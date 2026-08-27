// SPDX-License-Identifier: AGPL-3.0-only

//! Loading one DSA block: TP sharding plus the three load-time transforms the runtime
//! cannot do per token.
//!
//! Takes a `load` closure yielding an uploaded BF16 tensor by layer-relative name, rather
//! than a `WeightStore`, so the transforms are testable and the loader wiring stays one
//! call site.
//!
//! # The three transforms, and why each is here rather than in `decode`
//!
//! 1. **`q_absorb`** — `q_b_proj` pre-multiplied by `kv_b_proj`'s K half, so Q arrives in
//!    the 512-dim latent space the decode kernel dots against. Doing it per token would be
//!    a second GEMM on the critical path for a weight that never changes.
//! 2. **`weights_proj` scaled by `index_heads^-0.5`** — `dsa_index_scores` does not apply
//!    the factor. Folding it into the weight is exact (a positive scalar) and free.
//! 3. **`ape` upconverted BF16 → F32** — the kernel's parameter is `const float*` while
//!    the checkpoint stores BF16. This is the #341/#347 dtype-mismatch class: reading it
//!    at the wrong width is silent.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::Glm5NextDsaConfig;
use super::layer::Glm5NextDsaWeights;
use super::tp::{DsaShard, DsaTpPlan};

/// A tensor as the checkpoint holds it: full (unsharded) BF16 values on the host.
pub type LoadFn<'a> = &'a dyn Fn(&str) -> Result<Vec<f32>>;

/// `q_absorb[h*kvl + c][k] = Σ_r kv_b[h*(nope+vd) + r][c] · q_b[h*nope + r][k]`.
///
/// Per head this is `W_k^Tᐧq_b` — an `A^T B` contraction, which the `A @ B^T` GEMM kernel
/// cannot express without a transpose, so it runs on the host once at load.
///
/// 🪤 `q_b_proj` and `kv_b_proj` carry **different per-head widths** (`qk_head_dim` = 256
/// vs `nope + v_head_dim` = 512). Using one stride for the other still yields a
/// well-formed 2-D tensor of plausible values.
pub fn absorb_q(
    cfg: &Glm5NextDsaConfig,
    q_b: &[f32],
    kv_b: &[f32],
    full_heads: usize,
) -> Result<Vec<f32>> {
    let (nope, vd, kvl, ql) = (
        cfg.qk_nope_head_dim,
        cfg.v_head_dim,
        cfg.kv_lora_rank,
        cfg.q_lora_rank,
    );
    let qk = cfg.qk_head_dim();
    if q_b.len() != full_heads * qk * ql {
        bail!(
            "absorb_q: q_b_proj has {} elems, expected {}",
            q_b.len(),
            full_heads * qk * ql
        );
    }
    if kv_b.len() != full_heads * (nope + vd) * kvl {
        bail!(
            "absorb_q: kv_b_proj has {} elems, expected {}",
            kv_b.len(),
            full_heads * (nope + vd) * kvl
        );
    }
    // NoPE: qk_head_dim == qk_nope_head_dim, so the K half of kv_b lines up with the whole
    // of q_b. A rope section would need the rope rows carried separately and is refused.
    if cfg.qk_rope_head_dim != 0 {
        bail!(
            "absorb_q: NoPE only; qk_rope_head_dim is {}",
            cfg.qk_rope_head_dim
        );
    }

    let mut out = vec![0f32; full_heads * kvl * ql];
    for h in 0..full_heads {
        let kv_base = h * (nope + vd);
        let qb_base = h * nope;
        for c in 0..kvl {
            for k in 0..ql {
                let mut acc = 0f32;
                for r in 0..nope {
                    acc += kv_b[(kv_base + r) * kvl + c] * q_b[(qb_base + r) * ql + k];
                }
                out[(h * kvl + c) * ql + k] = acc;
            }
        }
    }
    Ok(out)
}

/// Rows `[start, end)` of a `[rows, row_elems]` row-major tensor.
fn row_slice(v: &[f32], row_elems: usize, start: usize, end: usize) -> Vec<f32> {
    v[start * row_elems..end * row_elems].to_vec()
}

/// Column range `[start, end)` of every row — the row-parallel case (`o_proj`).
fn col_slice(v: &[f32], row_elems: usize, start: usize, end: usize) -> Vec<f32> {
    v.chunks(row_elems)
        .flat_map(|r| r[start..end].iter().copied())
        .collect()
}

/// Apply one tensor's shard plan to full host values.
pub fn shard_host(plan: &super::tp::DsaTensorPlan, full: &[f32]) -> Vec<f32> {
    match plan.kind {
        DsaShard::Replicated => full.to_vec(),
        DsaShard::HeadRows => row_slice(
            full,
            plan.full_row_elems,
            plan.src_row_offset,
            plan.src_row_offset + plan.local_rows,
        ),
        DsaShard::HeadCols => col_slice(
            full,
            plan.full_row_elems,
            plan.src_col_offset,
            plan.src_col_offset + plan.local_row_elems,
        ),
    }
}

fn up_bf16(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = v
        .iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
        .collect();
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}
fn up_f32(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}

/// Bind one DSA block for this rank.
pub fn build_dsa_weights(
    gpu: &dyn GpuBackend,
    cfg: &Glm5NextDsaConfig,
    plan: &DsaTpPlan,
    load: LoadFn<'_>,
) -> Result<Glm5NextDsaWeights> {
    let get = |n: &str| -> Result<Vec<f32>> { load(&format!("self_attn.{n}")) };
    let shard = |n: &'static str, full: Vec<f32>| -> Result<Vec<f32>> {
        let p = plan
            .get(n)
            .ok_or_else(|| anyhow::anyhow!("no shard plan for {n}"))?;
        Ok(shard_host(p, &full))
    };

    // ── transform 1: absorb Q into latent space, THEN shard by head ──
    // Absorption is over full heads because it pairs q_b and kv_b head-for-head; slicing
    // first would pair this rank's q_b heads with the wrong kv_b rows.
    let q_absorb_full = absorb_q(
        cfg,
        &get("q_b_proj.weight")?,
        &get("kv_b_proj.weight")?,
        plan.full_heads,
    )?;
    let per_head = cfg.kv_lora_rank;
    let start = plan.tp_rank * plan.local_heads * per_head;
    let len = plan.local_heads * per_head;
    let q_absorb = row_slice(&q_absorb_full, cfg.q_lora_rank, start, start + len);

    // ── transform 2: fold index_heads^-0.5 into weights_proj ──
    let scale = (cfg.index_heads as f32).powf(-0.5);
    let weights_proj: Vec<f32> = get("indexer.weights_proj.weight")?
        .iter()
        .map(|x| x * scale)
        .collect();

    // ── transform 3: ape BF16 on disk -> F32 for the kernel ──
    let ape = get("indexer.index_kpool_compress_ape")?;

    Ok(Glm5NextDsaWeights {
        q_a_proj: up_bf16(gpu, &get("q_a_proj.weight")?)?,
        q_a_layernorm: up_bf16(gpu, &get("q_a_layernorm.weight")?)?,
        q_absorb: up_bf16(gpu, &q_absorb)?,
        kv_a_proj: up_bf16(gpu, &get("kv_a_proj_with_mqa.weight")?)?,
        kv_a_layernorm: up_bf16(gpu, &get("kv_a_layernorm.weight")?)?,
        o_proj: up_bf16(gpu, &shard("o_proj", get("o_proj.weight")?)?)?,
        wk: up_bf16(gpu, &get("indexer.wk.weight")?)?,
        k_norm_weight: up_bf16(gpu, &get("indexer.k_norm.weight")?)?,
        // 🪤 REQUIRED — LayerNorm bias, not optional.
        k_norm_bias: up_bf16(gpu, &get("indexer.k_norm.bias")?)?,
        compress_gate: up_bf16(gpu, &get("indexer.index_kpool_compress_gate")?)?,
        wq_b: up_bf16(gpu, &get("indexer.wq_b.weight")?)?,
        weights_proj: up_bf16(gpu, &weights_proj)?,
        ape: up_f32(gpu, &ape)?,
    })
}

#[cfg(test)]
mod tests;
