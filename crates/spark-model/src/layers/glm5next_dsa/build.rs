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
//! 3. **`o_absorb`** — `o_proj` pre-multiplied by `kv_b_proj`'s V half, the counterpart of
//!    `q_absorb` and the half that was missing. The decode kernel consumes the latent KV
//!    directly, so its output is `[local_heads, kv_lora_rank]` in LATENT space; the raw
//!    checkpoint `o_proj` expects `[local_heads, v_head_dim]` in V space. Feeding one to the
//!    other is not a numeric drift — the GEMM reads `kv_lora_rank / v_head_dim` = 2x past
//!    the end of every weight row. Measured 2026-08-28: `CUDA_ERROR_ILLEGAL_ADDRESS` at
//!    layer 3, `grid=[256,1,1] block=[16,16,1]`, on the first forward that reached it.
//! 4. **`ape` upconverted BF16 → F32** — the kernel's parameter is `const float*` while
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
/// Worker count for the absorb loops. Shared spelling with
/// `mistral_loader::loader_impl::phase_qk_absorbed`, including the
/// `ATLAS_MLA_ABSORB_THREADS` override, so one lever tunes both.
fn absorb_threads(rows: usize) -> usize {
    let want = std::env::var("ATLAS_MLA_ABSORB_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });
    want.clamp(1, rows.max(1))
}

/// A disjoint span of `absorb_q` output rows, row `row0` onward.
///
/// 🔴 The accumulation order over `r` is IDENTICAL to the original nest, so
/// every output element is the same f32 sequence of adds — this is a memory
/// layout and threading change, not a numerics one. The original indexed
/// `kv_b[(kv_base+r)*kvl + c]` inside the `k` loop, re-loading a strided scalar
/// `ql` times; hoisting it makes the inner loop a contiguous walk of `q_b`'s
/// row and cuts `kv_b` loads per output row from `nope*ql` to `nope`.
#[allow(clippy::too_many_arguments)]
fn absorb_q_rows(
    out: &mut [f32],
    row0: usize,
    kv_b: &[f32],
    q_b: &[f32],
    kvl: usize,
    ql: usize,
    nope: usize,
    vd: usize,
) {
    for (i, acc) in out.chunks_exact_mut(ql).enumerate() {
        let row = row0 + i;
        let h = row / kvl;
        let c = row % kvl;
        let kv_base = h * (nope + vd);
        let qb_base = h * nope;
        acc.fill(0.0);
        for r in 0..nope {
            let w = kv_b[(kv_base + r) * kvl + c];
            let q_row = &q_b[(qb_base + r) * ql..][..ql];
            for (a, &q) in acc.iter_mut().zip(q_row.iter()) {
                *a += q * w;
            }
        }
    }
}

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

    let rows = full_heads * kvl;
    let mut out = vec![0f32; rows * ql];
    let threads = absorb_threads(rows);
    if threads <= 1 {
        absorb_q_rows(&mut out, 0, kv_b, q_b, kvl, ql, nope, vd);
    } else {
        let rows_per = rows.div_ceil(threads);
        std::thread::scope(|scope| {
            for (chunk_idx, chunk) in out.chunks_mut(rows_per * ql).enumerate() {
                let row0 = chunk_idx * rows_per;
                scope.spawn(move || {
                    absorb_q_rows(chunk, row0, kv_b, q_b, kvl, ql, nope, vd);
                });
            }
        });
    }
    Ok(out)
}

/// `o_absorb[i][h*kvl + c] = Σ_r o_proj[i][h*vd + r] · kv_b[h*(nope+vd) + nope + r][c]`.
///
/// The output-side twin of [`absorb_q`]. Absorbed MLA is a PAIR of transforms — Q into the
/// latent space on the way in, the output projection back out of it on the way out — and
/// shipping only the first leaves the decode kernel's latent output being read by a
/// V-space weight.
///
/// 🪤 Unlike `absorb_q`, this one is done AFTER sharding, and that is not an optimisation
/// that happens to be safe — it is exact. `absorb_q` pairs `q_b` head `h` with `kv_b` head
/// `h`, so slicing before pairing would cross heads. Here both operands are indexed by the
/// SAME `h` and the head axis is a plain outer sum, so this rank's heads never touch
/// another rank's rows. Doing it on full heads would double the load-time cost for an
/// identical result.
///
/// 🪤 `kv_b`'s V half starts at `nope`, not 0. Using the K half compiles, runs, and gives a
/// well-formed wrong answer — the same trap `absorb_q` documents from the other side.
pub fn absorb_o(
    cfg: &Glm5NextDsaConfig,
    o_local: &[f32],
    kv_b_local: &[f32],
    local_heads: usize,
) -> Result<Vec<f32>> {
    let (nope, vd, kvl, hidden) = (
        cfg.qk_nope_head_dim,
        cfg.v_head_dim,
        cfg.kv_lora_rank,
        cfg.hidden,
    );
    if cfg.qk_rope_head_dim != 0 {
        bail!(
            "absorb_o: NoPE only; qk_rope_head_dim is {}",
            cfg.qk_rope_head_dim
        );
    }
    let in_w = local_heads * vd;
    let out_w = local_heads * kvl;
    if o_local.len() != hidden * in_w {
        bail!(
            "absorb_o: o_proj has {} elems, expected {} ([{hidden}, {in_w}])",
            o_local.len(),
            hidden * in_w
        );
    }
    if kv_b_local.len() != local_heads * (nope + vd) * kvl {
        bail!(
            "absorb_o: kv_b_proj slice has {} elems, expected {}",
            kv_b_local.len(),
            local_heads * (nope + vd) * kvl
        );
    }

    let mut out = vec![0f32; hidden * out_w];
    // Rows are independent and contiguous, so a plain row split is the whole story.
    // Single-threaded this is `hidden * local_heads * kvl * vd` MACs — 17 G at GLM-5.3's
    // TP=2 shape, per layer, times 11 layers, on the GB10's CPU. That is minutes of load
    // time, paid on every bring-up.
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, hidden);
    let rows = hidden.div_ceil(threads);
    std::thread::scope(|sc| {
        for (o_chunk, i_chunk) in out
            .chunks_mut(rows * out_w)
            .zip(o_local.chunks(rows * in_w))
        {
            sc.spawn(move || {
                for (dst_row, src_row) in o_chunk.chunks_mut(out_w).zip(i_chunk.chunks(in_w)) {
                    for h in 0..local_heads {
                        let dst = &mut dst_row[h * kvl..(h + 1) * kvl];
                        let v_base = h * (nope + vd) + nope;
                        for r in 0..vd {
                            let w = src_row[h * vd + r];
                            let src = &kv_b_local[(v_base + r) * kvl..(v_base + r) * kvl + kvl];
                            for (d, k) in dst.iter_mut().zip(src) {
                                *d += w * k;
                            }
                        }
                    }
                }
            });
        }
    });
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

    let kv_b = get("kv_b_proj.weight")?;

    // ── transform 1: absorb Q into latent space, THEN shard by head ──
    // Absorption is over full heads because it pairs q_b and kv_b head-for-head; slicing
    // first would pair this rank's q_b heads with the wrong kv_b rows.
    let q_absorb_full = absorb_q(cfg, &get("q_b_proj.weight")?, &kv_b, plan.full_heads)?;
    let per_head = cfg.kv_lora_rank;
    let start = plan.tp_rank * plan.local_heads * per_head;
    let len = plan.local_heads * per_head;
    let q_absorb = row_slice(&q_absorb_full, cfg.q_lora_rank, start, start + len);

    // ── transform 3: absorb the output projection back OUT of latent space ──
    // Sharded first (see `absorb_o`): both operands index the same head, so this rank's
    // slice is exact and costs half the arithmetic.
    let kv_b_rows = cfg.qk_nope_head_dim + cfg.v_head_dim;
    let kv_b_local = row_slice(
        &kv_b,
        cfg.kv_lora_rank,
        plan.tp_rank * plan.local_heads * kv_b_rows,
        (plan.tp_rank + 1) * plan.local_heads * kv_b_rows,
    );
    let o_absorb = absorb_o(
        cfg,
        &shard("o_proj", get("o_proj.weight")?)?,
        &kv_b_local,
        plan.local_heads,
    )?;

    // ── transform 2: fold index_heads^-0.5 into weights_proj ──
    let scale = (cfg.index_heads as f32).powf(-0.5);
    let weights_proj: Vec<f32> = get("indexer.weights_proj.weight")?
        .iter()
        .map(|x| x * scale)
        .collect();

    // ── transform 4: ape BF16 on disk -> F32 for the kernel ──
    let ape = get("indexer.index_kpool_compress_ape")?;

    Ok(Glm5NextDsaWeights {
        q_a_proj: up_bf16(gpu, &get("q_a_proj.weight")?)?,
        q_a_layernorm: up_bf16(gpu, &get("q_a_layernorm.weight")?)?,
        q_absorb: up_bf16(gpu, &q_absorb)?,
        kv_a_proj: up_bf16(gpu, &get("kv_a_proj_with_mqa.weight")?)?,
        kv_a_layernorm: up_bf16(gpu, &get("kv_a_layernorm.weight")?)?,
        o_absorb: up_bf16(gpu, &o_absorb)?,
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
