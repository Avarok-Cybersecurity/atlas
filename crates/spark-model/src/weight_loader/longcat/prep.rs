// SPDX-License-Identifier: AGPL-3.0-only

//! LongCat MLA weight preparation: the two load-time transforms that let the
//! shared (DeepSeek/Mistral-lineage) MLA runtime serve LongCat unchanged.
//!
//! 1. ROPE CONVENTION. Atlas's rope kernels are `rotate_half`: they pair
//!    element `i` with `i + rope/2`. LongCat (like DeepSeek HF) stores the
//!    rope slice INTERLEAVED — its `apply_rotary_pos_emb_interleave` first
//!    de-interleaves `[x0,x1,x2,x3,…] → [x0,x2,…,x1,x3,…]` and then applies
//!    the same rotate_half math. That de-interleave is a fixed PERMUTATION of
//!    the projection's OUTPUT rows, so folding it into the weights at load
//!    (rows `j → 2j`, `j + rope/2 → 2j+1`) makes the runtime kernel produce
//!    exactly the reference's rotated values with no kernel change.
//!
//!    Applies to the rope rows of `q_b_proj` (per head, rows
//!    `nope..nope+rope` of each `qk_head_dim` block) and the trailing `rope`
//!    rows of `kv_a_proj_with_mqa`.
//!
//! 2. MLA LoRA SCALING. LongCat multiplies `q_pass`/`q_rot` by
//!    `sqrt(hidden/q_lora_rank)` and `k_pass` by `sqrt(hidden/kv_lora_rank)`
//!    (`mla_scale_q_lora` / `mla_scale_kv_lora`). Both fold into weights:
//!      - q: scale ALL of `q_b_proj` (both nope and rope halves are scaled).
//!      - kv: scale `kv_a_layernorm.weight` — `k_pass` is the norm's OUTPUT,
//!        and RMSNorm is scale-invariant in its input, so scaling the norm
//!        gain reproduces `scale * norm(x)` exactly. `k_rot` correctly stays
//!        UNSCALED: it bypasses the norm (it is split off before it).
//!
//! 3. HEAD-WIDTH PADDING. Atlas's MLA prefill assembles K at stride
//!    `qk_nope + qk_rope` and V at stride `v_head_dim`, then runs one
//!    FlashAttention whose head width is a COMPILE-TIME constant. Every MLA
//!    model shipped so far has `qk_nope + qk_rope == v_head_dim` (Mistral
//!    64+64=128, DeepSeek-V4 448+64=512), so the two strides agree and match
//!    HDIM. LongCat is the first where they do NOT: 128+64=192 vs 128. And
//!    192 cannot be compiled — common's kernels `#error` because it does not
//!    vectorize (4 or 8 BF16 per lane; 192/32 = 6).
//!
//!    So both are padded to 256, the stock HDIM, entirely in the weights:
//!    ```text
//!    q_b_proj  per head [nope 128 | rope 64]  -> [nope 128 | 0 x64 | rope 64]
//!    kv_b_proj per head [nope 128 | v 128]    -> [nope 128 | 0 x64 | v 128 | 0 x128]
//!    o_proj    per head reads v 128           -> reads 256, pad columns ZERO
//!    ```
//!    The zero pad in `o_proj` is what makes this exact: whatever lands in
//!    the padded V lanes is multiplied by zero. The padded Q lanes are zero
//!    too, so the padded K lanes cannot affect any score either.
//!
//!    The softmax scale must stay `1/sqrt(192)`, not `1/sqrt(256)`, so the
//!    loader sets it explicitly via `set_attn_scale_override`.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use crate::weight_map::{DenseWeight, dense};

const BF16: usize = 2;

fn to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn to_bf16(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| {
            let bits = x.to_bits();
            // round-to-nearest-even, matching the repo's other host converters
            let r = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16;
            r.to_le_bytes()
        })
        .collect()
}

/// De-interleave `rows` rope rows in place: source row `2j` becomes dest row
/// `j`, source `2j+1` becomes dest `j + rows/2` (the reference's
/// `view(-1, d/2, 2).transpose(4,3)` on the OUTPUT axis).
fn deinterleave_rope_rows(host: &mut [u8], base_row: usize, rows: usize, cols: usize) {
    let half = rows / 2;
    let row_bytes = cols * BF16;
    let start = base_row * row_bytes;
    let src: Vec<u8> = host[start..start + rows * row_bytes].to_vec();
    for j in 0..half {
        let (a, b) = (2 * j, 2 * j + 1);
        host[start + j * row_bytes..start + (j + 1) * row_bytes]
            .copy_from_slice(&src[a * row_bytes..(a + 1) * row_bytes]);
        host[start + (half + j) * row_bytes..start + (half + j + 1) * row_bytes]
            .copy_from_slice(&src[b * row_bytes..(b + 1) * row_bytes]);
    }
}

fn upload(host: &[u8], gpu: &dyn GpuBackend) -> Result<DevicePtr> {
    let p = gpu.alloc(host.len())?;
    gpu.copy_h2d(host, p)?;
    Ok(p)
}

/// `q_b_proj` `[n_heads*qk_head_dim, q_lora]`: de-interleave each head's rope
/// rows, fold `mla_scale_q_lora`, and PAD each head to `padded_hd` with the
/// rope block moved to the tail so it lines up with where
/// `mla_kv_assemble_batched` writes K's rope (`[padded_nope, padded_nope+rope)`).
/// Layout per head becomes `[nope | zeros | rope]`.
pub(super) fn prep_q_b(
    store: &WeightStore,
    name: &str,
    n_heads: usize,
    nope: usize,
    rope: usize,
    q_lora: usize,
    scale_q: f32,
    padded_hd: usize,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = dense(store, name)?;
    let hd = nope + rope;
    anyhow::ensure!(
        padded_hd >= hd,
        "longcat prep: padded_hd {padded_hd} < {hd}"
    );
    let bytes = n_heads * hd * q_lora * BF16;
    let mut host = vec![0u8; bytes];
    gpu.copy_d2h(w.weight, &mut host)
        .with_context(|| format!("longcat prep: d2h {name}"))?;
    for head in 0..n_heads {
        deinterleave_rope_rows(&mut host, head * hd + nope, rope, q_lora);
    }
    if (scale_q - 1.0).abs() > f32::EPSILON {
        let mut f = to_f32(&host);
        for v in &mut f {
            *v *= scale_q;
        }
        host = to_bf16(&f);
    }
    // Pad: [nope | rope] -> [nope | zeros | rope] at width `padded_hd`.
    let row = q_lora * BF16;
    let padded_nope = padded_hd - rope;
    let mut out = vec![0u8; n_heads * padded_hd * row];
    for head in 0..n_heads {
        let src = head * hd * row;
        let dst = head * padded_hd * row;
        out[dst..dst + nope * row].copy_from_slice(&host[src..src + nope * row]);
        let rsrc = src + nope * row;
        let rdst = dst + padded_nope * row;
        out[rdst..rdst + rope * row].copy_from_slice(&host[rsrc..rsrc + rope * row]);
    }
    Ok(DenseWeight {
        weight: upload(&out, gpu)?,
    })
}

/// `kv_b_proj` `[n_heads*(nope+v_dim), kv_lora]`: pad each head's block from
/// `[nope | v]` to `[nope | zeros | v | zeros]` so the assembled K stride
/// (`padded_nope + rope`) and V stride (`padded_v`) are BOTH `padded_hd`.
pub(super) fn prep_kv_b(
    store: &WeightStore,
    name: &str,
    n_heads: usize,
    nope: usize,
    v_dim: usize,
    rope: usize,
    kv_lora: usize,
    padded_hd: usize,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = dense(store, name)?;
    let row = kv_lora * BF16;
    let src_stride = nope + v_dim;
    let mut host = vec![0u8; n_heads * src_stride * row];
    gpu.copy_d2h(w.weight, &mut host)
        .with_context(|| format!("longcat prep: d2h {name}"))?;
    let padded_nope = padded_hd - rope;
    let dst_stride = padded_nope + padded_hd; // [nope-part | v-part]
    let mut out = vec![0u8; n_heads * dst_stride * row];
    for head in 0..n_heads {
        let src = head * src_stride * row;
        let dst = head * dst_stride * row;
        out[dst..dst + nope * row].copy_from_slice(&host[src..src + nope * row]);
        let vsrc = src + nope * row;
        let vdst = dst + padded_nope * row;
        out[vdst..vdst + v_dim * row].copy_from_slice(&host[vsrc..vsrc + v_dim * row]);
    }
    Ok(DenseWeight {
        weight: upload(&out, gpu)?,
    })
}

/// `o_proj` `[hidden, n_heads*v_dim]`: widen to `[hidden, n_heads*padded_hd]`
/// with ZERO columns in each head's pad lanes. Those zeros are what make the
/// whole padding scheme exact — whatever the attention leaves in a padded V
/// lane is multiplied by zero here.
pub(super) fn prep_o_proj(
    store: &WeightStore,
    name: &str,
    hidden: usize,
    n_heads: usize,
    v_dim: usize,
    padded_hd: usize,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = dense(store, name)?;
    let src_cols = n_heads * v_dim;
    let dst_cols = n_heads * padded_hd;
    let mut host = vec![0u8; hidden * src_cols * BF16];
    gpu.copy_d2h(w.weight, &mut host)
        .with_context(|| format!("longcat prep: d2h {name}"))?;
    let mut out = vec![0u8; hidden * dst_cols * BF16];
    for r in 0..hidden {
        for head in 0..n_heads {
            let s = (r * src_cols + head * v_dim) * BF16;
            let d = (r * dst_cols + head * padded_hd) * BF16;
            out[d..d + v_dim * BF16].copy_from_slice(&host[s..s + v_dim * BF16]);
        }
    }
    Ok(DenseWeight {
        weight: upload(&out, gpu)?,
    })
}

/// `kv_a_proj_with_mqa` `[kv_lora + rope, hidden]`: de-interleave the trailing
/// rope rows (k_rot). NOT scaled — `k_rot` bypasses `kv_a_layernorm`.
pub(super) fn prep_kv_a(
    store: &WeightStore,
    name: &str,
    kv_lora: usize,
    rope: usize,
    hidden: usize,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = dense(store, name)?;
    let bytes = (kv_lora + rope) * hidden * BF16;
    let mut host = vec![0u8; bytes];
    gpu.copy_d2h(w.weight, &mut host)
        .with_context(|| format!("longcat prep: d2h {name}"))?;
    deinterleave_rope_rows(&mut host, kv_lora, rope, hidden);
    Ok(DenseWeight {
        weight: upload(&host, gpu)?,
    })
}

/// `kv_a_layernorm.weight` `[kv_lora]` scaled by `mla_scale_kv_lora` — the
/// fold that reproduces the reference's `k_pass * sqrt(hidden/kv_lora)`.
pub(super) fn prep_kv_a_norm(
    store: &WeightStore,
    name: &str,
    kv_lora: usize,
    scale_kv: f32,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = dense(store, name)?;
    if (scale_kv - 1.0).abs() <= f32::EPSILON {
        return Ok(w);
    }
    let mut host = vec![0u8; kv_lora * BF16];
    gpu.copy_d2h(w.weight, &mut host)?;
    let mut f = to_f32(&host);
    for v in &mut f {
        *v *= scale_kv;
    }
    Ok(DenseWeight {
        weight: upload(&to_bf16(&f), gpu)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deinterleave_moves_even_rows_first() {
        // 4 rope rows of width 1: [a,b,c,d] (interleaved pairs (a,b),(c,d))
        // → [a,c,b,d]: evens to the front half, odds to the back half, which
        // is exactly what rotate_half then pairs as (a,b) and (c,d).
        let vals: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut host = to_bf16(&vals);
        deinterleave_rope_rows(&mut host, 0, 4, 1);
        assert_eq!(to_f32(&host), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn deinterleave_respects_base_row_and_width() {
        // 2 leading rows untouched, then 4 rope rows of width 2.
        let vals: [f32; 12] = [
            -1.0, -1.0, -2.0, -2.0, // leading (nope) rows
            1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
        ];
        let mut host = to_bf16(&vals);
        deinterleave_rope_rows(&mut host, 2, 4, 2);
        let got = to_f32(&host);
        assert_eq!(&got[..4], &[-1.0, -1.0, -2.0, -2.0]);
        assert_eq!(&got[4..], &[1.0, 1.5, 3.0, 3.5, 2.0, 2.5, 4.0, 4.5]);
    }
}
