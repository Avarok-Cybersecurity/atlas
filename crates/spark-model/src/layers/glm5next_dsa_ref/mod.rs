// SPDX-License-Identifier: AGPL-3.0-only
//! GLM-5.3-Flash **DSA (DeepSeek Sparse Attention) + kpool indexer CPU reference** — Slice 8.
//!
//! Design artifact, not a production path. Nothing here runs on GPU and no checkpoint tensor is
//! bound. Its only job is to pin the equations in Atlas-shaped code against goldens produced by
//! HuggingFace `transformers` **5.16.1** itself, before a CUDA kernel is written.
//!
//! The indexer is proven **before** the MLA on purpose: a wrong top-k still produces perfectly
//! plausible attention output, so an MLA test built on a broken selection passes and poisons
//! everything downstream.
//!
//! # Traps this module encodes
//!
//! * 🪤 **`k_norm` is a `nn.LayerNorm`, not an RMSNorm** — it subtracts the mean **and** it has a
//!   **bias**. Every other norm in GLM-5.3 is an RMSNorm without bias. The checkpoint carries
//!   `indexer.k_norm.bias`, which is the tell; a loader that binds only `.weight` silently drops
//!   it, and mean-subtraction is invisible in shapes.
//! * 🪤 **The pool softmax runs over the POOL-SLOT axis, per channel** — `softmax(dim=2)` over
//!   `[pool, slot, head_dim]`. It is not a softmax over `head_dim` and not over pools. Getting
//!   the axis wrong still yields a well-formed weighted average.
//! * 🪤 **Pooling starts at the FIRST VALID TOKEN, not at slot 0.** With left padding
//!   `[P, P, A, B, C, D]`, pool 0 is `[A, B, C, D]`. Pools are formed from `first_key + offset`.
//! * 🪤 **A pool is valid only if ALL `kpool` slots are valid.** A trailing partial pool is
//!   never a pool; it is handled by the separate tail append. So a 7-token sequence has
//!   **one** pool, not two.
//! * 🪤 **NoPE: `qk_rope_head_dim = 0`.** `kv_a_proj_with_mqa` emits `kv_lora_rank + 0`, and the
//!   `k_rot` slice is zero-width — a no-op copy, not a padded RoPE. Do not inherit DeepSeek's
//!   assumption that the rope section exists.
//! * 🔴 **`-1` is the invalid sentinel and the destination must be FULLY written.** vLLM's day-0
//!   GLM DSA bug was a `torch.empty` top-k buffer whose tail was never written when the valid
//!   pool count fell below the budget, so uninitialised memory became "token indices". Every
//!   function here writes all `out_width` entries unconditionally.

/// Indexer + MLA geometry, read from the checkpoint config — never defaulted.
#[derive(Clone, Copy, Debug)]
pub struct DsaDims {
    pub hidden: usize,
    /// Indexer heads (`index_n_heads`), NOT the MLA head count.
    pub index_heads: usize,
    /// Indexer head dim (`index_head_dim`), NOT the MLA head dim.
    pub index_head_dim: usize,
    pub index_kpool: usize,
    pub index_topk: usize,
    pub always_select_tail: bool,
    pub q_lora_rank: usize,
    // ── MLA ──
    pub heads: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    /// **Zero** on GLM-5.3-Flash. Kept explicit so a nonzero value is a loud change.
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
}

impl DsaDims {
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
    /// Pools selected per query: `index_topk / index_kpool`, capped by how many pools exist.
    pub fn select_k(&self, n_pools: usize) -> usize {
        (self.index_topk / self.index_kpool).min(n_pools)
    }
    /// Width of the emitted index row. The tail adds `kpool - 1` slots.
    pub fn out_width(&self) -> usize {
        self.index_topk
            + if self.always_select_tail {
                self.index_kpool - 1
            } else {
                0
            }
    }
    pub fn is_nope(&self) -> bool {
        self.qk_rope_head_dim == 0
    }
}

/// The invalid-index sentinel. Chosen by the reference implementation, not by us.
pub const INVALID: i32 = -1;

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `nn.LayerNorm` over the trailing `d`: mean-subtract, variance-normalise, then `w * x + b`.
///
/// 🪤 Not an RMSNorm. The mean subtraction and the bias are both real, and both are invisible
/// from tensor shapes — `indexer.k_norm.bias` existing in the checkpoint is the only tell.
pub fn layer_norm(x: &[f32], w: &[f32], b: &[f32], d: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (row_in, row_out) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let mean = row_in.iter().sum::<f32>() / d as f32;
        let var = row_in.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..d {
            row_out[i] = (row_in[i] - mean) * inv * w[i] + b[i];
        }
    }
    out
}

/// RMSNorm without bias — the norm every OTHER GLM module uses.
pub fn rms_norm(x: &[f32], w: &[f32], d: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (row_in, row_out) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let inv = 1.0 / (row_in.iter().map(|v| v * v).sum::<f32>() / d as f32 + eps).sqrt();
        for i in 0..d {
            row_out[i] = row_in[i] * inv * w[i];
        }
    }
    out
}

/// `y = x @ w^T` for `x: [m, k]`, `w: [n, k]` (torch `Linear` layout), no bias.
pub fn linear(x: &[f32], m: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += x[row * k + i] * w[col * k + i];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

/// The compressed k-pool candidates.
pub struct Pools {
    /// `[n_pools, index_head_dim]` — softmax-weighted average of the pool's keys.
    pub keys: Vec<f32>,
    /// `[n_pools, index_kpool]` — raw token index per slot, [`INVALID`] where the slot is not real.
    pub indices: Vec<i32>,
    /// `[n_pools]` — a pool counts only when EVERY slot is valid.
    pub valid: Vec<u8>,
    pub n_pools: usize,
}

/// Build the pools. `k`/`gate` are `[seq, index_head_dim]`, `valid` is `[seq]`.
///
/// Mirrors `Glm5NextTextIndexer.get_pooled_states`.
pub fn pool_states(
    k: &[f32],
    gate: &[f32],
    valid: &[u8],
    ape: &[f32],
    dims: DsaDims,
    seq: usize,
) -> Pools {
    let (d, kp) = (dims.index_head_dim, dims.index_kpool);
    let n_pools = seq.div_ceil(kp);
    // 🪤 Pooling starts at the first VALID token, so left padding is skipped rather than pooled.
    let first_key = valid.iter().position(|v| *v != 0).unwrap_or(seq) as i64;

    let mut keys = vec![0.0f32; n_pools * d];
    let mut indices = vec![INVALID; n_pools * kp];
    let mut pvalid = vec![0u8; n_pools];
    let mut logits = vec![0.0f32; kp];

    for p in 0..n_pools {
        let mut slot_valid = [false; 64];
        let mut slot_idx = [0usize; 64];
        let mut all = true;
        for s in 0..kp {
            let raw = first_key + (p * kp + s) as i64;
            let in_range = raw >= 0 && (raw as usize) < seq;
            let ok = in_range && valid[raw as usize] != 0;
            slot_valid[s] = ok;
            slot_idx[s] = if in_range { raw as usize } else { 0 };
            all &= ok;
            indices[p * kp + s] = if ok { raw as i32 } else { INVALID };
        }
        pvalid[p] = all as u8;

        // 🪤 softmax over the SLOT axis, independently per channel.
        for dd in 0..d {
            let mut mx = f32::NEG_INFINITY;
            for s in 0..kp {
                logits[s] = if slot_valid[s] {
                    gate[slot_idx[s] * d + dd] + ape[s * d + dd]
                } else {
                    f32::NEG_INFINITY
                };
                mx = mx.max(logits[s]);
            }
            let mut sum = 0.0f32;
            for s in 0..kp {
                let e = if logits[s] == f32::NEG_INFINITY {
                    0.0
                } else {
                    (logits[s] - mx).exp()
                };
                logits[s] = e;
                sum += e;
            }
            // A fully invalid pool softmaxes to NaN in torch; HF calls `nan_to_num`, i.e. 0.
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            let mut acc = 0.0f32;
            for s in 0..kp {
                if slot_valid[s] {
                    acc += logits[s] * inv * k[slot_idx[s] * d + dd];
                }
            }
            keys[p * d + dd] = acc;
        }
    }
    // 🔴 HF returns `pool_keys[:, keep]` with `keep = pool_valid.any(0)`: pools invalid for
    // EVERY batch element are dropped from the axis. This shrinks `n_pools`, and `select_k` is
    // derived from the COMPACTED count — so the selection budget is data-dependent, not a pure
    // function of sequence length. A 7-token sequence has ONE pool and a budget of ONE.
    let keep: Vec<usize> = (0..n_pools).filter(|p| pvalid[*p] != 0).collect();
    if keep.len() == n_pools {
        return Pools {
            keys,
            indices,
            valid: pvalid,
            n_pools,
        };
    }
    let mut ck = vec![0.0f32; keep.len() * d];
    let mut ci = vec![INVALID; keep.len() * kp];
    let mut cv = vec![0u8; keep.len()];
    for (j, p) in keep.iter().enumerate() {
        ck[j * d..(j + 1) * d].copy_from_slice(&keys[p * d..(p + 1) * d]);
        ci[j * kp..(j + 1) * kp].copy_from_slice(&indices[p * kp..(p + 1) * kp]);
        cv[j] = pvalid[*p];
    }
    Pools {
        keys: ck,
        indices: ci,
        valid: cv,
        n_pools: keep.len(),
    }
}

/// The original pool ids that survive HF's `keep = pool_valid.any(0)` compaction.
///
/// Derived from padding and sequence length alone, so a caller can compute it before launching
/// anything — which is why the GPU path computes the full grid and then gathers.
pub fn kept_pools(valid: &[u8], dims: DsaDims, seq: usize) -> Vec<i32> {
    let kp = dims.index_kpool;
    let n_pools = seq.div_ceil(kp);
    let first_key = valid.iter().position(|v| *v != 0).unwrap_or(seq) as i64;
    (0..n_pools)
        .filter(|p| {
            (0..kp).all(|s| {
                let raw = first_key + (p * kp + s) as i64;
                raw >= 0 && (raw as usize) < seq && valid[raw as usize] != 0
            })
        })
        .map(|p| p as i32)
        .collect()
}

/// Per-(query, pool) index score. Mirrors the `matmul → relu → head-weighted sum` chain.
///
/// `q` is `[q_rows, index_heads, index_head_dim]`, `weights` is `[q_rows, index_heads]` and is
/// expected to ALREADY carry the `index_heads^-0.5` factor.
pub fn index_scores(
    q: &[f32],
    weights: &[f32],
    pools: &Pools,
    dims: DsaDims,
    q_rows: usize,
) -> Vec<f32> {
    let (h, d, p) = (dims.index_heads, dims.index_head_dim, pools.n_pools);
    let scale = (d as f32).powf(-0.5);
    let mut out = vec![0.0f32; q_rows * p];
    for r in 0..q_rows {
        for pp in 0..p {
            let mut acc = 0.0f32;
            for hh in 0..h {
                let mut dot = 0.0f32;
                for dd in 0..d {
                    dot += q[(r * h + hh) * d + dd] * pools.keys[pp * d + dd];
                }
                // ReLU AFTER the scale. relu(s·x) == s·relu(x) for s > 0, so the two orders
                // agree — but only because the scale is positive; keep it explicit.
                acc += weights[r * h + hh] * (scale * dot).max(0.0);
            }
            out[r * p + pp] = acc;
        }
    }
    out
}

/// Which keys a query at `q_pos` may see: causal **and** not padding.
pub fn visible(valid_keys: &[u8], q_pos: usize, key_idx: usize) -> bool {
    key_idx <= q_pos && valid_keys[key_idx] != 0
}

/// Select up to `select_k` pools per query.
///
/// 🔴 **Deterministic tiebreak: higher score first, then SMALLER pool index.** The reference uses
/// `torch.topk`, whose tie order is implementation-defined, so the reference's own pool
/// *identities* are not a legal target on a tied row — only the selected *set*, and only when the
/// tie does not straddle the cutoff. This function pins a total order so Atlas is reproducible
/// regardless.
pub fn topk_pools(
    scores: &[f32],
    valid_candidates: &[u8],
    n_pools: usize,
    q_rows: usize,
    select_k: usize,
) -> Vec<i32> {
    let mut out = vec![INVALID; q_rows * select_k];
    let mut buf: Vec<(f32, usize)> = Vec::with_capacity(n_pools);
    for r in 0..q_rows {
        buf.clear();
        for p in 0..n_pools {
            let s = if valid_candidates[r * n_pools + p] != 0 {
                scores[r * n_pools + p]
            } else {
                f32::MIN
            };
            buf.push((s, p));
        }
        buf.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        for (j, (_, p)) in buf.iter().take(select_k).enumerate() {
            out[r * select_k + j] = *p as i32;
        }
    }
    out
}

/// Expand selected pools into raw token indices, append the visible tail, pad with [`INVALID`].
///
/// 🔴 The output is **fully written** for all `out_width` entries on every row, unconditionally.
/// That is the structural answer to vLLM's day-0 defect, where a `torch.empty` buffer's tail
/// stayed uninitialised whenever the valid pool count fell below the budget and uninitialised
/// memory was read back as token indices.
#[allow(clippy::too_many_arguments)]
pub fn expand_selection(
    selected: &[i32],
    pools: &Pools,
    valid_candidates: &[u8],
    valid_keys: &[u8],
    q_positions: &[usize],
    q_mask: &[u8],
    dims: DsaDims,
    seq: usize,
    select_k: usize,
) -> Vec<i32> {
    let (kp, width) = (dims.index_kpool, dims.out_width());
    let q_rows = q_positions.len();
    let mut out = vec![INVALID; q_rows * width];

    let first_key = valid_keys.iter().position(|v| *v != 0).unwrap_or(seq) as i64;
    for r in 0..q_rows {
        let row = &mut out[r * width..(r + 1) * width];
        if q_mask[r] == 0 {
            continue; // already all-INVALID; a padded query selects nothing
        }
        let mut w = 0usize;
        for j in 0..select_k {
            let p = selected[r * select_k + j];
            let ok = p >= 0 && valid_candidates[r * pools.n_pools + p as usize] != 0;
            for s in 0..kp {
                row[w] = if ok {
                    pools.indices[p as usize * kp + s]
                } else {
                    INVALID
                };
                w += 1;
            }
        }
        if dims.always_select_tail {
            // The in-progress (incomplete) pool, as RAW indices.
            let vis_count = (0..seq)
                .filter(|k| visible(valid_keys, q_positions[r], *k))
                .count();
            let tail_count = vis_count % kp;
            let tail_start = first_key + vis_count as i64 - tail_count as i64;
            for t in 0..kp - 1 {
                let idx = tail_start + t as i64;
                let ok = t < tail_count
                    && idx >= 0
                    && (idx as usize) < seq
                    && visible(valid_keys, q_positions[r], idx as usize);
                row[w] = if ok { idx as i32 } else { INVALID };
                w += 1;
            }
        }
        debug_assert_eq!(
            w,
            width.min(select_k * kp + if dims.always_select_tail { kp - 1 } else { 0 })
        );
        // Everything past `w` is already INVALID from the initial fill — never uninitialised.
    }
    out
}

/// Turn an index row into the boolean visibility mask the attention consumes.
///
/// Mirrors HF's `scatter_add` + `.ne(0)`: **duplicate indices collapse**, so a repeated token is
/// attended once, not twice. Out-of-range and [`INVALID`] entries are dropped.
pub fn topk_to_mask(topk: &[i32], q_rows: usize, width: usize, kv_len: usize) -> Vec<u8> {
    let mut mask = vec![0u8; q_rows * kv_len];
    for r in 0..q_rows {
        for j in 0..width {
            let i = topk[r * width + j];
            if i >= 0 && (i as usize) < kv_len {
                mask[r * kv_len + i as usize] = 1;
            }
        }
    }
    mask
}

/// NoPE MLA over a per-query selected key set.
///
/// `q`: `[q_rows, heads, qk_head_dim]`, `k`: `[kv_len, heads, qk_head_dim]`,
/// `v`: `[kv_len, heads, v_head_dim]`, `mask`: `[q_rows, kv_len]`.
///
/// 🪤 NoPE means `qk_head_dim == qk_nope_head_dim` and there is **no rope section to skip**.
/// The scale is `qk_head_dim^-0.5` over the FULL head dim, which on GLM-5.3 equals
/// `qk_nope_head_dim^-0.5` only because the rope part is zero-width — do not hardcode either.
pub fn mla_masked_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &[u8],
    dims: DsaDims,
    q_rows: usize,
    kv_len: usize,
) -> Vec<f32> {
    let (h, qd, vd) = (dims.heads, dims.qk_head_dim(), dims.v_head_dim);
    let scale = (qd as f32).powf(-0.5);
    let mut out = vec![0.0f32; q_rows * h * vd];
    for r in 0..q_rows {
        for hh in 0..h {
            // Online (flash-style) softmax: one pass, no [kv_len] score buffer.
            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0f32;
            let mut acc = vec![0.0f32; vd];
            for kk in 0..kv_len {
                if mask[r * kv_len + kk] == 0 {
                    continue;
                }
                let mut dot = 0.0f32;
                for dd in 0..qd {
                    dot += q[(r * h + hh) * qd + dd] * k[(kk * h + hh) * qd + dd];
                }
                let s = dot * scale;
                let m_new = m.max(s);
                let corr = if m == f32::NEG_INFINITY {
                    0.0
                } else {
                    (m - m_new).exp()
                };
                let p = (s - m_new).exp();
                l = l * corr + p;
                for dd in 0..vd {
                    acc[dd] = acc[dd] * corr + p * v[(kk * h + hh) * vd + dd];
                }
                m = m_new;
            }
            let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
            for dd in 0..vd {
                out[(r * h + hh) * vd + dd] = acc[dd] * inv;
            }
        }
    }
    out
}

/// Expand the compressed latent into per-head K and V.
///
/// 🪤 NoPE: `k_rot` is zero-width, so the `key_states[..., nope:]` copy HF performs is a no-op.
/// Reproducing it as a padded RoPE section would change `qk_head_dim` and the scale with it.
pub fn expand_kv(kv_c: &[f32], w_kv_b: &[f32], dims: DsaDims, seq: usize) -> (Vec<f32>, Vec<f32>) {
    let (h, nope, vd, r) = (
        dims.heads,
        dims.qk_nope_head_dim,
        dims.v_head_dim,
        dims.kv_lora_rank,
    );
    assert_eq!(dims.qk_rope_head_dim, 0, "expand_kv is the NoPE path");
    let wide = linear(kv_c, seq, r, w_kv_b, h * (nope + vd));
    let mut k = vec![0.0f32; seq * h * nope];
    let mut v = vec![0.0f32; seq * h * vd];
    for t in 0..seq {
        for hh in 0..h {
            let src = t * h * (nope + vd) + hh * (nope + vd);
            k[(t * h + hh) * nope..(t * h + hh) * nope + nope]
                .copy_from_slice(&wide[src..src + nope]);
            v[(t * h + hh) * vd..(t * h + hh) * vd + vd]
                .copy_from_slice(&wide[src + nope..src + nope + vd]);
        }
    }
    (k, v)
}

/// Sigmoid re-exported for the microtest's convenience; the DSA path itself does not gate.
pub fn sigmoid_f32(x: f32) -> f32 {
    sigmoid(x)
}

#[cfg(test)]
mod tests;
