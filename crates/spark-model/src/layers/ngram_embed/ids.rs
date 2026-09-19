// SPDX-License-Identifier: AGPL-3.0-only

//! The n-gram id core: EOS-aware right-shift and the polynomial rolling
//! hash. Pure token-id arithmetic — no device, no weights — which is why
//! it can be tested bit-exactly against the Python reference.

use super::NgramDims;

fn shift_right_ignore_eos(ctx: &[u32], n: usize, eos: u32) -> Vec<u32> {
    let len = ctx.len();
    let mut out = vec![0u32; len];
    let mut prev = 0usize;
    for (pos, &tok) in ctx.iter().enumerate() {
        if tok == eos {
            let end = pos + 1;
            if end - prev > n {
                out[prev + n..end].copy_from_slice(&ctx[prev..end - n]);
            }
            prev = end;
        }
    }
    if prev < len && len - prev > n {
        out[prev + n..len].copy_from_slice(&ctx[prev..len - n]);
    }
    out
}

/// Compute the row ids for EVERY table over `ctx` (the n-1 cached context
/// tokens followed by the new tokens). Returns `num_tables` vectors of
/// `ctx.len()` ids each, table-major in reference index order
/// (`(ngram-2)*K + split`); callers slice the last `seq_len` entries.
pub fn ngram_ids(dims: &NgramDims, ctx: &[u32]) -> Vec<Vec<u64>> {
    let mut out = Vec::with_capacity(dims.num_tables());
    // shift_d computed once per d, shared across splits (reference computes
    // shifted_ids once per n-gram size; d ranges 1..=N-1).
    let shifts: Vec<Vec<u32>> = (1..dims.neighbor_num)
        .map(|d| shift_right_ignore_eos(ctx, d, dims.eos_token_id))
        .collect();
    for ngram in 2..=dims.neighbor_num {
        for split in 0..dims.split_num {
            let index = (ngram - 2) * dims.split_num + split;
            let t = dims.table_rows(index);
            let mods = dims.vocab_mods(ngram, split);
            let ids = ctx
                .iter()
                .enumerate()
                .map(|(pos, &x)| {
                    let mut acc = x as u64;
                    for (d, &m) in mods.iter().enumerate() {
                        acc += shifts[d][pos] as u64 * m;
                    }
                    acc % t
                })
                .collect();
            out.push(ids);
        }
    }
    out
}

// ── GPU module ───────────────────────────────────────────────────────────
