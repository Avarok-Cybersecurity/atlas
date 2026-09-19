// SPDX-License-Identifier: AGPL-3.0-only

//! PLE n-gram row ids: EOS-aware right-shift plus the multiply-XOR hash.
//!
//! Pure token-id arithmetic — no device, no weights — which is why it is
//! tested bit-exactly against the reference in `tests.rs` and runs in CI.
//!
//! **This does NOT transfer from LongCat (#746).** LongCat accumulates a
//! polynomial rolling hash (`acc += shift[d] * m[d]`, then `% rows`). Qwen
//! multiplies each shifted token by a SplitMix64-derived odd multiplier and
//! **XOR**s them, then takes a per-head prime modulus and adds a per-head
//! offset into one 320M-row table. The two produce different rows from the
//! same tokens, and both produce VALID rows — so a mix-up is silent.
//!
//! Reference: `Qwen4ExpTextNGramEmbedding.forward` /
//! `_shift_right_ignore_eos`, `bench/qwen4_exp/ref/modeling_qwen4_exp.py`.

/// Geometry for one PLE site, read from the checkpoint rather than derived.
///
/// `multipliers`, `head_vocab_sizes` and `head_offsets` are all SHIPPED
/// (`ple_embedding.layer_multipliers` / `.ngram_heads_vocab_sizes` /
/// `.ngram_heads_offsets`). The reference can derive them — from SplitMix64
/// and a prime search — and `bench/qwen4_exp/ple_golden.py` confirms the
/// derivation reproduces the shipped values exactly. Reading them is still
/// right: it cannot drift when the reference does.
#[derive(Clone, Debug)]
pub struct PleIdDims {
    /// `ngram_size` (3). Also the conv dilation, elsewhere.
    pub ngram_size: usize,
    /// `heads_per_ngram` (8). Heads are grouped by n-gram order:
    /// `[0, heads_per_ngram)` uses order 2, the next block order 3, and so on.
    pub heads_per_ngram: usize,
    /// `layer_multipliers[ngram_size]`, always odd.
    pub multipliers: Vec<u64>,
    /// `ngram_heads_vocab_sizes[ngram_heads]` — a distinct prime per head.
    pub head_vocab_sizes: Vec<u64>,
    /// `ngram_heads_offsets[ngram_heads]` — where each head's range starts
    /// in the single concatenated table.
    pub head_offsets: Vec<u64>,
    pub eos_token_id: u32,
}

impl PleIdDims {
    /// `(ngram_size - 1) * heads_per_ngram` — 16 here. Times `head_dim`
    /// (160) this is `ple_embed_dim` (2560): the head slices are
    /// CONCATENATED, not summed as LongCat's are.
    pub fn ngram_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// How many previous tokens a decode step must carry to reproduce
    /// prefill's ids. `ngram_size - 1` = 2.
    pub fn context_len(&self) -> usize {
        self.ngram_size - 1
    }

    /// Validate against the reference's invariants. Called once at load; a
    /// mismatch here is a checkpoint we do not understand, not something to
    /// paper over.
    pub fn validate(&self) -> anyhow::Result<()> {
        let heads = self.ngram_heads();
        anyhow::ensure!(
            self.multipliers.len() == self.ngram_size,
            "PLE: layer_multipliers has {} entries, expected ngram_size={}",
            self.multipliers.len(),
            self.ngram_size
        );
        anyhow::ensure!(
            self.head_vocab_sizes.len() == heads && self.head_offsets.len() == heads,
            "PLE: head vocab/offsets are {}/{}, expected ngram_heads={heads}",
            self.head_vocab_sizes.len(),
            self.head_offsets.len()
        );
        // The reference builds these as `2 * (splitmix64(..) % half) + 1`.
        // An even multiplier would collapse the low bit of every id.
        for (i, m) in self.multipliers.iter().enumerate() {
            anyhow::ensure!(
                m % 2 == 1,
                "PLE: layer_multipliers[{i}] = {m} is even; the reference \
                 derives `2*x + 1`, so this checkpoint is not what we think"
            );
        }
        anyhow::ensure!(
            self.head_vocab_sizes.iter().all(|v| *v > 0),
            "PLE: a head vocab size is 0 — modulus would divide by zero"
        );
        Ok(())
    }
}

/// Right-shift by `shift`, refusing to read across an EOS boundary.
///
/// Positions whose source would fall before the current EOS-delimited
/// segment get EOS instead. Transcribed from `_shift_right_ignore_eos`:
/// `previous_eos` is the last EOS position STRICTLY BEFORE each index, so a
/// token sitting on an EOS starts a segment at the previous one.
fn shift_right_ignore_eos(tokens: &[u32], shift: usize, eos: u32) -> Vec<u32> {
    if shift == 0 {
        return tokens.to_vec();
    }
    let mut out = vec![eos; tokens.len()];
    // `prev_eos` tracks `previous_eos_inclusive` shifted by one, i.e. the
    // cummax over indices strictly less than `pos`.
    let mut prev_eos: i64 = -1;
    let mut seen_eos: i64 = -1;
    for pos in 0..tokens.len() {
        let segment_start = prev_eos + 1;
        let position_in_segment = pos as i64 - segment_start;
        let source = pos as i64 - shift as i64;
        if position_in_segment >= shift as i64 && source >= 0 {
            out[pos] = tokens[source as usize];
        }
        if tokens[pos] == eos {
            seen_eos = pos as i64;
        }
        prev_eos = seen_eos;
    }
    out
}

/// Row ids for every head, one row per token in `tokens`.
///
/// `tokens` must already be `context ++ new`, where `context` is the
/// `context_len` preceding tokens (EOS-filled at the start of a sequence).
/// Returns `[tokens.len()][ngram_heads]`; callers slice off the last
/// `new.len()` rows, exactly as the reference's
/// `torch.cat(blocks, dim=-1)[:, -input_ids.shape[1]:]` does.
pub fn ple_ngram_ids(dims: &PleIdDims, tokens: &[u32]) -> Vec<Vec<u64>> {
    let heads = dims.ngram_heads();
    let shifted: Vec<Vec<u32>> = (0..dims.ngram_size)
        .map(|s| shift_right_ignore_eos(tokens, s, dims.eos_token_id))
        .collect();

    let mut out = vec![vec![0u64; heads]; tokens.len()];
    for ngram in 2..=dims.ngram_size {
        let start = (ngram - 2) * dims.heads_per_ngram;
        for (pos, row) in out.iter_mut().enumerate() {
            // `mixed = shifted[0]*m[0]`, then XOR `shifted[p]*m[p]` for
            // p in 1..ngram. Wrapping is the reference's semantics too:
            // torch int64 multiply wraps, and the multipliers are bounded by
            // `(2^63 - 1) / vocab_size` precisely so it does not.
            let mut mixed = (shifted[0][pos] as u64).wrapping_mul(dims.multipliers[0]);
            for p in 1..ngram {
                mixed ^= (shifted[p][pos] as u64).wrapping_mul(dims.multipliers[p]);
            }
            for h in start..start + dims.heads_per_ngram {
                row[h] = mixed % dims.head_vocab_sizes[h] + dims.head_offsets[h];
            }
        }
    }
    out
}
