// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the one u32 encoding of an [`AcceptRequest`] that both the kernel
//! (`kernels/ada/common/spec_accept_sample_ada.cu`) and the CPU [`super::reference`] read.
//!
//! Layout (little-endian u32 words):
//!
//! | words | content |
//! |---|---|
//! | 0..16 | header, indices in [`h`] |
//! | 16..48 | one draft per position ([`super::NO_DRAFT`] where none) |
//! | 48.. | edit descriptors, [`DESC_WORDS`] words each: regime 0's, then regime 1's |
//!
//! Descriptor words: token id; flags ([`e`] bits, live bias count in bits 4..6); repetition count;
//! delta (f32 bits); bias 0 and bias 1 (f32 bits).

use super::{
    AcceptMode, AcceptRequest, EditDesc, JobResult, MAX_EDITS, MAX_POS, MAX_REGIMES, NO_DRAFT,
};

/// Header word indices (the kernel's `H_*`).
pub mod h {
    /// Vocabulary size.
    pub const VOCAB: usize = 0;
    /// Verify positions.
    pub const NPOS: usize = 1;
    /// Regimes (1 or 2).
    pub const NREG: usize = 2;
    /// [`super::FLAG_SAMPLE`] or 0.
    pub const FLAGS: usize = 3;
    /// Temperature (f32 bits).
    pub const TEMP: usize = 4;
    /// Top-k (0 = none).
    pub const TOPK: usize = 5;
    /// Top-p (f32 bits).
    pub const TOPP: usize = 6;
    /// Min-p (f32 bits).
    pub const MINP: usize = 7;
    /// Repetition penalty (f32 bits).
    pub const REP: usize = 8;
    /// Rows below this mask the EOS-flagged edits.
    pub const EOS_UNTIL: usize = 9;
    /// RNG key, low word.
    pub const KEY_LO: usize = 10;
    /// RNG key, high word.
    pub const KEY_HI: usize = 11;
    /// Absolute position of row 0, low word.
    pub const POS_LO: usize = 12;
    /// Absolute position of row 0, high word.
    pub const POS_HI: usize = 13;
    /// Regime 0's descriptor count.
    pub const NEDIT0: usize = 14;
    /// Regime 1's descriptor count.
    pub const NEDIT1: usize = 15;
}

/// Descriptor flag bits (the kernel's `E_*`).
pub mod e {
    /// Always -inf.
    pub const MASK: u32 = 1;
    /// -inf below `eos_until`.
    pub const EOS: u32 = 2;
    /// `x - 12.0` when not masked.
    pub const SUB12: u32 = 4;
    /// `x - delta`.
    pub const DELTA: u32 = 8;
    /// The live bias count sits at this shift (2 bits).
    pub const NBIAS_SHIFT: u32 = 4;
}

/// Header flag: sample (else greedy).
pub const FLAG_SAMPLE: u32 = 1;
/// First draft word.
pub const DRAFTS_AT: usize = 16;
/// First descriptor word.
pub const DESC_AT: usize = DRAFTS_AT + MAX_POS;
/// Words per descriptor.
pub const DESC_WORDS: usize = 6;

/// Words the device request buffer must hold for the largest request.
pub fn capacity_words() -> usize {
    DESC_AT + MAX_REGIMES * MAX_EDITS * DESC_WORDS
}

/// Bytes of the device results buffer for the largest request.
pub fn results_bytes() -> usize {
    MAX_POS * MAX_REGIMES * 8
}

/// Encode a validated request (see [`super::validate`]).
pub fn encode(req: &AcceptRequest<'_>) -> Vec<u32> {
    let n_edits: usize = req.regimes.iter().map(Vec::len).sum();
    let mut w = vec![0u32; DESC_AT + n_edits * DESC_WORDS];
    w[h::VOCAB] = req.vocab as u32;
    w[h::NPOS] = req.n_pos as u32;
    w[h::NREG] = req.regimes.len() as u32;
    match req.mode {
        AcceptMode::Greedy => {
            w[h::TEMP] = 1.0f32.to_bits();
            w[h::TOPP] = 1.0f32.to_bits();
        }
        AcceptMode::Sample(p) => {
            w[h::FLAGS] = FLAG_SAMPLE;
            w[h::TEMP] = p.temperature.to_bits();
            w[h::TOPK] = p.top_k;
            w[h::TOPP] = p.top_p.to_bits();
            w[h::MINP] = p.min_p.to_bits();
        }
    }
    w[h::REP] = req.rep_penalty.to_bits();
    w[h::EOS_UNTIL] = req.eos_until;
    w[h::KEY_LO] = req.rng_key as u32;
    w[h::KEY_HI] = (req.rng_key >> 32) as u32;
    w[h::POS_LO] = req.pos_base as u32;
    w[h::POS_HI] = (req.pos_base >> 32) as u32;
    w[h::NEDIT0] = req.regimes.first().map_or(0, Vec::len) as u32;
    w[h::NEDIT1] = req.regimes.get(1).map_or(0, Vec::len) as u32;
    for p in 0..MAX_POS {
        w[DRAFTS_AT + p] = req.drafts.get(p).copied().unwrap_or(NO_DRAFT);
    }
    let mut at = DESC_AT;
    for edits in req.regimes {
        for d in edits {
            w[at..at + DESC_WORDS].copy_from_slice(&desc_words(d));
            at += DESC_WORDS;
        }
    }
    w
}

/// One descriptor's six words.
pub fn desc_words(d: &EditDesc) -> [u32; DESC_WORDS] {
    let mut flags = 0u32;
    if d.mask {
        flags |= e::MASK;
    }
    if d.eos {
        flags |= e::EOS;
    }
    if d.sub12 {
        flags |= e::SUB12;
    }
    if d.delta.is_some() {
        flags |= e::DELTA;
    }
    flags |= (u32::from(d.n_bias) & 3) << e::NBIAS_SHIFT;
    [
        d.token,
        flags,
        d.rep_count,
        d.delta.unwrap_or(0.0).to_bits(),
        d.bias[0].to_bits(),
        d.bias[1].to_bits(),
    ]
}

/// Decode one descriptor (the reference reads the blob back through this, like the kernel).
pub fn desc_from_words(w: &[u32]) -> EditDesc {
    let flags = w[1];
    EditDesc {
        token: w[0],
        mask: flags & e::MASK != 0,
        eos: flags & e::EOS != 0,
        sub12: flags & e::SUB12 != 0,
        rep_count: w[2],
        delta: (flags & e::DELTA != 0).then_some(f32::from_bits(w[3])),
        bias: [f32::from_bits(w[4]), f32::from_bits(w[5])],
        n_bias: ((flags >> e::NBIAS_SHIFT) & 3) as u8,
    }
}

/// Results buffer bytes back to [`JobResult`]s.
pub fn decode_results(bytes: &[u8], jobs: usize) -> Vec<JobResult> {
    bytes
        .chunks_exact(8)
        .take(jobs)
        .map(|c| JobResult {
            pick: u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
            status: u32::from_le_bytes([c[4], c[5], c[6], c[7]]),
        })
        .collect()
}
