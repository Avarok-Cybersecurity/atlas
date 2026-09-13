// SPDX-License-Identifier: AGPL-3.0-only

//! The env switches and row-index arithmetic the mHC verify reads,
//! split out of `verify_hc.rs` to keep that file under the 500-line
//! cap. Re-exported from there, so every `verify_hc::NAME` path that
//! callers already use still resolves.

use super::*;

/// `ATLAS_QWEN4EXP_MTP_HC_COMMIT=0` reverts to the pre-fix fused split, which
/// leaves the SSM verify intermediates unwritten. Diagnostic only.
pub(crate) fn hc_verify_publishes_intermediates() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MTP_HC_COMMIT").as_deref() != Ok("0"))
}

/// `ATLAS_QWEN4EXP_MTP_ROLLBACK=1` — the same switch `rollback_verify_rows`
/// reads (`trait_impl/mod.rs`), duplicated here so the batched arm can refuse
/// the incompatible combination at the point of use rather than desyncing.
pub(crate) fn rollback_armed() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MTP_ROLLBACK").as_deref() == Ok("1"))
}

/// The verify rows an mHC verify of width `k` must publish an SSM
/// intermediate for.
///
/// The last row is excluded on purpose: `commit_accepted_prefix` reads
/// `commit_rewind_index(num_accepted)` and short-circuits at
/// `num_accepted == k`, so the highest index it can ever read is `k - 2`.
/// `hc_publish_covers_every_commit_rewind` pins that agreement.
/// Opt back into the FLA chunked GDN scan inside the mHC verify.
///
/// Default OFF: the verify has to reproduce `decode()` bit-for-bit or greedy
/// speculation is not lossless, and the chunked scan does not. This exists so
/// the cost of the sequential path can be A/B'd, not because the chunked one is
/// ever correct here.
pub(crate) fn verify_uses_fla_scan() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MTP_VERIFY_FLA").as_deref() == Ok("1"))
}

/// Element strides of the three per-row attention metadata streams, as
/// `prefill_b_upload_meta_at` packs them and as `layer::AttnMetadataDev`
/// documents them. They are NOT the same width, which is the whole reason
/// these are named constants: `positions` is `[N] u32`, `slots` is `[N] i64`
/// (`fill_slots_from_block_table` / `reshape_and_cache` both take i64), and
/// `seq_lens` is `[N] i32`. Bumping the slot pointer by 4 instead of 8 aims
/// the KV write at the wrong cache slot for every row past 0 -- silently,
/// because the value there is still a plausible slot index.
pub(super) const VERIFY_POS_STRIDE: usize = 4;
pub(super) const VERIFY_SLOT_STRIDE: usize = 8;
pub(super) const VERIFY_SEQ_LEN_STRIDE: usize = 4;

/// Byte offset, inside the verify's metadata block at `meta_base`, of the
/// `[k]` i32 per-row `seq_len` array.
///
/// The pack owns `[0, slot_offset)` for the position streams and
/// `[slot_offset, slot_offset + k*8)` for the i64 slot table
/// (`prefill_b_upload_paged` fills exactly `k` entries there). The seq_len
/// array goes immediately past it, 4-byte aligned.
pub(crate) fn verify_row_seq_len_offset(slot_offset: usize, k: usize) -> usize {
    (slot_offset + k * VERIFY_SLOT_STRIDE).next_multiple_of(VERIFY_SEQ_LEN_STRIDE)
}

/// The DEVICE `seq_len` value row `t` of a verify based at `base_seq_len`
/// must present to the paged-decode attention: the number of VISIBLE KEYS,
/// which includes the row's own token because `write_kv_cache` runs first.
pub(crate) fn verify_row_seq_len_value(base_seq_len: usize, t: usize) -> i32 {
    (base_seq_len + t + 1) as i32
}

/// The HOST `seq_len` argument for row `t`'s `Layer::decode`. That parameter
/// is the PRE-APPEND length -- serial decode of the token at absolute
/// position `p` passes `p` (see `attention_forward.rs`, the QSA
/// `decode_select` call site) -- so it is one less than the device value.
pub(crate) fn verify_row_decode_seq_len(base_seq_len: usize, t: usize) -> usize {
    base_seq_len + t
}

/// Default-ON. `ATLAS_QWEN4EXP_MTP_HC_ATTN_DECODE=0` puts the attention
/// layers back on the K-row `prefill()` body.
pub(crate) fn verify_attn_decode_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MTP_HC_ATTN_DECODE").as_deref() != Ok("0"))
}

/// EXPERIMENTAL, default-OFF. `ATLAS_QWEN4EXP_MTP_HC_SSM_DECODE=1` puts the
/// GDN layers on the same one-row `decode()` body the attention layers take,
/// making the WHOLE verify decode-shaped. Only valid at k == 1 (see the call
/// site); the check there refuses anything wider rather than corrupting the
/// commit rewind.
pub(crate) fn verify_ssm_decode_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MTP_HC_SSM_DECODE").as_deref() == Ok("1"))
}

pub(crate) fn hc_publish_rows(k: usize) -> std::ops::Range<usize> {
    0..k.saturating_sub(1)
}

/// `ATLAS_QWEN4EXP_MTP_AUX_COMMIT=0` disables the auxiliary-carry commit.
/// Diagnostic only — with it off, every rejected verify row leaves PLE's
/// rolling conv/history and QSA's marks one row ahead of the sequence.
///
/// It replaces `ATLAS_QWEN4EXP_MTP_ROLLBACK=1`, which was the same rollback in
/// arm-to-use polarity AND wrong: reachable only from the K=2 reject branch,
/// and hard-wired to snapshot row 0.
pub(crate) fn hc_verify_commits_aux() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MTP_AUX_COMMIT").as_deref() != Ok("0"))
}

/// The aux snapshot row a commit of `num_accepted` out of `k` verify rows must
/// restore, or `None` when the live carries are already correct.
///
/// This is the auxiliary-carry twin of `commit_rewind_index`, and it delegates
/// to it rather than restating the arithmetic: the PLE carry and the SSM state
/// are rewound by the SAME commit and must land on the SAME token, so if the
/// two indices could drift the model would run one carry a row out from the
/// other. `hc_publish_rows` is the range both are published over.
///
/// `None` at `num_accepted == k` mirrors `commit_accepted_prefix`'s
/// short-circuit: nothing was discarded, so nothing is restored.
pub(crate) fn verify_aux_restore_row(num_accepted: usize, k: usize) -> Option<usize> {
    if num_accepted == 0 || num_accepted >= k {
        return None;
    }
    Some(commit_rewind_index(num_accepted))
}
