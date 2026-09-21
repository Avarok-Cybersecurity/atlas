// SPDX-License-Identifier: AGPL-3.0-only

//! EP wire protocol + fire/skip plan computation for the decode-time Marconi
//! checkpoint. Split out of `decode_checkpoint.rs` during the ≤500-line split
//! (pure move) — none of this touches `&self`, so it is testable with no GPU
//! and no container (see `prefill_b/snap_agree_tests.rs`).

use anyhow::{Result, bail};

/// A109 (2026-09-16): EP worker command — "save the decode-time Marconi
/// checkpoint rank 0 just saved", at the same `(slot, token, session)`.
///
/// **Invariant: every rank saves the same `(slot, token, session)` decode
/// checkpoint, or none does.** Before this command the head saved decode
/// checkpoints from a bare local call in scheduler-only code and the EP wire
/// protocol had no counterpart, so under TP=2/EP=2 rank 1 never held them
/// (HANDOFF-30 §6 — accidental, scheduler-sited). The A100 rank-agreed
/// restore then correctly refused every restore only one rank could serve,
/// and the head's extra snapshots evicted its own prefill checkpoints.
///
/// Wire shape: the `(seq_id, cmd)` preamble, then ONE bulk broadcast of
/// [`EP_CKPT_WORDS`] u32 — see [`encode_ckpt_payload`].
///
/// A113 (2026-09-16): moved from `0xFFFF_FFF6` to `0xFFFF_FFF8` — that value
/// was independently assigned on the DFlash lane (`EP_CMD_VERIFY_KGAMMA`,
/// `speculative.rs`), and the two lanes never built against each other until
/// campaign integration surfaced the collision as an `unreachable_patterns`
/// compile error, not a textual merge conflict.
pub(in crate::model) const EP_CMD_DECODE_CKPT: u32 = 0xFFFF_FFF8;

// Opcode band, checked at compile time. Worker commands must sit ABOVE the
// token range (`0..=0xFFFF_FFEF` is dispatched as a decode token id) and must
// not collide with a code already on the wire: F0 prefill chunk, F1
// alloc-slot, F2/F3/F4 verify K=2/3/4, F5 MTP propose, F6/F7 reserved for the
// DFlash lane, FF shutdown, E0 batched decode (matched before the token
// fallthrough).
const _: () = assert!(
    EP_CMD_DECODE_CKPT > 0xFFFF_FFEF,
    "EP_CMD_DECODE_CKPT would be dispatched as a decode token id"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFE0,
    "collides with batched decode"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFF0,
    "collides with prefill chunk"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFF1,
    "collides with alloc-slot"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFF2,
    "collides with verify K=2"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFF3,
    "collides with verify K=3"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFF4,
    "collides with verify K=4"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != crate::speculative::EP_CMD_MTP_PROPOSE,
    "collides with MTP propose"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFF6,
    "reserved: DFlash EP_CMD_VERIFY_KGAMMA (A113)"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFF7,
    "reserved: DFlash ctx-commit (A113)"
);
const _: () = assert!(EP_CMD_DECODE_CKPT != 0xFFFF_FFFF, "collides with shutdown");

/// Payload width of [`EP_CMD_DECODE_CKPT`], in u32 words.
pub(in crate::model) const EP_CKPT_WORDS: usize = 6;

/// What a decode checkpoint covers: the registered token count and the
/// block-table prefix it was taken over. Rank-agreed by construction — the
/// head computes it, every worker receives it verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::model) struct CkptPlan {
    /// TRUE coverage of the saved state (`seq.tokens.len()`), which under MTP
    /// can exceed `end_block * block_size` by 1..=bs+2 — see the note in
    /// `decode_ckpt_save_and_register`.
    pub snap_tokens: usize,
    /// Complete KV blocks the checkpoint spans.
    pub end_block: usize,
}

/// Everything the fire/skip decision reads, as plain values — so the decision
/// (and therefore "was an EP command emitted at all?") is testable with no
/// GPU and no container.
#[derive(Debug, Clone, Copy)]
pub(in crate::model) struct CkptInputs {
    /// `ssm_snapshots.is_enabled() && prefix_cache.is_active()` — the
    /// prefix-cache enable predicate (`AVAROK_GLM53_PREFIX_CACHE_UNPROVEN` +
    /// `--enable-prefix-caching`). FALSE ⇒ no save, and no EP command.
    pub enabled: bool,
    pub num_ssm_layers: usize,
    pub hss_window_start: usize,
    pub slot_idx: usize,
    pub tokens_len: usize,
    pub block_size: usize,
    pub block_table_len: usize,
    pub last_ckpt_block: usize,
    /// `AVAROK_DECODE_CKPT_BLOCKS`, already defaulted.
    pub interval: usize,
}

/// The cheap half of [`decode_ckpt_plan`], split out so the hot decode path
/// can bail before reading the env var or taking the KV lock — and so the two
/// cannot drift apart.
pub(in crate::model) fn ckpt_preconditions(
    enabled: bool,
    num_ssm_layers: usize,
    hss_window_start: usize,
    slot_idx: usize,
) -> bool {
    enabled && num_ssm_layers != 0 && hss_window_start == 0 && slot_idx != usize::MAX
}

/// The rank-0 fire/skip decision, as a pure function. `None` ⇒ nothing is
/// saved and no [`EP_CMD_DECODE_CKPT`] goes on the wire.
///
/// The vision-pad veto is deliberately NOT here: it scans the whole token
/// slice, so the caller applies it after this (cheap) decision says "fire".
pub(in crate::model) fn decode_ckpt_plan(i: &CkptInputs) -> Option<CkptPlan> {
    if !ckpt_preconditions(i.enabled, i.num_ssm_layers, i.hss_window_start, i.slot_idx) {
        return None;
    }
    if i.block_size == 0 || i.interval == 0 {
        return None;
    }
    // Derive the block count from tokens.len() (what we slice + cache), NOT
    // seq_len: under MTP seq_len can transiently exceed tokens.len() (verify
    // bonus position), which would overrun the token slice.
    let end_block = i.tokens_len / i.block_size;
    if end_block == 0
        || !end_block.is_multiple_of(i.interval)
        || end_block == i.last_ckpt_block
        // Only checkpoint blocks that physically exist. NOTE: the prefill-era
        // `kv_valid_tokens` guard does NOT apply here — that field tracks the
        // contiguous KV-written prefix during PREFILL and is never advanced by
        // decode, so it would wrongly veto every decode checkpoint past the
        // prompt length. During decode each token writes its KV inline, so all
        // `end_block` complete blocks are fully written.
        || end_block > i.block_table_len
    {
        return None;
    }
    Some(CkptPlan {
        snap_tokens: i.tokens_len,
        end_block,
    })
}

/// Encode the [`EP_CMD_DECODE_CKPT`] payload: plan + the head's session and
/// adapter identity, little-end word first for each u64.
pub(in crate::model) fn encode_ckpt_payload(
    plan: CkptPlan,
    session_hash: u64,
    adapter_id: u64,
) -> [u32; EP_CKPT_WORDS] {
    [
        plan.snap_tokens as u32,
        plan.end_block as u32,
        session_hash as u32,
        (session_hash >> 32) as u32,
        adapter_id as u32,
        (adapter_id >> 32) as u32,
    ]
}

/// Inverse of [`encode_ckpt_payload`]. Returns `(plan, session_hash,
/// adapter_id)` exactly as rank 0 sent them.
pub(in crate::model) fn decode_ckpt_payload(w: &[u32]) -> Result<(CkptPlan, u64, u64)> {
    if w.len() != EP_CKPT_WORDS {
        bail!(
            "EP_CMD_DECODE_CKPT: payload is {} words, expected {EP_CKPT_WORDS}",
            w.len()
        );
    }
    Ok((
        CkptPlan {
            snap_tokens: w[0] as usize,
            end_block: w[1] as usize,
        },
        w[2] as u64 | ((w[3] as u64) << 32),
        w[4] as u64 | ((w[5] as u64) << 32),
    ))
}
