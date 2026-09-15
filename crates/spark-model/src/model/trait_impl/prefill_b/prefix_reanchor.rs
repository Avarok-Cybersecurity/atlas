// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 2 helper: resolve the Marconi SSM anchor for a prefix match,
//! re-anchoring BELOW a declined exact full-prompt leaf.
//!
//! `PrefixCache::lookup` returns the DEEPEST snapshot at or below the matched
//! prefix. For a request whose prompt is ENTIRELY cached (`matched == total`)
//! that is the finish leaf saved at `total` — and `prefix_lookup.rs` declines
//! it: the exact-leaf shortcut is bypassed by default (unsound by
//! construction; `ATLAS_MARCONI_EXACT=1` re-enables it), and even when enabled
//! a hidden-less finish leaf cannot produce the first token's logits. The
//! decline used to fall straight through to "no SSM snapshot — recomputing
//! all KV": a full cold prefill (6-7 s at 3.3K on qwen4exp) while the
//! block-aligned intermediate checkpoint saved seconds earlier for that very
//! prefix (e.g. token 3264 of 3286) sat unused in the pool. An intermediate
//! anchor at `d < total` is exactly the established warm-hit path
//! (`snap_tok < matched == total`, the "leaf evicted" case): restore
//! state@d, replay `[d, total)` — so the fix is to re-select the deepest
//! snapshot STRICTLY below the prompt and take that path.
//!
//! Every restore-site guard (`marconi_min_tokens`, tail session gate,
//! aux-state presence for PLE/QSA, `matched <= total`) still runs on the
//! re-anchored snapshot in `prefix_lookup.rs`; only the anchor SELECTION
//! changes, the exact-leaf shortcut itself is untouched.

use anyhow::Result;
use spark_runtime::prefix_cache::PrefixMatch;

use super::super::super::types::TransformerModel;

/// `ATLAS_MARCONI_EXACT=1` re-enables the exact full-prompt leaf shortcut
/// (A/B only). Single reader shared with the restore site's `bypass_exact`.
pub(super) fn marconi_exact_enabled() -> bool {
    std::env::var("ATLAS_MARCONI_EXACT").as_deref() == Ok("1")
}

/// Pure decision: does the restore site DECLINE the anchor at `depth` as an
/// exact full-prompt leaf? Mirrors `bypass_exact` (default: every exact leaf)
/// and `exact_without_hidden` (shortcut enabled, leaf has no stashed hidden)
/// in `prefix_lookup.rs`. An intermediate anchor (`depth < matched`) or a
/// warm multi-turn hit (`matched < total`) is never declined here.
pub(super) fn exact_leaf_declined(
    depth: usize,
    matched: usize,
    total: usize,
    exact_enabled: bool,
    has_hidden: bool,
) -> bool {
    depth > 0 && depth == matched && matched == total && (!exact_enabled || !has_hidden)
}

/// Depth cap for the re-anchor lookup (`lookup_ssm_anchor`'s cap is
/// inclusive): the deepest snapshot at least TWO tokens below the prompt.
/// Strictly below is not enough — an anchor at `total - 1` leaves exactly one
/// token to replay, and `forward_layers` routes a one-token replay through the
/// decode-layer shortcut, which has no `kv_write_start` floor and would
/// rewrite position `total - 1`'s K/V (already cached, in a block shared with
/// the prefix cache) with GEMV-rounded values. Only meaningful after
/// [`exact_leaf_declined`] returned true, which implies `total > 0`.
pub(super) fn reanchor_cap(total: usize) -> usize {
    total.saturating_sub(2)
}

impl TransformerModel {
    /// Effective SSM snapshot for `prefix_match` — `eff_ssm_snapshot` plus the
    /// re-anchor below a declined exact leaf. On a re-anchor `prefix_match`'s
    /// `ssm_*` fields are rewritten to the new anchor so every downstream
    /// reader (`ssm_snapshot_tokens`, `ssm_snapshot_is_tail`) sees it; the KV
    /// half is untouched. Returns `(eff_snapshot, eff_snapshot_tokens)`.
    pub(super) fn prefill_b_resolve_ssm_anchor(
        &self,
        tokens: &[u32],
        prefix_match: &mut PrefixMatch,
        total: usize,
        session_hash: u64,
        adapter_id: u64,
        stream: u64,
    ) -> Result<(Option<usize>, usize)> {
        let matched = prefix_match.matched_tokens;
        let exact_enabled = marconi_exact_enabled();
        // (1) BEFORE any tier fault-in: the default bypass declines an exact
        // leaf on depth alone, so decide now and never fault in a leaf that
        // will not be restored.
        let depth = prefix_match.ssm_anchor().depth();
        let mut tried = false;
        if exact_leaf_declined(depth, matched, total, exact_enabled, true) {
            tried = true;
            self.reanchor_below_prompt(
                tokens,
                prefix_match,
                total,
                session_hash,
                adapter_id,
                "bypassed by default; ATLAS_MARCONI_EXACT=1 re-enables",
            )?;
        }
        let (eff_snapshot, eff_snapshot_tokens) =
            self.eff_ssm_snapshot(prefix_match, session_hash, stream);
        // (2) AFTER the leaf is resident: with the shortcut enabled, a
        // hidden-less finish leaf is declined too (`exact_without_hidden`).
        if !tried
            && let Some(snap_id) = eff_snapshot
            && exact_leaf_declined(
                eff_snapshot_tokens,
                matched,
                total,
                exact_enabled,
                self.ssm_snapshots.has_hidden(snap_id),
            )
            && self.reanchor_below_prompt(
                tokens,
                prefix_match,
                total,
                session_hash,
                adapter_id,
                "finish leaf has no stashed hidden",
            )?
        {
            return Ok(self.eff_ssm_snapshot(prefix_match, session_hash, stream));
        }
        Ok((eff_snapshot, eff_snapshot_tokens))
    }

    /// Re-select the deepest snapshot strictly below the prompt and install
    /// it on `prefix_match`. Returns `false` (match untouched) when none
    /// qualifies — the caller then falls through to the pre-existing
    /// full-recompute path unchanged.
    ///
    /// Multi-rank worlds (EP or TP): the F83 sync in `prefix_lookup.rs` agrees
    /// only on `matched`; the SSM skip point is rank-local, and a rank that
    /// holds the intermediate while another lost it to LRU/reclaim would
    /// replay different token counts into the MoE all-reduce (size mismatch /
    /// deadlock). So the re-anchor depth is AGREED like `matched`: min-reduce
    /// the local depth, each deeper rank re-selects at the agreed cap, repeat
    /// until every rank holds the same depth (min == max) or some rank holds
    /// nothing (agreed 0) — then every rank re-anchors at that depth or no
    /// rank does. Bounded rounds; the decision is a pure function of the
    /// collective values, so it is identical on all ranks by construction.
    fn reanchor_below_prompt(
        &self,
        tokens: &[u32],
        prefix_match: &mut PrefixMatch,
        total: usize,
        session_hash: u64,
        adapter_id: u64,
        reason: &str,
    ) -> Result<bool> {
        let mut anchor = self.prefix_cache.lookup_ssm_anchor(
            tokens,
            reanchor_cap(total),
            session_hash,
            adapter_id,
        );
        if self.multi_rank_protocol_active() {
            let mut depth = if anchor.is_some() {
                anchor.depth() as u32
            } else {
                0
            };
            let mut agreed_all = false;
            for _round in 0..4 {
                let lo = self.ep_min_u32(depth)?;
                let hi = u32::MAX - self.ep_min_u32(u32::MAX - depth)?;
                if lo == hi {
                    agreed_all = true;
                    break;
                }
                if lo == 0 {
                    depth = 0;
                    break;
                }
                if depth > lo {
                    anchor = self.prefix_cache.lookup_ssm_anchor(
                        tokens,
                        lo as usize,
                        session_hash,
                        adapter_id,
                    );
                    depth = if anchor.is_some() {
                        anchor.depth() as u32
                    } else {
                        0
                    };
                }
            }
            if !agreed_all {
                // Final collective decision, identical on every rank.
                let lo = self.ep_min_u32(depth)?;
                let hi = u32::MAX - self.ep_min_u32(u32::MAX - depth)?;
                agreed_all = lo == hi && lo > 0;
            }
            if !agreed_all || depth == 0 {
                tracing::info!(
                    "Marconi exact leaf at token {total} declined ({reason}); ranks hold no \
                     common snapshot below the prompt — full recompute on every rank"
                );
                return Ok(false);
            }
        }
        if !anchor.is_some() {
            tracing::debug!(
                "Marconi exact leaf at token {total} declined ({reason}); no snapshot \
                 below the prompt — falling through to full recompute"
            );
            return Ok(false);
        }
        let depth = anchor.depth();
        tracing::info!(
            "Marconi exact leaf at token {total} declined ({reason}); re-anchored on the \
             deepest snapshot below the prompt: token {depth} ({} SSM tokens to replay, \
             tail={}, resident={})",
            total.saturating_sub(depth),
            anchor.is_tail,
            anchor.snapshot.is_some(),
        );
        prefix_match.set_ssm_anchor(anchor);
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::{exact_leaf_declined, reanchor_cap};

    const T: usize = 3286; // the measured repro: 206 blocks, leaf at 3286
    const CKPT: usize = 3264; // block-aligned intermediate checkpoint (204 blocks)

    #[test]
    fn default_bypass_declines_every_exact_leaf() {
        // Full-prompt hit on the exact leaf: declined regardless of hidden.
        assert!(exact_leaf_declined(T, T, T, false, true));
        assert!(exact_leaf_declined(T, T, T, false, false));
    }

    #[test]
    fn shortcut_enabled_declines_only_hidden_less_leaves() {
        assert!(!exact_leaf_declined(T, T, T, true, true));
        assert!(exact_leaf_declined(T, T, T, true, false));
    }

    #[test]
    fn intermediate_and_warm_turn_anchors_are_never_declined() {
        // Intermediate checkpoint under a full-prompt match (the re-anchor
        // target itself): always accepted.
        assert!(!exact_leaf_declined(CKPT, T, T, false, true));
        assert!(!exact_leaf_declined(CKPT, T, T, true, false));
        // Warm multi-turn hit: matched < total, even when the anchor covers
        // the whole match.
        assert!(!exact_leaf_declined(CKPT, CKPT, T, false, false));
        // No anchor at all.
        assert!(!exact_leaf_declined(0, 0, 0, false, true));
        assert!(!exact_leaf_declined(0, T, T, false, true));
    }

    #[test]
    fn reanchor_cap_leaves_at_least_two_tokens_to_replay() {
        assert_eq!(reanchor_cap(T), T - 2);
        // The checkpoint qualifies under the cap; the leaf and a `total - 1`
        // anchor (one-token replay = decode shortcut, no KV-write floor) do
        // not.
        assert!(CKPT <= reanchor_cap(T));
        assert!(T > reanchor_cap(T));
        assert!(T - 1 > reanchor_cap(T));
        assert_eq!(reanchor_cap(0), 0);
        assert_eq!(reanchor_cap(1), 0);
    }
}
