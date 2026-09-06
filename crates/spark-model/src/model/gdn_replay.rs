// SPDX-License-Identifier: AGPL-3.0-only

//! Which GDN prefill recurrence a pass may take, so a warm restore replays
//! what a cold prefill computed.
//!
//! # The defect this closes
//!
//! With `--enable-prefix-caching` a hybrid-SSM prompt can be prefilled two
//! ways. COLD it runs `[0, total)` split at the chunk boundaries
//! `prefill_chunk_dispatch` picks. WARM it restores the recurrent state from a
//! Marconi checkpoint at `snap_tok` and replays only `[snap_tok, total)`. The
//! two passes must agree — the warm pass reuses the cold pass's cached KV for
//! `[snap_tok, matched)` and its own output feeds the same decode.
//!
//! They did not agree, because the two passes took DIFFERENT KERNELS. The FLA
//! chunked recurrence groups tokens into a 64-wide chunk grid anchored at the
//! START OF THE PASS, so it is a function of where the pass was cut; the
//! token-sequential ladder (register-resident -> WY4 -> persistent -> split4)
//! carries H forward one token at a time and is not. `gdn_exact_replay` already
//! forced the token-sequential ladder for the warm replay — and ONLY for the
//! warm replay, so the cold pass that produced the cached KV kept running FLA
//! and the boundary crossed kernels.
//!
//! Measured on gx10-9959 (qwen3.8-flash-next EXL3 2.05bpw, 490-token prompt,
//! cold chunk `[464, 490)` against the warm replay of the same 26 tokens from
//! the same restored state): the first divergence is layer 0 stage `raw_recur`
//! — the recurrence output — at relative L2 3.045e-03 with EVERY input to that
//! kernel bit-identical (0.000e+00). It is a STEP at the first layer, not a
//! ramp, and it grows ~27x with depth (post-MoE relL2 L0 3.28e-03 -> L35
//! 8.83e-02). Forcing the cold pass onto the same token-sequential kernel made
//! all 420 tapped (layer, stage) points bit-identical.
//!
//! # The rule
//!
//! When the prefix cache is active on a model with SSM layers, EVERY prefill
//! pass takes the token-sequential ladder — not just the warm ones. A cold
//! pass is what a later warm pass is compared against, so it is exactly as
//! load-bearing as the replay.
//!
//! The kill switch is `ModelLevers::gdn_fla_under_prefix_cache`
//! (`ATLAS_GDN_FLA_UNDER_PREFIX_CACHE=1`), which restores the previous
//! behaviour for an A/B. It does not disable the warm-replay force —
//! `marconi_skip` still pins the replay, as before.

/// Must this prefill pass take the token-sequential (chunk-grid-free) GDN
/// recurrence?
///
/// Split out as a free function over plain values so the decision table is
/// testable without a GPU, a model, or a mutated process environment.
///
/// * `marconi_skip` — this pass continues from a restored Marconi snapshot.
///   Pinned since 2026-06-10; the warm replay was never allowed on FLA.
/// * `prefix_cache_active` — a warm restore is POSSIBLE for this model, so
///   this pass may later be the cold reference a replay is compared against.
/// * `has_ssm_layers` — no GDN layers, nothing to decide.
/// * `allow_fla_under_prefix_cache` — the kill switch; restores the split
///   behaviour that produced the 3.045e-03 layer-0 step.
pub(crate) fn prefill_recurrence_must_be_grid_free(
    marconi_skip: bool,
    prefix_cache_active: bool,
    has_ssm_layers: bool,
    allow_fla_under_prefix_cache: bool,
) -> bool {
    if marconi_skip {
        return true;
    }
    prefix_cache_active && has_ssm_layers && !allow_fla_under_prefix_cache
}

impl super::types::TransformerModel {
    /// `ForwardContext::gdn_exact_replay` for a prefill pass.
    ///
    /// Call this at every prefill `ForwardContext` construction instead of
    /// passing `marconi_skip` (or a bare `false`) through — the whole point is
    /// that the COLD pass and the WARM pass answer the same way.
    pub(in crate::model) fn gdn_exact_replay_for_prefill(&self, marconi_skip: bool) -> bool {
        let active = self.prefix_cache.is_active();
        let has_ssm = self.config.num_ssm_layers() > 0;
        let allow_fla = self.levers.gdn_fla_under_prefix_cache;
        let grid_free =
            prefill_recurrence_must_be_grid_free(marconi_skip, active, has_ssm, allow_fla);
        // One line per model, so a serve log says which contract is in force.
        // Log-once latch (`ModelStats::once`) rather than a static: an operator
        // who swaps models must see the new model's answer, not a swallowed
        // duplicate of the previous one.
        if active && has_ssm && self.stats.once("log:gdn_prefill_grid_free") {
            if allow_fla {
                tracing::warn!(
                    "GDN prefill: FLA left ENABLED under prefix caching \
                     (ATLAS_GDN_FLA_UNDER_PREFIX_CACHE=1). Cold and warm passes take \
                     DIFFERENT recurrence kernels across a Marconi restore boundary — \
                     measured 3.045e-03 relative divergence at layer 0. Diagnostic only."
                );
            } else {
                tracing::info!(
                    "GDN prefill: token-sequential recurrence forced for ALL prefill passes \
                     (prefix caching active on a hybrid-SSM model), so a warm Marconi \
                     replay recomputes exactly what the cold pass cached. \
                     ATLAS_GDN_FLA_UNDER_PREFIX_CACHE=1 restores the split behaviour."
                );
            }
        }
        grid_free
    }
}

#[cfg(test)]
mod tests {
    use super::prefill_recurrence_must_be_grid_free as decide;

    /// The regression. Before the fix the cold pass answered `false` while the
    /// warm replay answered `true`, so the two sides of a Marconi boundary ran
    /// different recurrence kernels. Both must now answer the same.
    #[test]
    fn cold_and_warm_agree_when_the_prefix_cache_is_active() {
        for marconi_skip in [false, true] {
            assert!(
                decide(marconi_skip, true, true, false),
                "prefix cache active + SSM layers: every prefill pass must be \
                 grid-free (marconi_skip={marconi_skip})"
            );
        }
    }

    /// With no prefix cache there is no restore boundary to be equal across,
    /// so the cold pass keeps the faster chunked kernel.
    #[test]
    fn a_cold_pass_without_a_prefix_cache_keeps_the_chunked_kernel() {
        assert!(!decide(false, false, true, false));
        // ... and a warm replay is impossible, but if one is ever signalled it
        // still pins the token-sequential ladder.
        assert!(decide(true, false, true, false));
    }

    /// Nothing changes for a model with no GDN layers.
    #[test]
    fn a_model_without_ssm_layers_is_unaffected() {
        assert!(!decide(false, true, false, false));
        assert!(!decide(false, false, false, false));
    }

    /// The kill switch restores the pre-fix split — and only for cold passes.
    /// It must NOT re-enable FLA for the warm replay, which was already known
    /// to poison shared prefix-cache blocks (2026-06-10 warm-hit stutter).
    #[test]
    fn the_kill_switch_restores_the_split_but_never_unpins_the_replay() {
        assert!(!decide(false, true, true, true), "cold pass returns to FLA");
        assert!(
            decide(true, true, true, true),
            "warm replay stays token-sequential regardless of the kill switch"
        );
    }
}
