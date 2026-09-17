// SPDX-License-Identifier: AGPL-3.0-only

//! How the mHC (hyper-connection) decode bodies dispatch the FFN across an
//! `n`-row decode batch.
//!
//! Both mHC layer kinds used to run the MoE FFN as a PER-SEQUENCE loop, so the
//! single most expensive component of a decode step scaled linearly with the
//! concurrency `C` even though the scheduler dispatches the batch once:
//!
//!   * `qwen3_ssm/trait_decode_multi_seq/hc.rs`      — fused only at `n == 2`
//!   * `qwen3_attention/trait_impl/multi_seq/mod.rs` — per-row for every `n`
//!
//! Measured on the EXL3 2.05bpw qwen4_exp checkpoint (dgx-00, distinct-prompt
//! harness, ISL 1024 / OSL 256) by routing the same batch through the existing
//! batched MoE arm: C=4 aggregate 15.1 -> 23.9 tok/s (step p50 217 -> 122 ms),
//! C=8 16.5 -> 32.6 tok/s (step p50 404 -> 159 ms), C=1 unchanged.
//!
//! The dispatch decision is a pure function of `(n, has_ffn, kill switch,
//! site)` so it can be asserted on the CPU-only CI runner, away from device
//! code. `n == 1` deliberately stays on the per-row path: the single-token
//! `FfnComponent::forward` kernels are the shipping C=1 numerics and this
//! change must not perturb them.

/// The FFN dispatch shape chosen for one mHC decode step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HcFfnPlan {
    /// No FFN component on this layer (Nemotron-H standalone attention): the
    /// mixed rows pass through to `hc_post` unchanged, exactly as the per-row
    /// loop's `FfnComponent::forward` (which returns its input) left them.
    Passthrough,
    /// One `FfnComponent::forward` per row — `n` dispatches. The pre-existing
    /// behaviour, kept for `n == 1` (byte-for-byte) and for the kill switch.
    PerRow,
    /// The fused token-PAIR kernel (`FfnComponent::forward_k2`), output at
    /// `moe_output()[0..2)`. Already batched, already shipping — routing
    /// `n == 2` anywhere else would change its numerics for no measured win
    /// (the C=2 A/B moved ~5%, all of it from the attention layers).
    FusedK2,
    /// One `FfnComponent::forward_token_major_decode(n)` for the whole batch,
    /// output at `moe_output()[0..n)`. That entry delegates per arm — EXL3
    /// native -> `forward_exl3_decode`, LoRA-resident and FP8/BF16/T-layout ->
    /// `forward_batched`, NVFP4 -> the token-major kernels, dense FFN ->
    /// `forward_batched` — so every arm the per-row loop could serve keeps a
    /// defined n-row path, and each issues exactly ONE expert-parallel
    /// all-reduce for the whole batch instead of `n` single-row collectives.
    Batched,
}

impl HcFfnPlan {
    /// Number of FFN dispatches this plan issues for an `n`-row batch.
    ///
    /// This is the quantity the defect was about: the per-row loop issues `n`
    /// (48 sublayers × n single-row MoE forwards per step); every batched plan
    /// issues one.
    pub fn ffn_dispatches(self, n: usize) -> usize {
        match self {
            Self::Passthrough => 0,
            Self::PerRow => n,
            Self::FusedK2 | Self::Batched => 1,
        }
    }
}

/// Pick the FFN dispatch shape for an mHC decode step.
///
/// * `n` — rows in this step (the scheduler's `padded_batch_n`, so 1, 2, 4, 8,
///   12, 16, 24, 32, … — never 3).
/// * `has_ffn` — `!FfnComponent::is_none()`.
/// * `batched` — the `hc_batched_moe_decode` model-config key (default true;
///   false restores the per-row loop).
/// * `fused_k2_at_2` — keep the shipping fused K=2 kernel at `n == 2`. True on
///   the GDN/SSM site (where that arm already ran, and whose numerics stay
///   byte-identical); false on the attention site, which had no fused arm at
///   all and was measured on the batched one.
pub fn hc_ffn_plan(n: usize, has_ffn: bool, batched: bool, fused_k2_at_2: bool) -> HcFfnPlan {
    if !has_ffn {
        return HcFfnPlan::Passthrough;
    }
    // n == 1: the single-token arm is the C=1 numerics contract.
    // !batched: the operator kill switch.
    if n <= 1 || !batched {
        return HcFfnPlan::PerRow;
    }
    if fused_k2_at_2 && n == 2 {
        return HcFfnPlan::FusedK2;
    }
    HcFfnPlan::Batched
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE regression this module exists for: at C=8 the FFN must be dispatched
    /// ONCE, not eight times, on both mHC layer kinds.
    #[test]
    fn batch_of_eight_is_one_dispatch_on_both_sites() {
        for fused_k2_at_2 in [true, false] {
            let plan = hc_ffn_plan(8, true, true, fused_k2_at_2);
            assert_eq!(plan, HcFfnPlan::Batched);
            assert_eq!(plan.ffn_dispatches(8), 1);
        }
    }

    /// Every rung of the padding ladder above 2 batches (`n == 3` is
    /// unreachable — `padded_batch_n` maps it to 4 — but is correct here).
    #[test]
    fn every_ladder_rung_above_two_batches() {
        for n in [3usize, 4, 8, 12, 16, 24, 32, 48, 64] {
            let plan = hc_ffn_plan(n, true, true, true);
            assert_eq!(plan, HcFfnPlan::Batched, "n={n}");
            assert_eq!(plan.ffn_dispatches(n), 1, "n={n}");
        }
    }

    /// C=1 keeps the single-token arm, on both sites.
    #[test]
    fn single_row_stays_per_row() {
        assert_eq!(hc_ffn_plan(1, true, true, true), HcFfnPlan::PerRow);
        assert_eq!(hc_ffn_plan(1, true, true, false), HcFfnPlan::PerRow);
        assert_eq!(hc_ffn_plan(1, true, true, true).ffn_dispatches(1), 1);
    }

    /// `n == 2` keeps the shipping fused pair kernel on the GDN site and takes
    /// the batched arm on the attention site (which never had a fused arm).
    #[test]
    fn pair_batch_respects_the_site() {
        assert_eq!(hc_ffn_plan(2, true, true, true), HcFfnPlan::FusedK2);
        assert_eq!(hc_ffn_plan(2, true, true, false), HcFfnPlan::Batched);
        assert_eq!(hc_ffn_plan(2, true, true, true).ffn_dispatches(2), 1);
    }

    /// Kill switch (`hc_batched_moe_decode = false`) restores the per-row loop
    /// at every n — the assertion that fails if the key is ignored.
    #[test]
    fn kill_switch_restores_the_per_row_loop() {
        for n in [1usize, 2, 4, 8, 16] {
            let plan = hc_ffn_plan(n, true, false, true);
            assert_eq!(plan, HcFfnPlan::PerRow, "n={n}");
            assert_eq!(plan.ffn_dispatches(n), n, "n={n}");
        }
    }

    /// A layer with no FFN (standalone attention) routes its rows straight to
    /// `hc_post` and dispatches nothing — with or without the kill switch.
    #[test]
    fn no_ffn_layer_passes_through() {
        for batched in [true, false] {
            for n in [1usize, 2, 8] {
                let plan = hc_ffn_plan(n, false, batched, true);
                assert_eq!(plan, HcFfnPlan::Passthrough, "n={n} batched={batched}");
                assert_eq!(plan.ffn_dispatches(n), 0, "n={n}");
            }
        }
    }
}
