// SPDX-License-Identifier: AGPL-3.0-only

//! [`ModelStats`] — diagnostic counters and one-shot latches owned by the model.
//!
//! Sibling to [`ModelLevers`](super::ModelLevers): the levers say what a model's
//! kernels *do*, this records what they *did*. Both are owned by
//! `TransformerModel` and lent to every `ForwardContext`.
//!
//! Telemetry is not exempt from scoping. A counter that spans a model swap
//! averages two models together and describes neither, and a one-shot dump
//! latch that has already fired suppresses the *next* model's dump — the exact
//! artifact someone asked for by setting the flag. Nothing here changes
//! generation, so the failure is a wrong measurement rather than a wrong
//! answer; for a diagnostic those are the same kind of defect.
//!
//! Counters are atomics because they are mutated through the shared `&` that
//! `ForwardContext` hands out. The change from the statics they replace is
//! where they live, not how they are written.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Per-model diagnostic state.
#[derive(Debug, Default)]
pub struct ModelStats {
    /// MoE expert-union sampling (`ModelLevers::moe_union_stats`): calls seen,
    /// calls sampled, and the running unique-expert / slot totals behind the
    /// periodic aggregate line.
    pub moe_union: MoeUnionStats,
    /// One-shot latches for the `ATLAS_*_DUMP` diagnostics. A latch is per
    /// model so a swap re-arms the dump instead of silently swallowing it.
    pub dumped: DumpLatches,
}

/// Expert-union sampling counters for one model.
#[derive(Debug, Default)]
pub struct MoeUnionStats {
    pub calls: AtomicU64,
    pub samples: AtomicU64,
    pub unique_sum: AtomicU64,
    pub slots_sum: AtomicU64,
}

/// One-shot diagnostic latches for one model.
#[derive(Debug, Default)]
pub struct DumpLatches {
    /// Token-id dump (`impl_b3`) — fires on the first forward only.
    pub tokens: AtomicBool,
    /// FP8 weight-quantization diagnostic (`weight_map::loaders_fp8`) — a
    /// count, not a bool: it reports the first few tensors and then goes quiet.
    pub quant_diag: AtomicU64,
}

impl ModelStats {
    pub fn new() -> Self {
        Self::default()
    }
}

impl DumpLatches {
    /// `true` exactly once per model for a given latch.
    pub fn take(flag: &AtomicBool) -> bool {
        !flag.swap(true, Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_one_shot_latch_fires_once_per_model() {
        let a = ModelStats::new();
        assert!(DumpLatches::take(&a.dumped.tokens), "first forward dumps");
        assert!(!DumpLatches::take(&a.dumped.tokens), "and only the first");
    }

    #[test]
    fn a_second_model_re_arms_the_latch() {
        // The property a `static AtomicBool` could not have: after a swap, the
        // flag the operator set is still honoured. Held process-wide, the dump
        // they asked for is silently swallowed because the previous model
        // already consumed the single shot.
        let a = ModelStats::new();
        let b = ModelStats::new();
        assert!(DumpLatches::take(&a.dumped.tokens));
        assert!(DumpLatches::take(&b.dumped.tokens), "a new model dumps too");
    }

    #[test]
    fn two_models_count_expert_unions_independently() {
        let a = ModelStats::new();
        let b = ModelStats::new();
        a.moe_union.calls.fetch_add(9, Ordering::Relaxed);
        assert_eq!(a.moe_union.calls.load(Ordering::Relaxed), 9);
        assert_eq!(
            b.moe_union.calls.load(Ordering::Relaxed),
            0,
            "a second model starts clean rather than inheriting a mean"
        );
    }
}
