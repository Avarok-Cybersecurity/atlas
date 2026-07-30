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
///
/// The named fields are the latches with a caller that already holds the
/// struct; [`keyed`](DumpLatches::keyed) covers the long tail of
/// `ATLAS_*_DUMP` gates, which are numerous, scattered, and identical in
/// shape — a field each would be noise, and a static each is the bug.
#[derive(Debug, Default)]
pub struct DumpLatches {
    /// Ad-hoc latches, keyed by a `&'static str` naming the dump.
    fired: std::sync::Mutex<std::collections::BTreeSet<&'static str>>,
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

    /// `true` the first time THIS model reaches `key`, `false` after.
    ///
    /// The general per-model latch. Log-dedup gates used a `static Once` each,
    /// which is correct for "print this line once" and wrong for "print this
    /// line once per model": after a swap the new model's kernel-route and
    /// fallback lines — the ones that say which path a model actually took —
    /// were suppressed by the previous model's shot, and those lines exist to
    /// be read when a model behaves unexpectedly.
    ///
    /// Namespace keys by purpose (`"log:..."`, `"dump:..."`) so two unrelated
    /// sites cannot collide.
    pub fn once(&self, key: &'static str) -> bool {
        self.dumped.keyed(key)
    }
}

impl DumpLatches {
    /// `true` exactly once per model for a given latch.
    pub fn take(flag: &AtomicBool) -> bool {
        !flag.swap(true, Ordering::Relaxed)
    }

    /// `true` exactly once per model for `key`. Use for the `ATLAS_*_DUMP`
    /// gates that would otherwise each grow a `static AtomicBool`.
    ///
    /// Call it only when the dump is actually wanted — it consumes the shot,
    /// so gating on the env var FIRST keeps a disabled dump from burning it.
    pub fn keyed(&self, key: &'static str) -> bool {
        self.fired
            .lock()
            .expect("dump latches poisoned")
            .insert(key)
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
    fn a_keyed_latch_fires_once_per_key_per_model() {
        let a = ModelStats::new();
        assert!(a.dumped.keyed("dflash_block"));
        assert!(!a.dumped.keyed("dflash_block"), "and only once");
        assert!(a.dumped.keyed("dflash_ctx"), "a different dump is separate");
        assert!(
            ModelStats::new().dumped.keyed("dflash_block"),
            "and a new model re-arms every key"
        );
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
