// SPDX-License-Identifier: AGPL-3.0-only

//! Background token-mask prewarm, overlapped with prompt prefill (#918).
//!
//! WHAT THIS MOVES OFF THE CRITICAL PATH
//! -------------------------------------
//! [`xgrammar::CompiledGrammar::compile_top_k_masks`] is the dominant
//! term of a cold grammar-constrained request. Measured on CPU (M5 Max,
//! release, Qwen3 ByteLevel-BPE tokenizer, 151,669-token vocabulary, the
//! coherency gate's `get_weather` schema through
//! `compile_qwen3_coder_tool_grammar`, single-shot):
//!
//! ```text
//! grammar construction      4.9-5.5 ms
//! mask prewarm          596.7-621.2 ms   (123 reachable scanable states)
//! matcher + first fill      0.1-0.3 ms
//! ```
//!
//! #918 reports ~3.2 s for the same phase on the H100 host at 248,320
//! tokens. `compile_grammar_state` runs BEFORE the prompt forward pass
//! (`prefill_a_step.rs` / `prefill_b_step.rs`), so all of it lands in
//! front of the first token even though the GPU is about to be busy for
//! 370-1,400 ms with the prefill itself. Running the prewarm on its own
//! thread hides that much of it; the first constrained sample joins.
//! Measured on the same CPU harness, moving it off the request thread
//! releases that thread after 4.9 ms instead of 624.8 ms (n=3), so the
//! whole remainder is available to overlap the forward pass.
//!
//! WHY THIS IS NOT A PARALLEL PREWARM
//! ----------------------------------
//! An H100 experiment that made the prewarm loop itself four-way
//! parallel was measured and abandoned: four workers contending on the
//! shared `Arc<GrammarData>` made cold latency WORSE (5.28-6.57 s vs
//! 4.82 s). This changes nothing about that loop — `compile_top_k_masks`
//! stays the serial JIT walk it is in tree, and `GrammarCompiler`'s
//! `max_threads` stays recorded-but-unused — it only moves the whole
//! serial walk off the request thread. No two workers ever touch one
//! grammar.
//!
//! Kill switch: `ATLAS_GRAMMAR_ASYNC_PREWARM=0` restores the synchronous
//! prewarm.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use xgrammar::CompiledGrammar;

use super::PrewarmHook;

/// A rendezvous the background worker waits on before doing any work.
///
/// Always `None` in production. The ordering test passes one so it can
/// hold the prewarm open and prove that (a) `GrammarState` construction
/// did not block on it and (b) the first mask fill did. It is a plain
/// field rather than a `#[cfg(test)]` global precisely so a gated state
/// in one test can never stall a `GrammarState` built anywhere else.
pub(super) type PrewarmGate = Arc<std::sync::Barrier>;

/// A prewarm that is either still running on its own thread or finished.
///
/// Dropping a `Pending` prewarm — an abandoned or cancelled request —
/// detaches the worker rather than cancelling it. That is deliberate:
/// the work is bounded (one `compile_top_k_masks` pass), it holds only
/// its own `CompiledGrammar` clone, and its result lands in the shared
/// cross-grammar cache where the NEXT request benefits from it.
pub(super) enum Prewarm {
    /// Running. The `JoinHandle` yields the number of masks warmed; the
    /// flag lets callers ask "done?" without joining.
    Pending {
        handle: std::thread::JoinHandle<usize>,
        finished: Arc<AtomicBool>,
    },
    /// Joined, or never spawned (kill switch / thread-spawn failure).
    Done(usize),
}

impl Prewarm {
    /// Start warming the `k` costliest masks of `compiled`.
    ///
    /// Returns immediately when the async path is enabled and a thread
    /// could be spawned; otherwise does the work inline, which is
    /// exactly the pre-#918 behaviour.
    pub(super) fn start(compiled: &CompiledGrammar, k: usize) -> Self {
        Self::start_with(compiled, k, None, overlap_enabled(), None)
    }

    /// The whole policy, with the overlap decision passed in rather
    /// than read from the environment — so a test can exercise BOTH
    /// paths without mutating a process-wide variable other tests in
    /// the same binary are concurrently reading.
    pub(super) fn start_with(
        compiled: &CompiledGrammar,
        k: usize,
        gate: Option<PrewarmGate>,
        overlap: bool,
        on_warm: Option<PrewarmHook>,
    ) -> Self {
        if !overlap {
            // The pre-#918 behaviour: warm inline, on this thread. The
            // gate is a background-worker rendezvous and has no meaning
            // here, so it is dropped rather than waited on.
            drop(gate);
            let warmed = compiled.compile_top_k_masks(k);
            if let Some(hook) = on_warm {
                hook(warmed);
            }
            return Prewarm::Done(warmed);
        }
        let owned = compiled.clone();
        let finished = Arc::new(AtomicBool::new(false));
        let done_flag = Arc::clone(&finished);
        let spawned = std::thread::Builder::new()
            .name("grammar-prewarm".to_string())
            .spawn(move || {
                if let Some(gate) = gate {
                    gate.wait();
                }
                let warmed = owned.compile_top_k_masks(k);
                // Persist AFTER the masks are in the shared cache but
                // BEFORE the finished flag: the request's first fill
                // then never races the snapshot write, and the write
                // still costs the request nothing — it is on this
                // thread, which the prefill is already hiding.
                if let Some(hook) = on_warm {
                    hook(warmed);
                }
                done_flag.store(true, Ordering::Release);
                warmed
            });
        match spawned {
            Ok(handle) => Prewarm::Pending { handle, finished },
            // Under thread pressure, do the work here rather than
            // leaving the masks cold — a cold decode loop costs ~41 ms
            // per token (see `FORCED_TOKEN_TOP_K`), far worse than a
            // slow prefill.
            Err(e) => {
                tracing::debug!("Grammar: prewarm thread spawn failed ({e}); warming inline");
                Prewarm::Done(compiled.compile_top_k_masks(k))
            }
        }
    }

    /// Block until the masks are warm. Idempotent; called at every site
    /// that is about to read a token mask.
    pub(super) fn wait(&mut self) -> usize {
        match self {
            Prewarm::Done(n) => *n,
            Prewarm::Pending { .. } => {
                let Prewarm::Pending { handle, .. } = std::mem::replace(self, Prewarm::Done(0))
                else {
                    unreachable!("just matched Pending");
                };
                // A panicked prewarm is not fatal: the mask cache is
                // only a cache, and the matcher recomputes lazily.
                let warmed = handle.join().unwrap_or_else(|_| {
                    tracing::warn!("Grammar: background mask prewarm panicked; masks stay lazy");
                    0
                });
                *self = Prewarm::Done(warmed);
                warmed
            }
        }
    }

    /// Whether the worker has already finished, without joining it.
    #[cfg(test)]
    pub(super) fn is_finished(&self) -> bool {
        match self {
            Prewarm::Done(_) => true,
            Prewarm::Pending { finished, .. } => finished.load(Ordering::Acquire),
        }
    }
}

/// The serve-path overlap decision. See [`overlap_enabled`].
pub(super) fn overlap_enabled_for_serve() -> bool {
    overlap_enabled()
}

/// Whether the prewarm overlaps prefill, read ONCE per process: the
/// answer cannot change while a server runs, and a `getenv` per
/// grammar-bearing request is pure overhead.
fn overlap_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        overlap_from_env(std::env::var("ATLAS_GRAMMAR_ASYNC_PREWARM").ok().as_deref())
    })
}

/// `ATLAS_GRAMMAR_ASYNC_PREWARM=0` (or `false`/`off`/`no`) restores the
/// synchronous prewarm. Anything else — including unset — overlaps.
pub(super) fn overlap_from_env(value: Option<&str>) -> bool {
    !matches!(
        value
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "off" | "no"
    )
}
