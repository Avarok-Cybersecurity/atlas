// SPDX-License-Identifier: AGPL-3.0-only

//! Model-level shared launch state for EVERY cooperative EXL3 launch — the
//! MoE arm, the dense GDN/attention arms AND the LM head: the split-K locks
//! buffer, the once-resolved SM count,
//! and the ordering machinery that keeps two cooperative/spin-barrier
//! kernels from ever being partially co-resident on the device.
//!
//! Why one state per model (MoE arm AND dense GDN/attention arms share it):
//!
//!  * The gemm/mgemm split-K spinlocks live in the `locks` buffer, so two
//!    in-flight GEMMs on one locks buffer corrupt each other's counters.
//!  * Cooperative launches need full co-residency; a persistent MoE kernel
//!    holding most SMs while a cooperative dense GEMM waits for the rest (or
//!    the reverse) is the deadlock class the MoE milestone hit in serving.
//!  * Atlas runs prefill and decode on DIFFERENT CUDA streams, overlapping
//!    at C >= 2, and Q12 cache co-dispatch runs two prefills from two host
//!    threads. Host-side ordering alone therefore proves nothing about
//!    device-side ordering.
//!
//! The rule is one DISPATCH SECTION at a time, globally:
//!
//!  1. Host: a `Mutex` held for the whole section (RAII [`Exl3Section`]).
//!     A second host thread BLOCKS until the section ends — it is never
//!     refused. Sections only enqueue asynchronous launches (the MoE prefill
//!     tier's single D2H waits on its own stream's work, which never depends
//!     on the blocked thread), so blocking cannot deadlock the host.
//!  2. Device: a CUDA event `fence` is recorded on the section's stream when
//!     the section ends. A later section on a DIFFERENT stream first makes
//!     its stream wait on the fence — host serialization orders the
//!     enqueues, the fence orders the execution. Same-stream sections are
//!     already stream-ordered and skip the wait (the common decode case).
//!
//! Nested sections on ONE thread would self-deadlock on the mutex, so entry
//! detects re-entrancy through a thread-local flag and fails loudly.
//!
//! Cooperative launches are never CUDA-graph-capturable: every arm that
//! dispatches through this state must sit behind the model's
//! `exl3_graph_veto` and refuse `graph_capture` itself.

use std::cell::Cell;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops::exl3_matmul::{EXL3_LOCKS_BYTES, exl3_locks_alloc};

thread_local! {
    /// `true` while this thread holds an [`Exl3Section`] — re-entrancy guard.
    static IN_SECTION: Cell<bool> = const { Cell::new(false) };
}

/// One per model. Allocated at load, inside the util pledge (locks are 4.2
/// MB); shared by `Exl3MoeState` and every dense stage through an `Arc`.
#[derive(Debug)]
pub struct Exl3LaunchState {
    /// Per-model cooperative-launch locks ([`EXL3_LOCKS_BYTES`], zeroed
    /// once; the kernels' protocols self-reset — never re-zero).
    pub locks: DevicePtr,
    /// Resolved once at construction (the trait forbids per-launch queries).
    pub sm_count: u32,
    /// CUDA event recorded on the dispatching stream at the end of every
    /// section. GPU-side only — the host never blocks on it.
    fence: u64,
    /// Host serialization + the stream the most recent section launched on
    /// (0 = none yet). Guarded by the same mutex the section holds.
    last_stream: Mutex<u64>,
}

/// RAII token for one dispatch section — see [`Exl3LaunchState::section`].
/// Dropping it records the fence on the section's stream, remembers the
/// stream, and releases the host claim (in that order: the mutex guard is a
/// field, so it drops after this type's `Drop` body).
pub struct Exl3Section<'a> {
    st: &'a Exl3LaunchState,
    gpu: &'a dyn GpuBackend,
    stream: u64,
    guard: MutexGuard<'a, u64>,
}

impl Drop for Exl3Section<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.gpu.record_event(self.st.fence, self.stream) {
            // Cannot propagate from Drop; the next cross-stream section would
            // then wait on a stale fence, so make the failure visible.
            tracing::error!(
                "EXL3 launch state: fence record failed on stream {:#x}: {e}",
                self.stream
            );
        }
        *self.guard = self.stream;
        IN_SECTION.with(|f| f.set(false));
    }
}

impl Exl3Section<'_> {
    /// The stream this section launches on.
    pub fn stream(&self) -> u64 {
        self.stream
    }
}

impl Exl3LaunchState {
    /// Allocate + zero the locks buffer, create the fence event, resolve
    /// the SM count. All-or-nothing.
    pub fn new(gpu: &dyn GpuBackend) -> Result<Self> {
        let sm_count = gpu.sm_count()?;
        let locks = exl3_locks_alloc(gpu)?;
        let fence = match gpu.create_event() {
            Ok(e) => e,
            Err(e) => {
                gpu.free(locks).ok();
                return Err(e);
            }
        };
        tracing::info!(
            "EXL3 launch state allocated: locks {} KB + fence event (ONE dispatch \
             section at a time across MoE + dense arms; cross-stream sections \
             fence on the device)",
            EXL3_LOCKS_BYTES / 1024
        );
        Ok(Self {
            locks,
            sm_count,
            fence,
            last_stream: Mutex::new(0),
        })
    }

    /// Get the model-shared state, creating it on first use (the loader
    /// threads one `Option` cache through its per-layer loop, like
    /// `Exl3MoeState::get_or_create`). Resolves through [`Self::shared`] so
    /// a holder built outside the loader (the LM head, in the factory) lands
    /// on the SAME state.
    pub fn get_or_create(
        cache: &mut Option<Arc<Exl3LaunchState>>,
        gpu: &dyn GpuBackend,
    ) -> Result<Arc<Exl3LaunchState>> {
        if let Some(s) = cache {
            return Ok(s.clone());
        }
        let s = Self::shared(gpu)?;
        *cache = Some(s.clone());
        Ok(s)
    }

    /// The process's live launch state, created if none is alive. Held only
    /// WEAKLY here: every strong holder (MoE state, dense stage, LM head)
    /// belongs to one model, so when that model is dropped the state dies
    /// with it and the next model builds a fresh one against its own
    /// backend — no stale locks buffer survives a hot-swap. Atlas serves one
    /// model per process/GPU, which is what makes a process-wide anchor the
    /// right scope: the loader (layers) and the factory (LM head) build
    /// their pieces in different places and must agree on ONE section
    /// mutex / fence / locks buffer, or the "one cooperative section at a
    /// time" invariant is not global.
    pub fn shared(gpu: &dyn GpuBackend) -> Result<Arc<Exl3LaunchState>> {
        static SHARED: Mutex<Weak<Exl3LaunchState>> = Mutex::new(Weak::new());
        let mut slot = SHARED.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(s) = slot.upgrade() {
            return Ok(s);
        }
        let s = Arc::new(Self::new(gpu)?);
        *slot = Arc::downgrade(&s);
        Ok(s)
    }

    /// Claim the shared state for one dispatch section on `stream`. Blocks
    /// while another host thread holds a section (normal under co-dispatched
    /// prefill at C >= 2 — never refused). A stream change is normal too
    /// (prefill vs decode streams) and is made safe by ordering this stream
    /// behind the previous section's fence on the device. Re-entry from the
    /// thread that already holds a section is a contract breach and fails
    /// loudly instead of self-deadlocking.
    pub fn section<'a>(&'a self, gpu: &'a dyn GpuBackend, stream: u64) -> Result<Exl3Section<'a>> {
        if IN_SECTION.with(|f| f.replace(true)) {
            bail!(
                "EXL3 launch state: nested dispatch section on one thread (an arm \
                 opened a section while another was still held — take ONE section \
                 around the whole layer dispatch)"
            );
        }
        let guard = self
            .last_stream
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let prev = *guard;
        if prev != 0
            && prev != stream
            && let Err(e) = gpu.stream_wait_event(stream, self.fence)
        {
            IN_SECTION.with(|f| f.set(false));
            return Err(e);
        }
        Ok(Exl3Section {
            st: self,
            gpu,
            stream,
            guard,
        })
    }

    /// Free the locks + destroy the fence. Without an explicit caller this
    /// is reclaimed by `sweep_unreleased` at teardown (documented backstop).
    /// Only call once every `Arc` holder is gone.
    pub fn release(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.locks)?;
        gpu.destroy_event(self.fence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    #[test]
    fn section_serializes_and_fences_on_stream_change() {
        let gpu = MockGpuBackend::new();
        let st = Exl3LaunchState::new(&gpu).unwrap();
        {
            let s = st.section(&gpu, 0x10).unwrap();
            assert_eq!(s.stream(), 0x10);
            // Re-entry on the same thread must fail, not deadlock.
            assert!(st.section(&gpu, 0x10).is_err());
        }
        assert_eq!(*st.last_stream.lock().unwrap(), 0x10);
        // A second section on another stream is admitted (it waits on the
        // fence device-side; the mock accepts the wait).
        let s2 = st.section(&gpu, 0x20).unwrap();
        drop(s2);
        assert_eq!(*st.last_stream.lock().unwrap(), 0x20);
        // The re-entrancy flag is cleared after a refused entry too.
        let s3 = st.section(&gpu, 0x20).unwrap();
        drop(s3);
    }
}
