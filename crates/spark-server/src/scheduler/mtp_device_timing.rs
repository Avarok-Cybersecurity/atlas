// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: device-event phase timing of the MTP round
//! (`AVAROK_MTP_TIMING=2`), the part of the "unexplained MTP round cost"
//! investigation that host `Instant`s cannot see.
//!
//! # What it measures
//!
//! A *mark* records a CUDA event on the compute stream and closes an interval:
//! the time from the previous mark to this one is charged to the mark's
//! [`Phase`] (the phase that just ENDED). Because the events sit in the stream,
//! the interval is what the stream experienced, GPU execution plus any idle
//! gap the host left (a host pipeline that makes the GPU wait shows up as the
//! interval that contains it). [`Phase::Start`] opens a step and charges the
//! time since the last mark of the previous step to `gap`; so the step
//! decomposes into gap, propose, verify, readback, pipeline, commit,
//! bootstrap and other, with no overlap and nothing lost.
//!
//! # Cost
//!
//! Off (the default) a mark is one `OnceLock` read and a branch: no event, no
//! lock. On, a mark is one `cuEventRecord` and a short uncontended lock. No
//! host sync is ever added: events are read back with `cuEventElapsedTime`,
//! and one whose end is not yet complete simply waits for the next
//! [`Phase::Start`] poll. Windows of [`SUMMARY_PERIOD`] steps are logged and
//! added to `atlas_mtp_device_phase_microseconds_total{phase}`. A backend that
//! cannot time events (no `create_timing_event`) disables the timer with one
//! log line; nothing else changes.
//!
//! # Wiring
//!
//! * `verify_k2_step.rs` (the K=2 path, today's default) marks every phase;
//!   `verify_k3_step.rs` carries one [`step_start`] line, which times the
//!   round cadence (`gap`) only.
//! * Batched C=8 verify and the K=4 serial path are marked by the lanes that
//!   own them (`mark(Phase::.., model, stream)`); the call accepts a `Model`
//!   or a `GpuBackend` through [`DeviceClock`].
//!
//! The clock goes through `ModelPortExt::timing_event_*` for a `Model`
//! (`model/trait_impl/port_ext.rs`) and `GpuBackend::{create_timing_event,
//! record_event, event_elapsed_ms}` underneath.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use spark_model::traits::Model;
use spark_runtime::gpu::GpuBackend;

use crate::api::port_metrics as pm;

/// Steps per logged window.
pub const SUMMARY_PERIOD: u32 = 25;
/// Closed windows kept waiting for their events to complete; older ones are
/// dropped (and their events recycled) so a lagging stream cannot grow memory.
const MAX_PENDING: usize = 16;
const BUCKETS: usize = 8;

/// The phase an interval is charged to when its closing mark is placed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Phase {
    /// Step entry. Charges the time since the previous step's last mark to `gap`.
    Start = 0,
    /// The drafter chain (propose steps, grammar mask).
    Propose,
    /// The verify forward.
    Verify,
    /// The device-to-host readback of the verify result.
    Readback,
    /// The host per-position pipeline (dequant, processors, penalties, argmax).
    Pipeline,
    /// The SSM prefix commit.
    Commit,
    /// A bootstrap step (decode plus propose).
    Bootstrap,
    /// Everything else inside a step (emit, state bookkeeping).
    Other,
}

impl Phase {
    /// The 3.13 brief's name for [`Phase::Verify`] ("verify forward"); the batched
    /// verify step (lane L10) spells it this way.
    #[allow(non_upper_case_globals)]
    pub const VerifyForward: Phase = Phase::Verify;
}

const NAMES: [&str; BUCKETS] = [
    "gap",
    "propose",
    "verify",
    "readback",
    "pipeline",
    "commit",
    "bootstrap",
    "other",
];

/// What the timer needs from a device: timing-enabled events and their
/// distance. Implemented for `dyn GpuBackend` and `dyn Model`, so a call site
/// passes whichever handle it holds.
pub trait DeviceClock {
    /// A new event with timing enabled; `Err` when the device cannot time.
    fn timing_event_create(&self) -> anyhow::Result<u64>;
    /// Record `event` on `stream`.
    fn timing_event_record(&self, event: u64, stream: u64) -> anyhow::Result<()>;
    /// Milliseconds from `start` to `end`; `Ok(None)` while `end` is incomplete.
    fn timing_event_elapsed_ms(&self, start: u64, end: u64) -> anyhow::Result<Option<f32>>;
}

impl<'a> DeviceClock for dyn GpuBackend + 'a {
    fn timing_event_create(&self) -> anyhow::Result<u64> {
        self.create_timing_event()
    }
    fn timing_event_record(&self, event: u64, stream: u64) -> anyhow::Result<()> {
        self.record_event(event, stream)
    }
    fn timing_event_elapsed_ms(&self, start: u64, end: u64) -> anyhow::Result<Option<f32>> {
        self.event_elapsed_ms(start, end)
    }
}

impl<'a> DeviceClock for dyn Model + 'a {
    fn timing_event_create(&self) -> anyhow::Result<u64> {
        self.ext()?.timing_event_create()
    }
    fn timing_event_record(&self, event: u64, stream: u64) -> anyhow::Result<()> {
        self.ext()?.timing_event_record(event, stream)
    }
    fn timing_event_elapsed_ms(&self, start: u64, end: u64) -> anyhow::Result<Option<f32>> {
        self.ext()?.timing_event_elapsed_ms(start, end)
    }
}

/// `Model::port_ext`, or an error naming why this model cannot time events.
trait ModelExt {
    fn ext(&self) -> anyhow::Result<&dyn spark_model::traits::port_ext::ModelPortExt>;
}

impl<'a> ModelExt for dyn Model + 'a {
    fn ext(&self) -> anyhow::Result<&dyn spark_model::traits::port_ext::ModelPortExt> {
        self.port_ext()
            .ok_or_else(|| anyhow::anyhow!("this model has no port extension (no timing events)"))
    }
}

#[derive(Default)]
struct Inner {
    /// A clock error disabled the timer for the rest of the process.
    dead: bool,
    /// Completed events ready for reuse.
    pool: Vec<u64>,
    /// The open window: `(closing phase, event)`; entry 0 is the origin.
    seg: Vec<(Option<Phase>, u64)>,
    /// Non-closing `Start` marks in the open window after its origin.
    starts: u32,
    /// Closed windows waiting for their last event to complete.
    pending: VecDeque<Vec<(Option<Phase>, u64)>>,
}

/// The phase timer. The process has one ([`mark`]); tests build their own.
pub struct DeviceTimer {
    inner: Mutex<Inner>,
    total_us: [AtomicU64; BUCKETS],
    steps: AtomicU64,
    windows: AtomicU64,
}

impl Default for DeviceTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceTimer {
    pub const fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                dead: false,
                pool: Vec::new(),
                seg: Vec::new(),
                starts: 0,
                pending: VecDeque::new(),
            }),
            total_us: [const { AtomicU64::new(0) }; BUCKETS],
            steps: AtomicU64::new(0),
            windows: AtomicU64::new(0),
        }
    }

    /// Record a mark. See the module docs for what it charges.
    pub fn mark<C: DeviceClock + ?Sized>(&self, phase: Phase, clock: &C, stream: u64) {
        let mut guard = self.inner.lock();
        let st = &mut *guard;
        if st.dead {
            return;
        }
        if phase == Phase::Start {
            self.poll(st, clock);
        }
        let Some(ev) = Self::record(st, clock, stream) else {
            return;
        };
        if st.seg.is_empty() {
            // The first mark of a window is its origin: nothing is charged to it.
            st.seg.push((None, ev));
            return;
        }
        if phase == Phase::Start && st.starts + 1 >= SUMMARY_PERIOD {
            // The `SUMMARY_PERIOD`-th `Start` after the origin closes this
            // window (its gap is the last interval of the last step); the next
            // window starts from a fresh origin event recorded at the same point.
            st.seg.push((Some(phase), ev));
            let Some(origin) = Self::record(st, clock, stream) else {
                return;
            };
            let done = std::mem::replace(&mut st.seg, vec![(None, origin)]);
            st.starts = 0;
            if st.pending.len() >= MAX_PENDING
                && let Some(old) = st.pending.pop_front()
            {
                st.pool.extend(old.iter().map(|&(_, e)| e));
            }
            st.pending.push_back(done);
            return;
        }
        if phase == Phase::Start {
            st.starts += 1;
        }
        st.seg.push((Some(phase), ev));
    }

    fn record<C: DeviceClock + ?Sized>(st: &mut Inner, clock: &C, stream: u64) -> Option<u64> {
        let ev = match st.pool.pop() {
            Some(e) => e,
            None => match clock.timing_event_create() {
                Ok(e) => e,
                Err(e) => {
                    Self::kill(st, &e);
                    return None;
                }
            },
        };
        if let Err(e) = clock.timing_event_record(ev, stream) {
            Self::kill(st, &e);
            return None;
        }
        Some(ev)
    }

    fn kill(st: &mut Inner, e: &anyhow::Error) {
        st.dead = true;
        st.seg.clear();
        st.pending.clear();
        tracing::info!("MTP device timing disabled: {e:#}");
    }

    /// Read back every closed window whose last event has completed.
    fn poll<C: DeviceClock + ?Sized>(&self, st: &mut Inner, clock: &C) {
        while let Some(front) = st.pending.front() {
            if let [.., prev, last] = front.as_slice() {
                match clock.timing_event_elapsed_ms(prev.1, last.1) {
                    Ok(None) => return,
                    Err(e) => {
                        Self::kill(st, &e);
                        return;
                    }
                    Ok(Some(_)) => {}
                }
            }
            let seg = st.pending.pop_front().unwrap_or_default();
            let mut us = [0.0f64; BUCKETS];
            let mut steps = 0u32;
            for w in seg.windows(2) {
                let Some(phase) = w[1].0 else { continue };
                if phase == Phase::Start {
                    steps += 1;
                }
                if let Ok(Some(ms)) = clock.timing_event_elapsed_ms(w[0].1, w[1].1) {
                    us[phase as usize] += f64::from(ms) * 1000.0;
                }
            }
            st.pool.extend(seg.iter().map(|&(_, e)| e));
            self.publish(&us, steps);
        }
    }

    fn publish(&self, us: &[f64; BUCKETS], steps: u32) {
        for (total, v) in self.total_us.iter().zip(us) {
            total.fetch_add(v.round() as u64, Ordering::Relaxed);
        }
        self.steps.fetch_add(u64::from(steps), Ordering::Relaxed);
        self.windows.fetch_add(1, Ordering::Relaxed);
        use std::fmt::Write as _;
        let mut line = String::new();
        for (name, v) in NAMES.iter().zip(us) {
            if *v > 0.0 {
                let _ = write!(
                    line,
                    " {name}={:.2}ms",
                    v / 1000.0 / f64::from(steps.max(1))
                );
            }
        }
        tracing::info!("MTP device timing [{steps} steps, per step]:{line}");
    }

    /// Microseconds charged to each phase so far, in [`Phase`] order.
    pub fn totals_us(&self) -> [u64; BUCKETS] {
        std::array::from_fn(|i| self.total_us[i].load(Ordering::Relaxed))
    }

    fn render(&self, out: &mut String) {
        let totals = self.totals_us();
        let labels: Vec<String> = NAMES.iter().map(|n| format!("phase=\"{n}\"")).collect();
        let series: Vec<(&str, u64)> = labels.iter().map(String::as_str).zip(totals).collect();
        pm::labeled_counter(
            out,
            "atlas_mtp_device_phase_microseconds_total",
            "Stream time of MTP rounds by phase (AVAROK_MTP_TIMING=2)",
            &series,
        );
        pm::counter(
            out,
            "atlas_mtp_device_phase_steps_total",
            "MTP steps covered by the device phase timer",
            self.steps.load(Ordering::Relaxed),
        );
    }
}

static TIMER: DeviceTimer = DeviceTimer::new();

/// Is `AVAROK_MTP_TIMING=2` set? Read once.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| super::mtp_timing::env_level() >= 2)
}

/// Record a phase mark on `stream` through `clock` (a `Model` or a
/// `GpuBackend`). One branch when `AVAROK_MTP_TIMING=2` is not set.
pub fn mark<C: DeviceClock + ?Sized>(phase: Phase, clock: &C, stream: u64) {
    if enabled() {
        TIMER.mark(phase, clock, stream);
    }
}

/// [`mark`] on a model's default (compute) stream: the one-line call site.
pub fn mark_model(phase: Phase, model: &dyn Model) {
    if enabled() {
        TIMER.mark(phase, model, model.default_stream());
    }
}

/// Open a step ([`Phase::Start`]).
pub fn step_start(model: &dyn Model) {
    mark_model(Phase::Start, model);
}

/// The `/metrics` series of the process timer.
pub fn render_prometheus(out: &mut String) {
    TIMER.render(out);
}

#[cfg(test)]
#[path = "mtp_device_timing_tests.rs"]
mod tests;
