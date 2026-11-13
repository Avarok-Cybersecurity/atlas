// SPDX-License-Identifier: AGPL-3.0-only
// Design translated from vLLM (https://github.com/vllm-project/vllm, commit 8a36f08), vllm/v1/kv_offload/cpu/gpu_worker.py:543-731 (pooled start and end events per transfer, completion polled with event.query), Copyright contributors to the vLLM project, Apache-2.0; no code copied. Changes: Rust event pool over `GpuBackend` events, timing pairs only where the backend supports them, host clock fallback.

//! Responsibility: the tier's CUDA events. Completion events come from a free list (one per
//! in-flight write-back batch or restore, returned once its record is done). A [`Timer`] times
//! one transfer with two timing events when the backend has them (`create_timing_event`,
//! `event_elapsed_ms`), else by the host clock from submission to observed completion.

use std::time::Duration;

use anyhow::Result;

use crate::gpu::GpuBackend;

pub(crate) struct EventPool {
    free: Vec<u64>,
    /// None until the first timer asked; then whether timing events work here.
    timing: Option<bool>,
}

impl EventPool {
    pub(crate) fn new() -> Self {
        Self {
            free: Vec::new(),
            timing: None,
        }
    }

    pub(crate) fn take(&mut self, gpu: &dyn GpuBackend) -> Result<u64> {
        match self.free.pop() {
            Some(e) => Ok(e),
            None => gpu.create_event(),
        }
    }

    pub(crate) fn give(&mut self, ev: u64) {
        self.free.push(ev);
    }

    pub(crate) fn destroy_all(&mut self, gpu: &dyn GpuBackend) {
        for ev in self.free.drain(..) {
            let _ = gpu.destroy_event(ev);
        }
    }

    /// Start timing a transfer about to be enqueued on `stream`.
    pub(crate) fn start_timer(
        &mut self,
        gpu: &dyn GpuBackend,
        stream: u64,
        now: Duration,
    ) -> Timer {
        let mut timer = Timer {
            started: now,
            events: None,
        };
        if self.timing == Some(false) {
            return timer;
        }
        let pair = gpu
            .create_timing_event()
            .and_then(|a| match gpu.create_timing_event() {
                Ok(b) => Ok((a, b)),
                Err(e) => {
                    let _ = gpu.destroy_event(a);
                    Err(e)
                }
            });
        match pair {
            Ok((a, b)) => {
                if gpu.record_event(a, stream).is_ok() {
                    self.timing = Some(true);
                    timer.events = Some((a, b));
                } else {
                    let _ = gpu.destroy_event(a);
                    let _ = gpu.destroy_event(b);
                }
            }
            Err(_) => self.timing = Some(false),
        }
        timer
    }
}

/// Times one transfer. Drop it through [`Timer::elapsed_secs`] or [`Timer::discard`].
pub(crate) struct Timer {
    started: Duration,
    events: Option<(u64, u64)>,
}

impl Timer {
    /// Mark the end of the transfer on `stream` (enqueue BEFORE the completion event).
    pub(crate) fn stop(&mut self, gpu: &dyn GpuBackend, stream: u64) {
        let Some((a, b)) = self.events else {
            return;
        };
        if gpu.record_event(b, stream).is_err() {
            let _ = gpu.destroy_event(a);
            let _ = gpu.destroy_event(b);
            self.events = None;
        }
    }

    /// Seconds the transfer took, read once its completion event fired. Device time when
    /// timing events work, else host time from submission to `now` (an upper bound).
    pub(crate) fn elapsed_secs(&mut self, gpu: &dyn GpuBackend, now: Duration) -> f64 {
        let host = now.saturating_sub(self.started).as_secs_f64();
        let Some((a, b)) = self.events.take() else {
            return host;
        };
        let device = gpu.event_elapsed_ms(a, b);
        let _ = gpu.destroy_event(a);
        let _ = gpu.destroy_event(b);
        match device {
            Ok(Some(ms)) if ms.is_finite() && ms >= 0.0 => f64::from(ms) / 1e3,
            _ => host,
        }
    }

    /// Give the timing events back without reading them (failed or abandoned transfer).
    pub(crate) fn discard(&mut self, gpu: &dyn GpuBackend) {
        if let Some((a, b)) = self.events.take() {
            let _ = gpu.destroy_event(a);
            let _ = gpu.destroy_event(b);
        }
    }
}
