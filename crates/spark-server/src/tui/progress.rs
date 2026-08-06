// SPDX-License-Identifier: AGPL-3.0-only

//! ProgressModel — the Main tab's startup state machine.
//!
//! Fed typed [`ProgressEvent`]s from the capture layer; renders nothing
//! itself. Tracks the 12-phase checklist with per-phase wall times, the
//! GB-weighted overall bar, the current-shard bar, layer progress, a GPU
//! memory history for the MEM sparkline, and a load-rate/ETA estimate.

use std::time::Instant;

use super::capture_layer::ProgressEvent;

/// Display names for the 12 serve() phases, in fixed order. Indexes match the
//  `phase` field emitted by the instrumented call sites.
pub const PHASE_NAMES: [&str; 12] = [
    "banner",
    "model resolve",
    "config",
    "gpu init",
    "topology",
    "weight load",
    "kv cache",
    "kernel audit",
    "tokenizer",
    "scheduler",
    "router",
    "listening",
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PhaseState {
    Pending,
    Running,
    Done,
}

#[derive(Clone, Debug)]
pub struct Phase {
    pub name: &'static str,
    pub state: PhaseState,
    pub started: Option<Instant>,
    pub secs: f64,
}

/// Startup progress, ready to render.
pub struct ProgressModel {
    pub phases: Vec<Phase>,
    pub started_at: Instant,
    /// Weight-load denominator from preflight (GB on disk).
    pub disk_gb: f64,
    pub shard: u64,
    pub shard_total: u64,
    pub shard_name: String,
    pub layer: u64,
    pub layer_total: u64,
    pub gpu_used_gb: f64,
    pub gpu_free_gb: f64,
    /// GPU-used history for the MEM sparkline (bounded).
    pub mem_history: Vec<u64>,
    /// Ready flag + final port; flips the whole UI into SERVING.
    pub ready: bool,
    pub port: u16,
    pub ready_in_secs: f64,
    /// The weight-load window, which is what GB/s must be measured over.
    /// `load_started` is stamped by the FIRST shard event -- NOT process start,
    /// which would fold CUDA init, model resolution and preflight into the
    /// divisor and under-report the rate. `load_secs` freezes the window when the
    /// last shard lands; without it a finished load keeps dividing a constant
    /// number of bytes by a growing elapsed time, so the displayed rate decays
    /// toward zero for as long as the server runs.
    load_started: Option<Instant>,
    load_secs: Option<f64>,
    /// Smoothed displayed fractions (motion spec: d += (t-d)*0.35).
    disp_overall: f64,
    disp_shard: f64,
    last_shard_seen: u64,
}

impl Default for ProgressModel {
    fn default() -> Self {
        Self {
            phases: PHASE_NAMES
                .iter()
                .map(|n| Phase {
                    name: n,
                    state: PhaseState::Pending,
                    started: None,
                    secs: 0.0,
                })
                .collect(),
            started_at: Instant::now(),
            disk_gb: 0.0,
            shard: 0,
            shard_total: 0,
            shard_name: String::new(),
            layer: 0,
            layer_total: 0,
            gpu_used_gb: 0.0,
            gpu_free_gb: 0.0,
            mem_history: Vec::new(),
            ready: false,
            port: 0,
            ready_in_secs: 0.0,
            load_started: None,
            load_secs: None,
            disp_overall: 0.0,
            disp_shard: 0.0,
            last_shard_seen: 0,
        }
    }
}

impl ProgressModel {
    /// Start over for a new model load.
    ///
    /// **Deliberately `Self::default()`, not a field-by-field clear.** A load
    /// leaves state in every field — `ready`, the per-phase `Done` marks, the
    /// frozen `load_secs` window, `started_at` — and a hand-written reset that
    /// misses one renders the *second* load as already finished. Reassigning
    /// the whole struct cannot miss a field, and a field added later is reset
    /// for free.
    ///
    /// Without this, `enter_phase` only advances `Pending → Running`, so a
    /// phase re-entered after the first load never leaves `Done`; `ready` is
    /// never cleared; and `freeze_load_window` keeps the FIRST close, so the
    /// second load's GB/s is never measured.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn apply(&mut self, ev: ProgressEvent) {
        match ev {
            ProgressEvent::Phase { phase, .. } => self.enter_phase(phase as usize),
            ProgressEvent::Preflight { disk_gb, free_gb } => {
                self.disk_gb = disk_gb;
                self.gpu_free_gb = free_gb;
            }
            ProgressEvent::ShardStart { shard, total, name } => {
                // First shard opens the load window.
                self.load_started.get_or_insert_with(Instant::now);
                self.shard = shard;
                self.shard_total = total;
                self.shard_name = name;
                if shard != self.last_shard_seen {
                    // Shard rollover snaps to 0 (a backward-easing bar reads
                    // as an error, per the motion spec).
                    self.disp_shard = 0.0;
                    self.last_shard_seen = shard;
                }
            }
            ProgressEvent::ShardDone {
                shard,
                total,
                used_gb,
                free_gb,
            } => {
                self.shard = shard;
                self.shard_total = total;
                self.gpu_used_gb = used_gb;
                self.gpu_free_gb = free_gb;
                self.mem_history.push((used_gb * 10.0) as u64);
                if self.mem_history.len() > 64 {
                    self.mem_history.remove(0);
                }
                self.disp_shard = 1.0;
                // `shard` is 1-based: the last shard done closes the window.
                if total > 0 && shard >= total {
                    self.freeze_load_window();
                }
            }
            ProgressEvent::Layer { layer, total } => {
                self.layer = layer;
                self.layer_total = total;
            }
            ProgressEvent::Ready { port } => {
                self.ready = true;
                self.port = port;
                self.ready_in_secs = self.started_at.elapsed().as_secs_f64();
                // Backstop: a load that never emits its final shard_done still
                // stops the clock here rather than running forever.
                self.freeze_load_window();
                for p in &mut self.phases {
                    if p.state != PhaseState::Done {
                        Self::finish(p);
                    }
                }
            }
        }
    }

    fn enter_phase(&mut self, idx: usize) {
        for (i, p) in self.phases.iter_mut().enumerate() {
            match i.cmp(&idx) {
                std::cmp::Ordering::Less => {
                    if p.state != PhaseState::Done {
                        Self::finish(p);
                    }
                }
                std::cmp::Ordering::Equal => {
                    if p.state == PhaseState::Pending {
                        p.state = PhaseState::Running;
                        p.started = Some(Instant::now());
                    }
                }
                std::cmp::Ordering::Greater => {}
            }
        }
    }

    fn finish(p: &mut Phase) {
        if let Some(s) = p.started {
            p.secs = s.elapsed().as_secs_f64();
        }
        p.state = PhaseState::Done;
    }

    /// Overall weight-load fraction, GB-weighted when the preflight total is
    /// known, else shard-count-weighted.
    pub fn overall_target(&self) -> f64 {
        if self.shard_total == 0 {
            return if self.ready { 1.0 } else { 0.0 };
        }
        (self.shard as f64 / self.shard_total as f64).clamp(0.0, 1.0)
    }

    pub fn shard_target(&self) -> f64 {
        // Within-shard tensor progress isn't streamed; the shard bar advances
        // start(0) -> done(1), smoothed by the easing below.
        self.disp_shard
    }

    /// Advance displayed fractions one tick (10 Hz): d += (t-d)*0.35.
    pub fn ease_tick(&mut self) {
        let t = self.overall_target();
        self.disp_overall += (t - self.disp_overall) * 0.35;
        if (t - self.disp_overall).abs() < 0.002 {
            self.disp_overall = t;
        }
    }

    pub fn displayed_overall(&self) -> f64 {
        self.disp_overall
    }

    /// Stop the load clock, keeping the first close (later events must not extend
    /// a window that is already measured).
    fn freeze_load_window(&mut self) {
        if self.load_secs.is_none()
            && let Some(t) = self.load_started
        {
            self.load_secs = Some(t.elapsed().as_secs_f64());
        }
    }

    /// Seconds the weight load took, once it is over. `None` while still loading.
    pub fn load_secs(&self) -> Option<f64> {
        self.load_secs
    }

    /// Load rate (GB/s) and ETA seconds, both measured over the weight-load window
    /// only. Once the window is frozen the rate is a fixed measurement of what the
    /// load actually achieved, and the ETA is 0.
    pub fn rate_eta(&self) -> Option<(f64, f64)> {
        if self.disk_gb <= 0.0 || self.shard == 0 || self.shard_total == 0 {
            return None;
        }
        let elapsed = match self.load_secs {
            Some(s) => s,
            None => self.load_started?.elapsed().as_secs_f64(),
        }
        .max(0.1);
        let frac = (self.shard as f64 / self.shard_total as f64).clamp(0.0, 1.0);
        let gb_done = self.disk_gb * frac;
        let rate = gb_done / elapsed;
        if rate <= 0.0 {
            return None;
        }
        Some((rate, (self.disk_gb - gb_done) / rate))
    }

    /// Counts for the panel title: (done, total, cumulative secs).
    pub fn phase_counts(&self) -> (usize, usize, f64) {
        let done = self
            .phases
            .iter()
            .filter(|p| p.state == PhaseState::Done)
            .count();
        let secs = self.started_at.elapsed().as_secs_f64();
        (done, self.phases.len(), secs)
    }
}

#[cfg(test)]
#[path = "progress_tests.rs"]
mod tests;
