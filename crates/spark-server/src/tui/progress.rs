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
mod tests {
    use super::*;

    #[test]
    fn phase_entry_closes_earlier_phases() {
        let mut m = ProgressModel::default();
        m.apply(ProgressEvent::Phase {
            phase: 0,
            name: "banner".into(),
        });
        m.apply(ProgressEvent::Phase {
            phase: 3,
            name: "gpu init".into(),
        });
        assert_eq!(m.phases[0].state, PhaseState::Done);
        assert_eq!(m.phases[1].state, PhaseState::Done);
        assert_eq!(m.phases[3].state, PhaseState::Running);
        assert_eq!(m.phases[4].state, PhaseState::Pending);
    }

    #[test]
    fn ready_completes_everything() {
        let mut m = ProgressModel::default();
        m.apply(ProgressEvent::Phase {
            phase: 5,
            name: "weight load".into(),
        });
        m.apply(ProgressEvent::Ready { port: 8888 });
        assert!(m.ready);
        assert!(m.phases.iter().all(|p| p.state == PhaseState::Done));
    }

    #[test]
    fn shard_rollover_snaps_shard_bar() {
        let mut m = ProgressModel::default();
        m.apply(ProgressEvent::ShardDone {
            shard: 1,
            total: 4,
            used_gb: 2.0,
            free_gb: 100.0,
        });
        assert_eq!(m.shard_target(), 1.0);
        m.apply(ProgressEvent::ShardStart {
            shard: 2,
            total: 4,
            name: "s2".into(),
        });
        assert_eq!(m.shard_target(), 0.0);
    }

    /// GB/s must be measured over the weight-load window, not since process start,
    /// and must stop moving once the load is over. Previously it divided a constant
    /// number of bytes by an ever-growing elapsed time, so a finished load's rate
    /// decayed toward zero for as long as the server ran -- and the pre-load time
    /// (CUDA init, model resolution, preflight) was in the divisor throughout.
    #[test]
    fn load_rate_is_windowed_and_freezes_at_the_last_shard() {
        let ms = |n| std::thread::sleep(std::time::Duration::from_millis(n));
        let mut m = ProgressModel::default();
        m.apply(ProgressEvent::Preflight {
            disk_gb: 10.0,
            free_gb: 100.0,
        });
        ms(120); // pre-load work that must NOT count against the rate
        for shard in 1..=2u64 {
            m.apply(ProgressEvent::ShardStart {
                shard,
                total: 2,
                name: "s".into(),
            });
            ms(60);
            m.apply(ProgressEvent::ShardDone {
                shard,
                total: 2,
                used_gb: 1.0,
                free_gb: 9.0,
            });
        }
        let secs = m.load_secs().expect("last shard closes the window");
        assert!(
            secs < m.started_at.elapsed().as_secs_f64() - 0.1,
            "window {secs}s must exclude the 120ms of pre-load work"
        );
        let (rate, eta) = m.rate_eta().expect("rate known once shards are in");
        assert_eq!(eta, 0.0, "nothing left to load");
        assert!(rate > 0.0);

        ms(120);
        let (rate_later, _) = m.rate_eta().unwrap();
        assert_eq!(rate, rate_later, "a finished load's rate must not drift");
    }
}
