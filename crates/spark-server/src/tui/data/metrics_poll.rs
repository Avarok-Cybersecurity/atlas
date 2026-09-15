// SPDX-License-Identifier: AGPL-3.0-only

//! 1 Hz metrics sampler: diffs the monotone prometheus counters into rates,
//! reads the scheduler snapshot and the free-standing GPU/memory accessors,
//! and keeps bounded histories for the Stats sparklines/charts.

use std::collections::VecDeque;
use std::time::Instant;

use crate::metrics;
use crate::scheduler::snapshot::SchedulerSnapshot;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Bounded numeric history for a sparkline/chart.
#[derive(Default)]
pub struct History {
    pub points: VecDeque<f64>,
    cap: usize,
}

impl History {
    fn new(cap: usize) -> Self {
        Self {
            points: VecDeque::with_capacity(cap),
            cap,
        }
    }
    fn push(&mut self, v: f64) {
        if self.points.len() == self.cap {
            self.points.pop_front();
        }
        self.points.push_back(v);
    }
    pub fn as_u64(&self) -> Vec<u64> {
        self.points
            .iter()
            .map(|v| (v.max(0.0) * 10.0) as u64)
            .collect()
    }
}

/// One TTFT histogram snapshot: (upper_bound_secs, cumulative_count).
pub type TtftBuckets = Vec<(f64, u64)>;

pub struct StatsModel {
    // Totals (straight reads).
    pub requests_total: u64,
    pub requests_active: i64,
    pub gen_tokens_total: u64,
    pub prompt_tokens_total: u64,
    pub tool_calls_total: u64,
    pub bytes_in_total: u64,
    pub bytes_out_total: u64,
    // Rates (diffed per sample).
    pub gen_tps: f64,
    pub prompt_tps: f64,
    pub req_rate: f64,
    pub bytes_in_rate: f64,
    pub bytes_out_rate: f64,
    // Histories.
    pub gen_tps_history: History,
    /// Prefill ingest rate history. Kept SEPARATE from `gen_tps_history`
    /// rather than sharing the throughput chart: prefill runs ~890 tok/s
    /// against ~40 tok/s for generation, so on one axis the generation line
    /// flattens to the baseline and stops being readable.
    pub prompt_tps_history: History,
    pub req_history: History,
    pub queue_history: History,
    pub entropy_history: History,
    // TTFT.
    pub ttft_buckets: TtftBuckets,
    pub ttft_p50_ms: Option<f64>,
    pub ttft_p90_ms: Option<f64>,
    // Cache & sampler.
    pub prefix_hit_rate: Option<f64>,
    pub prefix_hit_tokens: u64,
    pub entropy: f64,
    // Spec decode accept per K: (k_label, accepted, total).
    pub spec_accept: Vec<(String, u64, u64)>,
    // Memory.
    /// True once a real device reading has been taken.
    ///
    /// ★ The three figures below are plain `f64` and default to 0.0, so on a
    /// box with no GPU or no NVML they render as `atlas 0.0 GB · free 0.0`
    /// with a 0 % gauge — a MEASUREMENT OF ZERO rather than "unavailable".
    /// This file already gets that right for TTFT (an `Option` that renders as
    /// `—`); the GPU tile did not. Nothing else on this dashboard fabricates a
    /// number, which is why the fabricated one stood out.
    pub gpu_known: bool,
    pub gpu_free_gb: f64,
    pub gpu_total_gb: f64,
    pub atlas_used_gb: f64,
    pub host_avail_gb: f64,
    pub host_total_gb: f64,
    // Scheduler.
    pub sched: Option<SchedulerSnapshot>,
    // Internals.
    last_sample: Option<(Instant, Prev)>,
}

#[derive(Clone, Copy)]
struct Prev {
    requests: u64,
    gen_tok: u64,
    decoded: u64,
    prompt: u64,
    bytes_in: u64,
    bytes_out: u64,
}

impl Default for StatsModel {
    fn default() -> Self {
        Self {
            requests_total: 0,
            requests_active: 0,
            gen_tokens_total: 0,
            prompt_tokens_total: 0,
            tool_calls_total: 0,
            bytes_in_total: 0,
            bytes_out_total: 0,
            gen_tps: 0.0,
            prompt_tps: 0.0,
            req_rate: 0.0,
            bytes_in_rate: 0.0,
            bytes_out_rate: 0.0,
            gen_tps_history: History::new(120),
            prompt_tps_history: History::new(120),
            req_history: History::new(16),
            queue_history: History::new(24),
            entropy_history: History::new(24),
            ttft_buckets: Vec::new(),
            ttft_p50_ms: None,
            ttft_p90_ms: None,
            prefix_hit_rate: None,
            prefix_hit_tokens: 0,
            entropy: 0.0,
            spec_accept: Vec::new(),
            gpu_known: false,
            gpu_free_gb: 0.0,
            gpu_total_gb: 0.0,
            atlas_used_gb: 0.0,
            host_avail_gb: 0.0,
            host_total_gb: 0.0,
            sched: None,
            last_sample: None,
        }
    }
}

/// Percentile from cumulative histogram buckets (upper-bound interpolation).
fn hist_percentile(buckets: &TtftBuckets, p: f64) -> Option<f64> {
    let total = buckets.last().map(|(_, c)| *c)?;
    if total == 0 {
        return None;
    }
    let rank = (total as f64 * p).ceil() as u64;
    for (ub, cum) in buckets {
        if *cum >= rank {
            return Some(ub * 1000.0);
        }
    }
    None
}

impl StatsModel {
    /// Take one sample. Call at ~1 Hz from the TUI event loop.
    pub fn sample(&mut self, run: Option<&crate::tui::RunHandles>) {
        let now = Instant::now();
        self.requests_total = metrics::REQUESTS_TOTAL.get();
        self.requests_active = metrics::REQUESTS_ACTIVE.get();
        self.gen_tokens_total = metrics::GENERATION_TOKENS_TOTAL.get();
        let decoded_total = metrics::DECODED_TOKENS_TOTAL.get();
        self.prompt_tokens_total = metrics::PROMPT_TOKENS_TOTAL.get();
        self.tool_calls_total = metrics::TOOL_CALLS_TOTAL.get();
        self.bytes_in_total = metrics::HTTP_BYTES_IN.get();
        self.bytes_out_total = metrics::HTTP_BYTES_OUT.get();

        if let Some((t0, prev)) = self.last_sample {
            let dt = now.duration_since(t0).as_secs_f64().max(0.05);
            // Prefer the per-token counter: it advances DURING generation, so
            // this is a live rate. `GENERATION_TOKENS_TOTAL` only moves when a
            // request finishes, which reads as 0 tok/s then one spike — fall
            // back to it only for paths that never stream (blocking requests),
            // where a lump at completion is all there is.
            let decoded_delta = decoded_total.saturating_sub(prev.decoded);
            let gen_delta = self.gen_tokens_total.saturating_sub(prev.gen_tok);
            self.gen_tps = decoded_delta.max(gen_delta) as f64 / dt;
            self.prompt_tps = (self.prompt_tokens_total.saturating_sub(prev.prompt)) as f64 / dt;
            self.req_rate = (self.requests_total.saturating_sub(prev.requests)) as f64 / dt;
            self.bytes_in_rate = (self.bytes_in_total.saturating_sub(prev.bytes_in)) as f64 / dt;
            self.bytes_out_rate = (self.bytes_out_total.saturating_sub(prev.bytes_out)) as f64 / dt;
            self.gen_tps_history.push(self.gen_tps);
            self.prompt_tps_history.push(self.prompt_tps);
            self.req_history.push(self.req_rate);
        }
        self.last_sample = Some((
            now,
            Prev {
                requests: self.requests_total,
                gen_tok: self.gen_tokens_total,
                decoded: decoded_total,
                prompt: self.prompt_tokens_total,
                bytes_in: self.bytes_in_total,
                bytes_out: self.bytes_out_total,
            },
        ));

        // TTFT histogram via the prometheus proto (bucket bounds + counts).
        self.ttft_buckets.clear();
        for mf in prometheus::gather() {
            if mf.name() != "atlas_time_to_first_token_seconds" {
                continue;
            }
            if let Some(m) = mf.get_metric().first() {
                for b in m.get_histogram().get_bucket() {
                    self.ttft_buckets
                        .push((b.upper_bound(), b.cumulative_count()));
                }
            }
        }
        self.ttft_p50_ms = hist_percentile(&self.ttft_buckets, 0.50);
        self.ttft_p90_ms = hist_percentile(&self.ttft_buckets, 0.90);

        // Spec-decode accept per K from the labeled counter vec.
        self.spec_accept = spec_accept_from_gather();

        // Prefix cache + sampler globals. Scoped to the CURRENT model: the
        // cumulative counters behind these are what /metrics exports, and
        // after a swap they still carry the previous model's cache activity,
        // which is not what this pane is describing.
        let (hits, misses, hit_tokens) = spark_runtime::run_metrics::cache_counts_this_run();
        self.prefix_hit_tokens = hit_tokens;
        self.prefix_hit_rate = (hits + misses > 0).then(|| hits as f64 / (hits + misses) as f64);
        self.entropy = spark_runtime::sampler::last_entropy() as f64;
        self.entropy_history.push(self.entropy);

        // Memory.
        // Both reads must land: `atlas_used` is a DIFFERENCE of the two, so
        // one without the other is not a smaller truth, it is a wrong number.
        match (
            super::gpu_free_bytes(),
            spark_runtime::gpu::baseline_free_bytes(),
        ) {
            (Some(free), Some(baseline)) => {
                self.gpu_free_gb = free as f64 / GIB;
                self.gpu_total_gb = baseline as f64 / GIB;
                self.atlas_used_gb = (self.gpu_total_gb - self.gpu_free_gb).max(0.0);
                self.gpu_known = true;
            }
            _ => self.gpu_known = false,
        }
        if let Some((avail, total)) = host_mem_gb() {
            self.host_avail_gb = avail;
            self.host_total_gb = total;
        }

        // Scheduler snapshot.
        self.sched = run.and_then(|r| r.snapshot.read());
        if let Some(s) = self.sched {
            self.queue_history.push(s.pending_len as f64);
        }
    }
}

fn spec_accept_from_gather() -> Vec<(String, u64, u64)> {
    use std::collections::BTreeMap;
    let mut per_k: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for mf in prometheus::gather() {
        if mf.name() != "atlas_spec_decode_verify_total" {
            continue;
        }
        for m in mf.get_metric() {
            let mut k = String::new();
            let mut outcome = String::new();
            for l in m.get_label() {
                match l.name() {
                    "k" => k = l.value().to_string(),
                    "outcome" => outcome = l.value().to_string(),
                    _ => {}
                }
            }
            let e = per_k.entry(k).or_default();
            let v = m.get_counter().get_value() as u64;
            e.1 += v;
            if outcome.contains("accept") {
                e.0 += v;
            }
        }
    }
    per_k.into_iter().map(|(k, (a, t))| (k, a, t)).collect()
}

/// (available, total) host RAM in GB, from /proc/meminfo.
fn host_mem_gb() -> Option<(f64, f64)> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let grab = |key: &str| -> Option<f64> {
        text.lines()
            .find(|l| l.starts_with(key))?
            .split_whitespace()
            .nth(1)?
            .parse::<f64>()
            .ok()
            .map(|kb| kb / (1024.0 * 1024.0))
    };
    Some((grab("MemAvailable:")?, grab("MemTotal:")?))
}
