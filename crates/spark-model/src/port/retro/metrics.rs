// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the `atlas_retro_*` `/metrics` series of the RetroInfer
//! port (master plan 4.2: the namespace is L33's), kept in this module's
//! atomics and formatted here (spark-model is below spark-server, so it writes
//! its own `# HELP` / `# TYPE` lines in the `port_metrics` style). With the
//! row off every series renders as zero, and the labeled families render their
//! fixed label sets, so scrapers see a stable shape on every target.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Why a decode layer call ran dense instead of retro attention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    /// The context is below `--retro-min-context`.
    ShortContext,
    /// No index for the sequence (not built yet, or no room in the arena).
    NoIndex,
    /// The arena could not hold the index or the step's buffers.
    Budget,
    /// Some rows of the batch are served and some are not.
    MixedBatch,
    /// The step is being captured into a CUDA graph (retro runs eager).
    Graph,
    /// A retro kernel failed or is absent; retro is off for the serve.
    Kernel,
    /// The layer's geometry or the verify width is not served.
    Shape,
}

impl Fallback {
    pub const ALL: [Fallback; 7] = [
        Fallback::ShortContext,
        Fallback::NoIndex,
        Fallback::Budget,
        Fallback::MixedBatch,
        Fallback::Graph,
        Fallback::Kernel,
        Fallback::Shape,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Fallback::ShortContext => "short_context",
            Fallback::NoIndex => "no_index",
            Fallback::Budget => "budget",
            Fallback::MixedBatch => "mixed_batch",
            Fallback::Graph => "graph",
            Fallback::Kernel => "kernel",
            Fallback::Shape => "shape",
        }
    }
}

/// Running means of the shadow measures for one layer.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShadowAgg {
    pub samples: u64,
    pub mass_exact: f64,
    pub mass_total: f64,
    pub out_rel_err: f64,
    pub jaccard: f64,
}

/// The series' state. `GLOBAL` is the serve's; tests build their own.
#[derive(Default)]
pub struct Metrics {
    active: AtomicU64,
    decode: AtomicU64,
    fallback: [AtomicU64; 7],
    build_nanos: AtomicU64,
    index_bytes: AtomicU64,
    /// Touched ratio times 1e6.
    touched_ppm: AtomicU64,
    updates: AtomicU64,
    union_size: AtomicU64,
    shadow: Mutex<BTreeMap<u32, ShadowAgg>>,
}

pub static GLOBAL: std::sync::LazyLock<Metrics> = std::sync::LazyLock::new(Metrics::default);

impl Metrics {
    pub fn set_active(&self, on: bool) {
        self.active.store(u64::from(on), Ordering::Relaxed);
    }
    pub fn count_decode(&self, n: u64) {
        self.decode.fetch_add(n, Ordering::Relaxed);
    }
    pub fn count_fallback(&self, why: Fallback) {
        let i = Fallback::ALL.iter().position(|f| *f == why).unwrap_or(0);
        self.fallback[i].fetch_add(1, Ordering::Relaxed);
    }
    pub fn fallback(&self, why: Fallback) -> u64 {
        let i = Fallback::ALL.iter().position(|f| *f == why).unwrap_or(0);
        self.fallback[i].load(Ordering::Relaxed)
    }
    pub fn add_build_secs(&self, s: f64) {
        self.build_nanos
            .fetch_add((s.max(0.0) * 1e9) as u64, Ordering::Relaxed);
    }
    pub fn build_secs(&self) -> f64 {
        self.build_nanos.load(Ordering::Relaxed) as f64 / 1e9
    }
    pub fn set_index_bytes(&self, b: u64) {
        self.index_bytes.store(b, Ordering::Relaxed);
    }
    pub fn set_touched_ratio(&self, r: f64) {
        self.touched_ppm
            .store((r.clamp(0.0, 1.0) * 1e6) as u64, Ordering::Relaxed);
    }
    pub fn count_update(&self) {
        self.updates.fetch_add(1, Ordering::Relaxed);
    }
    pub fn updates(&self) -> u64 {
        self.updates.load(Ordering::Relaxed)
    }
    pub fn decodes(&self) -> u64 {
        self.decode.load(Ordering::Relaxed)
    }
    pub fn set_union_size(&self, n: u64) {
        self.union_size.store(n, Ordering::Relaxed);
    }
    /// Fold one shadow sample of `layer` into its running means.
    pub fn record_shadow(
        &self,
        layer: u32,
        mass_exact: f64,
        mass_total: f64,
        out_rel_err: f64,
        jaccard: f64,
    ) {
        let mut m = self
            .shadow
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let a = m.entry(layer).or_default();
        let n = a.samples as f64;
        let upd = |old: f64, new: f64| (old * n + new) / (n + 1.0);
        a.mass_exact = upd(a.mass_exact, mass_exact);
        a.mass_total = upd(a.mass_total, mass_total);
        a.out_rel_err = upd(a.out_rel_err, out_rel_err);
        a.jaccard = upd(a.jaccard, jaccard);
        a.samples += 1;
    }
    pub fn shadow(&self, layer: u32) -> Option<ShadowAgg> {
        self.shadow
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&layer)
            .copied()
    }

    /// Append the series to the Prometheus text.
    pub fn render(&self, out: &mut String) {
        let ld = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let one = |out: &mut String, name: &str, kind: &str, help: &str, v: String| {
            let _ = writeln!(
                out,
                "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {v}"
            );
        };
        one(
            out,
            "atlas_retro_active",
            "gauge",
            "1 while RetroInfer retrieval attention replaces dense decode attention (approximate, opt-in).",
            ld(&self.active).to_string(),
        );
        one(
            out,
            "atlas_retro_decode_total",
            "counter",
            "Decode attention-layer calls served by RetroInfer retrieval attention.",
            ld(&self.decode).to_string(),
        );
        let _ = writeln!(
            out,
            "# HELP atlas_retro_dense_fallback_total Decode attention-layer calls that ran dense while retro was on, by reason.\n# TYPE atlas_retro_dense_fallback_total counter"
        );
        for (f, a) in Fallback::ALL.iter().zip(&self.fallback) {
            let _ = writeln!(
                out,
                "atlas_retro_dense_fallback_total{{reason=\"{}\"}} {}",
                f.label(),
                ld(a)
            );
        }
        one(
            out,
            "atlas_retro_index_build_seconds_total",
            "counter",
            "Seconds spent building and updating RetroInfer indexes.",
            format!("{:.6}", self.build_secs()),
        );
        one(
            out,
            "atlas_retro_index_bytes",
            "gauge",
            "Device bytes of the retro arena in use (live indexes, the index cache, step buffers).",
            ld(&self.index_bytes).to_string(),
        );
        one(
            out,
            "atlas_retro_tokens_touched_ratio",
            "gauge",
            "Exact tokens plus estimation centroids over the context, last retro decode call.",
            format!("{:.6}", ld(&self.touched_ppm) as f64 / 1e6),
        );
        let shadow = self
            .shadow
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let _ = writeln!(
            out,
            "# HELP atlas_retro_shadow_mass_recall Shadow mode: mean dense softmax mass inside the exact zone (zone=\"retrieval\") and the exact plus estimation zones (zone=\"retrieval_estimation\").\n# TYPE atlas_retro_shadow_mass_recall gauge"
        );
        for (l, a) in &shadow {
            let _ = writeln!(
                out,
                "atlas_retro_shadow_mass_recall{{layer=\"{l}\",zone=\"retrieval\"}} {:.6}",
                a.mass_exact
            );
            let _ = writeln!(
                out,
                "atlas_retro_shadow_mass_recall{{layer=\"{l}\",zone=\"retrieval_estimation\"}} {:.6}",
                a.mass_total
            );
        }
        let _ = writeln!(
            out,
            "# HELP atlas_retro_shadow_out_rel_err Shadow mode: mean relative L2 error of retro attention against dense attention.\n# TYPE atlas_retro_shadow_out_rel_err gauge"
        );
        for (l, a) in &shadow {
            let _ = writeln!(
                out,
                "atlas_retro_shadow_out_rel_err{{layer=\"{l}\"}} {:.6}",
                a.out_rel_err
            );
        }
        let _ = writeln!(
            out,
            "# HELP atlas_retro_shadow_jaccard Shadow mode: mean Jaccard of the exact-zone tokens against the dense top tokens.\n# TYPE atlas_retro_shadow_jaccard gauge"
        );
        for (l, a) in &shadow {
            let _ = writeln!(
                out,
                "atlas_retro_shadow_jaccard{{layer=\"{l}\"}} {:.6}",
                a.jaccard
            );
        }
        one(
            out,
            "atlas_retro_update_total",
            "counter",
            "Decode-time RetroInfer index updates (one update interval clustered and appended).",
            ld(&self.updates).to_string(),
        );
        one(
            out,
            "atlas_retro_verify_union_size",
            "gauge",
            "Retrieval clusters in the union of the last speculative verify step's rows.",
            ld(&self.union_size).to_string(),
        );
    }
}

/// Append the serve's `atlas_retro_*` series (the `port_metrics::render` line).
pub fn render_prometheus(out: &mut String) {
    GLOBAL.render(out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_every_series_at_zero() {
        let m = Metrics::default();
        let mut s = String::new();
        m.render(&mut s);
        for name in [
            "atlas_retro_active 0",
            "atlas_retro_decode_total 0",
            "atlas_retro_dense_fallback_total{reason=\"short_context\"} 0",
            "atlas_retro_dense_fallback_total{reason=\"graph\"} 0",
            "atlas_retro_index_build_seconds_total 0.000000",
            "atlas_retro_index_bytes 0",
            "atlas_retro_tokens_touched_ratio 0.000000",
            "# TYPE atlas_retro_shadow_mass_recall gauge",
            "# TYPE atlas_retro_shadow_out_rel_err gauge",
            "# TYPE atlas_retro_shadow_jaccard gauge",
            "atlas_retro_update_total 0",
            "atlas_retro_verify_union_size 0",
        ] {
            assert!(s.contains(name), "missing `{name}` in\n{s}");
        }
    }

    #[test]
    fn shadow_means_and_fallback_labels() {
        let m = Metrics::default();
        m.record_shadow(3, 0.9, 0.98, 0.02, 0.5);
        m.record_shadow(3, 0.7, 0.96, 0.04, 0.3);
        m.count_fallback(Fallback::MixedBatch);
        let a = m.shadow(3).unwrap();
        assert_eq!(a.samples, 2);
        assert!((a.mass_exact - 0.8).abs() < 1e-12);
        assert!((a.out_rel_err - 0.03).abs() < 1e-12);
        let mut s = String::new();
        m.render(&mut s);
        assert!(s.contains(
            "atlas_retro_shadow_mass_recall{layer=\"3\",zone=\"retrieval_estimation\"} 0.970000"
        ));
        assert!(s.contains("atlas_retro_dense_fallback_total{reason=\"mixed_batch\"} 1"));
    }
}
