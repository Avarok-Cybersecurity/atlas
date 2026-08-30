// SPDX-License-Identifier: AGPL-3.0-only

//! The pure half of the concurrency sweep's self-verdict — the C1 pattern
//! already applied to bfcl and decode-floor (commit a4837acb4), in a sibling
//! file for the 500-LoC cap. No endpoint, no I/O: every verdict path is
//! provable in unit tests.
//!
//! Under `--pull-request-gate` the floors are auto-filled from the selected
//! variant's BENCH.toml `min` bounds via `threshold_params`, so a run that
//! clears its committed ladder gets the PASS verdict the gate machinery
//! requires. All floors at the schema default 0.0 keep the standalone info
//! verdict exactly as before this existed.
//!
//! ★ Two rules the floors can never override:
//! * request ERRORS fail the affected sweep — an errored cell's numbers are
//!   not comparable, gating or not;
//! * VACUOUS cells make a gating run INCONCLUSIVE (a failing verdict), never
//!   PASS: an aggregate that divides undelivered tokens' wall time into real
//!   tokens cannot clear a throughput floor, however large it prints.
//! * a requested warm path without a material observed cached-prompt fraction
//!   is likewise INCONCLUSIVE: matching request bytes do not prove the cache
//!   served them, and a small shared template prefix is not the named setup.

use std::collections::BTreeMap;

use crate::result::Verdict;

/// Run-verdict floors on the ladder's aggregate tok/s, gate-filled per
/// variant. 0.0 is the documented OFF state for each (PCND) — a floor nobody
/// set gates nothing.
#[derive(Clone, Debug, Default)]
pub(crate) struct Floors {
    /// `(C, min aggregate tok/s)` for each gated rung.
    pub per_c: Vec<(usize, f64)>,
    /// Floor on `peak_aggregate_tok_s`.
    pub peak: f64,
}

impl Floors {
    /// Whether any floor is populated — the info/self-verdict switch.
    pub(crate) fn gating(&self) -> bool {
        self.peak > 0.0 || self.per_c.iter().any(|(_, f)| *f > 0.0)
    }
}

/// The sweep's run verdict. Pure over the metrics map the gate itself reads,
/// so the verdict and the record can never disagree about a rung's value.
///
/// ★ Deliberately STRICTER than gate scoring: this demands the raw value at
/// or above the floor, while `gate::scoring` allows value + noise to clear
/// the min. A sub-noise dip fails the run verdict even though scoring would
/// have passed it — safe conservatism, same as decode-floor's `verdict_for`.
pub(crate) fn sweep_verdict(
    metrics: &BTreeMap<String, f64>,
    cells: usize,
    errors: usize,
    vacuous: usize,
    cache_uncontrolled: usize,
    vacuity_floor_pct: f64,
    floors: &Floors,
) -> Verdict {
    if errors > 0 {
        // Errors invalidate the cells they landed in, gating or not.
        return Verdict::fail(format!(
            "{errors} request(s) failed — affected rows are not comparable"
        ));
    }
    if !floors.gating() {
        // The pre-gate behaviour, verbatim: a standalone sweep has no
        // committed ladder to be judged against.
        return if vacuous > 0 || cache_uncontrolled > 0 {
            Verdict::info(format!(
                "{cells} cells, {vacuous} below the vacuity floor ({vacuity_floor_pct:.0}% \
                 of osl), {cache_uncontrolled} without sufficient observed warm-cache use — flagged \
                 rows' tok/s are not comparable"
            ))
        } else {
            Verdict::info(format!(
                "{cells} cells, no request errors, all above the vacuity floor"
            ))
        };
    }
    if vacuous > 0 {
        return Verdict::fail(format!(
            "INCONCLUSIVE: {vacuous} of {cells} cells below the vacuity floor \
             ({vacuity_floor_pct:.0}% of osl) — undelivered tokens cannot clear a \
             throughput floor, whatever the aggregate prints"
        ));
    }
    if cache_uncontrolled > 0 {
        return Verdict::fail(format!(
            "INCONCLUSIVE: {cache_uncontrolled} of {cells} cells requested warm-up but a \
             measured request did not report a material cached-prompt fraction"
        ));
    }
    let mut basis = Vec::new();
    for (c, floor) in floors.per_c.iter().filter(|(_, f)| *f > 0.0) {
        let key = format!("c{c}_aggregate_tok_s");
        let Some(value) = metrics.get(&key) else {
            // A gated rung the sweep never measured comparably must not pass
            // by omission — the floor demands the measurement, not just the
            // absence of a bad one.
            return Verdict::fail(format!(
                "INCONCLUSIVE: the C={c} floor is set ({floor:.1} tok/s) but the sweep \
                 produced no comparable C={c} cell to judge it on"
            ));
        };
        if *value < *floor {
            return Verdict::fail(format!(
                "BELOW THE C={c} FLOOR — {value:.1} aggregate tok/s vs the {floor:.1} floor"
            ));
        }
        basis.push(format!("C{c} {value:.1}/{floor:.1}"));
    }
    if floors.peak > 0.0 {
        let Some(value) = metrics.get("peak_aggregate_tok_s") else {
            return Verdict::fail(format!(
                "INCONCLUSIVE: the peak floor is set ({:.1} tok/s) but no comparable cell \
                 produced a peak to judge it on",
                floors.peak
            ));
        };
        if *value < floors.peak {
            return Verdict::fail(format!(
                "BELOW THE PEAK FLOOR — {value:.1} aggregate tok/s vs the {:.1} floor",
                floors.peak
            ));
        }
        basis.push(format!("peak {value:.1}/{:.1}", floors.peak));
    }
    Verdict::pass(format!(
        "{cells} cells, zero errors, zero vacuous — every populated floor met \
         ({})",
        basis.join(" · ")
    ))
}
