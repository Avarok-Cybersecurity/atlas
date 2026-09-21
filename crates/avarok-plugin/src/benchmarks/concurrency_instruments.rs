// SPDX-License-Identifier: AGPL-3.0-only

//! The concurrency sweep's per-cell INSTRUMENT keys (2026-09-20): both ITL
//! clocks, the arrival-gap (jitter) distribution, and the GPU-rail energy
//! of the measured batch. Split from `concurrency.rs` so the driver's diff
//! stays a handful of lines and the key set is readable in one place.
//!
//! # Naming
//!
//! The quantity is presented as ITL (Inter-Token Latency) but the repo
//! already stores it under `tpot`, so no `itl_*` key is minted for the same
//! number: `c{C}_tpot_*` is the CLIENT clock (like `c{C}_ttft_p50_ms`, which
//! is client-measured) and `c{C}_server_tpot_*` the SERVER clock (the
//! `server_` prefix `quick_speed` already uses). The delta between the two
//! is the SSE/transport overhead and is derivable, not stored.
//!
//! Every key is additive and info-only until a BENCH.toml bound names it.

use std::collections::BTreeMap;

use super::CellRow;
use crate::hardware::energy::EnergyWindow;

impl CellRow {
    /// The instrument keys for one rung, under `prefix` (`"c8_"`).
    ///
    /// A clock that was not measured emits nothing — a server predating
    /// `usage.decode_time_ms` yields no `server_tpot` key, never a zero.
    pub(super) fn instrument_metrics(
        &self,
        prefix: &str,
        idle: Option<&EnergyWindow>,
        m: &mut BTreeMap<String, f64>,
    ) {
        let pairs = [
            ("tpot_p50_ms", self.tpot.p50),
            ("tpot_p90_ms", self.tpot.p90),
            ("server_tpot_p50_ms", self.server_tpot.p50),
            ("server_tpot_p90_ms", self.server_tpot.p90),
        ];
        for (k, v) in pairs {
            if let Some(v) = v {
                m.insert(format!("{prefix}{k}"), v);
            }
        }
        if let Some(g) = &self.gaps {
            g.metrics(prefix, m);
        }
        if let Some(e) = &self.energy {
            e.metrics(prefix, self.tokens, idle, m);
        }
    }
}
