// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the `x_atlas` objects `/v1/models` gains with the
//! `model_swap_timing` row on (port sprint lane L41): which model is resident,
//! how many device bytes it holds and what a swap to a model is expected to
//! cost. Pure over a [`SwapStats`] and a [`Residency`], so the handler and the
//! tests build the same objects.

use std::collections::BTreeMap;

use serde::Serialize;

use super::SwapStats;

/// The loaded model and the device bytes it holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Residency {
    pub model: String,
    pub bytes: u64,
}

impl Residency {
    /// `ledger_live` is `spark_runtime::run_metrics::ledger_bytes().0`.
    pub fn new(model: String, ledger_live: usize) -> Self {
        Self {
            model,
            bytes: u64::try_from(ledger_live).unwrap_or(u64::MAX),
        }
    }
}

/// What kind of `/v1/models` entry an id is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryKind {
    /// The base model the process is serving.
    Base,
    /// A LoRA adapter resident next to it.
    Adapter,
    /// A cold adapter that faults in on first use.
    Staged,
}

/// The `x_atlas` object of one `/v1/models` entry. Absent fields are omitted,
/// never `null`: absence means "unknown".
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct XAtlas {
    pub resident: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resident_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub swap_seconds_estimate: Option<f64>,
}

/// The top-level `x_atlas` object of the `/v1/models` list.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ListXAtlas {
    pub resident_model: Option<String>,
    pub swap_seconds_estimate: BTreeMap<String, f64>,
}

/// Seconds rounded to a millisecond (a router does not need more).
pub fn round_ms(secs: f64) -> f64 {
    (secs * 1000.0).round() / 1000.0
}

/// The `x_atlas` of entry `id`.
///
/// The base model is resident when it is the loaded one, and then costs 0 to
/// stay on; otherwise its price is the median of its timed swaps, absent when
/// it never was timed. A resident adapter rides on the base model, so it is
/// resident and free; a cold adapter faults in without a model swap, so it has
/// no swap price.
pub fn entry_x_atlas(
    stats: &SwapStats,
    id: &str,
    kind: EntryKind,
    residency: Option<&Residency>,
) -> XAtlas {
    match kind {
        EntryKind::Base => {
            let live = residency.filter(|r| r.model == id);
            XAtlas {
                resident: live.is_some(),
                resident_bytes: live.map(|r| r.bytes),
                swap_seconds_estimate: if live.is_some() {
                    Some(0.0)
                } else {
                    stats.estimate(id).map(round_ms)
                },
            }
        }
        EntryKind::Adapter => XAtlas {
            resident: true,
            resident_bytes: None,
            swap_seconds_estimate: Some(0.0),
        },
        EntryKind::Staged => XAtlas {
            resident: false,
            resident_bytes: None,
            swap_seconds_estimate: None,
        },
    }
}

/// The list-level `x_atlas`: the resident model (or `null`) and the estimate of
/// every model with swap history, which prices models that are not loaded.
pub fn list_x_atlas(stats: &SwapStats, residency: Option<&Residency>) -> ListXAtlas {
    ListXAtlas {
        resident_model: residency.map(|r| r.model.clone()),
        swap_seconds_estimate: stats
            .estimates()
            .into_iter()
            .map(|(m, s)| (m, round_ms(s)))
            .collect(),
    }
}
