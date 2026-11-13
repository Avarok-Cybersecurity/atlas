// SPDX-License-Identifier: AGPL-3.0-only

//! `/v1/models` with and without the `model_swap_timing` row: the row off must
//! leave the response bytes alone, the row on adds `x_atlas` and nothing else.

use super::*;
use serde_json::{Value, json};

fn info(id: &str) -> ModelInfo {
    ModelInfo {
        id: id.to_string(),
        object: "model".to_string(),
        created: 0,
        owned_by: "avarok-spark".to_string(),
        max_model_len: Some(4096),
    }
}

/// Adapter first, then a cold staged name, then the base model: the order the
/// handler lists them in.
fn entries() -> Vec<(ModelInfo, EntryKind)> {
    vec![
        (info("lora-1"), EntryKind::Adapter),
        (info("lora-cold"), EntryKind::Staged),
        (info("live"), EntryKind::Base),
    ]
}

fn stats() -> SwapStats {
    let s = SwapStats::new();
    s.record_total("big", std::time::Duration::from_secs(90));
    s
}

fn residency() -> Residency {
    Residency::new("live".to_string(), 8_000_000_000)
}

fn body(on: bool, with_model: bool) -> Value {
    let r = residency();
    let list = list_body(
        if with_model { entries() } else { Vec::new() },
        with_model.then_some(&r),
        &stats(),
        on,
    );
    serde_json::to_value(list).expect("serializes")
}

#[test]
fn row_off_is_byte_identical_to_the_plain_list() {
    let r = residency();
    let off = list_body(entries(), Some(&r), &stats(), false);
    let plain = ModelListResponse {
        object: "list".to_string(),
        data: entries().into_iter().map(|(m, _)| m).collect(),
    };
    assert_eq!(
        serde_json::to_string(&off).expect("serializes"),
        serde_json::to_string(&plain).expect("serializes"),
    );
    let text = serde_json::to_string(&off).expect("serializes");
    assert!(!text.contains("x_atlas"), "{text}");
}

#[test]
fn row_off_with_no_model_is_the_empty_list() {
    let off = body(false, false);
    assert_eq!(off, json!({"object": "list", "data": []}));
}

#[test]
fn row_on_adds_x_atlas_to_every_entry_and_keeps_every_other_field() {
    let on = body(true, true);
    let off = body(false, true);
    let (on_data, off_data) = (
        on["data"].as_array().unwrap(),
        off["data"].as_array().unwrap(),
    );
    assert_eq!(on_data.len(), off_data.len());
    for (a, b) in on_data.iter().zip(off_data) {
        let mut stripped = a.clone();
        let x = stripped.as_object_mut().unwrap().remove("x_atlas");
        assert!(x.is_some(), "every entry has x_atlas: {a}");
        assert_eq!(&stripped, b, "nothing else changed");
    }
    assert_eq!(on["object"], "list");
}

#[test]
fn row_on_reports_residency_and_swap_cost_per_kind() {
    let on = body(true, true);
    let data = on["data"].as_array().unwrap();
    // The resident adapter: serving, free to select.
    assert_eq!(
        data[0]["x_atlas"],
        json!({"resident": true, "swap_seconds_estimate": 0.0})
    );
    // A cold staged name: not resident, no swap price.
    assert_eq!(data[1]["x_atlas"], json!({"resident": false}));
    // The base model: resident with its bytes, zero to stay.
    assert_eq!(
        data[2]["x_atlas"],
        json!({"resident": true, "resident_bytes": 8_000_000_000u64, "swap_seconds_estimate": 0.0})
    );
}

#[test]
fn row_on_lists_the_swap_estimates_of_models_that_are_not_loaded() {
    let on = body(true, true);
    assert_eq!(
        on["x_atlas"],
        json!({"resident_model": "live", "swap_seconds_estimate": {"big": 90.0}})
    );
}

#[test]
fn row_on_with_no_model_loaded_still_prices_a_load() {
    let on = body(true, false);
    assert_eq!(on["data"], json!([]));
    assert_eq!(
        on["x_atlas"],
        json!({"resident_model": null, "swap_seconds_estimate": {"big": 90.0}})
    );
}

#[test]
fn retrieve_is_untouched_with_the_row_off_and_gains_x_atlas_with_it_on() {
    let base = json!({"id": "live", "object": "model", "max_model_len": 4096});
    let r = residency();
    let s = stats();
    let off = with_x_atlas(base.clone(), "live", EntryKind::Base, Some(&r), &s, false);
    assert_eq!(off, base);
    let on = with_x_atlas(base.clone(), "live", EntryKind::Base, Some(&r), &s, true);
    assert_eq!(on["id"], "live");
    assert_eq!(on["x_atlas"]["resident"], true);
    assert_eq!(on["x_atlas"]["resident_bytes"], 8_000_000_000u64);
    let cold = with_x_atlas(base, "big", EntryKind::Base, Some(&r), &s, true);
    assert_eq!(
        cold["x_atlas"],
        json!({"resident": false, "swap_seconds_estimate": 90.0})
    );
}

#[test]
fn kinds_follow_what_the_loaded_state_serves() {
    let adapters = vec!["lora-1".to_string()];
    assert_eq!(kind_of("live", &adapters, "live"), EntryKind::Base);
    assert_eq!(kind_of("live", &adapters, "lora-1"), EntryKind::Adapter);
    assert_eq!(kind_of("live", &adapters, "lora-cold"), EntryKind::Staged);
}
