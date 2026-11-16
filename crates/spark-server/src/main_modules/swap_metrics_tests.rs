// SPDX-License-Identifier: AGPL-3.0-only

//! Oracles for the swap timer, the per-model history, the `/v1/models`
//! `x_atlas` objects and the `atlas_model_*` text. Every test builds its own
//! [`SwapStats`], so none shares state with another or with the process-wide
//! instance (whose row is read from the environment).

use super::atlas::{EntryKind, Residency, XAtlas, entry_x_atlas, list_x_atlas};
use super::render::{escape_label, render_with};
use super::*;

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

fn sleep(n: u64) {
    std::thread::sleep(ms(n));
}

fn phase(s: &SwapStats, p: Phase) -> (u64, f64) {
    s.snapshot().phases[p.index()]
}

fn outcome(s: &SwapStats, o: Outcome) -> u64 {
    s.snapshot().outcomes[o.index()]
}

#[test]
fn the_join_splits_at_the_scheduler_mark() {
    let t0 = Instant::now();
    let (start, mid, end) = (t0, t0 + ms(30), t0 + ms(50));
    assert_eq!(split_join(start, end, Some(mid)), (ms(30), Some(ms(20))));
    // The ends of the window belong to it.
    assert_eq!(split_join(start, end, Some(start)), (ms(0), Some(ms(50))));
    assert_eq!(split_join(start, end, Some(end)), (ms(50), Some(ms(0))));
}

#[test]
fn a_missing_or_stale_mark_leaves_the_whole_join_as_drain() {
    let t0 = Instant::now();
    let (start, end) = (t0 + ms(10), t0 + ms(50));
    assert_eq!(split_join(start, end, None), (ms(40), None));
    assert_eq!(split_join(start, end, Some(t0)), (ms(40), None), "before");
    assert_eq!(
        split_join(start, end, Some(t0 + ms(80))),
        (ms(40), None),
        "after"
    );
}

/// The fake loader sleeps; the histogram must observe the sleep, each phase in
/// its own series, and `total` must cover them all.
#[test]
fn a_sleeping_fake_loader_shows_up_in_the_load_phase_and_the_total() {
    let stats = SwapStats::new();
    let mut timer = SwapTimer::begin(&stats, true);
    sleep(15); // steps 1-3: in-flight holders finish, the scheduler drains
    stats.mark_teardown(); // the scheduler starts releasing the model
    sleep(10); // step 4
    timer.joined();
    sleep(40); // step 5: the fake loader
    timer.loaded();
    timer.finish(Some("model-b"), Outcome::Ok);

    let (dn, ds) = phase(&stats, Phase::Drain);
    let (tn, ts) = phase(&stats, Phase::Teardown);
    let (ln, ls) = phase(&stats, Phase::Load);
    let (n, total) = phase(&stats, Phase::Total);
    assert_eq!((dn, tn, ln, n), (1, 1, 1, 1));
    assert!(ds >= 0.015, "drain {ds}");
    assert!(ts >= 0.010, "teardown {ts}");
    assert!(ls >= 0.040, "load {ls}");
    assert!(total >= 0.065, "total {total}");
    assert!(total + 1e-9 >= ds + ts + ls, "total covers the parts");
    assert_eq!(outcome(&stats, Outcome::Ok), 1);
    let est = stats.estimate("model-b").expect("history");
    assert!((est - total).abs() < 1e-9, "one sample is its own median");
}

#[test]
fn without_a_mark_the_whole_join_is_drain_and_teardown_stays_empty() {
    let stats = SwapStats::new();
    let mut timer = SwapTimer::begin(&stats, true);
    sleep(20);
    timer.joined();
    timer.loaded();
    timer.finish(Some("m"), Outcome::Ok);
    assert!(phase(&stats, Phase::Drain).1 >= 0.020);
    assert_eq!(phase(&stats, Phase::Teardown), (0, 0.0));
}

#[test]
fn a_mark_left_before_the_swap_began_is_ignored() {
    let stats = SwapStats::new();
    stats.mark_teardown(); // a boot, or the previous swap
    sleep(5);
    let mut timer = SwapTimer::begin(&stats, true);
    sleep(5);
    timer.joined();
    assert_eq!(phase(&stats, Phase::Teardown).0, 0);
    // And a mark made during this swap is used.
    let mut timer2 = SwapTimer::begin(&stats, true);
    sleep(10);
    stats.mark_teardown();
    sleep(10);
    timer2.joined();
    let (n, s) = phase(&stats, Phase::Teardown);
    assert_eq!(n, 1);
    assert!(s >= 0.010, "{s}");
    timer.finish(None, Outcome::Failed);
    timer2.finish(None, Outcome::Failed);
}

#[test]
fn a_timer_that_is_off_records_nothing() {
    let stats = SwapStats::new();
    let mut timer = SwapTimer::begin(&stats, false);
    sleep(5);
    timer.joined();
    timer.loaded();
    timer.finish(Some("m"), Outcome::Ok);
    drop(timer);
    assert_eq!(stats.snapshot(), Snapshot::default());
    assert!(stats.estimates().is_empty());
}

#[test]
fn only_a_published_swap_adds_a_total_and_history() {
    let stats = SwapStats::new();
    for o in [Outcome::Restored, Outcome::Failed, Outcome::Refused] {
        let mut t = SwapTimer::begin(&stats, true);
        t.joined();
        t.finish(Some("m"), o);
    }
    assert_eq!(phase(&stats, Phase::Total).0, 0, "no total without an ok");
    assert!(stats.estimate("m").is_none());
    for o in [Outcome::Restored, Outcome::Failed, Outcome::Refused] {
        assert_eq!(outcome(&stats, o), 1);
    }
    assert_eq!(outcome(&stats, Outcome::Ok), 0);
}

#[test]
fn finish_counts_once_and_a_dropped_timer_counts_failed() {
    let stats = SwapStats::new();
    let mut t = SwapTimer::begin(&stats, true);
    t.finish(None, Outcome::Refused);
    t.finish(None, Outcome::Ok); // ignored: already finished
    drop(t);
    assert_eq!(outcome(&stats, Outcome::Refused), 1);
    assert_eq!(outcome(&stats, Outcome::Ok), 0);
    assert_eq!(outcome(&stats, Outcome::Failed), 0);
    // An early `?` return never reaches finish.
    drop(SwapTimer::begin(&stats, true));
    assert_eq!(outcome(&stats, Outcome::Failed), 1);
}

#[test]
fn the_history_keeps_the_last_n_per_model() {
    let stats = SwapStats::new();
    for i in 1..=(HISTORY as u64 + 3) {
        stats.record_total("a", Duration::from_secs(i));
    }
    let kept = stats.durations("a");
    assert_eq!(kept.len(), HISTORY);
    assert_eq!(
        kept.first().copied(),
        Some(4.0),
        "the oldest three fell off"
    );
    assert_eq!(kept.last().copied(), Some(HISTORY as f64 + 3.0));
    // Every record is also a `total` observation.
    assert_eq!(phase(&stats, Phase::Total).0, HISTORY as u64 + 3);
}

#[test]
fn the_estimate_is_the_median_and_absent_without_history() {
    let stats = SwapStats::new();
    assert_eq!(stats.estimate("a"), None);
    for s in [90, 60, 300] {
        stats.record_total("a", Duration::from_secs(s));
    }
    assert_eq!(stats.estimate("a"), Some(90.0), "odd count: the middle");
    stats.record_total("a", Duration::from_secs(120));
    assert_eq!(
        stats.estimate("a"),
        Some(105.0),
        "even count: mean of middle two"
    );
    assert_eq!(stats.estimate("b"), None);
    // One cold first load among warm ones does not drag the estimate.
    for s in [10, 11, 12, 11] {
        stats.record_total("c", Duration::from_secs(s));
    }
    stats.record_total("c", Duration::from_secs(200));
    assert_eq!(stats.estimate("c"), Some(11.0));
}

#[test]
fn the_number_of_tracked_models_is_bounded() {
    let stats = SwapStats::new();
    for i in 0..(MAX_MODELS + 5) {
        stats.record_total(&format!("m{i}"), ms(1));
    }
    let known = stats.estimates();
    assert_eq!(known.len(), MAX_MODELS);
    assert!(!known.contains_key("m0"), "the first id was dropped");
    assert!(known.contains_key(&format!("m{}", MAX_MODELS + 4)));
}

fn busy() -> SwapStats {
    let s = SwapStats::new();
    s.observe(Phase::Drain, ms(1500));
    s.observe(Phase::Load, Duration::from_secs(2));
    s.record_total("qwen \"b\"", Duration::from_secs(3));
    s.count(Outcome::Ok);
    s
}

#[test]
fn the_series_text_is_exact() {
    let mut out = String::new();
    render_with(&mut out, &busy(), true, 123);
    assert_eq!(
        out,
        "# HELP atlas_model_swap_seconds Hot-swap time by phase (drain, teardown, load, total), seconds\n\
         # TYPE atlas_model_swap_seconds summary\n\
         atlas_model_swap_seconds_sum{phase=\"drain\"} 1.500000\n\
         atlas_model_swap_seconds_count{phase=\"drain\"} 1\n\
         atlas_model_swap_seconds_sum{phase=\"teardown\"} 0.000000\n\
         atlas_model_swap_seconds_count{phase=\"teardown\"} 0\n\
         atlas_model_swap_seconds_sum{phase=\"load\"} 2.000000\n\
         atlas_model_swap_seconds_count{phase=\"load\"} 1\n\
         atlas_model_swap_seconds_sum{phase=\"total\"} 3.000000\n\
         atlas_model_swap_seconds_count{phase=\"total\"} 1\n\
         # HELP atlas_model_swap_total Hot-swaps by outcome (ok, restored, failed, refused)\n\
         # TYPE atlas_model_swap_total counter\n\
         atlas_model_swap_total{outcome=\"ok\"} 1\n\
         atlas_model_swap_total{outcome=\"restored\"} 0\n\
         atlas_model_swap_total{outcome=\"failed\"} 0\n\
         atlas_model_swap_total{outcome=\"refused\"} 0\n\
         # HELP atlas_model_swap_estimate_seconds Median of the last timed swaps to the model, seconds\n\
         # TYPE atlas_model_swap_estimate_seconds gauge\n\
         atlas_model_swap_estimate_seconds{model=\"qwen \\\"b\\\"\"} 3\n\
         # HELP atlas_model_resident_bytes Device bytes live in the allocation ledger for the loaded model\n\
         # TYPE atlas_model_resident_bytes gauge\n\
         atlas_model_resident_bytes 123\n"
    );
}

#[test]
fn with_the_row_off_every_series_is_zero_and_no_model_label_appears() {
    let mut out = String::new();
    render_with(&mut out, &busy(), false, 123);
    for line in out.lines().filter(|l| !l.starts_with('#')) {
        let value = line.rsplit(' ').next().expect("a value");
        assert!(
            matches!(value, "0" | "0.000000"),
            "row off must render zeros: {line}"
        );
    }
    assert!(!out.contains("model=\""), "{out}");
    // The same families exist, so a dashboard never sees a series vanish.
    let mut on = String::new();
    render_with(&mut on, &SwapStats::new(), true, 0);
    let families = |s: &str| {
        s.lines()
            .filter(|l| l.starts_with("# TYPE"))
            .map(str::to_string)
            .collect::<Vec<_>>()
    };
    assert_eq!(families(&out), families(&on));
}

#[test]
fn label_values_are_escaped() {
    assert_eq!(escape_label("a\\b\"c\nd"), "a\\\\b\\\"c\\nd");
    assert_eq!(escape_label("plain/model-1.5"), "plain/model-1.5");
}

fn residency() -> Residency {
    Residency::new("live".to_string(), 5_000_000_000)
}

#[test]
fn the_resident_base_model_is_resident_with_bytes_and_zero_swap_cost() {
    let stats = busy();
    let r = residency();
    let x = entry_x_atlas(&stats, "live", EntryKind::Base, Some(&r));
    assert_eq!(
        x,
        XAtlas {
            resident: true,
            resident_bytes: Some(5_000_000_000),
            swap_seconds_estimate: Some(0.0)
        }
    );
}

#[test]
fn a_base_model_that_is_not_loaded_is_priced_from_its_history() {
    let stats = busy();
    let r = residency();
    let x = entry_x_atlas(&stats, "qwen \"b\"", EntryKind::Base, Some(&r));
    assert!(!x.resident);
    assert_eq!(x.resident_bytes, None);
    assert_eq!(x.swap_seconds_estimate, Some(3.0));
    let never = entry_x_atlas(&stats, "never-swapped", EntryKind::Base, Some(&r));
    assert_eq!(never.swap_seconds_estimate, None, "unknown stays unknown");
    // Nothing loaded: nothing is resident, history still prices a load.
    let none = entry_x_atlas(&stats, "live", EntryKind::Base, None);
    assert!(!none.resident && none.resident_bytes.is_none());
}

#[test]
fn adapters_follow_the_base_and_staged_names_are_cold() {
    let stats = SwapStats::new();
    let r = residency();
    let a = entry_x_atlas(&stats, "lora-1", EntryKind::Adapter, Some(&r));
    assert!(a.resident && a.resident_bytes.is_none());
    assert_eq!(a.swap_seconds_estimate, Some(0.0));
    let s = entry_x_atlas(&stats, "lora-2", EntryKind::Staged, Some(&r));
    assert!(!s.resident);
    assert_eq!(s.swap_seconds_estimate, None);
}

#[test]
fn the_wire_names_are_the_ones_a_router_reads() {
    let stats = busy();
    let r = residency();
    let v = serde_json::to_value(entry_x_atlas(&stats, "live", EntryKind::Base, Some(&r)))
        .expect("serializes");
    assert_eq!(v["resident"], true);
    assert_eq!(v["resident_bytes"], 5_000_000_000u64);
    assert_eq!(v["swap_seconds_estimate"], 0.0);
    let cold = serde_json::to_value(entry_x_atlas(&stats, "x", EntryKind::Staged, None))
        .expect("serializes");
    let keys: Vec<&str> = cold
        .as_object()
        .expect("object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(keys, ["resident"], "absent fields are omitted, never null");
    let list = serde_json::to_value(list_x_atlas(&stats, Some(&r))).expect("serializes");
    assert_eq!(list["resident_model"], "live");
    assert_eq!(list["swap_seconds_estimate"]["qwen \"b\""], 3.0);
    let empty = serde_json::to_value(list_x_atlas(&SwapStats::new(), None)).expect("serializes");
    assert!(empty["resident_model"].is_null());
    assert_eq!(empty["swap_seconds_estimate"], serde_json::json!({}));
}

#[test]
fn the_estimate_is_rounded_to_a_millisecond() {
    let stats = SwapStats::new();
    stats.record_total("m", Duration::from_nanos(12_345_678_901));
    let x = entry_x_atlas(&stats, "m", EntryKind::Base, None);
    assert_eq!(x.swap_seconds_estimate, Some(12.346));
}

#[test]
fn residency_converts_the_ledger_bytes() {
    let r = Residency::new("m".to_string(), 7);
    assert_eq!((r.model.as_str(), r.bytes), ("m", 7));
}
