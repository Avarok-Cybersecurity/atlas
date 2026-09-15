// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use atlas_plugin::hardware::equivalence::HardwareFingerprint;
use std::sync::Mutex;

fn node(baseline: Option<f64>) -> Node {
    Node {
        addr: "10.10.10.3".into(),
        name: "dgx3".into(),
        node_id: "x".into(),
        signer: "s".into(),
        hardware: HardwareFingerprint {
            gpu: "NVIDIA GB10".into(),
            driver_major: Some(580),
            sm_clock_max_mhz: Some(3003.0),
            mem_total_kb: Some(127_601_400),
            thermal_alert: Some(false),
            hottest_chassis_c: baseline,
            postcheck_valid: None,
        },
        free_fraction: Some(0.95),
        built: true,
        local: false,
    }
}

/// The rule, with its hysteresis: park past +20 over baseline, resume only
/// within +5 — and the band between is "hold what you were doing".
#[test]
fn park_past_twenty_over_resume_within_five() {
    assert_eq!(judge(Some(40.0), Some(59.0), false), Verdict::Ready);
    assert_eq!(
        judge(Some(40.0), Some(60.0), false),
        Verdict::Ready,
        "the line is exclusive"
    );
    assert!(matches!(
        judge(Some(40.0), Some(61.0), false),
        Verdict::Park { .. }
    ));
    // Parked: 50 is still 10 over — hold; 45 is within 5 — go.
    assert!(matches!(
        judge(Some(40.0), Some(50.0), true),
        Verdict::Park { .. }
    ));
    assert_eq!(judge(Some(40.0), Some(45.0), true), Verdict::Ready);
    // Not parked and in the band: keep working.
    assert_eq!(judge(Some(40.0), Some(50.0), false), Verdict::Ready);
    // The 2026-09-15 pair: dgx3 at 68 against a 40 baseline is parked;
    // dgx2 at 55 against 43 is not.
    assert!(matches!(
        judge(Some(40.0), Some(68.0), false),
        Verdict::Park { .. }
    ));
    assert_eq!(judge(Some(43.0), Some(55.0), false), Verdict::Ready);
}

/// NEGATIVE CONTROL: a reading that cannot be taken parks nothing.
#[test]
fn a_blind_reading_never_parks() {
    assert_eq!(judge(None, Some(99.0), false), Verdict::Blind);
    assert_eq!(judge(Some(40.0), None, true), Verdict::Blind);
}

struct Scripted(Mutex<Vec<Option<f64>>>);
impl Probe for Scripted {
    fn hottest_chassis_c(&self, _: &Node) -> Option<f64> {
        let mut v = self.0.lock().unwrap();
        if v.len() > 1 { v.remove(0) } else { v[0] }
    }
}

/// The gate parks on the first hot reading, holds through the band, resumes
/// near baseline, and says each transition exactly once.
#[test]
fn the_gate_parks_holds_and_resumes_saying_so_once() {
    let n = node(Some(40.0));
    let p = Scripted(Mutex::new(vec![
        Some(65.0),
        Some(52.0),
        Some(47.0),
        Some(44.0),
        Some(44.0),
    ]));
    let said = Mutex::new(Vec::<String>::new());
    let say = |s: &str| said.lock().unwrap().push(s.to_string());
    let mut g = Gate::default();
    assert!(!g.may_take(&n, &p, &say), "65 over 40: parked");
    assert!(!g.may_take(&n, &p, &say), "52: still 12 over, hold");
    assert!(!g.may_take(&n, &p, &say), "47: 7 over, hold");
    assert!(g.may_take(&n, &p, &say), "44: within 5, resume");
    assert!(g.may_take(&n, &p, &say));
    let said = said.lock().unwrap();
    assert_eq!(said.len(), 2, "{said:?}");
    assert!(said[0].contains("parked until"), "{said:?}");
    assert!(said[1].contains("resuming"), "{said:?}");
}

/// A node with no baseline is never parked, and told once.
#[test]
fn a_node_without_a_baseline_is_never_parked() {
    let n = node(None);
    let p = Scripted(Mutex::new(vec![Some(99.0)]));
    let said = Mutex::new(Vec::<String>::new());
    let say = |s: &str| said.lock().unwrap().push(s.to_string());
    let mut g = Gate::default();
    assert!(g.may_take(&n, &p, &say));
    assert!(g.may_take(&n, &p, &say));
    assert_eq!(said.lock().unwrap().len(), 1);
}
