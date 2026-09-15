// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use atlas_plugin::hardware::equivalence::HardwareFingerprint;
use std::sync::Mutex;

fn node() -> Node {
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
            hottest_chassis_c: Some(40.0),
            postcheck_valid: None,
        },
        free_fraction: Some(0.95),
        built: true,
        local: false,
    }
}

fn r(c: f64) -> Reading {
    Reading {
        chassis_c: Some(c),
        throttled: Some(false),
    }
}

/// The rule, with its hysteresis: park at 80, resume at or below 70, and
/// the band between is "hold what you were doing". A loaded GB10 at 65-76
/// (2026-09-15) is working, not parked; the incident box at 89 is.
#[test]
fn park_at_eighty_resume_at_seventy() {
    assert_eq!(
        judge(r(76.0), false),
        Verdict::Ready,
        "a loaded box keeps working"
    );
    assert_eq!(judge(r(79.9), false), Verdict::Ready);
    assert!(
        matches!(judge(r(80.0), false), Verdict::Park { .. }),
        "the line is inclusive"
    );
    assert!(
        matches!(judge(r(89.0), false), Verdict::Park { .. }),
        "the incident box"
    );
    // Parked: 75 is still above 70 — hold; 70 — go.
    assert!(matches!(judge(r(75.0), true), Verdict::Park { .. }));
    assert_eq!(judge(r(70.0), true), Verdict::Ready);
    // Not parked and in the band: keep working.
    assert_eq!(judge(r(75.0), false), Verdict::Ready);
}

/// The driver's thermal-slowdown flag parks at ANY temperature, and holds a
/// parked node until it clears — the temperature alone does not release it.
#[test]
fn a_throttle_flag_parks_and_holds_whatever_the_temperature() {
    let hot = Reading {
        chassis_c: Some(60.0),
        throttled: Some(true),
    };
    assert!(matches!(
        judge(hot, false),
        Verdict::Park {
            throttled: true,
            ..
        }
    ));
    assert!(matches!(
        judge(hot, true),
        Verdict::Park {
            throttled: true,
            ..
        }
    ));
    let flag_only = Reading {
        chassis_c: None,
        throttled: Some(true),
    };
    assert!(matches!(
        judge(flag_only, false),
        Verdict::Park {
            throttled: true,
            ..
        }
    ));
}

/// NEGATIVE CONTROL: no temperature and no flag parks nothing.
#[test]
fn a_blind_reading_never_parks() {
    let blind = Reading {
        chassis_c: None,
        throttled: None,
    };
    assert_eq!(judge(blind, false), Verdict::Blind);
    assert_eq!(judge(blind, true), Verdict::Blind);
    let unknown_flag = Reading {
        chassis_c: Some(99.0),
        throttled: None,
    };
    assert!(
        matches!(
            judge(unknown_flag, false),
            Verdict::Park {
                throttled: false,
                ..
            }
        ),
        "temperature alone still parks"
    );
}

struct Scripted(Mutex<Vec<Reading>>);
impl Probe for Scripted {
    fn read(&self, _: &Node) -> Reading {
        let mut v = self.0.lock().unwrap();
        if v.len() > 1 { v.remove(0) } else { v[0] }
    }
}

/// The gate parks on the first hot reading, holds through the band, resumes
/// at the line, and says each transition exactly once.
#[test]
fn the_gate_parks_holds_and_resumes_saying_so_once() {
    let n = node();
    let p = Scripted(Mutex::new(vec![
        r(84.0),
        r(78.0),
        r(72.0),
        r(70.0),
        r(70.0),
    ]));
    let said = Mutex::new(Vec::<String>::new());
    let say = |s: &str| said.lock().unwrap().push(s.to_string());
    let mut g = Gate::default();
    assert!(!g.may_take(&n, &p, false, &say), "84: parked");
    assert!(!g.may_take(&n, &p, false, &say), "78: hold");
    assert!(!g.may_take(&n, &p, false, &say), "72: hold");
    assert!(g.may_take(&n, &p, false, &say), "70: resume");
    assert!(g.may_take(&n, &p, false, &say));
    let said = said.lock().unwrap();
    assert_eq!(said.len(), 2, "{said:?}");
    assert!(
        said[0].contains("parked") && said[0].contains("chassis 84"),
        "{said:?}"
    );
    assert!(said[1].contains("resuming"), "{said:?}");
}

/// A node that reports nothing is never parked, and told once.
#[test]
fn a_node_that_reports_nothing_is_never_parked() {
    let n = node();
    let p = Scripted(Mutex::new(vec![Reading {
        chassis_c: None,
        throttled: None,
    }]));
    let said = Mutex::new(Vec::<String>::new());
    let say = |s: &str| said.lock().unwrap().push(s.to_string());
    let mut g = Gate::default();
    assert!(g.may_take(&n, &p, false, &say));
    assert!(g.may_take(&n, &p, false, &say));
    assert_eq!(said.lock().unwrap().len(), 1);
}

/// `--dangerous-ignore-thermals`: the same readings park nothing; the hot
/// spell is warned about once on the way in and cleared once on the way out.
#[test]
fn ignoring_thermals_warns_once_and_never_parks() {
    let n = node();
    let p = Scripted(Mutex::new(vec![
        r(84.0),
        r(88.0),
        r(75.0),
        r(70.0),
        r(70.0),
    ]));
    let said = Mutex::new(Vec::<String>::new());
    let say = |s: &str| said.lock().unwrap().push(s.to_string());
    let mut g = Gate::default();
    for _ in 0..5 {
        assert!(
            g.may_take(&n, &p, true, &say),
            "never parked under the flag"
        );
    }
    let said = said.lock().unwrap();
    assert_eq!(said.len(), 2, "{said:?}");
    assert!(
        said[0].contains("WARNING --dangerous-ignore-thermals"),
        "{said:?}"
    );
    assert!(said[0].contains("would be parked"), "{said:?}");
    assert!(said[1].contains("back below"), "{said:?}");
}
