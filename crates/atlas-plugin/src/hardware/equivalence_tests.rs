// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::hardware::state::{ThermalZone, ThrottleActive};

fn state(chassis: f64, thermal: Option<bool>) -> HardwareState {
    HardwareState {
        sm_clock_max_mhz: Some(3_003.0),
        mem_total_kb: Some(125_000_000),
        chassis_temps_c: Some(vec![
            ThermalZone {
                name: "acpitz".into(),
                temp_c: chassis,
            },
            ThermalZone {
                name: "acpitz".into(),
                temp_c: chassis - 5.0,
            },
        ]),
        throttle_active: ThrottleActive {
            sw_power_cap: Some(true),
            sw_thermal: thermal,
            hw_thermal: thermal,
            hw_power_brake: thermal,
        },
        ..Default::default()
    }
}

fn hw(gpu: &str, driver: &str) -> Hardware {
    Hardware {
        gpu: gpu.into(),
        driver: driver.into(),
        sm_clock_mhz: None,
        gpu_count: None,
        source: "test".into(),
    }
}

fn gb10(chassis: f64) -> HardwareFingerprint {
    HardwareFingerprint::from_live(
        &hw("NVIDIA GB10", "580.95.05"),
        &state(chassis, Some(false)),
    )
}

#[test]
fn two_healthy_gb10s_are_one_box_and_the_incident_pair_is_not() {
    assert_eq!(equivalent(&gb10(65.0), &gb10(70.0), &SPEED_SPREAD), Ok(()));
    // 8 °C apart: inside a day's swing on one box.
    assert_eq!(equivalent(&gb10(65.0), &gb10(73.0), &SPEED_SPREAD), Ok(()));
    // 13 °C apart: the 2026-09-15 fleet under load (55 vs 68 °C), accepted
    // since the limit went to 15.
    assert_eq!(equivalent(&gb10(55.0), &gb10(68.0), &SPEED_SPREAD), Ok(()));
    // The 2026-09-06 pair: 65 °C vs 89 °C, and a thermal reason on the hot one.
    let hot =
        HardwareFingerprint::from_live(&hw("NVIDIA GB10", "580.95.05"), &state(89.0, Some(true)));
    let why = equivalent(&gb10(65.0), &hot, &SPEED_SPREAD).unwrap_err();
    assert!(
        why.contains(&Mismatch::ChassisDelta {
            a: 65.0,
            b: 89.0,
            limit: 15.0
        }),
        "{why:?}"
    );
    assert!(
        why.contains(&Mismatch::ThermalAlert { a: false, b: true }),
        "{why:?}"
    );
    // 16 °C alone, no throttle: refused — the temperature is the signal.
    let why = equivalent(&gb10(65.0), &gb10(81.0), &SPEED_SPREAD).unwrap_err();
    assert_eq!(why.len(), 1, "{why:?}");
    assert!(matches!(why[0], Mismatch::ChassisDelta { .. }));
}

#[test]
fn every_static_field_is_checked_and_every_mismatch_is_named() {
    let base = gb10(65.0);
    // NEGATIVE CONTROLS, one field at a time.
    let mut other_gpu = base.clone();
    other_gpu.gpu = "NVIDIA H100".into();
    assert!(matches!(
        equivalent(&base, &other_gpu, &SPEED_SPREAD).unwrap_err()[..],
        [Mismatch::Gpu(..)]
    ));
    let mut other_driver = base.clone();
    other_driver.driver_major = Some(575);
    assert!(matches!(
        equivalent(&base, &other_driver, &SPEED_SPREAD).unwrap_err()[..],
        [Mismatch::DriverMajor(580, 575)]
    ));
    // Same major, different minor: fine.
    assert_eq!(driver_major("580.126.09"), Some(580));
    assert_eq!(driver_major(""), None);
    assert_eq!(driver_major("unknown"), None);
    let mut clock = base.clone();
    clock.sm_clock_max_mhz = Some(3_003.0 * 0.97);
    assert!(matches!(
        equivalent(&base, &clock, &SPEED_SPREAD).unwrap_err()[..],
        [Mismatch::ClockSpread { .. }]
    ));
    let mut clock_ok = base.clone();
    clock_ok.sm_clock_max_mhz = Some(3_003.0 * 0.995);
    assert_eq!(equivalent(&base, &clock_ok, &SPEED_SPREAD), Ok(()));
    let mut mem = base.clone();
    mem.mem_total_kb = Some(125_000_000 * 9 / 10);
    assert!(matches!(
        equivalent(&base, &mem, &SPEED_SPREAD).unwrap_err()[..],
        [Mismatch::MemSpread { .. }]
    ));
    let mut post = base.clone();
    post.postcheck_valid = Some(false);
    assert!(matches!(
        equivalent(&base, &post, &SPEED_SPREAD).unwrap_err()[..],
        [Mismatch::PostcheckInvalid]
    ));
    // A valid postcheck on one side and none on the other (live) is fine.
    let mut post_ok = base.clone();
    post_ok.postcheck_valid = Some(true);
    assert_eq!(equivalent(&base, &post_ok, &SPEED_SPREAD), Ok(()));
    // Every Display is non-empty and names the field.
    for m in [
        Mismatch::Gpu("a".into(), "b".into()),
        Mismatch::DriverMajor(1, 2),
        Mismatch::ClockSpread {
            a: 1.0,
            b: 2.0,
            limit: 0.01,
        },
        Mismatch::MemSpread {
            a: 1,
            b: 2,
            limit: 0.05,
        },
        Mismatch::ChassisDelta {
            a: 1.0,
            b: 2.0,
            limit: 15.0,
        },
        Mismatch::ThermalAlert { a: true, b: true },
        Mismatch::PostcheckInvalid,
        Mismatch::Undecidable("gpu"),
    ] {
        assert!(!m.to_string().is_empty());
    }
}

#[test]
fn a_missing_field_is_undecidable_which_is_not_equivalent() {
    let base = gb10(65.0);
    for (field, strip) in [
        (
            "gpu",
            Box::new(|f: &mut HardwareFingerprint| f.gpu.clear())
                as Box<dyn Fn(&mut HardwareFingerprint)>,
        ),
        ("driver", Box::new(|f| f.driver_major = None)),
        ("sm_clock_max_mhz", Box::new(|f| f.sm_clock_max_mhz = None)),
        ("mem_total_kb", Box::new(|f| f.mem_total_kb = None)),
        ("throttle reasons", Box::new(|f| f.thermal_alert = None)),
        (
            "chassis temperature",
            Box::new(|f| f.hottest_chassis_c = None),
        ),
    ] {
        let mut stripped = base.clone();
        strip(&mut stripped);
        let why = equivalent(&base, &stripped, &SPEED_SPREAD).expect_err(field);
        assert_eq!(why, vec![Mismatch::Undecidable(field)], "{field}");
        // Symmetric.
        let why = equivalent(&stripped, &base, &SPEED_SPREAD).expect_err(field);
        assert_eq!(why, vec![Mismatch::Undecidable(field)], "{field}");
    }
    // A box with NO state at all: from_live over a default state.
    let bare =
        HardwareFingerprint::from_live(&hw("NVIDIA GB10", "580.1"), &HardwareState::default());
    assert!(equivalent(&base, &bare, &SPEED_SPREAD).unwrap_err().len() >= 4);
}

#[test]
fn a_record_carries_its_before_capture_and_its_postcheck() {
    // A committed Speed-class record: the real shape, not a hand-built one.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace root");
    let dir = root.join(".benchmarks/decode-floor");
    let newest = std::fs::read_dir(&dir)
        .expect("decode-floor records are committed")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .max()
        .expect("at least one record");
    let rec = crate::gate::read_record(&newest).expect("parses");
    let fp = HardwareFingerprint::from_record(&rec);
    assert_eq!(fp.gpu, "NVIDIA GB10");
    assert_eq!(fp.driver_major, Some(580));
    assert_eq!(fp.sm_clock_max_mhz, Some(3_003.0));
    assert!(fp.mem_total_kb.is_some_and(|kb| kb > 100_000_000));
    assert_eq!(fp.thermal_alert, Some(false));
    assert!(fp.hottest_chassis_c.is_some());
    assert_eq!(fp.postcheck_valid, Some(true));
    // It is one box with a live GB10 at the same chassis temperature.
    let live = HardwareFingerprint::from_live(
        &hw("NVIDIA GB10", "580.95.05"),
        &state(fp.hottest_chassis_c.unwrap(), Some(false)),
    );
    let mut live = live;
    live.mem_total_kb = fp.mem_total_kb;
    assert_eq!(equivalent(&fp, &live, &SPEED_SPREAD), Ok(()));
    // A record with no report at all is undecidable on every live field.
    let mut bare = rec.clone();
    bare.hardware_state = None;
    let fp = HardwareFingerprint::from_record(&bare);
    assert_eq!(fp.postcheck_valid, None);
    let why = equivalent(&fp, &live, &SPEED_SPREAD).unwrap_err();
    assert!(
        why.iter().all(|m| matches!(m, Mismatch::Undecidable(_))),
        "{why:?}"
    );
    assert_eq!(why.len(), 4);
}
