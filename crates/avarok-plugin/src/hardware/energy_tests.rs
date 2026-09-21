// SPDX-License-Identifier: AGPL-3.0-only

use std::time::Duration;

use super::*;

fn at(t0: Instant, ms: u64) -> Instant {
    t0 + Duration::from_millis(ms)
}

fn reading(t0: Instant, ms: u64, w: f64) -> PowerSample {
    PowerSample {
        at: at(t0, ms),
        power_w: w,
        sw_power_cap: Some(true),
        hw_power_brake: Some(false),
    }
}

#[test]
fn a_reading_line_parses_and_a_not_available_rail_does_not_become_zero_watts() {
    let t0 = Instant::now();
    let s = parse_line("4.76, Not Active, Active", t0).unwrap();
    assert_eq!(s.power_w, 4.76);
    assert_eq!(s.sw_power_cap, Some(false));
    assert_eq!(s.hw_power_brake, Some(true));
    // The GB10 spellings for "no idea" must not integrate as 0 W.
    assert_eq!(parse_line("[N/A], Not Active, Not Active", t0), None);
    assert_eq!(parse_line("N/A, Active, Active", t0), None);
    assert_eq!(parse_line("", t0), None);
    assert_eq!(parse_line("-3.0, Active, Active", t0), None);
    // A flag cell the driver did not fill is unknown, not "not active".
    let bare = parse_line("61.2", t0).unwrap();
    assert_eq!((bare.sw_power_cap, bare.hw_power_brake), (None, None));
    let odd = parse_line("61.2, Sometimes, [N/A]", t0).unwrap();
    assert_eq!((odd.sw_power_cap, odd.hw_power_brake), (None, None));
}

/// The rule the joule count stands on: hold each reading forward to the
/// next, back-fill the first to the window start, hold the last to the
/// window end. `Σ dt` is exactly the window, so the mean is energy/window.
#[test]
fn integration_holds_each_reading_forward_and_spans_exactly_the_window() {
    let t0 = Instant::now();
    // Window [100 ms, 1100 ms]; readings at 250 (10 W), 500 (20 W), 750 (30 W),
    // 1000 (40 W). Held forward, the first back-filled to 100 ms and the last
    // held to 1100 ms: 10×0.4 + 20×0.25 + 30×0.25 + 40×0.1 = 20.5 J over 1.0 s.
    let samples = [
        reading(t0, 0, 999.0), // outside, before
        reading(t0, 250, 10.0),
        reading(t0, 500, 20.0),
        reading(t0, 750, 30.0),
        reading(t0, 1000, 40.0),
        reading(t0, 1250, 999.0), // outside, after
    ];
    let w = integrate(&samples, at(t0, 100), at(t0, 1100)).unwrap();
    assert_eq!(w.samples, 4);
    assert!((w.window_s - 1.0).abs() < 1e-9);
    assert!((w.energy_j - 20.5).abs() < 1e-9, "{w:?}");
    assert!((w.mean_power_w - 20.5).abs() < 1e-9);
    assert_eq!(w.max_power_w, 40.0);
    assert_eq!(w.sw_power_cap_frac, Some(1.0));
    assert_eq!(w.hw_power_brake_frac, Some(0.0));
}

/// A window with no evidence reports nothing, not zero joules.
#[test]
fn an_empty_or_unsampled_window_is_none() {
    let t0 = Instant::now();
    let samples = [reading(t0, 5000, 10.0)];
    assert_eq!(integrate(&samples, at(t0, 0), at(t0, 1000)), None);
    assert_eq!(integrate(&[], at(t0, 0), at(t0, 1000)), None);
    assert_eq!(integrate(&samples, at(t0, 6000), at(t0, 6000)), None);
    assert_eq!(integrate(&samples, at(t0, 6000), at(t0, 5000)), None);
}

/// The flag fractions distinguish "never reported" from "reported off".
#[test]
fn cap_fractions_are_over_reporting_samples_only_and_none_when_unreported() {
    let t0 = Instant::now();
    let mut samples = vec![reading(t0, 100, 10.0), reading(t0, 200, 10.0)];
    samples[1].sw_power_cap = Some(false);
    samples[0].hw_power_brake = None;
    samples[1].hw_power_brake = None;
    let w = integrate(&samples, at(t0, 0), at(t0, 300)).unwrap();
    assert_eq!(w.sw_power_cap_frac, Some(0.5));
    assert_eq!(w.hw_power_brake_frac, None);
    // Mixed: one reading reports the brake, the other does not — the
    // fraction is over the one that did.
    samples[0].hw_power_brake = Some(true);
    let w = integrate(&samples, at(t0, 0), at(t0, 300)).unwrap();
    assert_eq!(w.hw_power_brake_frac, Some(1.0));
}

#[test]
fn above_idle_subtracts_the_baseline_over_the_same_duration_and_may_go_negative() {
    let idle = EnergyWindow {
        window_s: 2.0,
        samples: 8,
        energy_j: 10.0,
        mean_power_w: 5.0,
        ..Default::default()
    };
    let busy = EnergyWindow {
        window_s: 30.0,
        samples: 120,
        energy_j: 1800.0,
        mean_power_w: 60.0,
        ..Default::default()
    };
    assert!((busy.above_idle_j(&idle) - 1650.0).abs() < 1e-9);
    let quieter = EnergyWindow {
        window_s: 10.0,
        energy_j: 40.0,
        mean_power_w: 4.0,
        ..busy
    };
    assert!(
        (quieter.above_idle_j(&idle) - -10.0).abs() < 1e-9,
        "not clamped"
    );
}

/// Joules and seconds add; the mean is re-derived; fractions weight by
/// sample count. This is the re-aggregation the "store primitives" rule
/// exists for.
#[test]
fn summing_windows_adds_the_additive_primitives_and_rederives_the_mean() {
    let a = EnergyWindow {
        window_s: 10.0,
        samples: 40,
        energy_j: 600.0,
        mean_power_w: 60.0,
        max_power_w: 70.0,
        sw_power_cap_frac: Some(1.0),
        hw_power_brake_frac: None,
    };
    let b = EnergyWindow {
        window_s: 30.0,
        samples: 120,
        energy_j: 900.0,
        mean_power_w: 30.0,
        max_power_w: 65.0,
        sw_power_cap_frac: Some(0.5),
        hw_power_brake_frac: Some(0.0),
    };
    let s = EnergyWindow::sum(&[a, b]).unwrap();
    assert_eq!(s.window_s, 40.0);
    assert_eq!(s.samples, 160);
    assert_eq!(s.energy_j, 1500.0);
    assert!((s.mean_power_w - 37.5).abs() < 1e-9);
    assert_eq!(s.max_power_w, 70.0);
    assert!((s.sw_power_cap_frac.unwrap() - (40.0 + 60.0) / 160.0).abs() < 1e-9);
    assert_eq!(
        s.hw_power_brake_frac,
        Some(0.0),
        "weighted over the reporting window only"
    );
    assert_eq!(EnergyWindow::sum(&[]), None);
}

#[test]
fn every_energy_key_names_the_rail_and_the_sample_count_rides_beside_the_joules() {
    let idle = EnergyWindow {
        window_s: 2.0,
        samples: 8,
        energy_j: 10.0,
        mean_power_w: 5.0,
        ..Default::default()
    };
    let w = EnergyWindow {
        window_s: 30.0,
        samples: 120,
        energy_j: 1800.0,
        mean_power_w: 60.0,
        max_power_w: 75.0,
        sw_power_cap_frac: Some(0.9),
        hw_power_brake_frac: None,
    };
    let mut m = BTreeMap::new();
    w.metrics("c8_", 2560, Some(&idle), &mut m);
    assert!(m.keys().all(|k| k.contains("gpu_rail")), "{m:?}");
    assert_eq!(m["c8_gpu_rail_energy_j"], 1800.0);
    assert_eq!(m["c8_gpu_rail_energy_window_tokens"], 2560.0);
    assert_eq!(m["c8_gpu_rail_power_samples"], 120.0);
    assert_eq!(m["c8_gpu_rail_mean_power_w"], 60.0);
    assert_eq!(m["c8_gpu_rail_max_power_w"], 75.0);
    assert_eq!(m["c8_gpu_rail_energy_window_s"], 30.0);
    assert_eq!(m["c8_gpu_rail_sw_power_cap_frac"], 0.9);
    assert!(
        !m.contains_key("c8_gpu_rail_hw_power_brake_frac"),
        "unreported stays absent"
    );
    assert!((m["c8_gpu_rail_energy_above_idle_j"] - 1650.0).abs() < 1e-9);
    // Without a baseline the above-idle key is absent, never zero.
    let mut bare = BTreeMap::new();
    w.metrics("", 915, None, &mut bare);
    assert!(!bare.contains_key("gpu_rail_energy_above_idle_j"));
    assert_eq!(bare["gpu_rail_energy_j"], 1800.0);
    // Joules and tokens are stored; no key stores their ratio.
    assert!(
        m.keys()
            .chain(bare.keys())
            .all(|k| !k.contains("per_token") && !k.contains("_per_") && !k.contains("tok_j")),
        "{m:?} {bare:?}"
    );
}

#[test]
fn sampler_cost_reports_its_cadence_and_never_hides_rejected_lines() {
    let mut m = BTreeMap::new();
    SamplerCost {
        cpu_s: Some(0.021),
        wall_s: 95.0,
        samples: 380,
        rejected_lines: 0,
    }
    .metrics(&mut m);
    assert_eq!(m["gpu_rail_sample_period_ms"], 250.0);
    assert_eq!(m["gpu_rail_sampler_cpu_s"], 0.021);
    assert_eq!(m["gpu_rail_sampler_wall_s"], 95.0);
    assert!(!m.contains_key("gpu_rail_sampler_rejected_lines"));
    let mut m = BTreeMap::new();
    let cost = SamplerCost {
        cpu_s: None,
        wall_s: 5.0,
        samples: 3,
        rejected_lines: 17,
    };
    cost.metrics(&mut m);
    assert_eq!(m["gpu_rail_sampler_rejected_lines"], 17.0);
    assert!(
        !m.contains_key("gpu_rail_sampler_cpu_s"),
        "unmeasured is absent, not 0"
    );
    assert!(cost.one_line().contains("unmeasured"));
    assert!(cost.one_line().contains("17 unparseable"));
}
