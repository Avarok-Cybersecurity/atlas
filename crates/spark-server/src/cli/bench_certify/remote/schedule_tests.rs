// SPDX-License-Identifier: AGPL-3.0-only
use super::super::super::plan::{Estimate, Unit};
use super::*;
use atlas_plugin::hardware::equivalence::HardwareFingerprint;

fn unit(id: &'static str, group: Option<&'static str>, class: Sensitivity, secs: u64) -> Unit {
    Unit {
        id,
        group,
        class,
        estimate: Estimate::Declared(secs),
        needs_confirmation: false,
    }
}

fn gb10(chassis: f64) -> HardwareFingerprint {
    HardwareFingerprint {
        gpu: "NVIDIA GB10".into(),
        driver_major: Some(580),
        sm_clock_max_mhz: Some(3003.0),
        mem_total_kb: Some(127_601_452),
        thermal_alert: Some(false),
        hottest_chassis_c: Some(chassis),
        postcheck_valid: None,
    }
}

fn node(addr: &str, chassis: f64, free: f64, built: bool) -> Node {
    Node {
        addr: addr.into(),
        name: addr.into(),
        node_id: String::new(),
        signer: "s".into(),
        hardware: gb10(chassis),
        free_fraction: Some(free),
        built,
        local: false,
    }
}

/// The real remaining set at a merged-down base: 12 gates, 18 units.
fn campaign() -> Vec<Unit> {
    let mut v = vec![
        unit("decode-floor", None, Sensitivity::Speed, 180),
        unit("ttft-warm-gate", None, Sensitivity::Speed, 160),
        unit("ttft-cold-gate", None, Sensitivity::Speed, 120),
        unit("agentic-webserver", None, Sensitivity::Speed, 600),
        unit("concurrency-sweep", None, Sensitivity::Speed, 1560),
        unit("concurrency-sweep-dflash2", None, Sensitivity::Speed, 300),
        unit("vision-fidelity", None, Sensitivity::Correctness, 120),
        unit("video-fidelity", None, Sensitivity::Correctness, 70),
        unit(
            "ssm-state-poisoning-gate",
            None,
            Sensitivity::Correctness,
            150,
        ),
        unit("kat-equality-gate", None, Sensitivity::Correctness, 4200),
    ];
    for s in ["a", "b", "c", "d"] {
        v.push(unit(
            Box::leak(format!("bfcl-subset-{s}").into_boxed_str()),
            Some("bfcl-subset"),
            Sensitivity::Correctness,
            1500,
        ));
        v.push(unit(
            Box::leak(format!("bfcl-subset-echolp-{s}").into_boxed_str()),
            Some("bfcl-subset-echolp"),
            Sensitivity::Correctness,
            1900,
        ));
    }
    v
}

#[test]
fn speed_mode_spreads_over_equivalent_boxes_and_bundles_otherwise() {
    assert_eq!(speed_mode(&[node("a", 65.0, 0.9, true)]), SpeedMode::Spread);
    assert_eq!(
        speed_mode(&[node("a", 65.0, 0.9, true), node("b", 70.0, 0.9, true)]),
        SpeedMode::Spread
    );
    // The incident pair: bundled on the box with more free memory, and the
    // warning names the field.
    let m = speed_mode(&[node("a", 65.0, 0.80, true), node("b", 89.0, 0.90, true)]);
    match m {
        SpeedMode::Bundle { node, why } => {
            assert_eq!(node, 1, "more free memory wins");
            assert!(why[0].contains("chassis 65 vs 89"), "{why:?}");
        }
        other => panic!("{other:?}"),
    }
    // Equal memory: the cooler box.
    let m = speed_mode(&[node("a", 65.0, 0.9, true), node("b", 89.0, 0.9, true)]);
    assert!(matches!(m, SpeedMode::Bundle { node: 0, .. }), "{m:?}");
    // NEGATIVE CONTROL: a node that cannot report its thermal state is not
    // equivalent to anything, even to an identical one.
    let mut blind = node("c", 65.0, 0.9, true);
    blind.hardware.hottest_chassis_c = None;
    assert!(matches!(
        speed_mode(&[node("a", 65.0, 0.9, true), blind]),
        SpeedMode::Bundle { .. }
    ));
}

#[test]
fn next_for_is_longest_first_with_shard_anti_affinity_and_the_speed_rule() {
    let units = campaign();
    let mut pending = vec![true; units.len()];
    let mut placed = vec![None; units.len()];
    // Spread: the longest unit of all first, on any node.
    let first = next_for(1, &units, &pending, &placed, &SpeedMode::Spread).unwrap();
    assert_eq!(units[first].id, "kat-equality-gate");
    pending[first] = false;
    placed[first] = Some(1);
    // Node 1 again: an echolp shard (1900) — and once it hosts one, the next
    // echolp shard yields to the longest OTHER unit (bfcl 1500 < sweep
    // 1560, so the sweep) rather than crowding.
    let e = next_for(1, &units, &pending, &placed, &SpeedMode::Spread).unwrap();
    assert!(
        units[e].id.starts_with("bfcl-subset-echolp-"),
        "{}",
        units[e].id
    );
    pending[e] = false;
    placed[e] = Some(1);
    let next = next_for(1, &units, &pending, &placed, &SpeedMode::Spread).unwrap();
    assert_eq!(
        units[next].id, "concurrency-sweep",
        "shard anti-affinity yields"
    );
    // Another node takes the next echolp shard freely.
    let e2 = next_for(0, &units, &pending, &placed, &SpeedMode::Spread).unwrap();
    assert!(units[e2].id.starts_with("bfcl-subset-echolp-"));
    // Bundle on node 0: node 1 never gets a Speed unit, even when only
    // Speed units remain.
    let only_speed: Vec<bool> = units
        .iter()
        .map(|u| u.class == Sensitivity::Speed)
        .collect();
    let bundle = SpeedMode::Bundle {
        node: 0,
        why: vec![],
    };
    assert_eq!(next_for(1, &units, &only_speed, &placed, &bundle), None);
    let s = next_for(0, &units, &only_speed, &placed, &bundle).unwrap();
    assert_eq!(units[s].id, "concurrency-sweep");
    // Nothing pending: None.
    assert_eq!(
        next_for(
            0,
            &units,
            &vec![false; units.len()],
            &placed,
            &SpeedMode::Spread
        ),
        None
    );
}

#[test]
fn the_simulation_is_work_conserving_and_near_optimal() {
    let units = campaign();
    let total: u64 = units.iter().map(Unit::secs).sum();
    let longest = units.iter().map(Unit::secs).max().unwrap();
    // One node: the serial sum.
    let one = simulate(
        &units,
        &[node("a", 65.0, 0.9, true)],
        &SpeedMode::Spread,
        1800,
    );
    assert_eq!(one.makespan_secs, total);
    assert_eq!(one.queues[0].len(), units.len());
    // Three equivalent nodes: every unit placed exactly once, makespan within
    // LPT's 4/3 bound of the trivial lower bound.
    let nodes = [
        node("a", 65.0, 0.9, true),
        node("b", 66.0, 0.9, true),
        node("c", 67.0, 0.9, true),
    ];
    let three = simulate(&units, &nodes, &SpeedMode::Spread, 1800);
    let placed: usize = three.queues.iter().map(Vec::len).sum();
    assert_eq!(placed, units.len());
    let lower = (total / 3).max(longest);
    assert!(
        three.makespan_secs <= lower * 4 / 3 + longest / 3,
        "makespan {} vs lower bound {lower}",
        three.makespan_secs
    );
    assert!(three.makespan_secs < total / 2, "{}", three.makespan_secs);
    // A node without the anchor built pays the build allowance once.
    let cold = [node("a", 65.0, 0.9, true), node("b", 66.0, 0.9, false)];
    let p = simulate(&units, &cold, &SpeedMode::Spread, 1800);
    let b_work: u64 = p.queues[1].iter().map(|&i| units[i].secs()).sum();
    assert_eq!(p.finish_at[1], b_work + 1800);
    // Bundle: every Speed unit is on the home node and nowhere else.
    let bundle = SpeedMode::Bundle {
        node: 0,
        why: vec![],
    };
    let p = simulate(&units, &nodes, &bundle, 0);
    for (k, q) in p.queues.iter().enumerate() {
        for &i in q {
            if units[i].class == Sensitivity::Speed {
                assert_eq!(k, 0, "{} landed on node {k}", units[i].id);
            }
        }
    }
    assert_eq!(p.queues.iter().map(Vec::len).sum::<usize>(), units.len());
}
