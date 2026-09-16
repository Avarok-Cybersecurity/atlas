// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// The committed GB10 timing limits.
fn timing() -> TimingLimits {
    TimingLimits {
        serve_allowance_s: 600,
        boot_timeout_s: 900,
        shard_overhead_s: 420,
        shard_floor_s: 300,
        build_allowance_s: 1800,
    }
}

fn statuses(pass: &[&str]) -> BTreeMap<String, GateStatus> {
    gate::REQUIRED_GATES
        .iter()
        .map(|id| {
            let s = if pass.contains(id) {
                GateStatus::Pass
            } else {
                GateStatus::Missing("no gate records committed".into())
            };
            (id.to_string(), s)
        })
        .collect()
}

#[test]
fn a_passing_gate_is_not_remaining() {
    let left = remaining(&statuses(&["decode-floor", "vision-fidelity"]), &[]).unwrap();
    assert_eq!(left.len(), gate::REQUIRED_GATES.len() - 2);
    assert!(!left.contains(&"decode-floor"));
    assert!(!left.contains(&"vision-fidelity"));
}

#[test]
fn a_failed_gate_is_remaining() {
    let mut s = statuses(&[]);
    s.insert(
        "decode-floor".into(),
        GateStatus::Fail(vec!["too slow".into()]),
    );
    assert!(remaining(&s, &[]).unwrap().contains(&"decode-floor"));
}

/// NEGATIVE CONTROL: naming a gate that already passes is an error, not a
/// silent re-run and not a silent skip.
#[test]
fn naming_an_already_passing_gate_is_refused() {
    let err = remaining(&statuses(&["decode-floor"]), &["decode-floor".into()]).unwrap_err();
    assert!(err.to_string().contains("already passes"), "{err}");
}

#[test]
fn a_group_expands_to_the_shards_of_a_fresh_partition_in_order() {
    let four = fresh_partition(4);
    assert_eq!(
        expand("bfcl-subset", &four),
        [
            ("bfcl-subset", Some((0, 4))),
            ("bfcl-subset", Some((1, 4))),
            ("bfcl-subset", Some((2, 4))),
            ("bfcl-subset", Some((3, 4))),
        ]
    );
    assert_eq!(expand("decode-floor", &four), [("decode-floor", None)]);
    // The count is the campaign's to choose; a plain gate ignores it.
    assert_eq!(
        expand("bfcl-subset", &fresh_partition(1)),
        [("bfcl-subset", Some((0, 1)))]
    );
    assert_eq!(expand("bfcl-subset", &fresh_partition(7)).len(), 7);
    assert_eq!(expand("decode-floor", &fresh_partition(7)).len(), 1);
}

#[test]
fn every_required_gate_yields_units_with_a_positive_estimate() {
    let all = units(
        &gate::REQUIRED_GATES,
        &|_| None,
        &fresh_partition(6),
        &timing(),
    )
    .unwrap();
    // 10 plain gates + 2 groups × 6 shards.
    assert_eq!(all.len(), gate::REQUIRED_GATES.len() - 2 + 12);
    for u in &all {
        assert!(u.secs() > 0, "{}", u.label());
        assert!(matches!(u.estimate, Estimate::Declared(_)), "{}", u.label());
    }
    let shards = all.iter().filter(|u| u.group.is_some()).count();
    assert_eq!(shards, 12);
    assert!(all.iter().all(|u| u.group.is_some() == u.shard.is_some()));
}

/// The shards a group still owes are the only ones planned: a banked shard
/// is not re-measured, a partition begun at another count is finished at
/// THAT count, and a plain gate is untouched by the question.
#[test]
fn a_group_expands_only_to_the_shards_it_still_owes() {
    let owed = |g: &'static gate::group::BenchmarkGroup| -> Vec<(usize, usize)> {
        if g.id == "bfcl-subset-echolp" {
            vec![(2, 4), (3, 4)]
        } else {
            fresh_partition(6)(g)
        }
    };
    let all = units(
        &["bfcl-subset-echolp", "bfcl-subset", "decode-floor"],
        &|_| None,
        &owed,
        &timing(),
    )
    .unwrap();
    let labels: Vec<String> = all.iter().map(Unit::label).collect();
    assert_eq!(
        labels,
        [
            "bfcl-subset-echolp[2/4]",
            "bfcl-subset-echolp[3/4]",
            "bfcl-subset[0/6]",
            "bfcl-subset[1/6]",
            "bfcl-subset[2/6]",
            "bfcl-subset[3/6]",
            "bfcl-subset[4/6]",
            "bfcl-subset[5/6]",
            "decode-floor",
        ]
    );
    // NEGATIVE CONTROL: a group that owes nothing yields no unit at all —
    // the campaign has nothing to run for it, and the verdict comes from the
    // final check.
    let none = |_: &'static gate::group::BenchmarkGroup| -> Vec<(usize, usize)> { vec![] };
    assert!(
        units(&["bfcl-subset"], &|_| None, &none, &timing())
            .unwrap()
            .is_empty()
    );
}

/// A shard's estimate is its group's divided by the count, floored: the
/// server start and warm-up do not shrink with the slice. And the child's
/// argument names exactly the slice the unit stands for.
#[test]
fn a_shards_estimate_is_the_groups_share_floored_and_its_param_names_the_slice() {
    let whole = units(
        &["bfcl-subset"],
        &|_| Some((6000, 1)),
        &fresh_partition(1),
        &timing(),
    )
    .unwrap();
    assert_eq!(whole[0].secs(), 6000 + timing().shard_overhead_s);
    assert_eq!(whole[0].shard_param().as_deref(), Some("shard=0/1"));
    assert_eq!(whole[0].file_stem(), "bfcl-subset-s0of1");
    let four = units(
        &["bfcl-subset"],
        &|_| Some((6000, 1)),
        &fresh_partition(4),
        &timing(),
    )
    .unwrap();
    assert_eq!(four[3].secs(), 1500 + timing().shard_overhead_s);
    assert_eq!(four[3].shard_param().as_deref(), Some("shard=3/4"));
    assert_eq!(four[3].label(), "bfcl-subset[3/4]");
    // Declared estimates divide the same way.
    let eight = units(&["bfcl-subset"], &|_| None, &fresh_partition(8), &timing()).unwrap();
    let declared = atlas_plugin::registry::find("bfcl-subset")
        .unwrap()
        .expected_secs;
    assert_eq!(eight[0].secs(), declared / 8 + timing().shard_overhead_s);
    // The floor: a hundred-way split is not a hundred one-minute runs.
    let thin = units(
        &["bfcl-subset"],
        &|_| Some((6000, 1)),
        &fresh_partition(100),
        &timing(),
    )
    .unwrap();
    assert_eq!(
        thin[0].secs(),
        (60 + timing().shard_overhead_s).max(timing().shard_floor_s)
    );
    // The overhead is what a six-way echolp shard measured: share 1260 s,
    // wall 1650-1873 s.
    assert!(
        (1650..=1873).contains(&shard_secs(7560, 6, &timing())),
        "{}",
        shard_secs(7560, 6, &timing())
    );
    // A plain gate has no shard, no param, and its stem is its id.
    let plain = units(&["decode-floor"], &|_| None, &fresh_partition(4), &timing()).unwrap();
    assert_eq!(plain[0].shard_param(), None);
    assert_eq!(plain[0].file_stem(), "decode-floor");
}

/// Two shards per box, so the list scheduler has slices to balance around
/// the long gates — and one box alone runs the whole draw, since a shard
/// costs a server start and there is nothing to balance against.
#[test]
fn the_default_shard_count_is_two_per_box_and_one_for_a_lone_box() {
    assert_eq!(shard_count(0), 1);
    assert_eq!(shard_count(1), 1);
    assert_eq!(shard_count(2), 4);
    assert_eq!(shard_count(3), 6);
}

/// The deadline pays for the server start-up first, then the scaled
/// estimate — so a 17 s bench is not killed at 51 s while its checkpoint
/// loads. The factor scales only the measured part.
#[test]
fn the_deadline_is_the_serve_allowance_plus_the_scaled_estimate() {
    let all = units(
        &["video-fidelity"],
        &|_| Some((17, 1)),
        &fresh_partition(4),
        &timing(),
    )
    .unwrap();
    assert_eq!(all[0].secs(), 17);
    assert_eq!(
        all[0].deadline(3.0),
        std::time::Duration::from_secs(600) + std::time::Duration::from_secs(51)
    );
    // The measured 40 s checkpoint load plus the bench fits at factor 1; the
    // rule this replaces (estimate × factor alone) did not.
    let load_plus_bench = std::time::Duration::from_secs(40 + 17);
    assert!(all[0].deadline(1.0) > load_plus_bench);
    assert!(std::time::Duration::from_secs(17 * 3) < load_plus_bench);
}

#[test]
fn a_measured_duration_beats_the_declared_one() {
    let all = units(
        &["decode-floor"],
        &|id| (id == "decode-floor").then_some((155, 1_789_000_000)),
        &fresh_partition(4),
        &timing(),
    )
    .unwrap();
    assert_eq!(
        all[0].estimate,
        Estimate::Measured {
            secs: 155,
            recorded_at: 1_789_000_000
        }
    );
    // A zero measurement is not a measurement.
    let all = units(
        &["decode-floor"],
        &|_| Some((0, 1)),
        &fresh_partition(4),
        &timing(),
    )
    .unwrap();
    assert!(matches!(all[0].estimate, Estimate::Declared(_)));
}

#[test]
fn local_order_puts_groups_first_then_speed_shortest_first() {
    let all = order_local(
        units(
            &gate::REQUIRED_GATES,
            &|_| None,
            &fresh_partition(4),
            &timing(),
        )
        .unwrap(),
    );
    let ids: Vec<String> = all.iter().map(Unit::label).collect();
    // The longest shard set (echolp) leads, then the golden shards.
    assert!(ids[0].starts_with("bfcl-subset-echolp["), "{ids:?}");
    assert!(ids[4].starts_with("bfcl-subset["), "{ids:?}");
    let first_plain = all.iter().position(|u| u.group.is_none()).unwrap();
    assert_eq!(first_plain, 8);
    let speed: Vec<u64> = all
        .iter()
        .filter(|u| u.group.is_none() && u.class == Sensitivity::Speed)
        .map(Unit::secs)
        .collect();
    assert!(speed.windows(2).all(|w| w[0] <= w[1]), "{speed:?}");
    // Deterministic.
    let again = order_local(
        units(
            &gate::REQUIRED_GATES,
            &|_| None,
            &fresh_partition(4),
            &timing(),
        )
        .unwrap(),
    );
    assert_eq!(ids, again.iter().map(Unit::label).collect::<Vec<_>>());
}

/// A measured shard run is scaled back to the whole draw before the planner
/// divides it again; a whole run, or a shard of one, is left alone.
#[test]
fn a_measured_shard_run_is_scaled_to_the_whole_draw() {
    use super::super::whole_draw_secs;
    let mut p = std::collections::BTreeMap::new();
    assert_eq!(whole_draw_secs(300, &p), 300);
    p.insert("shard".to_string(), "4/6".to_string());
    assert_eq!(whole_draw_secs(300, &p), 1800);
    p.insert("shard".to_string(), "0/1".to_string());
    assert_eq!(whole_draw_secs(300, &p), 300);
    p.insert("shard".to_string(), "inherit".to_string());
    assert_eq!(whole_draw_secs(300, &p), 300);
}
