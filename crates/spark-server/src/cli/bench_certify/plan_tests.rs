// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

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
fn a_group_expands_to_its_shards_in_order() {
    assert_eq!(
        expand("bfcl-subset", &all_members),
        [
            "bfcl-subset-a",
            "bfcl-subset-b",
            "bfcl-subset-c",
            "bfcl-subset-d"
        ]
    );
    assert_eq!(expand("decode-floor", &all_members), ["decode-floor"]);
}

#[test]
fn every_required_gate_yields_units_with_a_positive_estimate() {
    let all = units(&gate::REQUIRED_GATES, &|_| None, &all_members).unwrap();
    // 10 plain gates + 2 groups × 4 shards.
    assert_eq!(all.len(), gate::REQUIRED_GATES.len() - 2 + 8);
    for u in &all {
        assert!(u.secs() > 0, "{}", u.id);
        assert!(matches!(u.estimate, Estimate::Declared(_)), "{}", u.id);
    }
    let shards = all.iter().filter(|u| u.group.is_some()).count();
    assert_eq!(shards, 8);
}

/// The shards a group still owes are the only ones planned: a banked shard
/// is not re-measured, and a plain gate is untouched by the question.
#[test]
fn a_group_expands_only_to_the_shards_it_still_owes() {
    let owed = |g: &'static gate::group::BenchmarkGroup| -> Vec<&'static str> {
        if g.id == "bfcl-subset-echolp" {
            vec!["bfcl-subset-echolp-c", "bfcl-subset-echolp-d"]
        } else {
            g.members.to_vec()
        }
    };
    let all = units(
        &["bfcl-subset-echolp", "bfcl-subset", "decode-floor"],
        &|_| None,
        &owed,
    )
    .unwrap();
    let ids: Vec<&str> = all.iter().map(|u| u.id).collect();
    assert_eq!(
        ids,
        [
            "bfcl-subset-echolp-c",
            "bfcl-subset-echolp-d",
            "bfcl-subset-a",
            "bfcl-subset-b",
            "bfcl-subset-c",
            "bfcl-subset-d",
            "decode-floor",
        ]
    );
    // NEGATIVE CONTROL: a group that owes nothing yields no unit at all —
    // the campaign has nothing to run for it, and the verdict comes from the
    // final check.
    let none = |_: &'static gate::group::BenchmarkGroup| -> Vec<&'static str> { vec![] };
    assert!(
        units(&["bfcl-subset"], &|_| None, &none)
            .unwrap()
            .is_empty()
    );
}

/// The deadline pays for the server start-up first, then the scaled
/// estimate — so a 17 s bench is not killed at 51 s while its checkpoint
/// loads. The factor scales only the measured part.
#[test]
fn the_deadline_is_the_serve_allowance_plus_the_scaled_estimate() {
    let all = units(&["video-fidelity"], &|_| Some((17, 1)), &all_members).unwrap();
    assert_eq!(all[0].secs(), 17);
    assert_eq!(
        all[0].deadline(3.0),
        SERVE_ALLOWANCE + std::time::Duration::from_secs(51)
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
        &all_members,
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
    let all = units(&["decode-floor"], &|_| Some((0, 1)), &all_members).unwrap();
    assert!(matches!(all[0].estimate, Estimate::Declared(_)));
}

#[test]
fn local_order_puts_groups_first_then_speed_shortest_first() {
    let all = order_local(units(&gate::REQUIRED_GATES, &|_| None, &all_members).unwrap());
    let ids: Vec<&str> = all.iter().map(|u| u.id).collect();
    // The longest shard set (echolp) leads, then the golden shards.
    assert!(ids[0].starts_with("bfcl-subset-echolp-"), "{ids:?}");
    assert!(ids[4].starts_with("bfcl-subset-"), "{ids:?}");
    assert!(!ids[4].contains("echolp"), "{ids:?}");
    let first_plain = all.iter().position(|u| u.group.is_none()).unwrap();
    assert_eq!(first_plain, 8);
    let speed: Vec<u64> = all
        .iter()
        .filter(|u| u.group.is_none() && u.class == Sensitivity::Speed)
        .map(Unit::secs)
        .collect();
    assert!(speed.windows(2).all(|w| w[0] <= w[1]), "{speed:?}");
    // Deterministic.
    let again = order_local(units(&gate::REQUIRED_GATES, &|_| None, &all_members).unwrap());
    assert_eq!(ids, again.iter().map(|u| u.id).collect::<Vec<_>>());
}
