// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the `shard` parameter — the arbitrary-N split surface.
//!
//! Split out of `bfcl_tests.rs` to keep that file under the repository's
//! 500-LoC cap, the same way `mtp_carry_tests.rs` was split.

use super::*;

/// The `--param shard=i/n` surface, including every way it is rejected. Each
/// message must name the value and what was wrong with it, because this is the
/// only feedback an operator gets before a multi-hour run.
#[test]
fn the_shard_parameter_parses_and_refuses_precisely() {
    use super::dataset::Shard;
    assert_eq!(Shard::parse("2/7"), Ok(Shard { index: 2, count: 7 }));
    assert_eq!(Shard::parse(" 0 / 4 "), Ok(Shard { index: 0, count: 4 }));
    // The identity: running unsharded must be expressible as a VALUE, not as a
    // different command line. It is `0/1` — and NOT `1/1`, which is index 1 of
    // one shard and correctly refused. That asymmetry is the whole reason the
    // rejection message spells out "0-based".
    assert_eq!(Shard::parse("0/1"), Ok(Shard { index: 0, count: 1 }));
    assert!(
        Shard::parse("1/1").is_err(),
        "1/1 is index 1 of 1, not the whole draw"
    );

    // ★ 0-BASED, and the message says so. A 1-based reading would make `4/4`
    // look valid and silently skip shard 0 for a whole campaign.
    let e = Shard::parse("4/4").expect_err("index 4 of 4 is out of range");
    assert!(e.contains("0-based"), "{e}");
    assert!(e.contains("the last is 3"), "{e}");

    let e = Shard::parse("0/0").expect_err("zero shards");
    assert!(e.contains("at least 1"), "{e}");

    let e = Shard::parse("2-7").expect_err("wrong separator");
    assert!(e.contains("index/count"), "{e}");

    let e = Shard::parse("a/4").expect_err("non-numeric index");
    assert!(e.contains("not a number"), "{e}");
}

/// ★ THE TRAP. `shard` defaults to `inherit`, and this is why.
/// The four registered members set their slice in the constructor; if the
/// default meant "the whole draw", every `configure` — which the TUI and every
/// gate run call — would silently turn all four members into four copies of
/// the whole draw. The union would then be 4 × 995 rows and every sample would
/// be scored four times.
///
/// ★ HOW TO RUN THE CONTROL, because the obvious version of it is INERT.
/// Changing `INHERIT_SHARD` itself proves nothing: `configure` compares the
/// value against that same constant, so the two move together and the test
/// stays green whatever it is set to. The control must change ONLY the
/// ParamSpec's default to a real slice (`ParamValue::Text("0/1")`) while the
/// sentinel stays `inherit`. Done that way, this test and
/// `an_explicit_shard_value_overrides_and_an_empty_one_does_not` go red and
/// nothing else moves. Verified 2026-09-08 — the first attempt at this control
/// was the inert one and passed.
#[test]
fn a_shard_member_keeps_its_slice_under_default_parameters() {
    use super::dataset::Shard;
    let mut b = Bfcl::sharded(Variant::Subset, 2, 4);
    let values = ParamValues::defaults(&b.parameters());
    b.configure(&values).expect("defaults must configure");
    assert_eq!(
        b.shard,
        Some(Shard { index: 2, count: 4 }),
        "an empty `shard` param must leave the constructor's slice alone"
    );
}

/// The whole-draw benchmark stays whole under defaults, and an explicit value
/// still overrides either of them — the parameter is usable, not inert.
#[test]
fn an_explicit_shard_value_overrides_and_an_empty_one_does_not() {
    use super::dataset::Shard;
    let mut whole = Bfcl::new(Variant::Subset);
    let defaults = ParamValues::defaults(&whole.parameters());
    whole.configure(&defaults).expect("defaults");
    assert_eq!(whole.shard, None, "the whole draw stays whole");

    let mut values = defaults.clone();
    values.set("shard", ParamValue::Text("3/9".to_string()));
    whole.configure(&values).expect("explicit shard");
    assert_eq!(whole.shard, Some(Shard { index: 3, count: 9 }));

    // And it overrides a member's constructor slice too, which is the point of
    // having it at all: ad-hoc splits at arbitrary N.
    let mut member = Bfcl::sharded(Variant::Subset, 0, 4);
    member
        .configure(&values)
        .expect("explicit shard on a member");
    assert_eq!(member.shard, Some(Shard { index: 3, count: 9 }));
}

/// A bad value must be refused by `configure`, not carried into the run — the
/// parse is only useful if the call site actually consults it.
#[test]
fn configure_refuses_a_bad_shard_rather_than_ignoring_it() {
    let mut b = Bfcl::new(Variant::Subset);
    let mut values = ParamValues::defaults(&b.parameters());
    values.set("shard", ParamValue::Text("9/4".to_string()));
    let e = b.configure(&values).expect_err("9/4 is out of range");
    assert!(e.to_string().contains("out of range"), "{e}");
}
