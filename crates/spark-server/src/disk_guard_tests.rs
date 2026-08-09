// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn u(free_gb: f64, total_gb: f64) -> Usage {
    Usage {
        free: (free_gb * 1e9) as u64,
        total: (total_gb * 1e9) as u64,
    }
}

#[test]
fn a_healthy_disk_says_nothing() {
    // Silence is the common case and must stay silent — a warning that fires
    // on every start is one nobody reads when it matters.
    assert!(
        u(2900.0, 3700.0).warning(Path::new("/")).is_none(),
        "20% used"
    );
    assert!(
        u(370.0, 3700.0).warning(Path::new("/")).is_none(),
        "90% used"
    );
    assert!(
        u(120.0, 3700.0).warning(Path::new("/")).is_none(),
        "96.8% used"
    );
}

#[test]
fn the_threshold_is_where_it_is_documented_to_be() {
    // Exactly 97% warns; a hair under does not.
    assert!(
        u(111.0, 3700.0).warning(Path::new("/")).is_some(),
        "97.0% used"
    );
    assert!(
        u(112.0, 3700.0).warning(Path::new("/")).is_none(),
        "96.97% used"
    );
}

#[test]
fn a_nearly_full_disk_says_what_and_how_much() {
    let w = u(60.0, 3700.0)
        .warning(Path::new("/workspace/.cache/huggingface/hub"))
        .expect("98% must warn");
    assert!(w.contains("98%"), "states the percentage: {w}");
    assert!(w.contains("60.0 GB"), "and the absolute free space: {w}");
    assert!(w.contains("huggingface"), "and WHICH filesystem: {w}");
    // The reason it matters, because "disk nearly full" alone is not
    // actionable at 3am.
    assert!(w.contains("downloads") || w.contains("thrashing"), "{w}");
}

#[test]
fn a_full_disk_does_not_divide_by_zero_or_panic() {
    assert_eq!(u(0.0, 0.0).used_fraction(), 0.0, "unknown, not 100%");
    assert!(u(0.0, 0.0).warning(Path::new("/")).is_none());
    assert!(
        u(0.0, 3700.0).warning(Path::new("/")).is_some(),
        "100% used"
    );
}

#[test]
fn the_reading_is_real_on_this_filesystem() {
    // Guards the syscall wiring, not the threshold: a `usage` that always
    // returned None would make every test above vacuous in production.
    let got = usage(Path::new("/")).expect("/ is measurable on a unix box");
    assert!(got.total > 0);
    assert!(got.free <= got.total);
    let f = got.used_fraction();
    assert!((0.0..=1.0).contains(&f), "fraction out of range: {f}");
}
