// SPDX-License-Identifier: AGPL-3.0-only

use super::mtp_rows_to_trim;

#[test]
fn flag_off_is_exactly_the_legacy_behaviour() {
    // Legacy: trim only the rejected rows. These are the K=2/3/4 cases
    // the schedulers actually produce.
    assert_eq!(mtp_rows_to_trim(1, 0, false), 1); // K=2 reject
    assert_eq!(mtp_rows_to_trim(1, 1, false), 0); // K=2 accept
    assert_eq!(mtp_rows_to_trim(2, 0, false), 2); // K=3 reject
    assert_eq!(mtp_rows_to_trim(2, 1, false), 1); // K=3 accept-1
    assert_eq!(mtp_rows_to_trim(2, 2, false), 0); // K=3 accept-2
    assert_eq!(mtp_rows_to_trim(3, 3, false), 0); // K=4 accept-3
}

#[test]
fn flag_on_also_drops_accepted_rows_past_the_first() {
    // The first accepted draft used the TARGET hidden — it stays.
    assert_eq!(mtp_rows_to_trim(1, 1, true), 0); // K=2 accept: nothing extra
    assert_eq!(mtp_rows_to_trim(2, 1, true), 1); // K=3 accept-1: rejected only
    assert_eq!(mtp_rows_to_trim(2, 2, true), 1); // K=3 accept-2: drop draft 2
    assert_eq!(mtp_rows_to_trim(3, 2, true), 2); // K=4 accept-2: 1 rejected + 1
    assert_eq!(mtp_rows_to_trim(3, 3, true), 2); // K=4 accept-3: drop drafts 2,3
}

#[test]
fn full_reject_is_identical_with_and_without_the_flag() {
    // Nothing was accepted, so there is no drafter-hidden row to rebuild.
    for d in 0..8 {
        assert_eq!(mtp_rows_to_trim(d, 0, true), mtp_rows_to_trim(d, 0, false));
    }
}

#[test]
fn never_trims_more_rows_than_were_drafted() {
    for d in 0..8 {
        for a in 0..=d + 2 {
            assert!(mtp_rows_to_trim(d, a, true) <= d, "d={d} a={a}");
            assert!(mtp_rows_to_trim(d, a, false) <= d, "d={d} a={a}");
        }
    }
}
