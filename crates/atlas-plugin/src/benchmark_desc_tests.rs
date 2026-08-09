// SPDX-License-Identifier: AGPL-3.0-only

//! Every shipped benchmark must declare when its measurement last changed.

#[test]
fn every_benchmark_declares_an_updated_date() {
    for d in crate::registry::all() {
        assert!(
            !d.updated.is_empty(),
            "{} carries no updated date — a reader cannot tell whether two \
             runs of it are comparable",
            d.id
        );
        // ISO `YYYY-MM-DD`, so it sorts and reads the same as a recipe's.
        assert_eq!(d.updated.len(), 10, "{}: {:?}", d.id, d.updated);
        let parts: Vec<&str> = d.updated.split('-').collect();
        assert_eq!(parts.len(), 3, "{}: {:?}", d.id, d.updated);
        for p in parts {
            assert!(
                p.chars().all(|c| c.is_ascii_digit()),
                "{}: {:?} is not a date",
                d.id,
                d.updated
            );
        }
    }
}
