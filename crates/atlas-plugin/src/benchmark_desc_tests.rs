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
        for p in &parts {
            assert!(
                p.chars().all(|c| c.is_ascii_digit()),
                "{}: {:?} is not a date",
                d.id,
                d.updated
            );
        }

        let year: u32 = parts[0].parse().unwrap();
        let month: u32 = parts[1].parse().unwrap();
        let day: u32 = parts[2].parse().unwrap();
        #[allow(
            clippy::manual_is_multiple_of,
            reason = "keep the workspace's Rust 1.85 MSRV"
        )]
        let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
        let days_in_month = match month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 if leap => 29,
            2 => 28,
            _ => panic!("{}: {:?} has no such month", d.id, d.updated),
        };
        assert!(
            (1..=days_in_month).contains(&day),
            "{}: {:?} has no such day",
            d.id,
            d.updated
        );
    }
}
