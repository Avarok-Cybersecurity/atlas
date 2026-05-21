// SPDX-License-Identifier: AGPL-3.0-only
//
// Hash-combining helpers — port of the genuinely-reusable part of
// `cpp/support/utils.h`.
//
// Skipped from `utils.h` (C++-template machinery with no Rust need):
//   - `HashByMembers` / `XGRAMMAR_HASH_BY_MEMBERS`: replaced by
//     `#[derive(Hash)]`.
//   - `std::hash` specializations for pair/tuple/vector: Rust derives
//     these.
//   - `PartialResult` / `Result` / `ResultOk` / `ResultErr` /
//     `Convert`: replaced by the standard library `Result`.
//   - `TypedError`, `ThrowVariantError`, `GetMessageFromVariantError`:
//     replaced by `thiserror`-based error enums.
//   - `XGRAMMAR_UNREACHABLE`, `EqualByMembers` / `XGRAMMAR_EQUAL_BY_MEMBERS`:
//     replaced by `unreachable!()` and `#[derive(PartialEq, Eq)]`.
//
// Only `HashCombineBinary` / `HashCombine` carry over: they implement
// the specific boost-derived mixing function xgrammar relies on for
// structural hashes, which `#[derive(Hash)]` does not reproduce.

/// Mix `value` into `seed` in place.
///
/// Faithful to C++ `HashCombineBinary` — the boost `hash_combine`
/// 64-bit mixing step. Uses wrapping arithmetic to match C++ unsigned
/// overflow semantics.
pub fn hash_combine_binary(seed: &mut u64, value: u64) {
    *seed ^= value
        .wrapping_add(0x9e3779b97f4a7c15)
        .wrapping_add(*seed << 6)
        .wrapping_add(*seed >> 2);
}

/// Fold a slice of values into a single combined hash, starting from
/// seed `0`. Faithful to the variadic C++ `HashCombine`.
pub fn hash_combine(values: &[u64]) -> u64 {
    let mut seed = 0u64;
    for &v in values {
        hash_combine_binary(&mut seed, v);
    }
    seed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_binary_is_deterministic() {
        let mut a = 0u64;
        let mut b = 0u64;
        hash_combine_binary(&mut a, 42);
        hash_combine_binary(&mut b, 42);
        assert_eq!(a, b);
        assert_ne!(a, 0);
    }

    #[test]
    fn combine_binary_order_sensitive() {
        let mut a = 0u64;
        hash_combine_binary(&mut a, 1);
        hash_combine_binary(&mut a, 2);

        let mut b = 0u64;
        hash_combine_binary(&mut b, 2);
        hash_combine_binary(&mut b, 1);

        assert_ne!(a, b);
    }

    #[test]
    fn combine_matches_manual_folding() {
        let mut seed = 0u64;
        hash_combine_binary(&mut seed, 7);
        hash_combine_binary(&mut seed, 8);
        hash_combine_binary(&mut seed, 9);
        assert_eq!(hash_combine(&[7, 8, 9]), seed);
    }

    #[test]
    fn combine_empty_is_zero() {
        assert_eq!(hash_combine(&[]), 0);
    }

    #[test]
    fn combine_distinguishes_inputs() {
        assert_ne!(hash_combine(&[1, 2, 3]), hash_combine(&[1, 2, 4]));
        assert_ne!(hash_combine(&[1, 2]), hash_combine(&[2, 1]));
    }

    #[test]
    fn combine_no_overflow_panic() {
        // Large values must not panic (wrapping arithmetic).
        let mut seed = u64::MAX;
        hash_combine_binary(&mut seed, u64::MAX);
        let _ = hash_combine(&[u64::MAX, u64::MAX, u64::MAX]);
    }
}
