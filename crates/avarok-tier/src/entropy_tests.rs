// SPDX-License-Identifier: AGPL-3.0-only

//! `random_u64` sanity: it must succeed and actually vary — a constant (or
//! erroring) source would silently give every process the SAME decode client
//! salt, which is exactly the cross-client sharing the salt exists to prevent.

use super::random_u64;

#[test]
fn random_u64_succeeds_and_varies() {
    // 8 draws all identical has probability 2^-448 from a real entropy
    // source — this fails only if the source is broken (constant/zero).
    let draws: Vec<u64> = (0..8).map(|_| random_u64().expect("OS entropy")).collect();
    assert!(
        draws.iter().any(|&d| d != draws[0]),
        "8 identical draws — entropy source is degraded/constant: {draws:?}"
    );
}
