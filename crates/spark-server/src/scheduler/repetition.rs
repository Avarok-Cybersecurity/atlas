// SPDX-License-Identifier: AGPL-3.0-only

//! Fuzzy repetition detector + tests.

/// Detect intra-response fuzzy repetition loop in the tail of `tokens`.
///
/// Returns `Some((pattern_len, mis_a, mis_b))` if the last `3 * pattern_len`
/// tokens form three consecutive near-copies of a `pattern_len` window, where
/// each pair (windows A↔B and B↔C) has at most `pattern_len / 12` mismatches
/// (~8% Hamming tolerance, minimum 1). Returns `None` otherwise.
///
/// Requires at least 90 tokens and considers pattern lengths 15..=40 in
/// descending order — longer patterns win ties. Agentic / code output
/// naturally contains 2x boilerplate repeats (function signatures, XML tags);
/// real runaway loops repeat many times, so the 3x threshold cuts false
/// positives seen on alpha-2.43 (20-tok at 138 and 37-tok at 103 boundary hits).
pub fn detect_fuzzy_repetition(tokens: &[u32]) -> Option<(usize, usize, usize)> {
    let len = tokens.len();
    if len < 90 {
        return None;
    }
    for pattern_len in (15..=40).rev() {
        if len < pattern_len * 3 {
            continue;
        }
        let base = len - pattern_len * 3;
        let max_mismatches = (pattern_len / 12).max(1);
        let mut mis_a = 0usize;
        let mut mis_b = 0usize;
        for i in 0..pattern_len {
            if tokens[base + i] != tokens[base + pattern_len + i] {
                mis_a += 1;
            }
            if tokens[base + pattern_len + i] != tokens[base + 2 * pattern_len + i] {
                mis_b += 1;
            }
            if mis_a > max_mismatches && mis_b > max_mismatches {
                break;
            }
        }
        if mis_a <= max_mismatches && mis_b <= max_mismatches {
            return Some((pattern_len, mis_a, mis_b));
        }
    }
    None
}

#[cfg(test)]
mod fuzzy_repetition_tests {
    use super::detect_fuzzy_repetition;

    #[test]
    fn returns_none_below_minimum_output() {
        let tokens: Vec<u32> = (0..50).collect();
        assert_eq!(detect_fuzzy_repetition(&tokens), None);
    }

    #[test]
    fn returns_none_on_non_repeating_output() {
        // 120 distinct tokens — should never match a 15..=40 pattern-x3 window.
        let tokens: Vec<u32> = (0..120).collect();
        assert_eq!(detect_fuzzy_repetition(&tokens), None);
    }

    #[test]
    fn detects_exact_triple_repetition() {
        // 30 tokens of arbitrary prefix + three exact copies of a 20-token
        // pattern = 90 tokens total (minimum output length gate).
        let pattern: Vec<u32> = (1000..1020).collect();
        let mut tokens: Vec<u32> = (0..30).collect();
        for _ in 0..3 {
            tokens.extend_from_slice(&pattern);
        }
        let hit = detect_fuzzy_repetition(&tokens);
        assert!(
            matches!(hit, Some((20, 0, 0))),
            "expected exact 20-tok x3 detection, got {:?}",
            hit
        );
    }

    #[test]
    fn does_not_fire_on_double_boilerplate() {
        // Simulated agentic output where the model writes a function with two
        // similar signatures (2x near-match) then keeps going with novel code.
        // Under the old detector (pattern_len=20, max_mismatches=2), this
        // exact input triggered premature stops. Under the 3x rule it must not.
        let pattern: Vec<u32> = (500..520).collect();
        let mut tokens: Vec<u32> = (0..40).collect();
        tokens.extend_from_slice(&pattern);
        tokens.extend_from_slice(&pattern);
        // Follow up with ~60 tokens of novel code so we reach the min-length gate.
        for t in 600..660 {
            tokens.push(t);
        }
        assert_eq!(detect_fuzzy_repetition(&tokens), None);
    }

    #[test]
    fn detects_fuzzy_triple_within_tolerance() {
        // Three near-copies of a 24-token pattern, mutating one token in each
        // subsequent copy (1 mismatch per pair). 24/12 = 2, so 1 mismatch is
        // well within tolerance — must detect.
        let base: Vec<u32> = (2000..2024).collect();
        let mut copy_b = base.clone();
        copy_b[5] = 9999;
        let mut copy_c = copy_b.clone();
        copy_c[10] = 8888;
        let mut tokens: Vec<u32> = (0..20).collect();
        tokens.extend_from_slice(&base);
        tokens.extend_from_slice(&copy_b);
        tokens.extend_from_slice(&copy_c);
        let hit = detect_fuzzy_repetition(&tokens);
        assert!(
            hit.is_some(),
            "expected detection of near-identical triple, got None"
        );
    }

    #[test]
    fn does_not_fire_on_exact_boundary_double_match() {
        // Regression for alpha-2.43 false positive: 20-tok pattern x2 with 2
        // mismatches fired the old detector (max_mismatches = pattern_len/8 = 2
        // exact threshold). With 3x rule + /12 tolerance, the same input —
        // where only two copies exist — must not fire.
        let pattern: Vec<u32> = (3000..3020).collect();
        let mut mutated = pattern.clone();
        mutated[4] = 9001;
        mutated[12] = 9002;
        let mut tokens: Vec<u32> = (0..50).collect();
        tokens.extend_from_slice(&pattern);
        tokens.extend_from_slice(&mutated);
        // Append 60 tokens of novel output so overall length exceeds 90.
        for t in 4000..4060 {
            tokens.push(t);
        }
        assert_eq!(detect_fuzzy_repetition(&tokens), None);
    }
}
