// SPDX-License-Identifier: AGPL-3.0-only

//! The KNOWN_BAD controls of `native_fp8_ffn_gateup_fused_microtest`, and the
//! proof that each one MOVES the bytes the gate reads (#927, round-16 receipt
//! §2.2, Recommendation 3).
//!
//! # The defect this closes
//!
//! Round 16 ran the microtest on an H100 and it FAILED — `known-bad output was
//! admitted by the real oracle` — on a green kernel. The controls ran once, at
//! the first `M` of `ROWS` (= 5), and the second one zeroed the gate half's row
//! `MAX_M - 1` = row 15. At `M = 5` rows 5..16 are cuBLASLt PAD rows, which the
//! GEMM writes as zero on both arms, so zeroing row 15 changed nothing the
//! comparison reads, `equal_bytes` returned `Ok`, and the example panicked
//! before `M = 8`, `M = 16`, the two remaining controls, and every timing arm.
//! The lever's whole claim — fused µs against the pair's — went unmeasured for
//! the round.
//!
//! # The rule this file exists to enforce
//!
//! **A perturbation must move the graded quantity at every parameter the loop
//! runs at.** That is the same defect class as rounds 14/15's remnants control
//! (a `0.1 * rms` injection that could not move a `max_abs` which had grown
//! past it by `T = 4593`), and it is not visible in a diff of the control:
//! "zero row `MAX_M - 1`" reads as a perfectly good control until you know
//! which rows the GEMM filled.
//!
//! So the controls live here, as pure functions over host buffers, and are
//! graded two ways:
//!
//!   * ARMING, on device, at EVERY `M` the example walks — the caller requires
//!     a non-zero [`moved_bytes`] before it asks the oracle to refuse. A
//!     control that perturbs nothing now fails with the reason instead of with
//!     "the oracle admitted it";
//!   * ARMING, on the host, in the tests below, at `M ∈ {5, 8, 16}` over a
//!     synthetic buffer with the real pad-row structure. `cargo test -p
//!     spark-model --features cuda,gpu-examples --example
//!     native_fp8_ffn_gateup_fused_microtest` runs them with no GPU, so the
//!     round-16 failure is now reachable from a laptop.

/// Bytes per BF16 element, the layout of every buffer these controls perturb.
pub const BF16: usize = 2;

/// The four controls, in the order the example applies them.
pub const GATEUP_CONTROLS: [&str; 4] = ["one byte", "one row", "wrong half", "nonfinite"];

/// `spans[r]` = the byte (offset, length) of row `r`'s `width`-element slice
/// starting at column `col_off` of a `row_stride`-element row.
///
/// Shared with the example rather than restated there: the controls index the
/// SAME spans the oracle grades, and a second copy is how a control comes to
/// perturb a row the comparison does not read — which is the round-16 failure
/// in one sentence.
pub fn half_spans(
    rows: usize,
    row_stride: usize,
    col_off: usize,
    width: usize,
) -> Vec<(usize, usize)> {
    (0..rows)
        .map(|r| ((r * row_stride + col_off) * BF16, width * BF16))
        .collect()
}

/// Apply one KNOWN_BAD control to a copy of the fused output.
///
/// `m` is the LIVE row count of this rung — rows `m..` are cuBLASLt pad. Every
/// control lands inside `0..m`:
///
///   * `one byte` — flip one bit of gate row 0. The smallest defect the byte
///     gate claims to catch, and live at every `m >= 1`.
///   * `one row` — drop the LAST LIVE gate row, `m - 1`. Round 16 spelled this
///     `MAX_M - 1`, which is a pad row at every `m` below the pad width.
///   * `wrong half` — read the up half where the gate half belongs. The layout
///     error this gate exists to catch.
///   * `nonfinite` — a NaN in gate row 0, which `equal_bytes` refuses on its
///     own axis rather than by inequality.
pub fn apply_control(
    control: &str,
    bad: &mut [u8],
    m: usize,
    fused_gate: &[(usize, usize)],
    fused_up: &[(usize, usize)],
) {
    assert!(
        m >= 1 && m <= fused_gate.len(),
        "M={m} is outside the {} rows the spans cover",
        fused_gate.len()
    );
    match control {
        "one byte" => bad[fused_gate[0].0 + 2] ^= 1,
        "one row" => {
            let (o, len) = fused_gate[m - 1];
            bad[o..o + len].fill(0);
        }
        "wrong half" => {
            let (g, len) = fused_gate[0];
            let (u, _) = fused_up[0];
            bad.copy_within(u..u + len, g);
        }
        "nonfinite" => {
            bad[fused_gate[0].0..fused_gate[0].0 + 2].copy_from_slice(&0x7fc0_u16.to_le_bytes())
        }
        other => panic!("unknown KNOWN_BAD control {other:?}"),
    }
}

/// How many of the bytes the oracle GRADES the perturbation actually changed.
///
/// The arming half. `spans` must be the spans the comparison reads, not the
/// whole buffer: round 16's control did change a byte of the buffer — it wrote
/// zeros over zeros — and the buffer-wide question would have answered "yes"
/// while the graded question answered "no".
pub fn moved_bytes(bad: &[u8], clean: &[u8], spans: &[(usize, usize)]) -> usize {
    spans
        .iter()
        .map(|&(o, len)| {
            bad[o..o + len]
                .iter()
                .zip(&clean[o..o + len])
                .filter(|(a, b)| a != b)
                .count()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A narrow stand-in for the real `MAX_M = 16`, `INTER = 17408` geometry:
    /// same pad-row structure, four columns per half instead of 17 408.
    const ROWS: usize = 16;
    const INTER: usize = 4;

    /// `[gate | up]` over `ROWS` rows, with live rows `0..m` distinct and
    /// non-zero and pad rows `m..ROWS` all ZERO — which is what cuBLASLt
    /// leaves and what made round 16's control inert.
    fn fused(m: usize) -> Vec<u8> {
        let mut buf = vec![0_u8; ROWS * 2 * INTER * BF16];
        for r in 0..m {
            for c in 0..2 * INTER {
                // Distinct per (row, column) and never 0x0000 or 0x7fc0, so a
                // "moved" byte is the control's and nothing else's. The stride
                // is 0x0141 rather than 1 so consecutive elements differ in
                // BOTH bytes: with a stride of 1 only the low byte moves, and
                // "the whole row moved" would be half-true in a way the byte
                // counts below would quietly accept.
                let v = 0x3c01_u16.wrapping_add(((r * 2 * INTER + c) as u16).wrapping_mul(0x0141));
                let at = (r * 2 * INTER + c) * BF16;
                buf[at..at + BF16].copy_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    fn spans() -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
        (
            half_spans(ROWS, 2 * INTER, 0, INTER),
            half_spans(ROWS, 2 * INTER, INTER, INTER),
        )
    }

    /// THE regression. Every control must move a graded byte at every `M` the
    /// example walks — this is the assertion whose absence cost round 16 the
    /// gate+up timing arms.
    #[test]
    fn every_control_moves_a_graded_byte_at_every_m() {
        let (gate, up) = spans();
        for m in [5, 8, 16] {
            let clean = fused(m);
            for control in GATEUP_CONTROLS {
                let mut bad = clean.clone();
                apply_control(control, &mut bad, m, &gate, &up);
                assert!(
                    moved_bytes(&bad, &clean, &gate) > 0,
                    "KNOWN_BAD {control:?} perturbed nothing the gate half grades at M={m}",
                );
            }
        }
    }

    /// Round 16's spelling, kept executable so the defect cannot come back
    /// unseen: zeroing `MAX_M - 1` is a NO-OP at every `M` below the pad width
    /// and only works by accident at `M = MAX_M`.
    #[test]
    fn the_round_sixteen_spelling_is_inert_below_the_pad_width() {
        let (gate, _) = spans();
        for (m, expect_moved) in [(5, false), (8, false), (16, true)] {
            let clean = fused(m);
            let mut bad = clean.clone();
            let (o, len) = gate[ROWS - 1];
            bad[o..o + len].fill(0);
            assert_eq!(
                moved_bytes(&bad, &clean, &gate) > 0,
                expect_moved,
                "zeroing gate row {} at M={m}",
                ROWS - 1
            );
        }
    }

    /// The fix: the live row the control now targets is the LAST one the GEMM
    /// filled, so the whole row moves at every rung.
    #[test]
    fn the_live_row_control_drops_exactly_one_full_row() {
        let (gate, up) = spans();
        for m in [5, 8, 16] {
            let clean = fused(m);
            let mut bad = clean.clone();
            apply_control("one row", &mut bad, m, &gate, &up);
            assert_eq!(
                moved_bytes(&bad, &clean, &gate),
                INTER * BF16,
                "M={m}: the control must drop row {} and nothing else",
                m - 1
            );
        }
    }

    /// `wrong half` must not be satisfiable by the two halves happening to
    /// agree, and `one byte` must move exactly one byte.
    #[test]
    fn the_remaining_controls_land_where_their_names_say() {
        let (gate, up) = spans();
        let clean = fused(5);
        let mut bad = clean.clone();
        apply_control("one byte", &mut bad, 5, &gate, &up);
        assert_eq!(moved_bytes(&bad, &clean, &gate), 1);

        let mut bad = clean.clone();
        apply_control("wrong half", &mut bad, 5, &gate, &up);
        assert_eq!(
            moved_bytes(&bad, &clean, &gate),
            INTER * BF16,
            "the up half must overwrite the whole of gate row 0"
        );
        // …and it read the UP half, not some other gate row.
        let (g, len) = gate[0];
        let (u, _) = up[0];
        assert_eq!(&bad[g..g + len], &clean[u..u + len]);

        let mut bad = clean.clone();
        apply_control("nonfinite", &mut bad, 5, &gate, &up);
        assert_eq!(moved_bytes(&bad, &clean, &gate), 2);
    }

    #[test]
    #[should_panic(expected = "unknown KNOWN_BAD control")]
    fn an_unnamed_control_is_a_bug_not_a_silent_fallthrough() {
        let (gate, up) = spans();
        apply_control("typo", &mut fused(5), 5, &gate, &up);
    }
}
