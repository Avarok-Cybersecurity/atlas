// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the pure dispatch decisions every [`W4Linear`] consumer
//! shares: which call a row count takes (decode or prefill), when a gate+up
//! (or q/k/v) pair goes through `decode_pair`, how a long prefill is cut into
//! chunks the scratch can hold, and the shape checks that turn a mis-built
//! weight into a startup-time message instead of a wrong answer.
//!
//! Nothing here touches the GPU, so every rule is pinned by unit tests below.

use super::W4Linear;
use super::scratch::PREFILL_ROWS;

/// The call one projection takes for a given row count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjCall {
    /// `W4Linear::decode` with all rows at once (row-invariant bits).
    Decode,
    /// `W4Linear::prefill`, chunked by [`prefill_chunks`]; a decline falls
    /// back to the BF16 dequant GEMM.
    Prefill,
}

/// `rows <= max_decode_rows` decodes; anything wider is a prefill. `rows == 0`
/// is the caller's no-op and maps to `Decode` (no call is made for it).
pub fn proj_call(rows: usize, max_decode_rows: usize) -> ProjCall {
    if rows <= max_decode_rows {
        ProjCall::Decode
    } else {
        ProjCall::Prefill
    }
}

/// How two projections that share one input are dispatched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairCall {
    /// One `decode_pair` on the first weight (both fit the decode window).
    Pair,
    /// Each projection on its own [`proj_call`].
    Separate,
}

/// The pair goes through `decode_pair` exactly when BOTH members decode the
/// row count, so a fused pair never changes which route a row takes.
pub fn pair_call(rows: usize, a_max_rows: usize, b_max_rows: usize) -> PairCall {
    if rows <= a_max_rows.min(b_max_rows) {
        PairCall::Pair
    } else {
        PairCall::Separate
    }
}

/// Largest prefill chunk for a weight with this decode window: at least
/// [`PREFILL_ROWS`], and wide enough that an even split of any prefill keeps
/// every chunk above the decode window (see [`prefill_chunks`]).
pub fn prefill_chunk_cap(max_decode_rows: usize) -> usize {
    PREFILL_ROWS.max(4 * (max_decode_rows + 1))
}

/// Cut a prefill of `m` rows (`m > max_decode_rows`) into contiguous
/// `(first_row, rows)` chunks of at most `cap` rows, split as evenly as
/// possible, so that EVERY chunk is wider than `max_decode_rows` and the
/// prefill contract (`m > max_decode_rows()`) holds for each call. `cap`
/// comes from [`prefill_chunk_cap`]; a smaller `cap` is raised to it.
pub fn prefill_chunks(m: usize, cap: usize, max_decode_rows: usize) -> Vec<(usize, usize)> {
    if m == 0 {
        return Vec::new();
    }
    let cap = cap.max(prefill_chunk_cap(max_decode_rows));
    let n = m.div_ceil(cap);
    let (base, rem) = (m / n, m % n);
    let mut out = Vec::with_capacity(n);
    let mut first = 0;
    for i in 0..n {
        let rows = base + usize::from(i < rem);
        out.push((first, rows));
        first += rows;
    }
    out
}

/// Check one weight against the shape its call site multiplies by.
pub fn check_shape(w: &dyn W4Linear, n: usize, k: usize, what: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        w.n() == n && w.k() == k,
        "{what}: the installed {} weight is [{}, {}], the layer needs [{n}, {k}]",
        w.format(),
        w.n(),
        w.k()
    );
    anyhow::ensure!(
        w.max_decode_rows() >= 1,
        "{what}: the installed {} weight reports max_decode_rows 0",
        w.format()
    );
    Ok(())
}

/// The three SwiGLU projections against the model's `hidden` and
/// `intermediate` sizes: gate and up `[inter, hidden]`, down `[hidden, inter]`.
pub fn check_ffn(
    gate: &dyn W4Linear,
    up: &dyn W4Linear,
    down: &dyn W4Linear,
    hidden: usize,
    inter: usize,
) -> anyhow::Result<()> {
    check_shape(gate, inter, hidden, "dense FFN gate_proj")?;
    check_shape(up, inter, hidden, "dense FFN up_proj")?;
    check_shape(down, hidden, inter, "dense FFN down_proj")
}

/// The three SwiGLU projections against each other only (no model sizes):
/// what can be checked when the weights are installed.
pub fn ffn_consistent(gate: &dyn W4Linear, up: &dyn W4Linear, down: &dyn W4Linear) -> bool {
    gate.n() == up.n() && gate.k() == up.k() && down.k() == gate.n() && down.n() == gate.k()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::port::w4::fake::{FakeW4, log};

    #[test]
    fn decode_window_is_inclusive() {
        assert_eq!(proj_call(1, 16), ProjCall::Decode);
        assert_eq!(proj_call(16, 16), ProjCall::Decode);
        assert_eq!(proj_call(17, 16), ProjCall::Prefill);
        assert_eq!(proj_call(8, 8), ProjCall::Decode);
        assert_eq!(proj_call(9, 8), ProjCall::Prefill);
        assert_eq!(proj_call(64, 64), ProjCall::Decode);
    }

    #[test]
    fn a_pair_fuses_only_when_both_members_decode() {
        assert_eq!(pair_call(1, 16, 16), PairCall::Pair);
        assert_eq!(pair_call(16, 16, 16), PairCall::Pair);
        assert_eq!(pair_call(9, 16, 8), PairCall::Separate);
        assert_eq!(pair_call(8, 16, 8), PairCall::Pair);
        assert_eq!(pair_call(17, 16, 64), PairCall::Separate);
    }

    #[test]
    fn chunk_cap_never_drops_below_the_scratch_rows() {
        assert_eq!(prefill_chunk_cap(16), PREFILL_ROWS);
        assert_eq!(prefill_chunk_cap(64), PREFILL_ROWS);
        assert_eq!(prefill_chunk_cap(1000), 4004);
    }

    /// Every chunk is contiguous, inside the cap, wider than the decode
    /// window, and the chunks cover `m` exactly. Checked over every `m` from
    /// just above the window to past three caps, for three windows.
    #[test]
    fn chunks_cover_m_and_every_chunk_is_a_valid_prefill() {
        for max_rows in [8usize, 16, 64] {
            let cap = prefill_chunk_cap(max_rows);
            for m in (max_rows + 1)..(3 * cap + 7) {
                let chunks = prefill_chunks(m, cap, max_rows);
                let mut next = 0;
                for &(first, rows) in &chunks {
                    assert_eq!(first, next, "m={m}: chunks must be contiguous");
                    assert!(rows <= cap, "m={m}: chunk {rows} over cap {cap}");
                    assert!(
                        rows > max_rows,
                        "m={m}: chunk {rows} would be a decode call"
                    );
                    next += rows;
                }
                assert_eq!(next, m, "m={m}: chunks must cover every row");
                assert_eq!(chunks.len(), m.div_ceil(cap), "m={m}: fewest chunks");
            }
        }
    }

    #[test]
    fn chunks_are_even() {
        let cap = prefill_chunk_cap(16);
        assert_eq!(prefill_chunks(2048, cap, 16), vec![(0, 2048)]);
        assert_eq!(prefill_chunks(2049, cap, 16), vec![(0, 1025), (1025, 1024)]);
        assert_eq!(
            prefill_chunks(5000, cap, 16),
            vec![(0, 1667), (1667, 1667), (3334, 1666)]
        );
        assert!(prefill_chunks(0, cap, 16).is_empty());
    }

    #[test]
    fn a_small_cap_is_raised() {
        assert_eq!(prefill_chunks(100, 10, 16), vec![(0, 100)]);
    }

    #[test]
    fn shape_checks_name_the_projection_and_both_shapes() {
        let l = log();
        let gate = FakeW4::new("gate", 96, 32, 16, &l);
        let up = FakeW4::new("up", 96, 32, 16, &l);
        let down = FakeW4::new("down", 32, 96, 16, &l);
        check_ffn(&gate, &up, &down, 32, 96).unwrap();
        assert!(ffn_consistent(&gate, &up, &down));

        let bad_up = FakeW4::new("up", 96, 48, 16, &l);
        let e = check_ffn(&gate, &bad_up, &down, 32, 96)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("up_proj") && e.contains("[96, 48]") && e.contains("[96, 32]"),
            "{e}"
        );
        assert!(!ffn_consistent(&gate, &bad_up, &down));

        let e = check_ffn(&gate, &up, &down, 64, 96)
            .unwrap_err()
            .to_string();
        assert!(e.contains("gate_proj"), "{e}");

        let zero = FakeW4::new("gate", 96, 32, 0, &l);
        let e = check_ffn(&zero, &up, &down, 32, 96)
            .unwrap_err()
            .to_string();
        assert!(e.contains("max_decode_rows 0"), "{e}");
    }
}
