// SPDX-License-Identifier: AGPL-3.0-only

//! Drain-tail CUDA-graph borrowing: replay a WIDER captured graph instead of
//! capturing one per shrinking batch composition.
//!
//! THE MEASURED PROBLEM (dgx2, Qwen3.8-27B NVFP4, C=32 rung, d92fc2488): as a
//! C=32 wave drains 32→1, every composition the drain visits is a NEW
//! slot-vector key — the ramp admits the whole wave at once and never sees
//! n=31,30,…, so the drain re-captured 29 graphs (~7.5 s of capture work,
//! ~2.9 s of wall beyond the median finisher ≈ 2% of the rung).
//!
//! WHY THE EXISTING CACHES CANNOT HIT: both the batched-decode cache
//! (`decode_graph_key.rs`) and the batched-verify cache (`verify_e2.rs`) key
//! on the per-row SSM slot VECTOR, because a capture bakes each row's pool
//! state addresses. `padded_batch_n` buckets the WIDTH, but a drain step's
//! slot vector still differs from every captured one, so `padded_n` bucketing
//! alone cannot reuse anything.
//!
//! THE BORROW: `retire_finished_sequences` compacts survivors into contiguous
//! slots `[0..n)` and both dispatch paths sort by slot
//! (`decode_step.rs` / `mtp_step.rs`), so a drain batch's slot vector is a
//! PREFIX of the steady-state canonical vector. A captured graph whose key
//! starts with this batch's slots can therefore be replayed as-is: the active
//! rows sit at exactly the rows the scheduler reads (logits row `i`, verify
//! rows `off[i]..off[i]+k`), and the tail rows become padding. Tail rows are
//! safe if and only if every tail-baked slot is the dummy slot or a
//! CURRENTLY-FREE pool slot: pad lanes write garbage into those slots' pool
//! state, which is legal because `alloc_sequence_dispatch` zeroes h/conv
//! state on claim (and MTP intermediates/checkpoints are per-verify-step
//! scratch — `SsmStatePool::copy_slot` documents that nothing in them
//! survives a step). A slot claimed by a mid-prefill sequence VETOES the
//! borrow — the freeness check runs on the scheduler thread, which is the
//! only thread that claims or releases slots, so the answer is stable for
//! the duration of the dispatch.
//!
//! DETERMINISM: padding rows already exist on the decode path (every
//! `n < padded_n` batch computes dummy-slot pad rows today); borrowing only
//! widens the pad tail. All batched kernels on these paths are row-
//! independent (per-row attention/GDN lanes, per-row GEMM reductions,
//! per-token MoE routing), so active lanes' outputs do not depend on pad
//! lane contents. The throughput arbiter stays honest by construction: the
//! scheduler charges `record_decode(wall, active.len())` — the model layer
//! hides the padded width entirely, so timings are attributed to the ACTIVE
//! width (see the pin in `mtp_gate/tests.rs`).
//!
//! POLICY: borrow only graphs at most [`borrow_width_cap`] = 2 × the batch's
//! POWER-OF-TWO width bucket (one `mtp_gate` width regime above its own).
//! Wider than that, wasted pad lanes cost more over a drain segment than one
//! narrow capture, and the narrow capture is canonical — it is reused by
//! every later drain. With the window, a full 32→1 drain captures O(1)
//! graphs per family instead of ~29 (only the ladder's `k_drafts`
//! deepening below n≈13 still forces new verify shapes).
//!
//! Kill switch: `ATLAS_NO_GRAPH_BORROW` (PRESENCE disables, house
//! convention — `=0` is NOT off). Read once per process.

/// Borrowing enabled unless `ATLAS_NO_GRAPH_BORROW` is present.
pub(super) fn graph_borrow_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_NO_GRAPH_BORROW").is_none())
}

/// Widest captured graph an `n`-active batch may borrow: twice its
/// POWER-OF-TWO width bucket (`n.next_power_of_two()` — the same bucket the
/// `mtp_gate` width-regime arbiter uses), i.e. one regime above the batch's
/// own. Within the window the step stays bandwidth-bound-cheap; beyond it a
/// fresh narrow capture amortizes better over the drain segment (and seeds
/// the canonical key for later drains).
///
/// This was `2 * padded_batch_n(n)` in the first cut, which the 2026-08-16
/// dgx2 validation run (`/tmp/atlas-dtval-serve.log`) caught declining the
/// clean n=12-under-n=32 borrow: 12 is itself a padding-ladder rung, so its
/// padded bucket is 12 and the cap (24) rejected the only cached wider key
/// (32) — the engine then CAPTURED a fresh n=12 verify graph at 21:04:56
/// mid-drain. The power-of-two bucket (16) puts 32 inside the window, which
/// is what the drain needs: below n=13 the ladder deepens `k_drafts`
/// anyway, so n=9..12 is the LAST band the steady-state k=2 graph can
/// serve — declining there buys a capture and saves nothing later.
pub(super) fn borrow_width_cap(n: usize) -> usize {
    2 * n.next_power_of_two()
}

/// Dedup gate for borrow INFO logs. A drain band replays the same borrowed
/// graph for hundreds of consecutive steps; logging each replay would flood
/// the serve log, logging at `debug!` proved NOTHING at the production
/// `info` level (the 2026-08-16 validation could not tell "borrow engaged
/// silently" from "borrow never ran"). Log exactly the TRANSITIONS: one
/// line whenever the (exact key, borrowed key) pair differs from the
/// previous borrow — the same cardinality as the captures the borrow
/// replaced.
pub(super) struct BorrowLogGate(std::sync::atomic::AtomicU64);

impl BorrowLogGate {
    pub(super) const fn new() -> Self {
        Self(std::sync::atomic::AtomicU64::new(0))
    }

    /// True when this (exact, borrowed) pair is not the one last logged.
    /// FNV-1a over both keys; 0 is reserved for "nothing logged yet".
    pub(super) fn should_log(&self, exact: &[u32], borrowed: &[u32]) -> bool {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        // Lengths preserve the pair boundary. Without them, ([1], [2, 3])
        // and ([1, 2], [3]) hash the same concatenated values and a real
        // transition is suppressed.
        for v in std::iter::once(exact.len() as u64)
            .chain(exact.iter().map(|&v| u64::from(v)))
            .chain(std::iter::once(borrowed.len() as u64))
            .chain(borrowed.iter().map(|&v| u64::from(v)))
        {
            h = (h ^ v).wrapping_mul(0x0000_0100_0000_01b3);
        }
        let h = if h == 0 { 1 } else { h };
        self.0.swap(h, std::sync::atomic::Ordering::Relaxed) != h
    }
}

pub(super) static DECODE_BORROW_LOG: BorrowLogGate = BorrowLogGate::new();
pub(super) static VERIFY_BORROW_LOG: BorrowLogGate = BorrowLogGate::new();

/// Find a captured batched-decode key to replay for `active_slots`
/// (this batch's per-row SSM slots, batch order).
///
/// A candidate key `K` (row-slot vector, length = its captured width) is
/// borrowable iff:
///   * `n < K.len() <= borrow_width_cap(n)` — strictly wider than the active
///     batch (a same-length match IS the exact key) and inside the factor-2
///     window;
///   * `K[..n] == active_slots` — active rows land on rows baked with their
///     own slots, in order;
///   * every tail entry `K[n..]` satisfies `tail_ok` (caller passes "is the
///     dummy slot, or a currently-free pool slot").
///
/// Returns the NARROWEST borrowable key (fewest wasted pad lanes). Pure —
/// the cache scan is at most `batch_decode_graph_cap` (≤ 80) keys.
pub(super) fn find_borrowable_decode_key<'a>(
    active_slots: &[u32],
    keys: impl Iterator<Item = &'a Vec<u32>>,
    tail_ok: impl Fn(u32) -> bool,
) -> Option<Vec<u32>> {
    let n = active_slots.len();
    if n < 2 {
        return None;
    }
    let cap = borrow_width_cap(n);
    let mut best: Option<&'a Vec<u32>> = None;
    for k in keys {
        let m = k.len();
        if m <= n || m > cap {
            continue;
        }
        if best.is_some_and(|b| b.len() <= m) {
            continue;
        }
        if k[..n] != *active_slots {
            continue;
        }
        if k[n..].iter().all(|&s| tail_ok(s)) {
            best = Some(k);
        }
    }
    best.cloned()
}

/// A borrowable batched-verify graph: the cached key to replay plus the
/// ghost `(slot, k)` tail — the rows baked beyond the active batch, which
/// the caller must feed with pad metadata, pad embeds and synthesized WY
/// table entries.
pub(super) struct VerifyBorrow {
    pub(super) key: Vec<u32>,
    pub(super) ghosts: Vec<(u32, u32)>,
}

/// Find a captured batched-verify key to replay for this batch's exact key
/// (`verify_e2::verify_batched_graph_key` layout: `n` interleaved
/// `(slot, k)` pairs then one sentinel).
///
/// A candidate `K` with `m` pairs is borrowable iff:
///   * `n < m <= borrow_width_cap(n)`;
///   * the sentinels match (a table-less capture never replays a table-full
///     step or vice versa — same rule as the exact key);
///   * `K`'s first `n` pairs equal the active pairs — same slots AND the
///     same per-row depths, so `off[i]` (the scheduler's logits/stash row
///     base) is identical for every active sequence;
///   * every tail pair satisfies `tail_ok(slot, k)` (caller passes "slot is
///     currently free AND its tiered intermediate pool covers depth k").
///
/// Returns the candidate with the fewest ghost ROWS (Σ tail k).
pub(super) fn find_borrowable_verify_key<'a>(
    exact_key: &[u32],
    keys: impl Iterator<Item = &'a Vec<u32>>,
    tail_ok: impl Fn(u32, u32) -> bool,
) -> Option<VerifyBorrow> {
    // Interleaved pairs + sentinel: an exact key is always odd-length ≥ 5
    // (n ≥ 2 pairs). Anything else is not this cache's key shape.
    if exact_key.len() < 5 || exact_key.len().is_multiple_of(2) {
        return None;
    }
    let n = (exact_key.len() - 1) / 2;
    let sentinel = exact_key[exact_key.len() - 1];
    let prefix = &exact_key[..2 * n];
    let cap = borrow_width_cap(n);
    let mut best: Option<(&'a Vec<u32>, usize)> = None;
    for k in keys {
        if k.len() < 5 || k.len().is_multiple_of(2) {
            continue;
        }
        let m = (k.len() - 1) / 2;
        if m <= n || m > cap || k[k.len() - 1] != sentinel {
            continue;
        }
        if k[..2 * n] != *prefix {
            continue;
        }
        let tail = &k[2 * n..k.len() - 1];
        if !tail.chunks_exact(2).all(|p| tail_ok(p[0], p[1])) {
            continue;
        }
        let ghost_rows: usize = tail.chunks_exact(2).map(|p| p[1] as usize).sum();
        if best.is_none_or(|(_, r)| ghost_rows < r) {
            best = Some((k, ghost_rows));
        }
    }
    best.map(|(k, _)| VerifyBorrow {
        key: k.clone(),
        ghosts: k[2 * n..k.len() - 1]
            .chunks_exact(2)
            .map(|p| (p[0], p[1]))
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The width→bucket policy, pinned: a batch may borrow up to 2× its
    /// POWER-OF-TWO width bucket (one arbiter width-regime above its own).
    /// NOT the padding ladder: `2 * padded_batch_n(12) = 24` rejected the
    /// n=12-under-n=32 drain borrow on dgx2 (2026-08-16) because 12 is
    /// itself a ladder rung.
    #[test]
    fn borrow_width_cap_is_twice_the_power_of_two_bucket() {
        assert_eq!(borrow_width_cap(1), 2);
        assert_eq!(borrow_width_cap(2), 4);
        assert_eq!(borrow_width_cap(3), 8);
        assert_eq!(borrow_width_cap(8), 16);
        assert_eq!(borrow_width_cap(9), 32); // bucket 16 — NOT padded 12
        assert_eq!(borrow_width_cap(12), 32); // the dgx2 counter-example band
        assert_eq!(borrow_width_cap(13), 32);
        assert_eq!(borrow_width_cap(16), 32);
        assert_eq!(borrow_width_cap(17), 64); // bucket 32 (bs=64 boots)
        assert_eq!(borrow_width_cap(32), 64);
    }

    fn canonical(n: u32) -> Vec<u32> {
        (0..n).collect()
    }

    /// THE DRAIN SCENARIO: with the steady-state C=32 graph cached, every
    /// drain width inside the factor-2 window replays it — no re-capture.
    /// (Slots ≥ n are free during a drain: retirement compacts survivors
    /// into [0..n).)
    #[test]
    fn drain_widths_inside_the_bucket_replay_the_steady_state_graph() {
        let k32 = canonical(32);
        let cache = [k32.clone()];
        for n in 9..32usize {
            let active = canonical(n as u32);
            let found = find_borrowable_decode_key(&active, cache.iter(), |s| s >= n as u32);
            assert_eq!(
                found.as_ref(),
                Some(&k32),
                "drain width n={n} must replay the 32-wide graph, not capture"
            );
        }
    }

    /// Below the factor-2 window the borrow declines — one narrow capture is
    /// cheaper over the drain segment than 32-wide pad lanes at n≤8 (and
    /// the verify ladder deepens k there anyway, so the k=2 graph could not
    /// serve those widths for long).
    #[test]
    fn drain_widths_below_the_window_capture_a_narrow_canonical_graph() {
        let cache = [canonical(32)];
        for n in 2..=8usize {
            let active = canonical(n as u32);
            assert!(
                find_borrowable_decode_key(&active, cache.iter(), |s| s >= n as u32).is_none(),
                "n={n} is outside the 32-wide borrow window"
            );
        }
    }

    /// Once the first sub-window width captured (canonically, with dummy
    /// pads), the rest of its bucket borrows it — including keys whose tail
    /// mixes real free slots and the dummy slot.
    #[test]
    fn a_dummy_padded_canonical_key_serves_the_rest_of_its_bucket() {
        const DUMMY: u32 = 99;
        // Captured at n=11, padded to 12: rows [0..11) real, row 11 dummy.
        let mut k11 = canonical(11);
        k11.push(DUMMY);
        let cache = [k11.clone()];
        for n in 6..11usize {
            let active = canonical(n as u32);
            let found =
                find_borrowable_decode_key(&active, cache.iter(), |s| s == DUMMY || s >= n as u32);
            assert_eq!(
                found.as_ref(),
                Some(&k11),
                "n={n} must borrow the 12-row graph"
            );
        }
    }

    /// SAFETY VETO: a tail slot claimed by a mid-prefill sequence (not free)
    /// blocks the borrow — pad lanes would scribble on its SSM state.
    #[test]
    fn a_claimed_tail_slot_vetoes_the_borrow() {
        let cache = [canonical(32)];
        let active = canonical(20);
        // Slot 20 is claimed by a prefilling sequence; 21..32 free.
        let found = find_borrowable_decode_key(&active, cache.iter(), |s| s >= 21);
        assert!(found.is_none(), "claimed tail slot must veto the borrow");
    }

    /// A bootstrap SUBSET batch ({0,2,5}-style) or any permutation must not
    /// borrow: the baked rows would not match the active rows' slots.
    #[test]
    fn a_non_prefix_slot_vector_never_borrows() {
        let cache = [canonical(32)];
        assert!(find_borrowable_decode_key(&[0, 2, 5], cache.iter(), |_| true).is_none());
        assert!(find_borrowable_decode_key(&[1, 0, 2], cache.iter(), |_| true).is_none());
    }

    /// The narrowest borrowable graph wins (fewest wasted pad lanes).
    #[test]
    fn the_narrowest_candidate_is_preferred() {
        let cache = [canonical(32), canonical(24)];
        let active = canonical(17);
        let found = find_borrowable_decode_key(&active, cache.iter(), |_| true);
        assert_eq!(found, Some(canonical(24)));
    }

    /// A same-length key is the EXACT key's job, never a borrow; single-seq
    /// batches use the slot-keyed `decode()` cache instead.
    #[test]
    fn invalid_or_exact_width_keys_never_borrow() {
        let cache = [canonical(8)];
        assert!(find_borrowable_decode_key(&canonical(8), cache.iter(), |_| true).is_none());
        assert!(find_borrowable_decode_key(&[0], cache.iter(), |_| true).is_none());

        let valid = [vkey(&[(0, 2), (1, 2)], WY)];
        assert!(find_borrowable_verify_key(&[0, 2, 1, 2], valid.iter(), |_, _| true).is_none());
        assert!(find_borrowable_verify_key(&[0, 2, 1, 2, WY], valid.iter(), |_, _| true).is_none());
        let malformed_candidates = [vec![0, 2, 1, 2], vec![0, 2, 1, 2, 2, 2]];
        assert!(
            find_borrowable_verify_key(&[0, 2, 1, 2, WY], malformed_candidates.iter(), |_, _| true)
                .is_none()
        );
    }

    // ── batched-verify keys: interleaved (slot, k) pairs + sentinel ──

    fn vkey(pairs: &[(u32, u32)], sentinel: u32) -> Vec<u32> {
        let mut k: Vec<u32> = pairs.iter().flat_map(|&(s, d)| [s, d]).collect();
        k.push(sentinel);
        k
    }

    fn uniform(n: u32, k: u32, sentinel: u32) -> Vec<u32> {
        vkey(&(0..n).map(|s| (s, k)).collect::<Vec<_>>(), sentinel)
    }

    const WY: u32 = u32::MAX - 1; // tables-present sentinel

    /// THE DRAIN SCENARIO (verify): the steady-state n=32 k=2 graph serves
    /// every drain width in its window; ghosts carry the baked tail pairs.
    #[test]
    fn verify_drain_widths_replay_the_steady_state_graph_with_ghost_tails() {
        let k32 = uniform(32, 2, WY);
        let cache = [k32.clone()];
        for n in 9..32u32 {
            let exact = uniform(n, 2, WY);
            let found = find_borrowable_verify_key(&exact, cache.iter(), |s, _| s >= n)
                .unwrap_or_else(|| panic!("verify n={n} must borrow, not capture"));
            assert_eq!(found.key, k32);
            assert_eq!(
                found.ghosts,
                (n..32).map(|s| (s, 2)).collect::<Vec<_>>(),
                "ghost tail must be the baked (slot, k) pairs beyond the batch"
            );
        }
        let exact = uniform(8, 2, WY);
        assert!(
            find_borrowable_verify_key(&exact, cache.iter(), |s, _| s >= 8).is_none(),
            "n=8 is outside the 32-wide window"
        );
    }

    /// Borrow logging is transition-deduped: a drain band that replays one
    /// borrowed graph for hundreds of steps logs ONCE; every width change
    /// (new pair) logs again — the same cardinality as the captures the
    /// borrow replaced. Steady repeats of the same pair stay silent.
    #[test]
    fn borrow_log_gate_fires_once_per_transition() {
        let gate = BorrowLogGate::new();
        let k32 = canonical(32);
        let a = canonical(20);
        let b = canonical(19);
        assert!(gate.should_log(&a, &k32), "first borrow must log");
        assert!(!gate.should_log(&a, &k32), "same pair repeats silently");
        assert!(!gate.should_log(&a, &k32));
        assert!(gate.should_log(&b, &k32), "width change logs again");
        assert!(gate.should_log(&a, &k32), "returning to a prior pair logs");

        let boundary = BorrowLogGate::new();
        assert!(boundary.should_log(&[1], &[2, 3]));
        assert!(
            boundary.should_log(&[1, 2], &[3]),
            "moving the exact/borrowed boundary is a distinct transition"
        );
    }

    /// Depth is part of the row layout: a prefix whose ks differ must not
    /// borrow (off[i] would shift under the scheduler's row reads), and the
    /// sentinel must match exactly.
    #[test]
    fn verify_borrow_requires_matching_depths_and_sentinel() {
        let cache = [uniform(32, 2, WY)];
        let deeper = uniform(20, 3, WY);
        assert!(find_borrowable_verify_key(&deeper, cache.iter(), |_, _| true).is_none());
        let other_sentinel = uniform(20, 2, u32::MAX);
        assert!(find_borrowable_verify_key(&other_sentinel, cache.iter(), |_, _| true).is_none());
    }

    /// The tail check sees each ghost's DEPTH too — a slot whose tiered
    /// intermediate pool is too shallow for the baked k vetoes the borrow.
    #[test]
    fn verify_tail_check_receives_slot_and_depth() {
        let mut pairs: Vec<(u32, u32)> = (0..15).map(|s| (s, 2)).collect();
        pairs.push((15, 4)); // deep tail row
        let cache = [vkey(&pairs, WY)];
        let exact = uniform(15, 2, WY);
        // Tier check: slot 15 only covers k ≤ 3.
        let found = find_borrowable_verify_key(&exact, cache.iter(), |s, k| s >= 15 && k <= 3);
        assert!(
            found.is_none(),
            "shallow-tier tail slot must veto the borrow"
        );
        let found = find_borrowable_verify_key(&exact, cache.iter(), |s, k| s >= 15 && k <= 4);
        assert!(found.is_some());
    }

    /// Fewest ghost ROWS wins (Σ tail k, not pair count).
    #[test]
    fn verify_prefers_the_fewest_ghost_rows() {
        let mut more_pairs: Vec<(u32, u32)> = (0..16).map(|s| (s, 2)).collect();
        more_pairs.extend((16..24).map(|s| (s, 1))); // 8 pairs, 8 ghost rows
        let a = vkey(&more_pairs, WY);
        let mut fewer_pairs: Vec<(u32, u32)> = (0..16).map(|s| (s, 2)).collect();
        fewer_pairs.extend((16..19).map(|s| (s, 4))); // 3 pairs, 12 ghost rows
        let b = vkey(&fewer_pairs, WY);
        let cache = [b, a.clone()];
        let exact = uniform(16, 2, WY);
        let found = find_borrowable_verify_key(&exact, cache.iter(), |_, _| true).unwrap();
        assert_eq!(found.key, a);
        assert_eq!(found.ghosts.len(), 8, "pair count is not the cost metric");
        assert_eq!(found.ghosts.iter().map(|&(_, k)| k).sum::<u32>(), 8);
    }
}
