// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side contract tests for the grouped-GEMM prefill path.
//!
//! Split out of `forward_prefill_gemm.rs` to keep that file under the 500-LoC cap.
//! Nothing here touches a GPU: the numeric gate is
//! `examples/glm5next_moe_grouped_prefill_microtest.rs`. What IS covered here is the
//! routing algebra the grouped kernels read — the counting sort's contract and the grid
//! height — because both fail SILENTLY on the device (a dropped slot or a truncated
//! expert produces a well-formed wrong answer, never an error).

use super::*;

/// Host reference for `moe_sort_by_expert`: the contract the grouped path depends on.
/// Returns `(sorted_token_ids, sorted_expert_ids, expert_offsets, token_to_perm)`.
fn sort_ref(
    ids: &[u32],
    num_experts: usize,
    topk: usize,
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let te = ids.len();
    let mut counts = vec![0usize; num_experts];
    for &e in ids {
        counts[e as usize] += 1;
    }
    let mut offsets = vec![0i32; num_experts + 1];
    for e in 0..num_experts {
        offsets[e + 1] = offsets[e] + counts[e] as i32;
    }
    // The device kernel's Phase 4 uses `atomicAdd` per expert, so WITHIN an expert the
    // order is unspecified. This reference takes the in-order placement; every property
    // asserted below is order-independent within an expert.
    let mut cursor: Vec<i32> = offsets[..num_experts].to_vec();
    let mut stid = vec![-1i32; te];
    let mut seid = vec![-1i32; te];
    let mut t2p = vec![-1i32; te];
    for (i, &e) in ids.iter().enumerate() {
        let pos = cursor[e as usize];
        cursor[e as usize] += 1;
        stid[pos as usize] = (i / topk) as i32;
        seid[pos as usize] = e as i32;
        t2p[i] = pos;
    }
    (stid, seid, offsets, t2p)
}

fn routing(rows: usize, topk: usize, num_experts: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    let mut out = Vec::with_capacity(rows * topk);
    for _ in 0..rows {
        // top-k is a SET per row — no row may pick the same expert twice, or the
        // combine would double-count it.
        let mut picked: Vec<u32> = Vec::with_capacity(topk);
        while picked.len() < topk {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let e = ((s >> 33) as usize % num_experts) as u32;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        out.extend(picked);
    }
    out
}

/// The four invariants the grouped GEMM reads the sort through. Any one of them
/// breaking is a silently-wrong answer, not a crash: the GEMM would sweep the wrong
/// expert for a row, or `combine_indexed` would fetch another token's slot.
#[test]
fn sort_contract_holds_at_glm_shapes() {
    for &(rows, topk, ne) in &[(16usize, 8usize, 288usize), (256, 8, 288), (5, 4, 7)] {
        let ids = routing(rows, topk, ne, 0xC0FFEE ^ rows as u64);
        let (stid, seid, offsets, t2p) = sort_ref(&ids, ne, topk);
        let te = rows * topk;

        // (a) offsets is a prefix sum over exactly `te` slots.
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[ne], te as i32);
        for e in 0..ne {
            assert!(offsets[e + 1] >= offsets[e], "offsets not monotone at {e}");
        }
        // (b) every sorted position lies inside its expert's half-open range.
        for e in 0..ne {
            for p in offsets[e]..offsets[e + 1] {
                assert_eq!(seid[p as usize], e as i32, "expert block {e} is ragged");
            }
        }
        // (c) token_to_perm is a bijection onto [0, te).
        let mut seen = vec![false; te];
        for &p in &t2p {
            assert!(p >= 0 && (p as usize) < te, "perm {p} out of range");
            assert!(!seen[p as usize], "perm {p} claimed twice");
            seen[p as usize] = true;
        }
        // (d) the round trip the kernels actually make: the sorted row a slot maps to
        //     must carry that slot's TOKEN and that slot's EXPERT.
        for (i, &e) in ids.iter().enumerate() {
            let p = t2p[i] as usize;
            assert_eq!(stid[p], (i / topk) as i32, "slot {i} lost its token");
            assert_eq!(seid[p], e as i32, "slot {i} lost its expert");
        }
    }
}

/// The grid-height bound. Too small SILENTLY TRUNCATES an expert's rows (the kernel
/// has no way to report it); too large is the 97 %-empty launch this exists to avoid.
#[test]
fn max_m_tiles_covers_the_busiest_expert_and_never_exceeds_the_worst_case() {
    // 288 experts, 2048 slots, perfectly balanced at 7.1 → one 64-row tile.
    let balanced: Vec<i32> = (0..=288i32).map(|e| e * 2048 / 288).collect();
    assert_eq!(max_m_tiles_from_offsets(&balanced, 32, GROUPED_M_TILE), 1);

    // One expert takes everything: ceil(2048/64) = 32 tiles, exactly the worst case.
    let mut skewed = vec![0i32; 289];
    for o in skewed.iter_mut().skip(1) {
        *o = 2048;
    }
    assert_eq!(max_m_tiles_from_offsets(&skewed, 32, GROUPED_M_TILE), 32);

    // 65 rows on one expert needs TWO tiles — the off-by-one that would drop row 64.
    let mut sixty_five = vec![0i32; 289];
    for (e, o) in sixty_five.iter_mut().enumerate() {
        *o = if e == 0 { 0 } else { 65 };
    }
    assert_eq!(max_m_tiles_from_offsets(&sixty_five, 32, GROUPED_M_TILE), 2);

    // Empty routing still launches one tile — every CTA early-exits on M_expert <= 0.
    assert_eq!(max_m_tiles_from_offsets(&[0i32; 289], 1, GROUPED_M_TILE), 1);
}

/// Cross-check against the live routing shape: whatever the router picks, the bound
/// derived from the real offsets is never below what the busiest expert needs.
#[test]
fn max_m_tiles_is_never_short_for_a_real_routing() {
    for seed in 0..8u64 {
        let (rows, topk, ne) = (256usize, 8usize, 288usize);
        let ids = routing(rows, topk, ne, seed);
        let (_, _, offsets, _) = sort_ref(&ids, ne, topk);
        let worst = (rows * topk).div_ceil(GROUPED_M_TILE) as u32;
        let tiles = max_m_tiles_from_offsets(&offsets, worst, GROUPED_M_TILE);
        let busiest = (0..ne)
            .map(|e| offsets[e + 1] - offsets[e])
            .max()
            .unwrap_or(0) as u32;
        assert!(
            tiles * GROUPED_M_TILE as u32 >= busiest,
            "seed {seed}: {tiles} tiles cover {} rows, busiest expert has {busiest}",
            tiles * GROUPED_M_TILE as u32
        );
        assert!(
            tiles <= worst,
            "seed {seed}: {tiles} exceeds worst case {worst}"
        );
    }
}

/// The tile table is the dispatch's only source of grid geometry, so a wrong row is a
/// silently-wrong answer on the device. Pin every field against the kernel file.
#[test]
fn every_gemm_tile_matches_its_kernel_geometry() {
    for t in GEMM_TILES {
        assert!(
            t.name.starts_with("moe_w4a16_grouped_gemm_ptrtable"),
            "{} is not an entry point of moe_w4a16_grouped_gemm.cu",
            t.name
        );
        // SPLIT_N kernels are the `m16` family; everything else keeps the 64-row tile.
        let expect_m16 = t.name.contains("m16");
        assert_eq!(
            t.m_tile,
            if expect_m16 { 16 } else { 64 },
            "{}: m_tile must equal the kernel's M_TILE or rows are dropped",
            t.name
        );
        // An N_TILE of 128 is built with eight warps; 64 with four.
        let expect_n128 = t.name.contains("n128");
        assert_eq!(
            t.n_tile,
            if expect_n128 { 128 } else { 64 },
            "{}: n_tile must equal the kernel's NTILE or the output is half-written",
            t.name
        );
        assert_eq!(
            t.threads,
            if expect_n128 { 256 } else { 128 },
            "{}: block width must equal WARPS*32 or the cooperative load is short",
            t.name
        );
    }
}

/// 🪤 A typo in the env must not silently become the control arm.
#[test]
fn tile_selection_is_exact_and_rejects_unknown_names() {
    assert_eq!(select_gemm_tile("base").unwrap().name, GEMM_TILES[0].name);
    assert_eq!(select_gemm_tile("").unwrap().name, GEMM_TILES[0].name);
    assert_eq!(
        select_gemm_tile("bt_m16_k128").unwrap().name,
        DEFAULT_GEMM_TILE.name
    );
    assert_eq!(
        select_gemm_tile(DEFAULT_GEMM_TILE.name).unwrap().name,
        DEFAULT_GEMM_TILE.name
    );
    assert!(select_gemm_tile("bt_m16_k129").is_none());
    assert!(select_gemm_tile("m16").is_none());
}

/// The default tile's grid height is counted in 16s, so the bound must grow where the
/// 64-row one would not. This is the exact off-by-one that would drop rows 16..63 of an
/// expert if the dispatch kept using `GROUPED_M_TILE`.
#[test]
fn m16_tile_needs_more_rows_of_grid_than_the_base_tile() {
    let mut off = vec![0i32; 289];
    for (e, o) in off.iter_mut().enumerate() {
        *o = if e == 0 { 0 } else { 40 };
    }
    assert_eq!(max_m_tiles_from_offsets(&off, 128, 64), 1);
    assert_eq!(max_m_tiles_from_offsets(&off, 128, 16), 3);
    assert_eq!(DEFAULT_GEMM_TILE.m_tile, 16);
}
