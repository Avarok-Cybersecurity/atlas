// SPDX-License-Identifier: AGPL-3.0-only

//! Host tests for the pure per-expert table packer (`pack_expert_tables`) — the
//! correctness-critical layout that feeds the device `moe_lora_grouped_down`
//! fold. No GPU: verifies dense placement, unadapted `0` sentinels, table
//! length, and the router-only (empty) case.

use super::{
    ExpertTables, gather_bgmv_grids, gather_row_token, grouped_down_wc, pack_expert_tables,
};

#[test]
fn empty_entries_none() {
    // A router-only adapter installs no expert route.
    assert!(pack_expert_tables(&[]).is_none());
}

#[test]
fn single_expert_zero_padded_prefix() {
    // Expert id 3 adapted; ids 0..3 are unadapted -> 0/0.0 sentinels, len = 4.
    let t = pack_expert_tables(&[(3u16, 0xAAAA, 0xBBBB, 2.0)]).unwrap();
    assert_eq!(
        t,
        ExpertTables {
            a: vec![0, 0, 0, 0xAAAA],
            b: vec![0, 0, 0, 0xBBBB],
            scale: vec![0.0, 0.0, 0.0, 2.0],
            n_experts: 4,
        }
    );
}

#[test]
fn sparse_experts_dense_table() {
    // Experts 0 and 2 adapted, 1 is a gap -> zero at index 1, len = 3.
    let t = pack_expert_tables(&[(0u16, 0x10, 0x20, 0.5), (2u16, 0x30, 0x40, 0.25)]).unwrap();
    assert_eq!(t.a, vec![0x10, 0, 0x30]);
    assert_eq!(t.b, vec![0x20, 0, 0x40]);
    assert_eq!(t.scale, vec![0.5, 0.0, 0.25]);
    assert_eq!(t.n_experts, 3);
}

#[test]
fn entry_order_independent() {
    // Descending input order lands the same dense table.
    let t = pack_expert_tables(&[(2u16, 0x30, 0x40, 0.25), (0u16, 0x10, 0x20, 0.5)]).unwrap();
    assert_eq!(t.a, vec![0x10, 0, 0x30]);
    assert_eq!(t.scale, vec![0.5, 0.0, 0.25]);
}

#[test]
fn table_length_is_max_id_plus_one() {
    // n_experts is the table length (max id + 1), never the layer's full count.
    let t = pack_expert_tables(&[(7u16, 1, 2, 1.0)]).unwrap();
    assert_eq!(t.n_experts, 8);
    assert_eq!(t.a.len(), 8);
    assert_eq!(t.a[7], 1);
    assert_eq!(t.a[..7], [0u64; 7]);
}

// ── SOLID Incr-4 decode gather-fold planner ──────────────────────────────────

#[test]
fn gather_grids_single_token_decode() {
    // Qwen3.6-A3B decode: top_k=8, max_rank=16, inter=768, hidden=4096.
    // n_slots = 1*top_k = 8. Shrink outputs max_rank, expand outputs hidden.
    let (shrink, expand) = gather_bgmv_grids(16, 4096, 8);
    assert_eq!(shrink, [4, 8, 1]); // ceil(16/4)=4 rank groups, 8 flat rows
    assert_eq!(expand, [1024, 8, 1]); // ceil(4096/4)=1024 hidden groups
}

#[test]
fn gather_grids_verify_flat_rows() {
    // K=3 verify: n_slots = 3*top_k. Grid.y scales with flat (token,slot) rows;
    // grid.x is unchanged (per-output, independent of row count).
    let (shrink, expand) = gather_bgmv_grids(32, 4096, 24);
    assert_eq!(shrink, [8, 24, 1]);
    assert_eq!(expand, [1024, 24, 1]);
}

#[test]
fn gather_grids_rank_not_multiple_of_four_rounds_up() {
    // A padded rank of 6 still covers all outputs (ceil, not floor).
    let (shrink, _) = gather_bgmv_grids(6, 512, 8);
    assert_eq!(shrink[0], 2); // ceil(6/4) = 2
}

#[test]
fn row_token_decomposition_matches_kernel() {
    // token = row / top_k, mirroring the kernel's per-token row_adapter gather.
    assert_eq!(gather_row_token(0, 8), 0);
    assert_eq!(gather_row_token(7, 8), 0); // last slot of token 0
    assert_eq!(gather_row_token(8, 8), 1); // first slot of token 1
    assert_eq!(gather_row_token(23, 8), 2); // K=3 verify, token 2
}

#[test]
fn short_prefill_slot_rows_map_to_owning_token() {
    // Short (<=64-token) LoRA prefill through forward_batched folds each token
    // independently: for token t the fold sees n_slots=top_k flat rows
    // t*top_k .. t*top_k+top_k, and the gate/up x_gather=1 shrink reads x-row
    // (row / top_k) == t (the token's own input) for EVERY slot s. Pin that the
    // flat (token, slot) → token decomposition holds across a full short-prefill
    // token sweep, so no slot of token t ever gathers another token's activation.
    for top_k in [1u32, 2, 8] {
        for t in 0..64u32 {
            for s in 0..top_k {
                assert_eq!(
                    gather_row_token(t * top_k + s, top_k),
                    t,
                    "t={t} s={s} k={top_k}"
                );
            }
        }
    }
    // And the decode/prefill gather grid is per-token exact (n_slots = top_k for a
    // single token), matching the single-token decode replay the prefill mirrors.
    let (shrink, expand) = gather_bgmv_grids(16, 4096, 8);
    assert_eq!(shrink, [4, 8, 1]);
    assert_eq!(expand, [1024, 8, 1]);
}

// ── SOLID Incr-4 router fold (degenerate 1-"expert" gather, top_k=1) ──────────

#[test]
fn router_fold_top_k_one_row_is_token() {
    // The batched router fold drives the gather with top_k=1, so `row / top_k`
    // collapses to `row` — each flat row IS its token, and `row_adapter[row]` is
    // that token's own base/adapt sign (no per-token slice needed).
    for row in [0u32, 1, 3, 4, 255, 4095] {
        assert_eq!(gather_row_token(row, 1), row, "row={row}");
    }
}

#[test]
fn router_fold_grid_covers_all_experts_qwen36() {
    // Qwen3.6-35B-A3B router fold: n_out = num_experts = 256, C=4 concurrent
    // tokens, n_slots = n = 4 (one row per token, NOT expanded by top_k). Shrink
    // outputs max_rank, expand outputs all 256 experts.
    let (shrink, expand) = gather_bgmv_grids(16, 256, 4);
    assert_eq!(shrink, [4, 4, 1]); // ceil(16/4)=4 rank groups, 4 token rows
    assert_eq!(expand, [64, 4, 1]); // ceil(256/4)=64 expert groups
}

#[test]
fn router_fold_grid_single_token_decode() {
    // n=1 decode (the single-stream oracle the batched fold must match): one row.
    let (shrink, expand) = gather_bgmv_grids(32, 256, 1);
    assert_eq!(shrink, [8, 1, 1]);
    assert_eq!(expand, [64, 1, 1]);
}

// ── Prefill chunk-window grid math (grouped_down_wc) ──────────────────────────

/// Local re-derivation of the pre-chunk `wc = ceil(te/64).max(1)` so the tests
/// pin the full-window equivalence independently of `grouped_down_wc` (which
/// computes it via `spark_runtime::kernel_args::div_ceil` over a saturating
/// window). This oracle takes the pre-diffed row count directly and applies the
/// `.max(1)` clamp itself, so a regression in the launcher's window math is caught.
fn div_ceil64(n: u32) -> u32 {
    n.div_ceil(64).max(1)
}

#[test]
fn full_window_equals_prechunk_wc() {
    // A single window [0, te) must reproduce the old ceil(te/64) grid exactly —
    // this is the bit-identity (unchunked) path.
    for te in [0u32, 1, 63, 64, 65, 4096, 4097, 131072] {
        assert_eq!(grouped_down_wc(0, te), div_ceil64(te), "te={te}");
    }
}

#[test]
fn window_grid_covers_only_the_slice() {
    // A cap-sized window sizes grid.y to the window, NOT to te. cap=4096 ->
    // ceil(4096/64)=64 tiles regardless of how far into a huge te it sits.
    assert_eq!(grouped_down_wc(0, 4096), 64);
    assert_eq!(grouped_down_wc(4096, 8192), 64);
    assert_eq!(grouped_down_wc(128000, 131072), 48); // ceil(3072/64)=48 tail
}

#[test]
fn empty_window_still_launches_one_tile() {
    // Degenerate zero-row window clamps to 1 (kernel early-returns per expert).
    assert_eq!(grouped_down_wc(100, 100), 1);
    assert_eq!(grouped_down_wc(200, 100), 1); // row_end < row_offset saturates
}

#[test]
fn chunks_tile_range_without_gap_or_overlap() {
    // The hook loop `for off in (0..te).step_by(cap)` must partition [0, te)
    // contiguously (every row folded exactly once). Verify coverage + the
    // per-window wc matches the window length.
    let cap = 4096u32;
    for te in [1u32, 4096, 4097, 10000, 131072] {
        let mut off = 0u32;
        let mut covered = 0u32;
        let mut prev_end = 0u32;
        while off < te {
            let end = (off + cap).min(te);
            assert_eq!(off, prev_end, "gap/overlap at off={off} (te={te})");
            let window = end - off;
            assert!(window <= cap, "window {window} exceeds cap {cap}");
            assert_eq!(grouped_down_wc(off, end), div_ceil64(window), "te={te}");
            covered += window;
            prev_end = end;
            off = end;
        }
        assert_eq!(covered, te, "windows must cover exactly [0,te) for te={te}");
    }
}
