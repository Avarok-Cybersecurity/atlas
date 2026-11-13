// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU tests for the 32-row half of the `ffn_fp8_m16_tc` arm
//! (`fp8_m16_tc.rs`): which entry pair a width takes (the 16-row entries up to
//! 16 rows, `fp8_gemm_ada_m32` and `_m32_dual` for 17 to 32), their grids and
//! argument tails, and that each tile's launches are the same three. A child
//! of `fp8_m16_tc_tests.rs`, split out for the 500-line cap; it shares that
//! file's mock layer and context helpers. The kernel's numerics are the device
//! oracle's (`examples/fp8_ada_m16_tc_microtest.rs`).

use super::*;

#[test]
fn thirty_two_rows_use_the_m32_entries_and_sixteen_stay_on_the_m16_ones() {
    let gpu = MockGpuBackend::new();
    let layer = fp8_layer(&gpu);
    with_ctx(&gpu, false, |ctx, buffers| {
        let x = buffers.norm_output();
        for (rows, dual, plain) in [
            (TILE_ROWS, DUAL_K, PLAIN_K),
            (TILE_ROWS + 1, DUAL32_K, PLAIN32_K),
            (24, DUAL32_K, PLAIN32_K),
            (MAX_ROWS, DUAL32_K, PLAIN32_K),
        ] {
            let start = gpu.launch_count();
            assert!(
                layer
                    .fp8_m16_tc_arm_with(true, MIN, &kernels(), x, rows, ctx, 5)
                    .unwrap(),
                "rows={rows}"
            );
            assert_eq!(
                funcs(&gpu, start),
                vec![dual.0, ACT_MUL.0, plain.0],
                "rows={rows}"
            );
            let l = gpu.launches_snapshot();
            // Same grids and blocks on both tiles: the M tile is the
            // kernel's, the grid walks N only.
            assert_eq!(l[start].grid, [INTER.div_ceil(N_TILE), 1, 2]);
            assert_eq!(l[start + 2].grid, [H.div_ceil(N_TILE), 1, 1]);
            assert_eq!(l[start].block, [THREADS, 1, 1]);
            assert_eq!(
                l[start + 1].grid[0],
                (rows as u32 * INTER).div_ceil(256),
                "SwiGLU over all {rows} rows"
            );
            let tail = |i: usize| l[i].args[l[i].args.len() - 3..].to_vec();
            let m = rows as u32;
            assert_eq!(tail(start), vec![u32_arg(m), u32_arg(INTER), u32_arg(H)]);
            assert_eq!(
                tail(start + 2),
                vec![u32_arg(m), u32_arg(H), u32_arg(INTER)]
            );
            assert!(
                l[start + 2]
                    .args
                    .contains(&MockArg::Buffer(buffers.moe_output())),
                "rows={rows}: the result lands where every small-batch entry point writes"
            );
            assert!(l[start..].iter().all(|l| l.stream == 5), "rows={rows}");
        }
    });
}

#[test]
fn the_pair_for_a_width_is_the_tile_that_holds_it() {
    let k = kernels();
    let ids = |rows: usize| {
        let (plain, dual) = k.pair_for(rows);
        (plain.0, dual.0)
    };
    for rows in 1..=TILE_ROWS {
        assert_eq!(ids(rows), (PLAIN_K.0, DUAL_K.0), "rows={rows}");
    }
    for rows in TILE_ROWS + 1..=MAX_ROWS {
        assert_eq!(ids(rows), (PLAIN32_K.0, DUAL32_K.0), "rows={rows}");
    }
}

#[test]
fn a_width_above_the_32_row_tile_declines_even_with_every_handle_present() {
    let gpu = MockGpuBackend::new();
    let layer = fp8_layer(&gpu);
    with_ctx(&gpu, true, |ctx, buffers| {
        let start = gpu.launch_count();
        for rows in [MAX_ROWS + 1, 40, 64] {
            assert!(
                !layer
                    .fp8_m16_tc_arm_with(true, MIN, &kernels(), buffers.norm_output(), rows, ctx, 0)
                    .unwrap(),
                "rows={rows}"
            );
        }
        assert_eq!(gpu.launch_count(), start);
    });
}
