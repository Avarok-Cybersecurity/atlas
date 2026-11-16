// SPDX-License-Identifier: AGPL-3.0-only

//! CPU tests for the `gdn_verify_gemv` row: which row counts it serves, how
//! it splits them into passes, the launches it makes, and that nothing is
//! looked up with the row off. The kernel's bit-identity with the M = 1 GEMV
//! is `examples/dense_gemv_bf16_batchm_microtest` (device).

use super::*;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};

/// The 4B GDN reduction depths: hidden (in_proj_qkvz) and value_dim (out_proj).
const K_QKVZ: usize = 2560;
const K_OUT: usize = 4096;

#[test]
fn the_row_off_claims_nothing() {
    for rows in [1, 2, 3, 4, 8, 16, 32, 96] {
        assert!(!claims(false, false, true, rows, 2, K_OUT), "rows={rows}");
        assert!(!claims(false, true, true, rows, 3, K_QKVZ), "rows={rows}");
    }
}

#[test]
fn qkvz_starts_at_three_rows_and_out_proj_at_two() {
    // One row is the decode forward. Two QKVZ rows keep `dense_gemv_batch2`,
    // which is already one pass and bit-identical to decode.
    assert!(!claims(true, false, true, 1, 3, K_QKVZ));
    assert!(!claims(true, false, true, 2, 3, K_QKVZ));
    for rows in [3, 4, 5, 8, 16] {
        assert!(claims(true, false, true, rows, 3, K_QKVZ), "rows={rows}");
    }
    assert!(!claims(true, false, true, 1, 2, K_OUT));
    for rows in [2, 3, 4, 8, 16] {
        assert!(claims(true, false, true, rows, 2, K_OUT), "rows={rows}");
    }
}

#[test]
fn an_absent_kernel_or_a_depth_off_the_16_byte_grid_declines() {
    assert!(!claims(true, false, false, 4, 3, K_QKVZ), "no kernel");
    assert!(!claims(true, false, true, 4, 3, K_QKVZ + 4), "K % 8 != 0");
    assert!(claims(true, false, true, 4, 3, K_QKVZ + 8));
}

#[test]
fn above_one_pass_the_cublas_row_keeps_its_single_pass() {
    // Up to 16 rows this row wins whether or not `gdn_batched_gemm` is on.
    assert!(claims(true, true, true, ROWS_PER_PASS, 3, K_QKVZ));
    // Above it, cuBLASLt reads the weight once and this row would read it
    // twice or more, so it yields when that row is on...
    assert!(!claims(true, true, true, ROWS_PER_PASS + 1, 3, K_QKVZ));
    assert!(!claims(true, true, true, 32, 2, K_OUT));
    // ...and serves the width in several passes when it is not (the
    // alternative there is the scalar GEMM).
    assert!(claims(true, false, true, ROWS_PER_PASS + 1, 3, K_QKVZ));
    assert!(claims(true, false, true, 96, 2, K_OUT));
}

#[test]
fn passes_cover_every_row_once_in_chunks_of_sixteen() {
    assert_eq!(passes(0).count(), 0);
    assert_eq!(passes(3).collect::<Vec<_>>(), vec![(0, 3)]);
    assert_eq!(passes(16).collect::<Vec<_>>(), vec![(0, 16)]);
    assert_eq!(passes(17).collect::<Vec<_>>(), vec![(0, 16), (16, 1)]);
    assert_eq!(
        passes(35).collect::<Vec<_>>(),
        vec![(0, 16), (16, 16), (32, 3)]
    );
    for rows in 1..=96usize {
        let p: Vec<_> = passes(rows).collect();
        assert_eq!(p.iter().map(|(_, m)| m).sum::<usize>(), rows);
        assert!(p.iter().all(|&(_, m)| (1..=ROWS_PER_PASS).contains(&m)));
        assert_eq!(p.len(), rows.div_ceil(ROWS_PER_PASS));
    }
}

#[test]
fn launch_rows_is_one_batched_gemv_per_pass_at_the_right_offsets() {
    let gpu = MockGpuBackend::new();
    let (rows, n, k) = (35usize, 8u32, 16u32);
    let input = gpu.alloc(rows * k as usize * 2).unwrap();
    let output = gpu.alloc(rows * n as usize * 2).unwrap();
    let weight = DenseWeight {
        weight: gpu.alloc((n * k) as usize * 2).unwrap(),
    };
    let handle = KernelHandle(0x6D7);
    let start = gpu.launches_snapshot().len();
    launch_rows(&gpu, handle, input, &weight, output, rows, n, k, 5).unwrap();
    let l = gpu.launches_snapshot();
    let l = &l[start..];
    assert_eq!(l.len(), 3, "35 rows are 16 + 16 + 3");
    let u32_arg = |v: u32| MockArg::Bytes(v.to_le_bytes().to_vec());
    for (launch, (first, m)) in l.iter().zip([(0usize, 16u32), (16, 16), (32, 3)]) {
        assert_eq!(launch.func, handle.0);
        assert_eq!(launch.stream, 5);
        assert_eq!(launch.grid, [n.div_ceil(4), 1, 1]);
        assert_eq!(
            launch.args,
            vec![
                MockArg::Buffer(input.offset(first * k as usize * 2)),
                MockArg::Buffer(weight.weight),
                MockArg::Buffer(output.offset(first * n as usize * 2)),
                u32_arg(m),
                u32_arg(n),
                u32_arg(k),
                // Contiguous output rows: the pitch is N.
                u32_arg(n),
            ],
            "pass at row {first}"
        );
    }
}

#[test]
fn the_handle_comes_from_the_op_cache_and_an_absent_module_issues_no_lookup() {
    let gpu = MockGpuBackend::new();
    let before = gpu.kernel_lookups_snapshot().len();
    assert!(kernel(&gpu).0 != 0);
    assert_eq!(gpu.kernel_lookups_snapshot().len(), before + 1);
    assert!(kernel(&gpu).0 != 0);
    assert_eq!(
        gpu.kernel_lookups_snapshot().len(),
        before + 1,
        "the second read is a cache hit"
    );

    let bare = MockGpuBackend::new();
    bare.mark_module_absent(KERNEL.0);
    let before = bare.kernel_lookups_snapshot().len();
    assert_eq!(kernel(&bare).0, 0);
    assert_eq!(bare.kernel_lookups_snapshot().len(), before);
}

#[test]
fn with_the_row_off_the_process_predicates_look_nothing_up() {
    // The default environment resolves every port row off; this test is
    // meaningful only then.
    if row() {
        return;
    }
    let gpu = MockGpuBackend::new();
    let before = gpu.kernel_lookups_snapshot().len();
    prime(&gpu);
    for rows in [2, 3, 4, 8, 16, 32] {
        assert!(!qkvz_claims(&gpu, rows, K_QKVZ), "rows={rows}");
        assert!(!out_proj_claims(&gpu, rows, K_OUT), "rows={rows}");
    }
    assert_eq!(gpu.kernel_lookups_snapshot().len(), before);
}
