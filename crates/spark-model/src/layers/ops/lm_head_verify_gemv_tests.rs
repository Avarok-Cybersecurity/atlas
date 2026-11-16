// SPDX-License-Identifier: AGPL-3.0-only

//! CPU tests for the `lm_head_verify_gemv` row: which verify row counts it
//! serves and the launches it makes. Bit-identity of the kernel with the
//! M = 1 GEMV is `examples/dense_gemv_bf16_batchm_microtest` (device).

use super::*;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};

/// The 4B hidden size.
const H: usize = 2560;

#[test]
fn the_row_off_claims_nothing() {
    for rows in [1, 2, 8, 9, 16, 32, 160] {
        assert!(!claims(false, true, false, rows, H), "rows={rows}");
    }
}

#[test]
fn it_starts_above_the_band_the_batched_gemv_already_serves() {
    // 2..=8 rows are `lm_head_batched`'s own batched-GEMV arm, row or no row.
    for rows in [1, 2, 4, 8] {
        assert!(!claims(true, true, false, rows, H), "rows={rows}");
    }
    // 9 and up, to the verify row cap, in passes of 16.
    for rows in [9, 12, 16, 17, 32, 96, 160] {
        assert!(claims(true, true, false, rows, H), "rows={rows}");
    }
}

#[test]
fn an_absent_kernel_a_sharded_head_or_an_unaligned_depth_declines() {
    assert!(!claims(true, false, false, 16, H), "no kernel");
    assert!(!claims(true, true, true, 16, H), "vocab-parallel head");
    assert!(!claims(true, true, false, 16, H + 4), "K % 8 != 0");
    assert!(claims(true, true, false, 16, H + 8));
}

#[test]
fn sixteen_rows_are_one_pass_and_thirty_two_are_two() {
    let gpu = MockGpuBackend::new();
    let (v, h) = (24u32, 16u32);
    let hidden = gpu.alloc(32 * h as usize * 2).unwrap();
    let logits = gpu.alloc(32 * v as usize * 2).unwrap();
    let weight = DenseWeight {
        weight: gpu.alloc((v * h) as usize * 2).unwrap(),
    };
    let handle = KernelHandle(0x1A7);
    let u32_arg = |x: u32| MockArg::Bytes(x.to_le_bytes().to_vec());

    let start = gpu.launches_snapshot().len();
    launch_rows(&gpu, handle, hidden, &weight, logits, 16, v, h, 2).unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!(l.len() - start, 1, "one weight pass for 16 rows");
    assert_eq!(l[start].func, handle.0);
    assert_eq!(l[start].args[3], u32_arg(16));
    // Logit rows are a full vocabulary apart.
    assert_eq!(l[start].args[6], u32_arg(v));

    let start = l.len();
    launch_rows(&gpu, handle, hidden, &weight, logits, 32, v, h, 2).unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!(l.len() - start, 2, "two passes for 32 rows");
    assert_eq!(
        l[start + 1].args[0],
        MockArg::Buffer(hidden.offset(16 * h as usize * 2))
    );
    assert_eq!(
        l[start + 1].args[2],
        MockArg::Buffer(logits.offset(16 * v as usize * 2))
    );
    assert!(l[start..].iter().all(|x| x.stream == 2));
}
