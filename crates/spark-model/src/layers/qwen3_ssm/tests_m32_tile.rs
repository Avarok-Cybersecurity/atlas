// SPDX-License-Identifier: AGPL-3.0-only

//! G18 lever B on the batched GDN verify: the block-scaled FP8 `in_proj_qkvz`
//! / `out_proj` at 17..=32 rows dispatch the 32-row M-tile twin
//! (`w8a16_gemm_pipelined_m32`, nine parameters) instead of the 128-row tile
//! (seven). Child of `tests.rs`, sharing its native-FP8 layer and verify
//! harness. Numerics are the GPU oracle
//! (`examples/native_fp8_gdn_proj_m32_microtest`): bit-identical to the tile
//! it replaces.

use super::{native_fp8_gdn_layer, run_batched_verify, scalar_u32};
use crate::weight_map::Fp8Weight;
use avarok_core::config::ModelConfig;
use spark_runtime::gpu::KernelHandle;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};

const QKVZ_N: u32 = 12_288;
const H: u32 = 2_048;
const OUT_N: u32 = 2_048;
const VALUE_DIM: u32 = 4_096;

/// Every launch that consumed `weight`'s block-scaled pair at `m` rows, with
/// its parameter count — the only thing that tells the two tiles apart on
/// the mock (every handle resolves to the same id there).
fn projection_launches(gpu: &MockGpuBackend, weight: &Fp8Weight, m: u32) -> Vec<MockLaunch> {
    gpu.launches_snapshot()
        .into_iter()
        .filter(|l| {
            l.args.len() >= 7
                && l.args[1] == MockArg::Buffer(weight.weight)
                && l.args[2] == MockArg::Buffer(weight.row_scale)
                && scalar_u32(&l.args[4]) == Some(m)
        })
        .collect()
}

/// The 32-tile twin's launch: grid `(ceil(N/32), ceil(M/32))`, and the nine
/// parameters with the contiguous pitches `lda = K`, `ldc = N`.
fn assert_m32_tile(l: &MockLaunch, m: u32, n: u32, k: u32) {
    assert_eq!(l.args.len(), 9, "the twin takes lda/ldc");
    assert_eq!(l.grid, [n.div_ceil(32), m.div_ceil(32), 1]);
    assert_eq!(l.block, [256, 1, 1]);
    assert_eq!(scalar_u32(&l.args[5]), Some(n));
    assert_eq!(scalar_u32(&l.args[6]), Some(k));
    assert_eq!(scalar_u32(&l.args[7]), Some(k), "lda = K (contiguous)");
    assert_eq!(scalar_u32(&l.args[8]), Some(n), "ldc = N (contiguous)");
}

/// THE lever: R = 8 x 4 = 32 verify rows (the MoE's C=16, k=2 step) run
/// ONE 32-tile launch per projection — no 128-tile launch, no batch16
/// launch, for either weight.
#[test]
fn native_fp8_gdn_batched_verify_r32_takes_the_m32_tile() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_gdn_layer(&gpu, &config, true, true);
    run_batched_verify(&gpu, &config, &layer, &[4; 8]).unwrap();
    let qkvz = layer.qkvz_fp8w.as_ref().unwrap();
    let out = layer.out_proj_fp8w.as_ref().unwrap();

    let q = projection_launches(&gpu, qkvz, 32);
    assert_eq!(q.len(), 1, "one QKVZ launch at R=32: {q:?}");
    assert_m32_tile(&q[0], 32, QKVZ_N, H);
    let o = projection_launches(&gpu, out, 32);
    assert_eq!(o.len(), 1, "one out_proj launch at R=32: {o:?}");
    assert_m32_tile(&o[0], 32, OUT_N, VALUE_DIM);
}

/// Below the band the twin is not selected: R=16 stays on `w8a16_gemv_batch16`
/// (grid `ceil(N/4)`, seven parameters), bit-identical to the M=1 decode —
/// lever B moves 17..=32 only.
#[test]
fn native_fp8_gdn_batched_verify_r16_stays_on_batch16() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_gdn_layer(&gpu, &config, true, true);
    run_batched_verify(&gpu, &config, &layer, &[4; 4]).unwrap();
    let qkvz = layer.qkvz_fp8w.as_ref().unwrap();
    let out = layer.out_proj_fp8w.as_ref().unwrap();
    for (w, n) in [(qkvz, QKVZ_N), (out, OUT_N)] {
        let l = projection_launches(&gpu, w, 16);
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].args.len(), 7, "batch16 GEMV, not the twin");
        assert_eq!(l[0].grid, [n.div_ceil(4), 1, 1]);
    }
}

/// A shadow without the MAX_M=16 GEMV lands 5..=16 on the tile arm, which is
/// now the twin at those widths too (M=7 -> one 32-row tile, no padding to
/// 128). The 128-tile fallback without EITHER kernel is pinned in `tests.rs`.
#[test]
fn native_fp8_gdn_batched_verify_r7_without_batch16_takes_the_m32_tile() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let mut layer = native_fp8_gdn_layer(&gpu, &config, true, true);
    layer.w8a16_gemv_batch16_k = KernelHandle(0);
    run_batched_verify(&gpu, &config, &layer, &[4, 3]).unwrap();
    let qkvz = layer.qkvz_fp8w.as_ref().unwrap();
    let out = layer.out_proj_fp8w.as_ref().unwrap();
    let q = projection_launches(&gpu, qkvz, 7);
    assert_eq!(q.len(), 1);
    assert_m32_tile(&q[0], 7, QKVZ_N, H);
    let o = projection_launches(&gpu, out, 7);
    assert_eq!(o.len(), 1);
    assert_m32_tile(&o[0], 7, OUT_N, VALUE_DIM);
}

/// NEGATIVE CONTROL: a target without `w8a16_gemm_pipelined_m32` runs R=32
/// exactly as before — the 128-row tile (`grid.y = 1`, seven parameters) —
/// never a zero handle.
#[test]
fn native_fp8_gdn_batched_verify_r32_without_the_twin_keeps_the_128_tile() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let mut layer = native_fp8_gdn_layer(&gpu, &config, true, true);
    layer.w8a16_gemm_pipelined_m32_k = KernelHandle(0);
    run_batched_verify(&gpu, &config, &layer, &[4; 8]).unwrap();
    let qkvz = layer.qkvz_fp8w.as_ref().unwrap();
    let out = layer.out_proj_fp8w.as_ref().unwrap();
    for (w, n) in [(qkvz, QKVZ_N), (out, OUT_N)] {
        let l = projection_launches(&gpu, w, 32);
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].args.len(), 7, "the 128-tile kernel takes no pitches");
        assert_eq!(l[0].grid, [n.div_ceil(32), 1, 1]);
    }
}
