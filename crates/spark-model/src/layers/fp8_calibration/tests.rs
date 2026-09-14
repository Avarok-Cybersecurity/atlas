// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-glue tests for [`super::Fp8KvCalibration`]. The freeze DECISION is
//! tested without a GPU in `state_tests.rs`; these cover the wiring around it.

use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{Fp8KvCalibration, Fp8KvWriteTarget};

fn compact_source(src: &str) -> String {
    src.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Kernel-arg count of `reshape_and_cache_fp8`.
const RESHAPE_FP8_ARGS: usize = 13;

fn reshape_fp8_launches(gpu: &MockGpuBackend) -> usize {
    gpu.launches_snapshot()
        .iter()
        .filter(|l| l.args.len() == RESHAPE_FP8_ARGS)
        .count()
}

const NKV: u32 = 4;
const HD: u32 = 32;

fn target(gpu: &MockGpuBackend, slot: DevicePtr) -> Fp8KvWriteTarget {
    Fp8KvWriteTarget {
        kernel: gpu
            .kernel("reshape_and_cache", "reshape_and_cache_fp8")
            .unwrap(),
        k_pool: gpu.alloc(1 << 16).unwrap(),
        v_pool: gpu.alloc(1 << 16).unwrap(),
        block_size: 16,
        cache_stride: (16 * NKV * HD) as u64,
        key_stride: NKV * HD,
        value_stride: NKV * HD,
        slot,
    }
}

/// Atlas #919, the reported shape: a 13-token readiness probe must NOT end a
/// 256-token calibration window. Red before the fix — the calibrator froze on
/// the first observe, so `is_calibrating()` was already false here.
#[test]
fn a_readiness_probe_does_not_end_a_256_token_window() {
    let gpu = MockGpuBackend::new();
    let cal = Fp8KvCalibration::new(0, 256, 2.0, &gpu).expect("mock construct");
    let k = gpu.alloc(1 << 16).expect("k buf");
    let v = gpu.alloc(1 << 16).expect("v buf");
    let slot = gpu.alloc(256 * 8).expect("slot buf");
    let tgt = target(&gpu, slot);
    let stream = gpu.default_stream();

    cal.observe(&gpu, k, v, 13, NKV, HD, stream, &tgt)
        .expect("probe observe");
    assert!(
        cal.is_calibrating(),
        "a 13-token probe froze a 256-token window (#919)"
    );
    assert_eq!(
        cal.scales(),
        (super::PROVISIONAL_SCALE, super::PROVISIONAL_SCALE),
        "the window's own writes use the provisional scale"
    );

    cal.observe(&gpu, k, v, 243, NKV, HD, stream, &tgt)
        .expect("second observe");
    assert!(!cal.is_calibrating(), "the window must close at 256 tokens");
}

/// The freeze requantizes the staged window through `reshape_and_cache_fp8`
/// before the caller writes the crossing batch, and frees its staging buffers.
#[test]
fn the_freeze_replays_the_staged_window_and_releases_its_buffers() {
    let gpu = MockGpuBackend::new();
    let cal = Fp8KvCalibration::new(3, 64, 1.0, &gpu).expect("mock construct");
    let k = gpu.alloc(1 << 16).expect("k buf");
    let v = gpu.alloc(1 << 16).expect("v buf");
    let slot = gpu.alloc(64 * 8).expect("slot buf");
    let tgt = target(&gpu, slot);
    let stream = gpu.default_stream();

    for _ in 0..3 {
        cal.observe(&gpu, k, v, 16, NKV, HD, stream, &tgt)
            .expect("staged observe");
    }
    let before = gpu.alloc_count();
    // The mock hands out one KernelHandle for every lookup, so the requantizing
    // writes are told apart from the absmax reductions by arity:
    // `reshape_and_cache_fp8` takes 13 kernel args, `bf16_absmax` takes 3.
    let replays_before = reshape_fp8_launches(&gpu);

    cal.observe(&gpu, k, v, 16, NKV, HD, stream, &tgt)
        .expect("crossing observe");

    let replays = reshape_fp8_launches(&gpu) - replays_before;
    assert_eq!(
        replays, 3,
        "one requantizing write per staged batch, in write order"
    );
    assert!(
        gpu.alloc_count() < before,
        "staging buffers must be freed at the freeze"
    );
}

/// A window of 1 is the explicit "freeze immediately" mode: nothing is staged,
/// so the freeze replays nothing.
#[test]
fn a_one_token_window_stages_nothing() {
    let gpu = MockGpuBackend::new();
    let cal = Fp8KvCalibration::new(0, 1, 2.0, &gpu).expect("mock construct");
    let k = gpu.alloc(1 << 16).expect("k buf");
    let v = gpu.alloc(1 << 16).expect("v buf");
    let slot = gpu.alloc(64 * 8).expect("slot buf");
    let tgt = target(&gpu, slot);
    let stream = gpu.default_stream();

    let before = gpu.alloc_count();
    cal.observe(&gpu, k, v, 13, NKV, HD, stream, &tgt)
        .expect("observe");
    assert!(!cal.is_calibrating());
    assert_eq!(
        gpu.alloc_count(),
        before,
        "freeze-on-first-observe must not allocate staging"
    );
}

#[test]
fn only_plain_fp8_kv_runs_online_calibration() {
    use spark_runtime::kv_cache::KvCacheDtype as D;
    assert!(super::dtype_runs_online_fp8_kv_calibration(D::Fp8));
    assert!(!super::dtype_runs_online_fp8_kv_calibration(D::Bf16));
    assert!(!super::dtype_runs_online_fp8_kv_calibration(D::Nvfp4));
    assert!(!super::dtype_runs_online_fp8_kv_calibration(D::Turbo8));
    assert!(!super::dtype_runs_online_fp8_kv_calibration(D::Fp8KTurbo4V));
}

#[test]
fn graphs_ready_vacuous_when_no_calibrator() {
    assert!(super::graphs_ready_after_fp8_kv_cal([None, None]));
}

#[test]
fn graphs_ready_blocked_while_any_calibrator_is_warm() {
    assert!(!super::graphs_ready_after_fp8_kv_cal([
        None,
        Some(false),
        Some(true)
    ]));
}

#[test]
fn graphs_ready_when_every_calibrator_frozen() {
    assert!(super::graphs_ready_after_fp8_kv_cal([
        None,
        Some(true),
        Some(true)
    ]));
}

#[test]
fn graphs_stay_blocked_when_a_warm_layer_follows_a_frozen_layer() {
    assert!(!super::graphs_ready_after_fp8_kv_cal([
        None,
        Some(true),
        Some(false),
        None
    ]));
}

#[test]
fn attention_init_gates_calibrator_on_tokens_and_plain_fp8() {
    let src = compact_source(include_str!("../qwen3_attention/init.rs"));
    assert!(
        src.contains(
            "fp8_calibration:iffp8_calibration_tokens>0&&crate::layers::fp8_calibration::dtype_runs_online_fp8_kv_calibration(kv_dtype){Some(Fp8KvCalibration::new("
        ),
        "the attention initializer must require enabled tokens and an observing KV dtype"
    );
}

#[test]
fn decode_uses_shared_calibration_readiness() {
    let src = compact_source(include_str!("../../model/trait_impl/decode_a.rs"));
    assert!(
        src.contains(
            "fnfp8_calibration_frozen(&self)->bool{crate::layers::fp8_calibration::graphs_ready_after_fp8_kv_cal(self.layers.iter().map(|l|l.fp8_calibration_frozen()),)}"
        ),
        "decode readiness must aggregate every layer through the shared policy"
    );
}

#[test]
fn fused_verify_unsuppress_matches_decode_frozen_flag() {
    let src = compact_source(include_str!("../../model/trait_impl/verify_fused.rs"));
    assert!(
        src.contains(
            "&&self.fp8_calibration_frozen(){self.suppress_graphs.store(false,std::sync::atomic::Ordering::Relaxed);"
        ),
        "fused verify must unsuppress graphs from the frozen-state predicate"
    );
    assert!(
        !src.contains("calibration_tokens+10"),
        "old token-count gate kept fused verify eager for ~266 tokens"
    );
}
