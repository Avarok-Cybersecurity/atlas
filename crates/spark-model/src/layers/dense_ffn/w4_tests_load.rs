// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the load-time and precedence half of the dense FFN W4 arm's
//! mock tests (a child of `w4_tests.rs`, split for the 500-line cap): shape
//! refusal, the capture guard, the finalizers' scratch reservation and repack
//! skip, the BF16 overlay detach, FP8 precedence, row-off identity of the
//! hooks, and the LoRA refusal.

use super::*;
use crate::port::w4::W4Linear;
use crate::weight_map::{DenseWeight, Fp8Weight, WeightQuantFormat};

#[test]
fn shapes_are_checked_against_the_model_at_the_first_forward() {
    let gpu = MockGpuBackend::new();
    let log = fake::log();
    let gate: W4Box = Arc::new(FakeW4::new("gate", INTER, 64, 16, &log));
    let up: W4Box = Arc::new(FakeW4::new("up", INTER, 64, 16, &log));
    let down: W4Box = Arc::new(FakeW4::new("down", 64, INTER, 16, &log));
    let mut layer = bare_layer(&gpu);
    layer.set_w4(gate, up, down);
    with_ctx(&gpu, false, |ctx, b| {
        let e = layer
            .forward(b.norm_output(), ctx, STREAM)
            .unwrap_err()
            .to_string();
        assert!(e.contains("gate_proj") && e.contains("[256, 64]"), "{e}");
        assert!(log.lock().is_empty(), "nothing may run on a refused layer");
    });
}

#[test]
fn capture_before_the_scratch_is_reserved_is_refused_with_the_fix() {
    let gpu = MockGpuBackend::new();
    let f = fakes(16, |w| w);
    let mut layer = bare_layer(&gpu);
    install(&mut layer, &f);
    with_ctx(&gpu, true, |ctx, b| {
        let e = layer
            .forward(b.norm_output(), ctx, STREAM)
            .unwrap_err()
            .to_string();
        assert!(e.contains("finalize_w4_load"), "{e}");
    });
    // finalize_q4k_load is the loader's existing hook and reserves it.
    layer
        .finalize_q4k_load(&gpu, H as u32, INTER as u32, STREAM)
        .unwrap();
    with_ctx(&gpu, true, |ctx, b| {
        layer.forward(b.norm_output(), ctx, STREAM).unwrap();
    });
    let s = f.gate.scratches.lock()[0];
    let spec = layer.w4().unwrap().scratch_spec();
    assert_eq!(
        s,
        scratch::acquire(&gpu, &spec, STREAM).unwrap(),
        "the shared regions"
    );
    assert!(s.ws_bytes >= f.gate.scratch_bytes(16));
}

#[test]
fn the_finalizers_reserve_w4_scratch_and_skip_the_nvfp4_repacks() {
    let gpu = MockGpuBackend::new();
    let f = fakes(16, |w| w);
    let mut layer = bare_layer(&gpu);
    install(&mut layer, &f);
    for h in [
        &mut layer.q4k_mmq_nc_k,
        &mut layer.q4k_quant_act_k,
        &mut layer.q4k_quant_w_k,
        &mut layer.dequant_nvfp4_bf16_k,
        &mut layer.nvfp4_mmq_nc_k,
        &mut layer.nvfp4_quant_act_k,
        &mut layer.nvfp4_repack_k,
        &mut layer.nvfp4_silu_scaled_k,
    ] {
        *h = KernelHandle(0xBEEF);
    }
    let (allocs, launches) = (gpu.alloc_count(), gpu.launch_count());
    layer
        .finalize_q4k_load(&gpu, H as u32, INTER as u32, STREAM)
        .unwrap();
    assert_eq!(
        gpu.alloc_count(),
        allocs + 5,
        "ws, tickets, xq, xs and the BF16 target"
    );
    layer
        .finalize_nvfp4_mmq_load(&gpu, H as u32, INTER as u32, STREAM)
        .unwrap();
    assert_eq!(
        gpu.alloc_count(),
        allocs + 5,
        "no NVFP4 repack may be allocated"
    );
    assert_eq!(gpu.launch_count(), launches, "no kernel may be launched");
    // A second layer with the same shapes shares the regions.
    let f2 = fakes(16, |w| w);
    let mut second = bare_layer(&gpu);
    install(&mut second, &f2);
    let before_second = gpu.alloc_count();
    let expect = layer.w4().unwrap().scratch_spec().device_bytes();
    assert_eq!(second.finalize_w4_load(&gpu, STREAM).unwrap(), expect);
    assert_eq!(
        gpu.alloc_count(),
        before_second,
        "same shapes, same regions"
    );
    assert_eq!(bare_layer(&gpu).finalize_w4_load(&gpu, STREAM).unwrap(), 0);
}

#[test]
fn set_w4_detaches_a_bf16_overlay_and_reports_its_bytes() {
    let gpu = MockGpuBackend::new();
    let mut layer = bare_layer(&gpu);
    let bf16 = |bytes: usize| DenseWeight {
        weight: gpu.alloc(bytes).unwrap(),
    };
    layer.set_bf16_weights(
        bf16(INTER * H * 2),
        bf16(INTER * H * 2),
        bf16(H * INTER * 2),
    );
    let f = fakes(16, |w| w);
    assert_eq!(install(&mut layer, &f), 3 * INTER * H * 2);
    assert!(layer.bf16_weights.is_none() && layer.has_w4());
    let mut plain = bare_layer(&gpu);
    assert_eq!(install(&mut plain, &fakes(16, |w| w)), 0);
    assert_eq!(layer.w4().unwrap().device_bytes(), 3 * INTER * H / 2);
}

fn fp8_layer(gpu: &MockGpuBackend) -> DenseFfnLayer {
    let mut layer = bare_layer(gpu);
    layer.w8a16_gemm_k = KernelHandle(0xF08);
    let w = |n: usize, k: usize| Fp8Weight {
        weight: gpu.alloc(n * k).unwrap(),
        row_scale: gpu.alloc(n.div_ceil(128) * k.div_ceil(128) * 4).unwrap(),
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    };
    layer.set_fp8_weights(w(INTER, H), w(INTER, H), w(H, INTER));
    layer
}

#[test]
fn native_fp8_wins_over_w4_at_every_entry_point() {
    let gpu = MockGpuBackend::new();
    let mut layer = fp8_layer(&gpu);
    let f = fakes(16, |w| w);
    install(&mut layer, &f);
    with_ctx(&gpu, false, |ctx, b| {
        let input = b.norm_output();
        layer.forward(input, ctx, STREAM).unwrap();
        layer.forward_k2(input, ctx, STREAM).unwrap();
        layer.forward_k3(input, ctx, STREAM).unwrap();
        layer.forward_km(input, 4, ctx, STREAM).unwrap();
        layer.forward_prefill(input, 8, ctx, STREAM).unwrap();
        layer.forward_batched(input, 8, ctx, STREAM).unwrap();
        assert!(
            take(&f.log).is_empty(),
            "an FP8 layer must never reach the W4 weights"
        );
    });
}

/// Row-off identity of the hooks: with no W4 weights (and the FP8 verify row
/// off), the K=2/3/m entry points of a native overlay launch exactly what
/// `forward_prefill` launches, which is what they did before the hooks.
#[test]
fn without_w4_the_small_batch_hooks_are_inert() {
    let row_on = crate::layers::ops::port_rows::ffn_fp8_verify_gemv::row();
    let gpu = MockGpuBackend::new();
    let mut bf16 = bare_layer(&gpu);
    let dw = |bytes: usize| DenseWeight {
        weight: gpu.alloc(bytes).unwrap(),
    };
    bf16.set_bf16_weights(dw(INTER * H * 2), dw(INTER * H * 2), dw(H * INTER * 2));
    let fp8 = fp8_layer(&gpu);
    let mut layers = vec![("bf16", bf16)];
    if !row_on {
        layers.push(("fp8", fp8));
    }
    for (name, layer) in &layers {
        assert!(!layer.has_w4());
        with_ctx(&gpu, false, |ctx, b| {
            let input = b.norm_output();
            let allocs = gpu.alloc_count();
            type Run<'r> = Box<dyn Fn() + 'r>;
            let cases: Vec<(u32, Run)> = vec![
                (
                    2,
                    Box::new(|| layer.forward_k2(input, ctx, STREAM).unwrap()),
                ),
                (
                    3,
                    Box::new(|| layer.forward_k3(input, ctx, STREAM).unwrap()),
                ),
                (
                    4,
                    Box::new(|| layer.forward_km(input, 4, ctx, STREAM).unwrap()),
                ),
                (
                    8,
                    Box::new(|| layer.forward_km(input, 8, ctx, STREAM).unwrap()),
                ),
            ];
            for (m, run) in &cases {
                let s0 = gpu.launch_count();
                run();
                let hooked = trace_since(&gpu, s0);
                let s1 = gpu.launch_count();
                layer
                    .forward_prefill(input, *m as usize, ctx, STREAM)
                    .unwrap();
                let direct = trace_since(&gpu, s1);
                assert!(
                    !direct.is_empty(),
                    "{name} m={m}: the reference must dispatch"
                );
                assert_eq!(
                    hooked, direct,
                    "{name} m={m}: the hooks changed the dispatch"
                );
            }
            assert_eq!(
                gpu.alloc_count(),
                allocs,
                "{name}: no W4 scratch without W4 weights"
            );
        });
    }
}

#[test]
fn lora_is_refused_on_a_w4_layer() {
    let gpu = MockGpuBackend::new();
    let f = fakes(16, |w| w);
    let mut layer = w4_layer(&gpu, &f);
    let lora = crate::layers::ops::lora_delta::LoraFfnWeights {
        gate: None,
        up: None,
        down: None,
        kernels: crate::layers::ops::lora_delta::LoraKernels::new(&gpu).unwrap(),
    };
    let e = layer.set_lora_weights(lora).unwrap_err().to_string();
    assert!(e.contains("W4"), "{e}");
}
