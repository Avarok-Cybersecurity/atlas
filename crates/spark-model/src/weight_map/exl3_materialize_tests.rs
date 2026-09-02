// SPDX-License-Identifier: AGPL-3.0-only

//! Mock-backend tests for the EXL3 materialize pass (split from
//! `exl3_materialize.rs` for the 500-LoC cap).

use std::collections::HashMap;

use spark_runtime::gpu::mock::MockGpuBackend;

use super::*;

/// Dense gate OFF: every pre-existing test keeps its exact behavior.
const OFF: crate::weight_map::Exl3DenseFamilies = crate::weight_map::Exl3DenseFamilies::OFF;

fn t(gpu: &MockGpuBackend, shape: Vec<usize>, dtype: WeightDtype) -> WeightTensor {
    let bytes: usize = shape.iter().product::<usize>() * dtype.byte_size().max(1);
    WeightTensor {
        ptr: gpu.alloc(bytes.max(4)).unwrap(),
        shape,
        dtype,
    }
}

fn exl3_linear(gpu: &MockGpuBackend, m: &mut HashMap<String, WeightTensor>, p: &str, k: u32) {
    // [2560 -> 640] geometry, K bits.
    m.insert(
        format!("{p}.trellis"),
        t(gpu, vec![160, 40, 16 * k as usize], WeightDtype::UInt16),
    );
    m.insert(format!("{p}.suh"), t(gpu, vec![2560], WeightDtype::F16));
    m.insert(format!("{p}.svh"), t(gpu, vec![640], WeightDtype::F16));
    m.insert(format!("{p}.mul1"), t(gpu, vec![], WeightDtype::Int32));
}

#[test]
fn no_exl3_is_noop() {
    let gpu = MockGpuBackend::new();
    let mut m = HashMap::new();
    m.insert(
        "a.weight".to_string(),
        t(&gpu, vec![8, 8], WeightDtype::BF16),
    );
    let mut store = WeightStore::from_map(m);
    let stats = materialize_exl3(&gpu, &mut store).unwrap();
    assert_eq!(stats, Exl3MaterializeStats::default());
    assert!(store.contains("a.weight"));
}

#[test]
fn routes_experts_to_triplet_and_attention_to_bf16() {
    let gpu = MockGpuBackend::new();
    let mut m = HashMap::new();
    exl3_linear(&gpu, &mut m, "model.layers.0.mlp.experts.3.gate_proj", 4);
    exl3_linear(&gpu, &mut m, "model.layers.0.mlp.shared_expert.up_proj", 6);
    exl3_linear(&gpu, &mut m, "model.layers.0.linear_attn.in_proj_qkv", 6);
    // Bystander that must survive untouched.
    m.insert(
        "model.layers.0.norm.weight".to_string(),
        t(&gpu, vec![2560], WeightDtype::BF16),
    );
    let mut store = WeightStore::from_map(m);

    let stats = materialize_exl3(&gpu, &mut store).unwrap();
    assert_eq!(stats.quantized, 2);
    assert_eq!(stats.bf16, 1);

    // Expert: ModelOpt-style NVFP4 triplet, [n=640, k=2560].
    let ep = "model.layers.0.mlp.experts.3.gate_proj";
    let w = store.get(&format!("{ep}.weight")).unwrap();
    assert_eq!(w.dtype, WeightDtype::UInt8);
    assert_eq!(w.shape, vec![640, 1280]); // [n, k/2]
    let s = store.get(&format!("{ep}.weight_scale")).unwrap();
    assert_eq!(s.dtype, WeightDtype::FP8E4M3);
    assert_eq!(s.shape, vec![640, 160]); // [n, k/16]
    let s2 = store.get(&format!("{ep}.weight_scale_2")).unwrap();
    assert_eq!(s2.dtype, WeightDtype::FP32);

    // Attention: dense BF16 [out, in].
    let ap = "model.layers.0.linear_attn.in_proj_qkv";
    let w = store.get(&format!("{ap}.weight")).unwrap();
    assert_eq!(w.dtype, WeightDtype::BF16);
    assert_eq!(w.shape, vec![640, 2560]);

    // Every EXL3 source tensor is gone; the bystander survived.
    for p in [ep, ap, "model.layers.0.mlp.shared_expert.up_proj"] {
        for sfx in ["trellis", "suh", "svh", "mul1"] {
            assert!(!store.contains(&format!("{p}.{sfx}")), "{p}.{sfx} remains");
        }
    }
    assert!(store.contains("model.layers.0.norm.weight"));

    // Idempotent: second call is a no-op.
    let again = materialize_exl3(&gpu, &mut store).unwrap();
    assert_eq!(again, Exl3MaterializeStats::default());
}

#[test]
fn native_mode_keeps_supported_lm_head_packed() {
    let gpu = MockGpuBackend::new();
    let mut m = HashMap::new();
    exl3_linear(&gpu, &mut m, "lm_head", 4);
    exl3_linear(&gpu, &mut m, "model.layers.0.linear_attn.in_proj_qkv", 4);
    exl3_linear(&gpu, &mut m, "model.layers.0.mlp.experts.0.gate_proj", 4);
    let mut store = WeightStore::from_map(m);
    // Stamp the mul1 codebook flag — a fresh mock alloc reads back zero,
    // which is cb0/"3inst" and NOT natively supported.
    let flag = store.get("lm_head.mul1").unwrap().ptr;
    gpu.copy_h2d(&0x83DC_D12Du32.to_le_bytes(), flag).unwrap();

    let stats = materialize_exl3_impl(&gpu, &mut store, true, false, OFF).unwrap();
    assert_eq!(stats.kept_native, 1);
    assert_eq!(stats.quantized, 1); // the expert still lands as NVFP4
    assert_eq!(stats.bf16, 1); // the GDN linear still lands as BF16

    // lm_head stays packed — no BF16 rewrite, nothing freed.
    for sfx in ["trellis", "suh", "svh", "mul1"] {
        assert!(store.contains(&format!("lm_head.{sfx}")));
    }
    assert!(!store.contains("lm_head.weight"));
    // The non-served GDN linear materialized exactly as before.
    assert!(store.contains("model.layers.0.linear_attn.in_proj_qkv.weight"));
    assert!(!store.contains("model.layers.0.linear_attn.in_proj_qkv.trellis"));

    // Second call: still idempotent — keeps keeping, rewrites nothing.
    let again = materialize_exl3_impl(&gpu, &mut store, true, false, OFF).unwrap();
    assert_eq!(again.kept_native, 1);
    assert_eq!(again.quantized + again.bf16, 0);
}

#[test]
fn native_mode_unsupported_codebook_falls_back_to_bf16() {
    // cb0 (unset flag scalar) has no compiled kernels — lm_head must
    // materialize even when native mode asks for it.
    let gpu = MockGpuBackend::new();
    let mut m = HashMap::new();
    exl3_linear(&gpu, &mut m, "lm_head", 4);
    let mut store = WeightStore::from_map(m);
    let stats = materialize_exl3_impl(&gpu, &mut store, true, false, OFF).unwrap();
    assert_eq!(stats.kept_native, 0);
    assert_eq!(stats.bf16, 1);
    assert!(store.contains("lm_head.weight"));
    assert!(!store.contains("lm_head.trellis"));
}

#[test]
fn native_serve_set_and_support_envelope() {
    assert!(exl3_native_serves("lm_head"));
    // GDN / attention / ViT stay on the materialize path in milestone 1 —
    // their dispatch is not routed yet. Experts join the set ONLY under
    // the MoE gate (threaded explicitly here — env set_var races tests).
    assert!(!exl3_native_serves_with(
        "model.layers.0.linear_attn.in_proj_qkv",
        true,
        OFF
    ));
    assert!(!exl3_native_serves_with(
        "model.layers.3.self_attn.q_proj",
        true,
        OFF
    ));
    assert!(!exl3_native_serves_with(
        "model.visual.blocks.0.attn.q_proj",
        true,
        OFF
    ));
    assert!(!exl3_native_serves_with(
        "model.layers.0.mlp.experts.7.down_proj",
        false,
        OFF
    ));
    assert!(exl3_native_serves_with(
        "model.layers.0.mlp.experts.7.down_proj",
        true,
        OFF
    ));
    assert!(exl3_native_serves_with("lm_head", false, OFF));
    // Shared expert and MTP experts stay materialized under the MoE gate.
    assert!(!exl3_native_serves_with(
        "model.layers.0.mlp.shared_expert.up_proj",
        true,
        OFF
    ));
    assert!(!exl3_native_serves_with(
        "mtp.layers.0.mlp.experts.7.down_proj",
        true,
        OFF
    ));

    let w = |k_bits, cb| Exl3Weight {
        trellis: spark_runtime::gpu::DevicePtr(16),
        suh: spark_runtime::gpu::DevicePtr(32),
        svh: spark_runtime::gpu::DevicePtr(48),
        in_dim: 2560,
        out_dim: 248320,
        k_bits,
        cb,
    };
    // Every K with gemm instances: 2.05 (K=4), 3.05 (K=5), 4.05 (K=6)
    // and 6.05 (K=8 dense / K=6 lm_head) dense sets all qualify.
    for k in [2u32, 3, 4, 5, 6, 8] {
        assert!(exl3_native_supported(&w(k, Exl3Codebook::Mul1)), "K={k}");
        assert!(exl3_native_supported(&w(k, Exl3Codebook::Mcg)), "K={k}");
    }
    assert!(!exl3_native_supported(&w(4, Exl3Codebook::Inst3))); // cb0 not compiled
    assert!(!exl3_native_supported(&w(7, Exl3Codebook::Mul1))); // K=7 not compiled
    assert!(!exl3_native_supported(&w(1, Exl3Codebook::Mul1))); // K=1 not compiled
    let mut odd = w(4, Exl3Codebook::Mul1);
    odd.in_dim = 2504; // not %128
    assert!(!exl3_native_supported(&odd));
    // The dense arm admits exactly the K the dense dispatch accepts.
    for k in EXL3_NATIVE_DENSE_K_BITS {
        assert!(crate::layers::ops::exl3_gemm_serves_k(k));
    }
}

#[test]
fn native_mode_keeps_k6_lm_head_packed() {
    // 4.05bpw ships lm_head at K=6: no GEMV instance, gemm-only — kept.
    let gpu = MockGpuBackend::new();
    let mut m = HashMap::new();
    exl3_linear(&gpu, &mut m, "lm_head", 6);
    let mut store = WeightStore::from_map(m);
    stamp_mul1(&gpu, &store, "lm_head");
    let stats = materialize_exl3_impl(&gpu, &mut store, true, false, OFF).unwrap();
    assert_eq!(stats.kept_native, 1);
    assert!(store.contains("lm_head.trellis"));
    assert!(!store.contains("lm_head.weight"));

    // K=7 (5.05bpw's dense set) has no kernel: materializes to BF16 with
    // the envelope warning, exactly as before the widening.
    let mut m = HashMap::new();
    exl3_linear(&gpu, &mut m, "lm_head", 7);
    let mut store = WeightStore::from_map(m);
    stamp_mul1(&gpu, &store, "lm_head");
    let stats = materialize_exl3_impl(&gpu, &mut store, true, false, OFF).unwrap();
    assert_eq!(stats.kept_native, 0);
    assert_eq!(stats.bf16, 1);
    assert!(store.contains("lm_head.weight"));
}

/// One EXL3 linear at explicit `[in -> out]` geometry (the base helper is
/// pinned to gate/up's [2560 -> 640]; down runs [640 -> 2560]).
fn exl3_linear_dims(
    gpu: &MockGpuBackend,
    m: &mut HashMap<String, WeightTensor>,
    p: &str,
    k: u32,
    in_dim: usize,
    out_dim: usize,
) {
    m.insert(
        format!("{p}.trellis"),
        t(
            gpu,
            vec![in_dim / 16, out_dim / 16, 16 * k as usize],
            WeightDtype::UInt16,
        ),
    );
    m.insert(format!("{p}.suh"), t(gpu, vec![in_dim], WeightDtype::F16));
    m.insert(format!("{p}.svh"), t(gpu, vec![out_dim], WeightDtype::F16));
    m.insert(format!("{p}.mul1"), t(gpu, vec![], WeightDtype::Int32));
}

fn stamp_mul1(gpu: &MockGpuBackend, store: &WeightStore, p: &str) {
    let flag = store.get(&format!("{p}.mul1")).unwrap().ptr;
    gpu.copy_h2d(&0x83DC_D12Du32.to_le_bytes(), flag).unwrap();
}

#[test]
fn native_moe_keeps_uniform_layer_atomically_drops_mixed_layer() {
    let gpu = MockGpuBackend::new();
    let mut m = HashMap::new();
    let expert =
        |l: usize, e: usize, proj: &str| format!("model.layers.{l}.mlp.experts.{e}.{proj}");
    // Layer 0: two experts, uniform K=2 everywhere — kept whole.
    // Layer 1: expert 1's up_proj is K=3 against expert 0's K=2 — the
    // WHOLE layer must materialize (atomic: no partial keeps, or the
    // layer double-holds packed + NVFP4 copies).
    for l in 0..2usize {
        for e in 0..2usize {
            let up_k = if l == 1 && e == 1 { 3 } else { 2 };
            exl3_linear_dims(&gpu, &mut m, &expert(l, e, "gate_proj"), 2, 2560, 640);
            exl3_linear_dims(&gpu, &mut m, &expert(l, e, "up_proj"), up_k, 2560, 640);
            exl3_linear_dims(&gpu, &mut m, &expert(l, e, "down_proj"), 2, 640, 2560);
        }
    }
    // Excluded prefixes under the MoE gate: MTP experts and the shared
    // expert keep materializing to NVFP4 triplets.
    exl3_linear_dims(
        &gpu,
        &mut m,
        "mtp.layers.0.mlp.experts.0.gate_proj",
        2,
        2560,
        640,
    );
    exl3_linear_dims(
        &gpu,
        &mut m,
        "model.layers.0.mlp.shared_expert.up_proj",
        2,
        2560,
        640,
    );
    let mut store = WeightStore::from_map(m);
    for l in 0..2usize {
        for e in 0..2usize {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                stamp_mul1(&gpu, &store, &expert(l, e, proj));
            }
        }
    }
    stamp_mul1(&gpu, &store, "mtp.layers.0.mlp.experts.0.gate_proj");
    stamp_mul1(&gpu, &store, "model.layers.0.mlp.shared_expert.up_proj");

    let stats = materialize_exl3_impl(&gpu, &mut store, true, true, OFF).unwrap();
    assert_eq!(stats.kept_native, 6, "layer 0's six projections kept");
    assert_eq!(stats.kept_native_experts, 6);
    // Layer 1 (6) + MTP expert (1) + shared expert (1) -> NVFP4 triplets.
    assert_eq!(stats.quantized, 8);
    assert_eq!(stats.bf16, 0);
    // Memory accounting: per kept projection, packed = 2560*640*2/8 +
    // (2560+640)*2 + 4 = 416,004 B vs NVFP4 = 2560*640/2 + 2560*640/16
    // + 4 = 921,604 B (both orientations have the same element count).
    assert_eq!(stats.kept_packed_bytes, 6 * 416_004);
    assert_eq!(stats.nvfp4_equiv_bytes, 6 * 921_604);

    for e in 0..2usize {
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            // Layer 0: fully packed (no `.weight`, all sources present).
            let p0 = expert(0, e, proj);
            for sfx in ["trellis", "suh", "svh", "mul1"] {
                assert!(store.contains(&format!("{p0}.{sfx}")), "{p0}.{sfx} kept");
            }
            assert!(!store.contains(&format!("{p0}.weight")));
            // Layer 1: fully materialized — including its perfectly
            // uniform gate/down projections.
            let p1 = expert(1, e, proj);
            assert!(store.contains(&format!("{p1}.weight")), "{p1} materialized");
            assert!(!store.contains(&format!("{p1}.trellis")), "{p1} freed");
        }
    }
    for p in [
        "mtp.layers.0.mlp.experts.0.gate_proj",
        "model.layers.0.mlp.shared_expert.up_proj",
    ] {
        assert!(store.contains(&format!("{p}.weight")), "{p} materialized");
        assert!(
            store.contains(&format!("{p}.weight_scale")),
            "{p} is a triplet"
        );
        assert!(!store.contains(&format!("{p}.trellis")));
    }

    // Idempotent: the second pass keeps keeping and rewrites nothing.
    let again = materialize_exl3_impl(&gpu, &mut store, true, true, OFF).unwrap();
    assert_eq!(again.kept_native, 6);
    assert_eq!(again.kept_native_experts, 6);
    assert_eq!(again.quantized + again.bf16, 0);
}

#[test]
fn moe_gate_off_experts_materialize_exactly_as_before() {
    // native=1, moe=0: experts take the NVFP4 triplet path bit-for-bit
    // as today (the keep-vs-rewrite branch is exclusive).
    let gpu = MockGpuBackend::new();
    let mut m = HashMap::new();
    exl3_linear_dims(
        &gpu,
        &mut m,
        "model.layers.0.mlp.experts.0.gate_proj",
        2,
        2560,
        640,
    );
    let mut store = WeightStore::from_map(m);
    stamp_mul1(&gpu, &store, "model.layers.0.mlp.experts.0.gate_proj");
    let stats = materialize_exl3_impl(&gpu, &mut store, true, false, OFF).unwrap();
    assert_eq!(stats.kept_native, 0);
    assert_eq!(stats.quantized, 1);
    assert!(store.contains("model.layers.0.mlp.experts.0.gate_proj.weight"));
    assert!(!store.contains("model.layers.0.mlp.experts.0.gate_proj.trellis"));
}
