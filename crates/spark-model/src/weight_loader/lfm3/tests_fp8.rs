// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from HuggingFace transformers (models/lfm3, Apache-2.0, Copyright 2026 The HuggingFace Team) and vLLM (vllm/model_executor/models/lfm3.py, Apache-2.0, Copyright contributors to the vLLM project). Changes: ported from Python/Triton to Rust/CUDA for Atlas.

//! Responsibility: projection loading on the mock backend: BF16 passthrough,
//! the opt-in BF16 -> FP8 conversion (launch shape, source freed, ledger),
//! Liquid's native E4M3 projections with their scale grids, and the all-or-
//! nothing groups.

use spark_runtime::gpu::KernelHandle;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};
use spark_runtime::weights::WeightDtype;

use super::fp8::{Class, Fp8Loader, Group, Proj, Routes, fp8_weight_bytes};
use super::tests::{QUANT, bf16, fp8_tensor, quantize_launches, store_of};

#[test]
fn a_bf16_projection_with_the_route_off_is_a_zero_copy_pointer() {
    let gpu = MockGpuBackend::new();
    let t = bf16(&gpu, &[256, 128]);
    let ptr = t.ptr;
    let store = store_of(vec![("p.weight", t)]);
    let l = Fp8Loader::with_routes(&store, &gpu, Routes::default(), QUANT);
    match l.load(Class::Ffn, "p", 256, 128).unwrap() {
        Proj::Bf16(w) => assert_eq!(w.weight, ptr),
        Proj::Fp8(_) => panic!("route is off"),
    }
    assert!(quantize_launches(&gpu).is_empty());
}

#[test]
fn a_routed_projection_quantizes_once_frees_its_source_and_owns_the_result() {
    let gpu = MockGpuBackend::new();
    let t = bf16(&gpu, &[256, 128]);
    let src = t.ptr;
    let store = store_of(vec![("p.weight", t)]);
    let routes = Routes {
        ffn: true,
        attn: false,
        conv: false,
    };
    let l = Fp8Loader::with_routes(&store, &gpu, routes, QUANT);
    let Proj::Fp8(w) = l.load(Class::Ffn, "p", 256, 128).unwrap() else {
        panic!("routed projection must be E4M3");
    };
    assert_eq!((w.n, w.k), (256, 128));
    let launches = quantize_launches(&gpu);
    assert_eq!(launches.len(), 1);
    assert_eq!(launches[0].grid, [1, 2, 1], "[k/128, n/128, 1]");
    assert_eq!(launches[0].args[0], MockArg::Buffer(src));
    assert_eq!(launches[0].args[1], MockArg::Buffer(w.weight));
    assert!(gpu.read_alloc(src).is_none(), "BF16 source still live");
    let (wb, sb) = fp8_weight_bytes(256, 128);
    let owned: usize = store.derived().by_label().iter().map(|(_, b, _)| b).sum();
    assert_eq!(owned, wb + sb);

    // A class the route does not cover stays BF16.
    let store2 = store_of(vec![("q.weight", bf16(&gpu, &[256, 128]))]);
    let l2 = Fp8Loader::with_routes(&store2, &gpu, routes, QUANT);
    assert!(!l2.load(Class::Attn, "q", 256, 128).unwrap().is_fp8());
}

#[test]
fn an_extent_that_is_not_a_128_block_stays_bf16_even_when_routed() {
    let gpu = MockGpuBackend::new();
    let store = store_of(vec![("g.weight", bf16(&gpu, &[16, 256]))]);
    let routes = Routes {
        ffn: true,
        attn: true,
        conv: true,
    };
    let l = Fp8Loader::with_routes(&store, &gpu, routes, QUANT);
    assert!(!l.load(Class::Attn, "g", 16, 256).unwrap().is_fp8());
}

#[test]
fn a_projection_of_the_wrong_shape_or_dtype_is_refused() {
    let gpu = MockGpuBackend::new();
    let mut fp32 = bf16(&gpu, &[128, 128]);
    fp32.dtype = WeightDtype::FP32;
    let store = store_of(vec![
        ("a.weight", bf16(&gpu, &[128, 256])),
        ("b.weight", fp32),
    ]);
    let l = Fp8Loader::with_routes(&store, &gpu, Routes::default(), QUANT);
    let e = l.load(Class::Ffn, "a", 128, 128).err().unwrap().to_string();
    assert!(e.contains("[128, 256]") && e.contains("[128, 128]"), "{e}");
    let e = l.load(Class::Ffn, "b", 128, 128).err().unwrap().to_string();
    assert!(e.contains("unsupported projection dtype"), "{e}");
}

#[test]
fn the_vendor_e4m3_projection_stays_native_with_its_scale_grid_widened() {
    let gpu = MockGpuBackend::new();
    let w = fp8_tensor(&gpu, 256, 256);
    let ptr = w.ptr;
    let store = store_of(vec![
        ("p.weight", w),
        ("p.weight_scale", bf16(&gpu, &[2, 2])),
    ]);
    let l = Fp8Loader::with_routes(&store, &gpu, Routes::default(), QUANT);
    let Proj::Fp8(got) = l.load(Class::Ffn, "p", 256, 256).unwrap() else {
        panic!("vendor E4M3 must load native");
    };
    assert_eq!(got.weight, ptr, "the E4M3 bytes are the store's, uncopied");
    assert_eq!((got.n, got.k), (256, 256));
    assert!(
        quantize_launches(&gpu).is_empty(),
        "nothing is re-quantized"
    );
    let rows = store.derived().by_label();
    assert!(
        rows.iter().any(|(label, bytes, _)| {
            *label == "lfm3 fp8 block scale (widened)" && *bytes == 2 * 2 * 4
        }),
        "{rows:?}"
    );
}

#[test]
fn a_vendor_projection_with_a_missing_or_malformed_scale_is_refused() {
    let gpu = MockGpuBackend::new();
    let store = store_of(vec![
        ("a.weight", fp8_tensor(&gpu, 256, 256)),
        ("b.weight", fp8_tensor(&gpu, 256, 256)),
        ("b.weight_scale", bf16(&gpu, &[256])),
    ]);
    let l = Fp8Loader::with_routes(&store, &gpu, Routes::default(), QUANT);
    let e = l.load(Class::Ffn, "a", 256, 256).err().unwrap().to_string();
    assert!(e.contains("no block scale"), "{e}");
    let e = l.load(Class::Ffn, "b", 256, 256).err().unwrap().to_string();
    assert!(e.contains("block scale shape"), "{e}");
}

#[test]
fn a_mixed_group_converts_its_bf16_members_or_fails_loudly() {
    let gpu = MockGpuBackend::new();
    let store = store_of(vec![
        ("w1.weight", fp8_tensor(&gpu, 256, 128)),
        ("w1.weight_scale", bf16(&gpu, &[2, 1])),
        ("w3.weight", bf16(&gpu, &[256, 128])),
    ]);
    let l = Fp8Loader::with_routes(&store, &gpu, Routes::default(), QUANT);
    let Group::Fp8([a, b]) = l
        .load_group(Class::Ffn, [("w1", 256, 128), ("w3", 256, 128)])
        .unwrap()
    else {
        panic!("a group with an E4M3 member is FP8");
    };
    assert_eq!((a.n, b.n), (256, 256));
    assert_eq!(quantize_launches(&gpu).len(), 1, "only the BF16 member");

    let store = store_of(vec![
        ("w1.weight", fp8_tensor(&gpu, 256, 128)),
        ("w1.weight_scale", bf16(&gpu, &[2, 1])),
        ("w3.weight", bf16(&gpu, &[256, 128])),
    ]);
    let no_quant = Fp8Loader::with_routes(&store, &gpu, Routes::default(), KernelHandle(0));
    let e = no_quant
        .load_group(Class::Ffn, [("w1", 256, 128), ("w3", 256, 128)])
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("w3") && e.contains("unavailable"), "{e}");
}

#[test]
fn an_all_bf16_group_with_the_route_off_is_the_checkpoints_own_bytes() {
    let gpu = MockGpuBackend::new();
    let (a, b) = (bf16(&gpu, &[128, 128]), bf16(&gpu, &[128, 128]));
    let (pa, pb) = (a.ptr, b.ptr);
    let store = store_of(vec![("a.weight", a), ("b.weight", b)]);
    let l = Fp8Loader::with_routes(&store, &gpu, Routes::default(), QUANT);
    let Group::Bf16([ga, gb]) = l
        .load_group(Class::Attn, [("a", 128, 128), ("b", 128, 128)])
        .unwrap()
    else {
        panic!("route off keeps BF16");
    };
    assert_eq!((ga.weight, gb.weight), (pa, pb));
    assert!(quantize_launches(&gpu).is_empty());
}

// ── the conv pair (lane L38's layer keeps BF16 for prefill) ──

const H: usize = 256;

fn conv_store(gpu: &MockGpuBackend) -> spark_runtime::weights::WeightStore {
    store_of(vec![
        ("c.in_proj.weight", bf16(gpu, &[3 * H, H])),
        ("c.out_proj.weight", bf16(gpu, &[H, H])),
    ])
}

#[test]
fn the_conv_pair_is_plain_bf16_with_the_route_off() {
    let gpu = MockGpuBackend::new();
    let store = conv_store(&gpu);
    let l = Fp8Loader::with_routes(&store, &gpu, Routes::default(), QUANT);
    let p = l.load_conv_pair("c.in_proj", "c.out_proj", H).unwrap();
    assert!(p.fp8.is_none());
    assert_eq!(p.in_bf16.weight, store.get("c.in_proj.weight").unwrap().ptr);
    assert_eq!(
        p.out_bf16.weight,
        store.get("c.out_proj.weight").unwrap().ptr
    );
    assert!(quantize_launches(&gpu).is_empty());
}

#[test]
fn the_conv_route_adds_decode_copies_and_keeps_the_bf16_sources() {
    let gpu = MockGpuBackend::new();
    let store = conv_store(&gpu);
    let in_src = store.get("c.in_proj.weight").unwrap().ptr;
    let out_src = store.get("c.out_proj.weight").unwrap().ptr;
    let routes = Routes {
        ffn: false,
        attn: false,
        conv: true,
    };
    let l = Fp8Loader::with_routes(&store, &gpu, routes, QUANT);
    let p = l.load_conv_pair("c.in_proj", "c.out_proj", H).unwrap();
    let (fi, fo) = p.fp8.expect("both E4M3 copies");
    assert_eq!((fi.n as usize, fi.k as usize), (3 * H, H));
    assert_eq!((fo.n as usize, fo.k as usize), (H, H));
    assert_eq!(quantize_launches(&gpu).len(), 2);
    // Prefill reads BF16, so neither source is freed.
    assert_eq!((p.in_bf16.weight, p.out_bf16.weight), (in_src, out_src));
    assert!(gpu.read_alloc(in_src).is_some() && gpu.read_alloc(out_src).is_some());
    // The ledger owns both copies and their scale grids.
    let owned: usize = store.derived().by_label().iter().map(|(_, b, _)| b).sum();
    let (a, b) = (fp8_weight_bytes(3 * H, H), fp8_weight_bytes(H, H));
    assert_eq!(owned, a.0 + a.1 + b.0 + b.1);
    // The ffn and attention classes alone leave the convs untouched.
    let store2 = conv_store(&gpu);
    let only = Routes {
        ffn: true,
        attn: true,
        conv: false,
    };
    let l2 = Fp8Loader::with_routes(&store2, &gpu, only, QUANT);
    assert!(
        l2.load_conv_pair("c.in_proj", "c.out_proj", H)
            .unwrap()
            .fp8
            .is_none()
    );
}

#[test]
fn a_conv_projection_of_the_wrong_shape_is_refused() {
    let gpu = MockGpuBackend::new();
    let store = store_of(vec![
        ("c.in_proj.weight", bf16(&gpu, &[2 * H, H])),
        ("c.out_proj.weight", bf16(&gpu, &[H, H])),
    ]);
    let l = Fp8Loader::with_routes(&store, &gpu, Routes::default(), QUANT);
    let e = l
        .load_conv_pair("c.in_proj", "c.out_proj", H)
        .err()
        .unwrap()
        .to_string();
    assert!(
        e.contains("c.in_proj.weight") && e.contains("expected [768, 256]"),
        "{e}"
    );
}
