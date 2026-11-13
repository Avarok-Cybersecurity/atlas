// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: unit tests of `w4_proj.rs`: decode chunking, the shape rules, the install rules
//! of `try_set_w4_proj` / `set_w4_proj` on both loader layer shapes (BF16 still present, sources
//! already freed), and the scratch reservation.

use super::fixture::{Fakes, H, Rig, kv_dim, per_seq_qkv, q_dim, q_proj_dim};
use super::*;
use crate::port::w4::fake::FakeW4;
use crate::weight_map::{Fp8Weight, QuantWeight, QuantizedWeight, WeightQuantFormat};

#[test]
fn decode_chunks_cover_the_rows_in_windows() {
    let got: Vec<_> = decode_chunks(10, 4).collect();
    assert_eq!(got, vec![(0, 4), (4, 4), (8, 2)]);
    assert_eq!(decode_chunks(16, 16).collect::<Vec<_>>(), vec![(0, 16)]);
    assert_eq!(decode_chunks(1, 8).collect::<Vec<_>>(), vec![(0, 1)]);
    assert_eq!(decode_chunks(0, 8).count(), 0);
    // A zero window is treated as one row, never an infinite loop.
    assert_eq!(decode_chunks(3, 0).count(), 3);
    for rows in 1..70 {
        for max in [1usize, 3, 8, 16, 64] {
            let mut next = 0;
            for (first, len) in decode_chunks(rows, max) {
                assert_eq!(first, next);
                assert!((1..=max).contains(&len));
                next += len;
            }
            assert_eq!(next, rows);
        }
    }
}

#[test]
fn shape_refusal_checks_every_relation() {
    let l = crate::port::w4::fake::log();
    let fk = |n, k| FakeW4::new("w", n, k, 16, &l);
    let (q, k, v, o) = (fk(256, 128), fk(64, 128), fk(64, 128), fk(128, 128));
    assert!(
        shape_refusal(true, &q, &k, &v, &o).is_none(),
        "gated: q is twice o's input"
    );
    assert!(
        shape_refusal(false, &q, &k, &v, &o).is_some(),
        "ungated q must equal o's input"
    );
    let q1 = fk(128, 128);
    assert!(shape_refusal(false, &q1, &k, &v, &o).is_none());
    let bad_k = fk(64, 96);
    let e = shape_refusal(true, &q, &bad_k, &v, &o).unwrap();
    assert!(e.contains("input widths"), "{e}");
    let bad_v = fk(32, 128);
    assert!(
        shape_refusal(true, &q, &k, &bad_v, &o)
            .unwrap()
            .contains("v_proj")
    );
    let odd = fk(66, 128);
    let e = shape_refusal(true, &q, &odd, &odd, &o);
    assert!(e.is_none(), "n even and k % 8 is all alignment asks");
    let (q12, o12) = (fk(24, 12), fk(12, 12));
    let (k12, v12) = (fk(4, 12), fk(4, 12));
    assert!(
        shape_refusal(true, &q12, &k12, &v12, &o12)
            .unwrap()
            .contains("unaligned")
    );
    let zero = FakeW4::new("w", 64, 128, 0, &l);
    assert!(
        shape_refusal(true, &q, &zero, &v, &o)
            .unwrap()
            .contains("max_decode_rows")
    );
}

fn same_box(a: &W4Box, b: &std::sync::Arc<FakeW4>) -> bool {
    std::ptr::eq(
        std::sync::Arc::as_ptr(a) as *const (),
        std::sync::Arc::as_ptr(b) as *const (),
    )
}

#[test]
fn install_nulls_the_bf16_sources_and_reports_their_bytes() {
    let rig = Rig::new(true);
    let mut layer = rig.bf16_layer(true);
    assert!(
        !layer.w4_proj_installed(),
        "a fresh layer has no packed weights"
    );
    let fakes = Fakes::new(true, 16, false);
    let freed = fakes.install(&mut layer);
    let want = (q_proj_dim(true) * H + 2 * kv_dim() * H + H * q_dim()) * 2;
    assert_eq!(freed, want);
    assert!(layer.w4_proj_installed());
    assert!(layer.attn.q_proj.weight.is_null());
    assert!(layer.attn.k_proj.weight.is_null());
    assert!(layer.attn.v_proj.weight.is_null());
    assert!(layer.o_dense_bf16.is_none());
    let packed: usize = [&fakes.q, &fakes.k, &fakes.v, &fakes.o]
        .iter()
        .map(|f| f.device_bytes())
        .sum();
    assert_eq!(layer.w4_proj_device_bytes(), packed);
    // A second install is refused and changes nothing.
    let again = Fakes::new(true, 16, false);
    assert_eq!(again.install(&mut layer), 0);
    assert!(same_box(layer.w4_proj.get(W4Slot::Q).unwrap(), &fakes.q));
    assert!(same_box(layer.w4_proj.get(W4Slot::O).unwrap(), &fakes.o));
}

/// The loader's BF16 -> W4 route quantizes each source and frees it before the install (the
/// BF16 -> FP8 route's layer shape): every BF16 slot is NULL and no O-projection is installed.
/// That layer installs, and reports no bytes for the caller to free.
#[test]
fn a_layer_whose_sources_the_loader_freed_installs_and_frees_nothing() {
    let rig = Rig::new(true);
    let mut layer = rig.bf16_layer(true);
    let null = DenseWeight {
        weight: DevicePtr::NULL,
    };
    layer.attn.q_proj = null;
    layer.attn.k_proj = null;
    layer.attn.v_proj = null;
    layer.o_dense_bf16 = None;
    let fakes = Fakes::new(true, 16, false);
    let b = |f: &std::sync::Arc<FakeW4>| -> W4Box { f.clone() };
    let freed = layer
        .try_set_w4_proj(b(&fakes.q), b(&fakes.k), b(&fakes.v), b(&fakes.o))
        .unwrap();
    assert_eq!(freed, 0);
    assert!(layer.w4_proj_installed());

    // Only what is still present is reported: here the O-projection alone.
    let mut layer = rig.bf16_layer(true);
    layer.attn.q_proj = null;
    layer.attn.k_proj = null;
    layer.attn.v_proj = null;
    assert_eq!(
        Fakes::new(true, 16, false).install(&mut layer),
        H * q_dim() * 2
    );
}

#[test]
fn install_is_refused_on_a_layer_with_a_quantized_sidecar() {
    let rig = Rig::new(true);
    let mut layer = rig.bf16_layer(true);
    layer.o_weight = Some(QuantWeight::Fp8(Fp8Weight {
        weight: rig.gpu.alloc(H * q_dim()).unwrap(),
        row_scale: rig.gpu.alloc(4).unwrap(),
        n: H as u32,
        k: q_dim() as u32,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    }));
    let q_before = layer.attn.q_proj.weight;
    assert_eq!(Fakes::new(true, 16, false).install(&mut layer), 0);
    assert!(!layer.w4_proj_installed());
    assert_eq!(
        layer.attn.q_proj.weight, q_before,
        "a refused install keeps the BF16 weights"
    );
    assert!(layer.o_dense_bf16.is_some());

    // An NVFP4 o_proj in `attn` is a sidecar too.
    let mut layer = rig.bf16_layer(true);
    layer.attn.o_proj = QuantizedWeight {
        weight: rig.gpu.alloc(H * q_dim() / 2).unwrap(),
        ..QuantizedWeight::null()
    };
    let fakes = Fakes::new(true, 16, false);
    let b = |f: &std::sync::Arc<FakeW4>| -> W4Box { f.clone() };
    let e = layer
        .try_set_w4_proj(b(&fakes.q), b(&fakes.k), b(&fakes.v), b(&fakes.o))
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("quantized") && e.contains("attention layer 0"),
        "{e}"
    );
    assert!(!layer.w4_proj_installed());
}

#[test]
fn refusals_name_the_reason_and_leave_the_layer_unchanged() {
    let rig = Rig::new(true);
    let b = |f: &std::sync::Arc<FakeW4>| -> W4Box { f.clone() };
    let try_install = |layer: &mut Qwen3AttentionLayer, gated: bool| {
        let f = Fakes::new(gated, 16, false);
        layer.try_set_w4_proj(b(&f.q), b(&f.k), b(&f.v), b(&f.o))
    };
    // Ungated fakes on a gated layer: q is half the width the layer needs.
    let mut layer = rig.bf16_layer(true);
    let e = try_install(&mut layer, false).unwrap_err().to_string();
    assert!(e.contains("q_proj emits"), "{e}");
    assert!(!layer.w4_proj_installed() && !layer.attn.q_proj.weight.is_null());
    let mut layer = rig.bf16_layer(true);
    layer.k_eq_v = true;
    let e = try_install(&mut layer, true).unwrap_err().to_string();
    assert!(e.contains("K=V"), "{e}");
    assert!(layer.o_dense_bf16.is_some());
    // The 3.7.2 form turns every refusal into 0 bytes and an untouched layer.
    let mut layer = rig.bf16_layer(true);
    layer.k_eq_v = true;
    assert_eq!(Fakes::new(true, 16, false).install(&mut layer), 0);
    assert!(!layer.w4_proj_installed());
}

#[test]
fn prepare_reserves_once_and_an_empty_layer_reserves_nothing() {
    let rig = Rig::new(false);
    let empty = rig.bf16_layer(false);
    let allocs = rig.gpu.alloc_count();
    assert_eq!(empty.prepare_w4_proj(&rig.gpu, 0).unwrap(), 0);
    assert_eq!(rig.gpu.alloc_count(), allocs);
    assert!(
        empty.w4_ready(&rig.gpu, 0).is_err(),
        "no regions for a layer with nothing installed"
    );

    let mut layer = rig.bf16_layer(false);
    let fakes = Fakes::new(false, 16, false);
    fakes.install(&mut layer);
    let bytes = layer.prepare_w4_proj(&rig.gpu, 0).unwrap();
    let r = layer.w4_ready(&rig.gpu, 0).unwrap();
    assert_eq!(r.stage_bytes, STAGE_ROWS * per_seq_qkv(false));
    assert!(bytes >= r.stage_bytes + r.spec.device_bytes());
    let allocs = rig.gpu.alloc_count();
    assert_eq!(layer.prepare_w4_proj(&rig.gpu, 0).unwrap(), bytes);
    assert_eq!(layer.w4_ready(&rig.gpu, 0).unwrap(), r);
    assert_eq!(
        rig.gpu.alloc_count(),
        allocs,
        "a prepared layer never allocates again"
    );
    // A second layer of the same shape shares every region.
    let mut twin = rig.bf16_layer(false);
    Fakes::new(false, 16, false).install(&mut twin);
    // Building the twin allocates its own weights; only `prepare` must allocate nothing.
    let allocs = rig.gpu.alloc_count();
    twin.prepare_w4_proj(&rig.gpu, 0).unwrap();
    assert_eq!(rig.gpu.alloc_count(), allocs);
    assert_eq!(twin.w4_ready(&rig.gpu, 0).unwrap().stage, r.stage);
}
