// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the other-width install, untouched-layer, failed-quantize and rows-unset pins of the affine route.

use super::tests::{LP, bits, ffn_store, site, sources};
use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

#[test]
fn another_width_installs_through_lane_l17c_and_is_owned_by_the_ledger() {
    let gpu = MockGpuBackend::new();
    let (hidden, inter) = (256usize, 384usize);
    let store = ffn_store(&gpu, hidden, inter, WeightDtype::BF16);
    let src = sources(&store);
    let out = install_ffn(&bits(3).w4(), site(&gpu, &store, LP), hidden, inter)
        .unwrap()
        .unwrap();
    for w in &out {
        assert_eq!(w.format(), W4Format::Affine { bits: 3, group: 64 });
        let bufs = buffers_of(&**w);
        assert_eq!(bufs.len(), 3, "tiles and the two tables");
        assert_eq!(bufs.iter().map(|b| b.1).sum::<usize>(), w.device_bytes());
        // The prediction never under-states what the route keeps (it errs toward less credit).
        let predicted = super::super::w4_residency::weight_bytes(w.format(), w.n(), w.k());
        assert!(predicted >= w.device_bytes());
    }
    assert!(src.iter().all(|s| gpu.read_alloc(*s).is_none()));
    let kept: usize = out.iter().map(|w| w.device_bytes()).sum();
    assert_eq!(store.derived().bytes(), kept);
    store.derived().release(&gpu).expect("ledger release");
}

#[test]
fn a_layer_that_does_not_qualify_is_left_untouched() {
    let gpu = MockGpuBackend::new();
    for (store, hidden, inter) in [
        (ffn_store(&gpu, 96, 384, WeightDtype::BF16), 96, 384),
        (ffn_store(&gpu, 256, 384, WeightDtype::FP8E4M3), 256, 384),
        (ffn_store(&gpu, 256, 384, WeightDtype::BF16), 384, 256),
    ] {
        let src = sources(&store);
        let before = gpu.launches_snapshot().len();
        let r = install_ffn(&W4Opts::default(), site(&gpu, &store, LP), hidden, inter);
        assert!(r.unwrap().is_none());
        assert!(
            src.iter().all(|s| gpu.read_alloc(*s).is_some()),
            "nothing freed"
        );
        assert_eq!(gpu.launches_snapshot().len(), before, "nothing launched");
        assert!(store.derived().is_empty());
    }
    // A layer index the store does not have.
    let store = ffn_store(&gpu, 256, 384, WeightDtype::BF16);
    let r = install_ffn(
        &W4Opts::default(),
        site(&gpu, &store, "model.layers.9"),
        256,
        384,
    );
    assert!(r.unwrap().is_none());
}

#[test]
fn a_failed_quantize_names_the_tensor_and_the_override() {
    let gpu = MockGpuBackend::new();
    let store = ffn_store(&gpu, 256, 384, WeightDtype::BF16);
    // Group 16 divides both extents, so the layer qualifies, but the quantizer refuses the group.
    let opts = W4Opts {
        group: 16,
        ..W4Opts::default()
    };
    let e = install_ffn(&opts, site(&gpu, &store, LP), 256, 384).expect_err("not served");
    let text = format!("{e:#}");
    assert!(
        text.contains("model.layers.0.mlp.gate_proj.weight"),
        "{text}"
    );
    assert!(text.contains("AVAROK_DENSE_AFFINE4_FROM_BF16=0"), "{text}");
    // Nothing was freed: the first quantize failed before its free.
    assert!(sources(&store).iter().all(|s| gpu.read_alloc(*s).is_some()));
}

#[test]
fn with_the_rows_unset_every_route_is_off() {
    use crate::layers::ops::port_rows::{
        affine4_prefill_int8, attn_affine4_from_bf16, dense_affine4_from_bf16,
        gdn_affine4_from_bf16,
    };
    let envs = [
        dense_affine4_from_bf16::ENV,
        gdn_affine4_from_bf16::ENV,
        attn_affine4_from_bf16::ENV,
        affine4_prefill_int8::ENV,
    ];
    if envs.iter().any(|v| std::env::var_os(v).is_some()) {
        return; // the developer's shell set one; the table then is not the declaration's
    }
    let p = crate::layers::ops::target_defaults_port::resolved_port();
    assert!(!p.dense_affine4_from_bf16.value);
    assert!(!p.gdn_affine4_from_bf16.value);
    assert!(!p.attn_affine4_from_bf16.value);
    assert!(
        !p.affine4_prefill_int8.value,
        "off on every target at write time"
    );
    assert!(!crate::port::affine_opts::any_route_on());
}
