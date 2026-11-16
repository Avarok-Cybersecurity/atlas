// SPDX-License-Identifier: AGPL-3.0-only

//! Exact-integer pins for the 4-bit residency arithmetic on the 4B and 0.8B geometries.

use super::*;
use crate::weight_loader::qwen35_dense::w4_test_support::{ffn_store, qwen35_0p8b, qwen35_4b};
use spark_runtime::gpu::mock::MockGpuBackend;

const AFFINE4: W4Format = W4Format::Affine { bits: 4, group: 64 };

#[test]
fn weight_bytes_follow_each_formats_layout() {
    // Affine 4-bit g64 on the 4B gate [9216, 2560]: codes 11,796,480 + scales and biases 2 x 737,280 +
    // 144 tile tickets x 4.
    assert_eq!(
        weight_bytes(AFFINE4, 9216, 2560),
        11_796_480 + 2 * 737_280 + 576
    );
    assert_eq!(
        weight_bytes(AFFINE4, 9216, 2560),
        crate::port::affine::Layout::of(9216, 2560, 64).total() + 576
    );
    // NVFP4: half a byte a weight plus one E4M3 scale per 16.
    assert_eq!(weight_bytes(W4Format::Nvfp4, 9216, 2560), 13_271_040);
    // GGUF blocks of 256: IQ4_XS 136 bytes, Q4_K 144, Q6_K 210.
    assert_eq!(weight_bytes(W4Format::GgufIq4Xs, 256, 512), 256 * 2 * 136);
    assert_eq!(weight_bytes(W4Format::GgufQ4K, 256, 512), 256 * 2 * 144);
    assert_eq!(weight_bytes(W4Format::GgufQ6K, 256, 512), 256 * 2 * 210);
    // Marlin GPTQ g128: int4 codes plus one f16 scale per group; AWQ adds packed int4 zero points.
    let gptq = W4Format::MarlinInt4 {
        group: 128,
        zero_point: false,
    };
    let awq = W4Format::MarlinInt4 {
        group: 128,
        zero_point: true,
    };
    assert_eq!(weight_bytes(gptq, 1024, 1024), 1024 * 512 + 8 * 1024 * 2);
    assert_eq!(
        weight_bytes(awq, 1024, 1024),
        weight_bytes(gptq, 1024, 1024) + 8 * 1024 / 2
    );
    // A 3-bit affine weight is cheaper than its 4-bit twin, a 8-bit one dearer.
    let three = W4Format::Affine { bits: 3, group: 64 };
    let eight = W4Format::Affine { bits: 8, group: 64 };
    assert!(weight_bytes(three, 9216, 2560) < weight_bytes(AFFINE4, 9216, 2560));
    assert!(weight_bytes(eight, 9216, 2560) > weight_bytes(AFFINE4, 9216, 2560));
}

#[test]
fn qualification_is_k_a_whole_number_of_groups() {
    assert!(qualifies(AFFINE4, &[(96, 128), (128, 64)]));
    assert!(
        !qualifies(AFFINE4, &[(96, 128), (128, 96)]),
        "K 96 is not a whole g64"
    );
    assert!(
        qualifies(W4Format::Nvfp4, &[(96, 48)]),
        "NVFP4 groups are 16"
    );
    assert!(!qualifies(W4Format::Nvfp4, &[(96, 40)]));
    assert!(!qualifies(W4Format::Nvfp4, &[]), "nothing to quantize");
    assert!(!qualifies(W4Format::Nvfp4, &[(0, 64)]), "empty extent");
}

#[test]
fn projections_are_the_loaders_shapes_on_the_4b() {
    let c = qwen35_4b();
    assert_eq!(
        projections(&c, W4Role::Ffn),
        vec![(9216, 2560), (9216, 2560), (2560, 9216)]
    );
    assert_eq!(
        projections(&c, W4Role::Gdn),
        vec![(12_288, 2560), (2560, 4096)]
    );
    assert_eq!(
        projections(&c, W4Role::Attn),
        vec![(8192, 2560), (1024, 2560), (1024, 2560), (2560, 4096)]
    );
    assert_eq!(
        (
            layers_of(&c, W4Role::Ffn),
            layers_of(&c, W4Role::Gdn),
            layers_of(&c, W4Role::Attn)
        ),
        (32, 24, 8)
    );
}

#[test]
fn the_4b_ffn_credit_per_layer_and_per_model() {
    let c = qwen35_4b();
    let ffn = projections(&c, W4Role::Ffn);
    assert_eq!(layer_bf16_bytes(&ffn), 141_557_760);
    // NVFP4: 3 x 13,271,040.
    assert_eq!(layer_w4_bytes(W4Format::Nvfp4, &ffn), 39_813_120);
    assert_eq!(layer_freed_bytes(W4Format::Nvfp4, &ffn), 101_744_640);
    assert_eq!(32 * layer_freed_bytes(W4Format::Nvfp4, &ffn), 3_255_828_480);
    // Affine 4-bit g64: the same 4.5 bits a weight plus 2 x 576 + 160 ticket bytes.
    assert_eq!(layer_w4_bytes(AFFINE4, &ffn), 39_814_432);
    assert_eq!(layer_freed_bytes(AFFINE4, &ffn), 101_743_328);
    // Both free more than the FP8 route (2.26 GB on the 4B) and agree with its arithmetic on BF16.
    assert!(32 * layer_freed_bytes(W4Format::Nvfp4, &ffn) > 2_264_371_200);
    assert_eq!(
        crate::weight_loader::qwen35_dense::nvfp4_from_bf16::layer_freed_bytes(2560, 9216),
        layer_freed_bytes(W4Format::Nvfp4, &ffn)
    );
}

#[test]
fn the_0p8b_ffn_and_gdn_credit() {
    let c = qwen35_0p8b();
    let ffn = projections(&c, W4Role::Ffn);
    assert_eq!(layer_bf16_bytes(&ffn), 22_020_096);
    assert_eq!(layer_w4_bytes(W4Format::Nvfp4, &ffn), 6_193_152);
    assert_eq!(24 * layer_freed_bytes(W4Format::Nvfp4, &ffn), 379_846_656);
    let gdn = projections(&c, W4Role::Gdn);
    assert_eq!(gdn, vec![(8192, 1024), (1024, 2048)]);
    assert_eq!(layers_of(&c, W4Role::Gdn), 18);
    assert!(qualifies(AFFINE4, &gdn));
}

#[test]
fn the_scratch_floor_is_one_set_of_regions_at_the_widest_shape() {
    let c = qwen35_4b();
    let ffn_only = W4Formats {
        ffn: Some(W4Format::Nvfp4),
        ..Default::default()
    };
    // xq 2048 x 9216 + xs 2048 x 576 x 4 + BF16 9216 x 2560 x 2 + 64 KiB tickets + one 256-byte ws.
    assert_eq!(
        scratch_floor_bytes(&c, ffn_only),
        18_874_368 + 4_718_592 + 47_185_920 + 65_536 + 256
    );
    assert_eq!(scratch_floor_bytes(&c, W4Formats::default()), 0);
    // Adding the GDN raises the floor to its widest matrix ([12288, 2560] BF16), not by a second set.
    let both = W4Formats {
        gdn: Some(W4Format::Nvfp4),
        ..ffn_only
    };
    assert_eq!(
        scratch_floor_bytes(&c, both),
        18_874_368 + 4_718_592 + 12_288 * 2560 * 2 + 65_536 + 256
    );
}

#[test]
fn the_pre_load_credit_is_every_layer_minus_the_scratch() {
    let c = qwen35_4b();
    let f = W4Formats {
        ffn: Some(W4Format::Nvfp4),
        ..Default::default()
    };
    assert_eq!(predicted_free_bytes_for(&c, f), 3_255_828_480 - 70_844_672);
    assert_eq!(
        predicted_free_bytes_for(&c, W4Formats::default()),
        0,
        "every row off: no credit"
    );
    let all = W4Formats {
        ffn: Some(AFFINE4),
        gdn: Some(AFFINE4),
        attn: Some(AFFINE4),
    };
    let want = 32 * layer_freed_bytes(AFFINE4, &projections(&c, W4Role::Ffn))
        + 24 * layer_freed_bytes(AFFINE4, &projections(&c, W4Role::Gdn))
        + 8 * layer_freed_bytes(AFFINE4, &projections(&c, W4Role::Attn))
        - scratch_floor_bytes(&c, all);
    assert_eq!(predicted_free_bytes_for(&c, all), want);
    // A group the shapes do not divide credits nothing for that layer type.
    let g128 = W4Format::Affine {
        bits: 4,
        group: 128,
    };
    let mut odd = c.clone();
    odd.hidden_size = 2560 + 64;
    let only_ffn = W4Formats {
        ffn: Some(g128),
        ..Default::default()
    };
    assert_eq!(predicted_free_bytes_for(&odd, only_ffn), 0);
}

#[test]
fn the_post_load_credit_counts_the_ffn_layers_the_store_holds() {
    let gpu = MockGpuBackend::new();
    let mut c = qwen35_4b();
    c.hidden_size = 1024;
    c.intermediate_size = 2048;
    c.num_hidden_layers = 3;
    c.layer_types.truncate(3);
    // Only layers 0 and 1 are on the store.
    let store = ffn_store(&gpu, 2, 1024, 2048);
    let f = W4Formats {
        ffn: Some(W4Format::Nvfp4),
        ..Default::default()
    };
    let per = layer_freed_bytes(W4Format::Nvfp4, &projections(&c, W4Role::Ffn));
    assert_eq!(per, 12_582_912 - 3_538_944);
    let floor = scratch_floor_bytes(&c, f);
    assert_eq!(floor, 4_194_304 + 1_048_576 + 4_194_304 + 65_536 + 256);
    assert_eq!(deferred_free_bytes_for(&store, &c, f), 2 * per - floor);
    assert_eq!(predicted_free_bytes_for(&c, f), 3 * per - floor);
}

#[test]
fn formats_iterate_in_role_order() {
    let f = W4Formats {
        ffn: None,
        gdn: Some(W4Format::Nvfp4),
        attn: Some(AFFINE4),
    };
    let on: Vec<_> = f.on().collect();
    assert_eq!(
        on,
        vec![(W4Role::Gdn, W4Format::Nvfp4), (W4Role::Attn, AFFINE4)]
    );
    assert!(f.any());
    assert!(!W4Formats::default().any());
}

#[test]
fn every_row_off_predicts_nothing_and_looks_nothing_up() {
    // The empty environment of a unit test leaves every port row at its baked (off) declaration.
    let gpu = MockGpuBackend::new();
    let c = qwen35_4b();
    assert_eq!(predicted_free_bytes(&gpu, &c, 1, false), 0);
    assert!(
        gpu.kernel_lookups_snapshot().is_empty(),
        "with every 4-bit row off the preflight credit issues no kernel lookup"
    );
}

#[test]
fn a_marlin_served_projection_keeps_the_marlin_layout_and_others_keep_theirs() {
    use crate::port::marlin::route::marlin_bytes;
    let sms = 34;
    // The 4B gate [9216, 2560] is a shape Marlin serves (pinned by `route_tests`): codes at 4.125 bits a
    // weight plus the locks, which is less than the affine layout the old prediction used.
    let marlin = projection_w4_bytes(AFFINE4, 9216, 2560, Some(sms));
    assert_eq!(marlin, marlin_bytes(9216, 2560, false, sms));
    assert!(marlin < weight_bytes(AFFINE4, 9216, 2560));
    // Without the row's SM count the affine arithmetic is untouched.
    assert_eq!(
        projection_w4_bytes(AFFINE4, 9216, 2560, None),
        weight_bytes(AFFINE4, 9216, 2560)
    );
    // A shape Marlin does not serve (N = 100 has no layout) falls back to the affine layout per projection.
    assert_eq!(
        projection_w4_bytes(AFFINE4, 100, 2560, Some(sms)),
        weight_bytes(AFFINE4, 100, 2560)
    );
    // Only affine 4-bit becomes Marlin: NVFP4 and the 3-bit widths keep their own arithmetic.
    let three = W4Format::Affine { bits: 3, group: 64 };
    assert_eq!(
        projection_w4_bytes(W4Format::Nvfp4, 9216, 2560, Some(sms)),
        weight_bytes(W4Format::Nvfp4, 9216, 2560)
    );
    assert_eq!(
        projection_w4_bytes(three, 9216, 2560, Some(sms)),
        weight_bytes(three, 9216, 2560)
    );
}

#[test]
fn the_marlin_credit_is_larger_than_the_affine_credit_by_the_layout_difference() {
    let c = qwen35_4b();
    let f = W4Formats {
        ffn: Some(AFFINE4),
        ..Default::default()
    };
    let sms = 34;
    let plain = predicted_free_bytes_for(&c, f);
    let marlin = predicted_free_bytes_with(&c, f, Some(sms));
    assert_eq!(
        predicted_free_bytes_with(&c, f, None),
        plain,
        "None is the affine prediction"
    );
    let projs = projections(&c, W4Role::Ffn);
    let per_layer =
        layer_freed_bytes_with(AFFINE4, &projs, Some(sms)) - layer_freed_bytes(AFFINE4, &projs);
    assert!(per_layer > 0, "Marlin keeps fewer bytes than affine g64");
    assert_eq!(marlin - plain, 32 * per_layer);
}

#[test]
fn the_marlin_row_off_adds_no_lookup_and_no_sm_query() {
    // The unit-test environment leaves `marlin_w4a16` at its baked (off) declaration.
    let gpu = MockGpuBackend::new();
    let f = W4Formats {
        ffn: Some(AFFINE4),
        ..Default::default()
    };
    assert_eq!(marlin_sms(&gpu, &f), None);
    assert!(gpu.kernel_lookups_snapshot().is_empty());
}
