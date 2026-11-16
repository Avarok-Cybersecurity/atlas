// SPDX-License-Identifier: AGPL-3.0-only

//! The 4-bit install plan and its per-layer FFN install on the mock backend: route precedence, the boot
//! lines, the row-off path (no lookup, nothing touched), the NVFP4 FFN install order with its self-check
//! and residency tally, and the `/metrics` records.

use super::super::nvfp4_from_bf16::{Nvfp4FromBf16, RouteInputs as Nvfp4Inputs};
use super::super::w4_residency::W4Formats;
use super::*;
use crate::port::w4::fake::{FakeW4, log};
use crate::weight_loader::qwen35_dense::w4_test_support::{ffn_store, lever, live, qwen35_4b};
use spark_runtime::gpu::mock::MockGpuBackend;
use std::sync::Arc;

const H: usize = 128;
const I: usize = 256;

fn small_config() -> ModelConfig {
    let mut c = qwen35_4b();
    c.hidden_size = H;
    c.intermediate_size = I;
    c
}

fn on(role: W4Role, format: W4Format) -> RoutePlan {
    RoutePlan {
        format: Some(format),
        why: "on",
        ..RoutePlan::off(role)
    }
}

const AFFINE4: W4Format = W4Format::Affine { bits: 4, group: 64 };

#[test]
fn a_plan_is_on_exactly_when_it_has_a_format() {
    let off = RoutePlan::off(W4Role::Gdn);
    assert!(!off.on());
    assert_eq!(
        (off.role, off.format, off.why, off.from_env),
        (W4Role::Gdn, None, ROW_OFF, false)
    );
    assert!(on(W4Role::Ffn, W4Format::Nvfp4).on());
}

#[test]
fn the_nvfp4_table_maps_onto_a_plan() {
    let all = Nvfp4Inputs {
        lever: true,
        bf16_checkpoint: true,
        single_gpu: true,
        lora: false,
        kernels: true,
        fp8_ffn: false,
        affine_ffn: false,
    };
    let p = nvfp4_plan(true, all);
    assert_eq!(
        (p.role, p.format, p.why, p.from_env),
        (W4Role::Ffn, Some(W4Format::Nvfp4), "on", true)
    );
    let p = nvfp4_plan(
        false,
        Nvfp4Inputs {
            affine_ffn: true,
            ..all
        },
    );
    assert_eq!((p.format, p.why), (None, "the affine 4-bit FFN route wins"));
    let p = nvfp4_plan(
        false,
        Nvfp4Inputs {
            lever: false,
            ..all
        },
    );
    assert_eq!(p.why, ROW_OFF, "an off row prints no boot line");
}

#[test]
fn the_affine_ffn_route_wins_and_formats_follow_the_plans() {
    let mut p = LoadPlans::off();
    assert_eq!(p.formats(), W4Formats::default());
    p.ffn_nvfp4 = on(W4Role::Ffn, W4Format::Nvfp4);
    assert_eq!(p.ffn().format, Some(W4Format::Nvfp4));
    p.ffn_affine = on(W4Role::Ffn, AFFINE4);
    assert_eq!(p.ffn().format, Some(AFFINE4), "affine wins over NVFP4");
    p.gdn = on(W4Role::Gdn, W4Format::Affine { bits: 4, group: 32 });
    assert_eq!(
        p.formats(),
        W4Formats {
            ffn: Some(AFFINE4),
            gdn: Some(W4Format::Affine { bits: 4, group: 32 }),
            attn: None,
        }
    );
    let rows: Vec<_> = p.routes().iter().map(|(r, _)| r.row()).collect();
    assert_eq!(
        rows,
        [
            "dense_affine4_from_bf16",
            "dense_nvfp4_from_bf16",
            "gdn_affine4_from_bf16",
            "attn_affine4_from_bf16"
        ]
    );
}

#[test]
fn every_route_names_its_registry_env() {
    assert_eq!(Route::FfnAffine.env(), "AVAROK_DENSE_AFFINE4_FROM_BF16");
    assert_eq!(Route::FfnNvfp4.env(), "AVAROK_DENSE_NVFP4_FROM_BF16");
    assert_eq!(Route::GdnAffine.env(), "AVAROK_GDN_AFFINE4_FROM_BF16");
    assert_eq!(Route::AttnAffine.env(), "AVAROK_ATTN_AFFINE4_FROM_BF16");
}

#[test]
fn boot_lines_say_on_or_why_and_the_source() {
    assert_eq!(
        boot_line(Route::FfnNvfp4, &RoutePlan::off(W4Role::Ffn)),
        None
    );
    let mut p = on(W4Role::Ffn, W4Format::Nvfp4);
    p.from_env = true;
    let l = boot_line(Route::FfnNvfp4, &p).unwrap();
    assert_eq!(
        l,
        "[avarok] 4-bit route dense_nvfp4_from_bf16: ON, nvfp4 (row on from the environment). \
         AVAROK_DENSE_NVFP4_FROM_BF16=0 turns it off."
    );
    let d = RoutePlan {
        why: "tensor parallel",
        ..RoutePlan::off(W4Role::Attn)
    };
    let l = boot_line(Route::AttnAffine, &d).unwrap();
    assert!(
        l.contains("declined, tensor parallel") && l.contains("target declaration"),
        "{l}"
    );
    assert!(!l.contains('\u{2014}'));
}

#[test]
fn every_row_off_plans_nothing_looks_nothing_up_and_touches_nothing() {
    // The empty environment of a unit test leaves every 4-bit row at its baked (off) declaration.
    let gpu = MockGpuBackend::new();
    let config = small_config();
    let store = ffn_store(&gpu, 1, H, I);
    let plans = LoadPlans::plan(&gpu, true, 1, false, Fp8Wins::default());
    assert_eq!(plans.formats(), W4Formats::default());
    assert!(
        plans
            .routes()
            .iter()
            .all(|(r, p)| boot_line(*r, p).is_none())
    );
    let mut w4 = W4Load::resolve(
        &gpu,
        &store,
        &config,
        Nvfp4Variant::Bf16Raw,
        0,
        Fp8Wins::default(),
    );
    let mut res = DerivedResidency::default();
    assert!(
        w4.ffn_boxes("model.layers.0", 0, &mut res)
            .unwrap()
            .is_none()
    );
    assert!(!w4.ffn_packed("model.layers.0"));
    assert!(
        gpu.kernel_lookups_snapshot().is_empty(),
        "the boot audit is unchanged"
    );
    let gate = store
        .get("model.layers.0.mlp.gate_proj.weight")
        .unwrap()
        .ptr;
    assert!(live(&gpu, gate).is_some(), "the BF16 tensors stay");
    assert_eq!(res.kept, 0);
    assert!(w4.records().iter().all(|r| r.format.is_none()));
    assert!(plan_formats(&gpu, &config, true, 1, false) == W4Formats::default());
}

fn nvfp4_load<'a>(
    gpu: &'a MockGpuBackend,
    store: &'a WeightStore,
    config: &'a ModelConfig,
) -> W4Load<'a> {
    let (route, _) = Nvfp4FromBf16::resolve(gpu, lever(true), true, 1, false, false, false);
    let plans = LoadPlans {
        ffn_nvfp4: on(W4Role::Ffn, W4Format::Nvfp4),
        nvfp4: Some(route.expect("the mock resolves every kernel")),
        ..LoadPlans::off()
    };
    W4Load::with_plans(gpu, store, config, 0, plans)
}

#[test]
fn the_nvfp4_ffn_install_quantizes_frees_checks_and_tallies() {
    let gpu = MockGpuBackend::new();
    let config = small_config();
    let store = ffn_store(&gpu, 2, H, I);
    let mut w4 = nvfp4_load(&gpu, &store, &config);
    let mut res = DerivedResidency::default();
    let src: Vec<_> = (0..2)
        .flat_map(|l| {
            ["gate_proj", "up_proj", "down_proj"].map(|p| {
                store
                    .get(&format!("model.layers.{l}.mlp.{p}.weight"))
                    .unwrap()
                    .ptr
            })
        })
        .collect();

    let [g, u, d] = w4
        .ffn_boxes("model.layers.0", 0, &mut res)
        .unwrap()
        .expect("layer 0 takes the route");
    assert_eq!(
        (g.n(), g.k(), u.n(), u.k(), d.n(), d.k()),
        (I, H, I, H, H, I)
    );
    for p in &src[..3] {
        assert!(live(&gpu, *p).is_none(), "layer 0's BF16 sources are freed");
    }
    for p in &src[3..] {
        assert!(
            live(&gpu, *p).is_some(),
            "layer 1 is untouched until its turn"
        );
    }
    let per_layer = nvfp4_from_bf16::layer_device_bytes(H, I) as u64;
    assert_eq!(res.kept, per_layer, "the NVFP4 bytes are kept residency");
    assert!(
        w4.checks.checked(W4Role::Ffn) && !w4.checks.per_row(W4Role::Ffn),
        "self-check ran and passed"
    );

    w4.ffn_boxes("model.layers.1", 1, &mut res)
        .unwrap()
        .expect("layer 1 too");
    assert_eq!(res.kept, 2 * per_layer);
    let t = w4.tally[slot(W4Role::Ffn)];
    assert_eq!(
        t,
        Tally {
            format: Some(W4Format::Nvfp4),
            layers: 2,
            bytes: 2 * per_layer,
            bf16_freed: 2 * (3 * H * I * 2) as u64,
        }
    );
    let recs = w4.records();
    assert_eq!(recs.len(), W4Role::ALL.len());
    let ffn = recs.iter().find(|r| r.role == W4Role::Ffn).unwrap();
    assert_eq!(
        (ffn.format, ffn.layers, ffn.bytes, ffn.per_row),
        (Some(W4Format::Nvfp4), 2, 2 * per_layer, false)
    );
    assert!(
        recs.iter()
            .filter(|r| r.role != W4Role::Ffn)
            .all(|r| r.format.is_none())
    );
}

#[test]
fn a_layer_that_does_not_qualify_keeps_its_bf16_path() {
    let gpu = MockGpuBackend::new();
    let config = small_config();
    let store = ffn_store(&gpu, 1, H, I);
    let mut w4 = nvfp4_load(&gpu, &store, &config);
    let mut res = DerivedResidency::default();
    assert!(
        w4.ffn_boxes("model.layers.7", 7, &mut res)
            .unwrap()
            .is_none(),
        "not on the store"
    );
    assert_eq!(res.kept, 0);
    assert_eq!(w4.tally[slot(W4Role::Ffn)].layers, 0);
}

#[test]
fn the_tally_counts_device_bytes_and_only_from_bf16_sources_as_freed() {
    let l = log();
    let a: W4Box = Arc::new(FakeW4::new("a", 64, 32, 16, &l));
    let b: W4Box = Arc::new(FakeW4::new("b", 32, 64, 16, &l));
    let mut t = Tally::default();
    t.add(&[a.clone(), b.clone()], true);
    t.add(&[a, b], false);
    assert_eq!(t.layers, 2);
    assert_eq!(t.format, Some(AFFINE4), "the fakes report affine4/g64");
    assert_eq!(
        t.bf16_freed,
        (64 * 32 + 32 * 64) * 2,
        "only the first set came from BF16"
    );
}

#[test]
fn summary_lines_name_the_format_layers_and_bytes() {
    let t = Tally {
        format: Some(W4Format::Nvfp4),
        layers: 32,
        bytes: 1_274_019_840,
        bf16_freed: 4_529_848_320,
    };
    assert_eq!(
        summary_line(W4Role::Ffn, &t, false),
        "[avarok] 4-bit ffn: nvfp4 on 32 layers, 1.27 GB resident, 4.53 GB of BF16 replaced"
    );
    let packed = Tally {
        format: Some(W4Format::GgufQ4K),
        bf16_freed: 0,
        ..t
    };
    let l = summary_line(W4Role::Ffn, &packed, true);
    assert!(
        l.contains("kept packed from the checkpoint") && l.contains("one row at a time"),
        "{l}"
    );
    assert!(!l.contains('\u{2014}'));
}

#[test]
fn slots_are_distinct_per_role() {
    let s: Vec<_> = W4Role::ALL.iter().map(|&r| slot(r)).collect();
    assert_eq!(s, [0, 1, 2, 3]);
}
