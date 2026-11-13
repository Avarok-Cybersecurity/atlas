// SPDX-License-Identifier: AGPL-3.0-only

//! The BF16 -> NVFP4 FFN route: its decision table clause by clause, the lookup discipline, the per-layer
//! qualification, the residency arithmetic, and the quantize-then-free install on the mock backend.

use super::*;
use crate::layers::ops::target_defaults::Source;
use crate::port::w4::W4Format;
use crate::weight_loader::qwen35_dense::w4_test_support::{ffn_store, lever, live};
use spark_runtime::gpu::mock::MockGpuBackend;

const ALL: RouteInputs = RouteInputs {
    lever: true,
    bf16_checkpoint: true,
    single_gpu: true,
    lora: false,
    kernels: true,
    fp8_ffn: false,
    affine_ffn: false,
};

#[test]
fn the_table_declines_clause_by_clause_in_order() {
    assert!(route_selected(ALL));
    assert_eq!(why_declined(ALL), None);
    let cases: [(RouteInputs, &str); 7] = [
        (
            RouteInputs {
                lever: false,
                ..ALL
            },
            "row off",
        ),
        (
            RouteInputs {
                bf16_checkpoint: false,
                ..ALL
            },
            "not a plain BF16 checkpoint",
        ),
        (
            RouteInputs {
                single_gpu: false,
                ..ALL
            },
            "tensor parallel",
        ),
        (
            RouteInputs { lora: true, ..ALL },
            "a LoRA adapter is pending",
        ),
        (
            RouteInputs {
                fp8_ffn: true,
                ..ALL
            },
            "the FP8 FFN route wins",
        ),
        (
            RouteInputs {
                affine_ffn: true,
                ..ALL
            },
            "the affine 4-bit FFN route wins",
        ),
        (
            RouteInputs {
                kernels: false,
                ..ALL
            },
            "the NVFP4 quantizer or GEMV family is missing from this build",
        ),
    ];
    for (i, want) in cases {
        assert!(!route_selected(i), "{want}");
        assert_eq!(why_declined(i), Some(want));
    }
    // The first failing clause is reported, not a later one.
    let two = RouteInputs {
        lever: true,
        bf16_checkpoint: false,
        fp8_ffn: true,
        ..ALL
    };
    assert_eq!(why_declined(two), Some("not a plain BF16 checkpoint"));
}

#[test]
fn the_lever_off_looks_nothing_up() {
    let gpu = MockGpuBackend::new();
    let (r, i) = Nvfp4FromBf16::resolve(&gpu, lever(false), true, 1, false, false, false);
    assert!(r.is_none());
    assert!(!i.lever);
    assert!(
        gpu.kernel_lookups_snapshot().is_empty(),
        "the boot audit sees no new lookup"
    );
}

#[test]
fn an_earlier_decline_looks_nothing_up() {
    for (bf16, tp, lora, fp8, affine) in [
        (false, 1, false, false, false),
        (true, 2, false, false, false),
        (true, 1, true, false, false),
        (true, 1, false, true, false),
        (true, 1, false, false, true),
    ] {
        let gpu = MockGpuBackend::new();
        let (r, _) = Nvfp4FromBf16::resolve(&gpu, lever(true), bf16, tp, lora, fp8, affine);
        assert!(r.is_none());
        assert!(
            gpu.kernel_lookups_snapshot().is_empty(),
            "bf16={bf16} tp={tp} lora={lora} fp8={fp8} affine={affine}"
        );
    }
}

#[test]
fn the_lever_on_resolves_the_quantizer_and_the_gemv_family() {
    let gpu = MockGpuBackend::new();
    let env_on = Resolved {
        value: true,
        source: Source::Env,
    };
    let (r, i) = Nvfp4FromBf16::resolve(&gpu, env_on, true, 1, false, false, false);
    assert!(r.is_some() && i.kernels);
    let seen = gpu.kernel_lookups_snapshot();
    for f in ["nvfp4_global_absmax", "quantize_bf16_to_nvfp4"] {
        assert!(
            seen.contains(&(QUANT_MODULE.to_owned(), f.to_owned())),
            "{f}"
        );
    }
    assert!(
        seen.iter()
            .any(|(m, f)| m == nvfp4::GEMV_MODULE && f == "w4a16_gemv")
    );

    // A build without the quantizer declines on the kernel clause.
    let bare = MockGpuBackend::new();
    bare.mark_module_absent(QUANT_MODULE);
    let (r, i) = Nvfp4FromBf16::resolve(&bare, lever(true), true, 1, false, false, false);
    assert!(r.is_none());
    assert_eq!(
        why_declined(i),
        Some("the NVFP4 quantizer or GEMV family is missing from this build")
    );
}

#[test]
fn the_residency_arithmetic_on_the_4b_and_0p8b() {
    // 4B: gate/up [9216, 2560], down [2560, 9216]; 4.5 bits a weight.
    assert_eq!(layer_device_bytes(2560, 9216), 39_813_120);
    assert_eq!(layer_freed_bytes(2560, 9216), 141_557_760 - 39_813_120);
    // 0.8B: hidden 1024, FFN 3584.
    assert_eq!(layer_device_bytes(1024, 3584), 6_193_152);
    assert_eq!(layer_freed_bytes(1024, 3584), 22_020_096 - 6_193_152);
}

#[test]
fn qualification_is_the_fp8_routes_shape_rule() {
    let gpu = MockGpuBackend::new();
    let store = ffn_store(&gpu, 1, 128, 256);
    assert!(layer_qualifies(&store, "model.layers.0", 128, 256));
    assert!(
        !layer_qualifies(&store, "model.layers.0", 128, 512),
        "shape differs from the config"
    );
    assert!(
        !layer_qualifies(&store, "model.layers.1", 128, 256),
        "layer not on the store"
    );
}

#[test]
fn quantize_frees_each_source_and_adopts_two_buffers_per_projection() {
    let gpu = MockGpuBackend::new();
    let (h, inter) = (128, 256);
    let store = ffn_store(&gpu, 1, h, inter);
    let lp = "model.layers.0";
    let src: Vec<_> = ["gate_proj", "up_proj", "down_proj"]
        .iter()
        .map(|p| store.get(&format!("{lp}.mlp.{p}.weight")).unwrap().ptr)
        .collect();
    let (route, _) = Nvfp4FromBf16::resolve(&gpu, lever(true), true, 1, false, false, false);
    let route = route.expect("on");
    let before = store.derived().len();
    let [g, u, d] = route.quantize_ffn(&store, lp, h, inter, &gpu, 0).unwrap();
    for p in &src {
        assert!(live(&gpu, *p).is_none(), "every BF16 source is freed");
    }
    assert_eq!(
        store.derived().len(),
        before + 6,
        "codes and scales of three projections"
    );
    assert_eq!(
        store.derived().bytes(),
        layer_device_bytes(h, inter),
        "the ledger owns exactly the NVFP4 bytes"
    );
    assert_eq!(
        (g.n(), g.k(), u.n(), u.k(), d.n(), d.k()),
        (inter, h, inter, h, h, inter)
    );
    for w in [&g, &u, &d] {
        assert_eq!(w.format(), W4Format::Nvfp4);
    }
    assert_eq!(
        g.device_bytes() + u.device_bytes() + d.device_bytes(),
        layer_device_bytes(h, inter)
    );
}

#[test]
fn ledger_labels_are_stable() {
    assert_eq!(LABEL_CODES, "ffn nvfp4 from bf16");
    assert_eq!(LABEL_SCALES, "ffn nvfp4 from bf16 group scale");
}
