// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::gpu::KernelHandle;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};
use std::collections::HashMap;

use crate::layers::ops::target_defaults::{Resolved, Source};
use crate::weight_map::WeightQuantFormat;

fn lever(on: bool) -> Resolved<bool> {
    Resolved {
        value: on,
        source: Source::Target,
    }
}

/// Holo 3.1 4B / Qwen3.5-4B GDN: `[Q|K|V|Z]` = 2 x 16 x 128 + 2 x 32 x 128
/// = 12,288 rows, hidden 2,560, value_dim 32 x 128 = 4,096.
const FOUR_B: GdnShapes = GdnShapes {
    qkvz: 12_288,
    hidden: 2_560,
    value: 4_096,
};

#[test]
fn a_target_that_does_not_declare_the_route_looks_nothing_up() {
    let gpu = MockGpuBackend::new();
    assert!(GdnFp8FromBf16::resolve(&gpu, lever(false), Nvfp4Variant::Bf16Raw, 1, 0).is_none());
    assert!(gpu.kernel_lookups_snapshot().is_empty());
}

#[test]
fn the_route_takes_a_single_gpu_bf16_checkpoint_only() {
    let gpu = MockGpuBackend::new();
    assert!(GdnFp8FromBf16::resolve(&gpu, lever(true), Nvfp4Variant::Bf16Raw, 1, 0).is_some());
    let lookups = gpu.kernel_lookups_snapshot();
    assert!(lookups.contains(&(QUANT_MODULE.to_owned(), QUANT_MODULE.to_owned())));
    for (why, variant, tp, lora) in [
        (
            "native FP8 has its own arm",
            Nvfp4Variant::Fp8Dequanted,
            1,
            0,
        ),
        ("tensor parallel", Nvfp4Variant::Bf16Raw, 2, 0),
        ("LoRA pending", Nvfp4Variant::Bf16Raw, 1, 16),
    ] {
        assert!(
            GdnFp8FromBf16::resolve(&gpu, lever(true), variant, tp, lora).is_none(),
            "{why} must decline"
        );
    }
    let no_quant = MockGpuBackend::new();
    no_quant.mark_module_absent(QUANT_MODULE);
    let no_gemv = MockGpuBackend::new();
    no_gemv.mark_module_absent("fp8_gemv_ada");
    no_gemv.deny_kernel("w8a16_gemv", "w8a16_gemv");
    for (absent, bare) in [("quantizer", no_quant), ("M=1 GEMV", no_gemv)] {
        assert!(
            GdnFp8FromBf16::resolve(&bare, lever(true), Nvfp4Variant::Bf16Raw, 1, 0).is_none(),
            "no {absent}, no route"
        );
    }
}

#[test]
fn a_layer_qualifies_only_on_whole_128_blocks() {
    assert!(FOUR_B.qualify());
    // Qwen3.5-0.8B: [Q|K|V|Z] 2 x 16 x 128 + 2 x 16 x 128 = 8,192, hidden
    // 1,024, value 2,048.
    assert!(
        GdnShapes {
            qkvz: 8_192,
            hidden: 1_024,
            value: 2_048
        }
        .qualify()
    );
    for odd in [
        GdnShapes {
            hidden: 2_500,
            ..FOUR_B
        },
        GdnShapes {
            value: 4_000,
            ..FOUR_B
        },
        GdnShapes { qkvz: 0, ..FOUR_B },
    ] {
        assert!(!odd.qualify(), "{odd:?}");
    }
}

#[test]
fn the_audit_credit_is_bf16_minus_fp8_and_scales_per_layer() {
    // BF16: (12,288 x 2,560 + 2,560 x 4,096) x 2 = 83,886,080 bytes.
    // FP8: 41,943,040 bytes plus a 96 x 20 and a 20 x 32 FP32 grid (10,240).
    assert_eq!(FOUR_B.freed_bytes(), 83_886_080 - 41_943_040 - 10_240);
    // 24 GDN layers on the 4B: 1.006 GB handed back during the build.
    assert_eq!(24 * FOUR_B.freed_bytes(), 1_006_387_200);
    assert_eq!(
        GdnShapes {
            hidden: 0,
            ..FOUR_B
        }
        .freed_bytes(),
        0
    );
}

#[test]
fn quantize_gdn_launches_the_quantizer_twice_and_frees_both_bf16_sources() {
    let gpu = MockGpuBackend::new();
    let shapes = GdnShapes {
        qkvz: 384,
        hidden: 256,
        value: 512,
    };
    let store = WeightStore::from_map(HashMap::new());
    let qkvz = DenseWeight {
        weight: gpu.alloc(shapes.qkvz * shapes.hidden * 2).unwrap(),
    };
    let out = DenseWeight {
        weight: gpu.alloc(shapes.hidden * shapes.value * 2).unwrap(),
    };
    let route = GdnFp8FromBf16 {
        quant_k: KernelHandle(0xF8),
    };
    let mut residency = DerivedResidency::default();
    let (q, o) = route
        .quantize_gdn(&store, qkvz, out, shapes, &gpu, 0, &mut residency)
        .expect("quantize_gdn");

    let want = [(shapes.qkvz, shapes.hidden), (shapes.hidden, shapes.value)];
    for (w, (n, k)) in [&q, &o].into_iter().zip(want) {
        assert_eq!((w.n as usize, w.k as usize), (n, k));
        // The tag `set_fp8_decode_weights` and every FP8 consumer assert on.
        assert_eq!(w.scale_format, WeightQuantFormat::Fp8BlockScaled);
        let (wb, sb) = fp8_weight_bytes(n, k);
        assert_eq!(gpu.read_alloc(w.weight).map(|d| d.len()), Some(wb));
        assert_eq!(gpu.read_alloc(w.row_scale).map(|d| d.len()), Some(sb));
    }
    let launches: Vec<_> = gpu
        .launches_snapshot()
        .into_iter()
        .filter(|l| l.func == 0xF8)
        .collect();
    assert_eq!(launches.len(), 2);
    for ((l, (n, k)), (src, w)) in launches
        .iter()
        .zip(want)
        .zip([(qkvz.weight, &q), (out.weight, &o)])
    {
        assert_eq!(l.grid, [k.div_ceil(128) as u32, n.div_ceil(128) as u32, 1]);
        assert_eq!(l.args[0], MockArg::Buffer(src));
        assert_eq!(l.args[1], MockArg::Buffer(w.weight));
        assert_eq!(l.args[2], MockArg::Buffer(w.row_scale));
    }
    for src in [qkvz.weight, out.weight] {
        assert!(
            gpu.read_alloc(src).is_none(),
            "BF16 source {src} still live"
        );
    }
    let total: usize = want
        .iter()
        .map(|&(n, k)| {
            let (wb, sb) = fp8_weight_bytes(n, k);
            wb + sb
        })
        .sum();
    assert_eq!(residency.kept, total as u64);
    let owned: usize = store.derived().by_label().iter().map(|(_, b, _)| b).sum();
    assert_eq!(owned, total);
    let bf16 = (shapes.qkvz * shapes.hidden + shapes.hidden * shapes.value) * 2;
    assert_eq!(shapes.freed_bytes(), bf16 - total);
}
