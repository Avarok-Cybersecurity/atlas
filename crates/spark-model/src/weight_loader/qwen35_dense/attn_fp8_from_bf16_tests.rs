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

/// Holo 3.1 4B attention: 16 q heads x 256 gated (q_proj N = 8,192), 4 kv
/// heads x 256 (1,024), o_proj K = 4,096, hidden 2,560.
const FOUR_B: AttnShapes = AttnShapes {
    q_n: 8_192,
    kv_n: 1_024,
    o_k: 4_096,
    hidden: 2_560,
};

#[test]
fn a_target_that_does_not_declare_the_route_looks_nothing_up() {
    let gpu = MockGpuBackend::new();
    assert!(AttnFp8FromBf16::resolve(&gpu, lever(false), Nvfp4Variant::Bf16Raw, 1, 0).is_none());
    assert!(gpu.kernel_lookups_snapshot().is_empty());
}

#[test]
fn the_route_takes_a_single_gpu_bf16_checkpoint_only() {
    let gpu = MockGpuBackend::new();
    assert!(AttnFp8FromBf16::resolve(&gpu, lever(true), Nvfp4Variant::Bf16Raw, 1, 0).is_some());
    for (why, variant, tp, lora) in [
        (
            "native FP8 has its own overlay",
            Nvfp4Variant::Fp8Dequanted,
            1,
            0,
        ),
        ("tensor parallel", Nvfp4Variant::Bf16Raw, 2, 0),
        ("LoRA pending", Nvfp4Variant::Bf16Raw, 1, 16),
    ] {
        assert!(
            AttnFp8FromBf16::resolve(&gpu, lever(true), variant, tp, lora).is_none(),
            "{why} must decline"
        );
    }
    let no_quant = MockGpuBackend::new();
    no_quant.mark_module_absent(QUANT_MODULE);
    assert!(
        AttnFp8FromBf16::resolve(&no_quant, lever(true), Nvfp4Variant::Bf16Raw, 1, 0).is_none()
    );
}

#[test]
fn shapes_qualify_on_whole_128_blocks_only() {
    assert!(FOUR_B.qualify());
    // Holo 3.1 0.8B: 8 q heads x 256 gated, 2 kv heads x 256, hidden 1,024.
    assert!(
        AttnShapes {
            q_n: 4_096,
            kv_n: 512,
            o_k: 2_048,
            hidden: 1_024
        }
        .qualify()
    );
    assert!(
        !AttnShapes {
            kv_n: 576,
            ..FOUR_B
        }
        .qualify()
    );
    assert!(
        !AttnShapes {
            hidden: 0,
            ..FOUR_B
        }
        .qualify()
    );
}

#[test]
fn the_audit_credit_is_bf16_minus_fp8_and_scales_per_layer() {
    // BF16: (8,192 + 1,024 + 1,024) x 2,560 + 2,560 x 4,096 = 36,700,160
    // weights x 2 B. FP8: 36,700,160 B plus (64x20 + 8x20 + 8x20 + 20x32) x 4.
    assert_eq!(FOUR_B.freed_bytes(), 73_400_320 - 36_700_160 - 8_960);
    // 8 attention layers on the 4B: 0.29 GB handed back during the build.
    assert_eq!(8 * FOUR_B.freed_bytes(), 293_529_600);
}

#[test]
fn quantize_attn_launches_the_quantizer_per_projection_and_frees_every_source() {
    let gpu = MockGpuBackend::new();
    let shapes = AttnShapes {
        q_n: 512,
        kv_n: 128,
        o_k: 256,
        hidden: 384,
    };
    let store = WeightStore::from_map(HashMap::new());
    let src = shapes.projections().map(|(n, k)| DenseWeight {
        weight: gpu.alloc(n * k * 2).unwrap(),
    });
    let route = AttnFp8FromBf16 {
        quant_k: KernelHandle(0xF8),
    };
    let mut residency = DerivedResidency::default();
    let out = route
        .quantize_attn(&store, src, shapes, &gpu, 0, &mut residency)
        .expect("quantize_attn");

    let launches: Vec<_> = gpu
        .launches_snapshot()
        .into_iter()
        .filter(|l| l.func == 0xF8)
        .collect();
    assert_eq!(launches.len(), 4);
    let mut total = 0usize;
    for (((w, (n, k)), l), s) in out.iter().zip(shapes.projections()).zip(&launches).zip(src) {
        assert_eq!((w.n as usize, w.k as usize), (n, k));
        assert_eq!(w.scale_format, WeightQuantFormat::Fp8BlockScaled);
        assert_eq!(l.grid, [k.div_ceil(128) as u32, n.div_ceil(128) as u32, 1]);
        assert_eq!(l.args[0], MockArg::Buffer(s.weight));
        assert_eq!(l.args[1], MockArg::Buffer(w.weight));
        assert!(gpu.read_alloc(s.weight).is_none(), "BF16 source still live");
        let (wb, sb) = fp8_weight_bytes(n, k);
        total += wb + sb;
    }
    assert_eq!(residency.kept, total as u64);
    let owned: usize = store.derived().by_label().iter().map(|(_, b, _)| b).sum();
    assert_eq!(owned, total);
    let bf16: usize = shapes.projections().iter().map(|&(n, k)| n * k * 2).sum();
    assert_eq!(shapes.freed_bytes(), bf16 - total);
}
