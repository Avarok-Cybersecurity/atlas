// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::gpu::KernelHandle;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};
use spark_runtime::weights::WeightTensor;
use std::collections::HashMap;

use crate::layers::ops::target_defaults::{Resolved, Source};
use crate::weight_map::WeightQuantFormat;

fn lever(on: bool) -> Resolved<bool> {
    Resolved {
        value: on,
        source: Source::Target,
    }
}

#[test]
fn a_target_that_does_not_declare_the_route_looks_nothing_up() {
    let gpu = MockGpuBackend::new();
    assert!(Fp8FromBf16::resolve(&gpu, lever(false), Nvfp4Variant::Bf16Raw, 1, 0).is_none());
    assert!(
        gpu.kernel_lookups_snapshot().is_empty(),
        "the boot audit must see the same lookups as before on every other target"
    );
}

#[test]
fn a_declaring_target_takes_the_route_for_a_bf16_checkpoint_only() {
    let gpu = MockGpuBackend::new();
    assert!(Fp8FromBf16::resolve(&gpu, lever(true), Nvfp4Variant::Bf16Raw, 1, 0).is_some());
    assert!(
        gpu.kernel_lookups_snapshot()
            .contains(&(QUANT_MODULE.to_owned(), QUANT_MODULE.to_owned()))
    );
    assert!(Fp8FromBf16::resolve(&gpu, lever(true), Nvfp4Variant::Fp8Dequanted, 1, 0).is_none());
    assert!(Fp8FromBf16::resolve(&gpu, lever(true), Nvfp4Variant::Bf16Raw, 2, 0).is_none());
    assert!(Fp8FromBf16::resolve(&gpu, lever(true), Nvfp4Variant::Bf16Raw, 1, 16).is_none());
    let bare = MockGpuBackend::new();
    bare.mark_module_absent(QUANT_MODULE);
    assert!(
        Fp8FromBf16::resolve(&bare, lever(true), Nvfp4Variant::Bf16Raw, 1, 0).is_none(),
        "no quantizer in the tree, no route"
    );
}

const ALL: RouteInputs = RouteInputs {
    lever: true,
    bf16_checkpoint: true,
    single_gpu: true,
    lora: false,
    kernels: true,
};

#[test]
fn the_route_needs_every_clause() {
    assert!(route_selected(ALL));
    let off: [(&str, RouteInputs); 5] = [
        (
            "lever off",
            RouteInputs {
                lever: false,
                ..ALL
            },
        ),
        (
            "not a BF16 checkpoint",
            RouteInputs {
                bf16_checkpoint: false,
                ..ALL
            },
        ),
        (
            "tensor parallel",
            RouteInputs {
                single_gpu: false,
                ..ALL
            },
        ),
        ("LoRA pending", RouteInputs { lora: true, ..ALL }),
        (
            "kernels missing",
            RouteInputs {
                kernels: false,
                ..ALL
            },
        ),
    ];
    for (why, i) in off {
        assert!(!route_selected(i), "{why} must decline the route");
    }
}

fn tensor(gpu: &MockGpuBackend, shape: [usize; 2], dtype: WeightDtype) -> WeightTensor {
    let bytes = shape[0] * shape[1] * if dtype == WeightDtype::BF16 { 2 } else { 1 };
    WeightTensor {
        ptr: gpu.alloc(bytes).expect("mock alloc"),
        shape: shape.to_vec(),
        dtype,
    }
}

/// One dense layer's FFN on the store, `[inter, hidden]` gate/up and
/// `[hidden, inter]` down, all of `dtype`.
fn ffn_store(gpu: &MockGpuBackend, hidden: usize, inter: usize, dtype: WeightDtype) -> WeightStore {
    let lp = "model.layers.0";
    WeightStore::from_map(HashMap::from([
        (
            format!("{lp}.mlp.gate_proj.weight"),
            tensor(gpu, [inter, hidden], dtype),
        ),
        (
            format!("{lp}.mlp.up_proj.weight"),
            tensor(gpu, [inter, hidden], dtype),
        ),
        (
            format!("{lp}.mlp.down_proj.weight"),
            tensor(gpu, [hidden, inter], dtype),
        ),
    ]))
}

#[test]
fn a_layer_qualifies_only_with_bf16_tensors_of_the_config_shape() {
    let gpu = MockGpuBackend::new();
    let lp = "model.layers.0";
    let bf16 = ffn_store(&gpu, 256, 384, WeightDtype::BF16);
    assert!(layer_shapes_ok(&bf16, lp, 256, 384));
    assert!(!layer_shapes_ok(&bf16, lp, 384, 256), "transposed config");
    assert!(
        !layer_shapes_ok(&bf16, "model.layers.1", 256, 384),
        "missing layer"
    );
    let fp8 = ffn_store(&gpu, 256, 384, WeightDtype::FP8E4M3);
    assert!(
        !layer_shapes_ok(&fp8, lp, 256, 384),
        "already FP8: the native route's"
    );
    let odd = ffn_store(&gpu, 250, 384, WeightDtype::BF16);
    assert!(
        !layer_shapes_ok(&odd, lp, 250, 384),
        "hidden not a whole GEMV chunk"
    );
}

#[test]
fn quantize_ffn_launches_the_block_quantizer_per_projection_and_frees_the_bf16_source() {
    let gpu = MockGpuBackend::new();
    let (hidden, inter) = (256usize, 384usize);
    let store = ffn_store(&gpu, hidden, inter, WeightDtype::BF16);
    let lp = "model.layers.0";
    let src: Vec<_> = ["gate_proj", "up_proj", "down_proj"]
        .iter()
        .map(|p| store.get(&format!("{lp}.mlp.{p}.weight")).unwrap().ptr)
        .collect();

    let route = Fp8FromBf16 {
        quant_k: KernelHandle(0xF8),
    };
    let mut residency = DerivedResidency::default();
    let out = route
        .quantize_ffn(&store, lp, hidden, inter, &gpu, 0, &mut residency)
        .expect("quantize_ffn");

    // Shapes and the format tag every consumer asserts on.
    let want = [(inter, hidden), (inter, hidden), (hidden, inter)];
    for (w, (n, k)) in out.iter().zip(want) {
        assert_eq!((w.n as usize, w.k as usize), (n, k));
        assert_eq!(w.scale_format, WeightQuantFormat::Fp8BlockScaled);
        let (wb, sb) = fp8_weight_bytes(n, k);
        assert_eq!(gpu.read_alloc(w.weight).map(|d| d.len()), Some(wb));
        assert_eq!(gpu.read_alloc(w.row_scale).map(|d| d.len()), Some(sb));
    }

    // One launch per projection, on the quantizer's (k_blocks, n_blocks) grid,
    // reading the BF16 source and writing the two fresh buffers.
    let launches: Vec<_> = gpu
        .launches_snapshot()
        .into_iter()
        .filter(|l| l.func == 0xF8)
        .collect();
    assert_eq!(launches.len(), 3);
    for ((l, (n, k)), (s, w)) in launches.iter().zip(want).zip(src.iter().zip(&out)) {
        assert_eq!(l.grid, [k.div_ceil(128) as u32, n.div_ceil(128) as u32, 1]);
        assert_eq!(l.block, [256, 1, 1]);
        assert_eq!(l.args[0], MockArg::Buffer(*s));
        assert_eq!(l.args[1], MockArg::Buffer(w.weight));
        assert_eq!(l.args[2], MockArg::Buffer(w.row_scale));
    }

    // The BF16 sources are gone; the FP8 copies are owned by the ledger and
    // counted as kept.
    for s in src {
        assert!(
            gpu.read_alloc(s).is_none(),
            "BF16 source {s} still allocated"
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
    // Half the bytes of the BF16 weights, plus the scale grid.
    let bf16_bytes = 3 * hidden * inter * 2;
    assert!(total * 2 > bf16_bytes && total * 2 < bf16_bytes + bf16_bytes / 50);
}
