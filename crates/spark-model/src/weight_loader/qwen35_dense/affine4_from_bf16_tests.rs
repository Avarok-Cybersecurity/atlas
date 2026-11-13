// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the decision table, plan mapping and dense-FFN install of the affine route,
//! pinned clause by clause on the CPU and the mock backend.

use super::*;
use crate::port::affine::kernels::{LANE_MODULE, PACK_MODULE};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::WeightTensor;
use std::collections::HashMap;

const FACTS: LoadFacts = LoadFacts {
    bf16_checkpoint: true,
    tp_world_size: 1,
    lora: false,
    fp8_wins: false,
};

const ALL: AffineInputs = AffineInputs {
    lever: true,
    bf16_checkpoint: true,
    single_gpu: true,
    lora: false,
    fp8_wins: false,
    format: true,
    kernels: true,
};

pub(super) const LP: &str = "model.layers.0";

fn group(g: u16) -> AffineOpts {
    AffineOpts {
        group: g,
        ..AffineOpts::DEFAULT
    }
}

pub(super) fn bits(b: u8) -> AffineOpts {
    AffineOpts {
        bits: b,
        ..AffineOpts::DEFAULT
    }
}

pub(super) fn site<'a>(gpu: &'a MockGpuBackend, store: &'a WeightStore, lp: &'a str) -> Site<'a> {
    Site {
        gpu,
        store,
        stream: 0,
        lp,
    }
}

fn tensor(gpu: &MockGpuBackend, shape: [usize; 2], dtype: WeightDtype) -> WeightTensor {
    let per = if dtype == WeightDtype::BF16 { 2 } else { 1 };
    WeightTensor {
        ptr: gpu.alloc(shape[0] * shape[1] * per).expect("mock alloc"),
        shape: shape.to_vec(),
        dtype,
    }
}

/// Layer 0's FFN on a store: `[inter, hidden]` gate/up and `[hidden, inter]` down.
pub(super) fn ffn_store(
    gpu: &MockGpuBackend,
    hidden: usize,
    inter: usize,
    dtype: WeightDtype,
) -> WeightStore {
    WeightStore::from_map(HashMap::from([
        (
            format!("{LP}.mlp.gate_proj.weight"),
            tensor(gpu, [inter, hidden], dtype),
        ),
        (
            format!("{LP}.mlp.up_proj.weight"),
            tensor(gpu, [inter, hidden], dtype),
        ),
        (
            format!("{LP}.mlp.down_proj.weight"),
            tensor(gpu, [hidden, inter], dtype),
        ),
    ]))
}

pub(super) fn sources(store: &WeightStore) -> Vec<DevicePtr> {
    FFN_NAMES
        .iter()
        .map(|p| store.get(&format!("{LP}.mlp.{p}.weight")).unwrap().ptr)
        .collect()
}

#[test]
fn the_table_needs_every_clause_in_order() {
    assert_eq!(decide(ALL), Some(Ok(())));
    let off = AffineInputs {
        lever: false,
        ..ALL
    };
    assert_eq!(decide(off), None, "lever off is not a decline");
    let cases: [(Decline, AffineInputs); 6] = [
        (
            Decline::NotBf16,
            AffineInputs {
                bf16_checkpoint: false,
                ..ALL
            },
        ),
        (
            Decline::TensorParallel,
            AffineInputs {
                single_gpu: false,
                ..ALL
            },
        ),
        (Decline::Lora, AffineInputs { lora: true, ..ALL }),
        (
            Decline::Fp8Wins,
            AffineInputs {
                fp8_wins: true,
                ..ALL
            },
        ),
        (
            Decline::Format,
            AffineInputs {
                format: false,
                ..ALL
            },
        ),
        (
            Decline::Kernels,
            AffineInputs {
                kernels: false,
                ..ALL
            },
        ),
    ];
    for (want, i) in cases {
        assert_eq!(decide(i), Some(Err(want)), "{}", want.reason());
        assert!(
            !want.reason().contains('\u{2014}'),
            "no em dashes in boot text"
        );
    }
    // Two failing clauses report the earlier one: the boot line names the first thing to fix.
    let two = AffineInputs {
        lora: true,
        kernels: false,
        ..ALL
    };
    assert_eq!(decide(two), Some(Err(Decline::Lora)));
    let two = AffineInputs {
        fp8_wins: true,
        format: false,
        ..ALL
    };
    assert_eq!(decide(two), Some(Err(Decline::Fp8Wins)));
    let two = AffineInputs {
        bf16_checkpoint: false,
        single_gpu: false,
        ..ALL
    };
    assert_eq!(decide(two), Some(Err(Decline::NotBf16)));
}

#[test]
fn the_lever_off_looks_nothing_up() {
    let gpu = MockGpuBackend::new();
    for o in [AffineOpts::DEFAULT, bits(3), group(7)] {
        assert_eq!(verdict(&gpu, false, FACTS, o), Verdict::Off);
    }
    assert!(
        gpu.kernel_lookups_snapshot().is_empty(),
        "row off: the boot audit sees today's"
    );
}

#[test]
fn a_declaring_target_takes_the_route_for_a_single_gpu_bf16_checkpoint_only() {
    let gpu = MockGpuBackend::new();
    assert_eq!(verdict(&gpu, true, FACTS, AffineOpts::DEFAULT), Verdict::On);
    let lookups = gpu.kernel_lookups_snapshot();
    assert!(lookups.iter().any(|(m, _)| m == LANE_MODULE));
    assert!(lookups.iter().any(|(m, _)| m == PACK_MODULE));
    for (why, f, want) in [
        (
            "nothing BF16 to quantize",
            LoadFacts {
                bf16_checkpoint: false,
                ..FACTS
            },
            Decline::NotBf16,
        ),
        (
            "tensor parallel",
            LoadFacts {
                tp_world_size: 2,
                ..FACTS
            },
            Decline::TensorParallel,
        ),
        (
            "LoRA pending",
            LoadFacts {
                lora: true,
                ..FACTS
            },
            Decline::Lora,
        ),
    ] {
        let fresh = MockGpuBackend::new();
        let v = verdict(&fresh, true, f, AffineOpts::DEFAULT);
        assert_eq!(v, Verdict::Declined(want), "{why}");
        assert!(
            fresh.kernel_lookups_snapshot().is_empty(),
            "{why}: before the kernel clause"
        );
    }
    // A tensor-parallel size reported as 0 is a single GPU.
    let zero = LoadFacts {
        tp_world_size: 0,
        ..FACTS
    };
    assert_eq!(verdict(&gpu, true, zero, AffineOpts::DEFAULT), Verdict::On);
}

#[test]
fn fp8_wins_and_looks_nothing_up() {
    let gpu = MockGpuBackend::new();
    let f = LoadFacts {
        fp8_wins: true,
        ..FACTS
    };
    for o in [AffineOpts::DEFAULT, bits(3)] {
        assert_eq!(
            verdict(&gpu, true, f, o),
            Verdict::Declined(Decline::Fp8Wins)
        );
    }
    assert!(
        gpu.kernel_lookups_snapshot().is_empty(),
        "the kernel clause is not reached"
    );
}

#[test]
fn an_unserved_width_or_group_declines_before_any_lookup() {
    let gpu = MockGpuBackend::new();
    let three_at_32 = AffineOpts {
        bits: 3,
        group: 32,
        ..AffineOpts::DEFAULT
    };
    let deep_clip = AffineOpts {
        clip_search: 11,
        ..AffineOpts::DEFAULT
    };
    for o in [bits(7), group(16), three_at_32, deep_clip] {
        assert_eq!(
            verdict(&gpu, true, FACTS, o),
            Verdict::Declined(Decline::Format),
            "{o:?}"
        );
    }
    assert!(gpu.kernel_lookups_snapshot().is_empty());
}

#[test]
fn absent_kernels_decline_and_a_target_without_the_modules_looks_nothing_up() {
    for absent in [LANE_MODULE, PACK_MODULE] {
        let gpu = MockGpuBackend::new();
        gpu.mark_module_absent(absent);
        let v = verdict(&gpu, true, FACTS, AffineOpts::DEFAULT);
        assert_eq!(v, Verdict::Declined(Decline::Kernels), "{absent}");
        assert!(
            gpu.kernel_lookups_snapshot().is_empty(),
            "{absent} absent: has_module answers"
        );
    }
    // A module that is there but lacks one entry point (lane L17a's names) declines, never errors.
    for g in [32u16, 64, 128] {
        let gpu = MockGpuBackend::new();
        gpu.deny_kernel(LANE_MODULE, &format!("affine4_lane_gemv_g{g}"));
        let v = verdict(&gpu, true, FACTS, group(64));
        assert_eq!(v, Verdict::Declined(Decline::Kernels), "g{g} entry missing");
    }
    // Other widths need the widths module (lane L17c) and that width's entries.
    let gpu = MockGpuBackend::new();
    assert_eq!(verdict(&gpu, true, FACTS, bits(3)), Verdict::On);
    let want = (
        widths::kernels::MODULE.to_owned(),
        widths::kernels::gemv_name(3),
    );
    assert!(gpu.kernel_lookups_snapshot().contains(&want));
    let bare = MockGpuBackend::new();
    bare.mark_module_absent(widths::kernels::MODULE);
    assert_eq!(
        verdict(&bare, true, FACTS, bits(3)),
        Verdict::Declined(Decline::Kernels)
    );
    assert!(bare.kernel_lookups_snapshot().is_empty());
    let no_gemv = MockGpuBackend::new();
    no_gemv.deny_kernel(widths::kernels::MODULE, &widths::kernels::gemv_name(5));
    assert_eq!(
        verdict(&no_gemv, true, FACTS, bits(5)),
        Verdict::Declined(Decline::Kernels)
    );
}

#[test]
fn the_plan_carries_the_format_options_source_and_reason() {
    let on = route_plan(W4Role::Ffn, Verdict::On, Some(group(32)), true);
    assert_eq!(on.role, W4Role::Ffn);
    assert_eq!(on.format, Some(W4Format::Affine { bits: 4, group: 32 }));
    assert!(on.from_env);
    assert_eq!(on.why, "on");
    assert_eq!(on.opts, group(32).w4());
    assert_eq!(affine_opts_of(&on, W4Role::Ffn), Some(group(32).w4()));
    assert_eq!(
        affine_opts_of(&on, W4Role::Gdn),
        None,
        "another role's plan"
    );
    let int8 = AffineOpts {
        int8_prefill: true,
        ..AffineOpts::DEFAULT
    };
    let p = route_plan(W4Role::Attn, Verdict::On, Some(int8), false);
    assert!(affine_opts_of(&p, W4Role::Attn).is_some_and(|o| o.int8_prefill && o.bits == 4));
    let three = route_plan(W4Role::Gdn, Verdict::On, Some(bits(3)), false);
    assert_eq!(three.format, Some(W4Format::Affine { bits: 3, group: 64 }));

    let off = route_plan(W4Role::Ffn, Verdict::Off, None, false);
    assert_eq!(
        (off.format, off.why, off.opts),
        (None, ROW_OFF, W4Opts::default())
    );
    assert_eq!(
        off,
        RoutePlan::off(W4Role::Ffn),
        "the same inert plan lane L23b builds"
    );
    assert!(on.on() && !off.on());
    assert_eq!(affine_opts_of(&off, W4Role::Ffn), None);
    let d = Verdict::Declined(Decline::Fp8Wins);
    let fp8 = route_plan(W4Role::Gdn, d, Some(AffineOpts::DEFAULT), false);
    assert_eq!((fp8.format, fp8.why), (None, Decline::Fp8Wins.reason()));
    assert_eq!(affine_opts_of(&fp8, W4Role::Gdn), None);
    // A plan of another format (lane L23b's NVFP4 route) is never taken for an affine install.
    let base = route_plan(W4Role::Ffn, Verdict::On, None, false);
    let nv = RoutePlan {
        format: Some(W4Format::Nvfp4),
        ..base
    };
    assert_eq!(affine_opts_of(&nv, W4Role::Ffn), None);
}

#[test]
fn shapes_qualify_on_whole_groups_only() {
    assert!(shapes_ok(&ffn_projections(2560, 9216), 128));
    assert!(shapes_ok(&ffn_projections(96, 160), 32));
    assert!(
        !shapes_ok(&ffn_projections(96, 160), 64),
        "96 is not a whole 64-group"
    );
    assert!(!shapes_ok(&ffn_projections(0, 160), 32), "empty");
    assert!(!shapes_ok(&ffn_projections(2560, 9216), 0), "no group");
    assert!(!shapes_ok(&[], 64), "no projection");
    assert!(shapes_ok(&[(129, 256)], 64), "N needs no alignment");
}

#[test]
fn a_layer_qualifies_only_with_bf16_tensors_of_the_config_shape_and_whole_groups() {
    let gpu = MockGpuBackend::new();
    let bf16 = ffn_store(&gpu, 256, 384, WeightDtype::BF16);
    assert!(ffn_layer_ok(&bf16, LP, 256, 384, 64));
    assert!(ffn_layer_ok(&bf16, LP, 256, 384, 128));
    assert!(!ffn_layer_ok(&bf16, LP, 384, 256, 64), "transposed config");
    assert!(
        !ffn_layer_ok(&bf16, "model.layers.1", 256, 384, 64),
        "missing layer"
    );
    let fp8 = ffn_store(&gpu, 256, 384, WeightDtype::FP8E4M3);
    assert!(
        !ffn_layer_ok(&fp8, LP, 256, 384, 64),
        "already FP8: the native route's"
    );
    let odd = ffn_store(&gpu, 96, 384, WeightDtype::BF16);
    assert!(
        !ffn_layer_ok(&odd, LP, 96, 384, 64),
        "hidden 96 is not a whole 64-group"
    );
    assert!(ffn_layer_ok(&odd, LP, 96, 384, 32));
}

#[test]
fn install_quantizes_adopts_and_frees_each_projection() {
    let gpu = MockGpuBackend::new();
    let (hidden, inter) = (256usize, 384usize);
    let store = ffn_store(&gpu, hidden, inter, WeightDtype::BF16);
    let src = sources(&store);
    let live_before = gpu.live_alloc_count();
    let out = install_ffn(&W4Opts::default(), site(&gpu, &store, LP), hidden, inter)
        .expect("install")
        .expect("the layer qualifies");

    let mut total = 0;
    for (w, (n, k)) in out.iter().zip(ffn_projections(hidden, inter)) {
        assert_eq!((w.n(), w.k()), (n, k));
        assert_eq!(w.format(), W4Format::Affine { bits: 4, group: 64 });
        let bufs = buffers_of(&**w);
        assert_eq!(bufs.len(), 4, "codes, scales, biases, tickets");
        for (p, bytes) in &bufs {
            assert_eq!(
                gpu.read_alloc(*p).map(|d| d.len()),
                Some(*bytes),
                "buffer {p}"
            );
        }
        assert_eq!(bufs.iter().map(|b| b.1).sum::<usize>(), w.device_bytes());
        // Lane L23b's residency prediction is the bytes this install keeps (4 bits).
        let predicted = super::super::w4_residency::weight_bytes(w.format(), n, k);
        assert_eq!(predicted, w.device_bytes());
        total += w.device_bytes();
    }
    for s in src {
        assert!(
            gpu.read_alloc(s).is_none(),
            "BF16 source {s} still allocated"
        );
    }
    // Three sources gone, twelve weight buffers kept, no staging left behind.
    assert_eq!(gpu.live_alloc_count(), live_before - 3 + 12);
    let owned: HashMap<&str, usize> = store
        .derived()
        .by_label()
        .into_iter()
        .map(|(l, b, _)| (l, b))
        .collect();
    assert_eq!(
        owned.values().sum::<usize>(),
        total,
        "every buffer is owned by the ledger"
    );
    for l in labels(W4Role::Ffn) {
        assert!(owned.contains_key(l), "{l}");
    }
    // Teardown frees each buffer exactly once (the mock refuses a double free).
    store.derived().release(&gpu).expect("ledger release");
    assert_eq!(gpu.live_alloc_count(), live_before - 3);
    assert!(
        total < 3 * hidden * inter * 2,
        "the affine copy is smaller than its BF16"
    );
}
