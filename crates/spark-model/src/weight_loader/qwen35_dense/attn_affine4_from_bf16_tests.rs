// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the attention affine route's per-layer table and its quantize-then-free install on
//! the mock backend (q, k, v, o in order, every source freed, every buffer owned by the ledger).

use super::super::affine4_from_bf16::{
    LoadFacts, Verdict, buffers_of, labels, route_plan, verdict,
};
use super::*;
use crate::port::affine_opts::AffineOpts;
use crate::port::w4::W4Format;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::WeightStore;
use std::collections::HashMap;

/// Holo 3.1 4B / Qwen3.5-4B attention: 16 heads x 256 with a gated q (8,192 rows), 4 KV heads (1,024
/// rows), hidden 2,560, o's K 4,096.
const FOUR_B: AttnShapes = AttnShapes {
    q_n: 8_192,
    kv_n: 1_024,
    o_k: 4_096,
    hidden: 2_560,
};

/// A small attention layer the mock quantizes quickly.
const SMALL: AttnShapes = AttnShapes {
    q_n: 512,
    kv_n: 128,
    o_k: 256,
    hidden: 128,
};

const LP: &str = "model.layers.3";

fn site<'a>(gpu: &'a MockGpuBackend, store: &'a WeightStore) -> Site<'a> {
    Site {
        gpu,
        store,
        stream: 0,
        lp: LP,
    }
}

fn sources(gpu: &MockGpuBackend, s: AttnShapes) -> [DenseWeight; 4] {
    s.projections().map(|(n, k)| DenseWeight {
        weight: gpu.alloc(n * k * 2).expect("mock alloc"),
    })
}

#[test]
fn a_layer_qualifies_on_whole_groups_only() {
    for g in [32, 64, 128] {
        assert!(layer_ok(FOUR_B, g), "4B at group {g}");
    }
    assert!(
        !layer_ok(
            AttnShapes {
                hidden: 2_560 + 32,
                ..FOUR_B
            },
            64
        ),
        "hidden is q/k/v's K"
    );
    assert!(
        !layer_ok(
            AttnShapes {
                o_k: 4_096 + 16,
                ..FOUR_B
            },
            32
        ),
        "heads x head_dim is o's K"
    );
    assert!(!layer_ok(AttnShapes { kv_n: 0, ..FOUR_B }, 64), "empty");
    // N needs no alignment: the tiles pad rows to 128.
    assert!(layer_ok(
        AttnShapes {
            kv_n: 1_024 + 8,
            ..FOUR_B
        },
        64
    ));
}

#[test]
fn the_attention_plan_is_its_own_lever_and_fp8_wins() {
    let gpu = MockGpuBackend::new();
    let f = LoadFacts {
        bf16_checkpoint: true,
        tp_world_size: 1,
        lora: false,
        fp8_wins: false,
    };
    let on = route_plan(
        W4Role::Attn,
        verdict(&gpu, true, f, AffineOpts::DEFAULT),
        Some(AffineOpts::DEFAULT),
        false,
    );
    assert_eq!(
        (on.role, on.format),
        (W4Role::Attn, Some(W4Format::Affine { bits: 4, group: 64 }))
    );
    assert!(
        affine_opts_of(&on, W4Role::Attn).is_some() && affine_opts_of(&on, W4Role::Ffn).is_none()
    );
    let fresh = MockGpuBackend::new();
    let fp8 = verdict(
        &fresh,
        true,
        LoadFacts {
            fp8_wins: true,
            ..f
        },
        AffineOpts::DEFAULT,
    );
    assert!(
        matches!(fp8, Verdict::Declined(_)),
        "attn_fp8_from_bf16 on wins"
    );
    assert!(fresh.kernel_lookups_snapshot().is_empty());
    assert_eq!(verdict(&fresh, false, f, AffineOpts::DEFAULT), Verdict::Off);
}

#[test]
fn install_quantizes_q_k_v_o_in_order_and_frees_every_source() {
    let gpu = MockGpuBackend::new();
    let store = WeightStore::from_map(HashMap::new());
    let srcs = sources(&gpu, SMALL);
    let live_before = gpu.live_alloc_count();
    let four = install_attn(&W4Opts::default(), site(&gpu, &store), SMALL, srcs)
        .expect("install")
        .expect("the layer qualifies");
    for (w, (n, k)) in four.iter().zip(SMALL.projections()) {
        assert_eq!(
            (w.n(), w.k(), w.format()),
            (n, k, W4Format::Affine { bits: 4, group: 64 })
        );
        assert_eq!(
            buffers_of(&**w).iter().map(|b| b.1).sum::<usize>(),
            w.device_bytes()
        );
    }
    assert!(
        srcs.iter().all(|s| gpu.read_alloc(s.weight).is_none()),
        "every source freed"
    );
    assert_eq!(gpu.live_alloc_count(), live_before - 4 + 16);
    let owned: HashMap<&str, usize> = store
        .derived()
        .by_label()
        .into_iter()
        .map(|(l, b, _)| (l, b))
        .collect();
    assert_eq!(
        owned.values().sum::<usize>(),
        four.iter().map(|w| w.device_bytes()).sum::<usize>()
    );
    for l in labels(W4Role::Attn) {
        assert!(owned.contains_key(l), "{l}");
    }
    store.derived().release(&gpu).expect("ledger release");
    assert_eq!(gpu.live_alloc_count(), live_before - 4);
}

#[test]
fn a_layer_that_does_not_qualify_touches_no_source() {
    let gpu = MockGpuBackend::new();
    let store = WeightStore::from_map(HashMap::new());
    let odd = AttnShapes {
        hidden: 96,
        ..SMALL
    };
    let srcs = sources(&gpu, odd);
    let launches = gpu.launches_snapshot().len();
    assert!(
        install_attn(&W4Opts::default(), site(&gpu, &store), odd, srcs)
            .unwrap()
            .is_none()
    );
    assert!(srcs.iter().all(|s| gpu.read_alloc(s.weight).is_some()));
    assert_eq!(gpu.launches_snapshot().len(), launches);
    assert!(store.derived().is_empty());
}

#[test]
fn a_failed_quantize_names_the_tensor_and_the_attention_override() {
    let gpu = MockGpuBackend::new();
    let store = WeightStore::from_map(HashMap::new());
    let srcs = sources(&gpu, SMALL);
    let bad = W4Opts {
        group: 16,
        ..W4Opts::default()
    };
    let e =
        install_attn(&bad, site(&gpu, &store), SMALL, srcs).expect_err("group 16 is not served");
    let text = format!("{e:#}");
    assert!(
        text.contains("model.layers.3.self_attn.q_proj.weight")
            && text.contains("AVAROK_ATTN_AFFINE4_FROM_BF16=0"),
        "{text}"
    );
    assert!(
        srcs.iter().all(|s| gpu.read_alloc(s.weight).is_some()),
        "the first quantize failed before any free"
    );
}
