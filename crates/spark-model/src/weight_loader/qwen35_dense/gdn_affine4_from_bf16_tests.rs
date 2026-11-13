// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the GDN affine route's per-layer table and its quantize-then-free install on the
//! mock backend (both sources freed, the loader's concat taken out of the ledger first).

use super::super::affine4_from_bf16::{
    LoadFacts, Verdict, buffers_of, labels, route_plan, verdict,
};
use super::*;
use crate::port::affine_opts::resolve;
use crate::port::w4::W4Format;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::WeightStore;
use std::collections::HashMap;

/// Holo 3.1 4B / Qwen3.5-4B GDN: `[Q|K|V|Z]` = 2 x 16 x 128 + 2 x 32 x 128 = 12,288 rows, hidden
/// 2,560, value 32 x 128 = 4,096.
const FOUR_B: GdnShapes = GdnShapes {
    qkvz: 12_288,
    hidden: 2_560,
    value: 4_096,
};

/// Qwen3.5-0.8B: `[Q|K|V|Z]` 8,192, hidden 1,024, value 2,048.
const POINT_8B: GdnShapes = GdnShapes {
    qkvz: 8_192,
    hidden: 1_024,
    value: 2_048,
};

/// A small GDN layer the mock can quantize quickly.
const SMALL: GdnShapes = GdnShapes {
    qkvz: 384,
    hidden: 128,
    value: 256,
};

const LP: &str = "model.layers.0";

fn site<'a>(gpu: &'a MockGpuBackend, store: &'a WeightStore) -> Site<'a> {
    Site {
        gpu,
        store,
        stream: 0,
        lp: LP,
    }
}

fn buffer(gpu: &MockGpuBackend, n: usize, k: usize) -> DenseWeight {
    DenseWeight {
        weight: gpu.alloc(n * k * 2).expect("mock alloc"),
    }
}

#[test]
fn the_projections_are_qkvz_then_out_proj() {
    assert_eq!(gdn_projections(FOUR_B), [(12_288, 2_560), (2_560, 4_096)]);
    assert_eq!(gdn_projections(POINT_8B), [(8_192, 1_024), (1_024, 2_048)]);
}

#[test]
fn a_layer_qualifies_on_whole_groups_only() {
    for g in [32, 64, 128] {
        assert!(layer_ok(FOUR_B, g), "4B at group {g}");
        assert!(layer_ok(POINT_8B, g), "0.8B at group {g}");
    }
    assert!(
        !layer_ok(
            GdnShapes {
                hidden: 2_560 + 32,
                ..FOUR_B
            },
            64
        ),
        "hidden is qkvz's K"
    );
    assert!(
        !layer_ok(
            GdnShapes {
                value: 4_096 + 16,
                ..FOUR_B
            },
            32
        ),
        "value is out_proj's K"
    );
    assert!(!layer_ok(GdnShapes { qkvz: 0, ..FOUR_B }, 64), "empty");
    // N needs no alignment: the tiles pad rows to 128.
    assert!(layer_ok(
        GdnShapes {
            qkvz: 12_288 + 8,
            ..FOUR_B
        },
        64
    ));
}

#[test]
fn the_gdn_group_override_reaches_the_gdn_plan_only() {
    // `AVAROK_W4_GDN_GROUP=32` (resolved by `port::affine_opts`, here through its pure resolver): the
    // GDN plan carries group 32 while the FFN plan keeps the common group.
    let env = |n: &str| (n == crate::port::affine_opts::ENV_GDN_GROUP).then(|| "32".to_string());
    let gdn = resolve(W4Role::Gdn, Default::default(), env, false).unwrap();
    let ffn = resolve(W4Role::Ffn, Default::default(), env, false).unwrap();
    let gpu = MockGpuBackend::new();
    let f = LoadFacts {
        bf16_checkpoint: true,
        tp_world_size: 1,
        lora: false,
        fp8_wins: false,
    };
    let p = route_plan(W4Role::Gdn, verdict(&gpu, true, f, gdn), Some(gdn), true);
    assert_eq!(p.format, Some(W4Format::Affine { bits: 4, group: 32 }));
    assert_eq!(affine_opts_of(&p, W4Role::Gdn).map(|o| o.group), Some(32));
    assert_eq!(ffn.group, 64);
    // FP8 GDN on wins and looks nothing up.
    let fresh = MockGpuBackend::new();
    let fp8 = verdict(
        &fresh,
        true,
        LoadFacts {
            fp8_wins: true,
            ..f
        },
        gdn,
    );
    assert!(matches!(fp8, Verdict::Declined(_)));
    assert!(fresh.kernel_lookups_snapshot().is_empty());
}

#[test]
fn install_quantizes_both_projections_and_frees_both_sources() {
    let gpu = MockGpuBackend::new();
    let store = WeightStore::from_map(HashMap::new());
    let qkvz = buffer(&gpu, SMALL.qkvz, SMALL.hidden);
    let out = buffer(&gpu, SMALL.hidden, SMALL.value);
    // The loader's concat may be in the ledger: the install must take it out before freeing it.
    store.derived().adopt(
        "ssm qkvz bf16 concat",
        qkvz.weight,
        SMALL.qkvz * SMALL.hidden * 2,
    );
    let live_before = gpu.live_alloc_count();
    let [q, o] = install_gdn(&W4Opts::default(), site(&gpu, &store), SMALL, qkvz, out)
        .expect("install")
        .expect("the layer qualifies");
    assert_eq!((q.n(), q.k()), (SMALL.qkvz, SMALL.hidden));
    assert_eq!((o.n(), o.k()), (SMALL.hidden, SMALL.value));
    assert!(
        gpu.read_alloc(qkvz.weight).is_none() && gpu.read_alloc(out.weight).is_none(),
        "both sources freed"
    );
    assert_eq!(
        gpu.live_alloc_count(),
        live_before - 2 + 8,
        "two sources gone, eight weight buffers kept"
    );
    let owned: HashMap<&str, usize> = store
        .derived()
        .by_label()
        .into_iter()
        .map(|(l, b, _)| (l, b))
        .collect();
    assert!(
        !owned.contains_key("ssm qkvz bf16 concat"),
        "the freed concat left the ledger"
    );
    assert_eq!(
        owned.values().sum::<usize>(),
        q.device_bytes() + o.device_bytes()
    );
    for l in labels(W4Role::Gdn) {
        assert!(owned.contains_key(l), "{l}");
    }
    for w in [&q, &o] {
        assert_eq!(
            buffers_of(&**w).iter().map(|b| b.1).sum::<usize>(),
            w.device_bytes()
        );
    }
    // Teardown frees each buffer once: no double free of the concat, no leak.
    store.derived().release(&gpu).expect("ledger release");
    assert_eq!(gpu.live_alloc_count(), live_before - 2);
}

#[test]
fn a_layer_that_does_not_qualify_touches_neither_source() {
    let gpu = MockGpuBackend::new();
    let store = WeightStore::from_map(HashMap::new());
    let odd = GdnShapes {
        hidden: 96,
        ..SMALL
    };
    let qkvz = buffer(&gpu, odd.qkvz, odd.hidden);
    let out = buffer(&gpu, odd.hidden, odd.value);
    let launches = gpu.launches_snapshot().len();
    assert!(
        install_gdn(&W4Opts::default(), site(&gpu, &store), odd, qkvz, out)
            .unwrap()
            .is_none()
    );
    assert!(gpu.read_alloc(qkvz.weight).is_some() && gpu.read_alloc(out.weight).is_some());
    assert_eq!(gpu.launches_snapshot().len(), launches);
    // Group 32 serves the same layer.
    let w = install_gdn(
        &W4Opts {
            group: 32,
            ..W4Opts::default()
        },
        site(&gpu, &store),
        odd,
        qkvz,
        out,
    )
    .unwrap();
    assert!(w.is_some_and(|[a, _]| a.format() == W4Format::Affine { bits: 4, group: 32 }));
}

#[test]
fn a_failed_quantize_names_the_concat_and_the_gdn_override() {
    let gpu = MockGpuBackend::new();
    let store = WeightStore::from_map(HashMap::new());
    let qkvz = buffer(&gpu, SMALL.qkvz, SMALL.hidden);
    let out = buffer(&gpu, SMALL.hidden, SMALL.value);
    let bad = W4Opts {
        group: 16,
        ..W4Opts::default()
    };
    let e = install_gdn(&bad, site(&gpu, &store), SMALL, qkvz, out)
        .expect_err("group 16 is not served");
    let text = format!("{e:#}");
    assert!(
        text.contains("in_proj_qkvz (concat)") && text.contains("AVAROK_GDN_AFFINE4_FROM_BF16=0"),
        "{text}"
    );
    assert!(gpu.read_alloc(qkvz.weight).is_some() && gpu.read_alloc(out.weight).is_some());
}
