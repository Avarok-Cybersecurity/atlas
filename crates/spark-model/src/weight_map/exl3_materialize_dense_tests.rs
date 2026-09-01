// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for `exl3_materialize_dense.rs` (split out on the 500-LoC cap).

use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::exl3::Exl3Codebook;

use super::super::exl3_materialize_dense::*;

fn w(k_bits: u32, cb: Exl3Codebook, in_dim: usize, out_dim: usize) -> Exl3Weight {
    Exl3Weight {
        trellis: DevicePtr(16),
        suh: DevicePtr(32),
        svh: DevicePtr(48),
        in_dim,
        out_dim,
        k_bits,
        cb,
    }
}

#[test]
fn gate_combinations() {
    // OFF everywhere is fine; DENSE needs NATIVE.
    assert!(check_exl3_native_dense_gates(false, false, None, None).is_ok());
    assert!(check_exl3_native_dense_gates(true, false, None, None).is_ok());
    assert!(check_exl3_native_dense_gates(true, true, None, None).is_ok());
    assert!(check_exl3_native_dense_gates(false, true, None, None).is_err());
    // Sub-gates refine DENSE only.
    assert!(check_exl3_native_dense_gates(true, true, Some("0"), None).is_ok());
    assert!(check_exl3_native_dense_gates(true, true, Some("1"), None).is_ok());
    assert!(check_exl3_native_dense_gates(true, true, None, Some("0")).is_ok());
    // Both families are routed, so an explicit opt-in to either is honored.
    assert!(Exl3DenseFamily::Attn.routed());
    assert!(check_exl3_native_dense_gates(true, true, None, Some("1")).is_ok());
    assert!(check_exl3_native_dense_gates(true, false, Some("0"), None).is_err());
    assert!(check_exl3_native_dense_gates(true, false, None, Some("1")).is_err());
    assert!(check_exl3_native_dense_gates(true, true, Some("yes"), None).is_err());
}

#[test]
fn families_from_env_values() {
    assert_eq!(
        exl3_native_dense_families_with(false, Some("1"), Some("1")),
        Exl3DenseFamilies::OFF
    );
    assert_eq!(
        exl3_native_dense_families_with(true, None, None),
        Exl3DenseFamilies::ALL
    );
    assert_eq!(
        exl3_native_dense_families_with(true, Some("0"), None),
        Exl3DenseFamilies {
            gdn: false,
            attn: true
        }
    );
    assert_eq!(
        exl3_native_dense_families_with(true, None, Some("0")),
        Exl3DenseFamilies {
            gdn: true,
            attn: false
        }
    );
    assert!(!Exl3DenseFamilies::OFF.any());
}

#[test]
fn routed_leaves_are_a_subset_of_the_family() {
    for f in [Exl3DenseFamily::Gdn, Exl3DenseFamily::Attn] {
        for leaf in f.leaves() {
            assert!(f.all_leaves().contains(leaf), "{f:?} {leaf}");
        }
    }
    // Milestone scope: the whole GDN family and the whole attention family.
    assert_eq!(
        Exl3DenseFamily::Gdn.leaves(),
        &["in_proj_qkv", "in_proj_z", "out_proj"]
    );
    assert!(Exl3DenseFamily::Gdn.routed());
    assert_eq!(
        Exl3DenseFamily::Attn.leaves(),
        &["q_proj", "k_proj", "v_proj", "o_proj"]
    );
    assert!(Exl3DenseFamily::Attn.routed());
    assert_eq!(Exl3DenseFamily::Gdn.all_leaves().len(), 3);
    assert_eq!(Exl3DenseFamily::Attn.all_leaves().len(), 4);
}

#[test]
fn prefix_classification() {
    let lp = "model.language_model.layers.7";
    for leaf in ["in_proj_qkv", "in_proj_z", "out_proj"] {
        assert_eq!(
            exl3_dense_prefix_family(&format!("{lp}.linear_attn.{leaf}")),
            Some((lp, Exl3DenseFamily::Gdn)),
            "{leaf}"
        );
    }
    for leaf in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        assert_eq!(
            exl3_dense_prefix_family(&format!("{lp}.self_attn.{leaf}")),
            Some((lp, Exl3DenseFamily::Attn)),
            "{leaf}"
        );
    }
    // Not in the set: QSA indexer, BA projections, conv, experts, shared
    // expert, lm_head, ViT, and the whole MTP block.
    for p in [
        "model.language_model.layers.3.self_attn.indexer.index_qk_proj",
        "model.language_model.layers.3.linear_attn.in_proj_a",
        "model.language_model.layers.3.linear_attn.in_proj_b",
        "model.language_model.layers.3.linear_attn.conv1d",
        "model.language_model.layers.3.mlp.experts.0.gate_proj",
        "model.language_model.layers.3.mlp.shared_expert.up_proj",
        "lm_head",
        "model.visual.blocks.0.attn.qkv",
        "mtp.layers.0.self_attn.q_proj",
        "mtp.layers.0.linear_attn.out_proj",
        "mtp.fc_hidden",
    ] {
        assert_eq!(exl3_dense_prefix_family(p), None, "{p}");
    }
    // Gate refinement.
    let gdn_only = Exl3DenseFamilies {
        gdn: true,
        attn: false,
    };
    assert!(exl3_native_serves_dense(
        "model.layers.0.linear_attn.out_proj",
        gdn_only
    ));
    assert!(!exl3_native_serves_dense(
        "model.layers.0.self_attn.q_proj",
        gdn_only
    ));
    assert!(!exl3_native_serves_dense(
        "model.layers.0.linear_attn.out_proj",
        Exl3DenseFamilies::OFF
    ));
    assert!(exl3_native_serves_dense(
        "model.layers.0.self_attn.q_proj",
        Exl3DenseFamilies::ALL
    ));
    assert!(!exl3_native_serves_dense(
        "model.layers.0.self_attn.q_proj",
        Exl3DenseFamilies::OFF
    ));
}

#[test]
fn family_prefixes() {
    assert_eq!(
        Exl3DenseFamily::Gdn.prefixes("m.layers.1"),
        vec![
            "m.layers.1.linear_attn.in_proj_qkv",
            "m.layers.1.linear_attn.in_proj_z",
            "m.layers.1.linear_attn.out_proj",
        ]
    );
    assert_eq!(
        Exl3DenseFamily::Attn.prefixes("m.layers.2"),
        vec![
            "m.layers.2.self_attn.q_proj",
            "m.layers.2.self_attn.k_proj",
            "m.layers.2.self_attn.v_proj",
            "m.layers.2.self_attn.o_proj",
        ]
    );
}

/// qwen4_exp geometry: GDN qkv [2560->10240], z [2560->6144], out
/// [6144->2560]; attention q [2560->12288], k/v [2560->512], o
/// [6144->2560].
fn gdn_layer(m: &mut BTreeMap<String, Exl3Weight>, l: usize, k: u32, cb: Exl3Codebook) {
    let lp = format!("model.layers.{l}.linear_attn");
    m.insert(format!("{lp}.in_proj_qkv"), w(k, cb, 2560, 10240));
    m.insert(format!("{lp}.in_proj_z"), w(k, cb, 2560, 6144));
    m.insert(format!("{lp}.out_proj"), w(k, cb, 6144, 2560));
}
fn attn_layer(m: &mut BTreeMap<String, Exl3Weight>, l: usize, k: u32, cb: Exl3Codebook) {
    let lp = format!("model.layers.{l}.self_attn");
    m.insert(format!("{lp}.q_proj"), w(k, cb, 2560, 12288));
    m.insert(format!("{lp}.k_proj"), w(k, cb, 2560, 512));
    m.insert(format!("{lp}.v_proj"), w(k, cb, 2560, 512));
    m.insert(format!("{lp}.o_proj"), w(k, cb, 6144, 2560));
}

#[test]
fn keep_set_keeps_the_routed_leaves_only() {
    let mut m = BTreeMap::new();
    gdn_layer(&mut m, 0, 4, Exl3Codebook::Mul1);
    gdn_layer(&mut m, 1, 2, Exl3Codebook::Mcg);
    attn_layer(&mut m, 3, 4, Exl3Codebook::Mul1);
    let (keep, stats) = dense_keep_set(&m);
    // The whole GDN family of both layers and the whole attention family of
    // layer 3.
    assert_eq!(keep.len(), 10);
    for l in [0, 1] {
        for leaf in ["in_proj_qkv", "in_proj_z", "out_proj"] {
            assert!(
                keep.contains(&format!("model.layers.{l}.linear_attn.{leaf}")),
                "{l} {leaf}"
            );
        }
    }
    for leaf in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        assert!(keep.contains(&format!("model.layers.3.self_attn.{leaf}")));
    }
    assert_eq!(stats.gdn_layers_kept, 2);
    assert_eq!(stats.attn_layers_kept, 1);
    assert_eq!(
        stats.gdn_layers_materialized + stats.attn_layers_materialized,
        0
    );
    // BF16 equivalent: two x (qkv [10240 x 2560] + z [6144 x 2560] + out
    // [6144 x 2560]) + one x (q [12288 x 2560] + k/v [512 x 2560] x 2 + o
    // [2560 x 6144]).
    assert_eq!(
        stats.bf16_equiv_bytes,
        2 * (10240 + 6144 + 6144) * 2560 * 2 + (12288 + 512 + 512 + 6144) * 2560 * 2
    );
    assert!(stats.kept_packed_bytes < stats.bf16_equiv_bytes / 3);
}

#[test]
fn keep_set_bad_projection_drops_that_layer_only() {
    let mut m = BTreeMap::new();
    // Layer 0: out_proj at K=6 (outside {2,4}) — the WHOLE layer-0 family
    // materializes (in_proj pair included), atomically.
    gdn_layer(&mut m, 0, 4, Exl3Codebook::Mul1);
    m.insert(
        "model.layers.0.linear_attn.out_proj".to_string(),
        w(6, Exl3Codebook::Mul1, 6144, 2560),
    );
    // Layer 1: fine.
    gdn_layer(&mut m, 1, 4, Exl3Codebook::Mul1);
    // Layer 2: cb0 — out.
    gdn_layer(&mut m, 2, 4, Exl3Codebook::Inst3);
    // Layer 3: in_proj_z alone at K=3 — out (the pair is one unit with out).
    gdn_layer(&mut m, 3, 4, Exl3Codebook::Mul1);
    m.insert(
        "model.layers.3.linear_attn.in_proj_z".to_string(),
        w(3, Exl3Codebook::Mul1, 2560, 6144),
    );
    let (keep, stats) = dense_keep_set(&m);
    assert_eq!(keep.len(), 3);
    for leaf in ["in_proj_qkv", "in_proj_z", "out_proj"] {
        assert!(keep.contains(&format!("model.layers.1.linear_attn.{leaf}")));
    }
    assert_eq!(stats.gdn_layers_kept, 1);
    assert_eq!(stats.gdn_layers_materialized, 3);
    assert_eq!(stats.attn_layers_kept + stats.attn_layers_materialized, 0);
}

#[test]
fn keep_set_half_family_materializes_the_whole_layer() {
    // A layer whose out_proj shipped BF16 (not in the trellis map) while the
    // in_proj pair is EXL3: the family is incomplete, so NOTHING of it is
    // kept (one warn) — the pair materializes like any other dense linear.
    let mut m = BTreeMap::new();
    m.insert(
        "model.layers.0.linear_attn.in_proj_qkv".to_string(),
        w(4, Exl3Codebook::Mul1, 2560, 10240),
    );
    m.insert(
        "model.layers.0.linear_attn.in_proj_z".to_string(),
        w(4, Exl3Codebook::Mul1, 2560, 6144),
    );
    let (keep, stats) = dense_keep_set(&m);
    assert!(keep.is_empty());
    assert_eq!(stats.gdn_layers_kept, 0);
    assert_eq!(stats.gdn_layers_materialized, 1);
    assert_eq!(stats.kept_packed_bytes, 0);
}

#[test]
fn keep_set_ignores_non_dense_prefixes() {
    let mut m = BTreeMap::new();
    m.insert(
        "model.layers.0.mlp.experts.0.gate_proj".to_string(),
        w(4, Exl3Codebook::Mul1, 2560, 640),
    );
    m.insert(
        "lm_head".to_string(),
        w(4, Exl3Codebook::Mul1, 2560, 248320),
    );
    let (keep, stats) = dense_keep_set(&m);
    assert!(keep.is_empty());
    assert_eq!(stats, Exl3DenseKeepStats::default());
}

// ── Materialize-pass integration (mock store) ──

mod pass {
    use std::collections::HashMap;

    use spark_runtime::gpu::GpuBackend;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

    use super::super::super::exl3_materialize::materialize_exl3_impl;
    use super::super::super::{Exl3DenseFamilies, Exl3DenseFamily};

    fn t(gpu: &MockGpuBackend, shape: Vec<usize>, dtype: WeightDtype) -> WeightTensor {
        let bytes: usize = shape.iter().product::<usize>() * dtype.byte_size().max(1);
        WeightTensor {
            ptr: gpu.alloc(bytes.max(4)).unwrap(),
            shape,
            dtype,
        }
    }

    fn exl3_linear(
        gpu: &MockGpuBackend,
        m: &mut HashMap<String, WeightTensor>,
        p: &str,
        k: u32,
        in_dim: usize,
        out_dim: usize,
    ) {
        m.insert(
            format!("{p}.trellis"),
            t(
                gpu,
                vec![in_dim / 16, out_dim / 16, 16 * k as usize],
                WeightDtype::UInt16,
            ),
        );
        m.insert(format!("{p}.suh"), t(gpu, vec![in_dim], WeightDtype::F16));
        m.insert(format!("{p}.svh"), t(gpu, vec![out_dim], WeightDtype::F16));
        m.insert(format!("{p}.mul1"), t(gpu, vec![], WeightDtype::Int32));
    }

    fn stamp_mul1(gpu: &MockGpuBackend, store: &WeightStore, p: &str) {
        let flag = store.get(&format!("{p}.mul1")).unwrap().ptr;
        gpu.copy_h2d(&0x83DC_D12Du32.to_le_bytes(), flag).unwrap();
    }

    /// Two GDN layers (one at K=6, outside the envelope), one attention
    /// layer, one MTP attention layer (excluded), the QSA indexer (excluded).
    fn build() -> (MockGpuBackend, WeightStore, Vec<String>) {
        let gpu = MockGpuBackend::new();
        let mut m = HashMap::new();
        let mut all = Vec::new();
        for (l, k) in [(0usize, 4u32), (1, 6)] {
            let lp = format!("model.language_model.layers.{l}");
            for (leaf, i, o) in [
                ("in_proj_qkv", 2560, 10240),
                ("in_proj_z", 2560, 6144),
                ("out_proj", 6144, 2560),
            ] {
                let p = format!("{lp}.linear_attn.{leaf}");
                exl3_linear(&gpu, &mut m, &p, k, i, o);
                all.push(p);
            }
        }
        for lp in ["model.language_model.layers.3", "mtp.layers.0"] {
            for (leaf, i, o) in [
                ("q_proj", 2560, 12288),
                ("k_proj", 2560, 512),
                ("v_proj", 2560, 512),
                ("o_proj", 6144, 2560),
            ] {
                let p = format!("{lp}.self_attn.{leaf}");
                exl3_linear(&gpu, &mut m, &p, 4, i, o);
                all.push(p);
            }
        }
        let idx = "model.language_model.layers.3.self_attn.indexer.index_qk_proj".to_string();
        exl3_linear(&gpu, &mut m, &idx, 2, 2560, 640);
        all.push(idx);
        let store = WeightStore::from_map(m);
        for p in &all {
            stamp_mul1(&gpu, &store, p);
        }
        (gpu, store, all)
    }

    #[test]
    fn dense_gate_keeps_the_routed_set_atomically() {
        let (gpu, mut store, all) = build();
        let stats =
            materialize_exl3_impl(&gpu, &mut store, true, false, Exl3DenseFamilies::ALL).unwrap();
        // Layer 0's whole GDN family (3 tensors) and layer 3's whole attention
        // family (4 tensors) kept; layer 1 GDN (K=6), the MTP block (excluded
        // by prefix) and the indexer all materialize to BF16.
        assert_eq!(stats.kept_native, 7);
        assert_eq!(stats.bf16, all.len() - 7);
        assert_eq!(stats.quantized, 0);
        assert_eq!(stats.dense.gdn_layers_kept, 1);
        assert_eq!(stats.dense.gdn_layers_materialized, 1);
        assert_eq!(stats.dense.attn_layers_kept, 1);
        assert_eq!(stats.dense.attn_layers_materialized, 0);
        for p in &all {
            let kept = p.starts_with("model.language_model.layers.0.linear_attn.")
                || (p.starts_with("model.language_model.layers.3.self_attn.")
                    && !p.contains(".indexer."));
            assert_eq!(store.contains(&format!("{p}.trellis")), kept, "{p} trellis");
            assert_eq!(store.contains(&format!("{p}.weight")), !kept, "{p} weight");
        }
        // Idempotent.
        let again =
            materialize_exl3_impl(&gpu, &mut store, true, false, Exl3DenseFamilies::ALL).unwrap();
        assert_eq!(again.kept_native, 7);
        assert_eq!(again.bf16 + again.quantized, 0);
    }

    #[test]
    fn gdn_subgate_off_materializes_out_proj_too() {
        let (gpu, mut store, all) = build();
        let fam = Exl3DenseFamilies {
            gdn: false,
            attn: true,
        };
        let stats = materialize_exl3_impl(&gpu, &mut store, true, false, fam).unwrap();
        // Attention (layer 3) still kept; every GDN tensor materializes.
        assert_eq!(stats.kept_native, 4);
        assert_eq!(stats.bf16, all.len() - 4);
        assert_eq!(stats.dense.gdn_layers_kept, 0);
        assert_eq!(stats.dense.attn_layers_kept, 1);
        for leaf in ["in_proj_qkv", "in_proj_z", "out_proj"] {
            assert!(store.contains(&format!(
                "model.language_model.layers.0.linear_attn.{leaf}.weight"
            )));
        }
    }

    #[test]
    fn dense_gate_off_materializes_everything_as_before() {
        // native=1, dense OFF: every dense linear lands as BF16 exactly like
        // the MoE-only tree (the keep branch is never entered).
        let (gpu, mut store, all) = build();
        let stats =
            materialize_exl3_impl(&gpu, &mut store, true, false, Exl3DenseFamilies::OFF).unwrap();
        assert_eq!(stats.kept_native, 0);
        assert_eq!(stats.bf16, all.len());
        assert_eq!(stats.dense, Default::default());
        for p in &all {
            assert!(store.contains(&format!("{p}.weight")), "{p}");
            assert!(!store.contains(&format!("{p}.trellis")), "{p}");
        }
    }

    #[test]
    fn family_prefix_helpers_match_the_store_layout() {
        let (_, store, _) = build();
        let lp = "model.language_model.layers.0";
        let ps = Exl3DenseFamily::Gdn.prefixes(lp);
        assert!(!ps.is_empty());
        for p in ps {
            assert!(
                spark_runtime::weights::exl3::is_exl3_linear(&store, &p),
                "{p}"
            );
        }
    }
}
