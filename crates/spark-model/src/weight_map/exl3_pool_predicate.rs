// SPDX-License-Identifier: AGPL-3.0-only

//! The fast loader's EXL3 pool predicate: which trellis linears' four
//! tensors (`.trellis/.suh/.svh/.mul1`) are uploaded into one arena per
//! (shard, class) instead of one `cuMemAlloc` each. Child module of
//! `exl3_materialize.rs` (≤500 LoC split); re-exported from `weight_map`.
//!
//! The pool must admit exactly the prefixes the materialize pass will KEEP
//! packed — a non-kept prefix is materialized and freed, and inside an arena
//! its bytes could never be reclaimed (with the native gates off that is the
//! whole packed checkpoint held next to its NVFP4 rewrite). The prediction
//! here is header-only: prefix routing + trellis shape (K and the 128-
//! divisible geometry). Two inputs of the real keep decision are NOT known
//! at load — the codebook (the `.mul1` VALUE) and per-layer K/cb uniformity
//! — so a mixed or cb0 layer would be predicted "keep" and then materialized:
//! its arena bytes are stranded (counted and warned by the pass as
//! `stranded_pooled_bytes`), never freed twice. No shipped branch mixes K or
//! codebook, so in practice prediction and decision agree (pinned by the
//! tests below against `materialize_exl3_impl`).
//!
//! Kill switch: `ATLAS_EXL3_WEIGHT_POOL=0` disables pooling (every tensor
//! per-tensor, byte-identical to the pre-pool loader). Default ON whenever
//! `ATLAS_EXL3_NATIVE=1`; with native serving off nothing is kept, so
//! nothing is pooled and the predicate is not installed at all.

use spark_runtime::fast_weights::PoolPredicate;

use super::native::exl3_native_serves_with;
use crate::weight_map::{
    EXL3_NATIVE_DENSE_K_BITS, EXL3_NATIVE_MOE_K_BITS, Exl3DenseFamilies, exl3_native_serves_moe,
};

/// `ATLAS_EXL3_WEIGHT_POOL=0` turns the pool off. Read per call (load only).
pub fn exl3_weight_pool_enabled() -> bool {
    std::env::var("ATLAS_EXL3_WEIGHT_POOL").as_deref() != Ok("0")
}

/// Header-only prediction of the materialize pass's keep decision for
/// `prefix` with trellis shape `[in/16, out/16, 16*K]` under the given
/// gates. Env-independent (tests thread the gates explicitly).
pub fn exl3_pool_keep_predicted(
    prefix: &str,
    trellis_shape: &[usize],
    native: bool,
    native_moe: bool,
    dense: Exl3DenseFamilies,
) -> bool {
    if !native || !exl3_native_serves_with(prefix, native_moe, dense) {
        return false;
    }
    let [in16, out16, kx16] = trellis_shape else {
        return false;
    };
    if !kx16.is_multiple_of(16)
        || !(in16 * 16).is_multiple_of(128)
        || !(out16 * 16).is_multiple_of(128)
    {
        return false;
    }
    let k_bits = (kx16 / 16) as u32;
    if exl3_native_serves_moe(prefix) {
        EXL3_NATIVE_MOE_K_BITS.contains(&k_bits)
    } else {
        EXL3_NATIVE_DENSE_K_BITS.contains(&k_bits)
    }
}

/// The predicate to install on `FastSafetensorsLoader::pool_predicate`,
/// from the environment gates. `None` when nothing would be kept packed
/// (`ATLAS_EXL3_NATIVE` unset) or the kill switch is set. Reads the gates
/// once; gate VALIDATION stays with the materialize pass, which runs right
/// after the load and fails the boot on a misconfiguration either way.
pub fn exl3_fast_load_pool_predicate() -> Option<PoolPredicate> {
    let native = super::exl3_native_enabled();
    if !native || !exl3_weight_pool_enabled() {
        return None;
    }
    let native_moe = crate::weight_map::exl3_native_moe_enabled();
    let dense = crate::weight_map::exl3_native_dense_families();
    Some(std::sync::Arc::new(move |prefix: &str, shape: &[usize]| {
        exl3_pool_keep_predicted(prefix, shape, native, native_moe, dense)
    }))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use spark_runtime::gpu::GpuBackend;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

    use super::*;
    use crate::weight_map::materialize_exl3_impl;

    fn t(gpu: &MockGpuBackend, shape: Vec<usize>, dtype: WeightDtype) -> WeightTensor {
        let bytes: usize = shape.iter().product::<usize>() * dtype.byte_size().max(1);
        WeightTensor {
            ptr: gpu.alloc(bytes.max(4)).unwrap(),
            shape,
            dtype,
        }
    }

    /// [2560 -> 640] quartet at K bits, mul1 stamped to the MUL1 codebook.
    fn exl3_linear(gpu: &MockGpuBackend, m: &mut HashMap<String, WeightTensor>, p: &str, k: u32) {
        m.insert(
            format!("{p}.trellis"),
            t(gpu, vec![160, 40, 16 * k as usize], WeightDtype::UInt16),
        );
        m.insert(format!("{p}.suh"), t(gpu, vec![2560], WeightDtype::F16));
        m.insert(format!("{p}.svh"), t(gpu, vec![640], WeightDtype::F16));
        let flag = t(gpu, vec![], WeightDtype::Int32);
        gpu.copy_h2d(&0x83DC_D12Du32.to_le_bytes(), flag.ptr)
            .unwrap();
        m.insert(format!("{p}.mul1"), flag);
    }

    /// Full families (the dense keep-set is atomic per (layer, family) and
    /// needs every leaf present), a K=7 attention layer (no dense kernel),
    /// the shared expert and an MTP expert (never native).
    const PREFIXES: [(&str, u32); 12] = [
        ("lm_head", 4),
        ("model.layers.0.mlp.experts.0.gate_proj", 4),
        ("model.layers.0.mlp.experts.1.gate_proj", 4),
        ("model.layers.0.mlp.shared_expert.up_proj", 6),
        ("model.layers.0.linear_attn.in_proj_qkv", 6),
        ("model.layers.0.linear_attn.in_proj_z", 6),
        ("model.layers.0.linear_attn.out_proj", 6),
        ("model.layers.1.self_attn.q_proj", 7),
        ("model.layers.1.self_attn.k_proj", 7),
        ("model.layers.1.self_attn.v_proj", 7),
        ("model.layers.1.self_attn.o_proj", 7),
        ("mtp.layers.0.mlp.experts.0.gate_proj", 4),
    ];

    /// For every gate combination, the header-only prediction equals what
    /// the pass actually kept (`.trellis` still in the store afterwards).
    fn assert_prediction_matches_pass(native: bool, native_moe: bool, dense: Exl3DenseFamilies) {
        let gpu = MockGpuBackend::new();
        let mut m = HashMap::new();
        for (p, k) in PREFIXES {
            exl3_linear(&gpu, &mut m, p, k);
        }
        let predicted: Vec<bool> = PREFIXES
            .iter()
            .map(|(p, k)| {
                exl3_pool_keep_predicted(p, &[160, 40, 16 * *k as usize], native, native_moe, dense)
            })
            .collect();
        let mut store = WeightStore::from_map(m);
        materialize_exl3_impl(&gpu, &mut store, native, native_moe, dense).unwrap();
        for ((p, _), pred) in PREFIXES.iter().zip(predicted) {
            assert_eq!(
                pred,
                store.contains(&format!("{p}.trellis")),
                "{p}: predicted keep={pred} under native={native} moe={native_moe} dense={dense:?}"
            );
        }
    }

    #[test]
    fn prediction_matches_pass_gates_off() {
        assert_prediction_matches_pass(false, false, Exl3DenseFamilies::OFF);
    }

    #[test]
    fn prediction_matches_pass_native_only() {
        assert_prediction_matches_pass(true, false, Exl3DenseFamilies::OFF);
    }

    #[test]
    fn prediction_matches_pass_native_moe() {
        assert_prediction_matches_pass(true, true, Exl3DenseFamilies::OFF);
    }

    #[test]
    fn prediction_matches_pass_all_gates() {
        assert_prediction_matches_pass(true, true, Exl3DenseFamilies::ALL);
    }

    #[test]
    fn rejects_bad_geometry_and_unknown_k() {
        let all = Exl3DenseFamilies::ALL;
        // 2-D shape (not a trellis), non-128 dims, K=7 dense, K=8 expert.
        assert!(!exl3_pool_keep_predicted(
            "lm_head",
            &[160, 40],
            true,
            true,
            all
        ));
        assert!(!exl3_pool_keep_predicted(
            "lm_head",
            &[161, 40, 64],
            true,
            true,
            all
        ));
        assert!(!exl3_pool_keep_predicted(
            "lm_head",
            &[160, 40, 112],
            true,
            true,
            all
        ));
        assert!(exl3_pool_keep_predicted(
            "lm_head",
            &[160, 40, 128],
            true,
            true,
            all
        )); // K=8 dense ok
        let e = "model.layers.0.mlp.experts.0.up_proj";
        assert!(!exl3_pool_keep_predicted(
            e,
            &[160, 40, 128],
            true,
            true,
            all
        )); // K=8 expert: no
        assert!(exl3_pool_keep_predicted(e, &[160, 40, 32], true, true, all)); // K=2 expert: yes
        assert!(!exl3_pool_keep_predicted(
            e,
            &[160, 40, 32],
            true,
            false,
            all
        )); // moe gate off
    }
}
