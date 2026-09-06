// SPDX-License-Identifier: AGPL-3.0-only

//! Native EXL3 routed experts — the `ATLAS_EXL3_NATIVE_MOE=1` extension of
//! the materialize pass (see `exl3_materialize.rs` for the pass itself).
//!
//! This file owns the MoE-specific predicates and the per-layer ATOMIC
//! keep-set computation:
//!
//!  * [`exl3_native_moe_enabled`] / [`check_exl3_native_gates`] — the env
//!    gate. `ATLAS_EXL3_NATIVE_MOE=1` without `ATLAS_EXL3_NATIVE=1` is a
//!    hard ERROR (fail-loud house style), never a silent ignore.
//!  * [`exl3_native_serves_moe`] — the routed-expert prefix predicate:
//!    `.mlp.experts.N.{gate,up,down}_proj` only. `mtp.*` is EXCLUDED (MTP
//!    experts keep the NVFP4 triplet path) and `shared_expert` is EXCLUDED
//!    for this milestone (it must keep materializing so the fused
//!    shared-expert decode kernels keep their NVFP4 slot).
//!  * [`exl3_native_supported_moe`] — the per-tensor MoE kernel envelope:
//!    the intersection of the fused `exl3_moe` prefill kernel's fixed-K
//!    instance table {2,3,4,5,6} with mgemm's {2..6,8}, i.e. K in
//!    {2,3,4,5,6} ([`EXL3_NATIVE_MOE_K_BITS`]; K=8 experts exist on no
//!    shipped branch, so the fused kernel is not instantiated for it); cb
//!    MCG/MUL1 (cb0/"3inst" is not instantiated); both dims %128 (mgemm
//!    shape-2 needs k%32 and n%128 — %128 on both covers every projection
//!    orientation).
//!  * [`expert_keep_set`] — per-(layer, projection) K/cb UNIFORMITY (one
//!    mgemm launch decodes at ONE K/cb template), layer-uniform codebook
//!    (the fused `exl3_moe` prefill kernel runs gate/up/down under a single
//!    codebook) and the per-layer fused-K rule (uniform K in the table, or a
//!    mixed-K layer entirely inside the k0 runtime switch {2,3,4} —
//!    `ops::exl3_moe_fused_serves`), rolled up to an ATOMIC per-layer
//!    keep-or-materialize decision. No partial keeps: a layer that kept half
//!    its experts and then materialized the rest would double-hold memory,
//!    and a table mixing K would silently decode at the wrong bitrate.

use std::collections::{BTreeMap, HashSet};

use anyhow::{Result, bail};
use spark_runtime::weights::exl3::{Exl3Codebook, Exl3Weight};

/// `ATLAS_EXL3_NATIVE_MOE=1`: serve the routed experts natively from packed
/// trellis (requires `ATLAS_EXL3_NATIVE=1`). Read per call — load paths only.
pub fn exl3_native_moe_enabled() -> bool {
    std::env::var("ATLAS_EXL3_NATIVE_MOE").as_deref() == Ok("1")
}

/// Gate-combination validation: the MoE gate is an EXTENSION of the master
/// native gate, and setting it alone is a misconfiguration that must fail
/// loudly at load, not silently serve the materialized NVFP4 experts.
pub fn check_exl3_native_gates(native: bool, native_moe: bool) -> Result<()> {
    if native_moe && !native {
        bail!(
            "ATLAS_EXL3_NATIVE_MOE=1 requires ATLAS_EXL3_NATIVE=1 (the MoE \
             gate extends the native serving set; it cannot enable native \
             serving by itself) — set ATLAS_EXL3_NATIVE=1 or unset \
             ATLAS_EXL3_NATIVE_MOE"
        );
    }
    Ok(())
}

/// True for a ROUTED-expert projection prefix:
/// `...mlp.experts.N.{gate,up,down}_proj`.
///
/// Excludes `mtp.*` (MTP experts keep the NVFP4 triplet path — the MTP
/// loaders read triplets and their dispatch is not routed) and, by
/// construction of the pattern, `shared_expert` (kept materializing this
/// milestone so the fused shared-expert kernels keep working).
pub fn exl3_native_serves_moe(prefix: &str) -> bool {
    !(prefix.starts_with("mtp.") || prefix.contains(".mtp."))
        && prefix.contains(".mlp.experts.")
        && (prefix.ends_with(".gate_proj")
            || prefix.ends_with(".up_proj")
            || prefix.ends_with(".down_proj"))
}

/// The K values the routed-expert arm serves per tensor: the fused
/// `exl3_moe` prefill kernel's FIXED-K instance table
/// (`ops::EXL3_MOE_FUSED_K_BITS` = {2,3,4,5,6}; one definition). Per
/// shipped branch: experts are K=2 (2.05), 3 (3.05), 4 (4.05), 5 (5.05),
/// 6 (6.05), uniform across gate/up/down — every branch's routed experts
/// qualify. A MIXED-K layer additionally needs every K inside the k0
/// runtime-dispatch switch, `ops::EXL3_MOE_MIXED_K_BITS` = {2,3,4}
/// (`ops::exl3_moe_fused_serves`, applied per layer by [`expert_keep_set`]).
pub const EXL3_NATIVE_MOE_K_BITS: [u32; 5] = crate::layers::ops::EXL3_MOE_FUSED_K_BITS;

/// The per-TENSOR MoE kernel envelope (see module docs for the derivation).
/// Distinct from [`super::exl3_native_supported`]'s dense set
/// {2,3,4,5,6,8}: the expert path never touches the GEMV tier, but the fused
/// prefill kernel has no K=8 instance (see [`EXL3_NATIVE_MOE_K_BITS`]). The
/// per-LAYER mixed-K rule lives in [`expert_keep_set`].
pub fn exl3_native_supported_moe(w: &Exl3Weight) -> bool {
    EXL3_NATIVE_MOE_K_BITS.contains(&w.k_bits)
        && matches!(w.cb, Exl3Codebook::Mcg | Exl3Codebook::Mul1)
        && w.in_dim.is_multiple_of(128)
        && w.out_dim.is_multiple_of(128)
}

/// The layer-group key of an expert prefix: everything before
/// `.mlp.experts.` (e.g. `model.layers.7`). `None` for non-expert prefixes.
fn expert_layer_key(prefix: &str) -> Option<&str> {
    prefix.find(".mlp.experts.").map(|i| &prefix[..i])
}

/// The projection leaf (`gate_proj`/`up_proj`/`down_proj`) of an expert
/// prefix.
fn expert_proj_key(prefix: &str) -> &str {
    prefix.rsplit('.').next().unwrap_or(prefix)
}

/// Compute the set of expert prefixes to KEEP packed, atomically per layer.
///
/// `experts` maps every routed-expert prefix in the store (local experts
/// only — under EP the loader never uploads remote experts' tensors) to its
/// resolved [`Exl3Weight`]. A layer keeps ALL its expert tensors iff:
///  * every tensor is inside [`exl3_native_supported_moe`],
///  * within each projection, all experts share `(k_bits, cb)` (one mgemm
///    launch = one K/cb kernel template), and
///  * the codebook is uniform ACROSS the three projections (the fused
///    `exl3_moe` prefill kernel decodes gate/up/down under one codebook; K
///    may differ per projection).
///
/// Otherwise the WHOLE layer's experts materialize (with one warn naming
/// the layer and the reason) — no partial keeps.
pub(crate) fn expert_keep_set(experts: &BTreeMap<String, Exl3Weight>) -> HashSet<String> {
    // Group by layer.
    let mut layers: BTreeMap<&str, Vec<(&str, &Exl3Weight)>> = BTreeMap::new();
    for (p, w) in experts {
        if let Some(key) = expert_layer_key(p) {
            layers.entry(key).or_default().push((p.as_str(), w));
        }
    }

    let mut keep = HashSet::new();
    'layers: for (layer, tensors) in &layers {
        // Envelope check, per tensor.
        for (p, w) in tensors {
            if !exl3_native_supported_moe(w) {
                tracing::warn!(
                    "EXL3 native MoE: {layer} experts fall back to NVFP4 \
                     materialization — {p} is outside the MoE kernel envelope \
                     (K={} cb={:?} [{}x{}]; need K in {:?}, cb MCG/MUL1, \
                     dims %128). Atomic per layer: NO expert of this layer is \
                     kept packed.",
                    w.k_bits,
                    w.cb,
                    w.in_dim,
                    w.out_dim,
                    EXL3_NATIVE_MOE_K_BITS,
                );
                continue 'layers;
            }
        }
        // Per-projection (K, cb) uniformity + layer-uniform codebook.
        let mut per_proj: BTreeMap<&str, (u32, Exl3Codebook)> = BTreeMap::new();
        let mut layer_cb: Option<Exl3Codebook> = None;
        for (p, w) in tensors {
            let proj = expert_proj_key(p);
            let kc = (w.k_bits, w.cb);
            if *per_proj.entry(proj).or_insert(kc) != kc {
                tracing::warn!(
                    "EXL3 native MoE: {layer} experts fall back to NVFP4 \
                     materialization — {proj} mixes K/cb across experts \
                     ({p} has K={} cb={:?}); one mgemm launch decodes at ONE \
                     K/cb template. Atomic per layer: no partial keeps.",
                    w.k_bits,
                    w.cb,
                );
                continue 'layers;
            }
            if *layer_cb.get_or_insert(w.cb) != w.cb {
                tracing::warn!(
                    "EXL3 native MoE: {layer} experts fall back to NVFP4 \
                     materialization — codebook differs between projections \
                     ({p} has cb={:?}); the fused exl3_moe prefill kernel \
                     needs ONE codebook across gate/up/down.",
                    w.cb,
                );
                continue 'layers;
            }
        }
        // Per-layer fused-kernel rule: uniform K across gate/up/down takes
        // the fixed-K instance (any K in the envelope); a MIXED-K layer can
        // only use the k0 runtime-dispatch instance, whose switch covers
        // {2,3,4} — a K outside it would SILENTLY skip that projection's
        // GEMM in-kernel, so refuse here (no shipped branch mixes K).
        let ks: Vec<u32> = ["gate_proj", "up_proj", "down_proj"]
            .iter()
            .filter_map(|proj| per_proj.get(proj).map(|(k, _)| *k))
            .collect();
        if let [kg, ku, kd] = ks[..]
            && !crate::layers::ops::exl3_moe_fused_serves([kg, ku, kd])
        {
            tracing::warn!(
                "EXL3 native MoE: {layer} experts fall back to NVFP4 \
                 materialization — gate/up/down K={ks:?} is mixed and outside \
                 the fused kernel's runtime-dispatch set {:?} (uniform K may be \
                 any of {:?}). Atomic per layer: no partial keeps.",
                crate::layers::ops::EXL3_MOE_MIXED_K_BITS,
                crate::layers::ops::EXL3_MOE_FUSED_K_BITS,
            );
            continue 'layers;
        }
        for (p, _) in tensors {
            keep.insert((*p).to_string());
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use spark_runtime::gpu::DevicePtr;

    use super::*;

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
    fn gate_combination() {
        assert!(check_exl3_native_gates(false, false).is_ok());
        assert!(check_exl3_native_gates(true, false).is_ok());
        assert!(check_exl3_native_gates(true, true).is_ok());
        // MOE without NATIVE is a hard error, not a silent ignore.
        assert!(check_exl3_native_gates(false, true).is_err());
    }

    #[test]
    fn serves_moe_prefixes() {
        assert!(exl3_native_serves_moe(
            "model.layers.0.mlp.experts.3.gate_proj"
        ));
        assert!(exl3_native_serves_moe(
            "model.layers.47.mlp.experts.511.up_proj"
        ));
        assert!(exl3_native_serves_moe(
            "model.layers.9.mlp.experts.0.down_proj"
        ));
        // Shared expert stays on the materialize path this milestone.
        assert!(!exl3_native_serves_moe(
            "model.layers.0.mlp.shared_expert.gate_proj"
        ));
        // MTP experts keep the NVFP4 triplet path.
        assert!(!exl3_native_serves_moe(
            "mtp.layers.0.mlp.experts.3.gate_proj"
        ));
        assert!(!exl3_native_serves_moe(
            "model.mtp.layers.0.mlp.experts.3.up_proj"
        ));
        // Non-projection leaves and non-expert prefixes.
        assert!(!exl3_native_serves_moe(
            "model.layers.0.mlp.experts.3.gate_proj.suh"
        ));
        assert!(!exl3_native_serves_moe("model.layers.0.mlp.gate"));
        assert!(!exl3_native_serves_moe("lm_head"));
    }

    #[test]
    fn supported_moe_envelope() {
        // qwen4_exp shapes: gate/up [2560 -> 640], down [640 -> 2560]. Every
        // shipped branch's expert K (2/3/4/5/6 for 2.05/3.05/4.05/5.05/6.05).
        for k in [2, 3, 4, 5, 6] {
            assert!(exl3_native_supported_moe(&w(
                k,
                Exl3Codebook::Mul1,
                2560,
                640
            )));
            assert!(exl3_native_supported_moe(&w(
                k,
                Exl3Codebook::Mcg,
                640,
                2560
            )));
        }
        // K=8 is mgemm-compiled but OUTSIDE the fused exl3_moe table; K=1/7
        // have no kernels at all.
        for k in [1, 7, 8] {
            assert!(!exl3_native_supported_moe(&w(
                k,
                Exl3Codebook::Mul1,
                2560,
                640
            )));
        }
        // The fused set must stay inside the mgemm (decode-tier) envelope.
        assert!(
            EXL3_NATIVE_MOE_K_BITS
                .iter()
                .all(|k| crate::layers::ops::exl3_gemm_serves_k(*k))
        );
        // cb0 has no compiled instances.
        assert!(!exl3_native_supported_moe(&w(
            2,
            Exl3Codebook::Inst3,
            2560,
            640
        )));
        // dims must be %128.
        assert!(!exl3_native_supported_moe(&w(
            2,
            Exl3Codebook::Mul1,
            2504,
            640
        )));
        assert!(!exl3_native_supported_moe(&w(
            2,
            Exl3Codebook::Mul1,
            2560,
            600
        )));
    }

    fn ep(layer: usize, e: usize, proj: &str) -> String {
        format!("model.layers.{layer}.mlp.experts.{e}.{proj}")
    }

    #[test]
    fn keep_set_uniform_layer_kept_whole() {
        let mut m = BTreeMap::new();
        for e in 0..3 {
            m.insert(ep(0, e, "gate_proj"), w(2, Exl3Codebook::Mul1, 2560, 640));
            m.insert(ep(0, e, "up_proj"), w(2, Exl3Codebook::Mul1, 2560, 640));
            m.insert(ep(0, e, "down_proj"), w(3, Exl3Codebook::Mul1, 640, 2560));
        }
        let keep = expert_keep_set(&m);
        assert_eq!(keep.len(), 9, "K may differ ACROSS projections");
    }

    #[test]
    fn keep_set_mixed_k_layer_fully_falls_back() {
        let mut m = BTreeMap::new();
        // Layer 0: uniform (kept). Layer 1: expert 1's up_proj is K=3 while
        // expert 0's is K=2 — the WHOLE layer must materialize, including its
        // perfectly-uniform gate/down projections (atomicity: no partial keeps).
        for e in 0..2 {
            for proj in ["gate_proj", "up_proj"] {
                m.insert(ep(0, e, proj), w(2, Exl3Codebook::Mcg, 2560, 640));
            }
            m.insert(ep(0, e, "down_proj"), w(2, Exl3Codebook::Mcg, 640, 2560));
            let up_k = if e == 1 { 3 } else { 2 };
            m.insert(ep(1, e, "gate_proj"), w(2, Exl3Codebook::Mcg, 2560, 640));
            m.insert(ep(1, e, "up_proj"), w(up_k, Exl3Codebook::Mcg, 2560, 640));
            m.insert(ep(1, e, "down_proj"), w(2, Exl3Codebook::Mcg, 640, 2560));
        }
        let keep = expert_keep_set(&m);
        assert_eq!(keep.len(), 6);
        for e in 0..2 {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                assert!(keep.contains(&ep(0, e, proj)), "layer 0 fully kept");
                assert!(!keep.contains(&ep(1, e, proj)), "layer 1 fully dropped");
            }
        }
    }

    #[test]
    fn keep_set_mixed_codebook_across_projections_falls_back() {
        let mut m = BTreeMap::new();
        m.insert(ep(0, 0, "gate_proj"), w(2, Exl3Codebook::Mul1, 2560, 640));
        m.insert(ep(0, 0, "up_proj"), w(2, Exl3Codebook::Mul1, 2560, 640));
        // Uniform WITHIN the projection, but a different codebook than
        // gate/up — the fused prefill kernel cannot mix codebooks.
        m.insert(ep(0, 0, "down_proj"), w(2, Exl3Codebook::Mcg, 640, 2560));
        assert!(expert_keep_set(&m).is_empty());
    }

    #[test]
    fn keep_set_out_of_envelope_falls_back() {
        let mut m = BTreeMap::new();
        m.insert(ep(0, 0, "gate_proj"), w(8, Exl3Codebook::Mul1, 2560, 640));
        m.insert(ep(0, 0, "up_proj"), w(2, Exl3Codebook::Mul1, 2560, 640));
        m.insert(ep(0, 0, "down_proj"), w(2, Exl3Codebook::Mul1, 640, 2560));
        assert!(expert_keep_set(&m).is_empty());
    }

    #[test]
    fn keep_set_k5_k6_uniform_layers_kept_mixed_high_k_dropped() {
        // 5.05bpw (K=5) and 6.05bpw (K=6) expert layers take the fixed-K
        // fused instances. A MIXED-K layer can only use the k0 runtime
        // instance, whose switch covers {2,3,4}: gate/up K=6 + down K=5 is
        // refused (the kernel would silently skip the GEMM), while a mixed
        // layer inside {2,3,4} (gate/up K=2, down K=3) is kept.
        let mut m = BTreeMap::new();
        for e in 0..2 {
            m.insert(ep(0, e, "gate_proj"), w(5, Exl3Codebook::Mul1, 2560, 640));
            m.insert(ep(0, e, "up_proj"), w(5, Exl3Codebook::Mul1, 2560, 640));
            m.insert(ep(0, e, "down_proj"), w(5, Exl3Codebook::Mul1, 640, 2560));
            m.insert(ep(1, e, "gate_proj"), w(6, Exl3Codebook::Mul1, 2560, 640));
            m.insert(ep(1, e, "up_proj"), w(6, Exl3Codebook::Mul1, 2560, 640));
            m.insert(ep(1, e, "down_proj"), w(6, Exl3Codebook::Mul1, 640, 2560));
            m.insert(ep(2, e, "gate_proj"), w(6, Exl3Codebook::Mul1, 2560, 640));
            m.insert(ep(2, e, "up_proj"), w(6, Exl3Codebook::Mul1, 2560, 640));
            m.insert(ep(2, e, "down_proj"), w(5, Exl3Codebook::Mul1, 640, 2560));
            m.insert(ep(3, e, "gate_proj"), w(2, Exl3Codebook::Mul1, 2560, 640));
            m.insert(ep(3, e, "up_proj"), w(2, Exl3Codebook::Mul1, 2560, 640));
            m.insert(ep(3, e, "down_proj"), w(3, Exl3Codebook::Mul1, 640, 2560));
        }
        let keep = expert_keep_set(&m);
        assert_eq!(keep.len(), 18);
        for e in 0..2 {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                assert!(keep.contains(&ep(0, e, proj)), "uniform K=5 kept");
                assert!(keep.contains(&ep(1, e, proj)), "uniform K=6 kept");
                assert!(!keep.contains(&ep(2, e, proj)), "mixed 6/6/5 dropped");
                assert!(keep.contains(&ep(3, e, proj)), "mixed 2/2/3 kept");
            }
        }
        assert!(crate::layers::ops::exl3_moe_fused_serves([5, 5, 5]));
        assert!(crate::layers::ops::exl3_moe_fused_serves([2, 2, 3]));
        assert!(!crate::layers::ops::exl3_moe_fused_serves([6, 6, 5]));
        assert!(!crate::layers::ops::exl3_moe_fused_serves([8, 8, 8]));
    }
}
