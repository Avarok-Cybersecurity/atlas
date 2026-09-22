// SPDX-License-Identifier: AGPL-3.0-only

//! The **one** dtype conversion `glm5_next` weight loading is allowed to perform:
//! F32 on disk where the plan wants BF16.
//!
//! # Why this exists
//!
//! The reference checkpoint this port was built against
//! (`LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3`) stores every non-quantised
//! attention tensor at BF16. NVIDIA's official export,
//! `nvidia/GLM-5.3-Flash-NVFP4` (a ModelOpt export of the same model), stores a
//! large share of its non-quantised tensors at **F32** instead — the conv1d
//! weights among them:
//!
//! ```text
//! model.language_model.layers.N.self_attn.{q,k,v}_conv1d.weight
//!   LibertAIDAI : BF16 [8192, 1, 4]
//!   nvidia      : F32  [8192, 1, 4]
//! ```
//!
//! Nothing about the model changed — only the export's storage width. Before
//! this module the loader rejected that, and the rejection surfaced as a *byte
//! count* error in `KdaTpPlan` sharding, which reads as a shape bug and is not:
//!
//! ```text
//! q_conv1d: 131072 B on disk, the plan's full shape [8192, 4] x 2 B implies 65536 B
//! ```
//!
//! # What is and is not converted
//!
//! Converted: **F32 on disk where the plan says BF16**, and nothing else. That
//! rounds to nearest even via `half::bf16::from_f32` — the same cast
//! `upload_f32_as_bf16` already applies to norms and mHC in this loader, so
//! there is no second rounding convention in the port.
//!
//! The cast feeds the **raw-bytes path only** — `LayerSource::get`, i.e. the
//! KDA binder and the DSA verifier, the two places a fixed element width is
//! structural. `LayerSource::f32` is untouched and keeps reading the
//! checkpoint's own bytes, so the DSA absorb products and the F32 `ape` upload
//! still see every bit an F32 export carries; those already read either width.
//!
//! NOT converted, deliberately:
//!
//! * **BF16 on disk.** The mismatch branch is never entered, no copy is made,
//!   and the LibertAIDAI checkpoint loads byte-identically to before this
//!   module existed. That is the whole point of keying the cast off a
//!   *mismatch* rather than off a target dtype.
//! * **`A_log` and `dt_bias`.** The plan wants F32 and the KDA kernel signatures
//!   want F32. Both exports store them F32. Casting either to BF16 changes the
//!   gate — see the trap note at the top of
//!   [`crate::layers::glm5next_kda::binding`].
//! * **Anything quantised** (U8 NVFP4 payloads, F8_E4M3, their `weight_scale` /
//!   `weight_scale_2` siblings). Those never reach a `KdaTensorSource`; they are
//!   bound zero-copy off their device pointers and this module never sees them.
//! * **BF16 on disk where the plan wants F32.** There is no such tensor in
//!   either export, so there is no evidence to build the reverse cast on, and
//!   the binder's existing dtype error still fires if one ever appears.
//!
//! # The plan is read, not restated
//!
//! [`plan_dtype`] answers from the binders' own spec tables — `KDA_TENSORS` for
//! the KDA family, `dsa_tensor_specs` for the DSA family — so a spec change
//! moves the cast with it and cannot leave a stale copy here. `dsa_tensor_specs`
//! is shape-parameterised on a config this module has no access to at
//! `collect` time, but every one of its entries is BF16 by construction; the
//! test `every_dsa_spec_is_bf16` pins that to the table rather than to a
//! comment, so the day a DSA spec goes F32 the pin fails instead of this module
//! quietly casting it.

use anyhow::{Result, bail};
use spark_runtime::weights::WeightDtype;

use crate::layers::glm5next_kda::binding::{KDA_TENSORS, KdaDtype};

/// The dtype the plan wants for one **layer-relative** tensor name
/// (`self_attn.q_conv1d.weight`), or `None` for a tensor no attention spec
/// claims (norms, mHC, MLP — all of which reach the device through
/// `LayerSource::f32`, which already reads either width).
pub(super) fn plan_dtype(rel: &str) -> Option<KdaDtype> {
    if let Some(spec) = KDA_TENSORS.iter().find(|s| s.name == rel) {
        return Some(spec.dtype);
    }
    // Every `dsa_tensor_specs` entry is BF16 — pinned by `every_dsa_spec_is_bf16`.
    // `self_attn.*` names the KDA table does not claim are DSA's (or are
    // unrecognised, in which case the binder's "unclaimed tensor" error is the
    // one that must fire, not a silent skip here).
    if rel.starts_with("self_attn.") {
        return Some(KdaDtype::Bf16);
    }
    None
}

/// The plan-dtype copy of one tensor's host bytes, or `None` when the export
/// already stored it at the width the binders bind at.
///
/// `None` is the answer for every tensor of a checkpoint whose attention stack
/// is BF16 — no copy is made, nothing is allocated, and
/// [`super::LayerSource::get`] hands out the checkpoint's own bytes exactly as
/// it did before this module existed.
pub(super) fn cast_to_plan_dtype(
    rel: &str,
    on_disk: WeightDtype,
    bytes: &[u8],
) -> Result<Option<(WeightDtype, Vec<u8>)>> {
    if on_disk != WeightDtype::FP32 || plan_dtype(rel) != Some(KdaDtype::Bf16) {
        return Ok(None);
    }
    Ok(Some((
        WeightDtype::BF16,
        f32_bytes_to_bf16_bytes(rel, bytes)?,
    )))
}

/// Little-endian F32 bytes → little-endian BF16 bytes, round to nearest even.
///
/// 🪤 `chunks_exact` would silently drop a trailing partial element, which for a
/// truncated shard is exactly the corruption a strict loader exists to catch —
/// so the length is checked first and a bad length is an error, not a shorter
/// tensor.
fn f32_bytes_to_bf16_bytes(rel: &str, src: &[u8]) -> Result<Vec<u8>> {
    if !src.len().is_multiple_of(4) {
        bail!(
            "{rel}: {} B is not a whole number of F32 elements",
            src.len()
        );
    }
    Ok(src
        .chunks_exact(4)
        .flat_map(|c| {
            half::bf16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_le_bytes()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::glm5next_dsa::Glm5NextDsaConfig;
    use crate::layers::glm5next_dsa::binding::dsa_tensor_specs;

    fn f32_blob(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn bf16_blob(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
            .collect()
    }

    /// GLM-5.3's measured DSA geometry, copied from
    /// `glm5next_dsa::binding::tests` so this pin reads the real table.
    fn dsa_cfg() -> Glm5NextDsaConfig {
        Glm5NextDsaConfig {
            hidden: 4096,
            index_heads: 32,
            index_head_dim: 128,
            index_kpool: 4,
            index_topk: 2048,
            always_select_tail: true,
            local_heads: 64,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_nope_head_dim: 256,
            qk_rope_head_dim: 0,
            v_head_dim: 256,
            max_context: 16_384,
        }
    }

    /// The claim `plan_dtype`'s `self_attn.*` fallback rests on. If a DSA spec
    /// ever becomes F32, this fails rather than the fallback casting it away.
    #[test]
    fn every_dsa_spec_is_bf16() {
        for s in dsa_tensor_specs(&dsa_cfg(), 64) {
            assert_eq!(s.dtype, KdaDtype::Bf16, "{} is no longer BF16", s.name);
        }
    }

    /// The plan is read off the binder's table, F32 entries included.
    #[test]
    fn plan_dtype_matches_the_kda_spec_table() {
        assert_eq!(
            plan_dtype("self_attn.q_conv1d.weight"),
            Some(KdaDtype::Bf16)
        );
        assert_eq!(plan_dtype("self_attn.o_proj.weight"), Some(KdaDtype::Bf16));
        // The two the plan genuinely wants at F32.
        assert_eq!(plan_dtype("self_attn.A_log"), Some(KdaDtype::F32));
        assert_eq!(plan_dtype("self_attn.dt_bias"), Some(KdaDtype::F32));
        // DSA-only names: not in KDA_TENSORS, still BF16 in the plan.
        assert_eq!(
            plan_dtype("self_attn.indexer.k_norm.bias"),
            Some(KdaDtype::Bf16)
        );
        // Not an attention tensor: no plan entry, so no cast — `LayerSource::f32`
        // already reads these at either width.
        assert_eq!(plan_dtype("input_layernorm.weight"), None);
        assert_eq!(plan_dtype("hc_attn_fn"), None);
        assert_eq!(plan_dtype("mlp.gate.weight"), None);
    }

    /// The failing tensor from `nvidia/GLM-5.3-Flash-NVFP4`: F32 on disk where
    /// the plan says BF16. The bytes must come back as the BF16 the plan wants,
    /// bit-for-bit.
    #[test]
    fn f32_conv1d_is_cast_to_the_plan_bf16() {
        // Includes a tie that only round-to-nearest-EVEN gets right (0x3F81_8000
        // is exactly halfway between 0x3F81 and 0x3F82; RNE picks the even
        // 0x3F82) and one that rounds DOWN to stay even (0x3F80_8000 -> 0x3F80).
        let vals = [
            1.0f32,
            -2.5,
            0.0,
            f32::from_bits(0x3F80_8000),
            f32::from_bits(0x3F81_8000),
            1.0e-8,
        ];
        let (dt, out) = cast_to_plan_dtype(
            "self_attn.q_conv1d.weight",
            WeightDtype::FP32,
            &f32_blob(&vals),
        )
        .unwrap()
        .expect("F32 on disk where the plan says BF16 must be cast");
        assert_eq!(dt, WeightDtype::BF16);
        assert_eq!(out.len(), vals.len() * 2, "F32 -> BF16 halves the tensor");
        assert_eq!(out, bf16_blob(&vals));
        // Spelled out, so a change of rounding convention cannot pass by
        // agreeing with itself.
        assert_eq!(&out[0..2], &0x3F80u16.to_le_bytes());
        assert_eq!(&out[6..8], &0x3F80u16.to_le_bytes());
        assert_eq!(&out[8..10], &0x3F82u16.to_le_bytes());
    }

    /// A BF16 source takes the raw path: no cast, no copy, the checkpoint's own
    /// bytes. This is the LibertAIDAI checkpoint's entire attention stack, and
    /// the reason that checkpoint loads byte-identically to before.
    #[test]
    fn bf16_source_takes_the_raw_path() {
        let raw = bf16_blob(&[1.0, -2.5, 0.0, 7.75]);
        assert!(
            cast_to_plan_dtype("self_attn.q_conv1d.weight", WeightDtype::BF16, &raw)
                .unwrap()
                .is_none()
        );
    }

    /// F32 where the plan ALSO says F32 takes the raw path. Casting `A_log` or
    /// `dt_bias` to BF16 changes the KDA gate, and the kernels read them at 4 B.
    #[test]
    fn f32_stays_raw_where_the_plan_wants_f32() {
        let raw = f32_blob(&[1.0, -2.5, 0.0]);
        for name in ["self_attn.A_log", "self_attn.dt_bias"] {
            assert!(
                cast_to_plan_dtype(name, WeightDtype::FP32, &raw)
                    .unwrap()
                    .is_none(),
                "{name} must not be cast"
            );
        }
    }

    /// A tensor no attention spec claims is never cast, whatever its width —
    /// `LayerSource::f32` reads norms, mHC and the MLP at either width already.
    #[test]
    fn unclaimed_tensors_are_never_cast() {
        let raw = f32_blob(&[1.0, 2.0]);
        for name in ["input_layernorm.weight", "hc_attn_fn", "mlp.gate.weight"] {
            assert!(
                cast_to_plan_dtype(name, WeightDtype::FP32, &raw)
                    .unwrap()
                    .is_none(),
                "{name} must not be cast"
            );
        }
    }

    /// Quantised payloads never take the cast branch even if one somehow
    /// reached a claimed name.
    #[test]
    fn quantised_dtypes_are_left_alone() {
        let raw = vec![0xABu8; 16];
        for dt in [WeightDtype::UInt8, WeightDtype::FP8E4M3] {
            assert!(
                cast_to_plan_dtype("self_attn.q_conv1d.weight", dt, &raw)
                    .unwrap()
                    .is_none(),
                "{dt:?} must not be cast"
            );
        }
    }

    /// A truncated F32 blob is an error, not a silently shorter tensor.
    #[test]
    fn a_partial_f32_element_is_an_error() {
        let err = cast_to_plan_dtype("self_attn.q_conv1d.weight", WeightDtype::FP32, &[0u8; 10])
            .unwrap_err()
            .to_string();
        assert!(err.contains("whole number of F32 elements"), "{err}");
    }
}
