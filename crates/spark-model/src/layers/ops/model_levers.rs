// SPDX-License-Identifier: AGPL-3.0-only

//! Model-side kernel-path levers, resolved once and then carried.
//!
//! The second of the two lever categories on [`crate::layer::ForwardContext`]:
//!
//! * [`super::GemmDispatch`] — which GEMM implementation each projection takes.
//! * [`ModelLevers`] — everything else the model's kernel paths branch on:
//!   the SSM/GDN recurrence variant, FFN routing, MoE quantization, LoRA
//!   application mode, diagnostics.
//!
//! Both were `OnceLock<bool>` statics reading `ATLAS_*` at first touch. Two
//! problems with that, and only the first is about hot-swap:
//!
//! 1. A static outlives the model whose flags it encodes. Load a second model
//!    whose recipe sets different levers and the process keeps taking the
//!    previous model's branches — silently, because a cached `bool` cannot
//!    report that it is stale.
//! 2. It hides the dependency. A function that reads the environment through a
//!    static declares nothing in its signature, cannot be exercised with a
//!    different configuration without mutating the process, and gives the
//!    compiler nothing to check.
//!
//! Carrying it fixes both, and a site that forgets the field fails to build.

/// Kernel-path levers for one loaded model.
///
/// Plain `Copy` data resolved from the environment at model construction. Group
/// membership follows the subsystem the lever steers, so a reader can see at a
/// glance which part of the forward pass a flag reaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ModelLevers {
    // ── SSM / GDN recurrence ──
    /// Keep GDN recurrent state in registers across the prefill chunk loop.
    /// Default ON (the fold that shipped in PR #369, −7.25 % wall); the env var
    /// is an opt-OUT, which is why the field is stored positively and the
    /// resolution inverts it.
    pub gdn_regresident: bool,
    /// Batched FLA path for multi-sequence GDN decode.
    pub gdn_batched_fla: bool,
    /// WY17 GDN recurrence variant. Ships ON; `ATLAS_GDN_WY17=0` opts out.
    pub gdn_wy17: bool,
    /// WY-N GDN recurrence variant. Ships ON; `ATLAS_GDN_WYN=0` opts out.
    pub gdn_wyn: bool,

    // ── FFN / MoE ──
    /// Lossless single-warp decode GEMV (`w4a16_gemv_sw`, `w4a16_gemv_dual_sw`).
    /// Ships ON; `ATLAS_NO_GEMV_SW=1` restores the 64-thread kernels.
    pub gemv_sw: bool,
    /// Route decode FFN through the tile GEMM rather than the scalar GEMV.
    pub decode_ffn_via_gemm: bool,
    /// Small-M FFN GEMM tile shape. Ships ON; `ATLAS_FFN_SMALLM=0` opts out.
    pub ffn_small_m: bool,
    /// FP4 holo layout for the MoE down projection.
    pub holo_moe_down_fp4: bool,
    /// FP4 holo layout for the MoE gate/up projections.
    pub holo_moe_gateup_fp4: bool,
    /// Collect per-layer MoE expert-union statistics. Diagnostic.
    pub moe_union_stats: bool,
    /// `ATLAS_BEL_RESCALE=1` — undo the renormalization that hands absent
    /// experts' routing mass to substitutes, so every selected expert carries
    /// its TRUE softmax weight and the routed branch contributes only the
    /// share this serve can actually supply.
    ///
    /// OFF by default because it is a MEASURED REGRESSION on its own:
    /// Qwen3.6-35B-A3B-FP8, code-python at 37% of experts, 60 prompts —
    /// delta-CE went from +0.7888 to +1.0943 nats/token and top-1 agreement
    /// from 73.3% to 65.4%, worse in all twenty categories. Recovering the
    /// true weights shrinks the routed branch to rho of its usual total and
    /// nothing replaces the rest, so it trades a substitution error for a
    /// magnitude deficit — and the model turns out to mind the second one
    /// more. The lever stays because rho is the input a compensation term
    /// needs; the correction is only half a fix without one.
    pub bel_rescale: bool,
    /// Per-request MoE expert-activation telemetry (`--expert-telemetry`).
    /// Not from the environment: `TransformerModel::new` writes it from the
    /// serve arg, like `max_decode_seqs`.
    ///
    /// A boot decision rather than a per-request one because the staging
    /// buffer and the device-to-device copies that fill it must exist before
    /// decode-graph capture; a request cannot add a memcpy node to a graph
    /// that was already recorded. Which requests REPORT experts is then a
    /// per-request choice (`report_expert_metadata`).
    pub expert_telemetry: bool,

    // ── Attention ──
    /// Contiguous-attention path for the DFlash head.
    pub dflash_contig_attn: bool,

    // ── LoRA ──
    /// Apply LoRA eagerly at load instead of at each forward.
    pub lora_eager: bool,
    /// Allow hot rotation of LoRA adapters.
    pub lora_rotate: bool,

    // ── Diagnostics ──
    /// K=4 chain-widening diagnostics.
    pub k4_diag: bool,
    /// Per-layer hidden-state norm dumps on the Gemma-4 decode path. Heavy —
    /// one device-to-host copy per layer.
    pub gemma4_diag: bool,

    // ── Attention (cont.) ──
    /// BF16 tensor-core attention projections: dequant FP4 to BF16 and use a
    /// BF16 MMA instead of the default path, which crushes activations to FP8
    /// E4M3. Removes the FP8 prefill perturbation on those projections.
    pub bf16_tc_proj: bool,
    /// Configured max decode batch (`--max-batch-size`), the reference count
    /// the split-K attention split count is pinned to. Not from the
    /// environment: `TransformerModel::new` writes it from the serve arg.
    ///
    /// It pins DETERMINISM — the online-softmax split-merge is
    /// non-associative, so a sequence decoded alone must see the same
    /// reduction tree as one co-batched with fifteen others. Held in a
    /// `OnceLock` it was also idempotent, so a second model with a different
    /// max batch would silently keep the first model's split count.
    pub max_decode_seqs: u32,
    /// `ATLAS_MTP_SHADOW_TOPK=k` (0 = off, clamped to 8): the drafter D2Hs
    /// its logits and logs the top-k candidates. Observational only.
    pub shadow_topk: usize,
    /// `ATLAS_KV_POISON=1` — fill a fresh KV block with NaN instead of zero,
    /// the discriminator for the "unwritten fresh tail block read"
    /// hypothesis. A diagnostic that changes what the kernels READ, so it
    /// must not leak across a swap.
    pub kv_poison: bool,
    /// MTP drafter context policy (`ATLAS_NO_DRAFTER_CONTEXT` /
    /// `ATLAS_DRAFTER_PREFILL_ONLY`), resolved and logged once per model.
    /// The two halves are coupled — prefill without carry is a measured
    /// −927 ms/turn loss — so they travel as one value.
    pub drafter: crate::model::drafter_context::DrafterContext,
}

fn from_values(
    mut value: impl FnMut(&str) -> Option<String>,
    mut present: impl FnMut(&str) -> bool,
    shadow_topk: usize,
    drafter: crate::model::drafter_context::DrafterContext,
) -> ModelLevers {
    fn opt_in(value: Option<&str>) -> bool {
        value == Some("1")
    }
    fn opt_out(value: Option<&str>) -> bool {
        value != Some("0")
    }
    fn opt_in_truthy(value: Option<&str>) -> bool {
        value.is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    }

    ModelLevers {
        max_decode_seqs: 1,
        // Written by `TransformerModel::new` from the serve arg, not read
        // from the environment (see the field docs).
        expert_telemetry: false,
        bel_rescale: opt_in(value("ATLAS_BEL_RESCALE").as_deref()),
        shadow_topk,
        kv_poison: opt_in(value("ATLAS_KV_POISON").as_deref()),
        drafter,
        gdn_regresident: value("ATLAS_NO_GDN_REGRESIDENT").as_deref() != Some("1"),
        gdn_batched_fla: opt_in(value("ATLAS_GDN_BATCHED_FLA").as_deref()),
        gdn_wy17: opt_out(value("ATLAS_GDN_WY17").as_deref()),
        gdn_wyn: opt_out(value("ATLAS_GDN_WYN").as_deref()),
        ffn_small_m: opt_out(value("ATLAS_FFN_SMALLM").as_deref()),
        gemv_sw: super::gemv_sw::gemv_sw_from(value("ATLAS_NO_GEMV_SW").as_deref()),
        decode_ffn_via_gemm: opt_in(value("ATLAS_DECODE_FFN_VIA_GEMM").as_deref()),
        holo_moe_down_fp4: opt_in_truthy(value("ATLAS_HOLO_MOE_DOWN_FP4").as_deref()),
        holo_moe_gateup_fp4: opt_in_truthy(value("ATLAS_HOLO_MOE_GATEUP_FP4").as_deref()),
        moe_union_stats: opt_in(value("ATLAS_MOE_UNION_STATS").as_deref()),
        dflash_contig_attn: opt_in(value("ATLAS_DFLASH_CONTIG_ATTN").as_deref()),
        lora_eager: opt_in_truthy(value("ATLAS_LORA_EAGER").as_deref()),
        lora_rotate: opt_in_truthy(value("ATLAS_LORA_ROTATE").as_deref()),
        k4_diag: opt_in(value("ATLAS_K4_DIAG").as_deref()),
        gemma4_diag: opt_in_truthy(value("ATLAS_DIAG_GEMMA4").as_deref()),
        bf16_tc_proj: present("ATLAS_BF16_TC_PROJ"),
    }
}

impl ModelLevers {
    /// Resolve from the environment. Called once, when the model is built.
    pub fn from_env() -> Self {
        from_values(
            |var| std::env::var(var).ok(),
            |var| std::env::var_os(var).is_some(),
            crate::speculative::shadow_topk(),
            crate::model::drafter_context::resolve_from_env(),
        )
    }

    /// What a build resolves to with no `ATLAS_*` set — every opt-in off, the
    /// one opt-out lever on. Tests construct a context with this instead of
    /// mutating the process environment.
    pub fn defaults() -> Self {
        Self {
            max_decode_seqs: 1,
            shadow_topk: 0,
            kv_poison: false,
            drafter: crate::model::drafter_context::DrafterContext::BOTH,
            gdn_regresident: true,
            gdn_wy17: true,
            gdn_wyn: true,
            ffn_small_m: true,
            gemv_sw: true,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn resolve(values: &[(&str, &str)]) -> ModelLevers {
        let values: HashMap<_, _> = values.iter().copied().collect();
        from_values(
            |name| values.get(name).map(|value| (*value).to_owned()),
            |name| values.contains_key(name),
            0,
            crate::model::drafter_context::DrafterContext::BOTH,
        )
    }

    #[test]
    fn the_opt_out_lever_is_on_by_default_and_every_opt_in_is_off() {
        let d = ModelLevers::defaults();
        assert_eq!(
            resolve(&[]),
            d,
            "absent environment uses the public default"
        );
        assert_eq!(
            d,
            ModelLevers {
                gdn_regresident: true,
                gdn_wy17: true,
                gdn_wyn: true,
                gemv_sw: true,
                ffn_small_m: true,
                max_decode_seqs: 1,
                drafter: crate::model::drafter_context::DrafterContext::BOTH,
                ..ModelLevers::default()
            }
        );
    }

    #[test]
    fn exact_one_opt_ins_map_to_their_own_fields() {
        let cases = [
            ("ATLAS_KV_POISON", [true, false, false, false, false, false]),
            (
                "ATLAS_GDN_BATCHED_FLA",
                [false, true, false, false, false, false],
            ),
            (
                "ATLAS_DECODE_FFN_VIA_GEMM",
                [false, false, true, false, false, false],
            ),
            (
                "ATLAS_MOE_UNION_STATS",
                [false, false, false, true, false, false],
            ),
            (
                "ATLAS_DFLASH_CONTIG_ATTN",
                [false, false, false, false, true, false],
            ),
            ("ATLAS_K4_DIAG", [false, false, false, false, false, true]),
        ];
        for (name, expected) in cases {
            let d = resolve(&[(name, "1")]);
            assert_eq!(
                [
                    d.kv_poison,
                    d.gdn_batched_fla,
                    d.decode_ffn_via_gemm,
                    d.moe_union_stats,
                    d.dflash_contig_attn,
                    d.k4_diag
                ],
                expected,
                "{name}"
            );
        }
        assert!(!resolve(&[("ATLAS_K4_DIAG", "true")]).k4_diag);
    }

    #[test]
    fn truthy_opt_ins_map_independently_and_presence_is_distinct() {
        let cases = [
            (
                "ATLAS_HOLO_MOE_DOWN_FP4",
                [true, false, false, false, false],
            ),
            (
                "ATLAS_HOLO_MOE_GATEUP_FP4",
                [false, true, false, false, false],
            ),
            ("ATLAS_LORA_EAGER", [false, false, true, false, false]),
            ("ATLAS_LORA_ROTATE", [false, false, false, true, false]),
            ("ATLAS_DIAG_GEMMA4", [false, false, false, false, true]),
        ];
        for (name, expected) in cases {
            let d = resolve(&[(name, "TrUe")]);
            assert_eq!(
                [
                    d.holo_moe_down_fp4,
                    d.holo_moe_gateup_fp4,
                    d.lora_eager,
                    d.lora_rotate,
                    d.gemma4_diag
                ],
                expected,
                "{name}"
            );
        }
        assert!(resolve(&[("ATLAS_BF16_TC_PROJ", "0")]).bf16_tc_proj);
    }

    #[test]
    fn kill_switches_and_zero_opt_outs_keep_their_distinct_polarities() {
        let d = resolve(&[
            ("ATLAS_NO_GDN_REGRESIDENT", "1"),
            ("ATLAS_NO_GEMV_SW", "1"),
            ("ATLAS_GDN_WY17", "0"),
            ("ATLAS_GDN_WYN", "0"),
            ("ATLAS_FFN_SMALLM", "0"),
        ]);
        assert!(!d.gdn_regresident);
        assert!(!d.gemv_sw);
        assert!(!d.gdn_wy17);
        assert!(!d.gdn_wyn);
        assert!(!d.ffn_small_m);
        assert!(resolve(&[("ATLAS_NO_GDN_REGRESIDENT", "0")]).gdn_regresident);
        assert!(resolve(&[("ATLAS_GDN_WY17", "1")]).gdn_wy17);
    }

    #[test]
    fn externally_resolved_shadow_and_drafter_values_are_carried() {
        let d = from_values(
            |_| None,
            |_| false,
            7,
            crate::model::drafter_context::DrafterContext::OFF,
        );
        assert_eq!(d.shadow_topk, 7);
        assert_eq!(
            d.drafter,
            crate::model::drafter_context::DrafterContext::OFF
        );
    }
}
