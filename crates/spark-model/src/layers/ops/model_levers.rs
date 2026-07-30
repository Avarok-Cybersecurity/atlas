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
}

/// Opt-IN: off unless the variable is exactly `1`.
fn opt_in(var: &str) -> bool {
    std::env::var(var).ok().as_deref() == Some("1")
}

/// Opt-OUT: on unless the variable is exactly `0`.
fn opt_out(var: &str) -> bool {
    std::env::var(var).ok().as_deref() != Some("0")
}

/// Opt-IN accepting `true` as well as `1` — the LoRA levers' original spelling.
fn opt_in_truthy(var: &str) -> bool {
    std::env::var(var).is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

impl ModelLevers {
    /// Resolve from the environment. Called once, when the model is built.
    pub fn from_env() -> Self {
        Self {
            // `ATLAS_NO_GDN_REGRESIDENT` is a kill switch, so the variable is
            // negative and the field is positive: `!= "1"` means on.
            gdn_regresident: std::env::var("ATLAS_NO_GDN_REGRESIDENT").as_deref() != Ok("1"),
            gdn_batched_fla: opt_in("ATLAS_GDN_BATCHED_FLA"),
            // Opt-OUT — these three ship ON.
            gdn_wy17: opt_out("ATLAS_GDN_WY17"),
            gdn_wyn: opt_out("ATLAS_GDN_WYN"),
            ffn_small_m: opt_out("ATLAS_FFN_SMALLM"),
            decode_ffn_via_gemm: opt_in("ATLAS_DECODE_FFN_VIA_GEMM"),
            holo_moe_down_fp4: opt_in("ATLAS_HOLO_MOE_DOWN_FP4"),
            holo_moe_gateup_fp4: opt_in("ATLAS_HOLO_MOE_GATEUP_FP4"),
            moe_union_stats: opt_in("ATLAS_MOE_UNION_STATS"),
            dflash_contig_attn: opt_in("ATLAS_DFLASH_CONTIG_ATTN"),
            lora_eager: opt_in_truthy("ATLAS_LORA_EAGER"),
            lora_rotate: opt_in_truthy("ATLAS_LORA_ROTATE"),
            k4_diag: opt_in("ATLAS_K4_DIAG"),
        }
    }

    /// What a build resolves to with no `ATLAS_*` set — every opt-in off, the
    /// one opt-out lever on. Tests construct a context with this instead of
    /// mutating the process environment.
    pub fn defaults() -> Self {
        Self {
            gdn_regresident: true,
            gdn_wy17: true,
            gdn_wyn: true,
            ffn_small_m: true,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_opt_out_lever_is_on_by_default_and_every_opt_in_is_off() {
        let d = ModelLevers::defaults();
        // Four levers ship ON. Getting one of these senses backwards is a
        // silent behaviour change, which is why they are pinned here.
        assert!(d.gdn_regresident, "PR #369 folded this default-on");
        assert!(d.gdn_wy17 && d.gdn_wyn, "opt-OUT via =0");
        assert!(d.ffn_small_m, "opt-OUT via =0");
        assert!(!d.gdn_batched_fla);
        assert!(!d.decode_ffn_via_gemm);
        assert!(!d.lora_eager);
        assert!(!d.k4_diag);
    }

    #[test]
    fn derive_default_is_not_the_shipped_default() {
        // Guard against reaching for `ModelLevers::default()` and silently
        // turning off a lever that ships on. `defaults()` is the shipped shape.
        assert_ne!(ModelLevers::default(), ModelLevers::defaults());
        assert!(!ModelLevers::default().gdn_regresident);
    }

    #[test]
    fn two_models_can_hold_different_levers() {
        // The property a static could not have.
        let a = ModelLevers::defaults();
        let b = ModelLevers {
            gdn_batched_fla: true,
            ..ModelLevers::defaults()
        };
        assert_ne!(a, b);
        assert!(!a.gdn_batched_fla && b.gdn_batched_fla);
    }
}
