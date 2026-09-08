// SPDX-License-Identifier: AGPL-3.0-only

//! How [`ModelLevers`] is READ — the resolution table and the three
//! constructors, split from what the levers ARE.
//!
//! Split out of `model_levers.rs` at 483 lines to stay under the repository's
//! 500-LoC cap. The seam is deliberate rather than arbitrary: the parent file
//! is now the DECLARATION — one documented field per lever, which is what a
//! reader looking for "what does this flag do" wants — and this file is the
//! single place that touches the environment, which is what a reader looking
//! for "how is it spelled" wants. `from_values` stays pure over two closures
//! so tests drive the production resolution instead of a copy of it.

use super::ModelLevers;
use crate::layers::ops::gemv_sw;

pub(super) fn from_values(
    mut value: impl FnMut(&str) -> Option<String>,
    mut present: impl FnMut(&str) -> bool,
    shadow_topk: usize,
    drafter: crate::model::drafter_context::DrafterContext,
    // ★ Passed IN, like `shadow_topk` and `drafter`, NOT read here. Calling
    // `speculative::draft_conf_tau()` from inside this function broke its
    // purity: it reads the real environment whatever the closures say, so a
    // sibling test that set the variable made `resolve(&[])` return 0.99 and
    // `the_opt_out_lever_is_on_by_default_and_every_opt_in_is_off` failed
    // under parallel test execution. `from_values` is pure over its inputs;
    // that is the property the whole test suite rests on.
    draft_conf_tau: f32,
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
    /// `"1"` or `"true"` EXACTLY — case-SENSITIVE, unlike [`opt_in_truthy`].
    ///
    /// The two decode-graph levers spelled it `is_ok_and(|v| v == "1" || v ==
    /// "true")`, and widening them to accept `TRUE` would arm an experimental
    /// CUDA-graph capture on a spelling that previously did nothing — the
    /// direction that turns capture ON unexpectedly. Preserved rather than
    /// unified; the divergence between the two helpers is the point.
    fn opt_in_truthy_exact(value: Option<&str>) -> bool {
        matches!(value, Some("1") | Some("true"))
    }

    ModelLevers {
        max_decode_seqs: 1,
        shadow_topk,
        kv_poison: opt_in(value("ATLAS_KV_POISON").as_deref()),
        drafter,
        gdn_regresident: value("ATLAS_NO_GDN_REGRESIDENT").as_deref() != Some("1"),
        gdn_batched_fla: opt_in(value("ATLAS_GDN_BATCHED_FLA").as_deref()),
        gdn_wy17: opt_out(value("ATLAS_GDN_WY17").as_deref()),
        gdn_wyn: opt_out(value("ATLAS_GDN_WYN").as_deref()),
        ffn_small_m: opt_out(value("ATLAS_FFN_SMALLM").as_deref()),
        gemv_sw: gemv_sw::gemv_sw_from(value("ATLAS_NO_GEMV_SW").as_deref()),
        decode_ffn_via_gemm: opt_in(value("ATLAS_DECODE_FFN_VIA_GEMM").as_deref()),
        holo_moe_down_fp4: opt_in_truthy(value("ATLAS_HOLO_MOE_DOWN_FP4").as_deref()),
        holo_moe_gateup_fp4: opt_in_truthy(value("ATLAS_HOLO_MOE_GATEUP_FP4").as_deref()),
        moe_union_stats: opt_in(value("ATLAS_MOE_UNION_STATS").as_deref()),
        fp32_routing: opt_in(value("ATLAS_FP32_ROUTING").as_deref()),
        fp32_gate: opt_in(value("ATLAS_FP32_GATE").as_deref()),
        frankenstein_decode_via_prefill: opt_in(
            value("ATLAS_FRANKENSTEIN_DECODE_VIA_PREFILL").as_deref(),
        ),
        k2_diag: opt_in(value("ATLAS_K2_DIAG").as_deref()),
        dflash_debug_dump_full: opt_in(value("ATLAS_DFLASH_DEBUG_DUMP_FULL").as_deref()),
        mtp_debug_norms: opt_in(value("ATLAS_MTP_DEBUG_NORMS").as_deref()),
        draft_conf_tau,
        decode_split_silu: !present("ATLAS_NO_DECODE_SPLIT_SILU"),
        bf16_tc_prefill: present("ATLAS_BF16_TC_PREFILL"),
        fp8_m64_prefill: present("ATLAS_FP8_M64_PREFILL"),
        int8_prefill: present("ATLAS_INT8_PREFILL"),
        int8_faith5: present("ATLAS_INT8_FAITH5"),
        ffn_nvfp4_mmq: !present("ATLAS_NO_FFN_NVFP4_MMQ"),
        ffn_nvfp4_mmq_down: !present("ATLAS_NO_FFN_NVFP4_MMQ_DOWN"),
        ffn_mmq: present("ATLAS_FFN_MMQ"),
        ffn_mmq_down_q4k: present("ATLAS_FFN_MMQ_DOWN_Q4K"),
        fp4_prefill: present("ATLAS_FP4_PREFILL"),
        prefill_v2: !present("ATLAS_DISABLE_PREFILL_V2"),
        moe_grouped_cutlass: opt_in(value("ATLAS_HOLO_MOE_GROUPED_CUTLASS").as_deref()),
        moe_grouped_down: opt_in(value("ATLAS_HOLO_MOE_GROUPED_DOWN").as_deref()),
        moe_prefill_exact_tiles: match value("ATLAS_MOE_PREFILL_EXACT_TILES").as_deref() {
            Some("0") => Some(false),
            Some("1") => Some(true),
            _ => None,
        },
        moe_prefill_max_load_factor: value("ATLAS_MOE_PREFILL_MAX_LOAD_FACTOR")
            .as_deref()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&factor| factor > 0),
        moe_prefill_zero: opt_in(value("ATLAS_MOE_PREFILL_ZERO").as_deref()),
        moe_prefill_fp8_down: opt_in(value("ATLAS_MOE_PREFILL_FP8_DOWN").as_deref()),
        dflash_contig_attn: opt_in(value("ATLAS_DFLASH_CONTIG_ATTN").as_deref()),
        lora_eager: opt_in_truthy(value("ATLAS_LORA_EAGER").as_deref()),
        lora_rotate: opt_in_truthy(value("ATLAS_LORA_ROTATE").as_deref()),
        k4_diag: opt_in(value("ATLAS_K4_DIAG").as_deref()),
        gemma4_diag: opt_in_truthy(value("ATLAS_DIAG_GEMMA4").as_deref()),
        ssm_save_dump: present("ATLAS_SSM_SAVE_DUMP"),
        ep_graphs: opt_in_truthy_exact(value("ATLAS_EP_GRAPHS").as_deref()),
        gdn_decode_graph: opt_in_truthy_exact(value("ATLAS_GDN_DECODE_GRAPH").as_deref()),
        bf16_tc_proj: present("ATLAS_BF16_TC_PROJ"),
        weight_pre_rotated: opt_in_truthy(value("TQ_PLUS_WEIGHT_ROTATION").as_deref()),
        ssm_ms_profile: opt_in(value("ATLAS_SSM_MS_PROFILE").as_deref()),
        ssm_detail_profile: opt_in(value("ATLAS_SSM_DETAIL_PROFILE").as_deref()),
        ssm_gemv_batch4: opt_out(value("ATLAS_SSM_GEMV_BATCH4").as_deref()),
        gdn_fused_conv: opt_in(value("ATLAS_GDN_FUSED_CONV").as_deref()),
        moe_legacy_pertoken_decode: opt_in(value("ATLAS_MOE_LEGACY_PERTOKEN_DECODE").as_deref()),
    }
}

impl ModelLevers {
    /// The process-wide levers, resolved from the environment EXACTLY ONCE.
    ///
    /// ★ USE THIS, NOT [`Self::from_env`]. Every field here is a pure function
    /// of `ATLAS_*` environment variables, which cannot change after start —
    /// the runtime `set_var` that could have changed them was deliberately
    /// removed. So this is a process constant and must be computed once.
    ///
    /// It was not. `from_env` reads ~30 environment variables, each allocating
    /// a `String`, and three call sites invoked it from hot paths.
    /// MEASURED: 32,513 resolutions in a single `concurrency-sweep` — which
    /// matches 48 layers x ~680 prefills, i.e. once per layer per prefill from
    /// `qwen3_attention::prefill_weights`. Each of those also re-ran
    /// `drafter_context::resolve_from_env` and its logging.
    ///
    /// Returns a reference so callers cannot accidentally keep re-resolving;
    /// `ModelLevers` is `Copy`, so `*ModelLevers::get()` is free when an owned
    /// value is wanted.
    pub fn get() -> &'static Self {
        static LEVERS: std::sync::OnceLock<ModelLevers> = std::sync::OnceLock::new();
        LEVERS.get_or_init(Self::from_env)
    }

    /// Resolve from the environment, unconditionally.
    ///
    /// Prefer [`Self::get`]. This exists for the one caller that needs an OWNED,
    /// MUTABLE copy — the model build overwrites `max_decode_seqs` with the
    /// batch size — and for tests that want a fresh read. Calling it in a hot
    /// path re-reads every `ATLAS_*` variable.
    pub fn from_env() -> Self {
        from_values(
            |var| std::env::var(var).ok(),
            |var| std::env::var_os(var).is_some(),
            crate::speculative::shadow_topk(),
            crate::model::drafter_context::resolve_from_env(),
            crate::speculative::draft_conf_tau(),
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
            // Opt-out: ships ON, `ATLAS_SSM_GEMV_BATCH4=0` disables. Every
            // opt-out lever must appear here or
            // `the_opt_out_lever_is_on_by_default_and_every_opt_in_is_off`
            // fails — which is exactly how this line came to be written.
            ssm_gemv_batch4: true,
            // The dense-FFN opt-outs. Each ships ON and is disabled by the
            // PRESENCE of its variable, at any value — `=0` does not
            // re-enable them, which is why they are listed here explicitly
            // rather than left to `Default`.
            decode_split_silu: true,
            ffn_nvfp4_mmq: true,
            ffn_nvfp4_mmq_down: true,
            prefill_v2: true,
            ..Self::default()
        }
    }
}
