// SPDX-License-Identifier: AGPL-3.0-only

//! Model-side kernel-path levers, resolved once and then carried.
//!
//! # ★ THE ENVIRONMENT IS READ EXACTLY ONCE PER PROCESS. KEEP IT THAT WAY.
//!
//! Every `ATLAS_*` variable below is a process constant: nothing mutates the
//! environment after start (the runtime `set_var` that once could was
//! deliberately removed — see `main_modules/serve_load.rs` and `config.rs`).
//! So resolving them more than once is pure waste, and on a hot path it is
//! worse than waste:
//!
//! * `std::env::var` allocates a `String` per read, and
//! * it takes the PROCESS-WIDE environment lock, so concurrent readers
//!   SERIALISE against each other.
//!
//! MEASURED on GB10: one resolve of the ~30 variables here costs 0.57 us
//! single-threaded but **4.00 us at 8 threads and 5.76 us at 16** — the cost
//! grows with concurrency, which makes it invisible to any single-stream
//! benchmark. `from_env()` was called **32,513 times in one
//! `concurrency-sweep`** (48 layers x ~680 prefills) while its own doc claimed
//! it was "called once, when the model is built".
//!
//! **The rule for this module and anything like it:** read the environment in
//! ONE place, at ONE time, and pass the resolved value down. Use
//! [`ModelLevers::get`] for the process-wide copy; take `levers` from the
//! `ForwardContext` or the model when you already have one. If you find
//! yourself calling anything named `*_from_env`, `resolve_*` or `*_env()`
//! inside a function that runs per token, per layer, per forward pass or per
//! request, that is the bug this note exists to prevent.
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

    // ── Dense FFN: which GEMM each prefill/decode arm takes ──
    //
    // These twelve were read with `std::env::var_os` from inside
    // `DenseFfnLayer::forward` and `forward_prefill_inner`, i.e. once per
    // LAYER per decode token and once per layer per prefill chunk — and one
    // of them from inside a per-GEMM macro, so three times per layer. No
    // allocation (that is `var_os`'s advantage over `var`) but the same
    // process-wide environment lock, which serialises concurrent decode
    // threads. The load-time readers in `finalize_q4k_load` and
    // `finalize_nvfp4_mmq_load` are deliberately left where they are: they
    // run once per weight, at load.
    /// Split SiLU+down on the decode path: `silu_mul` into `gate_out`, then a
    /// separate `w4a16_decode_gemv` for down. Ships ON;
    /// `ATLAS_NO_DECODE_SPLIT_SILU` (presence) restores the fused kernel.
    /// A LoRA adapter pins this path on regardless — the fused alternative
    /// never materialises `silu(gate)*up`, which the down delta must
    /// contract over — so the call site is `levers.decode_split_silu ||
    /// self.lora.is_some()`.
    pub decode_split_silu: bool,
    /// `ATLAS_BF16_TC_PREFILL` (presence) — BF16 tensor-core prefill GEMM.
    /// Read here only; the usable gate is derived at the call site AFTER
    /// v1/v2 selection, from the handle actually launched. Gating on v1's
    /// handle while dispatching v2 admitted launches of a kernel the target
    /// may not carry.
    pub bf16_tc_prefill: bool,
    /// `ATLAS_FP8_M64_PREFILL` (presence) — m16n8k32 e4m3 M64 prefill GEMM,
    /// ~1.47x vs v2 BF16. Lossy (cosine 0.9997), so opt-in only.
    pub fp8_m64_prefill: bool,
    /// `ATLAS_INT8_PREFILL` (presence) — requant→`int8_gemm_faith2` prefill
    /// (cosine 0.999978 vs the host full-precision dequant GEMM).
    pub int8_prefill: bool,
    /// `ATLAS_INT8_FAITH5` (presence) — int32 per-sub-block accumulation,
    /// which breaks the MMA→scale dependency chain. Same kernel signature
    /// and launch geometry as faith2, so it is a handle swap.
    pub int8_faith5: bool,
    /// Vendored llama NVFP4 W4A4 MMQ for the gate/up prefill GEMMs
    /// (~80 TFLOP/s vs t_m128's ~51). Ships ON;
    /// `ATLAS_NO_FFN_NVFP4_MMQ` (presence) is the kill switch.
    pub ffn_nvfp4_mmq: bool,
    /// The same MMQ arm for the down projection — t_m128 runs the narrow-N
    /// down at only ~34 TFLOP/s. Ships ON; `ATLAS_NO_FFN_NVFP4_MMQ_DOWN`
    /// (presence) is the kill switch. Separate from
    /// [`Self::ffn_nvfp4_mmq`] because down is the heavy-tailed projection
    /// (W4A4 cosine 0.9961) and gets its own gate.
    pub ffn_nvfp4_mmq_down: bool,
    /// `ATLAS_FFN_MMQ` (presence) — Q4_K MMQ prefill arm.
    pub ffn_mmq: bool,
    /// `ATLAS_FFN_MMQ_DOWN_Q4K` (presence) — keep the down projection ON
    /// Q4_K instead of the near-lossless faith2 NVFP4 hybrid.
    ///
    /// Stores the POSITIVE of a variable whose call site reads the negative
    /// (`!levers.ffn_mmq_down_q4k`), the same shape as
    /// [`Self::moe_legacy_pertoken_decode`]. down = SiLU(gate)*up is
    /// heavy-tailed and Q4_K superblock scaling clips it — BFCL `multiple`
    /// −4.0%, which is why llama promotes only down→Q6_K.
    pub ffn_mmq_down_q4k: bool,
    /// `ATLAS_FP4_PREFILL` (presence) — native W4A4 FP4 tensor cores
    /// (sm_121a), NVFP4 weights used directly with no requant. Lossy
    /// (cos ~0.99 vs fp32).
    pub fp4_prefill: bool,
    /// The v2 BF16 t_m128 prefill kernel — faster and bit-identical to v1.
    /// Ships ON; `ATLAS_DISABLE_PREFILL_V2` (presence) forces v1 so the two
    /// can be compared for TTFT in one binary.
    pub prefill_v2: bool,

    // ── MoE routed prefill ──
    //
    // Read once per LAYER per prefill chunk from `forward_prefill_routed`,
    // and the CUTLASS gate is asked TWICE per call through a free function.
    /// `ATLAS_HOLO_MOE_GROUPED_CUTLASS=1` — single-launch CUTLASS grouped
    /// NVFP4 gate_up. Off by default; unset falls back to the hand-rolled
    /// fused FP4/FP8 grouped kernels.
    pub moe_grouped_cutlass: bool,
    /// `ATLAS_HOLO_MOE_GROUPED_DOWN=1` — take the down projection through
    /// the same CUTLASS grouped path. Requires
    /// [`Self::moe_grouped_cutlass`]; a separate gate because down consumes
    /// the already-expert-contiguous post-SiLU output and needs no gather.
    pub moe_grouped_down: bool,
    /// `ATLAS_MOE_PREFILL_EXACT_TILES=1|0` overrides the tile bound;
    /// `None` (unset) defers to the checkpoint — the win was measured on
    /// NVFP4, so the default is scoped to where it was measured.
    ///
    /// Tri-state on purpose. Measured: exact_tiles ON gave p90 +4.9% against
    /// a +5.0% limit (0.1% from failing the gate) and OFF gave p90 −5.0%,
    /// while the median barely moved either way (+0.1% vs −0.9%). Only the
    /// tail shows it, so both directions must stay reachable. Graph capture
    /// forces it off regardless — the bound is read back from device memory.
    pub moe_prefill_exact_tiles: Option<bool>,
    /// `ATLAS_MOE_PREFILL_MAX_LOAD_FACTOR=<n>` — cap the per-expert tile
    /// bound at n times the average when exact tiles are off. `None` (unset
    /// or `0`) means the worst case.
    pub moe_prefill_max_load_factor: Option<usize>,
    /// `ATLAS_MOE_PREFILL_ZERO=1` — memset the grouped scratch before
    /// dispatch. Implied by EP (`ctx.comm.is_some()`). In non-EP the sort
    /// produces a dense permutation over exactly the rows the grouped
    /// kernels write, so skipping the clear removes ~138 MB/layer on Holo.
    pub moe_prefill_zero: bool,
    /// `ATLAS_MOE_PREFILL_FP8_DOWN=1` — FP8 grouped GEMM for the routed
    /// down projection.
    pub moe_prefill_fp8_down: bool,

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
    /// The checkpoint's attention weights are ALREADY Hadamard-rotated at load
    /// (`TQ_PLUS_WEIGHT_ROTATION`), so the runtime must not rotate again.
    ///
    /// A property of the loaded checkpoint, and the SSOT for it. It previously
    /// had FIVE implementations of the same `=1`-or-`true` test — four raw
    /// `std::env::var` calls on attention paths (one per attention layer per
    /// DECODE TOKEN in `decode/attention_forward.rs`, one per layer per
    /// batched decode step in `multi_seq/attn.rs`, two per layer per prefill
    /// chunk) plus a fifth in the weight loader, whose `#[allow(dead_code)]`
    /// was stale — `attention_arms.rs` calls it. Reading this per token cost
    /// an allocation and the process-wide environment lock on the hottest path
    /// in the model, and five copies of one predicate is how a flag ends up
    /// decoded two different ways in one binary.
    pub weight_pre_rotated: bool,

    // ── SSM / GDN decode ──
    // These five ran on the batched-decode path — per SSM layer per decode
    // step, ~6-7M environment reads per sweep on a 36-SSM-layer hybrid, the
    // largest raw count in this crate. Three are diagnostics that are off in
    // every shipped configuration, and were paying a `String` allocation and
    // the process-wide environment lock to say so on every layer of every
    // token. Their neighbour `ssm_tc_proj_min_n()` in the same file was
    // already `OnceLock`'d with the note "Read ONCE — this site runs under
    // graph capture", so these were an inconsistency, not a design.
    /// Per-step multi-sequence SSM profiling dump.
    pub ssm_ms_profile: bool,
    /// Finer per-sub-step SSM profiling inside the batched recurrence.
    pub ssm_detail_profile: bool,
    /// Ships ON: use the batch-4 GEMV tier for the SSM projections when the
    /// kernel is resolved and n <= 16. `ATLAS_SSM_GEMV_BATCH4=0` opts out.
    pub ssm_gemv_batch4: bool,
    /// Fuse the GDN conv with the F32 norm when the head geometry allows.
    pub gdn_fused_conv: bool,
    /// Take the pre-token-major MoE decode kernel. The field stores the
    /// POSITIVE of the variable's name, so the call site reads
    /// `!levers.moe_legacy_pertoken_decode` for the default token-major path —
    /// the inversion lives here, once, rather than at the branch.
    pub moe_legacy_pertoken_decode: bool,
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

#[cfg(test)]
#[path = "model_levers_tests.rs"]
mod tests;

/// ★ Where the environment may be read at all. Kept next to the levers it
/// exists to protect, not inside `tests` — it guards other modules too.
#[cfg(test)]
#[path = "hot_path_env_guards.rs"]
mod hot_path_env_guards;
