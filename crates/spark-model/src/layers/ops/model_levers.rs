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
// `Eq` is deliberately absent since `draft_conf_tau` joined: it is an f32
// threshold. Comparing two resolutions is a test-only need and `PartialEq`
// covers it.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
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
    /// `ATLAS_FP32_ROUTING=1` — emit the MoE-input norm in FP32 so the gate
    /// GEMM routes at full precision, removing the bf16-store rounding that
    /// flips experts on gfx1151. Read once per LAYER per DECODE TOKEN from
    /// six call sites via `MoeFfnLayer::fp32_routing_active`, which also
    /// checks four weight/kernel preconditions — the lever is only the last
    /// term of that conjunction, which is why it lives here and the
    /// preconditions stay on the layer.
    pub fp32_routing: bool,
    /// `ATLAS_FP32_GATE=1` — the batched-gate sibling of
    /// [`Self::fp32_routing`].
    pub fp32_gate: bool,
    /// `ATLAS_FRANKENSTEIN_DECODE_VIA_PREFILL=1` — route the five DFlash
    /// capture layers' decode through the PREFILL MoE kernel, on the
    /// hypothesis that the decode MoE kernel is the dominant cause of low
    /// drafter acceptance. ~250 us per capture layer, so ~1.25 ms/token
    /// against a ~58 ms/token decode. Diagnostic; non-capture layers are
    /// untouched.
    pub frankenstein_decode_via_prefill: bool,
    /// `ATLAS_K2_DIAG=1` — K=2 routed-decode diagnostics.
    pub k2_diag: bool,

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
    /// `ATLAS_DFLASH_DEBUG_DUMP_FULL=1` — the model-side half of the DFlash
    /// full dump: emit the whole token sequence ONCE so a Python reference
    /// can run the same tokens through HF transformers.
    ///
    /// ★ The SAME variable that [`crate::layers::dflash_head::levers::DFlashLevers::debug_dump_full`]
    /// carries. Two structs, one flag — deliberately, because the two halves
    /// of the dump are armed together by design and the head is not reachable
    /// from `TransformerModel` (`proposer` is a `dyn DraftProposer`, so there
    /// is nothing to read the head's levers through without a downcast).
    /// `the_two_halves_of_the_dflash_dump_agree` pins that the two
    /// resolutions cannot drift, which is what makes the duplication safe —
    /// an unchecked second spelling of one lever is how
    /// `ATLAS_DSPARK_ANCHOR_BIAS` came to have two implementations.
    pub dflash_debug_dump_full: bool,
    /// `ATLAS_MTP_DEBUG_NORMS=1` — per-stage norm dumps inside the MTP
    /// drafter's `forward_one`, which asked for it FOUR times per drafted
    /// token, each read only to decide whether to do nothing.
    pub mtp_debug_norms: bool,
    /// `ATLAS_MTP_DRAFT_CONF=<t>` — confidence floor for submitting drafts
    /// to verification, clamped to `[0.0, 0.99]`. `0.0` (unset) disables.
    ///
    /// When the drafter's chain confidence (the min top-1 softmax prob
    /// across one propose's drafts) is below this, the drafts are discarded
    /// and the next step decodes serially, skipping a verify that would most
    /// likely reject. Economics at K=1 on the 35B MoE: verify ~35 ms for
    /// 1+accepted tokens against decode+propose ~21 ms for 1, so a draft is
    /// only worth verifying at p(accept) >~ 0.66. STAGED OFF pending its
    /// measured A/B.
    ///
    /// Three of its four readers asked per propose whether the feature was
    /// on, i.e. paid the environment lock to learn it was off. The fourth,
    /// `MtpHead::last_confidence`, is reached only when it is already ON, so
    /// it keeps its own read and its own contract — see the note there.
    pub draft_conf_tau: f32,
    /// `ATLAS_SSM_SAVE_DUMP` (presence) — the CBD scratch/SSM-state
    /// fingerprint probe. Asked THREE times per decode step by the decode
    /// path alone, each read only to decide whether to do nothing.
    pub ssm_save_dump: bool,

    // ── Decode graph capture ──
    /// `ATLAS_EP_GRAPHS=1|true` — allow CUDA-graph capture under expert
    /// parallelism. The EP all-reduce queues ncclSend/Recv plus a local add
    /// on the capture stream and NCCL >= 2.9 supports capture, so this MAY
    /// capture cleanly; env-gated so a deploy can revert instantly if
    /// capture crashes or replay hangs.
    pub ep_graphs: bool,
    /// `ATLAS_GDN_DECODE_GRAPH=1|true` — capture the whole single-token GDN
    /// HeadParallel TP decode forward (~130 kernels plus the per-layer TP
    /// all-reduces) into one replayable graph. Default OFF.
    pub gdn_decode_graph: bool,

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

/// How the levers above are READ. Private: the environment is touched in
/// exactly one place, and `ModelLevers::get` is the only way out of it.
#[path = "model_levers_resolve.rs"]
mod resolve;

#[cfg(test)]
#[path = "model_levers_tests.rs"]
mod tests;

/// ★ Where the environment may be read at all. Kept next to the levers it
/// exists to protect, not inside `tests` — it guards other modules too.
#[cfg(test)]
#[path = "hot_path_env_guards.rs"]
mod hot_path_env_guards;
