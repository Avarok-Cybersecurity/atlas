// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash block-diffusion draft head implementing [`DraftProposer`].
//!
//! Block-diffusion drafter (Z Lab, arXiv 2602.06036): a small Qwen3-architecture
//! transformer (8 layers, hidden=2048, GQA 32:4, head_dim=128) that emits γ=16
//! tokens **in a single forward pass** via bidirectional in-block attention.
//! Conditioned on five intermediate hidden states captured from the target
//! model at `target_layer_ids` (e.g., `[1, 10, 19, 28, 37]` for
//! Qwen3.6-35B-A3B-DFlash), projected through a single `fc` layer at model
//! entry — NOT per-layer KV injection (early plan was wrong; cf. vLLM
//! `qwen3_dflash.py`).
//!
//! Phase 1 deliverable: type + trait wiring. The actual γ-block forward kernel
//! (`inferspark_dflash_block_attn_fp8`) lands in Phase 2; until then `propose()`
//! returns the bonus token repeated `num_drafts` times so the verify path
//! degenerates to single-token decode (acceptance ~100% but no speedup).

use parking_lot::Mutex;
use std::any::Any;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_map::{DenseWeight, QuantizedWeight};

/// Kernel handles for the DFlash γ-block forward chain. All resolved once
/// at `BlockDiffusionDraftHead::from_weights` against the active GPU backend
/// (which compiles target-specific PTX at startup); subsequent
/// `propose()` calls just `KernelLaunch::new(...).launch(stream)`.
pub struct DflashKernels {
    pub rms_norm: KernelHandle,
    pub residual_rms_norm: KernelHandle,
    pub dense_gemv: KernelHandle,
    pub dense_gemm: KernelHandle,
    /// NVFP4 GEMM for the final logits when the shared lm_head is NVFP4
    /// (e.g. Holo): a BF16 `dense_gemm` on NVFP4-packed bytes reads garbage
    /// (and ~4× OOB → CUDA-700). `.0 == 0` when the target lm_head is BF16.
    pub w4a16_gemm: KernelHandle,
    pub dense_gemm_pipelined: KernelHandle,
    pub rope_qwen3: KernelHandle,
    pub reshape_cache_fp8: KernelHandle,
    /// BF16 KV cache writeback. Used by Phase 2 `precompute_ctx_kv` and
    /// the per-layer γ-block `reshape_and_cache` call to populate the
    /// drafter's BF16 paged cache before each `prefill_attention_paged_dflash`.
    pub reshape_cache_bf16: KernelHandle,
    pub prefill_attn_dflash_fp8: KernelHandle,
    /// BF16 paged-attention dispatcher for the DFlash γ-block.
    /// Calls `inferspark_prefill_paged` with `causal_mask_enabled=0`,
    /// reading BF16 K/V from the per-layer paged cache pool. Phase 2
    /// (Option B) drafter attention runs through this kernel; the FP8
    /// variant above is retained for a future quality-validated FP8 KV
    /// path. See `ops::prefill_attention_paged_dflash`.
    pub prefill_attn_dflash_bf16: KernelHandle,
    /// Phase 5 (CUDA graph) variant of `prefill_attn_dflash_bf16` that reads
    /// `kv_len` and `q_offset` from device pointers instead of taking them as
    /// kernel scalar args. Used by the graph-captured forward_block path so a
    /// single graph instance can be replayed across steps with different
    /// dynamic values written to the indirect-args buffer pre-launch.
    /// Resolves to kernel `inferspark_prefill_paged_indirect`.
    pub prefill_attn_dflash_bf16_indirect: KernelHandle,
    pub silu_mul: KernelHandle,
    pub residual_add: KernelHandle,
    pub argmax: KernelHandle,
    pub batched_embed: KernelHandle,
    /// Phase 2 Option B: builds `[count]` i32 slot indices on-device
    /// from a host-provided block_table. Used by propose.rs to populate
    /// the slot_mapping passed to reshape_and_cache and precompute_ctx_kv.
    pub fill_slots: KernelHandle,
    /// Non-paged prefill attention (used for the γ-block self-attention
    /// when there's no persistent K/V cache to walk).
    pub prefill_attn: KernelHandle,
    /// Phase G — BF16 → FP8 E4M3 per-row weight quantization. Used at
    /// model load time to convert the seven dense-GEMM drafter weights
    /// (q/k/v/o/gate/up/down) when `ATLAS_DFLASH_DRAFTER_FP8=1`. Never
    /// on the hot path.
    pub quantize_bf16_to_fp8: KernelHandle,
    /// Phase G — Row-scaled BF16 × FP8 → BF16 GEMM. Consumes the
    /// `Fp8DenseWeight` (FP8 weight + per-row f32 scale) produced at
    /// load time by `quantize_bf16_to_fp8`. Wraps
    /// `kernels/gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu fp8_gemm_t_row_scaled`.
    /// Replaces `dense_gemm_bf16` on the seven dense-GEMM call sites in
    /// `forward_block_layer_pre_attn` / `_post_attn` when
    /// `self.quant == DflashQuantization::Fp8Weights`.
    pub fp8_gemm_n128_row_scaled: KernelHandle,
    /// Phase G — Row-scaled BF16 × FP8 → BF16 GEMV (M=1) for the
    /// lm_head fall-back. At γ=16 vs vocab=248320 the row-scaled GEMM
    /// wastes 75% of its M_TILE; the GEMV in a γ-loop is faster.
    pub dense_gemv_fp8w: KernelHandle,
    /// Phase G — Small-M (M≤16) row-scaled FP8 GEMM. Drop-in replacement
    /// for `fp8_gemm_n128_row_scaled` when M=γ=16. Single warp per CTA,
    /// no wasted M_TILE rows. Used by the lm_head GEMM.
    pub fp8_gemm_n128_row_scaled_m16: KernelHandle,
}

/// Per-step scratch buffers for the γ-block forward.
///
/// Sized for `n_attn_slots = ctx_window + γ` rows, where ctx_window is the
/// max number of past target positions the drafter attends to per step. The
/// first `ctx_window` slots hold post-`fc` projected target context (K/V
/// only — Q is zero-padded); the next γ slots hold the noise tokens.
///
/// At γ=16 and ctx_window=γ=16: 32 rows × 2048 BF16 × ~10 buffers = ~1.3 MB
/// per head. lm_head logits buffer is the largest single alloc:
/// 32 × 248320 × 2 = 15 MB.
pub struct DflashScratch {
    pub stream_buf: DevicePtr,
    pub norm_buf: DevicePtr,
    pub q_buf: DevicePtr,
    pub k_buf: DevicePtr,
    pub v_buf: DevicePtr,
    pub attn_out: DevicePtr,
    pub mlp_intermediate: DevicePtr,
    pub mlp_up: DevicePtr,
    pub stream_acc: DevicePtr,
    /// `[ctx_window, draft_hidden]` BF16 — fc-projected + hidden_norm'd
    /// ctx for the most recent `ctx_window` target positions.
    pub fc_proj: DevicePtr,
    /// Phase 2 (Option B) scratch for `precompute_ctx_kv`: fused KV
    /// GEMM output, shape `[max_new_ctx, L * 2 * kv_dim]` BF16.
    /// `max_new_ctx` = `ctx_window` (worst case: first propose runs
    /// precompute over the entire prefix).
    pub fused_kv_out: DevicePtr,
    /// Phase 2 scratch: i32 slot mapping for the per-layer
    /// `reshape_and_cache` calls. Sized `[ctx_window]`.
    pub slot_mapping_dev: DevicePtr,
    /// Phase 5 (CUDA graph) scratch: 8 bytes (`[u32 kv_len, u32 q_offset]`)
    /// holding the per-call dynamic values that the indirect paged-attention
    /// kernel reads at entry. Host writes via `copy_h2d` BEFORE entering the
    /// captured region so the graph itself sees a stable device pointer.
    pub option_b_indirect_args_dev: DevicePtr,
    /// Phase E.2: pinned host buffer (`γ × 4` bytes) for the per-propose
    /// draft-token D2H copy. Allocated once at construction via
    /// `gpu.alloc_host_pinned`; the async D2H lands here without touching
    /// the system pageable allocator each call.
    ///
    /// Wrapped in `AtomicPtr` to keep `DflashScratch: Send + Sync` (the
    /// proposer is stored as `Arc<dyn DraftProposer>` which requires both
    /// auto-traits). Reads via `Ordering::Relaxed` are safe: the pointer
    /// itself never changes after construction; we only need atomic
    /// access for the Send/Sync bound, not for any actual concurrency.
    pub draft_tokens_host_pinned: std::sync::atomic::AtomicPtr<u8>,
    /// Phase E.2: CUDA event recorded against the draft-tokens D2H so the
    /// host can block on completion just before reading the pinned buffer,
    /// without a full `cuStreamSynchronize`. Created once at construction.
    pub draft_tokens_event: u64,
    pub logits: DevicePtr,
    pub draft_tokens_dev: DevicePtr,
    /// `[ctx_window + γ]` i32 positions. First ctx_window are
    /// historical target positions (decoded indices); last γ are
    /// the to-be-predicted noise positions.
    pub position_ids: DevicePtr,
}

/// Drafter-side weight precision. Defaults to BF16. **Phase G (2026-05-28)**
/// adds `Fp8Weights`, gated by env var `ATLAS_DFLASH_DRAFTER_FP8`. The
/// historical SM12.x acceptance collapse note applied to drafter FP8 KV
/// cache (different concern — bidirectional attention math); Phase G
/// targets weight FP8 only, so the risk surface is dynamic-range loss
/// in MLP intermediate activations, which per-row scales mitigate.
/// `--mtp-quantization fp8` is still not honored for the DFlash drafter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DflashQuantization {
    Bf16,
    /// Weight-only FP8: q/k/v/o/gate/up/down BF16 → FP8 E4M3 with per-row
    /// f32 scales at model load. Activations stay BF16; KV cache stays
    /// BF16. GEMMs use `fp8_gemm_n128` (BF16 × FP8 → BF16).
    Fp8Weights,
}

/// Per-drafter-layer Qwen3-style weights. Phase 1 is BF16-only; **Phase G**
/// (2026-05-28) adds optional FP8 weight fields populated at model load
/// when `ATLAS_DFLASH_DRAFTER_FP8=1`. The BF16 fields are always present
/// (Fp8 path falls back to them for any GEMM whose Fp8 weight is None).
#[allow(dead_code)]
pub struct DflashLayer {
    // Norms
    pub input_layernorm: DenseWeight,
    pub post_attention_layernorm: DenseWeight,
    // Attention (Qwen3: per-head Q/K RMSNorm)
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    pub o_proj: DenseWeight,
    pub q_norm: DenseWeight,
    pub k_norm: DenseWeight,
    // MLP
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,

    // Phase G — optional FP8 mirrors of the seven dense-GEMM weights.
    // Populated at load time when `ATLAS_DFLASH_DRAFTER_FP8=1`, consumed
    // by forward_block_layer_pre_attn / _post_attn when self.quant ==
    // DflashQuantization::Fp8Weights. None when BF16 path is active.
    pub q_proj_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    pub k_proj_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    pub v_proj_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    pub o_proj_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    pub gate_proj_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    pub up_proj_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    pub down_proj_fp8: Option<crate::weight_map::Fp8DenseWeight>,
}

/// Per-sequence DFlash drafter state. One paged KV cache per drafter layer
/// (8 typical), shared block table across layers since attention shape is
/// identical layer-to-layer for a vanilla Qwen3 architecture. Mirrors
/// `MtpProposerState` in spirit; the multi-layer cache keeps it distinct.
pub struct DflashProposerState {
    /// Block table for the drafter's KV cache (shared across all drafter layers).
    pub block_table: Vec<u32>,
    /// Current logical sequence length in the drafter's KV cache. Tracks how
    /// many target-aligned positions have been written via
    /// `precompute_and_store_context_kv`.
    pub seq_len: usize,
    /// Drafts produced in the last `propose()` call. `after_verify` consults
    /// this to know how many KV positions to roll back when the accept
    /// prefix is shorter than γ.
    pub last_num_drafted: usize,
    /// Whether the prompt-time `precompute_and_store_context_kv` has been
    /// called. The first `propose()` after model build needs to run prefill
    /// over the full prompt's captured hiddens; subsequent steps incrementally
    /// append the latest accepted tokens' projections.
    pub prefill_done: bool,
    /// Multi-token accumulator for captured target hidden states. Layout:
    /// `[max_ctx_len, 5 * target_hidden]` BF16 packed. The scheduler appends
    /// the model's `dflash_hidden_save` (latest decoded position's 5 hiddens)
    /// into slot `ctx_len` after each successful verify. `propose()` reads
    /// the full populated prefix and projects all positions through `fc`
    /// at forward time. Sized for `max_seq_len` total positions; not
    /// circular — fail-fast if exceeded (drafter can't handle longer
    /// context than allocated).
    pub ctx_hidden_acc: DevicePtr,
    /// Number of populated slots in `ctx_hidden_acc`. Capped at `max_ctx_len`.
    pub ctx_len: usize,
    /// Drafts accepted in the verify that immediately preceded this propose.
    /// Set by `after_verify` so propose can label row-0 with its TRUE position.
    pub last_num_accepted: usize,
    /// EAGLE-fix one-shot: when set, the next `propose()` skips its internal
    /// decode-append because the verify step (K=2 accept) already appended
    /// row 0 + row 1 in EAGLE order before calling propose. Consumed (reset to
    /// false) by propose. Only set under ATLAS_DFLASH_EAGLE_FIX=1.
    pub skip_next_decode_append: bool,
    /// Allocation cap for `ctx_hidden_acc` (in slot count). Mirrors the
    /// `max_seq_len` build arg so we can clamp without re-fetching it.
    pub max_ctx_len: usize,
    /// Width (bytes) of one `ctx_hidden_acc` slot — `5 * target_hidden * bf16`.
    /// Stored to avoid re-deriving on every append.
    pub ctx_slot_bytes: usize,

    // ─── Phase 2 Option B fields (paged KV cache for ctx) ───────────────
    /// Device-side block table for the drafter's paged KV cache. Allocated
    /// once at first propose with enough u32 slots to cover `max_seq_len`
    /// at block_size=16. Read by `prefill_attention_paged_dflash` to map
    /// logical block indices to physical pool block indices. Mirrors the
    /// host-side `block_table` Vec, copied to GPU after each `alloc_block`.
    pub block_table_dev: Option<DevicePtr>,
    /// Number of paged-cache slots populated with ctx K/V for this sequence.
    /// Distinct from `ctx_len` (which counts target_hidden_acc slots). The
    /// drafter writes one ctx K/V slot per accepted target token; the
    /// γ-block then attends over `[0..ctx_count_drafter+γ)`. Bumped by γ
    /// per propose (γ slots written for the noise rows) and trimmed in
    /// `after_verify` by `(γ - num_accepted)`.
    pub ctx_count_drafter: usize,
    /// Cap for `ctx_count_drafter`. Mirrors `block_table.len() * block_size`.
    pub max_ctx_count_drafter: usize,
    /// Phase I — incremental ctx precompute watermark. Number of ctx slots
    /// `[0..ctx_committed)` whose K/V is already valid in the paged cache
    /// from a prior propose. Each step we only precompute the new tail
    /// `[ctx_committed..ctx_len)` instead of rebuilding the whole prefix
    /// (the old O(ctx_len²) waste — see design doc §18). Reset to the
    /// current `ctx_len` on any rewind so stale slots can't be read.
    /// `0` forces a full rebuild (first propose, or the debug escape hatch).
    pub ctx_committed: usize,
    /// Phase I (v2) — per-slot TRUE absolute decoded position, stamped once
    /// when a ctx slot is appended and never recomputed. Indexed by ctx
    /// slot (parallel to `ctx_hidden_acc` slots, len == `ctx_len`). This is
    /// the vLLM convention: a cached token's rope position is fixed at
    /// insert time, so committed slots never go stale when later accepts
    /// shift the live `position`. Replaces the sliding `absolute_start_pos
    /// + i` formula in `precompute_ctx_kv`. Prefill positions are seeded
    /// `0..prompt_len` in `update_dflash_ctx_len_after_prefill`.
    pub ctx_positions: Vec<i32>,
}

impl ProposerState for DflashProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Block-diffusion draft head. Public API is the [`DraftProposer`] trait.
///
/// The drafter shares `embed_tokens` and `lm_head` with the target — these
/// are NOT in the drafter's safetensors checkpoint (verified against
/// `z-lab/Qwen3.6-35B-A3B-DFlash` commit 42d3b34). The constructor takes
/// the target's `embed_tokens_shared` and `lm_head_shared` device pointers
/// at build time and slots them in alongside the drafter's own `fc`,
/// `hidden_norm`, `norm`, and per-layer weights.
#[allow(dead_code)]
pub struct BlockDiffusionDraftHead {
    // Drafter-architecture config (mirrors the drafter's HF config.json).
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub draft_vocab_size: usize,
    pub gamma: usize,
    pub mask_token_id: u32,
    pub window_size: Option<usize>,
    /// `target_layer_ids`. Same data as `TransformerModel::dflash_capture_layers`,
    /// repeated here so the loader is the single source of truth; the model
    /// reads these to size its capture buffer.
    pub target_layer_ids: Vec<usize>,
    /// Target-side hidden_size (used for the `fc` projection input width:
    /// `target_layer_ids.len() * target_hidden_size`).
    pub target_hidden_size: usize,

    // === Weights shared with the target ===
    /// Target's embed_tokens GPU pointer. The drafter's checkpoint has no
    /// own embeddings — both vocab and embedding dim must match the target
    /// (Qwen3.6-35B-A3B-DFlash: vocab=248320, hidden=2048 — same as target).
    pub embed_tokens_shared: DevicePtr,
    /// Target's lm_head GPU pointer. Used for the drafter's per-position
    /// argmax over `[γ, vocab]` logits. Valid only when the target lm_head is
    /// BF16; when `lm_head_nvfp4` is `Some`, the NVFP4 path is used instead.
    pub lm_head_shared: DevicePtr,
    /// Target's NVFP4 lm_head (packed + scales), shared with the drafter for
    /// the final logits GEMM. `Some` when the target ships an NVFP4 lm_head
    /// (e.g. Holo) — required because a BF16 `dense_gemm` on the NVFP4 buffer
    /// reads garbage and OOB. `None` → use the BF16 `lm_head_shared`.
    pub lm_head_nvfp4: Option<QuantizedWeight>,
    /// Phase G — optional FP8 mirror of the shared lm_head weight,
    /// `[vocab_size, hidden_size]` FP8 E4M3 + per-row f32 scales.
    /// Built at model load when `ATLAS_DFLASH_DRAFTER_FP8=1`. Owned by
    /// the drafter (separate allocation from the shared BF16 ptr) since
    /// it must not mutate the target model's lm_head. `None` on the
    /// BF16 path.
    pub lm_head_shared_fp8: Option<crate::weight_map::Fp8DenseWeight>,

    // === Weights from the drafter checkpoint ===
    /// Hidden-norm applied to the projected target context before mixing
    /// with the embedded tokens (Qwen3-DFlash convention; see vLLM
    /// `DFlashQwen3Model.hidden_norm`).
    pub hidden_norm: DenseWeight,
    /// Final RMSNorm before LM head.
    pub norm: DenseWeight,
    /// `fc` projection — `[draft_hidden, target_layer_ids.len() * target_hidden_size]`
    /// BF16. Maps the stack of captured target hiddens to drafter's input space
    /// once at model entry. Replaces the earlier (incorrect) "per-layer KV
    /// injection" design.
    pub fc: DenseWeight,
    /// Optional draft-vocab-id → target-vocab-id remap. `None` when the
    /// drafter shares vocab with the target (Qwen3.6-35B-A3B-DFlash case:
    /// vocab_size == draft_vocab_size == 248320).
    pub draft_id_to_target_id: Option<DevicePtr>,
    /// Drafter transformer layers (8 for Qwen3.6-35B-A3B-DFlash).
    pub layers: Vec<DflashLayer>,

    /// Phase 2 (Option B) fused K/V projection across all L drafter layers.
    /// Shape: `[L × 2 × kv_dim, h]` BF16 — concatenated `[K0; V0; K1; V1; …]`
    /// (per-layer K then V interleaved). Built once at construction by
    /// `copy_d2d`-stitching the per-layer `k_proj.weight` and `v_proj.weight`
    /// pointers from `layers[i]`. Lets `precompute_ctx_kv` derive every
    /// drafter layer's ctx K/V via a single `dense_gemm` of shape
    /// `[new_ctx_count, h] × [h, L·2·kv_dim]` instead of 2·L per-layer GEMMs.
    ///
    /// `None` until Phase 2 lands the build (stage 1: kernel/dispatcher
    /// scaffolding; stage 2: this allocation + the precompute_ctx_kv module;
    /// stage 3: pyref bit-exact diff). Layout (K then V per layer) chosen
    /// to match vLLM's `_fused_kv_weight` in `qwen3_dflash.py:381-389`.
    pub fused_kv_weight: Option<DevicePtr>,

    /// Paged FP8 KV cache. One cache holding all `num_layers` drafter layers,
    /// laid out the same way the target's KV cache is — block-table-keyed,
    /// `num_layers × num_kv_heads × head_dim` per slot. Allocating a single
    /// multi-layer cache (vs. one per drafter layer) matches Atlas's existing
    /// `PagedKvCache` ABI and lets us reuse the existing `reshape_and_cache`
    /// kernel without per-layer dispatch overhead.
    pub kv_cache: Mutex<PagedKvCache>,

    /// Per-step scratch buffers (allocated once at construction, reused).
    pub scratch: DflashScratch,

    /// All kernel handles needed by `propose()` and the eventual prefill
    /// projection (`precompute_and_store_context_kv`).
    pub kernels: DflashKernels,

    /// Per-sequence ctx accumulator capacity (mirrors model's `max_seq_len`).
    /// Used by `alloc_state` to size each new sequence's `ctx_hidden_acc`.
    pub max_seq_len: usize,

    /// Pre-computed yarn inv_freq table (`[head_dim/2]` f32 on GPU).
    /// Drafter rope_scaling: factor=64, beta_fast=32, beta_slow=1,
    /// original_max_position_embeddings=4096 (per drafter config.json).
    pub yarn_inv_freq: DevicePtr,

    /// rope_theta (10000000 for Qwen3.6-DFlash). Stored to pass into the
    /// rope_yarn kernel each step.
    pub rope_theta: f32,

    /// rotary_dim. Drafter uses full-rotation (rotary_dim = head_dim = 128).
    pub rotary_dim: usize,

    /// RMSNorm epsilon (drafter inherits Qwen3 default 1e-6).
    pub rms_norm_eps: f32,

    /// Max number of past target positions injected into the drafter's K/V
    /// per step. Default γ — drafter sees at most γ ctx + γ noise = 2γ
    /// attention positions per step. ctx_window=0 disables ctx conditioning
    /// (degraded quality, ablation only).
    pub ctx_window: usize,

    // === Phase D (CUDA graph capture) → Phase F (piecewise) ===
    /// Per-subgraph captured handles. `None` until warm-up completes and
    /// the first capture pass lands; on the capture pass we fill this
    /// `Vec` with `2 × num_layers + 1` handles laid out as
    /// `[pre_0, post_0, pre_1, post_1, ..., pre_{N-1}, post_{N-1}, tail]`.
    /// Slot index = `layer_idx * 2 + half` for the layer halves
    /// (half = 0 for pre_attn, 1 for post_attn) and `num_layers * 2` for
    /// the tail (final norm + lm_head + argmax). `GraphHandle(0)` is the
    /// "empty capture" sentinel and means that slot replays eager.
    ///
    /// Phase F.2 (2026-05-28): replaces the single full-region capture
    /// with one capture per subgraph. Attention is NEVER captured —
    /// it's the natural sync barrier between captured subgraphs
    /// (vLLM piecewise convention). See design doc §15.
    pub propose_graphs: Mutex<Option<Vec<spark_runtime::gpu::GraphHandle>>>,
    /// When set, all `forward_block` calls run eagerly. Mirrors target-model
    /// `TransformerModel::suppress_graphs` so external code can disable
    /// graphs at runtime (e.g. while calibrating FP8 KV).
    pub suppress_graphs: std::sync::atomic::AtomicBool,
    /// How many eager warm-up calls we've executed against the graph path.
    /// Default warmup target is 2 (override via `ATLAS_DFLASH_PROPOSE_WARMUP_N`).
    /// Two eager passes warm the PTX→SASS cache, ramp GB10 clocks to steady
    /// state, and bring hot weight tiles into L2 before the capture freezes
    /// SASS variants the driver picks. Shared across all subgraphs — every
    /// subgraph captures on the same propose call after the warmup target
    /// is hit.
    pub propose_warmup_count: std::sync::atomic::AtomicUsize,

    // Quantization mode (BF16 only for Phase 1).
    pub quant: DflashQuantization,
}

mod forward_block;
mod forward_block_layer;
mod forward_block_layer_paged;
mod from_weights;
mod precompute_ctx_kv;
mod propose;

impl DraftProposer for BlockDiffusionDraftHead {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        // Per-seq ctx accumulator: `[max_seq_len, 5 * target_hidden] BF16`.
        // Sized once, re-used across the seq's lifetime; reset on
        // `free_state`. At max_seq_len=16384 and 5×2048 BF16: 320 MB per
        // seq — tolerable on a single Spark with max_batch_size=1; for
        // higher batch we may want to reduce to a smaller working window.
        let bf16 = 2usize;
        let ctx_slot_bytes = self.target_layer_ids.len() * self.target_hidden_size * bf16;
        let total = self.max_seq_len * ctx_slot_bytes;
        let ctx_hidden_acc = gpu.alloc(total)?;
        // Initialize to zero so stale data doesn't leak between sequences.
        gpu.memset(ctx_hidden_acc, 0, total)?;
        Ok(Box::new(DflashProposerState {
            block_table: Vec::with_capacity(64),
            seq_len: 0,
            last_num_drafted: 0,
            prefill_done: false,
            ctx_hidden_acc,
            ctx_len: 0,
            last_num_accepted: 0,
            skip_next_decode_append: false,
            max_ctx_len: self.max_seq_len,
            ctx_slot_bytes,
            // Phase 2 Option B: lazily allocated on first propose when
            // ATLAS_DFLASH_OPTION_B=1. None until then to keep alloc_state
            // cheap for sequences that never use Option B.
            block_table_dev: None,
            ctx_count_drafter: 0,
            max_ctx_count_drafter: 0,
            ctx_committed: 0,
            ctx_positions: Vec::new(),
        }))
    }

    fn propose(
        &self,
        last_token: u32,
        target_hidden: spark_runtime::gpu::DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
        draft_embed_target: Option<spark_runtime::gpu::DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        target_hidden_stack: Option<spark_runtime::gpu::DevicePtr>,
    ) -> Result<Vec<u32>> {
        self.propose_drafts(
            last_token,
            target_hidden,
            position,
            num_drafts,
            state,
            ctx,
            stream,
            draft_embed_target,
            grammar_bitmask,
            target_hidden_stack,
        )
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let dstate = state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid DFlash proposer state"))?;
        // Phase 1: no real KV trim because `propose()` is a stub. Phase 2
        // adds the rollback that drops `(last_num_drafted - num_accepted)`
        // tokens from each layer's paged cache.
        //
        // Phase I invariant: `ctx_committed` is the watermark of ctx slots
        // already precomputed into the paged cache. It is monotonic only as
        // long as `ctx_len` is monotonic (today it is — ctx is append-only
        // and never rewound here). IF a future rollback ever shrinks the
        // committed ctx (rewinds `ctx_len`), it MUST also reset
        // `dstate.ctx_committed = dstate.ctx_len` so the next propose
        // recomputes the rolled-back tail instead of reading stale K/V.
        // The `.min(ctx_len)` clamp in propose() is the defensive backstop.
        let _ = num_accepted;
        dstate.last_num_drafted = 0;
        Ok(())
    }

    fn free_state(&self, gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        // Phase 2 (Option B) reclaim: return the drafter's lazily-allocated
        // paged KV blocks to the pool on request completion. Without this the
        // ~257-block Option-B drafter cache (allocated in propose.rs when
        // block_table_dev.is_none()) is never freed, so the SECOND request to
        // a long-lived server starts with zero free drafter blocks and floods
        // "DFlash Option B: paged KV cache exhausted". Mirrors MtpHead::free_state.
        let dstate = match state.as_any_mut().downcast_mut::<DflashProposerState>() {
            Some(s) => s,
            // Phase 1 / non-DFlash proposer state: nothing allocated, nothing to free.
            None => return Ok(()),
        };
        if !dstate.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&dstate.block_table);
            dstate.block_table.clear();
        }
        // Free the per-seq ctx accumulator — the dominant per-request
        // allocation (`max_seq_len × 5 × target_hidden` BF16; ~320 MB at
        // max_seq_len=16384). `DevicePtr` has no Drop, so without this every
        // finished sequence leaks it for the server's lifetime. Guarded on a
        // non-null pointer so a double free_state is a no-op.
        if dstate.ctx_hidden_acc.0 != 0 {
            gpu.free(dstate.ctx_hidden_acc)?;
            dstate.ctx_hidden_acc = DevicePtr(0);
        }
        // Free the device-side block table (lazily allocated in propose.rs).
        if let Some(bt) = dstate.block_table_dev.take() {
            gpu.free(bt)?;
        }
        // Reset the lazy-alloc guard + watermarks so the NEXT request's first
        // propose re-allocates fresh blocks and re-precomputes ctx from a clean
        // slate (propose.rs gates alloc on block_table_dev.is_none()).
        dstate.max_ctx_count_drafter = 0;
        dstate.ctx_count_drafter = 0;
        dstate.ctx_committed = 0;
        dstate.ctx_positions.clear();
        dstate.seq_len = 0;
        dstate.ctx_len = 0;
        dstate.prefill_done = false;
        dstate.last_num_drafted = 0;
        dstate.last_num_accepted = 0;
        dstate.skip_next_decode_append = false;
        Ok(())
    }
}
