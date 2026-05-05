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
use crate::weight_map::DenseWeight;

/// Kernel handles for the DFlash γ-block forward chain. All resolved once
/// at `BlockDiffusionDraftHead::from_weights` against the active GPU backend
/// (which compiles target-specific PTX at startup); subsequent
/// `propose()` calls just `KernelLaunch::new(...).launch(stream)`.
pub struct DflashKernels {
    pub rms_norm: KernelHandle,
    pub residual_rms_norm: KernelHandle,
    pub dense_gemv: KernelHandle,
    pub dense_gemm: KernelHandle,
    pub rope_qwen3: KernelHandle,
    pub reshape_cache_fp8: KernelHandle,
    pub prefill_attn_dflash_fp8: KernelHandle,
    pub silu_mul: KernelHandle,
    pub residual_add: KernelHandle,
    pub argmax: KernelHandle,
    pub batched_embed: KernelHandle,
    /// Non-paged prefill attention (used for the γ-block self-attention
    /// when there's no persistent K/V cache to walk).
    pub prefill_attn: KernelHandle,
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
    pub logits: DevicePtr,
    pub draft_tokens_dev: DevicePtr,
    /// `[ctx_window + γ]` i32 positions. First ctx_window are
    /// historical target positions (decoded indices); last γ are
    /// the to-be-predicted noise positions.
    pub position_ids: DevicePtr,
}

/// Drafter-side weight precision. Defaults to BF16 because community
/// reports an FP8 acceptance-rate collapse on SM12.x; `--mtp-quantization fp8`
/// is intentionally not honored for the DFlash drafter (warned at build time).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DflashQuantization {
    Bf16,
}

/// Per-drafter-layer Qwen3-style weights. Phase 1 is BF16-only; FP8/NVFP4
/// drafter quantization is deferred (see `DflashQuantization`).
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
    /// Allocation cap for `ctx_hidden_acc` (in slot count). Mirrors the
    /// `max_seq_len` build arg so we can clamp without re-fetching it.
    pub max_ctx_len: usize,
    /// Width (bytes) of one `ctx_hidden_acc` slot — `5 * target_hidden * bf16`.
    /// Stored to avoid re-deriving on every append.
    pub ctx_slot_bytes: usize,
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
    /// argmax over `[γ, vocab]` logits.
    pub lm_head_shared: DevicePtr,

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

    // Quantization mode (BF16 only for Phase 1).
    pub quant: DflashQuantization,
}

mod forward_block;
mod forward_block_layer;
mod from_weights;
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
            max_ctx_len: self.max_seq_len,
            ctx_slot_bytes,
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
        let _ = num_accepted;
        dstate.last_num_drafted = 0;
        Ok(())
    }

    fn free_state(&self, _state: &mut dyn ProposerState) -> Result<()> {
        // Phase 1: nothing to free (no allocated KV blocks yet). Phase 2
        // reclaims paged blocks across all drafter layers.
        Ok(())
    }
}
