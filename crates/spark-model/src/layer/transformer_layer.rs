// SPDX-License-Identifier: AGPL-3.0-only

//! `TransformerLayer` trait — composable per-layer forward/decode hooks.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::{BatchedAttnMetadata, ForwardContext, GdnPrefillBuffers, LayerState};

mod default_loops;

/// Batched-verify WY pointer-table layout (SSOT — writer:
/// `TransformerModel::upload_verify_wy_tables`, reader: the qwen3_ssm
/// `GdnStates::Multi` batched conv+WY arm).
///
/// Per GDN layer, four device pointer tables laid back-to-back:
/// `[h_state | Hi0 | Hi1 | Hi2]`, each `VERIFY_WY_TABLE_SEQS` u64 entries
/// (one per batched-verify sequence, unused tail entries zero). The
/// per-layer slice handed to `decode_verify_multi` is
/// `VERIFY_WY_LAYER_STRIDE_BYTES` long. At K<4 only the first `k` tables
/// are filled (`[h | Hi_0..Hi_{k-2}]`); the layout/strides are constant so
/// the reader's offsets never depend on the ladder step.
///
/// 32 (2026-07-30, spec at n=32): THE batched-verify sequence envelope —
/// the stash slots, the `can_batch_verify` n-cap and these tables are all
/// sized from this one const. History: 4 → 16 (at 4 the upload returned
/// NULL for every n>4 batch, silently declining the cross-sequence conv+WY
/// fast path at exactly the concurrencies it was built for) → 32 (the
/// 32:1 ladder rung: n=32 × k=2 = 64 verify rows; kernels index a table by
/// sequence and receive each table's base pointer separately, so widening
/// is a pure host-side layout change). 48 GDN layers x 4 tables x 32
/// entries x 8 B = 48 KB.
pub const VERIFY_WY_TABLE_SEQS: usize = 32;
/// Tables per GDN layer: h_state + Hi0..Hi2 (K=4 verify → 3 intermediates).
pub const VERIFY_WY_TABLES_PER_LAYER: usize = 4;
/// Bytes between consecutive tables within a layer slice.
pub const VERIFY_WY_TABLE_STRIDE_BYTES: usize = VERIFY_WY_TABLE_SEQS * 8;
/// Bytes between consecutive GDN layers' table slices.
pub const VERIFY_WY_LAYER_STRIDE_BYTES: usize =
    VERIFY_WY_TABLES_PER_LAYER * VERIFY_WY_TABLE_STRIDE_BYTES;

pub trait TransformerLayer: Send + Sync {
    /// True when this layer's PREFILL attends only over the tokens it is
    /// handed, so a prefix-cache skip would hide the cached prefix from
    /// attention entirely. MLA layers on the paged path do; everything else
    /// reads the paged cache and is unaffected.
    fn uses_local_mla_prefill(&self) -> bool {
        false
    }

    /// `&mut dyn Any` downcast hook for post-construction weight overlays (e.g.
    /// the LoRA install walk). Default `None`; overlay-capable layers override.
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }

    /// Whether this layer's ONLINE FP8-KV calibration has frozen its scale.
    /// `None` = this layer runs no online calibration (non-attention layer,
    /// static checkpoint scales, or a non-FP8 KV dtype). The scheduler's
    /// graph-suppression gate keys off this rather than a token count: the
    /// scale freezes on the FIRST observe, so waiting `calibration_tokens`
    /// tokens would run ~256+ eager steps for a calibration that finished
    /// immediately.
    fn fp8_calibration_frozen(&self) -> Option<bool> {
        None
    }

    /// Decode one token through this layer, modifying `hidden` in-place.
    ///
    /// # Arguments
    /// * `hidden` - [1, hidden_size] BF16, read and written
    /// * `residual` - [1, hidden_size] BF16, scratch for residual stream
    /// * `state` - Per-layer state (empty for attention, SSM state for recurrent)
    /// * `kv_cache` - Paged KV cache (may be mutated for block allocation)
    /// * `seq_len` - Current sequence length (for position encoding + cache)
    /// * `block_table` - Sequence's block table (may grow if new blocks needed)
    /// * `ctx` - Shared forward context (buffers, gpu, config)
    /// * `stream` - CUDA stream handle
    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        // `--high-speed-swap` disk-side IDs parallel to `block_table` (Phase
        // 6.1.c). Layer-agnostic: the same ID indexes a slot in every
        // layer's on-disk file. Empty when the feature is disabled.
        disk_block_ids: &mut Vec<u32>,
        // Per-layer offload progress (Phase 6.1.d critical fix). Layer L
        // reads/writes `disk_last_offloaded_per_layer[L]`. Each layer's
        // offload runs independently because each layer writes its own
        // K/V to a separate region of the on-disk file. Empty when HSS
        // is disabled; SSM/MoE layers ignore it.
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()>;

    /// Prefill N tokens through this layer using GEMM-batched projections.
    ///
    /// Used during prompt processing: reads weight matrices once for all N
    /// tokens (GEMM M=N) instead of N separate GEMV calls. Attention uses
    /// Flash Attention on contiguous Q/K/V. SSM/GDN recurrence remains
    /// sequential per-token.
    ///
    /// # Arguments
    /// * `hidden` - [N, hidden_size] BF16, read and written
    /// * `residual` - [N, hidden_size] BF16, scratch for residual stream
    /// * `num_tokens` - Number of tokens (N)
    /// * `state` - Per-layer state (SSM state updated sequentially)
    /// * `kv_cache` - Paged KV cache (attention layers write K/V for all N)
    /// * `seq_len_start` - Sequence position of first token (usually 0)
    /// * `block_table` - Block table for KV cache (pre-allocated for N tokens)
    /// * `ctx` - Shared forward context (buffers, gpu, config)
    /// * `stream` - CUDA stream handle
    ///
    /// Default: falls back to sequential single-token decode calls.
    ///
    /// `kv_write_start`: number of tokens whose KV cache entries are already
    /// populated (prefix caching). Attention layers skip KV writes for
    /// positions `< kv_write_start`. SSM layers ignore this (recurrent).
    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        default_loops::prefill_default(
            self,
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len_start,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )
    }

    /// Two-phase SSM prefill — Phase 1: projections and GDN input staging.
    ///
    /// Runs RMS norm, QKVZ projection, BA+gates, conv1d, and L2 norm for a
    /// chunk of `num_tokens` tokens, then copies the GDN inputs (packed QKV,
    /// gate/beta, Z) into the full-sequence `gdn_bufs` at `token_offset`.
    ///
    /// Does NOT run the GDN recurrence — that happens in `prefill_gdn_full`
    /// after all chunks have staged their inputs.
    ///
    /// Attention layers: default falls back to full `prefill` (no phasing).
    #[allow(clippy::too_many_arguments)]
    fn prefill_phase1(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        kv_write_start: usize,
        gdn_bufs: &GdnPrefillBuffers,
        token_offset: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Default: fall back to full prefill (attention layers, non-SSM layers)
        let _ = (gdn_bufs, token_offset);
        self.prefill(
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len_start,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            kv_write_start,
            ctx,
            stream,
        )
    }

    /// M1 large-M batched Phase-1: token-parallel projections (RMS/QKVZ/BA-gates)
    /// over ALL stacked tokens in one large-M GEMM each. SSM-only; the caller
    /// runs `prefill_phase1_conv1d_one` per request then `prefill_phase1_l2_batched`.
    fn prefill_phase1_proj_batched(
        &self,
        hidden_stacked: DevicePtr,
        residual_stacked: DevicePtr,
        total_tokens: usize,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let _ = (
            hidden_stacked,
            residual_stacked,
            total_tokens,
            gdn_bufs,
            ctx,
            stream,
        );
        anyhow::bail!("prefill_phase1_proj_batched: only implemented for SSM layers")
    }

    /// M1: per-request conv1d tail (advances per-request conv_state), reading the
    /// request's slice of the stacked QKVZ scratch and writing into gdn_bufs.qkv.
    fn prefill_phase1_conv1d_one(
        &self,
        state: &mut dyn LayerState,
        token_offset: usize,
        len: usize,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let _ = (state, token_offset, len, gdn_bufs, ctx, stream);
        anyhow::bail!("prefill_phase1_conv1d_one: only implemented for SSM layers")
    }

    /// M1: batched L2 norm over the full stacked QKV buffer after all per-request
    /// conv1d tails have written their slices.
    fn prefill_phase1_l2_batched(
        &self,
        total_tokens: usize,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let _ = (total_tokens, gdn_bufs, ctx, stream);
        anyhow::bail!("prefill_phase1_l2_batched: only implemented for SSM layers")
    }

    /// Two-phase SSM prefill — Phase 2: GDN recurrence on the full sequence.
    ///
    /// Runs the WY4-persistent GDN kernel over all `total_len` tokens in
    /// `gdn_bufs` in a single launch. The kernel reads packed QKV and
    /// gate/beta from the full-sequence buffers and writes the GDN output.
    ///
    /// Only meaningful for SSM layers. Attention layers return `Ok(())`.
    fn prefill_gdn_full(
        &self,
        _state: &mut dyn LayerState,
        _gdn_bufs: &GdnPrefillBuffers,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        Ok(()) // No-op for attention layers
    }

    /// Q12 Path B: batched attention prefill across N stacked-input streams.
    ///
    /// Runs the full attention-layer prefill (rms_norm + residual, QKV proj,
    /// RoPE, KV-write, batched attention compute, O proj, post-attn norm,
    /// FFN, final residual) over `num_tokens = batch_size * chunk_len`
    /// stacked tokens, using `batched_meta` for per-stream metadata
    /// resolution.
    ///
    /// Default impl returns Err — only `Qwen3AttentionLayer` overrides.
    /// SSM/dense layers don't override (they have their own batched paths
    /// or work without batched metadata).
    ///
    /// Caller (model-level `prefill_attn_batched_layer`) is responsible for
    /// ensuring all streams share the same chunk_len, seq_len_start
    /// (q_offset), and that the layer is not MLA / not HDIM=512 / not HSS-
    /// engaged. The override bails Err if any unsupported case is detected.
    fn prefill_inner_batched_q12(
        &self,
        _hidden_stacked: DevicePtr,
        _residual_stacked: DevicePtr,
        _num_tokens: usize,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _batched_meta: &BatchedAttnMetadata,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!("prefill_inner_batched_q12: not implemented for this layer type")
    }

    /// Q12 Path B: batched GDN recurrence across N streams.
    ///
    /// Runs the same WY32 / persistent / split4 GDN kernel as
    /// `prefill_gdn_full` but with `batch_size = batch_size` and
    /// `h_state_ptrs` pointing to a device array of N per-stream h_state
    /// pointers (staged by `TransformerModel::stage_h_state_ptrs`).
    /// `gdn_bufs.qkv` / `gate_beta` / `output` are stacked across N
    /// streams contiguously: each stream's data lives at
    /// `b * chunk_len * conv_dim` (BF16) within the buffer.
    ///
    /// Default impl returns `Err` — the SSM layer override implements the
    /// actual batched dispatch using the kernel handles loaded in
    /// commit `8d07ca4`. Attention layers don't override (they don't
    /// have a GDN step).
    fn prefill_gdn_full_batched(
        &self,
        _h_state_ptrs: DevicePtr,
        _gdn_bufs: &GdnPrefillBuffers,
        _batch_size: u32,
        _chunk_len: u32,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!(
            "prefill_gdn_full_batched: layer does not implement batched GDN \
             — caller should fall back to per-stream prefill_gdn_full"
        )
    }

    /// VARLEN batched GDN: process ragged co-dispatch lengths via `cu_seqlens` in
    /// ONE `gdn_prefill_fla(batch=N, is_varlen)` call (replaces the non-uniform
    /// per-request loop → fills chunk_delta_h's 32→32N CTAs). Returns `Ok(true)`
    /// if it ran, `Ok(false)` if not eligible (caller falls back to the loop).
    /// Default (non-SSM layers, or FLA disabled): `Ok(false)`.
    #[allow(clippy::too_many_arguments)]
    fn prefill_gdn_full_batched_fla_varlen(
        &self,
        _h_state_ptrs: DevicePtr,
        _gdn_bufs: &GdnPrefillBuffers,
        _batch_size: u32,
        _cu_seqlens: DevicePtr,
        _max_num_chunks: u32,
        _total_nt: usize,
        _max_seqlen: u32,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<bool> {
        Ok(false)
    }

    /// Two-phase SSM prefill — Phase 3: post-GDN processing.
    ///
    /// Reads GDN output and Z gate from `gdn_bufs` at `token_offset`,
    /// then runs gated RMS norm, output projection, residual add, and MoE
    /// for the chunk of `num_tokens` tokens.
    ///
    /// Only meaningful for SSM layers. Attention layers return `Ok(())`.
    #[allow(clippy::too_many_arguments)]
    fn prefill_phase3(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _num_tokens: usize,
        _gdn_bufs: &GdnPrefillBuffers,
        _token_offset: usize,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        Ok(()) // No-op for attention layers
    }

    /// Returns true if this layer is an SSM layer (supports two-phase prefill).
    ///
    /// When true, the model loop can use `prefill_phase1` / `prefill_gdn_full` /
    /// `prefill_phase3` instead of the monolithic `prefill`.
    fn is_ssm_layer(&self) -> bool {
        false
    }

    /// Allocate the transposed MoE expert weights used by the coalesced
    /// prefill GEMM kernels. Called as a post-load pass from `factory::build`
    /// after LM-head NVFP4 quantization has freed BF16 headroom, so
    /// memory-tight EP configurations (e.g. MiniMax M2.7-NVFP4 EP=2) can
    /// fit the transpose that layer-0 preflight would otherwise reject.
    ///
    /// Default: no-op (non-MoE layers, and MoE layers whose loader already
    /// called `MoeLayer::transpose_for_prefill` inline during construction).
    fn transpose_moe_for_prefill(
        &mut self,
        _gpu: &dyn GpuBackend,
        _config: &ModelConfig,
    ) -> Result<()> {
        Ok(())
    }

    /// Like `transpose_moe_for_prefill` but only transposes the gate+up
    /// projections (skips the down projection), reducing the transpose cost
    /// from 3× to 2× per expert. Used as a memory-tight fallback by the
    /// MiniMax loader when full transpose doesn't fit.
    fn transpose_moe_gate_up_for_prefill(
        &mut self,
        _gpu: &dyn GpuBackend,
        _config: &ModelConfig,
    ) -> Result<()> {
        Ok(())
    }

    /// Wire a shared per-prefill `down_proj` transpose scratch into this
    /// layer's MoE block. Used as a memory-tight alternative to the
    /// persistent down transpose: factory allocates one shared scratch,
    /// every MoE layer reuses it layer-by-layer during sequential
    /// prefill. No-op for non-MoE layers and MoE layers that already
    /// have a persistent transposed down.
    fn set_moe_down_transpose_scratch(
        &mut self,
        _scratch_packed: DevicePtr,
        _scratch_scale: DevicePtr,
        _packed_ptrs_t: DevicePtr,
        _scale_ptrs_t: DevicePtr,
    ) {
    }

    /// Phase 8a unified-layout MoE transpose: build persistent transposed
    /// gate/up/down for all experts and free the untransposed copies.
    /// Phased flow keeps memory budget tight enough for MiniMax M2.7 EP=2.
    /// After this call, the untransposed-layout decode kernels can no
    /// longer execute correctly — `MoeLayer::use_t_layout_for_decode()` must
    /// gate dispatch to the `_t` decode kernels. Default no-op.
    fn transpose_moe_for_prefill_unified(
        &mut self,
        _gpu: &dyn GpuBackend,
        _config: &ModelConfig,
    ) -> Result<()> {
        Ok(())
    }

    /// Block C Path 2 hybrid-layout MoE transpose: build persistent
    /// transposed gate/up/down alongside the untransposed originals (no
    /// frees). Doubles MoE-weight memory but recovers the ~15 % decode
    /// regression of pure unified mode — decode + MTP verify dispatch
    /// keeps using the warp-reduction kernels on the originals while
    /// prefill (forward_batched) routes through transposed kernels.
    /// Caller must verify enough free memory before invocation. Default
    /// no-op for non-MoE layers.
    fn transpose_moe_for_prefill_hybrid(
        &mut self,
        _gpu: &dyn GpuBackend,
        _config: &ModelConfig,
    ) -> Result<()> {
        Ok(())
    }

    /// Decode K tokens through this layer using GEMM-batched projections.
    ///
    /// Used for speculative decode verification: processes multiple tokens
    /// per layer with GEMM for weight-heavy projections (amortizes bandwidth)
    /// and sequential ops for stateful/recurrent components.
    ///
    /// # Arguments
    /// * `hidden` - [K, hidden_size] BF16, read and written (K tokens contiguous)
    /// * `residual` - [K, hidden_size] BF16, scratch for residual stream
    /// * `num_tokens` - Number of tokens (K)
    /// * `state` - Per-layer state
    /// * `kv_cache` - Paged KV cache
    /// * `seq_len` - Starting sequence length (before these tokens)
    /// * `block_table` - Block table for KV cache
    /// * `ctx` - Shared context
    /// * `stream` - CUDA stream
    ///
    /// Default: falls back to sequential single-token decode calls.
    #[allow(clippy::too_many_arguments)]
    fn decode_batched(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        default_loops::decode_batched_default(
            self,
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )
    }

    /// Decode N sequences through this layer in a single batched call.
    ///
    /// Each sequence contributes 1 token. The weight matrices are loaded
    /// once and applied to all N sequences (amortizing memory bandwidth).
    ///
    /// # Arguments
    /// * `hidden` - [N, hidden_size] BF16, contiguous
    /// * `residual` - [N, hidden_size] BF16, contiguous
    /// * `num_seqs` - Number of sequences (N)
    /// * `states` - N per-layer states (one per sequence)
    /// * `kv_cache` - Shared paged KV cache
    /// * `ctx` - Forward context (attn_metadata contains N-sequence metadata)
    /// * `stream` - CUDA stream
    ///
    /// Default: falls back to N sequential single-token decode calls.
    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        default_loops::decode_multi_seq_default(
            self,
            hidden,
            residual,
            num_seqs,
            states,
            kv_cache,
            seq_lens,
            block_tables,
            ctx,
            stream,
        )
    }

    /// Batched MTP verify: `n_seqs` sequences × `k` tokens through this layer
    /// in ONE weight sweep (rows seq-major, `r = i*k + j`, contiguous in
    /// `hidden`/`residual`). Projections/FFN batch across all `n_seqs*k` rows;
    /// the stateful recurrence (conv/GDN) runs per-sequence against
    /// `states[i]` with row-offset buffer bases — per-sequence math is
    /// byte-identical to the single-sequence `decode_batched` K-token body.
    ///
    /// Only SSM layers override (attention layers are handled by the caller
    /// via `decode_multi_seq`, which already takes per-row block tables and
    /// seq lens). Default: unsupported.
    ///
    /// `wy_tables`: this layer's slice of the model-staged WY pointer tables
    /// (layout above, `VERIFY_WY_LAYER_STRIDE_BYTES`; refreshed pre-graph
    /// every step) enabling the single-launch table-form WY batch. NULL →
    /// the layer keeps its per-sequence WY path.
    #[allow(clippy::too_many_arguments)]
    fn decode_verify_multi<'a, 'b: 'a>(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _n_seqs: usize,
        _ks: &[usize],
        _states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _kv_cache: &mut PagedKvCache,
        _wy_tables: DevicePtr,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!("decode_verify_multi: unsupported for this layer type")
    }

    /// Allocate per-sequence state for this layer.
    ///
    /// Called once when a new sequence is created. Returns:
    /// - `EmptyLayerState` for pure attention layers
    /// - `SsmLayerState` for SSM/recurrent layers
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>>;
}
