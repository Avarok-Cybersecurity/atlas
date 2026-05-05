// SPDX-License-Identifier: AGPL-3.0-only

//! `TransformerLayer` trait — composable per-layer forward/decode hooks.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::{ForwardContext, GdnPrefillBuffers, LayerState};

pub trait TransformerLayer: Send + Sync {
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
        let h = ctx.config.hidden_size;
        for t in 0..num_tokens {
            let offset = t * h * 2; // BF16 = 2 bytes per element
            let h_t = hidden.offset(offset);
            let r_t = residual.offset(offset);
            self.decode(
                h_t,
                r_t,
                state,
                kv_cache,
                seq_len_start + t,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                ctx,
                stream,
            )?;
        }
        Ok(())
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
        let h = ctx.config.hidden_size;
        for t in 0..num_tokens {
            let offset = (t * h * 2) as u64; // BF16 = 2 bytes per element
            let h_t = hidden.offset(offset as usize);
            let r_t = residual.offset(offset as usize);
            self.decode(
                h_t,
                r_t,
                state,
                kv_cache,
                seq_len + t,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                ctx,
                stream,
            )?;
        }
        Ok(())
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
        let h = ctx.config.hidden_size;
        for i in 0..num_seqs {
            let offset = i * h * 2;
            let h_i = hidden.offset(offset);
            let r_i = residual.offset(offset);
            let mut bt = block_tables[i].clone();
            // Phase 6.1: per-seq disk_block_ids aren't threaded through this
            // default impl yet (chunked-prefill / batched-decode are Phase 6.2
            // scope). Pass empty stubs so the trait sig is satisfied; layers
            // that need disk IDs (attention) override decode_multi_seq.
            let mut stub_disk = Vec::<u32>::new();
            let mut stub_last_offloaded = Vec::<u32>::new();
            self.decode(
                h_i,
                r_i,
                states[i],
                kv_cache,
                seq_lens[i],
                &mut bt,
                &mut stub_disk,
                &mut stub_last_offloaded,
                ctx,
                stream,
            )?;
        }
        Ok(())
    }

    /// Allocate per-sequence state for this layer.
    ///
    /// Called once when a new sequence is created. Returns:
    /// - `EmptyLayerState` for pure attention layers
    /// - `SsmLayerState` for SSM/recurrent layers
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>>;
}
