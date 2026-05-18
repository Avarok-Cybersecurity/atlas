// SPDX-License-Identifier: AGPL-3.0-only

//! Composable transformer layer traits (SDD).
//!
//! Decouples the generic model loop (embed -> layers -> norm -> lm_head)
//! from layer-specific logic (attention vs SSM, MoE vs dense FFN).
//! Adding a new architecture only requires implementing [`TransformerLayer`]
//! for each layer type, not duplicating the model loop.

use std::any::Any;

use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

mod transformer_layer;
pub use transformer_layer::TransformerLayer;

/// Per-layer persistent state tracked across decode steps.
///
/// Attention layers use [`EmptyLayerState`] (KV lives in `PagedKvCache`).
/// SSM layers use [`SsmLayerState`] (recurrent h_state + conv_state).
/// Custom layers can implement this trait for arbitrary state.
pub trait LayerState: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Empty state for layers that store all persistent state externally
/// (e.g., attention layers where KV is in `PagedKvCache`).
pub struct EmptyLayerState;

impl LayerState for EmptyLayerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// SSM layer state: recurrent hidden state + conv1d sliding window.
///
/// Used by Mamba, Gated Delta Net (GDN), and similar recurrent layers.
pub struct SsmLayerState {
    /// Recurrent hidden state: [num_v_heads, v_dim, k_dim] in f32.
    pub h_state: DevicePtr,
    /// Conv1d sliding window state: [d_inner, d_conv] in f32.
    pub conv_state: DevicePtr,
    /// Checkpoint buffer for h_state (allocated lazily for speculative decode).
    pub h_state_checkpoint: Option<DevicePtr>,
    /// Checkpoint buffer for conv_state (allocated lazily for speculative decode).
    pub conv_state_checkpoint: Option<DevicePtr>,
    /// Intermediate h_state snapshots during batched verification.
    /// Element i holds h_state after processing verification token i.
    /// Used by rollback_ssm_states to restore to the correct position.
    pub h_state_intermediates: Vec<DevicePtr>,
    /// Intermediate conv_state snapshots during batched verification.
    pub conv_state_intermediates: Vec<DevicePtr>,
}

impl LayerState for SsmLayerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Pre-uploaded attention metadata device pointers.
///
/// Uploaded once per decode step in the model loop, reused across all
/// 12 attention layers. Eliminates 44 redundant H2D copies per step.
///
/// For batched decode (num_seqs > 1), arrays are contiguous:
/// - positions: `[N]` u32
/// - slots: `[N]` i64
/// - seq_lens: `[N]` i32
/// - block_table: `[N * max_blocks_per_seq]` i32 (row-major)
#[derive(Clone, Copy)]
pub struct AttnMetadataDev {
    /// Position values: `[N]` u32 at this device address. For multi-modal
    /// MRoPE this is the temporal (T) stream; callers set
    /// `positions_h`/`positions_w` to distinct buffers only when the token
    /// stream contains image or video patches.
    pub positions: DevicePtr,
    /// Height (H) position stream for MRoPE-interleaved. When identical
    /// to `positions` (same pointer) the rope reduces to scalar RoPE.
    /// Default: same as `positions`.
    pub positions_h: DevicePtr,
    /// Width (W) position stream for MRoPE-interleaved. Same fallback as
    /// `positions_h`.
    pub positions_w: DevicePtr,
    /// Slot mappings: `[N]` i64 at this device address.
    pub slot: DevicePtr,
    /// Sequence lengths (+1): `[N]` i32 at this device address.
    pub seq_len: DevicePtr,
    /// Block tables: `[N * max_blocks_per_seq]` i32 at this device address.
    pub block_table: DevicePtr,
    /// Number of blocks per sequence row in block_table.
    pub max_blocks_per_seq: u32,
    /// Number of sequences in this batch (1 for single-sequence decode).
    pub num_seqs: u32,
}

/// Q12 batched-prefill device-side metadata.
///
/// The single-stream `AttnMetadataDev` collapses per-stream pointers into
/// concrete device pointers because there's only one stream. For Q12 we
/// dispatch N concurrent prefilling streams through one batched kernel,
/// and the kernel takes:
///   - stacked positions / slot tables (one big buffer with all streams'
///     data concatenated in cu_seqlens order), and
///   - per-stream pointer arrays for block_table / seq_len / h_state.
///
/// Built once per `prefill_batch_chunk_dispatch` call by
/// `stage_batched_attn_metadata`; threaded through the model-level
/// per-layer batched dispatch (`prefill_attn_batched_layer`,
/// `prefill_ssm_batched_layer`) — see `model/trait_impl/prefill_b/batch.rs`.
pub struct BatchedAttnMetadata {
    /// Stacked positions across all streams: `[total_tokens]` u32 at this
    /// address. For MRoPE interleaved this is the temporal (T) stream.
    pub positions_stacked: DevicePtr,
    /// MRoPE H position stream, stacked. Equal to `positions_stacked` when
    /// MRoPE is disabled.
    pub positions_h_stacked: DevicePtr,
    /// MRoPE W position stream, stacked. Equal to `positions_stacked` when
    /// MRoPE is disabled.
    pub positions_w_stacked: DevicePtr,
    /// Stacked slot indices for KV writes: `[total_tokens]` i64.
    pub slot_stacked: DevicePtr,
    /// Per-stream block_table pointer array: `[batch_size]` of `DevicePtr`,
    /// each element pointing to a stream's chunked-prefill block_table.
    /// Used by `prefill_attention_paged_*_batched` kernels.
    pub block_table_ptrs: DevicePtr,
    /// Per-stream seq_len pointer array: `[batch_size]` of `DevicePtr`.
    pub seq_len_ptrs: DevicePtr,
    // Note: `h_state_ptrs` is NOT cached in BatchedAttnMetadata because
    // it's per-layer (each SSM layer's SsmLayerState has its own h_state
    // allocation). `prefill_ssm_batched_layer` stages h_state_ptrs JIT
    // per-layer-call into the model's scratch buffer.
    /// Number of batched streams.
    pub batch_size: u32,
    /// Per-stream chunk_len (SAME for all streams — scheduler-enforced
    /// constraint via `can_batch_prefill_only`).
    pub chunk_len: u32,
    /// Total tokens stacked across streams: `batch_size * chunk_len`.
    pub total_tokens: u32,
    /// Maximum block_table length across the batch (kernel uses for
    /// bounds checking; per-stream block_table reads via the pointer
    /// array dereference).
    pub max_blocks_per_seq: u32,
}

/// Device pointers to full-sequence GDN input/output buffers.
///
/// Used by the two-phase SSM prefill: phase 1 writes GDN inputs here,
/// phase 2 reads them for the single-launch GDN kernel, phase 3 reads output.
///
/// Uses a **packed QKV layout** matching the conv1d output: each token occupies
/// `conv_dim` contiguous BF16 elements as `[Q(key_dim) | K(key_dim) | V(value_dim)]`.
/// This allows simple contiguous memcpy from per-chunk conv1d output buffers.
/// The GDN kernel reads Q/K/V via stride parameters (`qk_stride = conv_dim`,
/// `v_stride = conv_dim`) to index into the packed layout.
pub struct GdnPrefillBuffers {
    /// Packed Q/K/V: [total_len, conv_dim] BF16.
    /// Layout per token: [Q(key_dim) | K(key_dim) | V(value_dim)].
    pub qkv: DevicePtr,
    /// Interleaved gate/beta: [total_len, 2*num_v_heads] FP32.
    /// Layout per token: [gate(nv) | beta(nv)].
    pub gate_beta: DevicePtr,
    /// GDN recurrence output: [total_len, value_dim] BF16.
    pub output: DevicePtr,
    /// Z gate for gated RMS norm: [total_len, value_dim] BF16.
    pub z: DevicePtr,
    /// Total number of tokens across all chunks.
    pub total_len: usize,
}

/// Shared context for a single forward pass step.
///
/// Provides access to GPU, buffers, and config without coupling
/// layer implementations to the model struct.
pub struct ForwardContext<'a> {
    /// Pre-allocated scratch buffers.
    pub buffers: &'a BufferArena,
    /// GPU backend for kernel launches and memory ops.
    pub gpu: &'a dyn GpuBackend,
    /// Model configuration (dimensions, hyperparameters).
    pub config: &'a ModelConfig,
    /// Pre-uploaded attention metadata (None if no attention layers).
    pub attn_metadata: Option<AttnMetadataDev>,
    /// Profile mode: sync+time per-operation within layers.
    pub profile: bool,
    /// Communication backend for expert parallelism (EP) all-reduce.
    /// None when running single-GPU (no distributed communication).
    pub comm: Option<&'a dyn spark_comm::CommBackend>,
    /// True when inside CUDA graph capture (between begin_capture/end_capture).
    /// MoE layers use sync all_reduce (capturable) instead of async (event-based).
    pub graph_capture: bool,
}

/// A single transformer layer performing the full per-layer computation.
///
/// Each layer encapsulates:
/// 1. Pre-norm -> attention/SSM -> residual add
/// 2. Post-norm -> FFN/MoE -> residual add
///
/// The generic model loop iterates `layers` without knowing whether
/// each is attention, SSM, MoE, or dense FFN.
#[cfg(test)]
mod tests;
