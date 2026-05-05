// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// Architecture-agnostic transformer model.
///
/// Composes `Vec<Box<dyn TransformerLayer>>` into a full forward pass.
/// Adding a new model only requires implementing [`TransformerLayer`]
/// for each layer type — the model loop stays unchanged.
#[allow(dead_code)]
pub struct TransformerModel {
    pub(super) config: ModelConfig,
    pub(super) embed_tokens: DenseWeight,
    pub(super) final_norm: DenseWeight,
    pub(super) lm_head_weight: DenseWeight,
    pub(super) lm_head_nvfp4: Option<QuantizedWeight>,
    pub(super) layers: Vec<Box<dyn TransformerLayer>>,
    pub(super) buffers: BufferArena,
    pub(super) kv_cache: Mutex<PagedKvCache>,
    pub(super) gpu: Box<dyn GpuBackend>,
    pub(super) rms_norm_kernel: KernelHandle,
    pub(super) bf16_to_f32_kernel: KernelHandle,
    pub(super) dense_gemv_kernel: KernelHandle,
    /// FP32-output variant of dense_gemv_bf16. Used by the LM head when
    /// `use_fp32_logits` is true, so the FP32 accumulator is preserved across
    /// the BF16-storage rounding boundary that flips greedy argmax tiebreaks
    /// on Gemma-4-31B (top-1 vs top-2 = 0.125 logit gap = exact BF16 step at
    /// value 16-32 → BF16 store snaps the wrong way and starts a stop-word
    /// loop). Loaded once at model init.
    pub(super) dense_gemv_fp32out_kernel: KernelHandle,
    pub(super) w4a16_gemv_kernel: KernelHandle,
    pub(super) w4a16_gemv_logits_kernel: KernelHandle, // FP32 output for LM head
    pub(super) w4a16_gemm_kernel: KernelHandle,
    pub(super) w4a16_gemv_batch2_kernel: KernelHandle,
    pub(super) dense_gemm_kernel: KernelHandle,
    pub(super) argmax_kernel: KernelHandle,
    pub(super) argmax_logits_kernel: KernelHandle, // FP32 argmax for logits
    pub(super) batched_embed_kernel: KernelHandle,
    pub(super) fill_slots_kernel: KernelHandle,
    /// Cached CUDA graph for single-sequence decode (layer loop + norm + LM head).
    /// CUDA graph cache for n=1 decode, keyed by `seq.slot_idx`. The captured
    /// graph has SSM h_state/conv_state pointers baked in as kernel arguments,
    /// so a graph captured for slot S can ONLY be replayed for slot S — replay
    /// for any other slot reads/writes the wrong sequence's recurrent state
    /// and produces gibberish for both sequences. With concurrent users we may
    /// alternate between slots in n=1 decode (e.g. via the per-seq fresh-decode
    /// fix in scheduler::step_decode_only), so we keep one graph per slot.
    pub(super) decode_graph: Mutex<std::collections::HashMap<usize, GraphHandle>>,
    /// Cached CUDA graphs for batched decode, keyed by padded batch size.
    pub(super) batch_decode_graphs: Mutex<HashMap<usize, GraphHandle>>,
    /// Pre-allocated SSM state pool for stable GPU addresses across graph replays.
    pub(super) ssm_pool: SsmStatePool,
    /// SSM state snapshot pool for Marconi prefix caching.
    pub(super) ssm_snapshots: SsmSnapshotPool,
    /// Fixed max blocks per sequence (max_seq_len / block_size + 1).
    /// Used as constant stride in attention metadata for CUDA graph compatibility.
    pub(super) max_blocks_per_seq: u32,
    /// Permanent KV cache block for padding sequences in batched decode.
    pub(super) dummy_kv_block: u32,
    /// Profile mode: skip graphs, sync+time each layer. Set ATLAS_PROFILE=1.
    pub(super) profile: bool,
    /// One-shot profile flag for the next prefill request only. Set
    /// ATLAS_PROFILE_FIRST=1 to capture per-step timing on the first prefill
    /// after startup without disabling CUDA graphs for subsequent decodes.
    /// Consumed (atomically swapped to false) by `prefill_chunk` / `prefill`.
    pub(super) profile_first_pending: std::sync::atomic::AtomicBool,
    /// When true, decode() skips CUDA graph capture/replay. Set during
    /// per-sequence batch decode to prevent SSM state pointer baking.
    pub(super) suppress_graphs: std::sync::atomic::AtomicBool,
    /// MTP draft proposer (built from mtp_weights at init).
    pub(super) proposer: Option<Arc<dyn DraftProposer>>,
    /// Dedicated buffer for saving hidden state before MTP head runs.
    /// Size: hidden_size * 4 bytes (one FP32 vector). MTP overwrites shared
    /// buffers (norm_output etc.), so the target hidden must be saved here first.
    pub(super) mtp_hidden_save: DevicePtr,
    /// DFlash 5-layer hidden-state stack. Allocated only when a
    /// `BlockDiffusionDraftHead` proposer is built. Layout:
    /// `[5 × hidden_size × bf16]` shallow-to-deep at the layer indices
    /// declared by `dflash_capture_layers`. Holds the most-recently-decoded
    /// token's intermediate hiddens; the drafter consumes them via its `fc`
    /// projection on the next propose() call. None for non-DFlash runs.
    pub(super) dflash_hidden_save: Option<DevicePtr>,
    /// Layer indices to capture for DFlash. Empty when DFlash is disabled.
    /// Sourced from drafter's `dflash_config.target_layer_ids` at model build.
    pub(super) dflash_capture_layers: Vec<usize>,
    /// Cached CUDA graphs for K=2 verification, **keyed by `seq.slot_idx`**.
    /// Same rationale as `decode_graph`: the captured graph has SSM
    /// h_state/conv_state pointers baked in as kernel arguments, so replay for
    /// a different slot writes to the wrong sequence's recurrent state. With
    /// concurrent users alternating through MTP verify, a single
    /// `Option<GraphHandle>` would corrupt both slots' SSM state.
    pub(super) verify2_graph: Mutex<std::collections::HashMap<usize, GraphHandle>>,
    /// Cached CUDA graphs for K=3 verification, keyed by `seq.slot_idx`.
    pub(super) verify3_graph: Mutex<std::collections::HashMap<usize, GraphHandle>>,
    /// Cached CUDA graphs for K=4 verification, keyed by `seq.slot_idx`.
    pub(super) verify4_graph: Mutex<std::collections::HashMap<usize, GraphHandle>>,
    /// Cached CUDA graphs for DFlash K=γ verification, keyed by
    /// `(seq.slot_idx, K)`. K is `tokens.len()` (γ+1 typically). One graph
    /// per (slot, K) — different γ values coexist via the K dimension.
    pub(super) verify_kgamma_graph: Mutex<std::collections::HashMap<(usize, usize), GraphHandle>>,
    /// Prefix cache for KV block reuse across requests.
    pub(super) prefix_cache: Box<dyn spark_runtime::prefix_cache::PrefixCache>,
    /// Secondary CUDA stream for pipelining checkpoint D2D with MTP propose.
    pub(super) secondary_stream: u64,
    /// CUDA event for GPU-side inter-stream synchronization (avoids CPU-blocking sync).
    pub(super) secondary_event: u64,
    /// Communication backend for expert parallelism (EP) all-reduce.
    /// None for single-GPU (no distributed communication needed).
    pub(super) comm: Option<std::sync::Arc<dyn spark_comm::CommBackend>>,
    /// Small GPU buffer for EP token broadcast (4 bytes).
    pub(super) ep_cmd_buf: DevicePtr,
    /// Self-speculative decoding mode: draft via layer-skipping (no MTP weights needed).
    pub(super) self_speculative: bool,
    /// Last token index passed to save_hidden_for_mtp (for EP broadcast to rank 1).
    pub(super) last_mtp_hidden_idx: std::sync::atomic::AtomicUsize,
    /// Optional vision encoder for VL models (Qwen3-VL).
    pub(super) vision_encoder: Option<crate::layers::VisionEncoder>,
    /// Number of patches encoded by the last prepare_vision_embed() call.
    /// 0 means no vision embeddings pending.
    pub(super) vision_embed_patches: Mutex<usize>,
    /// Per-image `(grid_h_post_merge, grid_w_post_merge)` from the most
    /// recent prepare_vision_embed() call. Used by MRoPE prefill to
    /// assign correct (h, w) spatial position IDs to each image patch
    /// token. Empty when no images are pending.
    pub(super) vision_image_grids: Mutex<Vec<(usize, usize)>>,
    /// Page-locked host staging for batched metadata H2D transfers.
    /// Allocated once at init via cuMemAllocHost, freed in Drop.
    ///
    /// Uses UnsafeCell (not Mutex) because TransformerModel is only accessed
    /// from the scheduler thread after construction. The Model trait requires
    /// Send+Sync for the move to the scheduler thread, but the model is never
    /// accessed from multiple threads simultaneously. A Mutex here caused a
    /// 500x EP=2 decode regression (50 tok/s → 0.1 tok/s) due to contention
    /// with the NCCL all-reduce path.
    pub(super) pinned_staging: std::cell::UnsafeCell<PinnedMetaStaging>,
    /// Save SSM snapshots every N blocks during chunked prefill.
    /// 0 = disabled (leaf-only). When > 0, intermediate checkpoints are saved
    /// at block boundaries, enabling partial prefix SSM restore.
    pub(super) ssm_checkpoint_interval: usize,
    /// Kernel handle for fused SSM state normalization (prevents state explosion
    /// during long chunked prefill — the SSM forgetting bug).
    pub(super) ssm_state_norm_kernel: KernelHandle,
    /// GPU buffer for ssm_state_clamp_norm_fused's pointer table [num_ssm_layers].
    pub(super) ssm_norm_ptrs_buf: DevicePtr,

    // ── Two-phase SSM prefill buffers ──
    // These hold GDN inputs/outputs for the full sequence, allowing the GDN
    // recurrence to run in a single kernel launch while GEMM projections are
    // processed in smaller chunks (memory-bounded).
    //
    // Allocated at model init for max_seq_len tokens. Reused across layers
    // (only one layer runs at a time) and across sequences.
    /// Packed QKV for two-phase SSM prefill: [max_seq_len, conv_dim] BF16.
    /// Layout per token: [Q(key_dim) | K(key_dim) | V(value_dim)].
    pub(super) gdn_buf_qkv: DevicePtr,
    /// Interleaved gate/beta for two-phase SSM prefill: [max_seq_len, 2*num_v_heads] FP32.
    /// Layout per token: [gate(nv) | beta(nv)].
    pub(super) gdn_buf_gate_beta: DevicePtr,
    /// Full-sequence GDN output: [max_seq_len, value_dim] BF16
    pub(super) gdn_buf_out: DevicePtr,
    /// Full-sequence Z gate (for gated RMS norm in phase 3): [max_seq_len, value_dim] BF16
    pub(super) gdn_buf_z: DevicePtr,
    /// Max sequence length these buffers were allocated for.
    pub(super) gdn_buf_max_len: usize,

    /// Logit softcapping kernel: logits = cap * tanh(logits / cap).
    /// KernelHandle(0) = disabled (no softcapping for this model).
    pub(super) logit_softcap_kernel: KernelHandle,
    /// FP32 variant of logit softcap. KernelHandle(0) when not loaded.
    /// Used when `use_fp32_logits` is true.
    pub(super) logit_softcap_fp32_kernel: KernelHandle,
    /// Whether the single-token decode LM head produces FP32 logits (rather
    /// than BF16). True when `config.use_fp32_residual()` AND the LM head is
    /// a dense BF16 weight (no NVFP4 quant). Drives:
    ///   - dense_gemv_bf16_fp32out kernel writes to `logits_fp32_buf`
    ///   - logit_softcap_fp32 kernel applied in place on the FP32 buffer
    ///   - sampler reads FP32 directly, skipping BF16→FP32 expansion
    ///
    /// Gated by config.model_type=="gemma4" via use_fp32_residual() — other
    /// models keep the BF16 path. Prefill / batched-decode lm_head still
    /// write BF16 to `buffers.logits()`; only single-token decode is FP32
    /// because the bug it fixes (greedy argmax tiebreak flip on the BF16
    /// representable boundary at value 16-32) only manifests there.
    pub(super) use_fp32_logits: bool,
    /// FP32 logits scratch [vocab_size × 4 bytes]. NULL when `use_fp32_logits`
    /// is false (no allocation).
    pub(super) logits_fp32_buf: DevicePtr,
    /// Embedding scale kernel: embeddings *= sqrt(hidden_size).
    /// KernelHandle(0) = disabled (no scaling for this model).
    pub(super) embed_scale_kernel: KernelHandle,
}

/// Pinned host memory staging buffer with reusable metadata Vecs.
pub(crate) struct PinnedMetaStaging {
    /// Page-locked host buffer (cuMemAllocHost).
    pub(super) ptr: *mut u8,
    /// Size in bytes.
    pub(super) bytes: usize,
    /// Reusable Vec<u32> for positions (avoids per-chunk heap allocation).
    pub(super) positions: Vec<u32>,
    pub(super) positions_h: Vec<u32>,
    pub(super) positions_w: Vec<u32>,
    /// Reusable Vec<i64> for slot mappings (avoids per-chunk heap allocation).
    pub(super) slots: Vec<i64>,
}

// SAFETY: TransformerModel is constructed on the main thread, then moved to
// the scheduler thread via Box<dyn Model>. After the move, ALL access
// (prefill, decode, batch_decode) happens on the single scheduler thread.
// The Model trait requires Send+Sync for the cross-thread move, but the
// Model is moved to the scheduler thread and accessed exclusively from there.
// UnsafeCell<PinnedMetaStaging> is not inherently Sync, but single-thread
// access is enforced at runtime by the scheduler architecture.
// The raw pointer in PinnedMetaStaging points to cuMemAllocHost memory which
// is process-global and valid from any thread.
unsafe impl Send for TransformerModel {}
// SAFETY: Model methods are only called from the scheduler thread. No concurrent &self access.
unsafe impl Sync for TransformerModel {}
