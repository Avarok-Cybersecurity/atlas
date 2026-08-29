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
pub use transformer_layer::{
    TransformerLayer, VERIFY_WY_LAYER_STRIDE_BYTES, VERIFY_WY_TABLE_SEQS,
    VERIFY_WY_TABLE_STRIDE_BYTES, VERIFY_WY_TABLES_PER_LAYER,
};

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

/// Attention-layer per-sequence state. KV lives in `PagedKvCache`; the only
/// resident piece is the QSA indexer carry on the 12 qwen4_exp
/// full-attention layers (Avarok #753 item B).
#[derive(Default)]
pub struct AttnLayerState {
    pub qsa: Option<crate::layers::qsa::QsaSeqState>,
}

impl LayerState for AttnLayerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

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
    /// Storage dtype of `h_state`: `false` = FP32, `true` = FP16
    /// (`--ssm-h-dtype f16`).
    ///
    /// This is the single source of truth for the h-state format. Which edge
    /// sets it depends on the POOL width:
    ///
    /// * FP32-sized pool (stage 1/2, `h_prefill_stage == None`): prefill is
    ///   the only FP32 writer and writes the slot in place, so the flag
    ///   starts `false` and the decode mixer
    ///   (`TransformerModel::ssm_h_to_f16_dispatch`) flips it exactly once
    ///   per sequence, on the first decode step. No caller has to know where
    ///   the prefill->decode edge is.
    /// * f16-SIZED pool (stage 3, `h_prefill_stage == Some`): the slot is
    ///   physically 2 bytes/element and can NEVER hold FP32, so the flag is
    ///   `true` from allocation onwards and the decode mixer is a no-op.
    ///   Prefill's FP32 kernels run over [`Self::h_prefill_stage`] instead.
    ///
    /// It rides through swap-out/swap-in because `state_io` mutates these
    /// states in place rather than rebuilding them.
    pub h_is_f16: bool,
    /// Stage-3 f16-SIZED pool ONLY (`--ssm-h-dtype f16-pool`): the FP32
    /// staging blob for THIS sequence's slot, which the GDN prefill widens
    /// `h_state` into before its FP32 kernels run and narrows back after.
    ///
    /// `None` — every configuration before stage 3 — means "the slot IS
    /// FP32-wide": prefill writes `h_state` in place exactly as it always
    /// has, and not one byte moves. The same blob is shared by every layer
    /// of the sequence (see `SsmStatePool::h_prefill_stage`).
    pub h_prefill_stage: Option<DevicePtr>,
    /// PLE per-sequence carry (n-gram history + dilated-conv state), present
    /// only on the layer that hosts a `PleLayer` (Avarok #753 item B: one per
    /// in-flight sequence, lazily created on the sequence's first pass).
    pub ple: Option<crate::layers::ple::PleSeqState>,
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
    /// M2 per-request LoRA routing: `[num_seqs]` i32 at this device address,
    /// one adapter SLOT index per row (`< 0` = base / no delta; pad rows are
    /// `-1`). Uploaded each decode step to a stable address (like positions /
    /// block_table), so the batched bgmv stays inside the captured decode
    /// graph. `DevicePtr(0)` on every non-routed path (single-seq decode,
    /// prefill, verify, MLA, MTP) — the bgmv apply sites no-op when it is null.
    pub seq_slot: DevicePtr,
    /// SOLID Incr-4 (batched decode MoE fold): `[num_seqs]` i32 per-row adapter
    /// map for the MoE expert gather-BGMV fold, at this device address. MoE
    /// semantics (distinct from `seq_slot`): `< 0` = base / no fold (device
    /// kernel skips the row); `>= 0` = fold the installed active adapter's
    /// per-expert delta on that row. Built by
    /// [`crate::lora::build_moe_row_adapter_decode`] and uploaded each decode
    /// step to a stable address (a dedicated fixed-address buffer,
    /// `TransformerModel::moe_row_adapter_buf`, alloc'd once at init), so the
    /// batched fold stays inside the captured decode graph and is route-agnostic across
    /// replays (base rows no-op individually). `DevicePtr(0)` when no adapter is
    /// resident and on every non-batched path (the fold hooks then fall back to
    /// the request-granularity `moe_route_gate`). NOT the `seq_slot` buffer —
    /// that resolves `-1 → active` (attention defer-to-active), which would fold
    /// the adapter onto base rows here.
    pub moe_row_adapter: DevicePtr,
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
    /// Per-stream chunk_len. In the legacy same-length path this is uniform; in
    /// the VARLEN path (`cu_seqlens` populated) it is the MAX per-stream length
    /// (retained only for buffer-bound/debug use — per-stream lengths come from
    /// `cu_seqlens`).
    pub chunk_len: u32,
    /// Total tokens stacked across streams. Legacy: `batch_size * chunk_len`.
    /// VARLEN: `Σ per-stream lengths` (= `cu_seqlens_host[batch_size]`).
    pub total_tokens: u32,
    /// VARLEN geometry: `[batch_size+1]` i32 prefix-sum of per-request token
    /// counts, on device (read by the GDN kernel + FlashInfer). `DevicePtr::NULL`
    /// in the legacy same-length path (callers fall back to `b*chunk_len`).
    pub cu_seqlens: DevicePtr,
    /// Host copy of `cu_seqlens` (`[batch_size+1]` i32) — FlashInfer's PrefillPlan
    /// dereferences the indptr on the CPU, and per-request slice offsets are
    /// computed host-side. Empty in the legacy path.
    pub cu_seqlens_host: Vec<i32>,
    /// VARLEN geometry: per-stream KV length `[batch_size]` i32 on device,
    /// `kv_lens[b] = chunk_start + per-stream token count`. The batched paged
    /// attention kernels need this per stream: a single scalar at the MAX
    /// makes short streams index their block_table past the blocks they
    /// actually own, and applies the wrong causal bound. `DevicePtr::NULL` in
    /// the legacy same-length path (kernels fall back to the scalar `kv_len`).
    pub kv_lens: DevicePtr,
    /// Host copy of `kv_lens`. Empty in the legacy path.
    pub kv_lens_host: Vec<i32>,
    /// Maximum block_table length across the batch (kernel uses for
    /// bounds checking; per-stream block_table reads via the pointer
    /// array dereference).
    pub max_blocks_per_seq: u32,
    /// Exact byte footprint of this metadata block within the scratch
    /// buffer (from `scratch_offset_bytes` to the end of `seq_len_ptrs`).
    /// SSOT for the caller's scratch-cursor advance — the per-SSM-layer
    /// `h_state_ptrs` slot is placed at `scratch_cursor + staged_bytes`, so
    /// an under-estimate here would overwrite the live `slot_stacked` array
    /// with device pointers and produce wild KV-cache slots (#110 bug #2).
    pub staged_bytes: usize,
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
    /// mHC highway ROW offset for this pass (#753 item B, mixed steps):
    /// the fused decode+prefill step gives the prefill chunk highway rows
    /// at `padded_n` so they live disjoint from the decode rows, mirroring
    /// the hidden/residual layout. 0 everywhere else.
    pub hc_row_offset: usize,
    /// GPU backend for kernel launches and memory ops.
    pub gpu: &'a dyn GpuBackend,
    /// Model configuration (dimensions, hyperparameters).
    pub config: &'a ModelConfig,
    /// Which GEMM implementation each projection takes. Carried rather than
    /// read from a static so it cannot outlive the model whose flags it
    /// encodes — see `layers::ops::GemmDispatch`.
    pub dispatch: &'a crate::layers::ops::GemmDispatch,
    /// Re-encoded copies of this model's weights, memoized for this model's
    /// lifetime. Carried rather than kept in a static keyed by device pointer,
    /// where a recycled address would HIT after a model swap.
    pub derived: &'a crate::layers::ops::DerivedWeights,
    /// Kernel-path levers for this model — the SSM/GDN variant, FFN routing,
    /// MoE quantization, LoRA mode, diagnostics. The non-GEMM half of the
    /// lever set; `dispatch` is the GEMM half.
    pub levers: &'a crate::layers::ops::ModelLevers,
    /// This model's diagnostic counters and one-shot dump latches. Carried
    /// for the same reason as `levers`: a counter that spans a model swap
    /// averages two models and describes neither, and a one-shot latch that
    /// already fired swallows the next model's dump.
    pub stats: &'a crate::layers::ops::ModelStats,
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
    /// True when this prefill pass continues from a restored Marconi SSM
    /// snapshot (warm prefix-cache hit). GDN layers must then take the
    /// bit-faithful WY4 recurrence instead of the FLA chunked kernel: FLA's
    /// chunk grid is anchored at the (arbitrary) snapshot offset and its
    /// bf16 intermediates drift vs the pass that originally produced the
    /// cached K/V, and the replay range [snap_tok, matched) is rewritten
    /// into SHARED prefix-cache blocks — non-exact recompute poisons them
    /// and the drift ratchets across turns (2026-06-10 warm-hit stutter).
    pub gdn_exact_replay: bool,
    /// Device `[num_tokens]` u32 token IDs for the tokens being processed this
    /// pass, in the SAME order the per-token MoE loop visits them. Required by
    /// DeepSeek-V4 hash-MoE layers (static `tid2eid[token_id]` routing); `None`
    /// for models without hash routing. Must be a STABLE address across the
    /// layer loop (and, under CUDA-graph decode, uploaded before each replay).
    pub token_ids: Option<DevicePtr>,
    /// HOST copy of the same token ids, when the caller had them in hand
    /// (decode always does — it uploads `token_ids` FROM this value; chunked
    /// prefill likewise). PLE computes its n-gram ids on the host, and
    /// reading them back off the device costs a synchronous D2H per decode
    /// step — pure overhead, and capture-unsupported inside a CUDA graph.
    pub host_token_ids: Option<&'a [u32]>,
    /// #30 (routed-prefill precision): the REQUEST slot's per-layer LoRA pairs,
    /// GLOBAL-layer-indexed (`len == num_hidden_layers`), set ONLY at the prefill
    /// entries and ONLY when the request routes to a NON-active slot. `Some` makes
    /// the K/V/O prefill apply sites select the request slot's pair and fold it
    /// through the SAME dense `apply_lora_delta` (dense_gemm_tc) the ACTIVE adapter
    /// uses — numerically identical to serving that adapter active, instead of the
    /// per-row bgmv (whose fp accumulation order tips razor-margin tokens). `None`
    /// (active/base request, no LoRA, and every decode/verify/mtp/moe pass) leaves
    /// the installed-active-pair path byte-identical. Prefill runs eager
    /// (`graph_capture: false`) so this per-pass CPU borrow is safe.
    pub routed_lora_layers: Option<&'a [Option<crate::lora::LoraLayerWeights>]>,
    /// Default-ON mid-chunk SSM tail capture (opt-out `ATLAS_SSM_TAIL_MIDCHUNK=0`).
    ///
    /// `Some` only on the single prefill pass whose local token range spans
    /// the block-floored matched-prefix boundary `tb`. GDN/SSM layers then
    /// split their recurrent (h_state) and conv (conv_state) kernels at
    /// `cap_local` and copy the @tb state into the reserved snapshot slot.
    /// `None` (default) => no split, byte-identical to prior behavior.
    pub midchunk_capture: Option<MidchunkCapture<'a>>,
    /// Feature-1 MoE-LoRA per-request fold decision for this forward pass,
    /// resolved by `TransformerModel::moe_lora_route` from the owning request's
    /// `adapter_slot`. Governs the prefill router/expert fold hooks
    /// (`layers/moe/lora.rs`). Ignored when no MoE adapter is installed
    /// (`self.lora == None` short-circuits first — byte-identical off). Default
    /// `Fold` keeps legacy single-request call sites unchanged.
    pub moe_lora_route: MoeLoraRoute,
}

/// Per-pass descriptor for mid-chunk SSM tail capture. Points at the reserved
/// Marconi snapshot slot's per-SSM-layer destination buffers (already offset to
/// the slot) plus the split point in local (chunk) token coordinates.
///
/// `ssm_layer_counter` is a fresh per-pass counter: each SSM layer's prefill
/// increments it once, in model order, so the value indexes `h_dsts`/`conv_dsts`
/// (which are in the same SSM-layer order as the snapshot pool).
pub struct MidchunkCapture<'a> {
    /// Split point in local token coordinates: capture state AFTER this many
    /// tokens (== `tb - proc_start`).
    pub cap_local: usize,
    /// Per-SSM-layer h_state snapshot destination (offset to the reserved slot).
    pub h_dsts: &'a [DevicePtr],
    /// Per-SSM-layer conv_state snapshot destination (offset to the reserved slot).
    pub conv_dsts: &'a [DevicePtr],
    /// Bytes per layer of h_state.
    pub h_bytes: usize,
    /// Bytes per layer of conv_state.
    pub conv_bytes: usize,
    /// Fresh per-pass SSM-layer ordinal counter (model order == pool order).
    pub ssm_layer_counter: &'a std::sync::atomic::AtomicUsize,
    /// Optional SECOND capture one KV block earlier, at `tb - block_size`
    /// (local split point `cap_local - block_size`). `Some` only when the pass
    /// also covers that point. On ~5/19 warm turns the next turn's block-floored
    /// `matched_tokens` lands exactly `tb - block_size` (generation-suffix /
    /// retokenize divergence), one block short of the tail; registering this
    /// earlier restore point makes those turns zero-replay too.
    pub cap_local_early: Option<usize>,
    /// Per-SSM-layer h_state dst for the `tb - block_size` slot (offset applied).
    pub h_dsts_early: &'a [DevicePtr],
    /// Per-SSM-layer conv_state dst for the `tb - block_size` slot.
    pub conv_dsts_early: &'a [DevicePtr],
}

/// Feature-1 MoE-LoRA fold decision for a single forward pass.
///
/// The MoE router/expert delta is a SINGLE globally-installed adapter (phase 1):
/// this gate makes the prefill fold per-request without a device kernel, exactly
/// mirroring the attention BGMV `seq_slot < 0` skip but at request granularity.
/// A base request pays nothing; a packed/mixed batch refuses loudly rather than
/// fold one adapter onto every row (the device-side per-row fold that would let
/// a mixed batch skip base rows individually is the documented follow-up —
/// `docs/design/lora-solid.md` Incr 1/3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MoeLoraRoute {
    /// Single-request pass whose request owns the installed active MoE adapter:
    /// FOLD. Genuine single-seq decode now folds the router + expert gate/up/down
    /// deltas at this altitude (mirroring prefill); the remaining back-compat
    /// paths (multi-seq / verify) still bail via `reject_decode_lora` before the
    /// fold, so their `Fold` value is inert there. Also the default.
    #[default]
    Fold,
    /// Single-request pass that is base (`adapter_slot < 0`, no adapter) or
    /// routes to a different, non-installed adapter: SKIP the fold entirely.
    /// Base tokens pay nothing (request-granularity mirror of the attention
    /// BGMV `seq_slot < 0` early-return).
    Skip,
    /// Multi-request / packed / codispatch batch whose per-row adapter identity
    /// cannot be honored without the device-side per-row fold (follow-up): the
    /// fold REFUSES loudly rather than mis-apply one adapter to every row.
    Refuse,
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
