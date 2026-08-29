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
use crate::weight_map::{DenseWeight, Fp8DenseWeight, MtpWeights, QuantizedWeight};

/// Architecture-agnostic transformer model.
///
/// Composes `Vec<Box<dyn TransformerLayer>>` into a full forward pass.
/// Adding a new model only requires implementing [`TransformerLayer`]
/// for each layer type — the model loop stays unchanged.
#[allow(dead_code)]
/// Rows in the drafter catch-up hidden ring (see `mtp_catchup_ring`):
/// 512 covers the gate's 256-token serial re-probe interval with 2x margin.
pub(super) const MTP_CATCHUP_RING_ROWS: usize = 512;

pub struct TransformerModel {
    pub(super) config: ModelConfig,
    /// Which GEMM implementation each projection takes, resolved from the
    /// environment when this model was built. Owned here and borrowed by every
    /// `ForwardContext` this model creates, so the choice cannot outlive the
    /// model — the property nine `OnceLock` statics could not have.
    pub(super) dispatch: crate::layers::ops::GemmDispatch,
    /// Weight re-encodings derived on demand and memoized for this model.
    /// Dropped with the model, so no entry can outlive the allocation it
    /// describes.
    pub(super) derived: crate::layers::ops::DerivedWeights,
    /// The weight ledger this model was built from.
    ///
    /// Held for TEARDOWN, not for lookup: the layers already copied the
    /// pointers they need out of it during construction. It is the only
    /// structure that knows every weight allocation, and it used to be dropped
    /// at the end of `startup()` — leaving that memory live with nothing able
    /// to free it. `None` once released.
    pub(super) weight_store: Option<spark_runtime::weights::WeightStore>,
    /// Non-GEMM kernel-path levers, resolved at model construction.
    pub(super) levers: crate::layers::ops::ModelLevers,
    /// Diagnostic counters and one-shot dump latches for this model. Sibling
    /// to `levers`: what the kernels did, rather than what they do.
    pub(super) stats: crate::layers::ops::ModelStats,
    pub(super) embed_tokens: DenseWeight,
    /// Fused n-gram input embedding (LongCat family), when the architecture
    /// has one. `Mutex` because the forward path is `&self` while the row
    /// cache mutates on lookup; the lock is taken once per embed, which is
    /// nothing beside a transformer forward.
    pub(super) ngram_embed: Option<std::sync::Mutex<crate::layers::ngram_embed::NgramEmbedding>>,
    pub(super) final_norm: DenseWeight,
    pub(super) lm_head_weight: DenseWeight,
    pub(super) lm_head_nvfp4: Option<QuantizedWeight>,
    /// TRANSPOSED `[K/2, ldb]` twin of `lm_head_nvfp4` + its PADDED row stride.
    ///
    /// The pad is load-bearing: the tile GEMM reads B with 16-byte `cp.async`,
    /// which needs a 16-byte-aligned source, and row r sits at `r * stride`.
    /// This checkpoint's vocab is 248077 — ODD — so an unpadded stride misaligns
    /// 15 of every 16 k-rows and faults with CUDA 716. Padded to 248192.
    ///
    /// ADDITIVE: never replaces or aliases `lm_head_nvfp4`, so every existing
    /// holder (including the `draft_lm_head_nvfp4` copy at `impl_a1.rs:157`)
    /// keeps a valid row-major pointer. Built once, immutable, never freed —
    /// so each per-`padded_n` CUDA graph binds one (kernel, tensor) pair.
    /// `None` under `ATLAS_NO_LMHEAD_TGEMM=1`.
    pub(super) lm_head_nvfp4_t: Option<(QuantizedWeight, u32)>,
    /// Runtime FP8 E4M3 LM head (per-row scales), decoded via `w8a16_gemv`.
    /// `Some` only when `--lm-head-dtype fp8` was requested; mutually exclusive
    /// with `lm_head_nvfp4` (that stays `None` on the FP8 path). Additive: when
    /// `None`, the NVFP4/BF16 LM-head dispatch is byte-identical to before.
    pub(super) lm_head_fp8: Option<Fp8DenseWeight>,
    pub(super) layers: Vec<Box<dyn TransformerLayer>>,
    pub(super) buffers: BufferArena,
    /// Startup-static LoRA adapter (pool + per-layer pairs + M2 pointer
    /// tables). `None` = no adapter. Installed post-construction via
    /// `set_lora_weights`, which also copies the per-layer pairs into the
    /// layer structs; kept here as the owner of the pool/tables and for
    /// status introspection.
    pub(super) lora: Option<crate::lora::LoraWeights>,
    /// True when runtime adapter rotation is ARMED: `ATLAS_LORA_ROTATE=1`, or
    /// `$ATLAS_LORA_PEER` set. Armed ⇒ decode runs eager (no CUDA-graph
    /// capture) so a `set_active_lora` re-point is immediately live
    /// (eager-on-rotate). `false` (single startup adapter, no rotation env)
    /// keeps the decode-graph path byte-identical to today.
    pub(super) lora_rotatable: bool,
    pub(super) kv_cache: Mutex<PagedKvCache>,
    pub(super) gpu: Box<dyn GpuBackend>,
    /// TQ+ InnerQ calibration driver, when `TURBO_INNERQ` is set. Owned here
    /// rather than parked in a static: it writes `__device__` globals in THIS
    /// model's modules, so it must not outlive the model. Reached from the
    /// scheduler through `Model::poll_innerq`.
    #[cfg(feature = "cuda")]
    pub(super) innerq: Option<crate::layers::qwen3_attention::InnerQDriver>,
    pub(super) rms_norm_kernel: KernelHandle,
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
    /// Tile GEMM over the TRANSPOSED lm_head twin. 0 when absent.
    pub(super) w4a16_gemm_t_kernel: KernelHandle,
    /// LOSSLESS BF16-MMA tile GEMM over the same twin. Preferred for lm_head:
    /// `w4a16_gemm_t` downcasts activations BF16->FP8 E4M3, and lm_head is the
    /// layer where a near-tie argmax flip changes the emitted token. Memory
    /// records exactly that failure mode (stop/end-of-turn mis-ranking on DEEP
    /// agentic trajectories) for sub-bf16 lm_heads. Costs ~1% of step.
    /// 0 when absent. Kill switch: ATLAS_NO_LMHEAD_LOSSLESS=1.
    pub(super) w4a16_gemm_t_bf16_kernel: KernelHandle,
    pub(super) w4a16_gemm_kernel: KernelHandle,
    pub(super) w4a16_gemv_batch2_kernel: KernelHandle,
    /// Narrow `w4a16_gemv_batch{M}` family (M=4..8) for the K=3..8 verify
    /// lm_head (one weight read for all rows; nsys 2026-07-18: the M64-tile
    /// `w4a16_gemm` at M=4 cost 19.3 ms/verify-step on the 248320-row lm_head
    /// — 94% tile padding). Individual tiers are 0-handles when the target
    /// lacks them (dispatch falls back).
    pub(super) w4a16_batchm: crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers,
    pub(super) w4a16_gemv_batch16_kernel: KernelHandle,
    /// FP8 E4M3 LUT GEMV (M=1) for the FP8 LM head. Only used when
    /// `lm_head_fp8.is_some()`; loaded unconditionally (cheap handle) so the
    /// dispatch in `lm_head` / batched-decode / verify can reference it.
    pub(super) dense_gemv_fp8w_kernel: KernelHandle,
    /// FP8-weight dual-GEMV (batch=2): reads the FP8 weight once for both K=2
    /// verify tokens. Bit-identical to two `dense_gemv_fp8w` calls; halves the
    /// FP8 weight bandwidth for the lm_head on the MTP verify path.
    pub(super) dense_gemv_fp8w_batch2_kernel: KernelHandle,
    pub(super) dense_gemm_kernel: KernelHandle,
    /// Batched BF16 GEMV (M rows, one weight pass). Used for the BF16 lm_head
    /// at decode: reads the ~617 MB vocab weight once with coalesced uint4
    /// loads, vs the scalar dense_gemm_bf16 (16x16 FFMA, ~89 GB/s). 0 = absent.
    pub(super) dense_gemv_batchm_kernel: KernelHandle,
    pub(super) argmax_kernel: KernelHandle,
    /// Batched argmax (one block per row). 0 when the kernel set lacks it.
    pub(super) argmax_batch_kernel: KernelHandle,
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
    /// Cached CUDA graphs for batched decode, keyed by the per-row SSM pool
    /// slot VECTOR (`trait_impl/decode_graph_key.rs`) — the only per-sequence
    /// addresses a capture bakes. The old `padded_n` key was sound only while
    /// the batch was exactly slots `[0..n)` with `n == padded_n`; the MTP
    /// Phase-A bootstrap passes a slot SUBSET of the active set and would
    /// replay another subset's baked GDN pointers.
    /// Value = `(graph, last_use_tick)`; the `u64` alongside the map is the
    /// monotonically increasing tick. At `BATCH_DECODE_GRAPH_CAP` entries the
    /// least-recently-used graph is destroyed and replaced.
    pub(super) batch_decode_graphs: Mutex<(HashMap<Vec<u32>, (GraphHandle, u64)>, u64)>,
    /// Pre-allocated SSM state pool for stable GPU addresses across graph replays.
    /// `Arc` so each `SequenceState` can hold a `SlotGuard` that releases its
    /// claimed slot on drop — guaranteeing the slot returns to the free list on
    /// EVERY sequence-exit path (normal finish, abort, error, swap-out failure,
    /// panic/unwind), not just the explicit `free_sequence`/`compact_sequence`
    /// sites. See `SsmStatePool::claim_guarded` / `SlotGuard`.
    pub(super) ssm_pool: Arc<SsmStatePool>,
    /// SSM state snapshot pool for Marconi prefix caching.
    pub(super) ssm_snapshots: SsmSnapshotPool,
    /// Optional SSM snapshot spill tier (`ATLAS_SSM_TIER`). `None` (default)
    /// keeps the drop-only reclaim path byte-identical; `Some` moves an evicted
    /// snapshot's bytes to the tier (keeping its index entry findable) so a warm
    /// turn faults it back instead of recomputing. Threaded into
    /// [`SsmSnapshotPool::reclaim_from_cache`] at every reclaim call site.
    pub(super) ssm_tier_store: Option<Arc<dyn super::ssm_tier::SnapshotBlobStore>>,
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
    /// Batched-verify hidden stash: `[8, hidden_size]` BF16 — one RAW-hidden
    /// row per batched-verify sequence (n ≤ 8 envelope). Every drafter
    /// `forward_one` writes its hidden into `buffers.hidden_states()`
    /// (mtp_multi.rs), so seq 0's propose clobbers seq 1..n's verify hidden
    /// rows; the batched verdict path copies each sequence's accepted-row
    /// hidden here FIRST (`stash_verify_hidden_rows`), then feeds the drafter
    /// from the stash (`save_hidden_for_mtp_from_stash`). NULL without MTP.
    pub(super) verify_hidden_stash: DevicePtr,
    /// ATLAS_MTP_CATCHUP: circular per-position final-hidden ring captured
    /// during serial-decode stretches (BF16 rows, slot = position % ring
    /// len). Feeds the drafter catch-up on the next propose. NULL when the
    /// feature is off or no proposer exists.
    pub(super) mtp_catchup_ring: DevicePtr,
    /// (first_position, count) of the contiguous position range currently
    /// resident in the ring; a non-contiguous capture resets the range.
    pub(super) mtp_catchup_meta: parking_lot::Mutex<(usize, usize)>,
    /// ATLAS_MTP_DRAFTER_PREFILL: per-position final-layer hidden capture for
    /// the whole prompt, `[max_seq_len, hidden_size]` BF16 (~335 MB at 32k /
    /// h=5120). NULL unless the env is set AND an MTP proposer is built.
    /// Filled contiguously by the prefill chunk epilogues; consumed once by
    /// the drafter-prefill pass on the first propose() of a sequence.
    pub(super) mtp_prefill_hidden: DevicePtr,
    /// Row capacity of `mtp_prefill_hidden` (== max_seq_len at alloc; 0 when
    /// the feature is off). SSOT for the capture bounds check.
    pub(super) mtp_prefill_capacity: usize,
    /// Rows of `mtp_prefill_hidden` captured contiguously from position 0 for
    /// the CURRENT sequence. Reset to 0 on `alloc_sequence`; a chunk whose
    /// start does not extend the contiguous range (prefix-cache reuse, warm
    /// restore) leaves it stale-short, which safely disables drafter-prefill
    /// for that sequence (coverage check at the propose site).
    pub(super) mtp_prefill_capture_len: std::sync::atomic::AtomicUsize,
    /// Monotonic generation of the single-slot capture above. Bumped every
    /// time a chunk-0 prefill (re)starts the capture; the restarting
    /// sequence is stamped with the new value (`SequenceState::
    /// mtp_capture_gen`). Appends and the drafter-prefill consume require
    /// `stamp == current generation`, so at C>=2 a sequence whose capture
    /// was overwritten by ANOTHER sequence's prefill skips the drafter
    /// prefill instead of pairing its tokens with foreign hiddens. The
    /// current value IS the latest capture's generation (single atomic,
    /// SSOT). 0 = no capture ever started (matches the fresh-seq stamp 0,
    /// which is harmless: `captured >= prompt_len >= 2` fails at len 0).
    pub(super) mtp_prefill_capture_gen: std::sync::atomic::AtomicU64,
    /// ATLAS_MTP_CARRY_DRAFTER: the previous turn's drafter KV, held so the
    /// next turn of the same session can adopt it instead of rebuilding
    /// (1136 ms at 12k rows) or — as today — silently going without. Single
    /// slot: MTP is gated `active.len() == 1` on every spec path, and one slot
    /// makes block ownership unambiguous (blocks are owned here XOR by a live
    /// sequence). `None` when the feature is off or nothing has been carried.
    pub(super) mtp_carry: parking_lot::Mutex<Option<super::mtp_carry::CarriedDrafter>>,
    /// Absolute position interval `[lo, hi)` of `mtp_prefill_hidden` rows
    /// written by the CURRENT sequence's prefill chunks. Reset per
    /// `alloc_sequence`, so a warm-turn append can only ever read hiddens this
    /// turn computed — which is why the carry path cannot inherit another
    /// sequence's hiddens the way the legacy `mtp_prefill_capture_len` path
    /// can. Only maintained when ATLAS_MTP_CARRY_DRAFTER is on.
    pub(super) mtp_store_range: parking_lot::Mutex<(usize, usize)>,
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
    /// Row capacity of `dflash_hidden_save` (the K-row EAGLE capture buffer).
    /// `try_dflash_capture_all` must never write past this many rows. Single
    /// source of truth for the buffer's KMAX; 0 when DFlash is disabled.
    pub(super) dflash_hidden_save_rows: usize,
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
    /// Cached CUDA graphs for the BATCHED K-row verify (verify_e), keyed by
    /// the batch's ssm-pool slot VECTOR (+ the per-seq row count K + a
    /// wy-tables-present sentinel). Slot-vector keying is what a per-slot
    /// key cannot give at n>1: the captured graph bakes every sequence's
    /// h_state/conv_state/intermediate pointers, so it may only replay for
    /// the exact same slot assignment in the same batch order (K is in the
    /// key because a graph also bakes the R = n*K launch dimensions).
    /// Attention metadata/block tables/embeds live at fixed scratch
    /// addresses refreshed pre-replay (decode_a2 pattern).
    /// Value = `(graph, last_use_tick)`; the `u64` alongside the map is the
    /// monotonically increasing tick. At `VERIFY_BATCHED_GRAPH_CAP` entries
    /// the least-recently-used graph is destroyed and replaced (slot vectors
    /// churn with request turnover — the old insert-only map went
    /// permanently eager after 32 distinct vectors on long serves).
    pub(super) verify_batched_graphs:
        Mutex<(std::collections::HashMap<Vec<u32>, (GraphHandle, u64)>, u64)>,
    /// Batched-verify WY pointer-table staging: `num_ssm_layers` slices of
    /// `crate::layer::VERIFY_WY_LAYER_STRIDE_BYTES` ([h|Hi0|Hi1|Hi2] × 4
    /// u64 entries each) at a FIXED device address, refreshed pre-graph every
    /// batched verify step (`upload_verify_wy_tables`). Enables the
    /// single-launch table-form `gdn_decode_wy4` in the batched GDN arm.
    /// NULL without an MTP proposer (path self-gates).
    pub(super) verify_wy_tables: DevicePtr,
    /// Encoded key of the bytes CURRENTLY staged in `verify_wy_tables`, or
    /// `None` when nothing has been staged (the buffer is memset to zero at
    /// allocation, which no key describes).
    ///
    /// `upload_verify_wy_tables` ran a 48 KB host build + a 48 KB H2D on
    /// EVERY n>=2 verify step. The staged bytes are a pure function of
    /// `(k, ssm-slot vector in batch order, ghost (slot, depth) pairs)` —
    /// see `verify_wy_cache_key` for the enumeration and the proof — so a
    /// step whose key matches what is already on the device may skip both.
    /// Kill switch `ATLAS_NO_VERIFY_WY_CACHE` (PRESENCE) restores the
    /// unconditional re-stage.
    pub(super) verify_wy_cache: Mutex<Option<Vec<u64>>>,
    /// Cached CUDA graphs for DFlash K=γ verification, keyed by
    /// `(seq.slot_idx, K)`. K is `tokens.len()` (γ+1 typically). One graph
    /// per (slot, K) — different γ values coexist via the K dimension.
    pub(super) verify_kgamma_graph: Mutex<std::collections::HashMap<(usize, usize), GraphHandle>>,
    /// Cached CUDA graphs for the DFlash decode+verify fused pass, keyed by
    /// `(seq.slot_idx, M)` where M = tokens.len() = 1 + num_drafts.
    /// Replaces the separate `decode_graph` (M=1) + `verify{k}_graph` (M=k)
    /// on the DFlash path with a single M-row weight sweep.
    pub(super) fused_graph: Mutex<std::collections::HashMap<(usize, usize), GraphHandle>>,
    /// Prefix cache for KV block reuse across requests.
    pub(super) prefix_cache: Box<dyn spark_runtime::prefix_cache::PrefixCache>,
    /// Secondary CUDA stream for pipelining checkpoint D2D with MTP propose.
    pub(super) secondary_stream: u64,
    /// CUDA event for GPU-side inter-stream synchronization (avoids CPU-blocking sync).
    pub(super) secondary_event: u64,
    /// CUDA event ordering SSM-snapshot SAVES (on the default stream) before a
    /// later warm Marconi RESTORE (on the prefill stream). Marconi saves
    /// (`decode_marconi_checkpoint`, `finish_leaf_snapshot`, prefill-time
    /// `prefill_save_snapshot`) record this event after their D2D copies; a
    /// warm restore in `prefill_b_prefix_lookup` waits on it before reading the
    /// snapshot region. Without this cross-stream edge, under concurrent
    /// batched traffic the restore (prefill stream) can read a snapshot slot
    /// whose save D2D (default stream) has not yet completed — restoring stale
    /// / torn SSM recurrent state and diverging the warm decode from the cold
    /// reference (the prefix-cache × hybrid-SSM warm-restore corruption).
    pub(super) snapshot_event: u64,
    /// Communication backend for expert parallelism (EP) all-reduce.
    /// None for single-GPU (no distributed communication needed).
    pub(super) comm: Option<std::sync::Arc<dyn spark_comm::CommBackend>>,
    /// Small GPU buffer for EP token broadcast (4 bytes).
    pub(super) ep_cmd_buf: DevicePtr,
    /// EP wire-protocol version. When true, the seq_id-preamble protocol
    /// extension from atlas#99 is active — every command broadcast is
    /// preceded by a `seq_id` broadcast so the worker can dispatch
    /// slot-bound work into the right `SequenceState` slot. When false,
    /// the legacy single-sequence protocol is used. Set at construction
    /// from `ATLAS_EP_PROTOCOL` env var; both ranks must agree.
    pub(super) ep_protocol_v2: bool,
    /// Self-speculative decoding mode: draft via layer-skipping (no MTP weights needed).
    pub(super) self_speculative: bool,
    /// Last token index passed to save_hidden_for_mtp (for EP broadcast to rank 1).
    pub(super) last_mtp_hidden_idx: std::sync::atomic::AtomicUsize,
    /// Optional vision encoder for VL models (Qwen3-VL).
    pub(super) vision_encoder: Option<crate::layers::VisionEncoder>,
    /// Number of patches encoded by the last prepare_vision_embed() call.
    /// 0 means no vision embeddings pending.
    pub(super) vision_embed_patches: Mutex<usize>,
    /// Per-ITEM `(t_len, grid_h_post_merge, grid_w_post_merge)` from the most
    /// recent prepare_vision_embed() call. Used by MRoPE prefill to assign
    /// correct (t, h, w) position IDs to each vision pad token. Empty when no
    /// vision input is pending.
    ///
    /// `t_len` is the number of TEMPORAL GROUPS the item spans: 1 for a still
    /// image, `frames / temporal_patch_size` for a video. It is per item and
    /// not per encoder row on purpose — a video feeds `t_len` rows to the ViT
    /// but occupies ONE contiguous pad run, and the position builder has to
    /// treat that run as a single item whose T advances rather than as
    /// `t_len` unrelated images (which would restart T and mis-advance the
    /// running position for everything after it).
    pub(super) vision_image_grids: Mutex<Vec<(usize, usize, usize)>>,
    /// Co-dispatched batched-ViT slice base for the NEXT prefill_chunk. When a
    /// tick batches >=2 image requests into one buf_out, each request's chunk-0
    /// splice/MRoPE must read its OWN slice: `vision_row_base` = first buf_out
    /// row, `vision_grid_base` = first vision_image_grids index, and
    /// `vision_owned_images` bounds the grid scan. All 0 ⇒ legacy (read from
    /// row 0 / grid 0). Set right before prefill_chunk, reset to 0 right after.
    pub(super) vision_row_base: Mutex<usize>,
    pub(super) vision_grid_base: Mutex<usize>,
    pub(super) vision_owned_images: Mutex<usize>,
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
    /// FP16 h-state twin of the above (`ATLAS_SSM_H_FP16`). Selected from the
    /// sequence's own `SsmLayerState::h_is_f16`, so the dispatch reads the
    /// invariant rather than assuming it.
    pub(super) ssm_state_norm_f16_kernel: KernelHandle,
    /// GPU buffer for ssm_state_clamp_norm_fused's pointer table `[num_ssm_layers]`.
    pub(super) ssm_norm_ptrs_buf: DevicePtr,
    /// One-shot FP32 -> FP16 h-state converter (`ATLAS_SSM_H_FP16`).
    pub(super) ssm_h_f32_to_f16_kernel: KernelHandle,
    /// Its widening inverse. Used ONLY by the stage-3 f16-SIZED pool
    /// (`--ssm-h-dtype f16-pool`) on the BATCHED prefill path, whose GDN
    /// kernels take a device pointer TABLE and so cannot be wrapped inside
    /// the layer the way the single-stream ladder is. Zero otherwise.
    pub(super) ssm_h_f16_to_f32_kernel: KernelHandle,
    /// Staging buffer for it, one layer wide (`h_bytes / 2`). The conversion is
    /// a narrowing compaction and CANNOT be done in place: thread `2i`'s write
    /// lands inside thread `i`'s read with nothing ordering them. Allocated
    /// lazily on first use, so a serve without the flag pays nothing.
    pub(super) ssm_h_f16_scratch: std::sync::OnceLock<DevicePtr>,

    /// SOLID Incr-4: dedicated persistent GPU buffer for the batched-decode MoE
    /// per-row fold map `[max_batch_size]` i32 (`< 0` = base skip, `>= 0` = fold
    /// the active adapter). Allocated ONCE at init (fixed device address),
    /// refreshed per decode step via copy_h2d_async — graph-capture-safe exactly
    /// like the GDN buffers, and now DISTINCT from the old +160 metadata gap so
    /// seq_slot@+128 reclaims its full +128..+256 range (concurrent-LoRA decode
    /// cap 8 → 32). Always allocated (cheap, max_batch_size·4 B); never touched
    /// when self.lora is None (upload_moe_row_adapter returns DevicePtr(0)).
    pub(super) moe_row_adapter_buf: DevicePtr,

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
    /// than BF16). The FP32 logits path required an FP32 residual stream as a
    /// precondition; with the residual stream now always BF16, this is always
    /// false and the BF16 logits path is always taken.
    pub(super) use_fp32_logits: bool,
    /// FP32 logits scratch [vocab_size × 4 bytes]. NULL when `use_fp32_logits`
    /// is false (no allocation).
    pub(super) logits_fp32_buf: DevicePtr,
    /// Embedding scale kernel: embeddings *= sqrt(hidden_size).
    /// KernelHandle(0) = disabled (no scaling for this model).
    pub(super) embed_scale_kernel: KernelHandle,
    /// Feature-2 token overlay: per-adapter-slot embed/lm_head row-override
    /// tables. `None` ⇒ feature OFF ⇒ every overlay forward hook early-returns
    /// (byte-identical to a no-overlay build). Built in `set_lora_weights`
    /// (Stage 2) from the resident pool's Stage-1 raw uploads.
    pub(super) overlays: Option<crate::lora::TokenOverlaySet>,
    /// Feature-2 token overlay kernels, resolved once at construction via
    /// `try_kernel` (null-on-miss ⇒ overlay silently unused on an older image).
    pub(super) overlay_kernels: crate::layers::ops::token_overlay::OverlayKernels,
    /// Feature-2 per-forward overlay route: the current request's `adapter_slot`,
    /// stamped at each `Model::{prefill,decode,...}` entry (the scheduler drives
    /// the model serially on one thread, so a plain atomic is sufficient). The
    /// overlay hooks resolve it through `routed_prefill_slot` so a request that
    /// selects a NON-active pool adapter gets THAT adapter's overlay, not the
    /// pool's active one. `i32::MIN` marks a mixed-adapter decode batch (per-token
    /// `seq_slot` routing deferred to SOLID Incr-4) ⇒ the hooks skip.
    pub(super) overlay_route_slot: std::sync::atomic::AtomicI32,
    /// Feature-1 per-decode MoE route, stamped from the decode batch's adapter
    /// slots at each `Model::{decode,decode_batch,mixed_forward}` entry (the
    /// decode/verify `ForwardContext`s read it instead of a hardcoded `Fold`).
    /// A pure-base decode batch resolves to `Skip` so base requests decode
    /// normally even while an adapter is resident; any adapter-using row makes
    /// the batch `Fold`/`Refuse`, which `reject_decode_lora` turns into a loud
    /// bail (the decode-fold is SOLID Incr-4). Encoded 0=Skip 1=Fold 2=Refuse.
    pub(super) decode_moe_route: std::sync::atomic::AtomicI32,
}

/// Pinned host memory staging buffer with reusable metadata Vecs.
pub(crate) struct PinnedMetaStaging {
    /// Page-locked host buffer (cuMemAllocHost).
    pub(super) ptr: *mut u8,
    /// Size in bytes.
    pub(super) bytes: usize,
    /// Reusable `Vec<u32>` for positions (avoids per-chunk heap allocation).
    pub(super) positions: Vec<u32>,
    pub(super) positions_h: Vec<u32>,
    pub(super) positions_w: Vec<u32>,
    /// Reusable `Vec<i64>` for slot mappings (avoids per-chunk heap allocation).
    pub(super) slots: Vec<i64>,
}

impl PinnedMetaStaging {
    /// The ONLY way to write this buffer: a bounds-checked cursor. See
    /// [`crate::model::pinned_pack`] for why the rule lives there and not in
    /// each of the five call sites that pack it.
    ///
    /// `dest_bytes` is how much room the DEVICE destination has, and it is
    /// required rather than defaulted because it is the bound that was missing.
    /// `bytes` here equals `sizes.scratch` exactly (`impl_a1.rs` allocates
    /// `scratch.max(64 KiB)` and `sizes.rs` already floors scratch at 64 KiB),
    /// but every one of these packs is uploaded to `scratch().offset(k)` for
    /// some non-zero `k`. So a pack that fits the HOST staging buffer can still
    /// run `k` bytes off the end of the DEVICE allocation, and checking only
    /// `cursor <= stg.bytes` — which is all the old code did — never sees it.
    /// The packer's capacity is the smaller of the two ends.
    ///
    /// Takes `&self` rather than `&mut self` on purpose — the bytes it writes
    /// are the separate `cuMemAllocHost` region `ptr` refers to, not this
    /// struct, so a shared borrow is enough and callers can still read the
    /// reusable source `Vec`s alongside it.
    pub(crate) fn packer_for(
        &self,
        dest_bytes: usize,
    ) -> crate::model::pinned_pack::PinnedPacker<'_> {
        // SAFETY: `ptr`/`bytes` are the `alloc_host_pinned` region installed in
        // `impl_a1.rs` and released in `drop.rs`; it is live for the model's
        // lifetime, zeroed at allocation (the trait's contract), and only ever
        // touched from the single scheduler thread — the same invariant that
        // `unsafe impl Sync for TransformerModel` above rests on. The capacity
        // handed over is `min(host room, device room)`, never more than the
        // allocation.
        unsafe {
            crate::model::pinned_pack::PinnedPacker::new(self.ptr, self.bytes.min(dest_bytes))
        }
    }
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

/// Release every pool this model owns, newest first.
///
/// Construction order is buffers → kv cache → ssm pools → derived, so release
/// runs the reverse. `Teardown` is used rather than a hand-rolled sequence
/// because it attempts every resource even after one fails: a half-torn-down
/// GPU is worse than a reported error.
///
/// NOT released here: the weights. `build_model` takes `store: &WeightStore`
/// and the layers only copy pointers out of it, so this model does not own
/// them — the host that retained the store releases it after this returns.
impl TransformerModel {
    /// Hand the model the ledger of its own weights, for teardown.
    pub fn adopt_weight_store(&mut self, store: spark_runtime::weights::WeightStore) {
        self.weight_store = Some(store);
    }

    pub(super) fn release_pools(&mut self) -> anyhow::Result<()> {
        use atlas_core::scope::ModelResource;

        let gpu: &dyn GpuBackend = self.gpu.as_ref();
        let mut first_error: Option<anyhow::Error> = None;
        let mut attempt = |label: &'static str, r: anyhow::Result<()>| {
            if let Err(e) = r
                && first_error.is_none()
            {
                first_error = Some(e.context(label));
            }
        };

        attempt("derived weights", self.derived.release(gpu));
        attempt("ssm snapshots", self.ssm_snapshots.release(gpu));
        // The pool is Arc'd because slots are handed out to sequences. A live
        // clone here means something still holds a slot, which is a drain bug,
        // not a teardown one — so it is reported rather than forced.
        match std::sync::Arc::get_mut(&mut self.ssm_pool) {
            Some(pool) => attempt("ssm state pool", pool.release(gpu)),
            None => attempt(
                "ssm state pool",
                Err(anyhow::anyhow!(
                    "{} handle(s) still hold the SSM pool — a sequence was not \
                     released before teardown",
                    std::sync::Arc::strong_count(&self.ssm_pool) - 1
                )),
            ),
        }
        attempt("kv cache", self.kv_cache.lock().release(gpu));
        attempt("buffer arena", self.buffers.release(gpu));
        // Weights LAST: the layers hold pointers into them, so they must not be
        // freed until everything that reads them is gone.
        if let Some(mut store) = self.weight_store.take() {
            attempt("weight store", store.release(gpu));
        }
        // LAST: whatever the owners above did not cover. Chiefly the loaders'
        // fused weights, which live in layer structs and belong to no pool.
        // Every pointer freed above has already left the ledger, so this
        // cannot double-free — it only ever sees what was missed.
        let swept = gpu.sweep_unreleased();
        if swept > 0 {
            tracing::warn!(
                "teardown swept {swept} allocation(s) that no ModelResource \
                 released — they are reclaimed, but each one is memory whose \
                 owner is unaccounted for"
            );
        }

        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
