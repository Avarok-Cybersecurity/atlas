// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! DeepSeek-V4.1 Flash: one transformer block on the generic model.
//!
//! The generic `TransformerModel` owns the embedding, the FP32 mHC highway
//! (`BufferArena::hc_streams`, `[T, hc, H]`), the final norm and the LM head;
//! each layer here does everything in between, in the reference's order
//! (`deepseek_v41_ref::model::forward`, 39/39 against DeepSeek's golden):
//!
//! ```text
//! layer 0 only:        hc_expand(embedding -> streams)
//! engram layers only:  streams += engram(token n-grams)          (EngramV41)
//! attention:           (pre_a, post_a, comb_a) = hc_mixes(streams, attn site)
//!                      x = rmsnorm(collapse(streams, pre_prev))   <- DELAYED pre
//!                      streams = hc_post(attn(x), streams, post_a, comb_a)
//! ffn:                 (pre_f, post_f, comb_f) = hc_mixes(streams, ffn site)
//!                      x = rmsnorm(collapse(streams, pre_a))
//!                      streams = hc_post(moe(x), streams, post_f, comb_f)
//!                      pre_prev = pre_f
//! last layer only:     hidden = collapse(streams, pre_prev)      (no learned head)
//! ```
//!
//! `pre_prev` starts as the one-hot on stream 0 (the reference's initial
//! pre-mix) and travels with the sequence through [`V41Runtime`]. So do the
//! shared attention slots, the engram hasher and the expert cache: V4.1's
//! attention is shared ACROSS layers (four kv sources write, forty read), which
//! no per-layer state can express, so one runtime object is shared by all
//! forty layers through an `Arc` and guarded by mutexes. One sequence at a
//! time; multi-sequence decode and CUDA-graph capture are declined through the
//! layer hooks.
//!
//! Routed experts never sit in HBM: the cache is a page-locked, device-visible
//! arena the GPU reads in place (`ExpertLru` + `ExpertSliceMap`), filled by
//! pread from the seven shards; the engram tables are read by row (`EngramRowReader`).

use std::sync::{Arc, Mutex};

use spark_runtime::gpu::{DevicePtr, KernelHandle};
use spark_runtime::weights::expert_stream::{
    EngramRowReader, ExpertLru, ExpertSliceMap, PinnedArena,
};

use crate::layer::LayerState;
use crate::layers::attn_v41::{
    AttnV41, AttnV41Cfg, AttnV41LayerState, AttnV41LayerWeights, LayerRole, SharedV41,
};
use crate::layers::engram_v41::{EngramHashTables, EngramHasher, EngramV41};
use crate::layers::moe_v41::{MoeV41, MoeV41Cfg, MoeV41LayerWeights};
use crate::layers::qwen3_attention::HcSiteWeights;
use crate::weight_map::DenseWeight;

mod step;
mod trait_impl;

/// Everything the forty layers share for the ONE sequence in flight.
pub struct V41Runtime {
    pub attn_cfg: AttnV41Cfg,
    pub moe_cfg: MoeV41Cfg,
    pub attn: Mutex<AttnV41>,
    pub moe: Mutex<MoeV41>,
    pub engram: Mutex<EngramV41>,
    pub lru: Mutex<ExpertLru>,
    /// Kept alive for the cache's lifetime; freed with the runtime.
    pub arena: PinnedArena,
    pub slices: ExpertSliceMap,
    pub rows: EngramRowReader,
    pub tables: Arc<EngramHashTables>,
    pub hasher: Mutex<EngramHasher>,
    pub shared: Mutex<SharedV41>,
    /// The step's engram hashes `[tokens, n_engram_layers, cols]`, computed by
    /// the first engram layer and reused by the second.
    pub step_hashes: Mutex<Option<Vec<i64>>>,
    /// The delayed `pre` mix `[max_tokens, hc]` f32 on the device.
    pub pre_prev: DevicePtr,
    /// `[max_tokens, (2 + hc) * hc]` f32 scratch for the hc mixes (dot -> finish)
    pub mixes_s: DevicePtr,
    pub reader_threads: usize,
    pub n_layers: usize,
    pub hc_mult: usize,
    pub hidden: usize,
    pub sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub norm_eps: f32,
    // per-step scratch, guarded by the same one-sequence contract
    pub pre_a: DevicePtr,
    pub pre_f: DevicePtr,
    pub post_s: DevicePtr,
    pub comb_s: DevicePtr,
    pub attn_in: DevicePtr,
    pub max_tokens: usize,
    /// per-step totals across layers, printed by the last layer when ATLAS_DS41_DIAG=1
    pub step_moe: Mutex<crate::layers::moe_v41::MoeV41Timing>,
    pub step_attn_ms: Mutex<f64>,
    pub step_engram_ms: Mutex<f64>,
    pub step_start: Mutex<Option<std::time::Instant>>,
}

// SAFETY: every raw device/host pointer here names memory the runtime owns
// for its whole life; access is serialised by the mutexes and the
// one-sequence contract.
unsafe impl Send for V41Runtime {}
unsafe impl Sync for V41Runtime {}

pub struct V41LayerState {
    pub attn: AttnV41LayerState,
}

impl LayerState for V41LayerState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

pub struct DeepSeekV41Layer {
    pub idx: usize,
    pub role: LayerRole,
    pub rt: Arc<V41Runtime>,
    pub attn_w: AttnV41LayerWeights,
    pub moe_w: MoeV41LayerWeights,
    /// `Some(index into engram_layer_ids)` on engram layers.
    pub engram_index: Option<usize>,
    pub hc_attn: HcSiteWeights,
    pub hc_ffn: HcSiteWeights,
    pub attn_norm: DenseWeight,
    pub ffn_norm: DenseWeight,
    pub k_hc_expand: KernelHandle,
    pub k_hc_post: KernelHandle,
    pub k_mixes_dot: KernelHandle,
    pub k_mixes_finish: KernelHandle,
    pub k_collapse: KernelHandle,
    pub k_rms_norm: KernelHandle,
}
