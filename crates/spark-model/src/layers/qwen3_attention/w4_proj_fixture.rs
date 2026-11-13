// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: mock-backend fixtures for the packed (W4) attention dispatch tests (port lane
//! L22b): a small plain-BF16 attention layer, its forward context, recording fakes of the four
//! projections (`port::w4::fake`), and a CPU oracle of what a computing fake writes.

use std::sync::Arc;

use avarok_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;

use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::FfnComponent;
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::layers::qwen3_attention::Qwen3AttentionLayer;
use crate::port::w4::W4Box;
use crate::port::w4::fake::{Call, FakeW4, Log, log};
use crate::weight_map::{AttentionWeights, DenseWeight, QuantizedWeight};

/// Hidden width, heads and head dim of the fixture layer (tiny, so computing fakes are cheap).
pub(crate) const H: usize = 128;
pub(crate) const NQ: usize = 2;
pub(crate) const NKV: usize = 1;
pub(crate) const HD: usize = 64;
/// Token capacity of the fixture's buffer arena.
pub(crate) const MAX_TOKENS: usize = 160;

pub(crate) const fn q_dim() -> usize {
    NQ * HD
}

pub(crate) const fn q_proj_dim(gated: bool) -> usize {
    if gated { 2 * q_dim() } else { q_dim() }
}

pub(crate) const fn kv_dim() -> usize {
    NKV * HD
}

/// Per-sequence `[Q | K | V]` stride in bytes of the batched decode buffer.
pub(crate) const fn per_seq_qkv(gated: bool) -> usize {
    (q_proj_dim(gated) + 2 * kv_dim()) * 2
}

/// Everything a forward context borrows.
pub(crate) struct Rig {
    pub gpu: MockGpuBackend,
    pub config: ModelConfig,
    pub buffers: BufferArena,
    pub dispatch: GemmDispatch,
    pub derived: DerivedWeights,
    pub levers: ModelLevers,
    pub stats: ModelStats,
}

impl Rig {
    pub(crate) fn new(gated: bool) -> Self {
        let gpu = MockGpuBackend::new();
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.hidden_size = H;
        config.intermediate_size = 128;
        config.num_attention_heads = NQ;
        config.num_key_value_heads = NKV;
        config.head_dim = HD;
        config.num_experts = 1;
        config.num_experts_per_tok = 1;
        config.moe_intermediate_size = 128;
        config.vocab_size = 128;
        config.attn_gated = gated;
        let buffers = BufferArena::new(&config, MAX_TOKENS, 256, 16, 8, &gpu).unwrap();
        Self {
            gpu,
            config,
            buffers,
            dispatch: GemmDispatch::defaults(),
            derived: DerivedWeights::new(),
            levers: ModelLevers::defaults(),
            stats: ModelStats::new(),
        }
    }

    pub(crate) fn ctx(&self) -> ForwardContext<'_> {
        ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: &self.gpu,
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            decode_step: false,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: MoeLoraRoute::Fold,
        }
    }

    fn dense(&self, n: usize, k: usize) -> DenseWeight {
        DenseWeight {
            weight: self.gpu.alloc(n * k * 2).unwrap(),
        }
    }

    /// A plain-BF16 attention layer: BF16 q/k/v in `attn`, the BF16 O-projection installed, no
    /// quantized sidecar (what the loader builds on a `Bf16Raw` checkpoint).
    pub(crate) fn bf16_layer(&self, gated: bool) -> Qwen3AttentionLayer {
        let norm = self.dense(1, H);
        let attn = AttentionWeights {
            q_proj: self.dense(q_proj_dim(gated), H),
            k_proj: self.dense(kv_dim(), H),
            v_proj: self.dense(kv_dim(), H),
            o_proj: QuantizedWeight::null(),
            q_norm: self.dense(1, HD),
            k_norm: self.dense(1, HD),
            q_norm_full: None,
            k_norm_full: None,
            k_scale: 1.0,
            v_scale: 1.0,
        };
        let mut layer = Qwen3AttentionLayer::new(
            norm,
            attn,
            norm,
            FfnComponent::None,
            0,
            None,
            None,
            None,
            &self.gpu,
            KvCacheDtype::Bf16,
            0,
            &self.config,
        )
        .unwrap();
        layer.gated = gated;
        // `new` may resolve FP8/NVFP4 helpers; the BF16 route carries none of them.
        layer.q_fp8 = None;
        layer.k_fp8 = None;
        layer.v_fp8 = None;
        layer.o_fp8 = None;
        layer.set_o_dense_bf16(self.dense(H, q_dim()));
        layer
    }

    /// Fill `[rows, k]` BF16 at `x` with small exact values; returns them as f32.
    pub(crate) fn fill_x(&self, x: DevicePtr, rows: usize, k: usize, salt: usize) -> Vec<f32> {
        let vals: Vec<f32> = (0..rows * k)
            .map(|i| (((i * 7 + salt * 3) % 9) as f32 - 4.0) / 4.0)
            .collect();
        let bytes: Vec<u8> = vals
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        self.gpu.copy_h2d(&bytes, x).unwrap();
        vals
    }

    pub(crate) fn read(&self, p: DevicePtr, bytes: usize) -> Vec<u8> {
        let mut out = vec![0u8; bytes];
        self.gpu.copy_d2h(p, &mut out).unwrap();
        out
    }
}

/// The four recording fakes of one layer and their shared log.
pub(crate) struct Fakes {
    pub log: Log,
    pub q: Arc<FakeW4>,
    pub k: Arc<FakeW4>,
    pub v: Arc<FakeW4>,
    pub o: Arc<FakeW4>,
}

/// Weight seeds of the computing fakes (q, k, v, o).
pub(crate) const SEEDS: [u32; 4] = [11, 23, 37, 41];

impl Fakes {
    /// Fakes shaped for the fixture layer with decode window `max_rows`; `computing` makes them
    /// write real outputs on the mock memory.
    pub(crate) fn new(gated: bool, max_rows: usize, computing: bool) -> Self {
        let log = log();
        let mk = |name, n, k, seed| {
            let f = FakeW4::new(name, n, k, max_rows, &log);
            Arc::new(if computing { f.computing(seed) } else { f })
        };
        Self {
            q: mk("q", q_proj_dim(gated), H, SEEDS[0]),
            k: mk("k", kv_dim(), H, SEEDS[1]),
            v: mk("v", kv_dim(), H, SEEDS[2]),
            o: mk("o", H, q_dim(), SEEDS[3]),
            log,
        }
    }

    pub(crate) fn install(&self, layer: &mut Qwen3AttentionLayer) -> usize {
        let b = |f: &Arc<FakeW4>| -> W4Box { f.clone() };
        layer.set_w4_proj(b(&self.q), b(&self.k), b(&self.v), b(&self.o))
    }

    pub(crate) fn calls(&self) -> Vec<(&'static str, Call)> {
        self.log.lock().clone()
    }
}

/// What a computing fake's decode writes for one input row: ascending-order f32 sums, BF16.
pub(crate) fn oracle_row(seed: u32, n: usize, x: &[f32]) -> Vec<u8> {
    (0..n)
        .flat_map(|j| {
            let mut acc = 0.0f32;
            for (i, &xv) in x.iter().enumerate() {
                acc += FakeW4::w(seed, j, i) * xv;
            }
            half::bf16::from_f32(acc).to_le_bytes()
        })
        .collect()
}
