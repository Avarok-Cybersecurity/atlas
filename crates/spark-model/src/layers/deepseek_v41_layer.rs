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
//! time; multi-sequence decode and the MODEL-level CUDA graph are declined
//! through the layer hooks. The layer captures its own single-token step
//! instead (`ATLAS_DS41_GRAPH=1`, see [`GraphMode`]): the host work in the
//! middle of every layer (the routing download and the expert fetch; on the
//! twelve kv/index source layers also the compressor group and the index
//! top-k) splits the step into graph segments with the host spans between.
//!
//! Routed experts never sit in HBM: the cache is a page-locked, device-visible
//! arena the GPU reads in place (`ExpertLru` + `ExpertSliceMap`), filled by
//! pread from the seven shards; the engram tables are read by row (`EngramRowReader`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::weights::expert_stream::{
    EngramRowReader, ExpertLru, ExpertSliceMap, PinnedArena,
};

use crate::layer::{ForwardContext, LayerState};
use crate::layers::attn_v41::{
    AttnV41, AttnV41Cfg, AttnV41LayerState, AttnV41LayerWeights, LayerRole, SharedV41,
};
use crate::layers::engram_v41::{ENGRAM_ROW_BYTES, EngramHashTables, EngramHasher, EngramV41};
use crate::layers::moe_v41::{MoeV41, MoeV41Cfg, MoeV41LayerWeights};
use crate::layers::ops;
use crate::layers::qwen3_attention::HcSiteWeights;
use crate::weight_map::DenseWeight;

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
    /// Set when a capture failed: every segment from then on runs eagerly
    /// (graphs already captured keep replaying; they are valid).
    pub graph_disabled: AtomicBool,
}

/// How the single-token step runs. `ATLAS_DS41_GRAPH_ORACLE=1` runs every
/// layer BOTH ways and compares the highway bit for bit; `ATLAS_DS41_GRAPH=1`
/// replays the captured segments; anything else is the eager step. Read once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphMode {
    Off,
    On,
    Oracle,
}

pub fn graph_mode() -> GraphMode {
    static MODE: OnceLock<GraphMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let is_one = |k: &str| std::env::var(k).is_ok_and(|v| v == "1");
        if is_one("ATLAS_DS41_GRAPH_ORACLE") {
            GraphMode::Oracle
        } else if is_one("ATLAS_DS41_GRAPH") {
            GraphMode::On
        } else {
            GraphMode::Off
        }
    })
}

/// The pointers a layer's captured segments bake that are not the runtime's
/// own (those live as long as the runtime). Checked on every replay: a
/// different set means the graphs describe other buffers and are recaptured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Baked {
    hidden: DevicePtr,
    streams: DevicePtr,
    normed: DevicePtr,
    window: DevicePtr,
    /// the kv source's latent cache the captured `sparse_attn` reads
    rows_b: Option<DevicePtr>,
    attn_out: DevicePtr,
    moe_out: DevicePtr,
}

/// One layer's captured single-token step, per sequence (it bakes the
/// sequence's window ring and the kv source's cache).
///
/// Segment A = the attention site's mixes, collapse and norm, the attention
/// (capturable layers only), `hc_post`, the ffn site's mixes, collapse and
/// norm, the router GEMV. On the twelve kv/index source layers the attention
/// runs eagerly between `a[0]` (through the norm) and `a[1]` (from `hc_post`).
/// Host span: the routing download, the expert fetch, the plan uploads.
/// Segment B = the expert compute, `hc_post`, the delayed-mix copy, and on
/// the last layer the final collapse.
struct LayerGraphs {
    a: [Option<GraphHandle>; 2],
    b: Option<GraphHandle>,
    baked: Baked,
}

impl LayerGraphs {
    fn destroy(self, gpu: &dyn GpuBackend) -> Result<()> {
        for g in self.a.into_iter().chain([self.b]).flatten() {
            gpu.destroy_graph(g)?;
        }
        Ok(())
    }
}

// SAFETY: every raw device/host pointer here names memory the runtime owns
// for its whole life; access is serialised by the mutexes and the
// one-sequence contract.
unsafe impl Send for V41Runtime {}
unsafe impl Sync for V41Runtime {}

pub struct V41LayerState {
    pub attn: AttnV41LayerState,
    graphs: Option<LayerGraphs>,
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

fn diag_on() -> bool {
    std::env::var("ATLAS_DS41_DIAG").is_ok_and(|v| v == "1")
}

/// RMS of the first `n` f32 values at `p` (diagnostics only, synchronises).
fn diag_rms_f32(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> f32 {
    let mut b = vec![0u8; n * 4];
    if gpu.copy_d2h(p, &mut b).is_err() {
        return f32::NAN;
    }
    let v: Vec<f32> = b
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (v.iter().map(|x| x * x).sum::<f32>() / n.max(1) as f32).sqrt()
}

fn diag_rms_bf16(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> f32 {
    let mut b = vec![0u8; n * 2];
    if gpu.copy_d2h(p, &mut b).is_err() {
        return f32::NAN;
    }
    let v: Vec<f32> = b
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect();
    (v.iter().map(|x| x * x).sum::<f32>() / n.max(1) as f32).sqrt()
}

impl DeepSeekV41Layer {
    fn mixes(
        &self,
        gpu: &dyn GpuBackend,
        site: &HcSiteWeights,
        streams: DevicePtr,
        pre: DevicePtr,
        post: DevicePtr,
        comb: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let mix_hc = (2 + rt.hc_mult) * rt.hc_mult;
        // one block per (token, mix) for the 24 dot products over hc * H, then
        // the tiny epilogue; bit-identical to the one-block hc_v41_mixes
        KernelLaunch::new(gpu, self.k_mixes_dot)
            .grid([m as u32, mix_hc as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(streams)
            .arg_ptr(site.hc_fn)
            .arg_ptr(rt.mixes_s)
            .arg_u32(rt.hidden as u32)
            .arg_u32(rt.hc_mult as u32)
            .arg_f32(rt.norm_eps)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.k_mixes_finish)
            .grid([m as u32, 1, 1])
            .block([32, 1, 1])
            .arg_ptr(rt.mixes_s)
            .arg_ptr(site.hc_scale)
            .arg_ptr(site.hc_base)
            .arg_ptr(pre)
            .arg_ptr(post)
            .arg_ptr(comb)
            .arg_u32(rt.hc_mult as u32)
            .arg_u32(rt.sinkhorn_iters as u32)
            .arg_f32(rt.hc_eps)
            .launch(stream)
    }

    fn collapse(
        &self,
        gpu: &dyn GpuBackend,
        streams: DevicePtr,
        pre: DevicePtr,
        y: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_collapse)
            .grid([m as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(streams)
            .arg_ptr(pre)
            .arg_ptr(y)
            .arg_u32(self.rt.hidden as u32)
            .arg_u32(self.rt.hc_mult as u32)
            .launch(stream)
    }

    fn hc_post(
        &self,
        gpu: &dyn GpuBackend,
        block_out: DevicePtr,
        streams: DevicePtr,
        post: DevicePtr,
        comb: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        ops::hc_post(
            gpu,
            self.k_hc_post,
            block_out,
            streams,
            post,
            comb,
            streams,
            m as u32,
            self.rt.hidden as u32,
            self.rt.hc_mult as u32,
            stream,
        )
    }

    /// The token ids of this step, for the engram hash.
    fn step_token_ids(&self, ctx: &ForwardContext, m: usize) -> Result<Vec<u32>> {
        if let Some(ids) = ctx.host_token_ids {
            ensure!(
                ids.len() >= m,
                "host token ids: {} for {m} tokens",
                ids.len()
            );
            return Ok(ids[..m].to_vec());
        }
        let dev = ctx
            .token_ids
            .context("engram needs the step's token ids (none in the context)")?;
        let mut bytes = vec![0u8; m * 4];
        ctx.gpu.copy_d2h(dev, &mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    fn engram(
        &self,
        hi: usize,
        streams: DevicePtr,
        m: usize,
        start_pos: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let gpu = ctx.gpu;
        let hashes = {
            let mut cache = rt.step_hashes.lock().unwrap();
            if hi == 0 || cache.is_none() {
                let ids = self.step_token_ids(ctx, m)?;
                let mut hasher = rt.hasher.lock().unwrap();
                let h = hasher.hash(&ids, start_pos)?;
                *cache = Some(h.clone());
                h
            } else {
                cache.clone().unwrap()
            }
        };
        let hasher = rt.hasher.lock().unwrap();
        let row_ids = hasher.layer_row_ids(&hashes, m, hi);
        drop(hasher);
        let mut raw = vec![0u8; row_ids.len() * ENGRAM_ROW_BYTES];
        rt.rows.read_rows(self.idx, &row_ids, &mut raw)?;
        let engram = rt.engram.lock().unwrap();
        engram.rows_from_q2k(gpu, &raw, row_ids.len(), stream)?;
        engram.apply(gpu, self.idx, streams, m, stream)
    }

    /// One block for `m` tokens at `start_pos`, on the highway.
    fn step(
        &self,
        hidden: DevicePtr,
        m: usize,
        start_pos: usize,
        state: &mut dyn LayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let gpu = ctx.gpu;
        ensure!(
            m >= 1 && m <= rt.max_tokens,
            "deepseek-v4.1: {m} tokens exceeds the {}-token workspace",
            rt.max_tokens
        );
        ensure!(
            start_pos == 0 || m == 1,
            "deepseek-v4.1: chunked prefill is not supported (start {start_pos}, {m} tokens); raise --max-prefill-tokens"
        );
        let st = state
            .as_any_mut()
            .downcast_mut::<V41LayerState>()
            .context("deepseek-v4.1 layer given a foreign state")?;
        let streams = ctx.buffers.hc_streams();
        let (h, hc) = (rt.hidden, rt.hc_mult);

        if self.idx == 0 {
            if start_pos == 0 {
                // a new sequence: fresh shared slots, hasher, delayed mix
                *rt.shared.lock().unwrap() = SharedV41::default();
                rt.hasher.lock().unwrap().reset();
                *rt.step_hashes.lock().unwrap() = None;
            }
            // a new step: the captured step's device-side position and
            // selection are stale
            rt.attn.lock().unwrap().invalidate_decode_uploads();
            // the initial pre-mix is one-hot on stream 0
            let mut onehot = vec![0u8; m * hc * 4];
            for t in 0..m {
                onehot[t * hc * 4..t * hc * 4 + 4].copy_from_slice(&1f32.to_le_bytes());
            }
            gpu.copy_h2d(&onehot, rt.pre_prev)?;
            ops::hc_expand(
                gpu,
                self.k_hc_expand,
                hidden,
                streams,
                m as u32,
                h as u32,
                hc as u32,
                stream,
            )?;
        }
        if self.idx == 0 {
            *rt.step_start.lock().unwrap() = Some(std::time::Instant::now());
            *rt.step_moe.lock().unwrap() = Default::default();
            *rt.step_attn_ms.lock().unwrap() = 0.0;
            *rt.step_engram_ms.lock().unwrap() = 0.0;
        }
        let te = std::time::Instant::now();
        if let Some(hi) = self.engram_index
            && !std::env::var("ATLAS_DS41_NO_ENGRAM").is_ok_and(|v| v == "1")
        {
            self.engram(hi, streams, m, start_pos, ctx, stream)?;
        }
        *rt.step_engram_ms.lock().unwrap() += te.elapsed().as_secs_f64() * 1e3;
        let diag = diag_on();
        if diag {
            gpu.synchronize(stream)?;
            tracing::info!(
                "DS41 L{} in: streams rms {:.4} (token 0)",
                self.idx,
                diag_rms_f32(gpu, streams, hc * h)
            );
        }

        // the single-token step at a position > 0 is the captured one; prefill
        // (m > 1, or the one-token prompt at position 0) stays eager
        let mode = graph_mode();
        let graph = mode != GraphMode::Off && m == 1 && start_pos > 0 && !gpu.debug_sync_kernels();
        match (graph, mode) {
            (false, _) => self.step_eager(hidden, m, start_pos, st, ctx, stream),
            (true, GraphMode::Oracle) => self.step_oracle(hidden, start_pos, st, ctx, stream),
            (true, _) => self.step_graph(hidden, start_pos, st, ctx, stream),
        }
    }

    /// The eager step after the engram: every launch issued from the host,
    /// the attention and the MoE with their host work inline. This is the
    /// path `ATLAS_DS41_GRAPH` unset (or `0`) takes, and prefill always.
    fn step_eager(
        &self,
        hidden: DevicePtr,
        m: usize,
        start_pos: usize,
        st: &mut V41LayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let gpu = ctx.gpu;
        let streams = ctx.buffers.hc_streams();
        let (h, hc) = (rt.hidden, rt.hc_mult);
        let diag = diag_on();

        // attention
        self.mixes(
            gpu,
            &self.hc_attn,
            streams,
            rt.pre_a,
            rt.post_s,
            rt.comb_s,
            m,
            stream,
        )?;
        self.collapse(gpu, streams, rt.pre_prev, hidden, m, stream)?;
        let normed = ctx.buffers.norm_output();
        ops::rms_norm(
            gpu,
            self.k_rms_norm,
            hidden,
            &self.attn_norm,
            normed,
            m as u32,
            h as u32,
            rt.norm_eps,
            stream,
        )?;
        let ta = std::time::Instant::now();
        let attn_out = {
            let attn = rt.attn.lock().unwrap();
            let mut shared = rt.shared.lock().unwrap();
            let run = attn.forward(
                gpu,
                &self.attn_w,
                &mut st.attn,
                &mut shared,
                normed,
                m,
                start_pos,
                stream,
            )?;
            run.out
        };
        *rt.step_attn_ms.lock().unwrap() += ta.elapsed().as_secs_f64() * 1e3;
        if diag {
            gpu.synchronize(stream)?;
            tracing::info!(
                "DS41 L{} attn: in rms {:.4} normed rms {:.4} out rms {:.4}",
                self.idx,
                diag_rms_bf16(gpu, hidden, h),
                diag_rms_bf16(gpu, normed, h),
                diag_rms_bf16(gpu, attn_out, h)
            );
        }
        self.hc_post(gpu, attn_out, streams, rt.post_s, rt.comb_s, m, stream)?;

        // ffn
        self.mixes(
            gpu,
            &self.hc_ffn,
            streams,
            rt.pre_f,
            rt.post_s,
            rt.comb_s,
            m,
            stream,
        )?;
        self.collapse(gpu, streams, rt.pre_a, hidden, m, stream)?;
        ops::rms_norm(
            gpu,
            self.k_rms_norm,
            hidden,
            &self.ffn_norm,
            normed,
            m as u32,
            h as u32,
            rt.norm_eps,
            stream,
        )?;
        let moe_out = {
            let moe = rt.moe.lock().unwrap();
            let mut lru = rt.lru.lock().unwrap();
            let (out, _w, _i) = moe.forward(
                gpu,
                &self.moe_w,
                &mut lru,
                &rt.slices,
                normed,
                m,
                rt.reader_threads,
                stream,
            )?;
            rt.step_moe.lock().unwrap().add(&moe.last.get());
            out
        };
        if diag {
            gpu.synchronize(stream)?;
            tracing::info!(
                "DS41 L{} ffn: in rms {:.4} normed rms {:.4} out rms {:.4}",
                self.idx,
                diag_rms_bf16(gpu, hidden, h),
                diag_rms_bf16(gpu, normed, h),
                diag_rms_bf16(gpu, moe_out, h)
            );
        }
        self.hc_post(gpu, moe_out, streams, rt.post_s, rt.comb_s, m, stream)?;
        gpu.copy_d2d_async(rt.pre_f, rt.pre_prev, m * hc * 4, stream)?;

        if self.idx + 1 == rt.n_layers {
            self.step_line(m, start_pos, "eager");
            // no learned head on V4.1: the final collapse uses the last ffn pre
            self.collapse(gpu, streams, rt.pre_prev, hidden, m, stream)?;
            gpu.synchronize(stream)?;
        }
        Ok(())
    }

    /// Segment A's opening: the attention site's mixes, the delayed-pre
    /// collapse and the attention norm into `normed`. Device work only.
    fn seg_attn_in(
        &self,
        gpu: &dyn GpuBackend,
        streams: DevicePtr,
        hidden: DevicePtr,
        normed: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        self.mixes(
            gpu,
            &self.hc_attn,
            streams,
            rt.pre_a,
            rt.post_s,
            rt.comb_s,
            1,
            stream,
        )?;
        self.collapse(gpu, streams, rt.pre_prev, hidden, 1, stream)?;
        ops::rms_norm(
            gpu,
            self.k_rms_norm,
            hidden,
            &self.attn_norm,
            normed,
            1,
            rt.hidden as u32,
            rt.norm_eps,
            stream,
        )
    }

    /// Segment A's close: `hc_post` of the attention, the ffn site's mixes,
    /// the collapse, the ffn norm into `normed`, the router GEMV into the MoE's
    /// logits. Device work only.
    fn seg_ffn_in(
        &self,
        gpu: &dyn GpuBackend,
        moe: &MoeV41,
        attn_out: DevicePtr,
        streams: DevicePtr,
        hidden: DevicePtr,
        normed: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        self.hc_post(gpu, attn_out, streams, rt.post_s, rt.comb_s, 1, stream)?;
        self.mixes(
            gpu,
            &self.hc_ffn,
            streams,
            rt.pre_f,
            rt.post_s,
            rt.comb_s,
            1,
            stream,
        )?;
        self.collapse(gpu, streams, rt.pre_a, hidden, 1, stream)?;
        ops::rms_norm(
            gpu,
            self.k_rms_norm,
            hidden,
            &self.ffn_norm,
            normed,
            1,
            rt.hidden as u32,
            rt.norm_eps,
            stream,
        )?;
        moe.route_launch(gpu, &self.moe_w, normed, 1, stream)
    }

    /// Segment B: the expert compute from the staged plan, `hc_post`, the
    /// delayed-mix copy, and on the last layer the final collapse. Device
    /// work only.
    fn seg_ffn_out(
        &self,
        gpu: &dyn GpuBackend,
        moe: &MoeV41,
        ne: usize,
        streams: DevicePtr,
        hidden: DevicePtr,
        normed: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let moe_out = moe.compute_m1(gpu, &self.moe_w, normed, ne, stream)?;
        self.hc_post(gpu, moe_out, streams, rt.post_s, rt.comb_s, 1, stream)?;
        gpu.copy_d2d_async(rt.pre_f, rt.pre_prev, rt.hc_mult * 4, stream)?;
        if self.idx + 1 == rt.n_layers {
            // no learned head on V4.1: the final collapse uses the last ffn pre
            self.collapse(gpu, streams, rt.pre_prev, hidden, 1, stream)?;
        }
        Ok(())
    }

    /// Replay `slot`'s graph, or capture `body` into it (and run it) on the
    /// first pass. A capture that cannot begin or end runs `body` eagerly and
    /// disables further captures; an error INSIDE the body ends the capture
    /// (so the stream is usable again) and propagates.
    fn run_segment(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
        slot: &mut Option<GraphHandle>,
        body: impl Fn() -> Result<()>,
    ) -> Result<()> {
        let rt = &self.rt;
        if let Some(g) = slot {
            return gpu.launch_graph(*g, stream);
        }
        if rt.graph_disabled.load(Ordering::Relaxed) {
            return body();
        }
        if let Err(e) = gpu.begin_capture(stream) {
            tracing::warn!(
                "DS41 L{}: CUDA graph begin_capture failed ({e:#}); running eagerly and disabling capture",
                self.idx
            );
            rt.graph_disabled.store(true, Ordering::Relaxed);
            return body();
        }
        if let Err(e) = body() {
            gpu.abort_capture_if_active(stream);
            let msg = format!("{e:#}");
            let poison = msg.contains("status 900")
                || msg.contains("status 901")
                || msg.contains("STREAM_CAPTURE");
            if !poison {
                return Err(e.context(format!("DS41 L{} under CUDA graph capture", self.idx)));
            }
            // a capture RECORDS: nothing ran yet, so the eager body is the step
            tracing::warn!(
                "DS41 L{}: segment failed under capture ({msg}); running eagerly and disabling capture",
                self.idx
            );
            rt.graph_disabled.store(true, Ordering::Relaxed);
            return body();
        }
        match gpu.end_capture(stream) {
            Ok(g) => {
                *slot = Some(g);
                gpu.launch_graph(g, stream)
            }
            Err(e) => {
                tracing::warn!(
                    "DS41 L{}: CUDA graph end_capture failed ({e:#}); running eagerly and disabling capture",
                    self.idx
                );
                rt.graph_disabled.store(true, Ordering::Relaxed);
                body()
            }
        }
    }

    /// The captured single-token step: segment A (replayed or captured),
    /// the host span (routing download, expert fetch, plan uploads), segment
    /// B. Bit for bit the eager step: the kernels and their arguments are the
    /// same, the per-token inputs (position, selection, expert pointers,
    /// routing weights) are read from device buffers the host refills before
    /// each replay, and `sparse_attn` runs at a fixed, -1-padded `topk`.
    fn step_graph(
        &self,
        hidden: DevicePtr,
        start_pos: usize,
        st: &mut V41LayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let gpu = ctx.gpu;
        let streams = ctx.buffers.hc_streams();
        let normed = ctx.buffers.norm_output();
        let h = rt.hidden;
        let diag = diag_on();
        let V41LayerState {
            attn: st_attn,
            graphs,
        } = st;

        let ta = std::time::Instant::now();
        let mut attn = rt.attn.lock().unwrap();
        let mut shared = rt.shared.lock().unwrap();
        let moe = rt.moe.lock().unwrap();
        let capturable = attn.decode_capturable(&self.attn_w);
        // the host half of the attention: position and padded selection onto
        // the device (only what changed since the last upload)
        let rows_b = if capturable {
            attn.decode_prep(&self.attn_w, &shared, gpu, start_pos, stream)?
        } else {
            None
        };
        let baked = Baked {
            hidden,
            streams,
            normed,
            window: st_attn.window(),
            rows_b,
            attn_out: attn.out_ptr(),
            moe_out: moe.out_ptr(),
        };
        if let Some(g) = graphs
            && g.baked != baked
        {
            tracing::warn!(
                "DS41 L{}: captured step bakes other buffers ({:?} vs {:?}); recapturing",
                self.idx,
                g.baked,
                baked
            );
            if let Some(g) = graphs.take() {
                g.destroy(gpu)?;
            }
        }
        let graphs = graphs.get_or_insert(LayerGraphs {
            a: [None, None],
            b: None,
            baked,
        });

        // ── segment A ──
        if capturable {
            let attn_ref: &AttnV41 = &attn;
            let st_ref: &AttnV41LayerState = st_attn;
            self.run_segment(gpu, stream, &mut graphs.a[0], || {
                self.seg_attn_in(gpu, streams, hidden, normed, stream)?;
                let attn_out =
                    attn_ref.decode_body(gpu, &self.attn_w, st_ref, normed, rows_b, stream)?;
                self.seg_ffn_in(gpu, &moe, attn_out, streams, hidden, normed, stream)
            })?;
        } else {
            self.run_segment(gpu, stream, &mut graphs.a[0], || {
                self.seg_attn_in(gpu, streams, hidden, normed, stream)
            })?;
            // a kv/index source: the compressor group and the index top-k
            // are host work in the middle of the attention, so it runs eagerly
            let run = attn.forward(
                gpu,
                &self.attn_w,
                st_attn,
                &mut shared,
                normed,
                1,
                start_pos,
                stream,
            )?;
            // it wrote pos / head_pos / idx_dev for itself
            attn.invalidate_decode_uploads();
            let attn_out = run.out;
            self.run_segment(gpu, stream, &mut graphs.a[1], || {
                self.seg_ffn_in(gpu, &moe, attn_out, streams, hidden, normed, stream)
            })?;
        }
        *rt.step_attn_ms.lock().unwrap() += ta.elapsed().as_secs_f64() * 1e3;
        if diag {
            gpu.synchronize(stream)?;
            tracing::info!(
                "DS41 L{} attn: out rms {:.4}; ffn: in rms {:.4} normed rms {:.4} ({})",
                self.idx,
                diag_rms_bf16(gpu, attn.out_ptr(), h),
                diag_rms_bf16(gpu, hidden, h),
                diag_rms_bf16(gpu, normed, h),
                if capturable {
                    "graph"
                } else {
                    "graph+eager attention"
                }
            );
        }
        drop(shared);
        drop(attn);

        // ── the host span ──
        let stage = {
            let mut lru = rt.lru.lock().unwrap();
            moe.stage_m1(
                gpu,
                &self.moe_w,
                &mut lru,
                &rt.slices,
                rt.reader_threads,
                stream,
            )?
        };

        // ── segment B ──
        let tb = std::time::Instant::now();
        let ne = stage.ne;
        self.run_segment(gpu, stream, &mut graphs.b, || {
            self.seg_ffn_out(gpu, &moe, ne, streams, hidden, normed, stream)
        })?;
        if diag {
            gpu.synchronize(stream)?;
            tracing::info!(
                "DS41 L{} ffn: out rms {:.4} (graph)",
                self.idx,
                diag_rms_bf16(gpu, moe.out_ptr(), h)
            );
        }
        let mut timing = stage.timing;
        timing.compute_ms = tb.elapsed().as_secs_f64() * 1e3;
        rt.step_moe.lock().unwrap().add(&timing);
        drop(moe);

        if self.idx + 1 == rt.n_layers {
            self.step_line(1, start_pos, "graph");
            gpu.synchronize(stream)?;
        }
        Ok(())
    }

    /// `ATLAS_DS41_GRAPH_ORACLE=1`: the captured step, then the highway put
    /// back and the eager step, and the two outputs (the streams, the delayed
    /// mix, the collapsed hidden) compared bit for bit. Diagnostics: every
    /// layer runs twice.
    fn step_oracle(
        &self,
        hidden: DevicePtr,
        start_pos: usize,
        st: &mut V41LayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let gpu = ctx.gpu;
        let streams = ctx.buffers.hc_streams();
        let (h, hc) = (rt.hidden, rt.hc_mult);
        let d2h = |p: DevicePtr, n: usize| -> Result<Vec<u8>> {
            let mut b = vec![0u8; n];
            gpu.copy_d2h(p, &mut b)?;
            Ok(b)
        };
        gpu.synchronize(stream)?;
        let streams0 = d2h(streams, hc * h * 4)?;
        let pre0 = d2h(rt.pre_prev, hc * 4)?;

        self.step_graph(hidden, start_pos, st, ctx, stream)?;
        gpu.synchronize(stream)?;
        let streams_g = d2h(streams, hc * h * 4)?;
        let pre_g = d2h(rt.pre_prev, hc * 4)?;
        let hidden_g = d2h(hidden, h * 2)?;

        gpu.copy_h2d(&streams0, streams)?;
        gpu.copy_h2d(&pre0, rt.pre_prev)?;
        self.step_eager(hidden, 1, start_pos, st, ctx, stream)?;
        gpu.synchronize(stream)?;
        let streams_e = d2h(streams, hc * h * 4)?;
        let pre_e = d2h(rt.pre_prev, hc * 4)?;
        let hidden_e = d2h(hidden, h * 2)?;

        let diff = |a: &[u8], b: &[u8], w: usize| {
            a.chunks(w).zip(b.chunks(w)).filter(|(x, y)| x != y).count()
        };
        let ds = diff(&streams_g, &streams_e, 4);
        let dp = diff(&pre_g, &pre_e, 4);
        // the collapsed hidden is the layer's output only on the last layer;
        // elsewhere it is the ffn input scratch, which both paths write
        let dh = diff(&hidden_g, &hidden_e, 2);
        if ds + dp + dh == 0 {
            tracing::info!(
                "DS41 GRAPH ORACLE L{} pos {start_pos}: graph == eager (streams {} f32, pre {} f32, hidden {} bf16)",
                self.idx,
                hc * h,
                hc,
                h
            );
        } else {
            tracing::error!(
                "DS41 GRAPH ORACLE L{} pos {start_pos}: MISMATCH streams {ds}/{} pre {dp}/{} hidden {dh}/{}",
                self.idx,
                hc * h,
                hc,
                h
            );
        }
        Ok(())
    }

    /// The per-step diagnostic line (`ATLAS_DS41_DIAG=1`), from the last layer.
    fn step_line(&self, m: usize, start_pos: usize, how: &str) {
        if !diag_on() {
            return;
        }
        let rt = &self.rt;
        let total = rt
            .step_start
            .lock()
            .unwrap()
            .map(|t| t.elapsed().as_secs_f64() * 1e3)
            .unwrap_or(0.0);
        let mo = *rt.step_moe.lock().unwrap();
        // one guard at a time: two `lru.lock()` temporaries in a single
        // statement deadlock on the std Mutex (the first guard lives to the
        // end of the statement)
        let (resident, n_slots) = {
            let lru = rt.lru.lock().unwrap();
            (lru.resident(), lru.n_slots())
        };
        let attn_ms = *rt.step_attn_ms.lock().unwrap();
        let engram_ms = *rt.step_engram_ms.lock().unwrap();
        tracing::info!(
            "DS41 step ({how}): {m} tok pos {start_pos}: total {total:.0} ms = attn {:.0} + engram {:.0} + moe(route {:.0} fetch {:.0} compute {:.0}) ms; experts hit {} miss {} read {:.2} GiB; cache {}/{} resident",
            attn_ms,
            engram_ms,
            mo.route_ms,
            mo.fetch_ms,
            mo.compute_ms,
            mo.hits,
            mo.misses,
            mo.bytes_read as f64 / 1073741824.0,
            resident,
            n_slots
        );
    }
}

impl crate::layer::TransformerLayer for DeepSeekV41Layer {
    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        state: &mut dyn LayerState,
        _kv_cache: &mut spark_runtime::kv_cache::PagedKvCache,
        seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.step(hidden, 1, seq_len, state, ctx, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut spark_runtime::kv_cache::PagedKvCache,
        seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.step(hidden, num_tokens, seq_len_start, state, ctx, stream)
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(V41LayerState {
            attn: AttnV41LayerState::new(gpu, &self.rt.attn_cfg, self.role)?,
            graphs: None,
        }))
    }

    /// The captured segments bake this sequence's buffers; they go with it.
    fn release_state(&self, state: &mut dyn LayerState, gpu: &dyn GpuBackend) -> Result<()> {
        if let Some(st) = state.as_any_mut().downcast_mut::<V41LayerState>()
            && let Some(g) = st.graphs.take()
        {
            g.destroy(gpu)?;
        }
        Ok(())
    }

    fn decode_graph_unsupported(&self) -> bool {
        true
    }

    fn decode_multi_seq_unsupported(&self) -> bool {
        true
    }
}
