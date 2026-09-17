// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! DeepSeek-V4.1 Flash attention on the GPU: one forward for every layer.
//!
//! V4.1 has three kinds of attention layer, told apart by `compress_ratio`:
//! ratio 0 attends over a sliding window of its own fp8 latent rows; ratio 1
//! and ratio 2 attend over that window PLUS a set of compressed positions chosen
//! by an indexer from a latent cache that only the four `kv_source_layer_ids`
//! produce (the `SharedAttentionRuntime`: four layers write, forty read). The
//! index selection itself is computed by eight `index_source_layer_ids` and
//! reused by the layers between them; layer `candidate_source_layer` also
//! publishes a block-level candidate mask the later index sources respect.
//!
//! This module is the production form of `deepseek_v41_ref::compress::attention_any`,
//! stage for stage, and is held to it layer by layer in `attn_v41_tests.rs`:
//! * GEMMs on `common/dense_gemm_bf16` (f32 accumulate, bf16 out = `linear_bf16`);
//! * RMSNorm, RoPE, the fp8/fp4 quantisers, the compressor pooling, the
//!   indexer scores, sparse attention with the sink and the grouped output
//!   projection on `kernels/gb10/deepseek-v4-flash/nvfp4/attn_v41.cu`;
//! * the two top-k selections (candidate blocks, index positions) on the CPU
//!   through the reference's `torch_cpu_topk_set`, because the reference
//!   returns torch's CPU tie order among equal scores and a kernel that picks
//!   any other set diverges from it on exactly those queries. The scores are
//!   bf16-valued and few (`[tokens, width]`), so this costs a download, not a
//!   kernel.
//!
//! Layer state (window ring, compressor partial group, the two caches) and the
//! shared slots live on the device; the shared index selection and candidate
//! mask are host vectors, as the reference keeps them.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use crate::layers::deepseek_v41_ref::attn::{freqs_cis, window_topk_idxs};
use crate::layers::deepseek_v41_ref::compress::{
    FP4_BLOCK, LATENT_BLOCK, select_candidate_blocks, torch_cpu_topk_set, yarn_freqs_cis,
};
use crate::layers::ops;
use crate::layers::ops::{
    KQUANT_MODULE, Q2K_MMQ_SMEM, kquant_mmq_act_bytes, kquant_mmq_gemm, kquant_mmvq_w,
    kquant_q8_1_rows, kquant_q8_1_rows_bytes,
};
use crate::weight_map::DenseWeight;

/// An attention projection on the device: bf16 or the GGUF's raw `Q2_K`
/// blocks (`Q3_K` is never an attention type here and is refused).
pub use crate::layers::ops::ResidentMat as AttnMat;

const MODULE: &str = "attn_v41";
const GEMM_MODULE: &str = "gemm";
/// One warp per quantiser block; 8 warps per launch block.
const QUANT_BLOCKS_PER_LAUNCH: u32 = 8;

/// Model-wide attention geometry, from `ModelConfig`.
#[derive(Clone, Debug)]
pub struct AttnV41Cfg {
    pub dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub q_rank: usize,
    pub o_rank: usize,
    pub groups: usize,
    pub window: usize,
    pub eps: f32,
    pub index_heads: usize,
    pub index_hd: usize,
    pub index_topk: usize,
    pub cand_topk_blocks: usize,
    pub cand_block: usize,
    pub max_seq: usize,
    pub max_tokens: usize,
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub rope_factor: f32,
    pub orig_seq: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

impl AttnV41Cfg {
    pub fn gw(&self) -> usize {
        self.n_heads * self.head_dim / self.groups
    }
}

/// What one layer is, in the shared runtime.
#[derive(Clone, Copy, Debug, Default)]
pub struct LayerRole {
    pub ratio: usize,
    pub is_kv_source: bool,
    pub is_index_source: bool,
    pub is_candidate_source: bool,
    pub uses_candidates: bool,
}

/// Compressor weights (kv sources only). At ratio > 1 `kv` and `gate` are f32
/// `[hd, dim]` as the checkpoint stores them; at ratio 1 `kv` is bf16.
pub struct CompressorWeightsGpu {
    pub kv: DevicePtr,
    pub gate: Option<DevicePtr>,
    /// f32 `[hd]`
    pub norm: DevicePtr,
}

/// Indexer weights (index sources only); `wk` / `k_norm` on kv sources.
pub struct IndexerWeightsGpu {
    /// bf16 `[index_heads * index_hd, q_rank]`
    pub wq_b: DevicePtr,
    /// bf16 `[index_heads, dim]`
    pub weights_proj: DevicePtr,
    /// bf16 `[index_hd, hd]`
    pub wk: Option<DevicePtr>,
    /// f32 `[index_hd]`
    pub k_norm: Option<DevicePtr>,
}

/// One layer's resident attention weights. bf16 `[out, in]` unless noted.
pub struct AttnV41LayerWeights {
    pub role: LayerRole,
    /// f32 `[n_heads]`
    pub sink: DevicePtr,
    pub wq_a: AttnMat,
    /// f32 `[q_rank]`
    pub q_norm: DevicePtr,
    pub wq_b: AttnMat,
    pub wkv: AttnMat,
    /// f32 `[hd]`
    pub kv_norm: DevicePtr,
    /// `[groups * o_rank, gw]`
    pub wo_a: AttnMat,
    /// `[dim, groups * o_rank]`
    pub wo_b: AttnMat,
    pub comp: Option<CompressorWeightsGpu>,
    pub idx: Option<IndexerWeightsGpu>,
}

/// One layer's device state across prefill and decode.
pub struct AttnV41LayerState {
    /// bf16 `[window, hd]`
    window: DevicePtr,
    /// f32 `[ratio, hd]` x 2, the partial group of a ratio-2 compressor
    comp_state: Option<(DevicePtr, DevicePtr)>,
    /// bf16 `[max_groups, hd]` (kv sources)
    compress_kv: Option<DevicePtr>,
    /// bf16 `[max_groups, index_hd]` (kv sources)
    index_k: Option<DevicePtr>,
}

impl AttnV41LayerState {
    pub fn new(gpu: &dyn GpuBackend, c: &AttnV41Cfg, role: LayerRole) -> Result<Self> {
        let window = gpu.alloc(c.window * c.head_dim * 2)?;
        gpu.memset(window, 0, c.window * c.head_dim * 2)?;
        let (comp_state, compress_kv, index_k) = if role.is_kv_source {
            let ratio = role.ratio.max(1);
            let groups = c.max_seq / ratio;
            let a = gpu.alloc(ratio * c.head_dim * 4)?;
            let b = gpu.alloc(ratio * c.head_dim * 4)?;
            gpu.memset(a, 0, ratio * c.head_dim * 4)?;
            // score state starts at -inf as the reference's does
            let neg: Vec<u8> =
                std::iter::repeat_n(f32::NEG_INFINITY.to_le_bytes(), ratio * c.head_dim)
                    .flatten()
                    .collect();
            gpu.copy_h2d(&neg, b)?;
            let ckv = gpu.alloc(groups.max(1) * c.head_dim * 2)?;
            gpu.memset(ckv, 0, groups.max(1) * c.head_dim * 2)?;
            let ik = gpu.alloc(groups.max(1) * c.index_hd * 2)?;
            gpu.memset(ik, 0, groups.max(1) * c.index_hd * 2)?;
            (Some((a, b)), Some(ckv), Some(ik))
        } else {
            (None, None, None)
        };
        Ok(AttnV41LayerState {
            window,
            comp_state,
            compress_kv,
            index_k,
        })
    }

    /// The window ring, bf16 `[window, hd]`: a pointer a captured decode step bakes.
    pub fn window(&self) -> DevicePtr {
        self.window
    }

    /// Free the four device buffers this sequence owns and clear the
    /// pointers. Idempotent: `gpu.free` is null-safe and the taken options
    /// stay `None`, so a second call frees nothing. Takes `&mut self` because
    /// the state reaches `release_state` as `&mut dyn LayerState` and cannot
    /// be moved out of the box.
    pub fn free(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        let window = std::mem::replace(&mut self.window, DevicePtr::NULL);
        gpu.free(window)?;
        if let Some((a, b)) = self.comp_state.take() {
            gpu.free(a)?;
            gpu.free(b)?;
        }
        if let Some(p) = self.compress_kv.take() {
            gpu.free(p)?;
        }
        if let Some(p) = self.index_k.take() {
            gpu.free(p)?;
        }
        Ok(())
    }
}

/// The `SharedAttentionRuntime`: what source layers publish for the layers
/// below them.
#[derive(Default)]
pub struct SharedV41 {
    /// the current kv source's latent cache and how many groups it holds
    pub compress_kv: Option<DevicePtr>,
    pub compress_len: usize,
    /// the current kv source's index-key cache
    pub index_k: Option<DevicePtr>,
    /// `[queries][topk]`, offset for the window rows, -1 = absent
    pub topk_idxs: Vec<i32>,
    pub topk: usize,
    /// `[queries][cand_width]`
    pub candidates: Vec<bool>,
    pub cand_width: usize,
}

/// The intermediates of one layer's forward, for oracles and for the caller.
pub struct AttnV41Run {
    /// bf16 `[tokens, n_heads, hd]`, after RoPE
    pub q: DevicePtr,
    /// window rows: the chunk's own rows on prefill, the ring on decode
    pub rows_a: DevicePtr,
    pub rows_a_len: usize,
    /// compressed rows (the kv source's cache) and how many are attended
    pub rows_b: Option<DevicePtr>,
    pub rows_b_len: usize,
    /// `[tokens][topk]`, window slots then compressed positions (offset by `rows_a_len`)
    pub idx: Vec<i32>,
    pub topk: usize,
    /// bf16 `[tokens, n_heads, hd]`, pre inverse rotation
    pub o: DevicePtr,
    /// bf16 `[tokens, dim]`
    pub out: DevicePtr,
}

struct Kernels {
    gemm: KernelHandle,
    /// `dense_gemv_bf16` for the single-token step: the tiled GEMM spends
    /// 15 of its 16 rows idle at m = 1 (326 us a launch on GB10 vs the
    /// GEMV's bandwidth-bound pass over the same `[N, K]` weight)
    gemv: KernelHandle,
    /// the K-quant path for `AttnMat::Q2K`: q8_1 row quant + GEMV at m <= 8,
    /// D2S6 tile quant + MMQ above
    q8_rows: KernelHandle,
    mmvq_q2k_w: KernelHandle,
    quant_d2s6: KernelHandle,
    mmq_q2k_nc: KernelHandle,
    mmq_q2k_wc: KernelHandle,
    rmsnorm_bf16: KernelHandle,
    rmsnorm_f32: KernelHandle,
    rope: KernelHandle,
    act_quant: KernelHandle,
    fp4_quant: KernelHandle,
    gemm_f32: KernelHandle,
    pool: KernelHandle,
    index_score: KernelHandle,
    sparse_attn: KernelHandle,
    slice_cols: KernelHandle,
    scatter_cols: KernelHandle,
    scale_bf16: KernelHandle,
    /// the decode step's window ring write with the slot read on the device
    ring_put: KernelHandle,
}

/// The attention runtime: kernels, RoPE tables, and workspaces for up to
/// `max_tokens` positions per call.
pub struct AttnV41 {
    pub cfg: AttnV41Cfg,
    k: Kernels,
    fc_plain: DevicePtr,
    fc_yarn: DevicePtr,
    // workspaces
    /// q8_1 activations for the Q2_K projections (plain rows or MMQ tiles)
    a_q8: DevicePtr,
    qr_raw: DevicePtr,
    qr: DevicePtr,
    q: DevicePtr,
    kv_raw: DevicePtr,
    kv: DevicePtr,
    o: DevicePtr,
    o_rot: DevicePtr,
    og: DevicePtr,
    slice_in: DevicePtr,
    slice_out: DevicePtr,
    out: DevicePtr,
    pos: DevicePtr,
    head_pos: DevicePtr,
    idx_pos: DevicePtr,
    grp_pos: DevicePtr,
    idx_dev: DevicePtr,
    ckv: DevicePtr,
    cscore: DevicePtr,
    pooled: DevicePtr,
    latent_raw: DevicePtr,
    latent: DevicePtr,
    ik_raw: DevicePtr,
    ik: DevicePtr,
    iq: DevicePtr,
    iw_raw: DevicePtr,
    iw: DevicePtr,
    score: DevicePtr,
    /// What the graph-captured decode step last uploaded into `pos` /
    /// `head_pos` (the position) and `idx_dev` (the padded selection), so a
    /// step re-uploads only what changed. `None` = unknown, upload.
    decode_pos: Option<usize>,
    decode_idx: Option<Vec<i32>>,
}

fn upload_f32(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(bytes.len().max(4))?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}

fn upload_i32(gpu: &dyn GpuBackend, dst: DevicePtr, v: &[i32]) -> Result<()> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    gpu.copy_h2d(&bytes, dst)
}

/// Stream-ordered upload from a transient host vector (staged by the driver,
/// no stream drain), for the per-token inputs of the captured decode step.
fn upload_i32_async(gpu: &dyn GpuBackend, dst: DevicePtr, v: &[i32], stream: u64) -> Result<()> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    gpu.copy_h2d_async(&bytes, dst, stream)
}

fn at(p: DevicePtr, byte_off: usize) -> DevicePtr {
    DevicePtr(p.0 + byte_off as u64)
}

impl AttnV41 {
    pub fn new(gpu: &dyn GpuBackend, cfg: AttnV41Cfg) -> Result<Self> {
        ensure!(
            cfg.rope_dim.is_multiple_of(2) && cfg.rope_dim / 2 <= 256,
            "rope_dim {} not supported",
            cfg.rope_dim
        );
        ensure!(
            (cfg.head_dim * cfg.n_heads).is_multiple_of(cfg.groups),
            "n_heads * hd not divisible by groups"
        );
        ensure!(
            cfg.head_dim.is_multiple_of(32) && cfg.index_hd.is_multiple_of(32),
            "head dims must be multiples of the 32-block"
        );
        let k = Kernels {
            gemm: gpu.kernel(GEMM_MODULE, "dense_gemm_bf16")?,
            gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            q8_rows: gpu.kernel(KQUANT_MODULE, "kquant_q8_1_rows_bf16")?,
            mmvq_q2k_w: gpu.kernel(KQUANT_MODULE, "kquant_mmvq_q2_k_w")?,
            quant_d2s6: gpu.kernel(KQUANT_MODULE, "atlas_q8_1_quantize_d2s6_bf16")?,
            mmq_q2k_nc: gpu.kernel(KQUANT_MODULE, "atlas_q2_k_mmq128_nc")?,
            mmq_q2k_wc: gpu.kernel(KQUANT_MODULE, "atlas_q2_k_mmq128_wc")?,
            rmsnorm_bf16: gpu.kernel(MODULE, "attn_v41_rmsnorm_bf16")?,
            rmsnorm_f32: gpu.kernel(MODULE, "attn_v41_rmsnorm_f32")?,
            rope: gpu.kernel(MODULE, "attn_v41_rope")?,
            act_quant: gpu.kernel(MODULE, "attn_v41_act_quant_fp8")?,
            fp4_quant: gpu.kernel(MODULE, "attn_v41_fp4_quant")?,
            gemm_f32: gpu.kernel(MODULE, "attn_v41_gemm_f32")?,
            pool: gpu.kernel(MODULE, "attn_v41_pool")?,
            index_score: gpu.kernel(MODULE, "attn_v41_index_score")?,
            sparse_attn: gpu.kernel(MODULE, "attn_v41_sparse_attn")?,
            slice_cols: gpu.kernel(MODULE, "attn_v41_slice_cols")?,
            scatter_cols: gpu.kernel(MODULE, "attn_v41_scatter_cols")?,
            scale_bf16: gpu.kernel(MODULE, "attn_v41_scale_bf16")?,
            ring_put: gpu.kernel(MODULE, "attn_v41_ring_put")?,
        };
        let plain = freqs_cis(cfg.rope_dim, cfg.max_seq, cfg.rope_theta);
        let yarn = yarn_freqs_cis(
            cfg.rope_dim,
            cfg.max_seq,
            cfg.orig_seq,
            cfg.compress_rope_theta,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
        );
        let flat = |v: &[(f32, f32)]| -> Vec<f32> { v.iter().flat_map(|&(c, s)| [c, s]).collect() };
        let fc_plain = upload_f32(gpu, &flat(&plain))?;
        let fc_yarn = upload_f32(gpu, &flat(&yarn))?;
        let m = cfg.max_tokens;
        let (nh, hd, nhi, ihd) = (cfg.n_heads, cfg.head_dim, cfg.index_heads, cfg.index_hd);
        let max_width = cfg.max_seq;
        let max_topk = cfg.window + cfg.index_topk;
        let alloc = |bytes: usize| gpu.alloc(bytes.max(16));
        let kmax = cfg
            .dim
            .max(cfg.q_rank)
            .max(cfg.gw())
            .max(cfg.groups * cfg.o_rank);
        Ok(AttnV41 {
            a_q8: alloc(
                kquant_q8_1_rows_bytes(8, kmax as u32)
                    .max(kquant_mmq_act_bytes(m as u32, kmax as u32)),
            )?,
            qr_raw: alloc(m * cfg.q_rank * 2)?,
            qr: alloc(m * cfg.q_rank * 2)?,
            q: alloc(m * nh * hd * 2)?,
            kv_raw: alloc(m * hd * 2)?,
            kv: alloc(m * hd * 2)?,
            o: alloc(m * nh * hd * 2)?,
            o_rot: alloc(m * nh * hd * 2)?,
            og: alloc(m * cfg.groups * cfg.o_rank * 2)?,
            slice_in: alloc(m * cfg.gw() * 2)?,
            slice_out: alloc(m * cfg.o_rank * 2)?,
            out: alloc(m * cfg.dim * 2)?,
            pos: alloc(m * 4)?,
            head_pos: alloc(m * nh * 4)?,
            idx_pos: alloc(m * nhi * 4)?,
            grp_pos: alloc(m * 4)?,
            idx_dev: alloc(m * max_topk * 4)?,
            ckv: alloc(m * hd * 4)?,
            cscore: alloc(m * hd * 4)?,
            pooled: alloc(m * hd * 4)?,
            latent_raw: alloc(m * hd * 2)?,
            latent: alloc(m * hd * 2)?,
            ik_raw: alloc(m * ihd * 2)?,
            ik: alloc(m * ihd * 2)?,
            iq: alloc(m * nhi * ihd * 2)?,
            iw_raw: alloc(m * nhi * 2)?,
            iw: alloc(m * nhi * 2)?,
            score: alloc(m * max_width * 4)?,
            decode_pos: None,
            decode_idx: None,
            cfg,
            k,
            fc_plain,
            fc_yarn,
        })
    }

    fn gemm(
        &self,
        gpu: &dyn GpuBackend,
        a: DevicePtr,
        w: AttnMat,
        c: DevicePtr,
        m: usize,
        n: usize,
        kk: usize,
        stream: u64,
    ) -> Result<()> {
        let w = match w {
            AttnMat::Bf16(p) => p,
            AttnMat::Q3K(_) => anyhow::bail!("attention projections are never Q3_K"),
            AttnMat::Q2K(blocks) => {
                let (m, n, kk) = (m as u32, n as u32, kk as u32);
                if m <= 8 {
                    kquant_q8_1_rows(gpu, self.k.q8_rows, a, self.a_q8, m, kk, stream)?;
                    return kquant_mmvq_w(
                        gpu,
                        self.k.mmvq_q2k_w,
                        blocks,
                        self.a_q8,
                        c,
                        n,
                        kk,
                        m,
                        stream,
                    );
                }
                ops::quantize_act_q8_1(gpu, self.k.quant_d2s6, a, self.a_q8, m, kk, stream)?;
                return kquant_mmq_gemm(
                    gpu,
                    self.k.mmq_q2k_nc,
                    self.k.mmq_q2k_wc,
                    self.a_q8,
                    blocks,
                    c,
                    m,
                    n,
                    kk,
                    Q2K_MMQ_SMEM,
                    stream,
                );
            }
        };
        if m == 1 {
            return ops::dense_gemv(
                gpu,
                self.k.gemv,
                a,
                &DenseWeight { weight: w },
                c,
                n as u32,
                kk as u32,
                stream,
            );
        }
        ops::dense_gemm(
            gpu,
            self.k.gemm,
            a,
            &DenseWeight { weight: w },
            c,
            m as u32,
            n as u32,
            kk as u32,
            stream,
        )
    }

    fn rmsnorm(
        &self,
        gpu: &dyn GpuBackend,
        f32_in: bool,
        x: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        rows: usize,
        dim: usize,
        stream: u64,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        KernelLaunch::new(
            gpu,
            if f32_in {
                self.k.rmsnorm_f32
            } else {
                self.k.rmsnorm_bf16
            },
        )
        .grid([rows as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(w)
        .arg_ptr(out)
        .arg_u32(dim as u32)
        .arg_f32(self.cfg.eps)
        .launch(stream)
    }

    fn rope(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        pos: DevicePtr,
        rows: usize,
        row_len: usize,
        yarn: bool,
        inverse: bool,
        stream: u64,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        KernelLaunch::new(gpu, self.k.rope)
            .grid([rows as u32, 1, 1])
            .block([(self.cfg.rope_dim / 2).max(32) as u32, 1, 1])
            .arg_ptr(x)
            .arg_ptr(pos)
            .arg_ptr(if yarn { self.fc_yarn } else { self.fc_plain })
            .arg_u32(row_len as u32)
            .arg_u32(self.cfg.rope_dim as u32)
            .arg_u32(inverse as u32)
            .launch(stream)
    }

    fn act_quant(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        n_values: usize,
        stream: u64,
    ) -> Result<()> {
        let n_blocks = n_values / 32;
        if n_blocks == 0 {
            return Ok(());
        }
        KernelLaunch::new(gpu, self.k.act_quant)
            .grid([(n_blocks as u32).div_ceil(QUANT_BLOCKS_PER_LAUNCH), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_u32(n_blocks as u32)
            .launch(stream)
    }

    fn fp4_quant(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        n_values: usize,
        block: usize,
        e4m3_scale: bool,
        stream: u64,
    ) -> Result<()> {
        let n_blocks = n_values / block;
        if n_blocks == 0 {
            return Ok(());
        }
        KernelLaunch::new(gpu, self.k.fp4_quant)
            .grid([(n_blocks as u32).div_ceil(QUANT_BLOCKS_PER_LAUNCH), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_u32(n_blocks as u32)
            .arg_u32(block as u32)
            .arg_u32(e4m3_scale as u32)
            .launch(stream)
    }

    /// The compressor's latent for this call (pre-RoPE, bf16 in `self.latent`),
    /// or `None` while a ratio-2 group is still filling. Returns the group count.
    fn compressor(
        &self,
        gpu: &dyn GpuBackend,
        w: &AttnV41LayerWeights,
        st: &AttnV41LayerState,
        x: DevicePtr,
        m: usize,
        start_pos: usize,
        stream: u64,
    ) -> Result<Option<usize>> {
        let c = &self.cfg;
        let comp = w
            .comp
            .as_ref()
            .context("kv source without compressor weights")?;
        let hd = c.head_dim;
        let ratio = w.role.ratio;
        if ratio == 1 {
            self.gemm(
                gpu,
                x,
                AttnMat::Bf16(comp.kv),
                self.latent_raw,
                m,
                hd,
                c.dim,
                stream,
            )?;
            self.rmsnorm(
                gpu,
                false,
                self.latent_raw,
                comp.norm,
                self.latent,
                m,
                hd,
                stream,
            )?;
            return Ok(Some(m));
        }
        let gate = comp.gate.context("ratio > 1 compressor without wgate")?;
        let (kv_state, score_state) = st.comp_state.context("compressor state")?;
        let gemm_f32 = |wt: DevicePtr, out: DevicePtr| {
            KernelLaunch::new(gpu, self.k.gemm_f32)
                .grid([(hd as u32).div_ceil(16), (m as u32).div_ceil(16), 1])
                .block([16, 16, 1])
                .arg_ptr(x)
                .arg_ptr(wt)
                .arg_ptr(out)
                .arg_u32(m as u32)
                .arg_u32(hd as u32)
                .arg_u32(c.dim as u32)
                .launch(stream)
        };
        gemm_f32(comp.kv, self.ckv)?;
        gemm_f32(gate, self.cscore)?;
        let row = hd * 4;
        let (src_kv, src_score, groups) = if start_pos == 0 {
            let remainder = m % ratio;
            let cutoff = m - remainder;
            if remainder > 0 {
                gpu.copy_d2d_async(
                    at(self.ckv, cutoff * row),
                    kv_state,
                    remainder * row,
                    stream,
                )?;
                gpu.copy_d2d_async(
                    at(self.cscore, cutoff * row),
                    score_state,
                    remainder * row,
                    stream,
                )?;
            }
            if m < ratio {
                return Ok(None);
            }
            (self.ckv, self.cscore, cutoff / ratio)
        } else {
            let slot = start_pos % ratio;
            gpu.copy_d2d_async(self.ckv, at(kv_state, slot * row), row, stream)?;
            gpu.copy_d2d_async(self.cscore, at(score_state, slot * row), row, stream)?;
            if !(start_pos + 1).is_multiple_of(ratio) {
                return Ok(None);
            }
            (kv_state, score_state, 1)
        };
        KernelLaunch::new(gpu, self.k.pool)
            .grid([groups as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(src_kv)
            .arg_ptr(src_score)
            .arg_ptr(self.pooled)
            .arg_u32(ratio as u32)
            .arg_u32(hd as u32)
            .launch(stream)?;
        self.rmsnorm(
            gpu,
            true,
            self.pooled,
            comp.norm,
            self.latent,
            groups,
            hd,
            stream,
        )?;
        Ok(Some(groups))
    }

    /// Group positions of `groups` new latents: `g * ratio` on prefill, the
    /// group's first position on decode.
    fn group_positions(groups: usize, ratio: usize, start_pos: usize) -> Vec<i32> {
        if start_pos == 0 {
            (0..groups).map(|g| (g * ratio) as i32).collect()
        } else {
            vec![(start_pos + 1 - ratio) as i32]
        }
    }

    /// The indexer: publishes index keys (kv sources), scores the compressed
    /// positions, selects candidates and the top-k on the CPU. Returns
    /// `([tokens][topk], topk)` offset by `offset`.
    #[allow(clippy::too_many_arguments)]
    fn indexer(
        &self,
        gpu: &dyn GpuBackend,
        w: &AttnV41LayerWeights,
        st: &AttnV41LayerState,
        shared: &mut SharedV41,
        x: DevicePtr,
        latent_groups: Option<usize>,
        m: usize,
        start_pos: usize,
        offset: usize,
        stream: u64,
    ) -> Result<(Vec<i32>, usize)> {
        let c = &self.cfg;
        let iw = w
            .idx
            .as_ref()
            .context("index source without indexer weights")?;
        let ratio = w.role.ratio;
        let (nhi, ihd, hd) = (c.index_heads, c.index_hd, c.head_dim);
        let end_pos = start_pos + m;
        if let (Some(groups), Some(cache)) = (latent_groups, st.index_k) {
            let wk = iw.wk.context("kv source without index wk")?;
            let k_norm = iw.k_norm.context("kv source without index k_norm")?;
            self.gemm(
                gpu,
                self.latent,
                AttnMat::Bf16(wk),
                self.ik_raw,
                groups,
                ihd,
                hd,
                stream,
            )?;
            self.rmsnorm(
                gpu,
                false,
                self.ik_raw,
                k_norm,
                self.ik,
                groups,
                ihd,
                stream,
            )?;
            upload_i32(
                gpu,
                self.grp_pos,
                &Self::group_positions(groups, ratio, start_pos),
            )?;
            self.rope(gpu, self.ik, self.grp_pos, groups, ihd, true, false, stream)?;
            self.fp4_quant(gpu, self.ik, groups * ihd, FP4_BLOCK, false, stream)?;
            gpu.copy_d2d_async(
                self.ik,
                at(cache, (start_pos / ratio) * ihd * 2),
                groups * ihd * 2,
                stream,
            )?;
            shared.index_k = Some(cache);
        }
        let index_k = shared
            .index_k
            .context("indexer before any index keys were published")?;
        // queries
        self.gemm(
            gpu,
            self.qr,
            AttnMat::Bf16(iw.wq_b),
            self.iq,
            m,
            nhi * ihd,
            c.q_rank,
            stream,
        )?;
        let ihpos: Vec<i32> = (0..m)
            .flat_map(|t| std::iter::repeat_n((start_pos + t) as i32, nhi))
            .collect();
        upload_i32(gpu, self.idx_pos, &ihpos)?;
        self.rope(
            gpu,
            self.iq,
            self.idx_pos,
            m * nhi,
            ihd,
            true,
            false,
            stream,
        )?;
        self.fp4_quant(gpu, self.iq, m * nhi * ihd, FP4_BLOCK, false, stream)?;
        // head weights: bf16(bf16(x . wproj) * ihd^-0.5 * nh^-0.5)
        self.gemm(
            gpu,
            x,
            AttnMat::Bf16(iw.weights_proj),
            self.iw_raw,
            m,
            nhi,
            c.dim,
            stream,
        )?;
        let wscale = (ihd as f32).powf(-0.5) * (nhi as f32).powf(-0.5);
        KernelLaunch::new(gpu, self.k.scale_bf16)
            .grid([((m * nhi) as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.iw_raw)
            .arg_ptr(self.iw)
            .arg_u32((m * nhi) as u32)
            .arg_f32(wscale)
            .launch(stream)?;
        let width = end_pos / ratio;
        ensure!(
            width * m <= c.max_seq * c.max_tokens,
            "index width {width} x {m} exceeds the workspace"
        );
        KernelLaunch::new(gpu, self.k.index_score)
            .grid([(width as u32).div_ceil(256), m as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(self.iq)
            .arg_ptr(index_k)
            .arg_ptr(self.iw)
            .arg_ptr(self.score)
            .arg_u32(width as u32)
            .arg_u32(nhi as u32)
            .arg_u32(ihd as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut bytes = vec![0u8; m * width * 4];
        gpu.copy_d2h(self.score, &mut bytes)?;
        let mut score: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let compress_lens: Vec<usize> = if start_pos == 0 {
            (0..m).map(|t| (t + 1) / ratio).collect()
        } else {
            vec![width; m]
        };
        if start_pos == 0 {
            for t in 0..m {
                for p in compress_lens[t]..width {
                    score[t * width + p] = f32::NEG_INFINITY;
                }
            }
        }
        if w.role.is_candidate_source {
            shared.candidates = select_candidate_blocks(
                &score,
                m,
                width,
                &compress_lens,
                c.cand_topk_blocks,
                c.cand_block,
            );
            shared.cand_width = width;
        } else if w.role.uses_candidates {
            ensure!(
                shared.candidates.len() == m * width,
                "candidate mask is {} for {} x {width}",
                shared.candidates.len(),
                m
            );
            for (s, &keep) in score.iter_mut().zip(&shared.candidates) {
                if !keep {
                    *s = f32::NEG_INFINITY;
                }
            }
        }
        let topk = c.index_topk.min(end_pos / ratio);
        let mut out = Vec::with_capacity(m * topk);
        for t in 0..m {
            let row = &score[t * width..(t + 1) * width];
            let mut picked = torch_cpu_topk_set(row, topk);
            picked.sort_unstable();
            for i in picked {
                out.push(if i < compress_lens[t] {
                    (i + offset) as i32
                } else {
                    -1
                });
            }
        }
        Ok((out, topk))
    }

    /// q (low-rank, normed, up-projected, rotated) and kv (one latent row per
    /// token, normed, rotated, fp8) from the normed input `x`. Device work
    /// only: the positions are read from `pos` / `head_pos`.
    fn q_kv_projections(
        &self,
        gpu: &dyn GpuBackend,
        w: &AttnV41LayerWeights,
        x: DevicePtr,
        m: usize,
        yarn: bool,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (nh, hd, dim) = (c.n_heads, c.head_dim, c.dim);
        self.gemm(gpu, x, w.wq_a, self.qr_raw, m, c.q_rank, dim, stream)?;
        self.rmsnorm(
            gpu,
            false,
            self.qr_raw,
            w.q_norm,
            self.qr,
            m,
            c.q_rank,
            stream,
        )?;
        self.gemm(gpu, self.qr, w.wq_b, self.q, m, nh * hd, c.q_rank, stream)?;
        self.rope(gpu, self.q, self.head_pos, m * nh, hd, yarn, false, stream)?;
        self.gemm(gpu, x, w.wkv, self.kv_raw, m, hd, dim, stream)?;
        self.rmsnorm(gpu, false, self.kv_raw, w.kv_norm, self.kv, m, hd, stream)?;
        self.rope(gpu, self.kv, self.pos, m, hd, yarn, false, stream)?;
        self.act_quant(gpu, self.kv, m * hd, stream)
    }

    /// Sparse attention over the selection in `idx_dev` (`topk` entries a
    /// token, -1 = absent), the inverse rotation and the grouped output
    /// projection into `self.out`. Device work only.
    #[allow(clippy::too_many_arguments)]
    fn attend_and_project(
        &self,
        gpu: &dyn GpuBackend,
        w: &AttnV41LayerWeights,
        rows_a: DevicePtr,
        rows_a_len: usize,
        rows_b: Option<DevicePtr>,
        topk: usize,
        m: usize,
        yarn: bool,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (nh, hd, dim) = (c.n_heads, c.head_dim, c.dim);
        // sparse attention with the sink, then the inverse rotation
        let scale = (hd as f32).powf(-0.5);
        KernelLaunch::new(gpu, self.k.sparse_attn)
            .grid([m as u32, nh as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(self.q)
            .arg_ptr(rows_a)
            .arg_ptr(rows_b.unwrap_or(rows_a))
            .arg_u32(rows_a_len as u32)
            .arg_ptr(self.idx_dev)
            .arg_ptr(w.sink)
            .arg_ptr(self.o)
            .arg_u32(nh as u32)
            .arg_u32(hd as u32)
            .arg_u32(topk as u32)
            .arg_f32(scale)
            .launch(stream)?;
        // `run.o` is the pre-rotation output (the reference's `sa_o`); the
        // inverse rotation runs on a copy
        let o_copy = self.o_rot;
        gpu.copy_d2d_async(self.o, o_copy, m * nh * hd * 2, stream)?;
        self.rope(gpu, o_copy, self.head_pos, m * nh, hd, yarn, true, stream)?;

        // grouped low-rank output projection: og[t, g*o_rank + r] = o_g . wo_a[g*o_rank + r]
        let gw = c.gw();
        for g in 0..c.groups {
            KernelLaunch::new(gpu, self.k.slice_cols)
                .grid([m as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(o_copy)
                .arg_ptr(self.slice_in)
                .arg_u32((nh * hd) as u32)
                .arg_u32((g * gw) as u32)
                .arg_u32(gw as u32)
                .launch(stream)?;
            self.gemm(
                gpu,
                self.slice_in,
                w.wo_a.at_rows(g * c.o_rank, gw),
                self.slice_out,
                m,
                c.o_rank,
                gw,
                stream,
            )?;
            KernelLaunch::new(gpu, self.k.scatter_cols)
                .grid([m as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.slice_out)
                .arg_ptr(self.og)
                .arg_u32((c.groups * c.o_rank) as u32)
                .arg_u32((g * c.o_rank) as u32)
                .arg_u32(c.o_rank as u32)
                .launch(stream)?;
        }
        self.gemm(
            gpu,
            self.og,
            w.wo_b,
            self.out,
            m,
            dim,
            c.groups * c.o_rank,
            stream,
        )?;
        Ok(())
    }

    /// Can this layer's single-token step be captured into a CUDA graph?
    /// The kv and index sources compute the position's compressor group and
    /// the index top-k on the host in the middle of the forward (position
    /// parity, a score download); every other layer is kernels and copies
    /// from end to end once its per-token inputs are read from the device.
    pub fn decode_capturable(&self, w: &AttnV41LayerWeights) -> bool {
        !w.role.is_kv_source && !w.role.is_index_source
    }

    /// The `topk` a captured step launches `sparse_attn` with: the widest
    /// selection the layer can ever see, so the launch argument is constant
    /// and the selection is padded with -1 (which the kernel skips: a padded
    /// entry adds a zero to the denominator sum and nothing to the output,
    /// so the result is the eager step's bit for bit).
    pub fn decode_fixed_topk(&self, w: &AttnV41LayerWeights) -> usize {
        let c = &self.cfg;
        if w.role.ratio > 0 {
            (c.window + c.index_topk).min(2048)
        } else {
            c.window
        }
    }

    /// Forget what the captured decode step last uploaded; the next
    /// [`Self::decode_prep`] re-uploads. Call after any eager `forward` (it
    /// writes `pos` / `head_pos` / `idx_dev` for itself) and at a new step.
    pub fn invalidate_decode_uploads(&mut self) {
        self.decode_pos = None;
        self.decode_idx = None;
    }

    /// The HOST half of a capturable layer's single-token step at `start_pos`:
    /// the position into `pos` / `head_pos` and the padded selection (window
    /// slots, then the shared index selection, then -1) into `idx_dev`, each
    /// only when it differs from what is already there. Returns the compressed
    /// rows the captured `sparse_attn` reads (`None` on ratio-0 layers) — a
    /// pointer the graph bakes, for the caller to verify on every replay.
    pub fn decode_prep(
        &mut self,
        w: &AttnV41LayerWeights,
        shared: &SharedV41,
        gpu: &dyn GpuBackend,
        start_pos: usize,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        let c = &self.cfg;
        ensure!(
            self.decode_capturable(w),
            "attn_v41: decode_prep on a kv/index source layer"
        );
        ensure!(
            start_pos >= 1 && start_pos < c.max_seq,
            "attn_v41: position {start_pos} outside the decode range 1..{}",
            c.max_seq
        );
        if self.decode_pos != Some(start_pos) {
            let p = start_pos as i32;
            upload_i32_async(gpu, self.pos, &[p], stream)?;
            upload_i32_async(gpu, self.head_pos, &vec![p; c.n_heads], stream)?;
            self.decode_pos = Some(start_pos);
        }
        let fixed = self.decode_fixed_topk(w);
        let (mut idx, topk) = window_topk_idxs(c.window, 1, start_pos);
        debug_assert_eq!(topk, c.window);
        let rows_b = if w.role.ratio > 0 {
            ensure!(
                shared.topk_idxs.len() == shared.topk,
                "attn_v41: shared index selection is {} for one token x {}",
                shared.topk_idxs.len(),
                shared.topk
            );
            idx.extend_from_slice(&shared.topk_idxs);
            Some(
                shared
                    .compress_kv
                    .context("compressed layer before any kv source published")?,
            )
        } else {
            None
        };
        ensure!(
            idx.len() <= fixed,
            "attn_v41: selection {} exceeds the fixed {fixed}",
            idx.len()
        );
        idx.resize(fixed, -1);
        if self.decode_idx.as_deref() != Some(&idx[..]) {
            upload_i32_async(gpu, self.idx_dev, &idx, stream)?;
            self.decode_idx = Some(idx);
        }
        Ok(rows_b)
    }

    /// The DEVICE half of a capturable layer's single-token step: the
    /// projections, the window ring write at `pos % window` (slot read on
    /// the device), sparse attention over the padded selection and the output
    /// projection into the returned buffer. No host work, no per-token
    /// scalar arguments: capturable into a CUDA graph and replayed for any
    /// position once `decode_prep` has updated `pos` / `head_pos` / `idx_dev`.
    pub fn decode_body(
        &self,
        gpu: &dyn GpuBackend,
        w: &AttnV41LayerWeights,
        st: &AttnV41LayerState,
        x: DevicePtr,
        rows_b: Option<DevicePtr>,
        stream: u64,
    ) -> Result<DevicePtr> {
        let c = &self.cfg;
        ensure!(
            self.decode_capturable(w),
            "attn_v41: decode_body on a kv/index source layer"
        );
        ensure!(
            (w.role.ratio > 0) == rows_b.is_some(),
            "attn_v41: compressed rows {} on a ratio-{} layer",
            if rows_b.is_some() { "given" } else { "missing" },
            w.role.ratio
        );
        let yarn = w.role.ratio > 0;
        self.q_kv_projections(gpu, w, x, 1, yarn, stream)?;
        KernelLaunch::new(gpu, self.k.ring_put)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.kv)
            .arg_ptr(st.window)
            .arg_ptr(self.pos)
            .arg_u32(c.window as u32)
            .arg_u32(c.head_dim as u32)
            .launch(stream)?;
        self.attend_and_project(
            gpu,
            w,
            st.window,
            c.window,
            rows_b,
            self.decode_fixed_topk(w),
            1,
            yarn,
            stream,
        )?;
        Ok(self.out)
    }

    /// One layer's attention for `m` tokens at `start_pos`: `x` is the normed
    /// input `[m, dim]` bf16; the output is `[m, dim]` bf16 in `run.out`.
    pub fn forward(
        &self,
        gpu: &dyn GpuBackend,
        w: &AttnV41LayerWeights,
        st: &mut AttnV41LayerState,
        shared: &mut SharedV41,
        x: DevicePtr,
        m: usize,
        start_pos: usize,
        stream: u64,
    ) -> Result<AttnV41Run> {
        let c = &self.cfg;
        ensure!(
            m >= 1 && m <= c.max_tokens,
            "attn_v41: {m} tokens outside 1..={}",
            c.max_tokens
        );
        ensure!(
            start_pos + m <= c.max_seq,
            "attn_v41: position {} beyond max_seq {}",
            start_pos + m,
            c.max_seq
        );
        let (nh, hd) = (c.n_heads, c.head_dim);
        let yarn = w.role.ratio > 0;
        let pos: Vec<i32> = (0..m).map(|t| (start_pos + t) as i32).collect();
        let hpos: Vec<i32> = pos
            .iter()
            .flat_map(|&p| std::iter::repeat_n(p, nh))
            .collect();
        upload_i32(gpu, self.pos, &pos)?;
        upload_i32(gpu, self.head_pos, &hpos)?;

        self.q_kv_projections(gpu, w, x, m, yarn, stream)?;

        // the window ring (stream-ordered copies, no host sync)
        let win = c.window;
        let row = hd * 2;
        let (rows_a, rows_a_len) = if start_pos == 0 {
            if m <= win {
                gpu.copy_d2d_async(self.kv, st.window, m * row, stream)?;
            } else {
                let cutoff = m % win;
                let tail = at(self.kv, (m - win) * row);
                gpu.copy_d2d_async(
                    tail,
                    at(st.window, cutoff * row),
                    (win - cutoff) * row,
                    stream,
                )?;
                gpu.copy_d2d_async(
                    at(tail, (win - cutoff) * row),
                    st.window,
                    cutoff * row,
                    stream,
                )?;
            }
            (self.kv, m)
        } else {
            let slot = start_pos % win;
            gpu.copy_d2d_async(self.kv, at(st.window, slot * row), row, stream)?;
            (st.window, win)
        };
        let (mut idx, mut topk) = window_topk_idxs(win, m, start_pos);

        let (mut rows_b, mut rows_b_len) = (None, 0usize);
        if w.role.ratio > 0 {
            let ratio = w.role.ratio;
            let offset = rows_a_len;
            let compress_len = (start_pos + m) / ratio;
            let latent_groups = if w.role.is_kv_source {
                let g = self.compressor(gpu, w, st, x, m, start_pos, stream)?;
                shared.compress_kv = st.compress_kv;
                g
            } else {
                None
            };
            let (cidx, ctopk) = if !w.role.is_index_source {
                (shared.topk_idxs.clone(), shared.topk)
            } else if compress_len == 0 {
                (Vec::new(), 0)
            } else {
                let r = self.indexer(
                    gpu,
                    w,
                    st,
                    shared,
                    x,
                    latent_groups,
                    m,
                    start_pos,
                    offset,
                    stream,
                )?;
                shared.topk_idxs = r.0.clone();
                shared.topk = r.1;
                r
            };
            if let Some(groups) = latent_groups {
                let cache = st.compress_kv.context("kv source without a latent cache")?;
                upload_i32(
                    gpu,
                    self.grp_pos,
                    &Self::group_positions(groups, ratio, start_pos),
                )?;
                self.rope(
                    gpu,
                    self.latent,
                    self.grp_pos,
                    groups,
                    hd,
                    true,
                    false,
                    stream,
                )?;
                self.fp4_quant(gpu, self.latent, groups * hd, LATENT_BLOCK, true, stream)?;
                gpu.copy_d2d_async(
                    self.latent,
                    at(cache, (start_pos / ratio) * row),
                    groups * row,
                    stream,
                )?;
                shared.compress_kv = Some(cache);
                shared.compress_len = shared.compress_len.max(start_pos / ratio + groups);
            }
            rows_b = Some(
                shared
                    .compress_kv
                    .context("compressed layer before any kv source published")?,
            );
            rows_b_len = compress_len;
            ensure!(
                cidx.len() == m * ctopk,
                "index selection is {} for {m} x {ctopk}",
                cidx.len()
            );
            let mut merged = Vec::with_capacity(m * (topk + ctopk));
            for t in 0..m {
                merged.extend_from_slice(&idx[t * topk..(t + 1) * topk]);
                merged.extend_from_slice(&cidx[t * ctopk..(t + 1) * ctopk]);
            }
            idx = merged;
            topk += ctopk;
        }
        ensure!(
            topk <= win + c.index_topk && topk <= 2048,
            "attn_v41: topk {topk} exceeds the workspace"
        );
        upload_i32(gpu, self.idx_dev, &idx)?;

        self.attend_and_project(gpu, w, rows_a, rows_a_len, rows_b, topk, m, yarn, stream)?;
        // no host sync here: everything downstream runs on the same stream,
        // and the routing download in the MoE block drains it before any
        // expert slot can be rewritten
        Ok(AttnV41Run {
            q: self.q,
            rows_a,
            rows_a_len,
            rows_b,
            rows_b_len,
            idx,
            topk,
            o: self.o,
            out: self.out,
        })
    }

    /// The layer output buffer, bf16 `[max_tokens, dim]`: a pointer a captured
    /// decode step bakes.
    pub fn out_ptr(&self) -> DevicePtr {
        self.out
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        for p in [
            self.fc_plain,
            self.fc_yarn,
            self.qr_raw,
            self.qr,
            self.q,
            self.kv_raw,
            self.kv,
            self.o,
            self.o_rot,
            self.og,
            self.slice_in,
            self.slice_out,
            self.out,
            self.pos,
            self.head_pos,
            self.idx_pos,
            self.grp_pos,
            self.idx_dev,
            self.ckv,
            self.cscore,
            self.pooled,
            self.latent_raw,
            self.latent,
            self.ik_raw,
            self.ik,
            self.iq,
            self.iw_raw,
            self.iw,
            self.score,
        ] {
            gpu.free(p)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "attn_v41_tests.rs"]
mod tests;
