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
//!
//! Split into submodules:
//!   - `init`: `AttnV41::new` (kernels, RoPE tables, workspaces) and `free`
//!   - `primitives`: the GEMM/GEMV, RMSNorm, RoPE and quantiser launch wrappers
//!   - `sources`: the kv-source compressor and the index-source indexer
//!   - `forward`: one layer's attention forward

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

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
    /// The eight output groups of `wo_a` in one launch (decode, m <= 8).
    mmvq_q2k_groups_w: KernelHandle,
    quant_d2s6: KernelHandle,
    mmq_q2k_nc: KernelHandle,
    mmq_q2k_wc: KernelHandle,
    rmsnorm_bf16: KernelHandle,
    rmsnorm_f32: KernelHandle,
    rope: KernelHandle,
    act_quant: KernelHandle,
    fp4_quant: KernelHandle,
    gemm_f32: KernelHandle,
    /// the compressor's f32 product at m = 1, staged, same bits as `gemm_f32`
    gemv_f32_staged: KernelHandle,
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
    /// The captured step's selection for ratio-0 layers (window only), so a
    /// multi-layer capture can hold both selection classes at once.
    idx_dev_win: DevicePtr,
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
    decode_idx_win: Option<Vec<i32>>,
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

mod decode;
mod forward;
mod init;
mod primitives;
mod sources;

#[cfg(test)]
#[path = "attn_v41_tests.rs"]
mod tests;
