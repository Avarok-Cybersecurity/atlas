// SPDX-License-Identifier: AGPL-3.0-only
//
// GPU-side predictor: loads the three PTX modules emitted by build.rs,
// owns persistent device buffers for `P` (projection) and `A_g` (anchor
// vectors per block × kv_head), and exposes three operations:
//
//   1. project_q       — Q @ P, per decode step
//   2. project_kv_block — A_g[block, kv_head] = mean_token(K_block @ P), at write time
//   3. score_blocks    — q_proj @ A_g_seq.T, max-reduced per block
//
// Lossless contract: scores are *advisory* (eviction priority + prefetch
// order). Mispredictions never gate attention correctness.

use anyhow::{Context, Result, bail};
use half::bf16;
use std::ffi::c_void;

use crate::cuda_min::{
    CudaCtx, CudaModule, DeviceBuffer, copy_d_to_h_async, copy_h_to_d_async, launch_kernel,
    stream_sync,
};
use crate::projection::{PredictorShape, build_projection};

include!(concat!(env!("OUT_DIR"), "/storage_ptx.rs"));

#[derive(Clone, Copy, Debug)]
pub struct PredictorDims {
    pub num_layers: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub r: usize,
    pub block_size: usize,
    pub max_blocks: usize,
}

impl PredictorDims {
    pub fn validate(&self) -> Result<()> {
        if !self.num_q_heads.is_multiple_of(self.num_kv_heads) {
            bail!(
                "num_q_heads ({}) must divide num_kv_heads ({})",
                self.num_q_heads,
                self.num_kv_heads
            );
        }
        Ok(())
    }
    pub fn gqa_ratio(&self) -> i32 {
        (self.num_q_heads / self.num_kv_heads) as i32
    }
    pub fn a_g_bytes(&self) -> usize {
        // K_lr is stored per-token: [num_layers, num_blocks, num_kv_heads, block_size, r]
        self.num_layers * self.max_blocks * self.num_kv_heads * self.block_size * self.r * 2
    }
    pub fn per_layer_block_floats(&self) -> usize {
        self.num_kv_heads * self.block_size * self.r
    }
    pub fn p_bytes(&self) -> usize {
        self.head_dim * self.r * 2
    }
}

pub struct Predictor {
    dims: PredictorDims,
    _modules: Vec<CudaModule>,
    f_q_proj: u64,
    f_kv_proj: u64,
    f_score: u64,
    p_dev: DeviceBuffer,   // [head_dim, r] BF16, immutable after init
    a_g_dev: DeviceBuffer, // [num_layers, max_blocks, num_kv_heads, r] BF16
}

impl Predictor {
    pub fn new(ctx: &CudaCtx, dims: PredictorDims, projection_seed: u64) -> Result<Self> {
        Self::new_on_stream(ctx.stream, dims, projection_seed)
    }

    pub fn new_on_stream(stream: u64, dims: PredictorDims, projection_seed: u64) -> Result<Self> {
        dims.validate()?;
        // Load only the predictor PTX modules; tiled-attention kernels are
        // owned by `TiledAttention` so each subsystem stays self-contained.
        let mut modules: Vec<CudaModule> = Vec::new();
        let mut f_q_proj = 0u64;
        let mut f_kv_proj = 0u64;
        let mut f_score = 0u64;
        for entry in STORAGE_PTX.iter() {
            match entry.name {
                "q_lowrank_project" | "kv_lowrank_project" | "predictor_score" => {
                    let m = CudaModule::from_ptx(entry.ptx)
                        .with_context(|| format!("load PTX module {}", entry.name))?;
                    match entry.name {
                        "q_lowrank_project" => f_q_proj = m.function("q_lowrank_project")?,
                        "kv_lowrank_project" => f_kv_proj = m.function("kv_lowrank_project")?,
                        "predictor_score" => f_score = m.function("predictor_score")?,
                        _ => unreachable!(),
                    }
                    modules.push(m);
                }
                _ => {} // tiled-attention modules handled elsewhere
            }
        }
        if f_q_proj == 0 || f_kv_proj == 0 || f_score == 0 {
            bail!("missing predictor kernel function — PTX list incomplete");
        }
        // Allocate and upload P.
        let shape = PredictorShape::new(dims.head_dim, dims.r);
        let p_host = build_projection(shape, projection_seed);
        let p_dev = DeviceBuffer::new(dims.p_bytes())?;
        copy_h_to_d_async(
            p_dev.ptr,
            p_host.as_ptr() as *const c_void,
            dims.p_bytes(),
            stream,
        )?;
        // Allocate A_g (zeroed initially via cuMemAlloc which doesn't zero — but
        // unwritten slots are unused; the predictor never reads a slot that
        // hasn't been populated by `project_kv_block`).
        let a_g_dev = DeviceBuffer::new(dims.a_g_bytes())?;
        stream_sync(stream)?;
        Ok(Self {
            dims,
            _modules: modules,
            f_q_proj,
            f_kv_proj,
            f_score,
            p_dev,
            a_g_dev,
        })
    }

    /// `q` device pointer to `[num_q_heads, head_dim]` BF16.
    /// `q_proj` device pointer to `[num_q_heads, r]` BF16 output.
    pub fn project_q(&self, ctx: &CudaCtx, q: u64, q_proj: u64) -> Result<()> {
        self.project_q_on_stream(ctx.stream, q, q_proj)
    }

    /// Stream-only variant for production callers that already own a CUDA
    /// context (and therefore don't need the test-side `CudaCtx` wrapper).
    pub fn project_q_on_stream(&self, stream: u64, q: u64, q_proj: u64) -> Result<()> {
        let mut q_v = q;
        let mut p_v = self.p_dev.ptr;
        let mut o_v = q_proj;
        let mut nq = self.dims.num_q_heads as i32;
        let mut hd = self.dims.head_dim as i32;
        let mut r = self.dims.r as i32;
        let mut params = [
            &mut q_v as *mut _ as *mut c_void,
            &mut p_v as *mut _ as *mut c_void,
            &mut o_v as *mut _ as *mut c_void,
            &mut nq as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut r as *mut _ as *mut c_void,
        ];
        launch_kernel(
            self.f_q_proj,
            (self.dims.num_q_heads as u32, 1, 1),
            (self.dims.r as u32, 1, 1),
            0,
            stream,
            &mut params,
        )
    }

    /// Update A_g for a single (layer, block_id) slot from the K data the
    /// caller just wrote into the KV cache. `k_block` device ptr to
    /// `[block_size, num_kv_heads, head_dim]` BF16.
    pub fn project_kv_block(
        &self,
        ctx: &CudaCtx,
        layer: usize,
        block_id: usize,
        k_block: u64,
    ) -> Result<()> {
        self.project_kv_block_on_stream(ctx.stream, layer, block_id, k_block)
    }

    pub fn project_kv_block_on_stream(
        &self,
        stream: u64,
        layer: usize,
        block_id: usize,
        k_block: u64,
    ) -> Result<()> {
        if layer >= self.dims.num_layers || block_id >= self.dims.max_blocks {
            bail!("project_kv_block out of range: layer {layer}, block {block_id}");
        }
        let slot_floats = self.dims.per_layer_block_floats();
        let k_lr_slot = self.a_g_dev.ptr
            + (((layer * self.dims.max_blocks + block_id) * slot_floats) * 2) as u64;
        let mut k_v = k_block;
        let mut p_v = self.p_dev.ptr;
        let mut o_v = k_lr_slot;
        let mut bs = self.dims.block_size as i32;
        let mut nk = self.dims.num_kv_heads as i32;
        let mut hd = self.dims.head_dim as i32;
        let mut r = self.dims.r as i32;
        let mut params = [
            &mut k_v as *mut _ as *mut c_void,
            &mut p_v as *mut _ as *mut c_void,
            &mut o_v as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut nk as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut r as *mut _ as *mut c_void,
        ];
        launch_kernel(
            self.f_kv_proj,
            (
                self.dims.num_kv_heads as u32,
                self.dims.block_size as u32,
                1,
            ),
            (self.dims.r as u32, 1, 1),
            0,
            stream,
            &mut params,
        )
    }

    /// Score `num_active_blocks` (already-laid-out) blocks for the current
    /// layer. `q_proj` device ptr to `[num_q_heads, r]` BF16. `a_g_seq` is a
    /// device ptr to the active sequence's per-block anchors at this layer
    /// (`[num_active_blocks, num_kv_heads, r]` BF16). `scores_out` is a
    /// device ptr to a `[num_active_blocks]` f32 buffer.
    pub fn score_blocks(
        &self,
        ctx: &CudaCtx,
        q_proj: u64,
        k_lr_seq: u64,
        scores_out: u64,
        num_active_blocks: usize,
    ) -> Result<()> {
        self.score_blocks_on_stream(ctx.stream, q_proj, k_lr_seq, scores_out, num_active_blocks)
    }

    pub fn score_blocks_on_stream(
        &self,
        stream: u64,
        q_proj: u64,
        k_lr_seq: u64,
        scores_out: u64,
        num_active_blocks: usize,
    ) -> Result<()> {
        let mut q_v = q_proj;
        let mut a_v = k_lr_seq;
        let mut s_v = scores_out;
        let mut nq = self.dims.num_q_heads as i32;
        let mut nk = self.dims.num_kv_heads as i32;
        let mut bs = self.dims.block_size as i32;
        let mut r = self.dims.r as i32;
        let mut gqa = self.dims.gqa_ratio();
        let mut params = [
            &mut q_v as *mut _ as *mut c_void,
            &mut a_v as *mut _ as *mut c_void,
            &mut s_v as *mut _ as *mut c_void,
            &mut nq as *mut _ as *mut c_void,
            &mut nk as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut r as *mut _ as *mut c_void,
            &mut gqa as *mut _ as *mut c_void,
        ];
        launch_kernel(
            self.f_score,
            (num_active_blocks as u32, 1, 1),
            (128, 1, 1),
            0,
            stream,
            &mut params,
        )
    }

    pub fn dims(&self) -> PredictorDims {
        self.dims
    }
    pub fn a_g_dev_ptr(&self) -> u64 {
        self.a_g_dev.ptr
    }
}

/// Helper for tests / debug: copy the predictor's K_lr slot for a given
/// (layer, block) to host as BF16. Layout `[num_kv_heads, block_size, r]`.
pub fn read_k_lr_slot(
    ctx: &CudaCtx,
    pred: &Predictor,
    layer: usize,
    block: usize,
) -> Result<Vec<bf16>> {
    let dims = pred.dims();
    let n = dims.per_layer_block_floats();
    let slot = pred.a_g_dev.ptr + (((layer * dims.max_blocks + block) * n) * 2) as u64;
    let mut host = vec![bf16::from_f32(0.0); n];
    copy_d_to_h_async(host.as_mut_ptr() as *mut c_void, slot, n * 2, ctx.stream)?;
    stream_sync(ctx.stream)?;
    Ok(host)
}
