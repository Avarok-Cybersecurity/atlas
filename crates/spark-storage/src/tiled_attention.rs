// SPDX-License-Identifier: AGPL-3.0-only
//
// Host wrapper for the tiled decode-attention kernels. Owns the (m, l, o)
// running-state device buffers and exposes the per-step lifecycle:
//
//   1. `begin_step()`   — reset state to (-INF, 0, 0).
//   2. `step_tile(...)` — one launch per tile of blocks.
//   3. `finalize(...)`  — divide o by l and store as BF16 output.
//
// The state buffers are sized for the maximum (num_seqs, num_q_heads,
// head_dim) the engine will see, so they can be reused across steps without
// reallocation.

use anyhow::{Context, Result, bail};
use std::ffi::c_void;

use crate::cuda_min::{CudaCtx, CudaModule, DeviceBuffer, launch_kernel};

include!(concat!(env!("OUT_DIR"), "/storage_ptx.rs"));

unsafe extern "C" {
    fn cuMemsetD32Async(dst: u64, value: u32, count: usize, stream: u64) -> i32;
}

#[derive(Clone, Copy, Debug)]
pub struct TiledAttentionDims {
    pub max_seqs: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    pub tile_capacity: usize,
}

impl TiledAttentionDims {
    pub fn validate(&self) -> Result<()> {
        if !self.num_q_heads.is_multiple_of(self.num_kv_heads) {
            bail!(
                "num_q_heads ({}) must divide num_kv_heads ({})",
                self.num_q_heads,
                self.num_kv_heads
            );
        }
        if self.head_dim > 256 {
            bail!("head_dim {} exceeds MAX_HEAD_DIM=256", self.head_dim);
        }
        Ok(())
    }
    pub fn gqa_ratio(&self) -> i32 {
        (self.num_q_heads / self.num_kv_heads) as i32
    }
    fn n_q_slots(&self) -> usize {
        self.max_seqs * self.num_q_heads
    }
    pub fn m_bytes(&self) -> usize {
        self.n_q_slots() * 4
    }
    pub fn l_bytes(&self) -> usize {
        self.n_q_slots() * 4
    }
    pub fn o_bytes(&self) -> usize {
        self.n_q_slots() * self.head_dim * 4
    }
    pub fn output_bytes(&self) -> usize {
        self.n_q_slots() * self.head_dim * 2
    }
}

pub struct TiledAttention {
    dims: TiledAttentionDims,
    _modules: Vec<CudaModule>,
    f_step: u64,
    f_finalize: u64,
    pub m_state: DeviceBuffer,
    pub l_state: DeviceBuffer,
    pub o_state: DeviceBuffer,
}

const NEG_INF_F32_BITS: u32 = 0xFF800000;

impl TiledAttention {
    pub fn new(dims: TiledAttentionDims) -> Result<Self> {
        dims.validate()?;
        let mut modules = Vec::new();
        let mut f_step = 0u64;
        let mut f_finalize = 0u64;
        for entry in STORAGE_PTX.iter() {
            match entry.name {
                "paged_decode_attn_tiled" => {
                    let m = CudaModule::from_ptx(entry.ptx)
                        .with_context(|| format!("load {}", entry.name))?;
                    f_step = m.function("paged_decode_attn_tiled")?;
                    modules.push(m);
                }
                "attention_finalize" => {
                    let m = CudaModule::from_ptx(entry.ptx)
                        .with_context(|| format!("load {}", entry.name))?;
                    f_finalize = m.function("attention_finalize")?;
                    modules.push(m);
                }
                _ => {}
            }
        }
        if f_step == 0 || f_finalize == 0 {
            bail!("tiled-attention PTX modules missing");
        }
        let m_state = DeviceBuffer::new(dims.m_bytes())?;
        let l_state = DeviceBuffer::new(dims.l_bytes())?;
        let o_state = DeviceBuffer::new(dims.o_bytes())?;
        Ok(Self {
            dims,
            _modules: modules,
            f_step,
            f_finalize,
            m_state,
            l_state,
            o_state,
        })
    }

    /// Reset (m, l, o) for `num_seqs` sequences. Call once at the start of
    /// each decode step before the first `step_tile`.
    pub fn begin_step(&self, ctx: &CudaCtx, num_seqs: usize) -> Result<()> {
        self.begin_step_on_stream(ctx.stream, num_seqs)
    }

    pub fn begin_step_on_stream(&self, stream: u64, num_seqs: usize) -> Result<()> {
        if num_seqs > self.dims.max_seqs {
            bail!(
                "begin_step num_seqs {} > max_seqs {}",
                num_seqs,
                self.dims.max_seqs
            );
        }
        let n_q = num_seqs * self.dims.num_q_heads;
        let n_o = n_q * self.dims.head_dim;
        unsafe {
            let s = cuMemsetD32Async(self.m_state.ptr, NEG_INF_F32_BITS, n_q, stream);
            if s != 0 {
                bail!("cuMemsetD32Async m_state failed: {s}");
            }
            let s = cuMemsetD32Async(self.l_state.ptr, 0, n_q, stream);
            if s != 0 {
                bail!("cuMemsetD32Async l_state failed: {s}");
            }
            let s = cuMemsetD32Async(self.o_state.ptr, 0, n_o, stream);
            if s != 0 {
                bail!("cuMemsetD32Async o_state failed: {s}");
            }
        }
        Ok(())
    }

    /// Stride triple for the kernel-native paged K/V layout:
    /// `[num_blocks, block_size, num_kv_heads, head_dim]`. All in BF16
    /// elements.
    pub fn paged_strides(&self) -> (i64, i64, i64) {
        let blk = (self.dims.block_size * self.dims.num_kv_heads * self.dims.head_dim) as i64;
        let tok = (self.dims.num_kv_heads * self.dims.head_dim) as i64;
        let kvh = self.dims.head_dim as i64;
        (blk, tok, kvh)
    }

    /// Stride triple for the scratch-pool slot layout
    /// `[slot, K|V, kv_head, block_size, head_dim]`. Reads of contiguous-
    /// per-(kv_head) groups stay efficient on disk; the kernel pays a single
    /// extra multiply per address calculation.
    pub fn scratch_pool_strides(&self) -> (i64, i64, i64) {
        let kv_stripe = (self.dims.num_kv_heads * self.dims.block_size * self.dims.head_dim) as i64;
        let blk = 2 * kv_stripe; // K stripes + V stripes per slot
        let tok = self.dims.head_dim as i64;
        let kvh = (self.dims.block_size * self.dims.head_dim) as i64;
        (blk, tok, kvh)
    }

    /// One tile of blocks across `num_seqs` sequences. Stride triple
    /// (`blk_stride`, `tok_stride`, `kvh_stride`) selects the K/V layout —
    /// see [`paged_strides`](Self::paged_strides) and
    /// [`scratch_pool_strides`](Self::scratch_pool_strides).
    #[allow(clippy::too_many_arguments)]
    pub fn step_tile(
        &self,
        ctx: &CudaCtx,
        q: u64,
        k_pool: u64,
        v_pool: u64,
        tile_blocks: u64,
        tile_block_counts: u64,
        num_seqs: usize,
        blk_stride: i64,
        tok_stride: i64,
        kvh_stride: i64,
        last_block_valid_slots: i32,
    ) -> Result<()> {
        self.step_tile_on_stream(
            ctx.stream,
            q,
            k_pool,
            v_pool,
            tile_blocks,
            tile_block_counts,
            num_seqs,
            blk_stride,
            tok_stride,
            kvh_stride,
            last_block_valid_slots,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_tile_on_stream(
        &self,
        stream: u64,
        q: u64,
        k_pool: u64,
        v_pool: u64,
        tile_blocks: u64,
        tile_block_counts: u64,
        num_seqs: usize,
        blk_stride: i64,
        tok_stride: i64,
        kvh_stride: i64,
        last_block_valid_slots: i32,
    ) -> Result<()> {
        let mut q_v = q;
        let mut k_v = k_pool;
        let mut v_v = v_pool;
        let mut tb = tile_blocks;
        let mut tc = tile_block_counts;
        let mut m_v = self.m_state.ptr;
        let mut l_v = self.l_state.ptr;
        let mut o_v = self.o_state.ptr;
        let mut nq = self.dims.num_q_heads as i32;
        let mut nk = self.dims.num_kv_heads as i32;
        let mut hd = self.dims.head_dim as i32;
        let mut bs = self.dims.block_size as i32;
        let mut tcap = self.dims.tile_capacity as i32;
        let mut gqa = self.dims.gqa_ratio();
        let mut blk_s = blk_stride;
        let mut tok_s = tok_stride;
        let mut kvh_s = kvh_stride;
        let mut lbvs = last_block_valid_slots;
        let mut params = [
            &mut q_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
            &mut v_v as *mut _ as *mut c_void,
            &mut tb as *mut _ as *mut c_void,
            &mut tc as *mut _ as *mut c_void,
            &mut m_v as *mut _ as *mut c_void,
            &mut l_v as *mut _ as *mut c_void,
            &mut o_v as *mut _ as *mut c_void,
            &mut nq as *mut _ as *mut c_void,
            &mut nk as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut tcap as *mut _ as *mut c_void,
            &mut gqa as *mut _ as *mut c_void,
            &mut blk_s as *mut _ as *mut c_void,
            &mut tok_s as *mut _ as *mut c_void,
            &mut kvh_s as *mut _ as *mut c_void,
            &mut lbvs as *mut _ as *mut c_void,
        ];
        launch_kernel(
            self.f_step,
            (num_seqs as u32, self.dims.num_q_heads as u32, 1),
            (self.dims.head_dim as u32, 1, 1),
            0,
            stream,
            &mut params,
        )
    }

    /// Divide o_state by l_state and store as BF16 in `output`.
    pub fn finalize(&self, ctx: &CudaCtx, output: u64, num_seqs: usize) -> Result<()> {
        self.finalize_on_stream(ctx.stream, output, num_seqs)
    }

    pub fn finalize_on_stream(&self, stream: u64, output: u64, num_seqs: usize) -> Result<()> {
        let mut l_v = self.l_state.ptr;
        let mut o_v = self.o_state.ptr;
        let mut out_v = output;
        let mut nq = self.dims.num_q_heads as i32;
        let mut hd = self.dims.head_dim as i32;
        let mut params = [
            &mut l_v as *mut _ as *mut c_void,
            &mut o_v as *mut _ as *mut c_void,
            &mut out_v as *mut _ as *mut c_void,
            &mut nq as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        launch_kernel(
            self.f_finalize,
            (num_seqs as u32, self.dims.num_q_heads as u32, 1),
            (self.dims.head_dim as u32, 1, 1),
            0,
            stream,
            &mut params,
        )
    }

    pub fn dims(&self) -> TiledAttentionDims {
        self.dims
    }
}
