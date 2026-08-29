// SPDX-License-Identifier: AGPL-3.0-only

//! `Glm5NextDsaLayer` — the DSA block: NoPE MLA attention over indexer-selected tokens.
//!
//! Decode, end to end:
//!
//! ```text
//! hidden ─┬─ q_a_proj ─ RMSNorm ─┬─ q_absorb ────────────── Q (latent space)
//!         │                      └─ indexer.wq_b ────────── q_idx  ─┐
//!         ├─ indexer.wk ─ LayerNorm(w,b) ─ state.k_normed ──────────┤
//!         ├─ compress_gate ─────────────── state.gate ──────────────┼─ select_tokens
//!         ├─ weights_proj ──────────────── head weights ────────────┘        │
//!         └─ kv_a_proj ─ RMSNorm ─ FP8 ─── paged latent cache                │
//!                                                                            ▼
//!                                            glm5next_dsa_mla_decode_fp8 (gather)
//! ```
//!
//! # 🪤 Four silent-wrong-answer traps this file exists to hold
//!
//! * **Two RMSNorm kernels differ only by a `+1`.** `rms_norm` computes
//!   `x * rms * (1 + w)`; `rms_norm_vanilla` computes `x * rms * w`. Same signature, same
//!   shapes. GLM is plain, so every norm here takes the *vanilla* entry point.
//! * **`indexer.k_norm` is an `nn.LayerNorm` with a bias**, not an RMSNorm at all — mean
//!   subtraction plus a bias term. It takes `nllb_layernorm_bf16(x, w, b, …)`.
//! * **`weights_proj` output must already carry `index_heads^-0.5`.** `dsa_index_scores`
//!   does not apply it. Folded into the weight at load — see [`Glm5NextDsaWeights`].
//! * **Q must be absorbed into latent space before it reaches the decode kernel.** The
//!   kernel dots Q against the 512-dim latent directly, so `q_absorb` is `q_b_proj`
//!   pre-multiplied by `kv_b_proj`'s K half. A raw `q_b_proj` is the right shape per head
//!   (256 vs 512 is not) but the wrong space.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::PagedKvCache;

use super::attend::{DsaDecodeInputs, DsaDecodePaging, Glm5NextDsaDecodeKernel, decode_attention};
use super::select::{DsaSelectInputs, DsaSelectScratch, select_tokens};
use super::state::Glm5NextDsaState;
use super::{Glm5NextDsaConfig, Glm5NextDsaKernels};
use crate::layer::{ForwardContext, LayerState, TransformerLayer};

const GEMM_TILE: u32 = 16;

/// GEMM launch: `C[M, N] = A[M, K] @ B[N, K]^T`. Grid `(ceil(N/16), ceil(M/16))`,
/// block `(16, 16)` — one thread per output element.
fn gemm(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([
            (n as u32).div_ceil(GEMM_TILE),
            (m as u32).div_ceil(GEMM_TILE),
            1,
        ])
        .block([GEMM_TILE, GEMM_TILE, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .launch(stream)?;
    Ok(())
}

/// Every kernel a DSA block launches, beyond the selection set.
#[derive(Clone, Copy)]
pub struct Glm5NextDsaLayerKernels {
    /// `C = A @ B^T`, BF16 out.
    pub gemm: KernelHandle,
    /// Same, FP32 out — the selector wants `q_idx` and the head weights in FP32.
    pub gemm_f32: KernelHandle,
    /// 🪤 **vanilla** = `x * rms * w`. The other `rms_norm` adds 1 to the weight.
    pub rms_norm: KernelHandle,
    /// RMSNorm + FP8 + paged slot write, GLM-target.
    pub latent_write: KernelHandle,
}

impl Glm5NextDsaLayerKernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            // 🪤 Module is "gemm", NOT the file stem. `common/KERNEL.toml` [modules] maps
            // `dense_gemm_bf16 = "gemm"`, and an unlisted .cu takes its stem — so the two
            // conventions coexist and only the TOML says which applies. Guessing the stem
            // here resolved to nothing and would have failed at first construction.
            gemm: gpu.kernel("gemm", "dense_gemm_bf16")?,
            gemm_f32: gpu.kernel("gemm", "dense_gemm_bf16_f32out")?,
            rms_norm: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            latent_write: gpu
                .kernel("glm5next_mla_latent_write", "glm5next_mla_latent_write_fp8")?,
        })
    }
}

/// One DSA block's weights, already sharded for this rank.
pub struct Glm5NextDsaWeights {
    // ── MLA ──
    pub q_a_proj: DevicePtr,
    pub q_a_layernorm: DevicePtr,
    /// `[local_heads * kv_lora_rank, q_lora_rank]` BF16 — `q_b_proj` **absorbed** through
    /// `kv_b_proj`'s K half, so Q arrives in latent space. See the module header.
    pub q_absorb: DevicePtr,
    pub kv_a_proj: DevicePtr,
    pub kv_a_layernorm: DevicePtr,
    /// `[hidden, local_heads * kv_lora_rank]` BF16, row-parallel — all-reduced by the caller.
    ///
    /// 🪤 **Absorbed**, not the raw checkpoint `o_proj`: the decode kernel leaves its output
    /// in the 512-dim LATENT space, so the projection carries `kv_b_proj`'s V half folded in.
    /// The raw weight is `local_heads * v_head_dim` wide — half of this — and feeding the
    /// latent to it reads 2x past every row rather than merely computing the wrong thing.
    pub o_absorb: DevicePtr,
    // ── indexer (REPLICATED across ranks; see `tp`) ──
    pub wk: DevicePtr,
    pub k_norm_weight: DevicePtr,
    /// 🪤 REQUIRED. `k_norm` is a LayerNorm; a `.weight`-only bind silently drops the
    /// mean subtraction and the bias.
    pub k_norm_bias: DevicePtr,
    pub compress_gate: DevicePtr,
    pub wq_b: DevicePtr,
    /// 🪤 Pre-multiplied by `index_heads^-0.5` at load — `dsa_index_scores` does not scale.
    pub weights_proj: DevicePtr,
    /// `[index_kpool, index_head_dim]` **FP32**. 🪤 BF16 on disk; upconverted at load.
    pub ape: DevicePtr,
}

/// Scratch reused across decode steps. Allocated once per layer.
pub struct Glm5NextDsaWorkspace {
    q_a: DevicePtr,
    q_resid: DevicePtr,
    q_abs: DevicePtr,
    kv_a: DevicePtr,
    q_idx: DevicePtr,
    head_weights: DevicePtr,
    q_pos: DevicePtr,
    q_mask: DevicePtr,
    slot: DevicePtr,
    attn_out: DevicePtr,
    select: DsaSelectScratch,
}

impl Glm5NextDsaWorkspace {
    pub fn new(gpu: &dyn GpuBackend, cfg: &Glm5NextDsaConfig) -> Result<Self> {
        // Sized at the DSA context cap so a growing sequence never reallocates.
        let geom =
            super::select::DsaSelectGeometry::plan(cfg, super::state::max_dsa_context(cfg), 1)?;
        Ok(Self {
            q_a: gpu.alloc(cfg.q_lora_rank * 2)?,
            q_resid: gpu.alloc(cfg.q_lora_rank * 2)?,
            q_abs: gpu.alloc(cfg.local_heads * cfg.kv_lora_rank * 2)?,
            kv_a: gpu.alloc(cfg.kv_lora_rank * 2)?,
            q_idx: gpu.alloc(cfg.index_heads * cfg.index_head_dim * 4)?,
            head_weights: gpu.alloc(cfg.index_heads * 4)?,
            q_pos: gpu.alloc(4)?,
            q_mask: gpu.alloc(1)?,
            slot: gpu.alloc(8)?,
            attn_out: gpu.alloc(cfg.local_heads * cfg.kv_lora_rank * 2)?,
            select: DsaSelectScratch::alloc(gpu, cfg, &geom)?,
        })
    }
}

pub struct Glm5NextDsaLayer {
    pub cfg: Glm5NextDsaConfig,
    pub weights: Glm5NextDsaWeights,
    pub kernels: Glm5NextDsaLayerKernels,
    pub select_kernels: Glm5NextDsaKernels,
    pub decode_kernel: Glm5NextDsaDecodeKernel,
    pub workspace: Glm5NextDsaWorkspace,
    /// Index in the MODEL stack (0..num_hidden_layers). Diagnostics only.
    pub layer_idx: usize,
    /// Index in the KV POOL — the running ordinal over KV-cache-consuming layers,
    /// which for GLM-5.3 is 0..11 over the sparse layers, not 0..45.
    ///
    /// 🪤 These two are NOT interchangeable. The pool is sized to
    /// `ModelConfig::num_attention_layers()`; indexing it with `layer_idx` reads
    /// past the end of the allocation on every layer after the first.
    pub attn_layer_idx: usize,
    pub rms_eps: f32,
    /// FP8 latent-cache scale. Reads and writes must agree; the write takes `1/scale`.
    pub kv_scale: f32,
}

impl Glm5NextDsaLayer {
    /// Project `hidden` into the indexer cache at position `pos`, then advance.
    ///
    /// Writes `k_normed` and `gate` **directly into the state rows** rather than through a
    /// staging buffer: the selector reads `k[raw * D + d]` over the whole context, so the
    /// cache is the natural destination and a copy would buy nothing.
    pub fn indexer_forward(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        state: &mut Glm5NextDsaState,
        stream: u64,
    ) -> Result<()> {
        let d = self.cfg.index_head_dim;
        let pos = state.len();
        let off = state.row_offset(pos);

        // k_raw -> the state row, then LayerNorm in place.
        gemm(
            gpu,
            self.kernels.gemm,
            hidden,
            self.weights.wk,
            state.k_normed.offset(off),
            1,
            d,
            self.cfg.hidden,
            stream,
        )?;
        // 🪤 LayerNorm WITH BIAS, in place, one row.
        KernelLaunch::new(gpu, self.select_kernels.k_norm)
            .grid([1, 1, 1])
            .block([d.min(1024) as u32, 1, 1])
            .shared_mem((d.min(1024) * 4) as u32)
            .arg_ptr(state.k_normed.offset(off))
            .arg_ptr(self.weights.k_norm_weight)
            .arg_ptr(self.weights.k_norm_bias)
            .arg_u32(1)
            .arg_u32(d as u32)
            .arg_f32(self.rms_eps)
            .launch(stream)?;

        gemm(
            gpu,
            self.kernels.gemm,
            hidden,
            self.weights.compress_gate,
            state.gate.offset(off),
            1,
            d,
            self.cfg.hidden,
            stream,
        )?;

        // Per-head selector weights, FP32 straight out of the GEMM, from the LAYER INPUT.
        // `weights_proj` is `[index_heads, hidden]` and the reference is
        // `weights_proj(hidden) * index_heads**-0.5`, with the scale already folded into the
        // weight at load (`build.rs` transform 2). Computed here rather than in
        // `select_and_attend` for the plain reason that this is the function that HAS
        // `hidden`; `select_and_attend` does not, which is how it came to read `q_resid`
        // instead and overrun it by 5120 bytes. See A55.
        gemm(
            gpu,
            self.kernels.gemm_f32,
            hidden,
            self.weights.weights_proj,
            self.workspace.head_weights,
            1,
            self.cfg.index_heads,
            self.cfg.hidden,
            stream,
        )?;

        // Validity is per position and this one is real.
        gpu.memset_async(state.valid.offset(pos), 1, 1, stream)?;
        state.advance(1)
    }

    /// Everything after the indexer write: selector inputs, selection, gather-attend.
    /// Leaves `[local_heads, kv_lora_rank]` BF16 in the workspace's `attn_out`.
    fn select_and_attend(
        &self,
        gpu: &dyn GpuBackend,
        state: &Glm5NextDsaState,
        kv_cache: &PagedKvCache,
        block_table_dev: DevicePtr,
        seq_lens_dev: DevicePtr,
        paging: &DsaDecodePaging,
        stream: u64,
    ) -> Result<DevicePtr> {
        let w = &self.workspace;
        let geom = state.geometry(&self.cfg, 1)?;

        // Selector Q and head weights, FP32 straight out of the GEMM.
        gemm(
            gpu,
            self.kernels.gemm_f32,
            w.q_resid,
            self.weights.wq_b,
            w.q_idx,
            1,
            self.cfg.index_heads * self.cfg.index_head_dim,
            self.cfg.q_lora_rank,
            stream,
        )?;
        // 🔴 `head_weights` is NOT computed here any more — see `indexer_forward`. It used to
        // be, from `w.q_resid` with `K = cfg.hidden`, which was wrong twice over: the
        // reference projects the LAYER INPUT (`gen_dsa_indexer_golden.py`:
        // `weights_proj(hidden) * NH**-0.5`), and `q_resid` is only `[q_lora_rank] = 1536`
        // BF16, so reading 4096 of them ran **5120 bytes past the end of the allocation**.
        // That out-of-bounds read was ANOMALIES A55: the head weights were a function of
        // whatever the allocator had placed after `q_resid`, which is why the model's output
        // moved when the heap moved, when allocations were zeroed, and when an unrelated
        // buffer was added. Found by red-zoning the allocator and bisecting the guard bands.

        let inputs = DsaSelectInputs {
            k_normed: state.k_normed,
            gate: state.gate,
            valid: state.valid,
            ape: self.weights.ape,
            q: w.q_idx,
            weights: w.head_weights,
            q_pos: w.q_pos,
            q_mask: w.q_mask,
            first_key: 0,
        };
        select_tokens(
            gpu,
            &self.select_kernels,
            &self.cfg,
            &geom,
            &inputs,
            &w.select,
            stream,
        )?;

        let pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
        decode_attention(
            gpu,
            self.decode_kernel,
            &self.cfg,
            &geom,
            paging,
            &DsaDecodeInputs {
                q: w.q_abs,
                k_cache: pool,
                v_cache: pool, // absorbed NoPE MLA: K and V are the same latent
                out: w.attn_out,
                block_tables: block_table_dev,
                seq_lens: seq_lens_dev,
                sel_indices: w.select.tokens(),
                k_scale: self.kv_scale,
                v_scale: self.kv_scale,
            },
            stream,
        )?;
        Ok(w.attn_out)
    }
}

impl TransformerLayer for Glm5NextDsaLayer {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(Glm5NextDsaState::alloc(gpu, &self.cfg)?))
    }

    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let st = state
            .as_any_mut()
            .downcast_mut::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!("Glm5NextDsaLayer got a state that is not Glm5NextDsaState")
            })?;
        if st.len() != seq_len {
            bail!(
                "DSA layer {}: indexer cache holds {} tokens but the sequence is at {}. \
                 The indexer stream must advance in lockstep with the KV cache — a drift \
                 selects over the wrong context.",
                self.layer_idx,
                st.len(),
                seq_len
            );
        }
        let gpu = ctx.gpu;
        let w = &self.workspace;

        // ── q path ──
        gemm(
            gpu,
            self.kernels.gemm,
            hidden,
            self.weights.q_a_proj,
            w.q_a,
            1,
            self.cfg.q_lora_rank,
            self.cfg.hidden,
            stream,
        )?;
        // 🪤 vanilla: x * rms * w, no `1 +`.
        KernelLaunch::new(gpu, self.kernels.rms_norm)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(w.q_a)
            .arg_ptr(self.weights.q_a_layernorm)
            .arg_ptr(w.q_resid)
            .arg_u32(self.cfg.q_lora_rank as u32)
            .arg_f32(self.rms_eps)
            .launch(stream)?;
        // Q absorbed into latent space in one GEMM.
        gemm(
            gpu,
            self.kernels.gemm,
            w.q_resid,
            self.weights.q_absorb,
            w.q_abs,
            1,
            self.cfg.local_heads * self.cfg.kv_lora_rank,
            self.cfg.q_lora_rank,
            stream,
        )?;

        // ── kv path: latent -> FP8 -> paged slot ──
        gemm(
            gpu,
            self.kernels.gemm,
            hidden,
            self.weights.kv_a_proj,
            w.kv_a,
            1,
            self.cfg.kv_lora_rank,
            self.cfg.hidden,
            stream,
        )?;
        let block_size = kv_cache.config().block_size;
        let logical = seq_len / block_size;
        let physical = *block_table.get(logical).ok_or_else(|| {
            anyhow::anyhow!(
                "DSA layer {}: block table has {} entries, needs logical block {logical} \
                 for position {seq_len}",
                self.layer_idx,
                block_table.len()
            )
        })? as usize;
        let slot = (physical * block_size + seq_len % block_size) as i64;
        gpu.copy_h2d(&slot.to_le_bytes(), w.slot)?;
        KernelLaunch::new(gpu, self.kernels.latent_write)
            .grid([1, 1, 1])
            .block([self.cfg.kv_lora_rank as u32, 1, 1])
            .arg_ptr(w.kv_a)
            .arg_ptr(self.weights.kv_a_layernorm)
            .arg_ptr(kv_cache.k_pool_ptr(self.attn_layer_idx))
            .arg_ptr(w.slot)
            .arg_u32(self.cfg.kv_lora_rank as u32)
            .arg_f32(self.rms_eps)
            .arg_f32(1.0 / self.kv_scale)
            .launch(stream)?;

        // ── indexer stream, then select + gather-attend ──
        self.indexer_forward(gpu, hidden, st, stream)?;

        let pos = seq_len as i32;
        gpu.copy_h2d(&pos.to_le_bytes(), w.q_pos)?;
        gpu.copy_h2d(&[1u8], w.q_mask)?;
        let bt: Vec<u8> = block_table.iter().flat_map(|b| b.to_le_bytes()).collect();
        let d_bt = gpu.alloc(bt.len().max(4))?;
        gpu.copy_h2d(&bt, d_bt)?;
        let d_sl = gpu.alloc(4)?;
        gpu.copy_h2d(&((seq_len + 1) as i32).to_le_bytes(), d_sl)?;

        let paging = DsaDecodePaging {
            num_seqs: 1,
            num_q_heads: self.cfg.local_heads,
            num_kv_heads: 1,
            max_blocks_per_seq: block_table.len(),
            block_size,
            cache_stride_bytes: (block_size * self.cfg.kv_lora_rank) as u64,
        };
        let attn = self.select_and_attend(gpu, st, kv_cache, d_bt, d_sl, &paging, stream)?;
        gpu.free(d_bt)?;
        gpu.free(d_sl)?;

        // ── output projection, row-parallel: the caller all-reduces ──
        gemm(
            gpu,
            self.kernels.gemm,
            attn,
            self.weights.o_absorb,
            hidden,
            1,
            self.cfg.hidden,
            self.cfg.local_heads * self.cfg.kv_lora_rank,
            stream,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
