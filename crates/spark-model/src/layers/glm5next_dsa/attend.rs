// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 DSA attention — the launcher for the selected-index NoPE MLA paged decode.
//!
//! Consumes what [`super::select`] produced: a `[q_rows, out_width]` i32 row of token
//! ids with `-1` holes. Gathers exactly those tokens from the paged FP8 latent cache.
//!
//! # Why this is not the masked kernel
//!
//! `dsa_mla_masked_attn` is an **oracle** (see [`super::MASKED_ATTN_MAX_KEYS`]). The
//! production path gathers per row, which is what HF says the reference *cannot* do
//! (`_supports_flash_attn = False`, "cannot be mapped to FA without a custom kernel that
//! can select on a per indices bases per row") and what vLLM ships. The gather is
//! **exactly equivalent**, not an approximation: the reference mask is pure set
//! membership, with duplicates collapsed and no additive weighting.
//!
//! # 🪤 NoPE, and why no `common/` kernel would do
//!
//! GLM-5.3 has `qk_rope_head_dim == 0`: the latent **is** the whole cache token.
//! `common/mla_paged_decode.cu` declares `kv_cache_dim` and never reads it (its strides
//! come from `#define ROPE_DIM 64`); `common/mla_paged_decode_fp8.cu` uses the runtime
//! stride but then overwrites dims 448–511 with rope taken from the *next* token. Both
//! fail silently. Hence a GLM-target kernel with no rope arm at all.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::{Glm5NextDsaConfig, select::DsaSelectGeometry};

/// Module name the DSA decode kernel resolves from — an unlisted `.cu` takes its file
/// stem, and this one lives in the `glm-5.3-flash` target, not `common/`.
pub const DSA_DECODE_MODULE: &str = "glm5next_dsa_mla_decode";

/// Threads per block: `NUM_WARPS * WARP_SIZE` in the kernel.
const DECODE_BLOCK: u32 = 256;

/// The selected-index MLA decode entry point.
#[derive(Clone, Copy)]
pub struct Glm5NextDsaDecodeKernel(KernelHandle);

impl Glm5NextDsaDecodeKernel {
    /// Resolved with `kernel()`, never `try_kernel`: a missing sparse decode entry point
    /// must be a hard error. Falling back to a dense path would be a correctness bug
    /// wearing a performance bug's clothes.
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self(gpu.kernel(
            DSA_DECODE_MODULE,
            "glm5next_dsa_mla_decode_fp8",
        )?))
    }
}

/// Everything the decode reads, all caller-owned.
#[derive(Debug, Clone, Copy)]
pub struct DsaDecodeInputs {
    /// `[num_q_heads * kv_lora_rank]` BF16 — this rank's absorbed queries.
    pub q: DevicePtr,
    /// FP8 paged latent cache. In absorbed NoPE MLA K and V are the **same** buffer;
    /// both are taken so a caller that splits them is not forced to lie.
    pub k_cache: DevicePtr,
    pub v_cache: DevicePtr,
    /// `[num_q_heads * kv_lora_rank]` BF16 output.
    pub out: DevicePtr,
    /// `[num_seqs, max_blocks_per_seq]` i32.
    pub block_tables: DevicePtr,
    /// `[num_seqs]` i32.
    pub seq_lens: DevicePtr,
    /// `[num_seqs, sel_width]` i32 — [`super::select::DsaSelectScratch::tokens`].
    pub sel_indices: DevicePtr,
    pub k_scale: f32,
    pub v_scale: f32,
}

/// Paging geometry the decode needs and the selection does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DsaDecodePaging {
    pub num_seqs: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub max_blocks_per_seq: usize,
    pub block_size: usize,
    pub cache_stride_bytes: u64,
}

impl DsaDecodePaging {
    /// 🪤 `block_size % index_kpool == 0` is required by the *selector*, not this kernel:
    /// pools are built over absolute positions, so a block that straddles a pool boundary
    /// makes a pool's tokens span two pages. The gather itself is per-token and would not
    /// notice — which is exactly why the check belongs here rather than nowhere.
    pub fn validate(&self, cfg: &Glm5NextDsaConfig) -> Result<()> {
        if self.num_seqs == 0 || self.num_q_heads == 0 {
            bail!(
                "DSA decode: degenerate launch ({} seqs, {} heads)",
                self.num_seqs,
                self.num_q_heads
            );
        }
        if self.block_size == 0 {
            bail!("DSA decode: block_size must be > 0");
        }
        if !self.block_size.is_multiple_of(cfg.index_kpool) {
            bail!(
                "DSA decode: block_size {} is not a multiple of index_kpool {} — a pool \
                 would straddle a page boundary",
                self.block_size,
                cfg.index_kpool
            );
        }
        if self.num_kv_heads != 1 {
            bail!(
                "DSA decode: MLA carries a single latent KV head, got {}",
                self.num_kv_heads
            );
        }
        Ok(())
    }
}

/// Launch the selected-index decode. Enqueued on `stream`, not synchronised.
///
/// `q_rows` in `geom` must equal `paging.num_seqs`: this is the decode path, one query
/// row per sequence. A mismatch would index the selection rows with the wrong stride,
/// so it is refused rather than trusted.
pub fn decode_attention(
    gpu: &dyn GpuBackend,
    kernel: Glm5NextDsaDecodeKernel,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    paging: &DsaDecodePaging,
    inputs: &DsaDecodeInputs,
    stream: u64,
) -> Result<()> {
    paging.validate(cfg)?;
    if geom.q_rows != paging.num_seqs {
        bail!(
            "DSA decode: selection has {} query rows but {} sequences are being decoded; \
             the selection row stride would be wrong",
            geom.q_rows,
            paging.num_seqs
        );
    }

    // The kernel tiles 512 latent dims across 32 lanes at 16 each. The host guard for
    // this is `Glm5NextDsaConfig::validate` (KERNEL_KV_LORA_DIM); restated at the launch
    // because a mismatch here is silent memory corruption, not an error.
    if cfg.kv_lora_rank != super::KERNEL_KV_LORA_DIM {
        bail!(
            "DSA decode: kv_lora_rank {} != kernel tiling {}",
            cfg.kv_lora_rank,
            super::KERNEL_KV_LORA_DIM
        );
    }

    KernelLaunch::new(gpu, kernel.0)
        .grid([paging.num_q_heads as u32, paging.num_seqs as u32, 1])
        .block([DECODE_BLOCK, 1, 1])
        .arg_ptr(inputs.q)
        .arg_ptr(inputs.k_cache)
        .arg_ptr(inputs.v_cache)
        .arg_ptr(inputs.out)
        .arg_ptr(inputs.block_tables)
        .arg_ptr(inputs.seq_lens)
        .arg_ptr(inputs.sel_indices)
        .arg_u32(geom.out_width as u32)
        .arg_u32(paging.max_blocks_per_seq as u32)
        .arg_u32(paging.num_q_heads as u32)
        .arg_u32(paging.num_kv_heads as u32)
        .arg_u32(cfg.kv_lora_rank as u32)
        .arg_u32(paging.block_size as u32)
        // 🪤 NoPE: the score scale is over the latent width, which IS the whole cache
        // token. DeepSeek-V4 divides by sqrt(576) because its token carries a rope tail.
        .arg_f32((cfg.kv_lora_rank as f32).powf(-0.5))
        .arg_f32(inputs.k_scale)
        .arg_f32(inputs.v_scale)
        .arg_u64(paging.cache_stride_bytes)
        .launch(stream)?;
    Ok(())
}

#[cfg(test)]
mod tests;
