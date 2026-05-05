// SPDX-License-Identifier: AGPL-3.0-only

//! Paged KV cache block allocator.
//!
//! Manages a pool of fixed-size blocks for attention KV storage.
//! Each block holds `block_size` token positions for all KV heads.

use crate::gpu::DevicePtr;
use anyhow::{Result, bail};

pub(crate) const NVFP4_GROUP_SIZE: usize = 16;

/// KV cache quantization dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvCacheDtype {
    /// 2 bytes per element.
    Bf16,
    /// 1 byte per element (FP8 E4M3 with per-tensor scale).
    Fp8,
    /// 0.5 bytes data + per-group FP8 scale (E2M1 packed nibbles).
    Nvfp4,
    /// 4-bit WHT + Lloyd-Max quantization (TurboQuant). Same byte layout as NVFP4
    /// but with Walsh-Hadamard rotation and optimal Gaussian codebook for ~2x
    /// lower MSE at the same bit rate.
    Turbo4,
    /// 3-bit WHT + Lloyd-Max (8 levels). 22% smaller than turbo4.
    Turbo3,
    /// WHT + FP8 E4M3. Same memory as FP8 but with outlier suppression.
    /// Enables FP8-level memory for models with large RMS norm weights.
    Turbo8,
}

impl std::fmt::Display for KvCacheDtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvCacheDtype::Bf16 => write!(f, "bf16"),
            KvCacheDtype::Fp8 => write!(f, "fp8"),
            KvCacheDtype::Nvfp4 => write!(f, "nvfp4"),
            KvCacheDtype::Turbo4 => write!(f, "turbo4"),
            KvCacheDtype::Turbo3 => write!(f, "turbo3"),
            KvCacheDtype::Turbo8 => write!(f, "turbo8"),
        }
    }
}

impl std::str::FromStr for KvCacheDtype {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "bf16" => Ok(KvCacheDtype::Bf16),
            "fp8" => Ok(KvCacheDtype::Fp8),
            "nvfp4" => Ok(KvCacheDtype::Nvfp4),
            "turbo4" => Ok(KvCacheDtype::Turbo4),
            "turbo3" => Ok(KvCacheDtype::Turbo3),
            "turbo8" => Ok(KvCacheDtype::Turbo8),
            other => bail!(
                "Unsupported --kv-cache-dtype '{other}'. Use 'fp8', 'bf16', 'nvfp4', 'turbo4', 'turbo3', or 'turbo8'."
            ),
        }
    }
}

/// Configuration for the paged KV cache.
pub struct KvCacheConfig {
    /// Tokens per block.
    pub block_size: usize,
    /// Number of KV heads.
    pub num_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Number of attention layers (only full_attention layers have KV cache).
    pub num_layers: usize,
    /// Quantization dtype for cache storage (uniform fallback).
    pub dtype: KvCacheDtype,
    /// Per-layer KV cache dtype override. When non-empty, `layer_dtypes[i]`
    /// specifies the dtype for attention layer `i`. When empty, all layers
    /// use the uniform `dtype` field (backward compatible).
    pub layer_dtypes: Vec<KvCacheDtype>,
    /// Per-layer (num_kv_heads, head_dim) overrides for heterogeneous
    /// attention models (e.g. Gemma-4 with sliding 16×256 vs full 4×512).
    /// When non-empty, `layer_dims[i]` specifies the (nkv, hd) for layer
    /// `i`; allocation and kernel stride computations use these per-layer
    /// values so writes/reads land at the correct offsets. When empty,
    /// all layers use the uniform `num_kv_heads`/`head_dim` (backward
    /// compatible — homogeneous models need no change).
    pub layer_dims: Vec<(usize, usize)>,
    /// `--high-speed-swap` HBM-shrink knob (Phase 6.1). When `Some(N)`,
    /// each sequence is capped at `N` HBM-resident blocks; older blocks
    /// are evicted to disk via `HighSpeedSwap` and read back on demand.
    /// `None` (default) preserves the existing behavior — sequences hold
    /// every block in HBM forever and rely on `--swap-space-gb` for
    /// admission control. The `try_evict_oldest_for_seq` helper below is
    /// only valid when this is `Some`.
    pub cache_blocks_per_seq: Option<u32>,
}

impl KvCacheConfig {
    /// Resolve the effective dtype for a given attention layer index.
    pub fn dtype_for_layer(&self, layer_idx: usize) -> KvCacheDtype {
        if layer_idx < self.layer_dtypes.len() {
            self.layer_dtypes[layer_idx]
        } else {
            self.dtype
        }
    }

    /// Bytes per block (K or V, not both) for a specific dtype and
    /// (num_kv_heads, head_dim) pair. Per-layer callers pass their layer's
    /// actual dimensions; homogeneous callers pass the global values.
    fn block_bytes_dims(&self, dtype: KvCacheDtype, nkv: usize, hd: usize) -> usize {
        let elems = self.block_size * nkv * hd;
        match dtype {
            KvCacheDtype::Bf16 => elems * 2,
            KvCacheDtype::Fp8 => elems,
            KvCacheDtype::Nvfp4 | KvCacheDtype::Turbo4 => {
                // Both NVFP4 and Turbo4 use 4-bit data + FP8 per-group scales.
                // Same byte layout, different codebook (E2M1 vs Lloyd-Max).
                let data = elems / 2; // 2 nibbles per byte
                let num_groups = elems / NVFP4_GROUP_SIZE;
                data + num_groups // +1 FP8 scale byte per group
            }
            KvCacheDtype::Turbo3 => {
                // 3-bit WHT + Lloyd-Max (8 levels). Packed: 8 values in 3 bytes.
                let data = elems * 3 / 8;
                let num_groups = elems / NVFP4_GROUP_SIZE;
                data + num_groups
            }
            KvCacheDtype::Turbo8 => {
                // WHT + FP8 E4M3 data + per-group BF16 scales.
                // 2026-04-28: scales upgraded from FP8 (1 byte) to BF16 (2 bytes)
                // because FP8's ~12% per-scale relative error compounds
                // catastrophically across MiniMax M2.7's 58 Turbo8 layers
                // (gibberish output). BF16 scales (~0.4% relative error)
                // keep compounding tractable. ~6% extra cache memory for
                // a 256× precision improvement on the per-group scaling.
                let num_groups = elems / NVFP4_GROUP_SIZE;
                elems + num_groups * 2 // 1 byte data + BF16 scale per group
            }
        }
    }

    /// Legacy name: bytes per block using global dims and a given dtype.
    /// Used when the caller has a uniform KV geometry; prefer
    /// `block_bytes_for_layer` for layer-aware paths.
    fn block_bytes_for_dtype(&self, dtype: KvCacheDtype) -> usize {
        self.block_bytes_dims(dtype, self.num_kv_heads, self.head_dim)
    }

    /// Bytes per block per layer (K or V, not both), using the uniform dtype.
    pub fn block_bytes(&self) -> usize {
        self.block_bytes_for_dtype(self.dtype)
    }

    /// (num_kv_heads, head_dim) for a specific attention layer.
    /// Falls back to the global values when no per-layer override is set.
    pub fn dims_for_layer(&self, layer_idx: usize) -> (usize, usize) {
        if layer_idx < self.layer_dims.len() {
            self.layer_dims[layer_idx]
        } else {
            (self.num_kv_heads, self.head_dim)
        }
    }

    /// Bytes per block for a specific attention layer (K or V, not both).
    /// Uses per-layer (nkv, hd) from `layer_dims` when set — this lets
    /// heterogeneous layers (Gemma-4 sliding vs full) allocate tight,
    /// correctly-sized pools so the kernel's per-layer stride matches
    /// the allocation layout.
    pub fn block_bytes_for_layer(&self, layer_idx: usize) -> usize {
        let (nkv, hd) = self.dims_for_layer(layer_idx);
        self.block_bytes_dims(self.dtype_for_layer(layer_idx), nkv, hd)
    }

    /// Bytes per block per layer (K + V combined).
    pub fn block_bytes_kv(&self) -> usize {
        self.block_bytes() * 2
    }

    /// Sum of K+V block bytes across all layers for one block slot.
    /// Accounts for mixed dtypes when layer_dtypes is set.
    pub fn block_bytes_kv_all_layers(&self) -> usize {
        if self.layer_dtypes.is_empty() {
            self.block_bytes_kv() * self.num_layers
        } else {
            (0..self.num_layers)
                .map(|i| self.block_bytes_for_layer(i) * 2)
                .sum()
        }
    }

    /// Cache stride in elements (for FP8/BF16 kernels).
    /// Elements per block = block_size * num_kv_heads * head_dim.
    pub fn cache_stride_elements(&self) -> usize {
        self.block_size * self.num_kv_heads * self.head_dim
    }

    /// NVFP4 data section bytes per block (packed E2M1 nibbles).
    pub fn nvfp4_data_bytes(&self) -> usize {
        self.block_size * self.num_kv_heads * self.head_dim / 2
    }

    /// NVFP4 scale section bytes per block (FP8 per-group scales).
    pub fn nvfp4_scale_bytes(&self) -> usize {
        self.block_size * self.num_kv_heads * self.head_dim / NVFP4_GROUP_SIZE
    }

    /// Turbo4 data section bytes (same layout as NVFP4: 4-bit packed).
    pub fn turbo4_data_bytes(&self) -> usize {
        self.nvfp4_data_bytes()
    }

    /// Turbo4 scale section bytes (same layout as NVFP4: FP8 per-group).
    pub fn turbo4_scale_bytes(&self) -> usize {
        self.nvfp4_scale_bytes()
    }

    /// Turbo3 data section bytes (3-bit packed: 8 values in 3 bytes).
    pub fn turbo3_data_bytes(&self) -> usize {
        let elems = self.block_size * self.num_kv_heads * self.head_dim;
        elems * 3 / 8
    }

    /// Turbo3 scale section bytes (FP8 per-group, same as turbo4).
    pub fn turbo3_scale_bytes(&self) -> usize {
        self.nvfp4_scale_bytes()
    }

    /// Turbo8 data section bytes (FP8 E4M3: 1 byte per element).
    pub fn turbo8_data_bytes(&self) -> usize {
        self.block_size * self.num_kv_heads * self.head_dim
    }

    /// Turbo8 scale section bytes — **BF16 per-group scales** (2 bytes
    /// each, vs the 1-byte FP8 scales NVFP4/Turbo3/Turbo4 use). The
    /// BF16 upgrade is what makes Turbo8 viable across many-layer models
    /// like MiniMax M2.7 (58 Turbo8 layers under auto HP=2). Returns
    /// `(num_groups) * 2` bytes total.
    pub fn turbo8_scale_bytes(&self) -> usize {
        // num_groups = elems / GROUP_SIZE; each scale is 2 bytes (BF16).
        self.nvfp4_scale_bytes() * 2
    }
}

/// Per-layer KV cache pool.
struct LayerPool {
    k_pool: DevicePtr,
    v_pool: DevicePtr,
    /// Stride between blocks in bytes.
    block_stride: usize,
    /// Effective dtype for this layer.
    dtype: KvCacheDtype,
}

/// Paged KV cache across all attention layers.
pub struct PagedKvCache {
    layers: Vec<LayerPool>,
    num_blocks: usize,
    free_blocks: Vec<u32>,
    /// Per-block reference count. Enables shared blocks (prefix caching).
    /// Default: 1 on alloc, freed when decremented to 0.
    block_ref_counts: Vec<u32>,
    config: KvCacheConfig,
}

mod paged_impl;
#[cfg(test)]
mod tests;
