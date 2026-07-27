// SPDX-License-Identifier: AGPL-3.0-only

//! Weight loading from safetensors files (SBIO IORouter for filesystem I/O).

use crate::gpu::{DevicePtr, GpuBackend};
use anyhow::{Result, bail};
use std::collections::HashMap;
use std::path::Path;

/// Advise the OS to evict a file's pages from the page cache.
///
/// On GB10 (unified memory), mmap'd safetensors share the GPU memory pool.
/// After copying tensors to GPU, the mmap pages linger in the page cache,
/// consuming memory that should be available for KV cache and inference buffers.
/// This function tells the kernel those pages are no longer needed.
#[cfg(target_os = "linux")]
pub(crate) fn evict_page_cache(file: &std::fs::File) {
    use std::os::unix::io::AsRawFd;
    // POSIX_FADV_DONTNEED = 4 on Linux (POSIX standard).
    // macOS lacks posix_fadvise — see the non-linux branch below.
    const POSIX_FADV_DONTNEED: libc::c_int = 4;
    unsafe {
        libc::posix_fadvise(file.as_raw_fd(), 0, 0, POSIX_FADV_DONTNEED);
    }
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn evict_page_cache(_file: &std::fs::File) {
    // No-op: macOS/BSD have no posix_fadvise. Apple Silicon UMA already
    // shares page cache with the GPU pool, so eviction is unnecessary.
}

/// Data type of a weight tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDtype {
    BF16,
    FP32,
    FP8E4M3,
    FP8E8M0,
    UInt8,
    Int64,
}

impl WeightDtype {
    pub fn byte_size(self) -> usize {
        match self {
            Self::BF16 => 2,
            Self::FP32 => 4,
            Self::FP8E4M3 => 1,
            Self::FP8E8M0 => 1,
            Self::UInt8 => 1,
            Self::Int64 => 8,
        }
    }

    fn from_safetensors(dtype: safetensors::Dtype) -> Result<Self> {
        match dtype {
            safetensors::Dtype::BF16 => Ok(Self::BF16),
            safetensors::Dtype::F32 => Ok(Self::FP32),
            safetensors::Dtype::U8 => Ok(Self::UInt8),
            // I8: raw 1-byte container for 4-bit-packed NVFP4 (DeepSeek-V4 MTP
            // experts). Treat as UInt8 — signedness is irrelevant for packed FP4.
            safetensors::Dtype::I8 => Ok(Self::UInt8),
            safetensors::Dtype::F8_E4M3 => Ok(Self::FP8E4M3),
            safetensors::Dtype::F8_E8M0 => Ok(Self::FP8E8M0),
            safetensors::Dtype::I64 => Ok(Self::Int64),
            other => bail!("Unsupported safetensors dtype: {other:?}"),
        }
    }

    /// Map a raw safetensors header dtype STRING (as it appears in the JSON
    /// header, e.g. `"BF16"`, `"F8_E4M3"`) to a [`WeightDtype`], factored out
    /// so the RDMA weight loader (which receives dtype as a wire string in the
    /// peer manifest, not a `safetensors::Dtype`) resolves it identically to
    /// the disk loaders — byte-identity depends on the two ends agreeing.
    pub fn from_safetensors_str(s: &str) -> Result<Self> {
        Ok(match s {
            "F32" => Self::FP32,
            "BF16" => Self::BF16,
            "U8" => Self::UInt8,
            // I8 is a 1-byte raw container (packed NVFP4); signedness is
            // irrelevant, treat as raw bytes exactly like the disk path.
            "I8" => Self::UInt8,
            "F8_E4M3" => Self::FP8E4M3,
            "F8_E8M0" => Self::FP8E8M0,
            "I64" => Self::Int64,
            other => bail!("Unsupported safetensors dtype '{other}'"),
        })
    }
}

/// Convert a little-endian IEEE-754 half-precision (F16) tensor byte buffer
/// to BF16 bytes. F16 and BF16 are both 2 bytes/element but have different
/// bit layouts (5-bit vs 8-bit exponent), so the bytes cannot be
/// reinterpreted — each value goes f16 → f32 (exact) → bf16
/// (round-to-nearest-even). Shared by both disk loaders so F16 checkpoints
/// (e.g. centml modelopt W4A4 exports, which ship all unquantized tensors as
/// F16) land in the store as BF16; [`WeightDtype`] itself stays closed to
/// store-legal dtypes and F16 can never appear on the RDMA wire.
pub(crate) fn f16_to_bf16_bytes(src: &[u8]) -> Vec<u8> {
    use half::{bf16, f16};
    debug_assert_eq!(src.len() % 2, 0, "F16 tensor byte length must be even");
    let mut out = Vec::with_capacity(src.len());
    for pair in src.chunks_exact(2) {
        let h = f16::from_le_bytes([pair[0], pair[1]]);
        out.extend_from_slice(&bf16::from_f32(h.to_f32()).to_le_bytes());
    }
    out
}

/// A weight tensor on the GPU.
pub struct WeightTensor {
    pub ptr: DevicePtr,
    pub shape: Vec<usize>,
    pub dtype: WeightDtype,
}

impl WeightTensor {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_size(&self) -> usize {
        self.num_elements() * self.dtype.byte_size()
    }
}

/// All model weights loaded onto the GPU, keyed by HuggingFace name.
pub struct WeightStore {
    weights: HashMap<String, WeightTensor>,
}

impl WeightStore {
    /// Create an empty weight store (for testing).
    pub fn empty() -> Self {
        Self {
            weights: HashMap::new(),
        }
    }

    /// Wrap a pre-built map. Used by alternate loaders (e.g.
    /// `fast_weights::FastSafetensorsLoader`, and the RDMA weight loader in
    /// `spark-storage`, which lives in a different crate and so needs this pub).
    pub fn from_map(weights: HashMap<String, WeightTensor>) -> Self {
        Self { weights }
    }

    /// Get a weight tensor by name. Fails fast if not found.
    pub fn get(&self, name: &str) -> Result<&WeightTensor> {
        self.weights
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Weight '{name}' not found in store"))
    }

    /// Check if a weight exists.
    pub fn contains(&self, name: &str) -> bool {
        self.weights.contains_key(name)
    }

    /// Number of loaded weights.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// True if no weights are loaded.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Iterator over all weight names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.weights.keys().map(|s| s.as_str())
    }

    /// Total bytes across all weight tensors on the GPU.
    pub fn total_bytes(&self) -> usize {
        self.weights.values().map(|w| w.byte_size()).sum()
    }

    /// Check if any tensor has FP8 dtype.
    pub fn has_fp8_weights(&self) -> bool {
        self.weights
            .values()
            .any(|w| matches!(w.dtype, WeightDtype::FP8E4M3))
    }
}

/// SBIO IORouter trait for weight loading.
pub trait WeightLoader {
    fn load(
        &self,
        model_dir: &Path,
        gpu: &dyn GpuBackend,
        oom_reserve_bytes: usize,
    ) -> Result<WeightStore>;
}

/// Loads weights from safetensors files using mmap.
pub struct SafetensorsLoader {
    /// EP rank (0-based). Only used when ep_world_size > 1.
    pub ep_rank: usize,
    /// EP world size. When > 1, remote expert tensors are skipped.
    pub ep_world_size: usize,
    /// Total number of MoE experts in the model (for EP partitioning).
    pub num_experts: usize,
    /// Override for the peak memory multiplier in the pre-flight OOM check.
    /// Set from QuantFormat::peak_memory_multiplier() in the caller.
    /// When None, the pre-flight uses its own heuristic (1.3x NVFP4 / 1.5x FP8).
    pub peak_memory_multiplier: Option<f64>,
}

impl Default for SafetensorsLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl SafetensorsLoader {
    /// Create a loader with no expert parallelism (loads all tensors).
    pub fn new() -> Self {
        Self {
            ep_rank: 0,
            ep_world_size: 1,
            num_experts: 0,
            peak_memory_multiplier: None,
        }
    }

    /// Create a loader with EP-aware filtering.
    pub fn with_ep(ep_rank: usize, ep_world_size: usize, num_experts: usize) -> Self {
        Self {
            ep_rank,
            ep_world_size,
            num_experts,
            peak_memory_multiplier: None,
        }
    }

    /// Check if a tensor should be skipped under EP.
    /// Skips `*.experts.{E}.*` tensors where E is not in local range.
    /// MTP head experts are never skipped (small, fully replicated).
    fn should_skip_tensor(&self, name: &str) -> bool {
        if self.ep_world_size <= 1 {
            return false;
        }
        // MTP head experts are small — always replicate, never shard.
        if name.starts_with("mtp.") {
            return false;
        }
        // Parse expert index from patterns like "*.experts.42.gate_proj*"
        if let Some(idx) = parse_expert_index(name) {
            let per_rank = self.num_experts / self.ep_world_size;
            let local_start = self.ep_rank * per_rank;
            let local_end = if self.ep_rank == self.ep_world_size - 1 {
                self.num_experts
            } else {
                local_start + per_rank
            };
            idx < local_start || idx >= local_end
        } else {
            false // Non-expert tensors are always loaded (replicated)
        }
    }
}

/// Parse expert index from tensor name (e.g. "model.layers.3.mlp.experts.42.gate_proj.weight" → 42).
pub fn parse_expert_index(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "experts" && i + 1 < parts.len() {
            return parts[i + 1].parse().ok();
        }
    }
    None
}

pub mod adapter;
mod loader;
pub mod mlx_int8;
pub(crate) use loader::estimate_load_bytes;
// Consumed by the unix-only fast-weights (O_DIRECT) loader path.
#[cfg(unix)]
pub(crate) use loader::{check_oom_guard, estimate_has_fp8};

#[cfg(test)]
mod from_str_tests {
    use super::WeightDtype;

    #[test]
    fn from_safetensors_str_matches_disk_mapping() {
        // The RDMA weight peer publishes these raw header strings; the client
        // must resolve them to the exact WeightDtype the disk loaders use, else
        // byte_size/shape diverge and logits break. Locks the closed mapping.
        use WeightDtype::*;
        for (s, want) in [
            ("F32", FP32),
            ("BF16", BF16),
            ("U8", UInt8),
            ("I8", UInt8), // packed NVFP4 raw container
            ("F8_E4M3", FP8E4M3),
            ("F8_E8M0", FP8E8M0),
            ("I64", Int64),
        ] {
            assert_eq!(
                WeightDtype::from_safetensors_str(s).unwrap(),
                want,
                "dtype {s}"
            );
        }
        // F16 is converted to BF16 at disk-load; a store (and therefore a
        // peer manifest) can never contain it, so the wire mapping rejects it.
        assert!(WeightDtype::from_safetensors_str("F16").is_err());
        assert!(WeightDtype::from_safetensors_str("bogus").is_err());
    }

    #[test]
    fn f16_bytes_convert_to_bf16_via_f32() {
        use half::{bf16, f16};
        // Cover sign, exact powers of two, a value needing mantissa rounding
        // (f16 has 10 mantissa bits, bf16 only 7), f16 max, and a subnormal.
        let vals = [0.0f32, 1.0, -1.5, 0.1, 65504.0, -6.1035156e-5];
        let src: Vec<u8> = vals
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect();
        let out = super::f16_to_bf16_bytes(&src);
        assert_eq!(out.len(), src.len());
        for (i, v) in vals.iter().enumerate() {
            let got = bf16::from_le_bytes([out[2 * i], out[2 * i + 1]]);
            let want = bf16::from_f32(f16::from_f32(*v).to_f32());
            assert_eq!(got, want, "value {v}");
        }
    }
}
