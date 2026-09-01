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
    /// IEEE fp16. Shipped by EXL3 checkpoints (Hadamard sign vectors `suh`/
    /// `svh` and the unquantized dense tensors); consumers convert to BF16
    /// at load where a path expects it.
    F16,
    FP8E4M3,
    FP8E8M0,
    UInt8,
    /// Opaque 2-byte container: EXL3 `.trellis` packed code words arrive as
    /// safetensors I16 but are raw bits, not integers — nothing may do
    /// arithmetic on them except the EXL3 decode kernels.
    UInt16,
    /// Raw i32 scalars — the EXL3 per-tensor codebook flag (`.mul1`, which
    /// stores the codebook's multiplier constant).
    Int32,
    Int64,
    /// Keep-packed PrismML ternary Q2_0 (ggml id 42): raw on-disk blocks stay
    /// 2-bit in VRAM (fp16 scale + 2-bit codes per group of `group` elements),
    /// dequantized in-kernel by the native `q2_0_gemv` decode path. Only
    /// produced by the GGUF loader under `ATLAS_GGUF_NATIVE_Q2=1`. Its byte
    /// footprint is NOT a per-element size (2-bit codes + an inline scale per
    /// group), so [`WeightDtype::byte_size`] returns 0 for this variant and the
    /// real size is computed in [`WeightTensor::byte_size`] (shape + group).
    PackedQ2_0 {
        group: u16,
    },
}

impl WeightDtype {
    /// Bytes per element for the fixed-width dtypes. Returns 0 for the
    /// block-based [`WeightDtype::PackedQ2_0`] — [`WeightTensor::byte_size`]
    /// handles that variant directly, and no caller multiplies its numel by this.
    pub fn byte_size(self) -> usize {
        match self {
            Self::BF16 => 2,
            Self::FP32 => 4,
            Self::F16 => 2,
            Self::FP8E4M3 => 1,
            Self::FP8E8M0 => 1,
            Self::UInt8 => 1,
            Self::UInt16 => 2,
            Self::Int32 => 4,
            Self::Int64 => 8,
            Self::PackedQ2_0 { .. } => 0,
        }
    }

    fn from_safetensors(dtype: safetensors::Dtype) -> Result<Self> {
        match dtype {
            safetensors::Dtype::BF16 => Ok(Self::BF16),
            safetensors::Dtype::F32 => Ok(Self::FP32),
            safetensors::Dtype::F16 => Ok(Self::F16),
            safetensors::Dtype::U8 => Ok(Self::UInt8),
            // I8: raw 1-byte container for 4-bit-packed NVFP4 (DeepSeek-V4 MTP
            // experts). Treat as UInt8 — signedness is irrelevant for packed FP4.
            safetensors::Dtype::I8 => Ok(Self::UInt8),
            // I16: raw 2-byte container for EXL3 packed trellis codes —
            // opaque bits, decoded only by the exl3_reconstruct kernels.
            safetensors::Dtype::I16 => Ok(Self::UInt16),
            // I32: raw scalar container (EXL3 codebook flag).
            safetensors::Dtype::I32 => Ok(Self::Int32),
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
            "F16" => Self::F16,
            "U8" => Self::UInt8,
            // I8 is a 1-byte raw container (packed NVFP4); signedness is
            // irrelevant, treat as raw bytes exactly like the disk path.
            "I8" => Self::UInt8,
            // I16 is a 2-byte raw container (EXL3 packed trellis codes),
            // mirroring the disk path's mapping.
            "I16" => Self::UInt16,
            // I32 raw scalar (EXL3 codebook flag), mirroring the disk path.
            "I32" => Self::Int32,
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
        match self.dtype {
            // Packed Q2_0: `n_blocks = numel / group` blocks of
            // `2 + group/4` bytes (34 @ g128, 18 @ g64) — the on-disk footprint.
            WeightDtype::PackedQ2_0 { group } => {
                let g = group as usize;
                debug_assert!(g == 128 || g == 64, "unexpected Q2_0 group {g}");
                let n_blocks = self.num_elements() / g.max(1);
                n_blocks * (2 + g / 4)
            }
            d => self.num_elements() * d.byte_size(),
        }
    }

    /// The Q2_0 group size if this tensor is keep-packed ternary, else `None`.
    pub fn q2_group(&self) -> Option<u16> {
        match self.dtype {
            WeightDtype::PackedQ2_0 { group } => Some(group),
            _ => None,
        }
    }

    /// True if this tensor holds keep-packed ternary Q2_0 blocks (id 42).
    pub fn is_packed_q2(&self) -> bool {
        matches!(self.dtype, WeightDtype::PackedQ2_0 { .. })
    }
}

/// All model weights loaded onto the GPU, keyed by HuggingFace name.
pub struct WeightStore {
    weights: HashMap<String, WeightTensor>,
    /// Tensors deliberately NOT uploaded, with where they live on disk.
    ///
    /// The n-gram embedding tables of the LongCat / Qwen3.8-Flash-Next family
    /// are 63 GB (LongCat-Lite) to ~102 GB (Flash-Next) of BF16. Uploading
    /// them through the generic path would exhaust a 121 GB unified box
    /// before any quantization could run — and on GB10 the fallback is
    /// `alloc_managed`, i.e. Linux swap, i.e. the documented kernel freeze.
    /// They are skipped at load and served either by streaming per-table
    /// quantize-on-load or straight off NVMe by `NgramRowCache`, both of
    /// which need only this (path, offset) locator.
    deferred: HashMap<String, DeferredTensor>,
}

/// Where a skipped tensor lives, so a consumer can read it in place.
#[derive(Clone, Debug)]
pub struct DeferredTensor {
    /// Shard file containing the tensor.
    pub path: std::path::PathBuf,
    /// ABSOLUTE byte offset of the tensor's first element in that file
    /// (safetensors header length + the tensor's `data_offsets[0]`).
    pub offset: u64,
    pub shape: Vec<usize>,
    pub dtype: WeightDtype,
}

impl WeightStore {
    /// Create an empty weight store (for testing).
    pub fn empty() -> Self {
        Self {
            weights: HashMap::new(),
            deferred: HashMap::new(),
        }
    }

    /// Record a tensor that was skipped at load, with its on-disk location.
    pub fn defer(&mut self, name: String, t: DeferredTensor) {
        self.deferred.insert(name, t);
    }

    /// Look up a deferred (not-uploaded) tensor's on-disk location.
    pub fn deferred(&self, name: &str) -> Option<&DeferredTensor> {
        self.deferred.get(name)
    }

    /// Every deferred tensor, name-sorted (NUMERIC on a trailing index, so
    /// `embedders.10` sorts after `embedders.2` — a lexicographic sort here
    /// silently mis-maps the n-gram tables, which cost a real debugging
    /// session the first time).
    pub fn deferred_sorted(&self) -> Vec<(&String, &DeferredTensor)> {
        let mut v: Vec<_> = self.deferred.iter().collect();
        v.sort_by_key(|(n, _)| split_trailing_index(n));
        v
    }

    /// Wrap a pre-built map. Used by alternate loaders (e.g.
    /// `fast_weights::FastSafetensorsLoader`, and the RDMA weight loader in
    /// `spark-storage`, which lives in a different crate and so needs this pub).
    pub fn from_map(weights: HashMap<String, WeightTensor>) -> Self {
        Self {
            weights,
            deferred: HashMap::new(),
        }
    }

    /// Insert (or replace) a tensor. Used by load-time materialization
    /// passes that rewrite a checkpoint's tensors into a layout the model
    /// loaders consume (the EXL3 pass: trellis triplets -> `.weight` [+
    /// NVFP4 scales]). Returns the displaced tensor, if any — the CALLER
    /// owns freeing its device memory (the store never frees).
    pub fn insert(&mut self, name: String, t: WeightTensor) -> Option<WeightTensor> {
        self.weights.insert(name, t)
    }

    /// Remove a tensor by name, returning it. As with [`Self::insert`], the
    /// caller owns freeing the returned tensor's device memory.
    pub fn remove(&mut self, name: &str) -> Option<WeightTensor> {
        self.weights.remove(name)
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

    /// Number of per-layer FP8 KV-cache scale tensors (`*.k_scale`) the
    /// checkpoint ships. `>0` means the model carries calibrated KV scales, so
    /// FP8 KV needs no online calibration; `0` means the scales default to 1.0
    /// (which clips BF16 into E4M3 range), so online calibration or a non-FP8 KV
    /// dtype is required. Used to log the right guidance at serve time.
    pub fn fp8_kv_scale_count(&self) -> usize {
        self.names().filter(|n| n.ends_with(".k_scale")).count()
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
    /// Skip the W4A4 `*.input_scale` activation scales at load.
    ///
    /// ModelOpt NVFP4 checkpoints ship one 0-dim F32 scalar per quantized
    /// projection. On a 512-expert model that is ~74k four-byte allocations,
    /// each taking a full allocation granule — GBs of padding for values
    /// Atlas never reads, because it serves w4a16 (BF16 activations) and the
    /// NVFP4 loader already treats the key as optional.
    ///
    /// OPT-IN: `step3p7` reads this key on its own path, so it must stay off
    /// unless the model's loader is known not to need it.
    pub skip_activation_scales: bool,
    /// Skip `mtp.*` tensors at load.
    ///
    /// For models whose loader deliberately does not build an MTP head,
    /// uploading its weights is pure waste — on Qwen3.8-Flash-Next that is a
    /// 1.49 GB expert shard plus the MTP backbone, held resident while the KV
    /// cache goes without.
    ///
    /// OPT-IN: a model that DOES build an MTP head must keep them, so this is
    /// set only where `load_mtp_weights` is known to return `None`.
    pub skip_mtp: bool,
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
            skip_activation_scales: false,
            skip_mtp: false,
        }
    }

    /// Create a loader with EP-aware filtering.
    pub fn with_ep(ep_rank: usize, ep_world_size: usize, num_experts: usize) -> Self {
        Self {
            ep_rank,
            ep_world_size,
            num_experts,
            peak_memory_multiplier: None,
            skip_activation_scales: false,
            skip_mtp: false,
        }
    }

    /// Check if a tensor should be skipped under EP.
    /// Skips `*.experts.{E}.*` tensors where E is not in local range.
    /// MTP head experts are never skipped (small, fully replicated).
    fn should_skip_tensor(&self, name: &str) -> bool {
        // MTP head weights for a model whose loader does not build one.
        if self.skip_mtp && name.starts_with("mtp.") {
            return true;
        }
        // W4A4 activation scales: never read on the w4a16 path (the NVFP4
        // loader falls back to `DevicePtr::NULL`), and 4-byte allocations are
        // almost pure granule padding at expert scale.
        if self.skip_activation_scales && name.ends_with(".input_scale") {
            return true;
        }
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

/// Split a tensor name into (everything but its last numeric path segment,
/// that segment as a number) so names sort NUMERICALLY on the index.
/// `embedders.2` must precede `embedders.10`; a plain lexicographic sort puts
/// `10` first and silently mis-maps every table after the ninth.
pub mod adapter;
pub mod exl3;
mod gguf;
mod loader;
pub mod mlx_int8;
pub use gguf::{GgufLoader, config_from_gguf_dir, find_gguf};
pub(crate) use loader::estimate_load_bytes;
// Platform-independent: consumed by the unix-only fast-weights (O_DIRECT) path
// AND by the GGUF loader, which builds everywhere. Gating this on `unix` broke
// the Windows CUDA build the moment `gguf.rs` started using it.
pub(crate) use loader::check_oom_guard;
// Consumed by the unix-only fast-weights (O_DIRECT) loader path.
#[cfg(unix)]
pub(crate) use loader::estimate_has_fp8;

mod name_utils;
pub(crate) use name_utils::split_trailing_index;
pub use name_utils::{is_ngram_table, parse_expert_index};

#[cfg(test)]
mod packed_q2_tests;
mod prefix_detect;
pub use prefix_detect::auto_detect_weight_prefix;

/// Release every weight tensor.
///
/// Safe to free per-entry because the loaders allocate per-tensor: the fast
/// path calls `gpu.alloc(meta.len)` once per tensor before inserting it
/// (`fast_weights/mod.rs:360-388`), and no loader inserts an `.offset()` view of
/// a shared block into this map. (Fused per-expert views DO exist — see
/// `weight_loader/step3p7.rs:93` — but they live in the layer structs that own
/// the fused allocation, not here, so this cannot double-free them.)
impl atlas_core::scope::ModelResource<dyn GpuBackend> for WeightStore {
    fn label(&self) -> &'static str {
        "weight store"
    }

    fn release(&mut self, gpu: &dyn GpuBackend) -> anyhow::Result<()> {
        let mut first_error = None;
        // `drain` rather than iterate: the map must not be left holding
        // pointers to memory that is gone, and it makes this idempotent.
        for (name, tensor) in self.weights.drain() {
            if let Err(e) = gpu.free(tensor.ptr)
                && first_error.is_none()
            {
                first_error = Some(e.context(format!("freeing weight {name}")));
            }
        }
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod teardown_tests;
