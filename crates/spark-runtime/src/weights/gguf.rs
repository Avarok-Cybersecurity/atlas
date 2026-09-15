// SPDX-License-Identifier: AGPL-3.0-only

//! Generic GGUF weight loader for Atlas.
//!
//! Loads any GGUF checkpoint the same way [`super::SafetensorsLoader`] loads
//! safetensors: mmap the file, walk its tensors, land each one GPU-resident in
//! a [`WeightStore`] keyed by HuggingFace name. Unlike safetensors, GGUF tensors
//! are quantized in-file, so each tensor is *dequantized to BF16* on the way in.
//!
//! Per the project design directive we **prefer GPU dequant**: upload the raw
//! quantized GGUF block bytes h2d, run a device dequant kernel that writes BF16
//! into fresh device memory, and hand that BF16 [`WeightTensor`] to the store.
//! The per-arch model loaders do the downstream NVFP4 requantize — this loader's
//! job ends at clean BF16. A pure-CPU reference dequant is the fallback for ggml
//! types lacking a GPU kernel and the correctness oracle under `MockGpuBackend`
//! (which cannot execute kernels).
//!
//! GGUF `dims` are ggml-order (fastest-varying first); Atlas/HF shapes are the
//! reverse, so each tensor's shape is reversed before it enters the store.
//!
//! The PrismML `Q2_0` (id 42) group size is not encoded in the type id. It
//! defaults to group-128 (the shipped Ternary-Bonsai layout); set
//! `ATLAS_GGUF_Q2_GROUP=64` for the fork-master group-64 layout.

mod config;
mod container;
mod dequant_cpu;
mod dequant_gpu;
mod names;
mod sidecar;
mod value_transform;

pub use config::config_from_gguf_dir;

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::{WeightDtype, WeightStore, WeightTensor, check_oom_guard, evict_page_cache};
use crate::gpu::{DevicePtr, GpuBackend};

/// Locate the backbone GGUF weight file in `dir`. Returns the
/// lexicographically-first non-mmproj `*.gguf` (also the first shard,
/// `*-00001-of-*`, of a split file). The mmproj vision sidecar is excluded here
/// and loaded separately (see `sidecar::find_mmproj`); a dir that is *only* an
/// mmproj falls back to the first file so the caller still gets a path to error
/// on.
pub fn find_gguf(dir: &Path) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("gguf"))
        .collect();
    candidates.sort();
    candidates
        .iter()
        .find(|p| !sidecar::is_mmproj(p))
        .cloned()
        .or_else(|| candidates.into_iter().next())
}

/// One resolved split-GGUF file set, in `split.no` order.
///
/// A split GGUF is `N` files that share one logical tensor table. llama.cpp
/// writes the full metadata KV block into shard 0 only; every shard carries
/// `split.no`, `split.count` and `split.tensors.count`, and each holds its own
/// slice of the tensors.
pub struct GgufShardSet {
    /// Shard paths ordered by `split.no`. Length 1 for an unsplit file.
    pub paths: Vec<PathBuf>,
    /// `split.tensors.count` from shard 0: the total across the whole set.
    pub total_tensors: usize,
}

/// Resolve every shard of a split GGUF, given the first one.
///
/// WHY THIS IS NOT "glob the directory": a directory can legitimately hold more
/// than one model, an in-progress download, or a leftover shard from a
/// different quant. The split keys are the only authority on what belongs to
/// this file set, so the count comes from `split.count` and membership is
/// checked per shard rather than assumed from the filename.
///
/// Refuses, rather than proceeding, when:
///   * the passed file is not `split.no == 0` (the caller resolved the wrong one)
///   * any sibling named by the convention is missing or unparseable
///   * a sibling disagrees about `split.count` or `split.tensors.count`
///   * the `split.no` values are not exactly `0..count`
///   * the per-shard tensor counts do not sum to `split.tensors.count`
///
/// That last check is the one that matters in practice: a download that lost a
/// shard still leaves every other shard individually valid and checksum-clean,
/// so nothing below the tensor-count sum catches it.
pub fn find_gguf_shards(first: &Path) -> Result<GgufShardSet> {
    let (_f, _m, g) = sidecar::open_gguf(first)?;
    let count = match g.get_u64("split.count") {
        None | Some(0) | Some(1) => {
            return Ok(GgufShardSet {
                paths: vec![first.to_path_buf()],
                total_tensors: g.tensors.len(),
            });
        }
        Some(c) => c as usize,
    };
    let no = g.get_u64("split.no").unwrap_or(0) as usize;
    if no != 0 {
        bail!(
            "GGUF split: {} reports split.no={no}, expected shard 0. \
             Point the loader at the `-00001-of-*` shard.",
            first.display()
        );
    }
    let total_tensors = g
        .get_u64("split.tensors.count")
        .context("GGUF split: shard 0 has 'split.count' but no 'split.tensors.count'")?
        as usize;

    let name = first
        .file_name()
        .and_then(|n| n.to_str())
        .context("GGUF split: shard path has no filename")?;
    // llama.cpp convention: `<stem>-00001-of-00007.gguf`, one-based on disk,
    // zero-based in `split.no`.
    let marker = format!("-00001-of-{count:05}.gguf");
    let stem = name.strip_suffix(&marker).with_context(|| {
        format!(
            "GGUF split: '{name}' declares split.count={count} but does not end in '{marker}'; \
             cannot derive sibling shard names"
        )
    })?;
    let dir = first.parent().unwrap_or_else(|| Path::new("."));

    let mut paths = vec![first.to_path_buf()];
    let mut seen = vec![0usize; count];
    seen[0] = g.tensors.len();
    for i in 1..count {
        let p = dir.join(format!("{stem}-{:05}-of-{count:05}.gguf", i + 1));
        if !p.exists() {
            bail!(
                "GGUF split: shard {} of {count} missing: {}",
                i + 1,
                p.display()
            );
        }
        let (_f2, _m2, g2) = sidecar::open_gguf(&p)
            .with_context(|| format!("GGUF split: failed to open shard {}", p.display()))?;
        let c2 = g2.get_u64("split.count").unwrap_or(0) as usize;
        if c2 != count {
            bail!(
                "GGUF split: {} reports split.count={c2}, shard 0 says {count}",
                p.display()
            );
        }
        let n2 = g2.get_u64("split.no").unwrap_or(usize::MAX as u64) as usize;
        if n2 != i {
            bail!(
                "GGUF split: {} reports split.no={n2}, expected {i} from its filename",
                p.display()
            );
        }
        if let Some(t2) = g2.get_u64("split.tensors.count")
            && t2 as usize != total_tensors
        {
            bail!(
                "GGUF split: {} reports split.tensors.count={t2}, shard 0 says {total_tensors}",
                p.display()
            );
        }
        seen[i] = g2.tensors.len();
        paths.push(p);
    }

    let summed: usize = seen.iter().sum();
    if summed != total_tensors {
        bail!(
            "GGUF split: shards hold {summed} tensors but 'split.tensors.count' is \
             {total_tensors} — the file set is incomplete or mismatched (per-shard: {seen:?})"
        );
    }
    Ok(GgufShardSet {
        paths,
        total_tensors,
    })
}

/// True when the native keep-packed Q2_0 decode path is enabled
/// (`ATLAS_GGUF_NATIVE_Q2=1`). Off by default: the loader dequants every id-42
/// tensor to BF16 exactly as before, so the default path is byte-identical.
/// When on, the "big" transform-free FFN projections (see
/// [`names::is_keep_packed_proj`]) are uploaded as raw `block_q2_0` blocks and
/// tagged [`WeightDtype::PackedQ2_0`] for in-kernel dequant at decode.
fn native_q2_enabled() -> bool {
    std::env::var("ATLAS_GGUF_NATIVE_Q2").ok().as_deref() == Some("1")
}

/// The id-42 PrismML group size, from `ATLAS_GGUF_Q2_GROUP` (default 128).
fn q2_group_usize() -> usize {
    match std::env::var("ATLAS_GGUF_Q2_GROUP").ok().as_deref() {
        Some("64") => 64,
        _ => 128,
    }
}

/// Map a group size to the container's `Q2Group` (for on-disk byte sizing).
fn q2_group_variant(g: usize) -> container::Q2Group {
    if g == 64 {
        container::Q2Group::G64
    } else {
        container::Q2Group::G128
    }
}

/// Loads weights from a GGUF file, dequantizing every tensor to BF16 on the GPU.
///
/// Mirrors [`super::SafetensorsLoader`] so the two are interchangeable behind the
/// [`super::WeightLoader`] trait and the serve call-site can pick one on file
/// type.
pub struct GgufLoader {
    /// EP rank (0-based). Only used when `ep_world_size > 1`.
    pub ep_rank: usize,
    /// EP world size. When > 1, remote expert slices are skipped.
    pub ep_world_size: usize,
    /// Total number of MoE experts in the model (for EP partitioning).
    pub num_experts: usize,
    /// Override for the peak-memory multiplier in the pre-flight OOM check.
    pub peak_memory_multiplier: Option<f64>,
}

impl Default for GgufLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufLoader {
    /// Create a loader with no expert parallelism (loads all tensors).
    pub fn new() -> Self {
        Self {
            ep_rank: 0,
            ep_world_size: 1,
            num_experts: 0,
            peak_memory_multiplier: None,
        }
    }

    /// Create a loader with EP-aware expert filtering.
    pub fn with_ep(ep_rank: usize, ep_world_size: usize, num_experts: usize) -> Self {
        Self {
            ep_rank,
            ep_world_size,
            num_experts,
            peak_memory_multiplier: None,
        }
    }

    /// True if expert `idx` lives on a remote EP rank and should be skipped.
    fn should_skip_expert(&self, idx: usize) -> bool {
        if self.ep_world_size <= 1 || self.num_experts == 0 {
            return false;
        }
        let per_rank = self.num_experts / self.ep_world_size;
        let local_start = self.ep_rank * per_rank;
        let local_end = if self.ep_rank == self.ep_world_size - 1 {
            self.num_experts
        } else {
            local_start + per_rank
        };
        idx < local_start || idx >= local_end
    }

    /// Split a dequantized, stacked expert buffer into per-expert `WeightTensor`s
    /// that alias offsets into the single BF16 device allocation. `shape[0]` is
    /// the expert count; each expert tensor is `shape[1..]`.
    fn emit_experts(
        &self,
        weights: &mut HashMap<String, WeightTensor>,
        base_ptr: DevicePtr,
        shape: &[usize],
        layer: usize,
        proj: &str,
        skipped: &mut usize,
    ) -> Result<()> {
        let count = *shape
            .first()
            .context("stacked expert tensor has no leading expert dimension")?;
        let per_elems: usize = shape[1..].iter().product();
        let per_bytes = per_elems * WeightDtype::BF16.byte_size();
        let expert_shape: Vec<usize> = shape[1..].to_vec();
        for e in 0..count {
            if self.should_skip_expert(e) {
                *skipped += 1;
                continue;
            }
            let ptr = base_ptr.offset(e * per_bytes);
            let name = names::expert_name(layer, proj, e);
            weights.insert(
                name,
                WeightTensor {
                    ptr,
                    shape: expert_shape.clone(),
                    dtype: WeightDtype::BF16,
                },
            );
        }
        Ok(())
    }
}

/// Dequant one tensor's raw block bytes to a BF16 device buffer. Prefer-GPU /
/// CPU-fallback: use the GPU kernel when one exists for `id` (and `force_cpu`
/// is unset), else the CPU reference dequant (host BF16 → single h2d).
fn dequant_to_device(
    gpu: &dyn GpuBackend,
    id: u32,
    raw: &[u8],
    num_elements: usize,
    q2_group: usize,
    force_cpu: bool,
) -> Result<DevicePtr> {
    if !force_cpu && dequant_gpu::supports(id) {
        let q_ptr = gpu.alloc(raw.len())?;
        gpu.copy_h2d(raw, q_ptr)?;
        let bf16_ptr = dequant_gpu::to_bf16(gpu, id, q_ptr, num_elements, q2_group)
            .with_context(|| format!("GPU dequant failed for ggml type {id}"))?;
        gpu.free(q_ptr)?;
        Ok(bf16_ptr)
    } else if dequant_cpu::supports(id) {
        let host = dequant_cpu::to_bf16_bytes(id, q2_group, raw, num_elements)
            .with_context(|| format!("CPU dequant failed for ggml type {id}"))?;
        debug_assert_eq!(host.len(), num_elements * WeightDtype::BF16.byte_size());
        let bf16_ptr = gpu.alloc(host.len())?;
        gpu.copy_h2d(&host, bf16_ptr)?;
        Ok(bf16_ptr)
    } else {
        bail!("No GPU or CPU dequant available for ggml type {id}");
    }
}

/// Pre-flight: estimate the total BF16 footprint (dequant expands quantized
/// blocks) and bail before allocating if it won't fit under the reserve.
fn preflight_oom(
    gpu: &dyn GpuBackend,
    est_bf16_bytes: usize,
    reserve_bytes: usize,
    multiplier: Option<f64>,
) -> Result<()> {
    // Small transient overhead: the raw quantized scratch buffer coexists with
    // its BF16 output for one tensor at a time (freed immediately after).
    let overhead = multiplier.unwrap_or(1.1);
    let peak = (est_bf16_bytes as f64 * overhead) as usize;
    let free = gpu.free_memory()?;
    let gb = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
    tracing::info!(
        "GGUF pre-flight: ~{:.2} GB BF16 after dequant, {:.1}x overhead = {:.2} GB peak, \
         {:.2} GB free, {:.1} GB reserve",
        gb(est_bf16_bytes),
        overhead,
        gb(peak),
        gb(free),
        gb(reserve_bytes),
    );
    if peak + reserve_bytes > free {
        bail!(
            "Pre-flight OOM: GGUF dequant to BF16 needs ~{:.2} GB peak + {:.1} GB reserve, \
             only {:.2} GB free. Use a smaller model or lower --oom-guard-mb.",
            gb(peak),
            gb(reserve_bytes),
            gb(free),
        );
    }
    Ok(())
}

impl super::WeightLoader for GgufLoader {
    fn load(
        &self,
        model_dir: &Path,
        gpu: &dyn GpuBackend,
        oom_reserve_bytes: usize,
    ) -> Result<WeightStore> {
        let path = find_gguf(model_dir)
            .with_context(|| format!("No .gguf file found in {}", model_dir.display()))?;
        tracing::info!("Loading GGUF weights from {}", path.display());

        let force_cpu = std::env::var("ATLAS_GGUF_FORCE_CPU").ok().as_deref() == Some("1");
        let native_q2 = native_q2_enabled();
        let q2_group = q2_group_usize();
        let q2_variant = q2_group_variant(q2_group);
        if native_q2 {
            tracing::info!(
                "ATLAS_GGUF_NATIVE_Q2=1: keeping id-42 FFN projections packed (group {q2_group})"
            );
        }

        // ── Backbone (text model) ──
        let (bb_file, bb_mmap, bb_gguf) = sidecar::open_gguf(&path)?;
        let arch = bb_gguf
            .get_str("general.architecture")
            .unwrap_or("llama")
            .to_lowercase();

        // Qwen3.5/3.6 GDN-hybrid GGUFs (llama.cpp `qwen35` converter) encode a
        // handful of GDN / RMSNorm tensor VALUES differently than Atlas's
        // kernels expect (norm +1 offset, `A_log = ln(-ssm_a)`, and a value-head
        // reorder). Read the GDN head geometry once so `load_pass` can invert
        // them per tensor (see `value_transform`).
        let is_qwen35 = value_transform::is_qwen35(&arch);
        let gdn = if is_qwen35 {
            value_transform::gdn_dims(&bb_gguf, &arch)
        } else {
            None
        };
        if is_qwen35 && gdn.is_none() {
            bail!(
                "GGUF arch '{arch}' is qwen35-family but the SSM metadata keys \
                 ({arch}.ssm.*) are missing; cannot apply GDN value transforms"
            );
        }

        // ── Optional mmproj vision-tower sidecar ──
        // Open it (if present) up front so the pre-flight OOM check covers both
        // files. mmaps are virtual, so holding two at once costs no RAM.
        let mmproj_path = sidecar::find_mmproj(model_dir, &path);
        let mmproj = match &mmproj_path {
            Some(mp) => {
                tracing::info!("Found mmproj vision sidecar {}", mp.display());
                Some(sidecar::open_gguf(mp)?)
            }
            None => None,
        };
        let mmproj_arch = mmproj.as_ref().map(|(_, _, g)| {
            g.get_str("general.architecture")
                .unwrap_or("clip")
                .to_lowercase()
        });

        // Pre-flight: combined BF16 footprint of both files.
        let mut est = sidecar::est_bf16(&bb_gguf, &arch);
        if let (Some((_, _, mm_gguf)), Some(mm_arch)) = (mmproj.as_ref(), mmproj_arch.as_ref()) {
            est += sidecar::est_bf16(mm_gguf, mm_arch);
        }
        preflight_oom(gpu, est, oom_reserve_bytes, self.peak_memory_multiplier)?;

        let mut weights: HashMap<String, WeightTensor> = HashMap::new();
        let mut skipped = 0usize;

        // Pass 1: backbone → weights.
        sidecar::load_pass(
            self,
            gpu,
            &bb_gguf,
            &bb_mmap,
            &arch,
            gdn,
            force_cpu,
            native_q2,
            q2_group,
            q2_variant,
            &mut weights,
            &mut skipped,
        )?;
        drop(bb_gguf);
        drop(bb_mmap);
        evict_page_cache(&bb_file);

        // Pass 2: mmproj sidecar → SAME weights map (clip names land under
        // `model.visual.*`, disjoint from the backbone's `model.layers.*`).
        // No GDN transforms (gdn = None) and no expert fan-out for clip.
        if let (Some((mm_file, mm_mmap, mm_gguf)), Some(mm_arch)) = (mmproj, mmproj_arch) {
            let before = weights.len();
            // mmproj is a `clip` tower — no qwen35 FFN names, so native_q2 is
            // irrelevant there; pass false to keep it on the plain BF16 path.
            sidecar::load_pass(
                self,
                gpu,
                &mm_gguf,
                &mm_mmap,
                &mm_arch,
                None,
                force_cpu,
                false,
                q2_group,
                q2_variant,
                &mut weights,
                &mut skipped,
            )?;
            tracing::info!(
                "Merged {} mmproj tensors (arch '{}') into the weight store",
                weights.len() - before,
                mm_arch,
            );
            drop(mm_gguf);
            drop(mm_mmap);
            evict_page_cache(&mm_file);
        }

        if skipped > 0 {
            tracing::info!("EP: skipped {} remote expert slices", skipped);
        }
        check_oom_guard(gpu, oom_reserve_bytes, "weight loading (GGUF)")?;
        tracing::info!("Loaded {} weight tensors (GGUF → BF16)", weights.len());
        Ok(WeightStore::from_map(weights))
    }
}

#[cfg(test)]
mod deepseek_v41_real_file_test;
#[cfg(test)]
mod deepseek_v41_stream_bench_test;
#[cfg(test)]
mod real_file_test;

#[cfg(all(test, feature = "cuda"))]
mod gpu_validate_test;

#[cfg(test)]
mod tests;
