// SPDX-License-Identifier: AGPL-3.0-only
//
// Sharded + single safetensors loaders. Split out of `loader.rs` to keep
// the parent under the 500-line cap.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::Path;

use super::super::{WeightDtype, WeightTensor, evict_page_cache};
use super::{SafetensorsIndex, check_oom_guard, estimate_has_fp8, estimate_load_bytes};
use crate::gpu::GpuBackend;

pub(super) fn load_sharded(
    model_dir: &Path,
    index_path: &Path,
    gpu: &dyn GpuBackend,
    oom_reserve_bytes: usize,
    skip_fn: &dyn Fn(&str) -> bool,
    peak_multiplier_override: Option<f64>,
) -> Result<HashMap<String, WeightTensor>> {
    let index_json = std::fs::read_to_string(index_path)
        .with_context(|| format!("Failed to read {}", index_path.display()))?;
    let index: SafetensorsIndex = serde_json::from_str(&index_json)?;

    let mut offload_logged = false; // Track if we've logged the managed memory fallback

    // Group tensors by shard to minimize mmap overhead
    let mut shard_to_tensors: HashMap<String, Vec<String>> = HashMap::new();
    for (tensor_name, shard_name) in &index.weight_map {
        shard_to_tensors
            .entry(shard_name.clone())
            .or_default()
            .push(tensor_name.clone());
    }

    // Pre-flight: estimate bytes from index with model-building overhead.
    let shard_files: Vec<std::path::PathBuf> =
        shard_to_tensors.keys().map(|s| model_dir.join(s)).collect();
    let estimated = estimate_load_bytes(&shard_files, skip_fn)?;
    let has_fp8 = estimate_has_fp8(&shard_files, skip_fn)?;
    let overhead_multiplier: f64 =
        peak_multiplier_override.unwrap_or(if has_fp8 { 1.5 } else { 1.3 });
    let peak_estimated = (estimated as f64 * overhead_multiplier) as usize;
    let free = gpu.free_memory()?;
    let free_gb = free as f64 / (1024.0 * 1024.0 * 1024.0);
    let est_gb = estimated as f64 / (1024.0 * 1024.0 * 1024.0);
    let peak_gb = peak_estimated as f64 / (1024.0 * 1024.0 * 1024.0);
    let reserve_gb = oom_reserve_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    tracing::info!(
        "Pre-flight estimate: {:.2} GB on-disk weights, {:.1}x overhead = {:.2} GB peak, \
         {:.2} GB free, {:.1} GB reserve (FP8: {})",
        est_gb,
        overhead_multiplier,
        peak_gb,
        free_gb,
        reserve_gb,
        has_fp8,
    );
    if peak_estimated + oom_reserve_bytes > free {
        bail!(
            "OOM pre-flight: model peak memory ({:.2} GB = {:.2} GB weights × {:.1}x \
             model-building overhead) + {:.1} GB reserve = {:.2} GB, \
             but only {:.2} GB GPU memory is available. \
             This model is too large. Use a smaller quantization (NVFP4 instead of FP8) \
             or add more GPUs for expert parallelism.",
            peak_gb,
            est_gb,
            overhead_multiplier,
            reserve_gb,
            peak_gb + reserve_gb,
            free_gb,
        );
    }

    let mut weights = HashMap::new();
    let mut skipped = 0usize;
    let total_shards = shard_to_tensors.len();
    let initial_free = free;

    for (i, (shard_name, tensor_names)) in shard_to_tensors.iter().enumerate() {
        let shard_path = model_dir.join(shard_name);
        tracing::info!(
            "Loading shard {}/{}: {} ({} tensors)",
            i + 1,
            total_shards,
            shard_name,
            tensor_names.len()
        );

        let file = std::fs::File::open(&shard_path)
            .with_context(|| format!("Failed to open {}", shard_path.display()))?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let tensors = safetensors::SafeTensors::deserialize(&mmap)?;

        for name in tensor_names {
            if skip_fn(name) {
                skipped += 1;
                continue;
            }
            let view = tensors.tensor(name)?;
            let dtype = WeightDtype::from_safetensors(view.dtype())?;
            let shape: Vec<usize> = view.shape().to_vec();
            let data = view.data();

            // Try GPU alloc first; if OOM, fall back to managed (UVM) memory.
            // On GB10 unified memory, managed alloc uses Linux swap for overflow.
            let ptr = match gpu.alloc(data.len()) {
                Ok(p) => {
                    gpu.copy_h2d(data, p)?;
                    p
                }
                Err(_) => {
                    if !offload_logged {
                        tracing::warn!(
                            "GPU alloc failed for {} ({} bytes) — switching to managed (UVM) memory. \
                             Weights will be paged via Linux swap (slower but avoids OOM).",
                            name,
                            data.len()
                        );
                        offload_logged = true;
                    }
                    let p = gpu.alloc_managed(data.len())?;
                    // Use CPU memcpy (not GPU copy_h2d) to avoid GPU page faults.
                    // Managed memory is CPU-accessible, so memcpy writes directly to
                    // CPU pages. The GPU will page-fault on first access during kernels.
                    unsafe {
                        std::ptr::copy_nonoverlapping(data.as_ptr(), p.0 as *mut u8, data.len());
                    }
                    p
                }
            };

            weights.insert(name.clone(), WeightTensor { ptr, shape, dtype });
        }

        // Drop mmap before evicting page cache — releases the mapping first.
        drop(tensors);
        drop(mmap);
        evict_page_cache(&file);

        // OOM guard: check free memory after each shard
        let free_now = gpu.free_memory()?;
        let used = initial_free.saturating_sub(free_now);
        tracing::info!(
            "  Shard {}/{} done — GPU memory: {:.2} GB used, {:.2} GB free",
            i + 1,
            total_shards,
            used as f64 / (1024.0 * 1024.0 * 1024.0),
            free_now as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        if !offload_logged {
            check_oom_guard(
                gpu,
                oom_reserve_bytes,
                &format!("weight loading (shard {}/{})", i + 1, total_shards),
            )?;
        }
    }

    if skipped > 0 {
        tracing::info!("EP: skipped {} remote expert tensors", skipped);
    }
    tracing::info!("Loaded {} weight tensors", weights.len());
    Ok(weights)
}

pub(super) fn load_single(
    path: &Path,
    gpu: &dyn GpuBackend,
    oom_reserve_bytes: usize,
    skip_fn: &dyn Fn(&str) -> bool,
) -> Result<HashMap<String, WeightTensor>> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let tensors = safetensors::SafeTensors::deserialize(&mmap)?;

    let mut weights = HashMap::new();
    for (name, view) in tensors.tensors() {
        if skip_fn(&name) {
            continue;
        }
        let dtype = WeightDtype::from_safetensors(view.dtype())?;
        let shape: Vec<usize> = view.shape().to_vec();
        let data = view.data();

        let ptr = gpu.alloc(data.len())?;
        gpu.copy_h2d(data, ptr)?;

        weights.insert(name, WeightTensor { ptr, shape, dtype });
    }

    // Drop mmap before evicting page cache.
    drop(tensors);
    drop(mmap);
    evict_page_cache(&file);

    // OOM guard after single-file load
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy())
        .unwrap_or_default();
    check_oom_guard(
        gpu,
        oom_reserve_bytes,
        &format!("weight loading ({file_name})"),
    )?;

    Ok(weights)
}
