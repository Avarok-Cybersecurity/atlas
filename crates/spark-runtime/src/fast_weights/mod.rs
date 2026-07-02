// SPDX-License-Identifier: AGPL-3.0-only

//! Fast safetensors loader (InstantTensor-style) — pure Rust.
//!
//! Two wins over the mmap-based loader in [`crate::weights`]:
//!
//! 1. **`O_DIRECT`** reads. Bypasses the OS page cache, so the bytes never
//!    compete with GPU allocations on GB10 unified memory. The mmap path
//!    already works around this with `POSIX_FADV_DONTNEED` post-load; here
//!    we avoid the pollution in the first place.
//! 2. **Pipelined read/copy**. One background reader thread fetches the
//!    next tensor while the main thread does `copy_h2d` for the current
//!    one. Overlaps disk I/O with the host→device memcpy.
//!
//! Behavioural parity with [`crate::weights::SafetensorsLoader`] is
//! preserved — same EP filtering, same OOM pre-flight, same UVM fallback
//! on GPU allocation failure, same extra-weights handling.

use crate::gpu::GpuBackend;
use crate::weights::{
    WeightLoader, WeightStore, WeightTensor, check_oom_guard, estimate_has_fp8,
    estimate_load_bytes, evict_page_cache, parse_expert_index,
};
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::mpsc::sync_channel;

mod direct_io;
mod header;

use header::{parse_header, resolve_shards};

/// Pure-Rust InstantTensor-style loader. Same public shape as
/// [`crate::weights::SafetensorsLoader`].
pub struct FastSafetensorsLoader {
    pub ep_rank: usize,
    pub ep_world_size: usize,
    pub num_experts: usize,
    pub peak_memory_multiplier: Option<f64>,
    /// When true (default), attempt `O_DIRECT`; fall back to buffered reads if
    /// the filesystem rejects it (tmpfs, overlayfs, some FUSE backends).
    pub try_direct_io: bool,
    /// Per-shard heuristic cap: if a shard's tensor count exceeds this,
    /// we skip `O_DIRECT` for that shard and fall back to buffered +
    /// pipelined reads even when [`Self::try_direct_io`] is `true`.
    ///
    /// Motivation: `O_DIRECT`'s 4 KiB-aligned per-tensor `pread` has a
    /// fixed syscall + copy overhead that kernel readahead amortises for
    /// free on the buffered path. Benchmarks on GB10 showed buffered wins
    /// above ~5k tensors/shard; O_DIRECT wins below. Set to [`usize::MAX`]
    /// to disable.
    pub direct_io_tensor_cap: usize,
}

/// Default tensor-count cap for per-shard `O_DIRECT`. Above this, the fast
/// loader uses buffered reads even when `try_direct_io = true`. See the
/// field doc on [`FastSafetensorsLoader::direct_io_tensor_cap`].
pub const DEFAULT_DIRECT_IO_TENSOR_CAP: usize = 5000;

impl Default for FastSafetensorsLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl FastSafetensorsLoader {
    pub fn new() -> Self {
        Self {
            ep_rank: 0,
            ep_world_size: 1,
            num_experts: 0,
            peak_memory_multiplier: None,
            try_direct_io: true,
            direct_io_tensor_cap: DEFAULT_DIRECT_IO_TENSOR_CAP,
        }
    }

    pub fn with_ep(ep_rank: usize, ep_world_size: usize, num_experts: usize) -> Self {
        Self {
            ep_rank,
            ep_world_size,
            num_experts,
            peak_memory_multiplier: None,
            try_direct_io: true,
            direct_io_tensor_cap: DEFAULT_DIRECT_IO_TENSOR_CAP,
        }
    }

    fn should_skip_tensor(&self, name: &str) -> bool {
        if self.ep_world_size <= 1 {
            return false;
        }
        if name.starts_with("mtp.") {
            return false;
        }
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
            false
        }
    }
}

impl WeightLoader for FastSafetensorsLoader {
    fn load(
        &self,
        model_dir: &Path,
        gpu: &dyn GpuBackend,
        oom_reserve_bytes: usize,
    ) -> Result<WeightStore> {
        let skip_fn = |name: &str| self.should_skip_tensor(name);

        // Resolve shard list (sharded index, single file, or unindexed shards).
        let (shard_files, tensor_to_shard): (Vec<PathBuf>, Option<HashMap<String, String>>) =
            resolve_shards(model_dir)?;

        // Pre-flight OOM estimate (identical to SafetensorsLoader).
        {
            let estimated = estimate_load_bytes(&shard_files, &skip_fn)?;
            let has_fp8 = estimate_has_fp8(&shard_files, &skip_fn)?;
            let mult = self
                .peak_memory_multiplier
                .unwrap_or(if has_fp8 { 1.5 } else { 1.3 });
            let peak = (estimated as f64 * mult) as usize;
            let free = gpu.free_memory()?;
            let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
            tracing::info!(
                "Fast-load pre-flight: {:.2} GB on-disk, {:.1}x overhead = {:.2} GB peak, \
                 {:.2} GB free, {:.1} GB reserve (FP8: {})",
                gib(estimated),
                mult,
                gib(peak),
                gib(free),
                gib(oom_reserve_bytes),
                has_fp8,
            );
            if peak + oom_reserve_bytes > free {
                bail!(
                    "OOM pre-flight: peak {:.2} GB + {:.2} GB reserve exceeds {:.2} GB free. \
                     Use a smaller quantization or add more GPUs for EP.",
                    gib(peak),
                    gib(oom_reserve_bytes),
                    gib(free),
                );
            }
        }

        // Load each shard. Loaded tensors filtered by EP rules upstream.
        let mut weights: HashMap<String, WeightTensor> = HashMap::new();
        let total_shards = shard_files.len();
        let initial_free = gpu.free_memory()?;
        let mut offload_logged = false;

        for (i, shard_path) in shard_files.iter().enumerate() {
            // When an index is present, only load the tensors it routes here;
            // otherwise load everything in the shard. `None` means "load all".
            let shard_name = shard_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            let tensor_filter: Option<Vec<String>> = tensor_to_shard.as_ref().map(|map| {
                map.iter()
                    .filter(|(_, s)| *s == shard_name)
                    .map(|(t, _)| t.clone())
                    .collect()
            });

            tracing::info!(
                "Fast-loading shard {}/{}: {}{}",
                i + 1,
                total_shards,
                shard_name,
                tensor_filter
                    .as_ref()
                    .map(|v| format!(" ({} tensors)", v.len()))
                    .unwrap_or_default(),
            );

            load_shard_fast(
                shard_path,
                tensor_filter.as_deref(),
                gpu,
                &skip_fn,
                self.try_direct_io,
                self.direct_io_tensor_cap,
                &mut weights,
                &mut offload_logged,
            )?;

            let free_now = gpu.free_memory().unwrap_or(0);
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
                    &format!("fast weight loading (shard {}/{})", i + 1, total_shards),
                )?;
            }
        }

        // Extra weights (e.g. MTP grafted from another quantization).
        let no_skip = |_: &str| false;
        let extra = model_dir.join("extra_weights.safetensors");
        if extra.exists() {
            tracing::info!("Fast-loading extra_weights.safetensors");
            let mut extra_offload = false;
            load_shard_fast(
                &extra,
                None,
                gpu,
                &no_skip,
                self.try_direct_io,
                self.direct_io_tensor_cap,
                &mut weights,
                &mut extra_offload,
            )?;
        }

        tracing::info!("Fast-loaded {} weight tensors", weights.len());
        Ok(WeightStore::from_map(weights))
    }
}

/// Load a single shard with O_DIRECT + pipelined read/copy.
///
/// Pipeline:
///   reader thread: pread tensor N into aligned buffer → sync_channel ──▶
///   main thread:   recv → copy_h2d → store tensor
///
/// The channel has capacity 1, so at any time the reader is ≤1 tensor
/// ahead of the copier. Memory overhead per shard: 2 × max_tensor_bytes
/// (rounded up to O_DIRECT alignment).
#[allow(clippy::too_many_arguments)]
fn load_shard_fast(
    shard_path: &Path,
    tensor_filter: Option<&[String]>,
    gpu: &dyn GpuBackend,
    skip_fn: &dyn Fn(&str) -> bool,
    try_direct_io: bool,
    direct_io_tensor_cap: usize,
    out: &mut HashMap<String, WeightTensor>,
    offload_logged: &mut bool,
) -> Result<()> {
    // Header parsing uses a buffered fd — header is a few KB, cache pollution
    // is negligible and buffered I/O handles short reads cleanly.
    let mut meta_file = File::open(shard_path)
        .with_context(|| format!("Failed to open {}", shard_path.display()))?;
    let mut tensors = parse_header(&mut meta_file)?;
    let file_size = meta_file.metadata()?.len();

    // Filter down to tensors we actually want (index filter + EP filter).
    if let Some(allow) = tensor_filter {
        let allow_set: std::collections::HashSet<&str> = allow.iter().map(|s| s.as_str()).collect();
        tensors.retain(|t| allow_set.contains(t.name.as_str()));
    }
    tensors.retain(|t| !skip_fn(&t.name));

    // Per-shard heuristic: above `direct_io_tensor_cap` tensors, O_DIRECT's
    // per-tensor syscall + 4 KiB alignment overhead costs more than kernel
    // readahead on the buffered path saves. Skip the direct-open attempt
    // entirely in that case — keeps the log clean and avoids a wasted fd.
    let wants_direct = try_direct_io && tensors.len() <= direct_io_tensor_cap;
    if try_direct_io && !wants_direct {
        tracing::info!(
            "  Shard has {} tensors (> {} cap) — using buffered+pipelined path",
            tensors.len(),
            direct_io_tensor_cap
        );
    }

    // File for data reads. Try O_DIRECT; if it fails, fall through to buffered.
    let (direct_file, using_direct) = match wants_direct
        .then(|| direct_io::open_direct(shard_path))
        .transpose()
    {
        Ok(Some(f)) => (Some(f), true),
        Ok(None) => (None, false),
        Err(e) => {
            tracing::warn!(
                "O_DIRECT open failed for {} ({e}); falling back to buffered reads",
                shard_path.display()
            );
            (None, false)
        }
    };
    let buffered_file = File::open(shard_path)?;
    let data_fd = direct_file.as_ref().unwrap_or(&buffered_file);

    // Pipelined reader: sends (tensor_index, aligned_buffer, slice_start) to main.
    type ReadMsg = (usize, direct_io::AlignedBuffer, usize);
    let (tx, rx) = sync_channel::<Result<ReadMsg>>(1);
    let tensors_for_reader: Vec<(u64, usize)> =
        tensors.iter().map(|t| (t.abs_offset, t.len)).collect();
    let raw_fd = {
        use std::os::unix::io::AsRawFd;
        data_fd.as_raw_fd()
    };

    let _ = file_size; // retained for future use (tail-fragment buffered read)
    let reader_handle = std::thread::spawn(move || {
        for (idx, (abs_offset, len)) in tensors_for_reader.iter().enumerate() {
            let msg = direct_io::read_tensor_aligned(raw_fd, *abs_offset, *len, using_direct)
                .map(|(buf, slice_start)| (idx, buf, slice_start));
            if tx.send(msg).is_err() {
                break; // receiver dropped
            }
        }
    });

    // Copier: drains the channel, does gpu alloc + copy_h2d, inserts into the map.
    for result in rx {
        let (idx, buf, slice_start) = result?;
        let meta = &tensors[idx];
        let src = &buf.as_slice()[slice_start..slice_start + meta.len];

        let ptr = match gpu.alloc(meta.len) {
            Ok(p) => {
                gpu.copy_h2d(src, p)?;
                p
            }
            Err(_) => {
                if !*offload_logged {
                    tracing::warn!(
                        "GPU alloc failed for {} ({} bytes) — switching to managed (UVM) memory",
                        meta.name,
                        meta.len
                    );
                    *offload_logged = true;
                }
                let p = gpu.alloc_managed(meta.len)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), p.0 as *mut u8, meta.len);
                }
                p
            }
        };

        out.insert(
            meta.name.clone(),
            WeightTensor {
                ptr,
                shape: meta.shape.clone(),
                dtype: meta.dtype,
            },
        );
    }

    reader_handle
        .join()
        .map_err(|_| anyhow::anyhow!("reader thread panicked"))?;

    // Release file handles, then advise the kernel to drop any pages we did
    // end up caching on the buffered fallback path. O_DIRECT reads never hit
    // the page cache, so the posix_fadvise is a no-op there but cheap.
    drop(direct_file);
    evict_page_cache(&buffered_file);
    drop(buffered_file);
    Ok(())
}
