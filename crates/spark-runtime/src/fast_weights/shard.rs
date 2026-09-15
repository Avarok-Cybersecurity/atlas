// SPDX-License-Identifier: AGPL-3.0-only

//! Per-shard load for [`super::FastSafetensorsLoader`]: header parse,
//! filtering, the O_DIRECT / buffered pipelined reader, and the copier that
//! uploads each tensor either into its pooled arena slot (`pool.rs`) or its
//! own allocation. Split from `mod.rs` for the ≤500 LoC cap.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;
use std::sync::mpsc::{Receiver, sync_channel};

use anyhow::{Context, Result};

use super::header::{TensorMeta, parse_header};
use super::pool::{PoolPlan, ShardArenas};
use super::{ShardOpts, ShardSink, direct_io, fs_probe};
use crate::gpu::{DevicePtr, GpuBackend};
use crate::weights::{DeferredTensor, WeightTensor, evict_page_cache, f16_to_bf16_bytes};

/// Load a single shard with O_DIRECT + pipelined read/copy.
///
/// Pipeline:
///   reader thread: pread tensor N into aligned buffer → sync_channel ──▶
///   main thread:   recv → copy_h2d → store tensor
///
/// The channel has capacity 1, so at any time the reader is ≤1 tensor
/// ahead of the copier. Memory overhead per shard: 2 × max_tensor_bytes
/// (rounded up to O_DIRECT alignment).
///
/// Tensors whose EXL3 prefix is in `opts.pooled_prefixes` are uploaded into
/// ONE arena per class allocated up front (`PoolPlan` / `ShardArenas`);
/// everything else takes the per-tensor `gpu.alloc` path exactly as before.
/// If the copy loop fails after the arenas exist they are freed here, once.
pub(super) fn load_shard_fast(
    shard_path: &Path,
    tensor_filter: Option<&[String]>,
    gpu: &dyn GpuBackend,
    skip_fn: &dyn Fn(&str) -> bool,
    opts: &ShardOpts<'_>,
    sink: &mut ShardSink<'_>,
) -> Result<()> {
    // Header parsing uses a buffered fd — header is a few KB, cache pollution
    // is negligible and buffered I/O handles short reads cleanly.
    let mut meta_file = File::open(shard_path)
        .with_context(|| format!("Failed to open {}", shard_path.display()))?;
    let mut tensors = parse_header(&mut meta_file)?;
    let file_size = meta_file.metadata()?.len();

    // Filter down to tensors we actually want (index filter + EP filter).
    if let Some(allow) = tensor_filter {
        let allow_set: HashSet<&str> = allow.iter().map(|s| s.as_str()).collect();
        tensors.retain(|t| allow_set.contains(t.name.as_str()));
    }
    // The n-gram embedding TABLES are never uploaded with the checkpoint —
    // 63 GB (LongCat-Lite) to ~102 GB (Flash-Next) of BF16 would exhaust a
    // 121 GB unified box before any quantization could run, and the fallback
    // on GB10 is managed memory, i.e. Linux swap, i.e. a kernel freeze. They
    // are recorded with their on-disk location and served either by streaming
    // per-table quantize-on-load or straight off NVMe by the row cache.
    let mut deferred_here: Vec<(String, DeferredTensor)> = Vec::new();
    tensors.retain(|t| {
        if crate::weights::is_ngram_table(&t.name) {
            deferred_here.push((
                t.name.clone(),
                DeferredTensor {
                    path: shard_path.to_path_buf(),
                    offset: t.abs_offset,
                    shape: t.shape.clone(),
                    dtype: t.dtype,
                },
            ));
            return false;
        }
        !skip_fn(&t.name)
    });
    if !deferred_here.is_empty() {
        tracing::info!(
            "Deferred {} n-gram table(s) in {} — served from disk, not uploaded",
            deferred_here.len(),
            shard_path.display()
        );
        sink.deferred.extend(deferred_here);
    }

    // Pooled layout: decided from the metas alone, before any byte is read.
    let plan = PoolPlan::build(&tensors, opts.pooled_prefixes);
    let arenas = ShardArenas::alloc(gpu, &plan, tensors.len(), sink.pool_fallback_logged)?;

    // Where do the bytes live? On a NETWORK mount O_DIRECT is the worst
    // option available — it bypasses the page cache and makes every tensor a
    // synchronous round trip — so the answer overrides the tensor-count
    // heuristic below and turns the shard prefetch on whether or not the
    // operator passed the flag. `None` = could not tell (non-Linux, statfs
    // failed): change nothing, keep the flag-driven behaviour.
    let net_fs = fs_probe::network_fs(shard_path).flatten();

    // Per-shard heuristic: above `direct_io_tensor_cap` tensors, O_DIRECT's
    // per-tensor syscall + 4 KiB alignment overhead costs more than kernel
    // readahead on the buffered path saves. Skip the direct-open attempt
    // entirely in that case — keeps the log clean and avoids a wasted fd.
    let under_cap = tensors.len() <= opts.direct_io_tensor_cap;
    let wants_direct = opts.try_direct_io && under_cap && net_fs.is_none();
    if opts.try_direct_io && !wants_direct {
        match net_fs {
            Some(fs) if under_cap => tracing::info!(
                "  Shard is on a {fs} mount — using buffered+pipelined path with \
                 prefetch (O_DIRECT would bypass the page cache and make every \
                 one of the {} tensors a network round trip)",
                tensors.len(),
            ),
            _ => tracing::info!(
                "  Shard has {} tensors (> {} cap) — using buffered+pipelined path",
                tensors.len(),
                opts.direct_io_tensor_cap
            ),
        }
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
    // Prefetch: requested, or implied by a network mount (that is what the
    // flag was added for — this just stops it depending on the operator
    // knowing where the checkpoint is mounted).
    if (opts.prefetch_shards || net_fs.is_some()) && !using_direct {
        advise_prefetch_shard(&buffered_file, shard_path, file_size);
    }

    // Pipelined reader: sends (tensor_index, aligned_buffer, slice_start) to main.
    let (tx, rx) = sync_channel::<Result<ReadMsg>>(1);
    let tensors_for_reader: Vec<(u64, usize)> =
        tensors.iter().map(|t| (t.abs_offset, t.len)).collect();
    let raw_fd = {
        use std::os::unix::io::AsRawFd;
        data_fd.as_raw_fd()
    };

    let reader_handle = std::thread::spawn(move || {
        for (idx, (abs_offset, len)) in tensors_for_reader.iter().enumerate() {
            let msg = direct_io::read_tensor_aligned(raw_fd, *abs_offset, *len, using_direct)
                .map(|(buf, slice_start)| (idx, buf, slice_start));
            if tx.send(msg).is_err() {
                break; // receiver dropped
            }
        }
    });

    // Copier: drains the channel, uploads into the arena slot or a fresh
    // allocation, inserts into the map. A failure mid-shard frees the
    // shard's arenas (once) before propagating — the per-tensor entries are
    // swept by the backend at drop, exactly as before.
    if let Err(e) = copy_tensors(rx, &tensors, gpu, arenas.as_ref(), sink) {
        if let Some(a) = arenas {
            a.free_all(gpu);
        }
        return Err(e);
    }

    reader_handle
        .join()
        .map_err(|_| anyhow::anyhow!("reader thread panicked"))?;

    if let Some(a) = arenas {
        tracing::info!(
            "  EXL3 weight pool: {} tensors / {:.2} GB into {} arena(s) — {} per-tensor \
             cuMemAllocs avoided, ~{:.2} GB of 2 MiB chunk-tail padding avoided \
             (GB10 driver model)",
            plan.pooled_tensors,
            plan.pooled_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            plan.arena_count(),
            plan.pooled_tensors.saturating_sub(plan.arena_count()),
            plan.padding_avoided as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        sink.arenas.extend(a.into_arenas());
    }

    // Release file handles, then advise the kernel to drop any pages we did
    // end up caching on the buffered fallback path. O_DIRECT reads never hit
    // the page cache, so the posix_fadvise is a no-op there but cheap.
    drop(direct_file);
    evict_page_cache(&buffered_file);
    drop(buffered_file);
    Ok(())
}

type ReadMsg = (usize, direct_io::AlignedBuffer, usize);

/// The copier half of the pipeline.
fn copy_tensors(
    rx: Receiver<Result<ReadMsg>>,
    tensors: &[TensorMeta],
    gpu: &dyn GpuBackend,
    arenas: Option<&ShardArenas>,
    sink: &mut ShardSink<'_>,
) -> Result<()> {
    for result in rx {
        let (idx, buf, slice_start) = result?;
        let meta = &tensors[idx];
        let raw = &buf.as_slice()[slice_start..slice_start + meta.len];
        // F16 shards: convert bytes to BF16 before upload (same length,
        // different bit layout — meta.dtype is already staged as BF16). The
        // EXL3 `.suh/.svh` sign vectors are exempt at header time and arrive
        // here with `from_f16 == false`, pooled or not.
        let converted: Vec<u8>;
        let src: &[u8] = if meta.from_f16 {
            converted = f16_to_bf16_bytes(raw);
            &converted
        } else {
            raw
        };

        let ptr = match arenas.and_then(|a| a.slot(idx)) {
            Some(slot) => {
                gpu.copy_h2d(src, slot)
                    .with_context(|| format!("uploading pooled tensor {}", meta.name))?;
                slot
            }
            None => upload_own(gpu, meta, src, sink.offload_logged)?,
        };

        sink.out.insert(
            meta.name.clone(),
            WeightTensor {
                ptr,
                shape: meta.shape.clone(),
                dtype: meta.dtype,
            },
        );
    }
    Ok(())
}

/// The per-tensor path: own allocation, managed (UVM) fallback on failure.
/// Byte-identical to the loader before pooling existed.
fn upload_own(
    gpu: &dyn GpuBackend,
    meta: &TensorMeta,
    src: &[u8],
    offload_logged: &mut bool,
) -> Result<DevicePtr> {
    match gpu.alloc(meta.len) {
        Ok(p) => {
            gpu.copy_h2d(src, p)?;
            Ok(p)
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
            Ok(p)
        }
    }
}

/// Scan every shard header once and return the EXL3 prefixes whose four
/// tensors will be pooled: those with a `.trellis` in the upload set for
/// which `predicate(prefix, trellis_shape)` holds. Model-wide (not per
/// shard) so a layer whose quartet straddles two shards still resolves K
/// for its aux tensors.
pub(super) fn select_pooled_prefixes(
    shard_files: &[std::path::PathBuf],
    tensor_to_shard: Option<&HashMap<String, String>>,
    skip_fn: &dyn Fn(&str) -> bool,
    predicate: &dyn Fn(&str, &[usize]) -> bool,
) -> Result<HashSet<String>> {
    let mut out = HashSet::new();
    for shard_path in shard_files {
        let mut f = File::open(shard_path)
            .with_context(|| format!("Failed to open {}", shard_path.display()))?;
        for t in parse_header(&mut f)? {
            let Some(prefix) = t.name.strip_suffix(".trellis") else {
                continue;
            };
            let indexed = tensor_to_shard.is_none_or(|m| m.contains_key(&t.name));
            if indexed && !skip_fn(&t.name) && predicate(prefix, &t.shape) {
                out.insert(prefix.to_string());
            }
        }
    }
    Ok(out)
}

#[cfg(target_os = "linux")]
fn advise_prefetch_shard(file: &File, shard_path: &Path, file_size: u64) {
    use std::os::unix::io::AsRawFd;

    let fd = file.as_raw_fd();
    let seq_rc = unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_SEQUENTIAL) };
    let willneed_rc = unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_WILLNEED) };
    if seq_rc == 0 && willneed_rc == 0 {
        tracing::info!(
            "  NFS/shard prefetch requested for {} ({:.2} GB)",
            shard_path.display(),
            file_size as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    } else {
        tracing::warn!(
            "  NFS/shard prefetch hint failed for {}: sequential_rc={}, willneed_rc={}",
            shard_path.display(),
            seq_rc,
            willneed_rc
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn advise_prefetch_shard(_file: &File, _shard_path: &Path, _file_size: u64) {}
