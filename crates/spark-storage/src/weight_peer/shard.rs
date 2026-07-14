// SPDX-License-Identifier: AGPL-3.0-only
//
// Shard resolution + safetensors header parsing + the warm RO mmap (unix,
// server-side). Builds a [`WeightManifest`] from a model directory and holds
// each shard file mapped PROT_READ so its pages stay resident for one-sided
// RDMA reads. No verbs here — the mmap's raw base is handed to `serve`'s
// `reg_mr` unchanged.

use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use super::manifest::{WeightManifest, WeightTensorRecord};

/// Resolve shard files (index / single / glob), parse each shard's header,
/// and assemble the [`WeightManifest`]. Mirrors the resolution order in
/// `spark_runtime::fast_weights::header::resolve_shards`.
pub(super) fn build_manifest(dir: &Path, model_id: &str) -> Result<(Vec<PathBuf>, WeightManifest)> {
    let (shard_paths, weight_map) = resolve_shards(dir)?;

    let mut shard_files = Vec::with_capacity(shard_paths.len());
    let mut shard_lens = Vec::with_capacity(shard_paths.len());
    let mut tensors: Vec<WeightTensorRecord> = Vec::new();

    for (idx, path) in shard_paths.iter().enumerate() {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        let len = std::fs::metadata(path)
            .with_context(|| format!("stat {}", path.display()))?
            .len();
        // Only publish tensors the index actually routes (weight_map). A
        // shard header may list orphan/tied/aux tensors absent from the
        // index; the disk loaders iterate weight_map keys and never load
        // those, so filtering here keeps the RDMA store byte-identical
        // (same key set, not a superset). No index (single/glob) => keep all.
        parse_shard_header(path, idx as u32, false, weight_map.as_ref(), &mut tensors)?;
        shard_files.push(name);
        shard_lens.push(len);
    }

    // extra_weights.safetensors: an extra shard whose tensors are NEVER
    // expert-skipped (grafted MTP etc.), exactly like the disk loaders.
    let extra = dir.join("extra_weights.safetensors");
    if extra.exists() {
        let idx = shard_paths.len();
        let mut shard_paths = shard_paths.clone();
        let len = std::fs::metadata(&extra)?.len();
        // extra_weights.safetensors is never in the index weight_map; its
        // tensors are always fully published (extra=true), so pass no filter.
        parse_shard_header(&extra, idx as u32, true, None, &mut tensors)?;
        shard_files.push("extra_weights.safetensors".to_string());
        shard_lens.push(len);
        shard_paths.push(extra);
        let manifest = WeightManifest {
            version: WeightManifest::VERSION,
            model_id: model_id.to_string(),
            shard_files,
            shard_lens,
            tensors,
        };
        return Ok((shard_paths, manifest));
    }

    let manifest = WeightManifest {
        version: WeightManifest::VERSION,
        model_id: model_id.to_string(),
        shard_files,
        shard_lens,
        tensors,
    };
    Ok((shard_paths, manifest))
}

/// Resolved shard set: the shard file paths (shard-index order) plus the
/// index `weight_map` (tensor name -> shard file) when an index exists, or
/// `None` for a single-file / glob checkpoint (keep every header tensor).
type ShardResolution = (Vec<PathBuf>, Option<HashMap<String, String>>);

/// Shard discovery, resolution order identical to the disk loaders:
/// (1) model.safetensors.index.json, else consolidated.safetensors.index.json;
/// (2) single model.safetensors; (3) glob model.safetensors-* / consolidated-*.
pub(super) fn resolve_shards(dir: &Path) -> Result<ShardResolution> {
    let index = dir.join("model.safetensors.index.json");
    let consolidated = dir.join("consolidated.safetensors.index.json");
    let actual = if index.exists() {
        Some(index)
    } else if consolidated.exists() {
        Some(consolidated)
    } else {
        None
    };

    if let Some(ip) = actual {
        let json =
            std::fs::read_to_string(&ip).with_context(|| format!("read {}", ip.display()))?;
        let v: Value = serde_json::from_str(&json)?;
        let map = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .context("index json missing weight_map object")?;
        let weight_map: HashMap<String, String> = map
            .iter()
            .filter_map(|(k, val)| val.as_str().map(|s| (k.clone(), s.to_string())))
            .collect();
        let mut shards: Vec<String> = weight_map
            .values()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        shards.sort();
        let files = shards.iter().map(|s| dir.join(s)).collect();
        return Ok((files, Some(weight_map)));
    }

    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok((vec![single], None));
    }

    // PEFT adapter dir: a single `adapter_model.safetensors` (its keys are
    // the lora_A/lora_B tensors, classified client-side). Staged for LoRA
    // rotation over the RDMA tier (`weight_lora_rdma`).
    let adapter = dir.join("adapter_model.safetensors");
    if adapter.exists() {
        return Ok((vec![adapter], None));
    }

    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                (n.starts_with("model.safetensors-") || n.starts_with("consolidated-"))
                    && n.ends_with(".safetensors")
            })
        })
        .collect();
    shards.sort();
    if shards.is_empty() {
        bail!("no safetensor files found in {}", dir.display());
    }
    Ok((shards, None))
}

/// Parse one shard's safetensors header, pushing a [`WeightTensorRecord`]
/// per tensor. Byte layout: `[u64 LE header_size][header_size JSON]`; data
/// section starts at `8 + header_size`; each tensor's `data_offsets` are
/// relative to that. We publish ABSOLUTE offsets so the client's
/// `remote_addr = shard_base + offset_in_shard` reads out of the whole-file
/// MR directly. Validates dtype against the disk loaders' closed set.
fn parse_shard_header(
    path: &Path,
    shard_index: u32,
    extra: bool,
    weight_map: Option<&HashMap<String, String>>,
    out: &mut Vec<WeightTensorRecord>,
) -> Result<()> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut size_buf = [0u8; 8];
    f.read_exact(&mut size_buf)
        .with_context(|| format!("read header size of {}", path.display()))?;
    let header_size = u64::from_le_bytes(size_buf) as usize;
    if header_size > 64 * 1024 * 1024 {
        bail!(
            "{}: safetensor header too large ({header_size} bytes)",
            path.display()
        );
    }
    let mut header_buf = vec![0u8; header_size];
    f.read_exact(&mut header_buf)
        .with_context(|| format!("read header of {}", path.display()))?;
    let data_start = 8 + header_size as u64;

    let json: Value = serde_json::from_slice(&header_buf)?;
    let obj = json
        .as_object()
        .with_context(|| format!("{}: header is not a JSON object", path.display()))?;

    for (name, info) in obj {
        if name == "__metadata__" {
            continue;
        }
        // Skip header tensors the index doesn't route (orphan/tied/aux) so the
        // published set matches the disk loaders' weight_map iteration exactly.
        if let Some(map) = weight_map
            && !map.contains_key(name)
        {
            continue;
        }
        let dtype = info["dtype"].as_str().unwrap_or("BF16").to_string();
        validate_dtype(&dtype, name)?;
        let shape: Vec<u64> = info["shape"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
            .unwrap_or_default();
        let offsets = info["data_offsets"]
            .as_array()
            .with_context(|| format!("tensor {name} missing data_offsets"))?;
        let rel_start = offsets[0].as_u64().context("bad data_offsets[0]")?;
        let rel_end = offsets[1].as_u64().context("bad data_offsets[1]")?;
        out.push(WeightTensorRecord {
            name: name.clone(),
            dtype,
            shape,
            offset_in_shard: data_start + rel_start,
            len: rel_end - rel_start,
            shard_index,
            extra,
        });
    }
    Ok(())
}

/// The disk loaders' closed dtype set (mirrors
/// `WeightDtype::from_safetensors_str`). Fail at STAGE time on anything
/// unsupported rather than shipping a bad manifest.
fn validate_dtype(dtype: &str, tensor: &str) -> Result<()> {
    match dtype {
        // F16 is accepted for PEFT LoRA adapter shards (default PEFT save
        // is F32, some export F16); the `weight_lora_rdma` client converts
        // F16/F32 → BF16 host-side before landing. The base-weight disk
        // loaders never see F16 (they load model.safetensors, not adapters).
        "F32" | "F16" | "BF16" | "U8" | "I8" | "F8_E4M3" | "F8_E8M0" | "I64" => Ok(()),
        other => bail!("unsupported safetensors dtype '{other}' for tensor {tensor}"),
    }
}

/// A read-only whole-file `mmap`, warmed (MADV_WILLNEED) and unmapped on
/// drop. Held persistently in a `StagedModel` so pages stay resident across
/// connections (the weight-cache property). `Send`/`Sync`: the raw pointer
/// is only ever handed to `ibv_reg_mr` as a base and read by the NIC; the
/// Rust side never dereferences it.
pub(super) struct Mmap {
    pub(super) addr: *mut libc::c_void,
    pub(super) len: usize,
}

// SAFETY: see the doc above — addr/len are an immutable mapping description;
// the memory is read only by the HCA, never mutated through this pointer.
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    pub(super) fn open_ro(path: &Path) -> Result<Self> {
        use std::os::fd::AsRawFd;
        let f = std::fs::File::open(path)?;
        let len = f.metadata()?.len() as usize;
        if len == 0 {
            bail!("empty shard file {}", path.display());
        }
        // SAFETY: fd is a valid open RO file; MAP_SHARED read mapping of
        // `len` bytes. The kernel keeps the mapping valid after the fd
        // closes.
        let addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                f.as_raw_fd(),
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            bail!(
                "mmap {} failed: {}",
                path.display(),
                std::io::Error::last_os_error()
            );
        }
        // Warm the pages into RAM so the first (and every) RDMA read hits
        // resident memory — the cache property. Best-effort.
        // SAFETY: addr/len came from the successful mmap above.
        unsafe { libc::posix_madvise(addr, len, libc::POSIX_MADV_WILLNEED) };
        Ok(Self { addr, len })
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        // SAFETY: addr/len came from a successful mmap and are unmapped once.
        unsafe { libc::munmap(self.addr, self.len) };
    }
}

#[cfg(test)]
#[path = "shard_tests.rs"]
mod tests;
