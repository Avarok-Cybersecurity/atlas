// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the safetensors side of the EXL3 reader: find a checkpoint's shard files, read
//! ONLY their JSON headers (no mmap, so a scan touches no data pages, which matters on unified-memory
//! targets), validate every entry's offsets against the file, and read one tensor's bytes on demand
//! with a positioned read.
//!
//! Shard discovery: `model.safetensors.index.json`'s `weight_map` when present (its file names must be
//! plain names in the directory), else every `*.safetensors` file in the directory, sorted. A tensor
//! name found in two shards is an error. Header entries are checked with overflow-checked arithmetic:
//! `end >= begin`, the data inside the file, and for a known dtype `end - begin == numel * size`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use safetensors::Dtype;

/// The safetensors crate's header size limit (100 MB).
const MAX_HEADER_BYTES: u64 = 100_000_000;

/// One tensor of a shard set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardEntry {
    /// Index into [`ShardSet::files`].
    pub shard: usize,
    /// The dtype, when the safetensors crate knows its name.
    pub dtype: Option<Dtype>,
    /// The dtype as written in the header.
    pub dtype_name: String,
    /// The shape as written.
    pub shape: Vec<usize>,
    /// Absolute byte offset of the data in the shard file.
    pub offset: u64,
    /// Bytes of data.
    pub len: u64,
}

impl ShardEntry {
    /// Elements (the product of the shape), if it does not overflow.
    pub fn numel(&self) -> Option<usize> {
        self.shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d))
    }

    /// True when the entry has exactly `dtype` and `shape`.
    pub fn is(&self, dtype: Dtype, shape: &[usize]) -> bool {
        self.dtype == Some(dtype) && self.shape == shape
    }
}

/// The headers of every shard of a checkpoint.
#[derive(Debug)]
pub struct ShardSet {
    dir: PathBuf,
    files: Vec<PathBuf>,
    entries: BTreeMap<String, ShardEntry>,
}

impl ShardSet {
    /// Reads the headers of `dir`'s shards (see the module doc for discovery).
    pub fn open(dir: &Path) -> Result<Self> {
        let files = shard_files(dir)?;
        Self::from_files(dir, files)
    }

    /// Reads the headers of the given shard files.
    pub fn from_files(dir: &Path, files: Vec<PathBuf>) -> Result<Self> {
        let mut entries = BTreeMap::new();
        for (shard, path) in files.iter().enumerate() {
            for (name, entry) in read_header(path, shard)? {
                if let Some(prev) = entries.insert(name.clone(), entry) {
                    bail!(
                        "tensor {name} is in two shards: {} and {}",
                        files[prev.shard].display(),
                        path.display()
                    );
                }
            }
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            files,
            entries,
        })
    }

    /// The checkpoint directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The shard files, in discovery order.
    pub fn files(&self) -> &[PathBuf] {
        &self.files
    }

    /// The entry of tensor `name`.
    pub fn get(&self, name: &str) -> Option<&ShardEntry> {
        self.entries.get(name)
    }

    /// True when the checkpoint has tensor `name`.
    pub fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Every tensor, sorted by name.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ShardEntry)> {
        self.entries.iter().map(|(n, e)| (n.as_str(), e))
    }

    /// Tensors in the set.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when the set has no tensors.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The bytes of `entry` (one positioned read; the file is opened for the call).
    pub fn read(&self, entry: &ShardEntry) -> Result<Vec<u8>> {
        let path = self
            .files
            .get(entry.shard)
            .context("EXL3 shard index out of range")?;
        let len = usize::try_from(entry.len).context("tensor larger than the address space")?;
        let mut file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        file.seek(SeekFrom::Start(entry.offset))?;
        let mut out = vec![0u8; len];
        file.read_exact(&mut out)
            .with_context(|| format!("short read of {len} bytes in {}", path.display()))?;
        Ok(out)
    }

    /// Advises the OS to drop every shard's cached pages (see `weights::evict_page_cache`); a loader
    /// calls it after it has copied the tensors it needs.
    pub fn release_page_cache(&self) {
        for path in &self.files {
            if let Ok(file) = File::open(path) {
                crate::weights::evict_page_cache(&file);
            }
        }
    }
}

/// The shard files of `dir`.
pub fn shard_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if index.is_file() {
        let text = std::fs::read_to_string(&index)
            .with_context(|| format!("failed to read {}", index.display()))?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse {}", index.display()))?;
        let map = json
            .get("weight_map")
            .and_then(|m| m.as_object())
            .with_context(|| format!("{} has no weight_map object", index.display()))?;
        let mut names = BTreeSet::new();
        for (tensor, file) in map {
            let file = file
                .as_str()
                .with_context(|| format!("weight_map entry {tensor} is not a string"))?;
            let plain = Path::new(file).components().count() == 1
                && !file.starts_with('.')
                && file.ends_with(".safetensors");
            if !plain {
                bail!(
                    "weight_map entry {tensor} names {file:?}, not a shard file in the directory"
                );
            }
            names.insert(file.to_string());
        }
        return Ok(names.into_iter().map(|n| dir.join(n)).collect());
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("failed to list {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        bail!("{} has no safetensors files", dir.display());
    }
    Ok(files)
}

/// The validated entries of one shard's header.
pub fn read_header(path: &Path, shard: usize) -> Result<Vec<(String, ShardEntry)>> {
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let file_len = file.metadata()?.len();
    let mut size = [0u8; 8];
    file.read_exact(&mut size)
        .with_context(|| format!("{} is too short for a safetensors header", path.display()))?;
    let header_len = u64::from_le_bytes(size);
    if header_len > MAX_HEADER_BYTES || header_len.checked_add(8).is_none_or(|e| e > file_len) {
        bail!(
            "{}: header of {header_len} bytes does not fit a {file_len}-byte file",
            path.display()
        );
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)?;
    let json: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header)
        .with_context(|| format!("{}: header is not a JSON object", path.display()))?;
    let data_start = 8 + header_len;
    let data_len = file_len - data_start;
    let mut out = Vec::with_capacity(json.len());
    for (name, info) in json {
        if name == "__metadata__" {
            continue;
        }
        let entry = parse_entry(&name, &info, shard, data_start, data_len)
            .with_context(|| format!("{}: bad header entry {name}", path.display()))?;
        out.push((name, entry));
    }
    Ok(out)
}

fn parse_entry(
    name: &str,
    info: &serde_json::Value,
    shard: usize,
    data_start: u64,
    data_len: u64,
) -> Result<ShardEntry> {
    let dtype_name = info
        .get("dtype")
        .and_then(|d| d.as_str())
        .context("missing dtype")?
        .to_string();
    let dtype = serde_json::from_value::<Dtype>(serde_json::Value::String(dtype_name.clone())).ok();
    let shape = info
        .get("shape")
        .and_then(|s| s.as_array())
        .context("missing shape")?
        .iter()
        .map(|d| {
            d.as_u64()
                .and_then(|d| usize::try_from(d).ok())
                .context("shape entry is not a non-negative integer")
        })
        .collect::<Result<Vec<usize>>>()?;
    let offsets = info
        .get("data_offsets")
        .and_then(|o| o.as_array())
        .context("missing data_offsets")?;
    let (Some(begin), Some(end)) = (
        offsets.first().and_then(|v| v.as_u64()),
        offsets.get(1).and_then(|v| v.as_u64()),
    ) else {
        bail!("data_offsets of {name} is not [begin, end]");
    };
    if offsets.len() != 2 || end < begin || end > data_len {
        bail!("data_offsets [{begin}, {end}] of {name} outside the {data_len} data bytes");
    }
    let entry = ShardEntry {
        shard,
        dtype,
        dtype_name,
        shape,
        offset: data_start + begin,
        len: end - begin,
    };
    if let Some(dt) = entry.dtype {
        let bytes = entry
            .numel()
            .and_then(|n| n.checked_mul(dt.bitsize()))
            .map(|bits| bits.div_ceil(8) as u64);
        if bytes != Some(entry.len) {
            bail!(
                "{name}: {} bytes for shape {:?} of {}",
                entry.len,
                entry.shape,
                entry.dtype_name
            );
        }
    }
    Ok(entry)
}
