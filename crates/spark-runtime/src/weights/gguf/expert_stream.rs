// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! DeepSeek-V4.1 Flash expert streaming from the split GGUF, by positional read.
//!
//! The routed experts of the Q2_K checkpoint are 40 layers x 384 experts of
//! 12.22 MiB each; they never fit and they are never expanded. Measured on the
//! real shards (`deepseek_v41_stream_bench_test.rs`): a CPU dequant of one expert
//! is 851 ms, and page-faulting an expert slice in through the mmap runs at
//! 0.33 GB/s cold and cached alike, while `pread` of the same 3.7 MiB slices
//! runs at 4.5 GB/s cold. So the loader's mmap is used for the header only, and
//! the expert bytes are read by [`pread`] straight into a page-locked arena the
//! GPU addresses in place (see [`super::expert_lru::ExpertLru`]); the K-quant
//! kernels consume the raw blocks from there.
//!
//! Three pieces:
//!   * [`ShardFiles`]: every shard opened once, header parsed, mmap dropped.
//!   * [`ExpertSliceMap`]: where expert `e` of layer `l` lives for each of the
//!     three stacked tensors (`blk.N.ffn_{gate,up,down}_exps.weight`, GGUF dims
//!     `[k, n, experts]`), and the read of one expert into one slot.
//!   * [`EngramRowReader`]: the ~30 GiB engram tables, one Q2_K block (84 B) per
//!     row, read by row id on demand. 48 rows per token.
//!
//! Oracle: bytes identical to the mmap path, held by
//! `deepseek_v41_stream_oracle_test.rs` on the real shards.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};

use super::container::{GgufFile, Q2Group, TensorInfo};
use super::sidecar;
use crate::weights::{find_gguf, find_gguf_shards};

pub use super::expert_lru::{ExpertLru, ExpertSlot, LruStats, PinnedArena};

/// Positional read of exactly `dst.len()` bytes at `offset`. No file position
/// is shared, so any number of threads may read one `File` at once.
pub fn pread(file: &File, offset: u64, dst: &mut [u8]) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(dst, offset)
            .with_context(|| format!("pread {} bytes at {offset}", dst.len()))
    }
    #[cfg(not(unix))]
    {
        let _ = (file, offset, dst);
        bail!("expert streaming needs positional reads (unix)")
    }
}

/// `pread`, then drop the range from the page cache: the bytes now live in
/// the arena and the cache copy is dead weight. With a 100 GiB page-locked
/// arena on a 121 GiB Spark, leaving 12 MiB of cache behind every miss made
/// the kernel reclaim into swap in the middle of a step (single steps of 2 to
/// 18 s on the 09-19 standard). `ATLAS_DS41_KEEP_PAGE_CACHE=1` keeps the old
/// behaviour.
pub fn pread_uncached(file: &File, offset: u64, dst: &mut [u8]) -> Result<()> {
    pread(file, offset, dst)?;
    #[cfg(target_os = "linux")]
    if !keep_page_cache() {
        use std::os::unix::io::AsRawFd;
        // SAFETY: an advisory call on an open descriptor; the kernel checks
        // the range.
        unsafe {
            libc::posix_fadvise(
                file.as_raw_fd(),
                offset as libc::off_t,
                dst.len() as libc::off_t,
                libc::POSIX_FADV_DONTNEED,
            );
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn keep_page_cache() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("ATLAS_DS41_KEEP_PAGE_CACHE").is_ok_and(|v| v == "1"))
}

struct Shard {
    path: PathBuf,
    file: File,
    gguf: GgufFile,
}

/// Every shard of a split GGUF, opened once for the process lifetime. The
/// header is parsed through the loader's own reader and the mmap is dropped:
/// nothing here goes through the page cache but the reads we ask for.
pub struct ShardFiles {
    shards: Vec<Shard>,
}

impl ShardFiles {
    /// Open the shard set that `first` (the `split.no == 0` file) names.
    pub fn open(first: &Path) -> Result<Self> {
        let set = find_gguf_shards(first)?;
        let mut shards = Vec::with_capacity(set.paths.len());
        for path in set.paths {
            let (file, mmap, gguf) = sidecar::open_gguf(&path)?;
            drop(mmap);
            shards.push(Shard { path, file, gguf });
        }
        Ok(ShardFiles { shards })
    }

    /// Open the model in `dir` (shard 0 found as the loader finds it).
    pub fn open_dir(dir: &Path) -> Result<Self> {
        let first = find_gguf(dir).with_context(|| format!("no GGUF in {}", dir.display()))?;
        Self::open(&first)
    }

    pub fn len(&self) -> usize {
        self.shards.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shards.is_empty()
    }

    pub fn path(&self, shard: usize) -> &Path {
        &self.shards[shard].path
    }

    pub fn file(&self, shard: usize) -> &File {
        &self.shards[shard].file
    }

    /// The parsed header of `shard` (shard 0 carries the model metadata).
    pub fn header(&self, shard: usize) -> &GgufFile {
        &self.shards[shard].gguf
    }

    /// `(shard, tensor, absolute byte offset)` of a tensor by GGUF name.
    pub fn locate(&self, name: &str) -> Option<(usize, &TensorInfo, u64)> {
        self.shards.iter().enumerate().find_map(|(i, s)| {
            s.gguf
                .tensor(name)
                .map(|t| (i, t, s.gguf.tensor_abs_offset(t) as u64))
        })
    }

    /// Every `(shard, tensor)` whose name ends with `suffix`.
    fn with_suffix<'a>(
        &'a self,
        suffix: &'a str,
    ) -> impl Iterator<Item = (usize, &'a TensorInfo)> + 'a {
        self.shards.iter().enumerate().flat_map(move |(i, s)| {
            s.gguf
                .tensors
                .iter()
                .filter(move |t| t.name.ends_with(suffix))
                .map(move |t| (i, t))
        })
    }

    fn abs_offset(&self, shard: usize, t: &TensorInfo) -> u64 {
        self.shards[shard].gguf.tensor_abs_offset(t) as u64
    }
}

/// `blk.N.<rest>` -> `N`.
fn block_layer(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("blk.")?;
    let end = rest.find('.')?;
    rest[..end].parse().ok()
}

/// Where one expert of one stacked projection lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliceLoc {
    pub shard: usize,
    /// Absolute byte offset of expert 0; expert `e` is at `base + e * bytes`.
    pub base: u64,
    /// Bytes per expert on disk.
    pub bytes: usize,
    /// Elements per expert (`k * n`).
    pub elems: usize,
    pub ggml_type_id: u32,
}

/// The three projections of one MoE layer.
#[derive(Clone, Copy, Debug)]
pub struct ExpertLayerLoc {
    pub layer: usize,
    pub gate: SliceLoc,
    pub up: SliceLoc,
    pub down: SliceLoc,
}

/// Byte layout of one expert inside an LRU slot: `[gate | up | down]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotLayout {
    pub gate_off: usize,
    pub up_off: usize,
    pub down_off: usize,
    pub gate_bytes: usize,
    pub up_bytes: usize,
    pub down_bytes: usize,
    /// Total bytes per expert = one slot.
    pub bytes: usize,
}

impl SlotLayout {
    pub fn new(gate_bytes: usize, up_bytes: usize, down_bytes: usize) -> Self {
        SlotLayout {
            gate_off: 0,
            up_off: gate_bytes,
            down_off: gate_bytes + up_bytes,
            gate_bytes,
            up_bytes,
            down_bytes,
            bytes: gate_bytes + up_bytes + down_bytes,
        }
    }
}

/// Anything the LRU can fill a slot from. The on-disk map implements it; the
/// unit tests implement it in memory.
pub trait ExpertSource: Sync {
    fn slot_layout(&self) -> SlotLayout;
    fn num_experts(&self) -> usize;
    /// Fill `dst` (exactly `slot_layout().bytes` long) with expert `expert` of
    /// layer `layer`, in slot layout.
    fn read_expert(&self, layer: u32, expert: u32, dst: &mut [u8]) -> Result<()>;
    /// Fill `dst` with bytes `off..off + dst.len()` of the expert's slot
    /// image, so one miss can be read by several threads at once. The
    /// default reads the whole expert into scratch; the on-disk map seeks.
    fn read_expert_range(&self, layer: u32, expert: u32, off: usize, dst: &mut [u8]) -> Result<()> {
        let mut whole = vec![0u8; self.slot_layout().bytes];
        self.read_expert(layer, expert, &mut whole)?;
        dst.copy_from_slice(&whole[off..off + dst.len()]);
        Ok(())
    }
}

/// The stacked expert tensors of every MoE layer, located once.
pub struct ExpertSliceMap {
    files: Arc<ShardFiles>,
    layers: Vec<ExpertLayerLoc>,
    by_layer: HashMap<usize, usize>,
    num_experts: usize,
    layout: SlotLayout,
}

impl ExpertSliceMap {
    pub fn new(files: Arc<ShardFiles>) -> Result<Self> {
        let mut parts: HashMap<usize, [Option<SliceLoc>; 3]> = HashMap::new();
        let mut num_experts: Option<usize> = None;
        for (which, suffix) in [
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        ]
        .iter()
        .enumerate()
        {
            for (shard, t) in files.with_suffix(suffix) {
                let Some(layer) = block_layer(&t.name) else {
                    continue;
                };
                ensure!(
                    t.dims.len() == 3,
                    "{}: expected a stacked 3-D expert tensor, got {:?}",
                    t.name,
                    t.dims
                );
                let (k, n, experts) = (t.dims[0], t.dims[1], t.dims[2]);
                match num_experts {
                    None => num_experts = Some(experts),
                    Some(e) => ensure!(
                        e == experts,
                        "{}: {experts} experts, others have {e}",
                        t.name
                    ),
                }
                let (qk, bb) = t.ggml_type.block_layout(Q2Group::G128)?;
                let elems = k * n;
                ensure!(
                    elems.is_multiple_of(qk),
                    "{}: {elems} elements not a multiple of the {qk}-block",
                    t.name
                );
                let loc = SliceLoc {
                    shard,
                    base: files.abs_offset(shard, t),
                    bytes: elems / qk * bb,
                    elems,
                    ggml_type_id: t.ggml_type.id(),
                };
                let slot = parts.entry(layer).or_default();
                ensure!(slot[which].is_none(), "{}: duplicate", t.name);
                slot[which] = Some(loc);
            }
        }
        let Some(num_experts) = num_experts else {
            bail!("no ffn_*_exps tensors in this GGUF")
        };
        let mut layers: Vec<ExpertLayerLoc> = parts
            .into_iter()
            .map(|(layer, [g, u, d])| {
                let need = |x: Option<SliceLoc>, w: &str| {
                    x.with_context(|| format!("layer {layer}: ffn_{w}_exps missing"))
                };
                Ok(ExpertLayerLoc {
                    layer,
                    gate: need(g, "gate")?,
                    up: need(u, "up")?,
                    down: need(d, "down")?,
                })
            })
            .collect::<Result<_>>()?;
        layers.sort_by_key(|l| l.layer);
        let first = layers[0];
        let layout = SlotLayout::new(first.gate.bytes, first.up.bytes, first.down.bytes);
        for l in &layers {
            ensure!(
                (l.gate.bytes, l.up.bytes, l.down.bytes)
                    == (layout.gate_bytes, layout.up_bytes, layout.down_bytes),
                "layer {}: expert byte sizes differ from layer {}",
                l.layer,
                first.layer
            );
        }
        let by_layer = layers
            .iter()
            .enumerate()
            .map(|(i, l)| (l.layer, i))
            .collect();
        Ok(ExpertSliceMap {
            files,
            layers,
            by_layer,
            num_experts,
            layout,
        })
    }

    /// MoE layers in ascending order.
    pub fn layers(&self) -> &[ExpertLayerLoc] {
        &self.layers
    }

    pub fn layer(&self, layer: usize) -> Option<&ExpertLayerLoc> {
        self.by_layer.get(&layer).map(|&i| &self.layers[i])
    }

    pub fn files(&self) -> &Arc<ShardFiles> {
        &self.files
    }

    /// Absolute `(shard, offset)` of one projection of one expert.
    pub fn slice_at(&self, loc: &SliceLoc, expert: usize) -> (usize, u64) {
        (loc.shard, loc.base + (expert * loc.bytes) as u64)
    }
}

impl ExpertSource for ExpertSliceMap {
    fn slot_layout(&self) -> SlotLayout {
        self.layout
    }

    fn num_experts(&self) -> usize {
        self.num_experts
    }

    fn read_expert(&self, layer: u32, expert: u32, dst: &mut [u8]) -> Result<()> {
        let l = self
            .layer(layer as usize)
            .with_context(|| format!("layer {layer} has no routed experts"))?;
        ensure!(
            (expert as usize) < self.num_experts,
            "expert {expert} >= {}",
            self.num_experts
        );
        ensure!(
            dst.len() == self.layout.bytes,
            "slot is {} bytes, expert is {}",
            dst.len(),
            self.layout.bytes
        );
        let lay = self.layout;
        for (loc, off, len) in [
            (&l.gate, lay.gate_off, lay.gate_bytes),
            (&l.up, lay.up_off, lay.up_bytes),
            (&l.down, lay.down_off, lay.down_bytes),
        ] {
            let (shard, at) = self.slice_at(loc, expert as usize);
            pread_uncached(self.files.file(shard), at, &mut dst[off..off + len])
                .with_context(|| format!("layer {layer} expert {expert} shard {shard}"))?;
        }
        Ok(())
    }

    fn read_expert_range(&self, layer: u32, expert: u32, off: usize, dst: &mut [u8]) -> Result<()> {
        let l = self
            .layer(layer as usize)
            .with_context(|| format!("layer {layer} has no routed experts"))?;
        ensure!(
            (expert as usize) < self.num_experts,
            "expert {expert} >= {}",
            self.num_experts
        );
        let lay = self.layout;
        let (lo, hi) = (off, off + dst.len());
        ensure!(
            hi <= lay.bytes,
            "range {lo}..{hi} exceeds the {} byte slot",
            lay.bytes
        );
        for (loc, s_off, s_len) in [
            (&l.gate, lay.gate_off, lay.gate_bytes),
            (&l.up, lay.up_off, lay.up_bytes),
            (&l.down, lay.down_off, lay.down_bytes),
        ] {
            let (a, b) = (lo.max(s_off), hi.min(s_off + s_len));
            if a >= b {
                continue;
            }
            let (shard, at) = self.slice_at(loc, expert as usize);
            pread_uncached(
                self.files.file(shard),
                at + (a - s_off) as u64,
                &mut dst[a - lo..b - lo],
            )
            .with_context(|| {
                format!("layer {layer} expert {expert} shard {shard} range {a}..{b}")
            })?;
        }
        Ok(())
    }
}

/// One engram table: `blk.N.engram_embd.weight`, GGUF dims `[head_dim, rows]`,
/// one quant block per row.
#[derive(Clone, Copy, Debug)]
pub struct EngramTable {
    pub layer: usize,
    pub shard: usize,
    pub base: u64,
    pub rows: usize,
    pub row_bytes: usize,
    pub head_dim: usize,
    pub ggml_type_id: u32,
}

/// Row reads from the engram tables. The tables are ~30 GiB each and a token
/// touches 24 rows per table, so nothing is cached: every call is `ids.len()`
/// positional reads of one block.
pub struct EngramRowReader {
    files: Arc<ShardFiles>,
    tables: Vec<EngramTable>,
}

impl EngramRowReader {
    pub fn new(files: Arc<ShardFiles>) -> Result<Self> {
        let mut tables = Vec::new();
        for (shard, t) in files.with_suffix("engram_embd.weight") {
            let Some(layer) = block_layer(&t.name) else {
                continue;
            };
            ensure!(
                t.dims.len() == 2,
                "{}: expected [head_dim, rows], got {:?}",
                t.name,
                t.dims
            );
            let (head_dim, rows) = (t.dims[0], t.dims[1]);
            let (qk, bb) = t.ggml_type.block_layout(Q2Group::G128)?;
            ensure!(
                qk == head_dim,
                "{}: head_dim {head_dim} is not one {qk}-element block per row",
                t.name
            );
            tables.push(EngramTable {
                layer,
                shard,
                base: files.abs_offset(shard, t),
                rows,
                row_bytes: bb,
                head_dim,
                ggml_type_id: t.ggml_type.id(),
            });
        }
        ensure!(!tables.is_empty(), "no engram_embd tensors in this GGUF");
        tables.sort_by_key(|t| t.layer);
        Ok(EngramRowReader { files, tables })
    }

    pub fn tables(&self) -> &[EngramTable] {
        &self.tables
    }

    pub fn table(&self, layer: usize) -> Option<&EngramTable> {
        self.tables.iter().find(|t| t.layer == layer)
    }

    /// Read rows `ids` of layer `layer`'s table into `dst`
    /// (`ids.len() * row_bytes` bytes, in `ids` order).
    pub fn read_rows(&self, layer: usize, ids: &[u64], dst: &mut [u8]) -> Result<()> {
        let t = self
            .table(layer)
            .with_context(|| format!("layer {layer} has no engram table"))?;
        ensure!(
            dst.len() == ids.len() * t.row_bytes,
            "dst is {} bytes for {} rows of {}",
            dst.len(),
            ids.len(),
            t.row_bytes
        );
        let file = self.files.file(t.shard);
        for (i, &id) in ids.iter().enumerate() {
            ensure!(
                (id as usize) < t.rows,
                "engram row {id} >= {} (layer {layer})",
                t.rows
            );
            let off = t.base + id * t.row_bytes as u64;
            pread(file, off, &mut dst[i * t.row_bytes..(i + 1) * t.row_bytes])
                .with_context(|| format!("engram layer {layer} row {id}"))?;
        }
        Ok(())
    }
}
