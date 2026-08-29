// SPDX-License-Identifier: AGPL-3.0-only

//! NVMe-backed row cache for the n-gram embedding tables.
//!
//! The n-gram tables of the LongCat / Qwen3.8-Flash-Next family are the
//! model's largest tensors by far (31.4 B params on LongCat-Flash-Lite,
//! ~51 B announced for Flash-Next) and simultaneously its *least*
//! bandwidth-hungry: a token touches exactly one row per table — 12 rows,
//! ~3 KB — regardless of sequence length. Pure capacity, near-zero
//! bandwidth, which makes them the best demotion candidate in the model.
//!
//! Design, and why it needs no CUDA kernel change:
//!
//! * The cache is a flat PINNED arena of `slots × row_stride` bytes. On
//!   GB10 pinned host memory is GPU-addressable at the SAME virtual address
//!   ([`ExpertArena`] asserts this), so the arena *is* a
//!   `[slots, dim]` device-side table.
//! * The n-gram row ids are computed HOST-side (they are a pure function of
//!   token ids), so a lookup resolves `row_id -> slot` on the host and hands
//!   the gather kernel the SLOT INDEX in place of the row id. `batched_embed`
//!   / `batched_embed_fp8` then run verbatim against the arena base.
//! * A miss reads the row straight off NVMe into its pinned slot — no
//!   `cuMemcpyHtoD` anywhere on the path.
//!
//! Eviction is CLOCK (second-chance): O(1), no per-hit bookkeeping, and it
//! approximates LRU well for the power-law access pattern these tables have.
//! Rows touched by the CURRENT batch are pinned so a large prefill can never
//! evict a row it is still about to read.
//!
//! O_DIRECT requires 4 KiB-aligned reads, while a row is typically 256 B
//! (FP8, dim 256). Reads are therefore issued as the containing 4 KiB block
//! into a bounce buffer and the row copied out — the block is the disk's
//! minimum transfer anyway, so this costs no extra I/O, only a 256 B host
//! memcpy. Cache capacity stays row-granular, which matters because the
//! hash scatters ids: neighbouring rows in a table are unrelated.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::expert_arena::ExpertArena;

/// O_DIRECT transfer granularity (also `ExpertArena`'s stride requirement).
pub(crate) const BLOCK: usize = 4096;

/// One table's on-NVMe backing file plus its resident row cache.
pub struct NgramRowCache {
    /// Flat pinned, GPU-addressable `[slots, row_stride]` region.
    arena: ExpertArena,
    /// Backing file: row `i` at byte offset `base_offset + i * row_stride`.
    /// `base_offset` lets the cache read STRAIGHT OUT OF A SAFETENSORS SHARD
    /// — a table is already a contiguous row-major blob there, so no repack
    /// or re-save is needed. Because that offset is only 8-byte aligned, a
    /// row may straddle a 4 KiB O_DIRECT block; `fetch_into` handles the seam.
    file: File,
    /// Additional backing files, for a segmented table whose shards do NOT all
    /// live in one safetensors shard. Index 0 is `file`; `Segments::shard_file`
    /// indexes into this list.
    ///
    /// Qwen3.8-Flash-Next needs this and LongCat does not: the released
    /// NVFP4 checkpoint spreads its 128 PLE shards across TEN
    /// `model-plefp8-*.safetensors` files, so requiring one file refused the
    /// model outright ("PLE: shard 2 lives in a different file from shard 0").
    extra_files: Vec<File>,
    base_offset: u64,
    /// SEGMENTED tables: one base offset per equal-sized shard.
    ///
    /// LongCat ships each n-gram table as ONE contiguous safetensors tensor,
    /// so `base_offset` alone locates every row. Qwen3.8-Flash-Next splits its
    /// single 320M-row table across 128 shard tensors which are NOT laid out
    /// consecutively in the file — the shards interleave with other weights,
    /// so a global row id needs its shard's own base. `None` keeps the
    /// original single-offset behaviour byte for byte.
    segments: Option<Segments>,
    /// Per-row scale file mirror (FP8 tables), `None` for BF16 tables.
    scales: Option<ScaleCache>,
    row_stride: usize,
    slots: usize,
    rows_total: u64,
    /// row_id -> slot.
    map: HashMap<u64, u32>,
    /// slot -> resident row id (`u64::MAX` = empty).
    slot_row: Vec<u64>,
    /// CLOCK reference bits.
    refbit: Vec<bool>,
    /// Slots pinned for the batch in flight (never evicted).
    pinned: Vec<bool>,
    hand: usize,
    bounce: AlignedBlock,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

/// A table split across equal-sized shards at scattered file offsets.
struct Segments {
    /// Byte offset of each shard's first row, indexed by shard.
    bases: Vec<u64>,
    /// Which backing file each shard lives in: 0 is `file`, n>0 indexes
    /// `extra_files[n - 1]`. All zeroes when the table is one file, which is
    /// the LongCat shape and stays byte-for-byte what it was.
    shard_file: Vec<usize>,
    /// Rows per shard. Every shard but conceivably the last holds exactly
    /// this many; `open_segmented` requires them all equal so the mapping is
    /// a divide rather than a search.
    rows_per: u64,
}

/// Per-row f32 scales for an FP8 table, mirrored into a device-visible
/// `[slots]` array indexed by SLOT (parallel to the arena).
struct ScaleCache {
    arena: ExpertArena,
    /// `None` for a table whose scale is a single constant for every row: the
    /// arena is filled once at open and never faulted. The released
    /// Qwen3.8-Flash-Next PLE table is that shape — one
    /// `ngram_embedding.weight_scale`, BF16, shape [1] — while LongCat's is
    /// per-row and reads from this file.
    file: Option<File>,
}

/// A 4 KiB-aligned host buffer for O_DIRECT reads.
pub(crate) struct AlignedBlock {
    buf: Vec<u8>,
    off: usize,
}

impl AlignedBlock {
    /// Two blocks: a row whose base offset is not 4 KiB-aligned (every row of
    /// a table read in place from a safetensors shard) can straddle one
    /// boundary, and two blocks always cover it since `row_stride <= BLOCK`.
    pub(crate) fn new() -> Self {
        // Over-allocate and take an aligned window (portable, no libc::memalign).
        let buf = vec![0u8; BLOCK * 3];
        let addr = buf.as_ptr() as usize;
        let off = (BLOCK - (addr % BLOCK)) % BLOCK;
        Self { buf, off }
    }
    /// `n` whole blocks of aligned scratch (`n <= 2`).
    pub(crate) fn blocks(&mut self, n: usize) -> &mut [u8] {
        &mut self.buf[self.off..self.off + n * BLOCK]
    }
}

impl NgramRowCache {
    /// Open `path` as the backing store for a table of `rows_total` rows of
    /// `row_stride` bytes, caching `slots` of them in pinned GPU-addressable
    /// memory. `scale_path` supplies the per-row f32 scales of an FP8 table.
    pub fn open(
        path: &Path,
        scale_path: Option<&Path>,
        rows_total: u64,
        row_stride: usize,
        slots: usize,
    ) -> Result<Self> {
        Self::open_at(path, 0, scale_path, rows_total, row_stride, slots)
    }

    /// As [`Self::open`], but the table starts at `base_offset` inside the
    /// file — the safetensors-shard case (`data_offsets[0]` + the header
    /// length), which needs no re-save of the checkpoint.
    #[allow(clippy::too_many_arguments)]
    pub fn open_at(
        path: &Path,
        base_offset: u64,
        scale_path: Option<&Path>,
        rows_total: u64,
        row_stride: usize,
        slots: usize,
    ) -> Result<Self> {
        if row_stride == 0 || slots == 0 {
            bail!("NgramRowCache: zero geometry (row_stride={row_stride}, slots={slots})");
        }
        if row_stride > BLOCK {
            bail!(
                "NgramRowCache: row_stride {row_stride} exceeds the {BLOCK}-byte \
                 O_DIRECT block; a row would span more than the two blocks the \
                 seam-handling fetch reads"
            );
        }
        // One flat pinned region: `slots * row_stride` bytes, rounded up to the
        // arena's 4 KiB stride requirement.
        let bytes = slots * row_stride;
        let blocks = bytes.div_ceil(BLOCK);
        let arena =
            ExpertArena::new(1, blocks as u32, BLOCK).context("NgramRowCache: pinned arena")?;
        let file = open_direct(path)?;
        let scales = match scale_path {
            Some(sp) => {
                let sbytes = slots * 4;
                let sblocks = sbytes.div_ceil(BLOCK);
                Some(ScaleCache {
                    arena: ExpertArena::new(1, sblocks as u32, BLOCK)
                        .context("NgramRowCache: scale arena")?,
                    file: Some(open_direct(sp)?),
                })
            }
            None => None,
        };
        Ok(Self {
            arena,
            file,
            base_offset,
            segments: None,
            extra_files: Vec::new(),
            scales,
            row_stride,
            slots,
            rows_total,
            map: HashMap::with_capacity(slots * 2),
            slot_row: vec![u64::MAX; slots],
            refbit: vec![false; slots],
            pinned: vec![false; slots],
            hand: 0,
            bounce: AlignedBlock::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        })
    }

    /// As [`Self::open_at`], but for a table split across equal-sized shards
    /// at SCATTERED file offsets — Qwen3.8-Flash-Next's PLE table, whose 128
    /// shard tensors are not laid out consecutively inside the safetensors
    /// file. `bases[i]` is shard `i`'s first row; every shard holds
    /// `rows_per_shard` rows.
    #[allow(clippy::too_many_arguments)]
    pub fn open_segmented(
        shards: &[(std::path::PathBuf, u64)],
        rows_per_shard: u64,
        scale_path: Option<&Path>,
        row_stride: usize,
        slots: usize,
    ) -> Result<Self> {
        if shards.is_empty() || rows_per_shard == 0 {
            bail!(
                "NgramRowCache: segmented table needs shards and rows \
                 (shards={}, rows_per_shard={rows_per_shard})",
                shards.len()
            );
        }
        // One File per DISTINCT path, in first-seen order, so a table split
        // across ten files costs ten descriptors rather than one per shard.
        // Shard 0's file is the cache's own `file`; the rest are `extra_files`.
        let mut order: Vec<&Path> = Vec::new();
        let mut shard_file = Vec::with_capacity(shards.len());
        for (p, _) in shards {
            let idx = order
                .iter()
                .position(|q| *q == p.as_path())
                .unwrap_or_else(|| {
                    order.push(p.as_path());
                    order.len() - 1
                });
            shard_file.push(idx);
        }
        let bases: Vec<u64> = shards.iter().map(|(_, o)| *o).collect();
        let rows_total = bases.len() as u64 * rows_per_shard;
        let mut c = Self::open_at(order[0], 0, scale_path, rows_total, row_stride, slots)?;
        for p in &order[1..] {
            c.extra_files.push(open_direct(p)?);
        }
        c.segments = Some(Segments {
            bases,
            shard_file,
            rows_per: rows_per_shard,
        });
        Ok(c)
    }

    /// Device VA of the cache's row table — the `embed_table` argument of the
    /// gather kernels, which then index it by SLOT.
    pub fn table_dev_va(&self) -> Result<u64> {
        self.arena.slot_dev_va(0, 0)
    }

    /// Give every slot the same scale, for an FP8 table quantized with ONE
    /// factor rather than per row.
    ///
    /// Filled once here instead of faulted per row: the value does not depend
    /// on which row landed in the slot, so a per-fault read would be the same
    /// four bytes fetched again for every miss. The gather kernel is the FP8
    /// one either way — it multiplies by `scales[slot]` and does not care where
    /// that came from.
    ///
    /// # Errors
    /// If the scale arena cannot be allocated.
    pub fn set_constant_scale(&mut self, scale: f32) -> Result<()> {
        let sbytes = self.slots * 4;
        let sblocks = sbytes.div_ceil(BLOCK);
        let arena = ExpertArena::new(1, sblocks as u32, BLOCK)
            .context("NgramRowCache: constant scale arena")?;
        let p = arena.slot_host_ptr(0, 0)?.cast::<f32>();
        for i in 0..self.slots {
            // SAFETY: the arena is at least `slots * 4` bytes and was just
            // allocated here, so nothing else holds a reference into it.
            unsafe { p.add(i).write(scale) };
        }
        self.scales = Some(ScaleCache { arena, file: None });
        Ok(())
    }

    /// Device VA of the `[slots]` f32 scale array (FP8 tables only).
    pub fn scale_dev_va(&self) -> Result<Option<u64>> {
        match &self.scales {
            Some(s) => Ok(Some(s.arena.slot_dev_va(0, 0)?)),
            None => Ok(None),
        }
    }

    pub fn stats(&self) -> (u64, u64, u64) {
        (self.hits, self.misses, self.evictions)
    }

    /// Resolve `row_ids` to slot indices, faulting misses in from NVMe.
    ///
    /// Every returned slot is PINNED for the caller's batch: the gather runs
    /// after this returns, so a later resolve in the same batch must not
    /// evict a row the kernel is about to read. Call [`Self::end_batch`] once
    /// the gather has been issued.
    pub fn resolve(&mut self, row_ids: &[u64], out_slots: &mut Vec<u32>) -> Result<()> {
        out_slots.clear();
        out_slots.reserve(row_ids.len());
        // Phase 1 — bookkeeping only: pin hits, assign a victim slot to every
        // miss (a repeated missing id hits the map on its second occurrence,
        // so each unique row faults once). No I/O under this loop.
        let mut jobs: Vec<crate::ngram_cache_fault::FaultJob> = Vec::new();
        for &id in row_ids {
            if id >= self.rows_total {
                bail!(
                    "NgramRowCache: row id {id} >= table rows {} (hash/table mismatch)",
                    self.rows_total
                );
            }
            let slot = match self.map.get(&id) {
                Some(&s) => {
                    self.hits += 1;
                    self.refbit[s as usize] = true;
                    self.pinned[s as usize] = true;
                    s
                }
                None => {
                    self.misses += 1;
                    let s = self.victim()?;
                    self.map.insert(id, s);
                    self.slot_row[s as usize] = id;
                    self.refbit[s as usize] = true;
                    self.pinned[s as usize] = true;
                    jobs.push(self.fault_job(id, s)?);
                    s
                }
            };
            out_slots.push(slot);
        }
        // Phase 2 — fault every miss in, parallel past a few (the serial
        // QD=1 pread-per-miss loop was the diverse-prefill stall).
        if !jobs.is_empty() {
            // Every backing file, in index order, so a job can name its own.
            let files: Vec<&File> = std::iter::once(&self.file)
                .chain(self.extra_files.iter())
                .collect();
            let r = crate::ngram_cache_fault::fault_all(
                &jobs,
                &files,
                self.scales.as_ref().and_then(|sc| sc.file.as_ref()),
                self.row_stride,
                &mut self.bounce,
            );
            if let Err(e) = r {
                // Roll the failed batch's map entries back: they were
                // inserted in phase 1 and now describe slots holding garbage.
                for j in &jobs {
                    self.map.remove(&j.row_id);
                    self.slot_row[j.slot as usize] = u64::MAX;
                    self.pinned[j.slot as usize] = false;
                    self.refbit[j.slot as usize] = false;
                }
                return Err(e);
            }
        }
        Ok(())
    }

    /// Resolve one miss to byte offsets + destination addresses — the
    /// bookkeeping-free half of the old `fetch_into`, consumed by
    /// [`crate::ngram_cache_fault::fault_all`].
    fn fault_job(&self, id: u64, slot: u32) -> Result<crate::ngram_cache_fault::FaultJob> {
        let (file_idx, byte) = self.row_byte(id);
        let block_off = byte - (byte % BLOCK as u64);
        let within = (byte - block_off) as usize;
        let nblocks = crate::ngram_cache_fault::nblocks_for(within, self.row_stride);
        // SAFETY: address arithmetic only; the fault worker writes the
        // disjoint `[dst, dst+row_stride)` region while the arena is live.
        let dst = unsafe {
            self.arena
                .slot_host_ptr(0, 0)?
                .add(slot as usize * self.row_stride)
        } as usize;
        let scale = match &self.scales {
            // A constant scale has no file and never faults: the arena was
            // filled at open and every slot already holds the right value.
            Some(sc) if sc.file.is_some() => {
                let sbyte = id * 4;
                let sblock = sbyte - (sbyte % BLOCK as u64);
                let swithin = (sbyte - sblock) as usize;
                // SAFETY: as above, 4-byte disjoint region.
                let sdst = unsafe { sc.arena.slot_host_ptr(0, 0)?.add(slot as usize * 4) };
                Some((sblock, swithin, sdst as usize))
            }
            // No scales at all, or a constant one already resident.
            Some(_) | None => None,
        };
        Ok(crate::ngram_cache_fault::FaultJob {
            row_id: id,
            slot,
            block_off,
            within,
            nblocks,
            dst,
            scale,
            file_idx,
        })
    }

    /// Release the batch's pins (call after the gather kernels are issued).
    pub fn end_batch(&mut self) {
        for p in &mut self.pinned {
            *p = false;
        }
    }

    /// CLOCK second-chance victim among the unpinned slots.
    fn victim(&mut self) -> Result<u32> {
        for _ in 0..(self.slots * 2) {
            let s = self.hand;
            self.hand = (self.hand + 1) % self.slots;
            if self.pinned[s] {
                continue;
            }
            if self.refbit[s] {
                self.refbit[s] = false;
                continue;
            }
            if self.slot_row[s] != u64::MAX {
                let old = self.slot_row[s];
                self.map.remove(&old);
                self.evictions += 1;
            }
            return Ok(s as u32);
        }
        bail!(
            "NgramRowCache: every one of {} slots is pinned by the batch in flight — \
             raise the cache size or lower max-prefill-tokens",
            self.slots
        )
    }

    /// Byte offset of row `id`, and which backing file holds it.
    ///
    /// The file index is part of the answer because a segmented table's shards
    /// may live in different safetensors files — an offset alone named a byte
    /// in the wrong one.
    fn row_byte(&self, id: u64) -> (usize, u64) {
        match &self.segments {
            None => (0, self.base_offset + id * self.row_stride as u64),
            Some(seg) => {
                let shard = (id / seg.rows_per) as usize;
                let local = id % seg.rows_per;
                (
                    seg.shard_file[shard],
                    seg.bases[shard] + local * self.row_stride as u64,
                )
            }
        }
    }
}

#[cfg(unix)]
fn open_direct(path: &Path) -> Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .with_context(|| format!("NgramRowCache: open {} (O_DIRECT)", path.display()))
}

#[cfg(not(unix))]
fn open_direct(path: &Path) -> Result<File> {
    File::open(path).with_context(|| format!("NgramRowCache: open {}", path.display()))
}

#[cfg(test)]
#[path = "ngram_cache/tests.rs"]
mod tests;
