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
const BLOCK: usize = 4096;

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
    base_offset: u64,
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

/// Per-row f32 scales for an FP8 table, mirrored into a device-visible
/// `[slots]` array indexed by SLOT (parallel to the arena).
struct ScaleCache {
    arena: ExpertArena,
    file: File,
}

/// A 4 KiB-aligned host buffer for O_DIRECT reads.
struct AlignedBlock {
    buf: Vec<u8>,
    off: usize,
}

impl AlignedBlock {
    /// Two blocks: a row whose base offset is not 4 KiB-aligned (every row of
    /// a table read in place from a safetensors shard) can straddle one
    /// boundary, and two blocks always cover it since `row_stride <= BLOCK`.
    fn new() -> Self {
        // Over-allocate and take an aligned window (portable, no libc::memalign).
        let buf = vec![0u8; BLOCK * 3];
        let addr = buf.as_ptr() as usize;
        let off = (BLOCK - (addr % BLOCK)) % BLOCK;
        Self { buf, off }
    }
    /// `n` whole blocks of aligned scratch (`n <= 2`).
    fn blocks(&mut self, n: usize) -> &mut [u8] {
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
                    file: open_direct(sp)?,
                })
            }
            None => None,
        };
        Ok(Self {
            arena,
            file,
            base_offset,
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

    /// Device VA of the cache's row table — the `embed_table` argument of the
    /// gather kernels, which then index it by SLOT.
    pub fn table_dev_va(&self) -> Result<u64> {
        self.arena.slot_dev_va(0, 0)
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
                    self.fetch_into(id, s)?;
                    s
                }
            };
            out_slots.push(slot);
        }
        Ok(())
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

    /// Read row `id` off NVMe straight into `slot`'s pinned (GPU-addressable)
    /// bytes, via the containing 4 KiB block.
    fn fetch_into(&mut self, id: u64, slot: u32) -> Result<()> {
        let byte = self.base_offset + id * self.row_stride as u64;
        let block_off = byte - (byte % BLOCK as u64);
        let within = (byte - block_off) as usize;
        // One block unless the row crosses the boundary (possible whenever the
        // table's base offset is not 4 KiB-aligned, i.e. reading in place from
        // a safetensors shard).
        let nblocks = if within + self.row_stride > BLOCK {
            2
        } else {
            1
        };
        atlas_tier::pio::read_exact_at(&self.file, self.bounce.blocks(nblocks), block_off)
            .with_context(|| format!("NgramRowCache: read row {id}"))?;
        // SAFETY: slot < self.slots and the arena holds slots*row_stride bytes.
        let dst = unsafe {
            let base = self.arena.slot_host_ptr(0, 0)?;
            std::slice::from_raw_parts_mut(
                base.add(slot as usize * self.row_stride),
                self.row_stride,
            )
        };
        dst.copy_from_slice(&self.bounce.blocks(nblocks)[within..within + self.row_stride]);

        if let Some(sc) = &self.scales {
            let sbyte = id * 4;
            let sblock = sbyte - (sbyte % BLOCK as u64);
            let swithin = (sbyte - sblock) as usize;
            atlas_tier::pio::read_exact_at(&sc.file, self.bounce.blocks(1), sblock)
                .with_context(|| format!("NgramRowCache: read scale {id}"))?;
            // SAFETY: slot < slots, scale arena holds slots*4 bytes.
            let sdst = unsafe {
                let base = sc.arena.slot_host_ptr(0, 0)?;
                std::slice::from_raw_parts_mut(base.add(slot as usize * 4), 4)
            };
            sdst.copy_from_slice(&self.bounce.blocks(1)[swithin..swithin + 4]);
        }

        self.map.insert(id, slot);
        self.slot_row[slot as usize] = id;
        self.refbit[slot as usize] = true;
        self.pinned[slot as usize] = true;
        Ok(())
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
mod tests {
    use super::*;

    #[test]
    fn aligned_scratch_is_4k_aligned_and_two_blocks() {
        let mut b = AlignedBlock::new();
        let s = b.blocks(2);
        assert_eq!(s.len(), BLOCK * 2);
        assert_eq!(s.as_ptr() as usize % BLOCK, 0);
    }

    #[test]
    fn row_wider_than_a_block_is_refused() {
        // A row larger than one block could span three with an unaligned base.
        let msg = match NgramRowCache::open(Path::new("/nonexistent"), None, 10, BLOCK + 8, 4) {
            Ok(_) => panic!("expected refusal for oversize row_stride"),
            Err(e) => format!("{e:#}"),
        };
        assert!(msg.contains("O_DIRECT block"), "{msg}");
    }

    /// The seam arithmetic: with a base offset that is only 8-byte aligned
    /// (what a safetensors shard gives), rows land at every phase relative to
    /// the 4 KiB block, and the covering span must stay within two blocks.
    #[test]
    fn straddling_rows_are_covered_by_two_blocks() {
        for base in [0u64, 8, 1234568, 4095, 4097] {
            for stride in [256usize, 512, 4096] {
                for id in [0u64, 1, 7, 8, 1023] {
                    let byte = base + id * stride as u64;
                    let block_off = byte - (byte % BLOCK as u64);
                    let within = (byte - block_off) as usize;
                    let n = if within + stride > BLOCK { 2 } else { 1 };
                    assert!(
                        within + stride <= n * BLOCK,
                        "base={base} stride={stride} id={id} within={within} n={n}"
                    );
                }
            }
        }
    }
}
