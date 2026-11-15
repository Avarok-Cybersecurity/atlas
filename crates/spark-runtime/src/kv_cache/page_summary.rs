// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm after Quest (page min/max upper-bound selection) and the page-sparse prototype in
// arczhi/llama-tierkv scripts-5060ti/proto/proto_page_sparse.cu (MIT, Copyright (c) 2023-2026 The
// ggml authors; file has no header of its own). Kernels rewritten for Atlas paged KV; changes: the
// summary is keyed by the physical KV block that leads each page (graph-safe: every index a kernel
// needs comes from the device block table), with a per-head completeness mask, a content
// fingerprint and a slot lock, by Atlas contributors. No prototype source was read; the formula is
// taken from a written description of the method.

//! Responsibility: the device layout of the Quest page summaries (row
//! `attn_page_sparse`, off by default on every target): per attention
//! layer, per page-leading physical KV block and per KV head, the channel-wise
//! minimum and maximum of the page's keys as f16 (`kmin`, `kmax`), plus a
//! 32-byte meta record. Pure layout arithmetic and the one allocation; the
//! kernels are `kernels/ada/common/attn_page_sparse_ada.cu` and the host glue
//! is `spark_model::port::sparse_quest`.
//!
//! # Pages and slots
//! A page is `page_blocks` consecutive logical blocks of one sequence
//! (64 tokens: 4 Atlas blocks of 16). Its summary lives in the slot of the
//! physical block that holds the page's first token, so a kernel finds it from
//! the block table alone (no host state, so CUDA-graph replays stay correct)
//! and sequences that share a cached prefix block share its summary. Slots of
//! blocks that never lead a page are unused; the arena has one slot per
//! physical block of the pool (`cap_blocks`), sized by the model build where
//! the KV budget is computed, and charged to that budget.
//!
//! # Meta record (per layer, slot and KV head; four u64 words)
//! - word 0: completeness mask, bit `t` set when the page's token `t` was
//!   folded into the summary since the slot was last reset;
//! - word 1: fingerprint, the first 8 bytes of the head's key at the page's
//!   first token when the summary was last written (detects a reused or
//!   restored block whose summary is stale);
//! - word 2: the slot lock (low 32 bits), held by one CTA or warp while it
//!   reads, merges and writes the record;
//! - word 3: reserved, zero.
//!
//! A summary is trusted only when its mask is full and its fingerprint matches
//! the key in the pool; otherwise the scoring kernel rebuilds it from the keys,
//! so a missed update costs one extra page read, never a wrong bound.
//!
//! # Off is byte-identical
//! Nothing here is allocated, sized or charged unless the row is on and the
//! selector registered (`spark_model::port::sparse_quest::budget`).

use anyhow::{Result, bail};

use crate::gpu::{DevicePtr, GpuBackend};

/// Tokens per page (Quest page; the scoring and selection unit).
pub const PAGE_TOKENS: usize = 64;

/// Largest supported `page_blocks` (block size 8).
pub const MAX_PAGE_BLOCKS: usize = 8;

/// u64 words in one meta record.
pub const META_WORDS: usize = 4;

/// Bytes of one meta record.
pub const META_BYTES: usize = META_WORDS * 8;

/// Largest head dim the kernels serve (16 channels per lane).
pub const MAX_HEAD_DIM: usize = 512;

/// Alignment of every region inside the arena.
const ALIGN: u64 = 256;

fn align_up(v: u64) -> u64 {
    v.div_ceil(ALIGN) * ALIGN
}

/// Blocks per page for a KV block size: `PAGE_TOKENS / block_size` when the
/// block size divides a page, `None` otherwise (the selector then refuses the
/// geometry and attention stays dense).
pub fn page_blocks(block_size: usize) -> Option<usize> {
    if block_size == 0 || block_size > PAGE_TOKENS || !PAGE_TOKENS.is_multiple_of(block_size) {
        return None;
    }
    let pb = PAGE_TOKENS / block_size;
    (pb <= MAX_PAGE_BLOCKS).then_some(pb)
}

/// The mask word of a page whose `page_tokens` tokens were all folded.
pub fn full_mask(page_tokens: usize) -> u64 {
    if page_tokens >= 64 {
        u64::MAX
    } else {
        (1u64 << page_tokens) - 1
    }
}

/// One attention layer's KV geometry, as the summary sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerGeom {
    pub kv_heads: u32,
    pub head_dim: u32,
}

impl LayerGeom {
    /// f16 elements of one slot's values: `kv_heads x [kmin; head_dim] [kmax; head_dim]`.
    pub fn slot_halves(self) -> u64 {
        u64::from(self.kv_heads) * 2 * u64::from(self.head_dim)
    }

    /// Bytes of one slot: values plus one meta record per head.
    pub fn slot_bytes(self) -> u64 {
        self.slot_halves() * 2 + u64::from(self.kv_heads) * META_BYTES as u64
    }
}

/// Where every layer's values and meta live inside one arena. All value
/// regions come first, then all meta regions, so the meta can be zeroed (every
/// mask empty, every lock free) with one memset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SummaryLayout {
    pub cap_blocks: u32,
    pub block_size: u32,
    pub page_blocks: u32,
    pub layers: Vec<LayerGeom>,
    vals_off: Vec<u64>,
    meta_off: Vec<u64>,
    meta_start: u64,
    total: u64,
}

impl SummaryLayout {
    /// The layout for `cap_blocks` physical blocks of `block_size` tokens and
    /// these attention layers. Refuses a geometry the kernels cannot serve.
    pub fn new(cap_blocks: u32, block_size: u32, layers: Vec<LayerGeom>) -> Result<Self> {
        let Some(pb) = page_blocks(block_size as usize) else {
            bail!(
                "attn_page_sparse: KV block size {block_size} does not divide a {PAGE_TOKENS}-token \
                 page (supported block sizes: 8, 16, 32, 64)"
            );
        };
        if layers.is_empty() {
            bail!("attn_page_sparse: the model has no attention layers to summarize");
        }
        for (i, l) in layers.iter().enumerate() {
            let hd = l.head_dim as usize;
            if l.kv_heads == 0 || hd == 0 || hd > MAX_HEAD_DIM || !hd.is_multiple_of(32) {
                bail!(
                    "attn_page_sparse: attention layer {i} has {} KV heads of head dim {hd}; the \
                     page summary kernels need a head dim that is a multiple of 32, at most \
                     {MAX_HEAD_DIM}",
                    l.kv_heads
                );
            }
        }
        let cap = u64::from(cap_blocks);
        let mut vals_off = Vec::with_capacity(layers.len());
        let mut off = 0u64;
        for l in &layers {
            vals_off.push(off);
            off = align_up(off + cap * l.slot_halves() * 2);
        }
        let meta_start = off;
        let mut meta_off = Vec::with_capacity(layers.len());
        for l in &layers {
            meta_off.push(off);
            off = align_up(off + cap * u64::from(l.kv_heads) * META_BYTES as u64);
        }
        Ok(Self {
            cap_blocks,
            block_size,
            page_blocks: pb as u32,
            layers,
            vals_off,
            meta_off,
            meta_start,
            total: off,
        })
    }

    /// Tokens per page.
    pub fn page_tokens(&self) -> u32 {
        self.page_blocks * self.block_size
    }

    /// Bytes of the whole arena.
    pub fn total_bytes(&self) -> u64 {
        self.total
    }

    /// The `(offset, bytes)` of the meta region (zeroed at allocation).
    pub fn meta_region(&self) -> (u64, u64) {
        (self.meta_start, self.total - self.meta_start)
    }

    /// Bytes per physical block across every layer (the budget charge rate).
    pub fn bytes_per_block(layers: &[LayerGeom]) -> u64 {
        layers.iter().map(|l| l.slot_bytes()).sum()
    }

    /// The f16 values of attention layer `layer`, `[cap_blocks][kv_heads][2][head_dim]`.
    pub fn layer_vals(&self, base: DevicePtr, layer: usize) -> DevicePtr {
        base.offset(self.vals_off[layer] as usize)
    }

    /// The meta records of attention layer `layer`, `[cap_blocks][kv_heads][META_WORDS]` u64.
    pub fn layer_meta(&self, base: DevicePtr, layer: usize) -> DevicePtr {
        base.offset(self.meta_off[layer] as usize)
    }

    /// Byte offset of one slot's values for one head inside the arena (host
    /// mirror of the kernels' indexing; tests and the microtest read it).
    pub fn vals_offset(&self, layer: usize, slot: u32, head: u32) -> u64 {
        let l = self.layers[layer];
        self.vals_off[layer]
            + (u64::from(slot) * u64::from(l.kv_heads) + u64::from(head))
                * 2
                * u64::from(l.head_dim)
                * 2
    }

    /// Byte offset of one slot's meta record for one head inside the arena.
    pub fn meta_offset(&self, layer: usize, slot: u32, head: u32) -> u64 {
        let l = self.layers[layer];
        self.meta_off[layer]
            + (u64::from(slot) * u64::from(l.kv_heads) + u64::from(head)) * META_BYTES as u64
    }
}

/// One allocated summary arena. Freed on drop through the backend that made it.
pub struct PageSummaryBuffers {
    pub base: DevicePtr,
    pub layout: SummaryLayout,
}

impl PageSummaryBuffers {
    /// Allocate the arena and zero its meta region on `stream` (every summary
    /// starts invalid, every lock free). Never call while `stream` is capturing.
    pub fn alloc(gpu: &dyn GpuBackend, layout: SummaryLayout, stream: u64) -> Result<Self> {
        let bytes = usize::try_from(layout.total_bytes())?;
        let base = gpu.alloc(bytes)?;
        let (off, len) = layout.meta_region();
        if let Err(e) = gpu.memset_async(base.offset(off as usize), 0, len as usize, stream) {
            let _ = gpu.free(base);
            return Err(e);
        }
        Ok(Self { base, layout })
    }

    /// Release the arena.
    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.base)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::mock::MockGpuBackend;

    fn qwen35_4b() -> Vec<LayerGeom> {
        vec![
            LayerGeom {
                kv_heads: 4,
                head_dim: 256
            };
            8
        ]
    }

    #[test]
    fn page_blocks_covers_the_supported_block_sizes() {
        assert_eq!(page_blocks(16), Some(4));
        assert_eq!(page_blocks(8), Some(8));
        assert_eq!(page_blocks(32), Some(2));
        assert_eq!(page_blocks(64), Some(1));
        for bad in [0, 4, 12, 24, 128] {
            assert_eq!(page_blocks(bad), None, "{bad}");
        }
    }

    #[test]
    fn full_mask_has_one_bit_per_page_token() {
        assert_eq!(full_mask(64), u64::MAX);
        assert_eq!(full_mask(32), 0xFFFF_FFFF);
        assert_eq!(full_mask(1), 1);
    }

    #[test]
    fn the_4b_rate_is_one_sixteenth_of_its_bf16_kv() {
        // 8 layers x 4 heads x 256 channels x (min, max) f16 = 32 KiB per block, plus
        // 8 x 4 meta records; a BF16 block of 16 tokens is 8 x 2 x 16 x 1024 x 2 B = 512 KiB.
        let per_block = SummaryLayout::bytes_per_block(&qwen35_4b());
        assert_eq!(per_block, 32 * 1024 + 8 * 4 * 32);
        let bf16_block = 8u64 * 2 * 16 * 1024 * 2;
        assert!(per_block * 15 < bf16_block && per_block * 17 > bf16_block);
    }

    #[test]
    fn regions_are_disjoint_aligned_and_cover_every_slot() {
        let l = SummaryLayout::new(1000, 16, qwen35_4b()).unwrap();
        assert_eq!(l.page_tokens(), 64);
        let (meta_start, meta_len) = l.meta_region();
        assert_eq!(meta_start % ALIGN, 0);
        assert_eq!(meta_start + meta_len, l.total_bytes());
        for layer in 0..8 {
            let last_val = l.vals_offset(layer, 999, 3) + 2 * 256 * 2;
            assert!(
                last_val <= meta_start,
                "layer {layer} values overrun the meta region"
            );
            if layer + 1 < 8 {
                assert!(last_val <= l.vals_offset(layer + 1, 0, 0));
                assert!(
                    l.meta_offset(layer, 999, 3) + META_BYTES as u64
                        <= l.meta_offset(layer + 1, 0, 0)
                );
            }
            assert_eq!(l.vals_offset(layer, 0, 0) % ALIGN, 0);
            assert_eq!(l.meta_offset(layer, 0, 0) % ALIGN, 0);
            assert!(l.meta_offset(layer, 999, 3) + META_BYTES as u64 <= l.total_bytes());
        }
        // Consecutive heads and slots are contiguous (the kernels' `(slot * kvh + h)` index).
        assert_eq!(l.vals_offset(2, 5, 1) - l.vals_offset(2, 5, 0), 2 * 256 * 2);
        assert_eq!(
            l.meta_offset(2, 6, 0) - l.meta_offset(2, 5, 3),
            META_BYTES as u64
        );
    }

    #[test]
    fn unsupported_geometry_is_refused() {
        assert!(SummaryLayout::new(10, 12, qwen35_4b()).is_err());
        assert!(SummaryLayout::new(10, 16, Vec::new()).is_err());
        let odd = vec![LayerGeom {
            kv_heads: 2,
            head_dim: 80,
        }];
        assert!(SummaryLayout::new(10, 16, odd).is_err());
        let wide = vec![LayerGeom {
            kv_heads: 1,
            head_dim: 1024,
        }];
        assert!(SummaryLayout::new(10, 16, wide).is_err());
    }

    #[test]
    fn alloc_zeroes_only_the_meta_region() {
        let gpu = MockGpuBackend::new();
        let layout = SummaryLayout::new(
            4,
            16,
            vec![LayerGeom {
                kv_heads: 2,
                head_dim: 64,
            }],
        )
        .unwrap();
        let bufs = PageSummaryBuffers::alloc(&gpu, layout.clone(), 0).unwrap();
        let data = gpu.read_alloc(bufs.base).unwrap();
        let (off, len) = layout.meta_region();
        assert!(
            data[off as usize..(off + len) as usize]
                .iter()
                .all(|b| *b == 0)
        );
        assert_eq!(data.len() as u64, layout.total_bytes());
        bufs.free(&gpu).unwrap();
    }
}
