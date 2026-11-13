// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the bounded view's table arithmetic on the host (port lane
//! L32): turning a block pick into a view shape, binding a shape to one
//! sequence by its anchors, and composing the view block table of one
//! attention row. [`build_row`] is the CPU twin (and oracle) of the device
//! kernel `kv_view_build_tables` (`kernels/ada/common/kv_view_table_ada.cu`).
//!
//! # The view
//! A view keeps, in chronological order, the selected pages that lie before
//! its tail, then every page from the tail page on (the recent window, the
//! question, and everything appended after the selection). The real block
//! table is never touched, so eviction, prefix sharing and free-on-finish
//! behave as without the view. K is stored after RoPE at its absolute
//! position, so dropping whole pages before the tail keeps the geometry of
//! every kept key to every new query; the view length is the real length
//! minus the dropped tokens, and the causal tail stays at the end.
//!
//! # Anchors
//! Decode steps are replayed from CUDA graphs with no host code, so the
//! device finds a row's view by content: a descriptor applies to a row whose
//! physical blocks at logical pages 0, `tail_page` and `last_page` (the last
//! page at selection) are the descriptor's anchors. The most specific match
//! (largest `last_page`) wins; a row with no match keeps its real table.

use std::ops::Range;

/// Words per device descriptor: `[active, a0, a_tail, a_last, last_page,
/// tail_page, n_sel, 0]` as `i32`.
pub const DESC_WORDS: usize = 8;
/// The `active` word of a live descriptor.
pub const DESC_ACTIVE: i32 = 1;

/// Pages a row of `len` tokens spans.
pub fn pages_of(len: u32, block_size: u32) -> u32 {
    len.div_ceil(block_size.max(1))
}

/// A selection in logical page terms, independent of any physical block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewShape {
    /// Selected pages before the tail, ascending, all below `tail_page`.
    pub sel_pages: Vec<u32>,
    /// First page kept in full: every page from here on is in the view.
    pub tail_page: u32,
}

impl ViewShape {
    /// Tokens the view drops (whole pages before the tail that were not selected).
    pub fn drop_tokens(&self, block_size: u32) -> u32 {
        (self.tail_page - self.sel_pages.len() as u32) * block_size
    }

    /// The pages and length the view of a row of `len` tokens holds, in
    /// physical block ids from `row`. `None` when the row is too short for
    /// the shape (it does not reach the tail page) or the shape is malformed.
    pub fn compose(&self, row: &[u32], len: u32, block_size: u32) -> Option<(Vec<u32>, u32)> {
        let pages = pages_of(len, block_size);
        let ok = self.tail_page < pages
            && pages as usize <= row.len()
            && self.sel_pages.len() as u32 <= self.tail_page
            && self.sel_pages.windows(2).all(|w| w[0] < w[1])
            && self.sel_pages.last().is_none_or(|&p| p < self.tail_page);
        if !ok {
            return None;
        }
        let mut out = Vec::with_capacity(self.sel_pages.len() + (pages - self.tail_page) as usize);
        out.extend(self.sel_pages.iter().map(|&p| row[p as usize]));
        out.extend_from_slice(&row[self.tail_page as usize..pages as usize]);
        Some((out, len - self.drop_tokens(block_size)))
    }
}

/// The view shape of a block pick. `picked` holds block indices (each block
/// is `block_pages` pages); it is sorted and deduplicated here, blocks at or
/// past the end are dropped and the last (possibly partial) block is always
/// kept. The tail is the longest run of consecutive picked blocks ending at
/// the last block; the other picked blocks become selected pages. `None` when
/// the view would drop nothing (the identity).
pub fn shape_from_pick(picked: &[u32], n_pages: u32, block_pages: u32) -> Option<ViewShape> {
    let bp = block_pages.max(1);
    if n_pages == 0 {
        return None;
    }
    let n_blocks = n_pages.div_ceil(bp);
    let mut blocks: Vec<u32> = picked.iter().copied().filter(|&b| b < n_blocks).collect();
    blocks.push(n_blocks - 1);
    blocks.sort_unstable();
    blocks.dedup();
    let mut tail_block = n_blocks - 1;
    for &b in blocks.iter().rev().skip(1) {
        if b + 1 == tail_block {
            tail_block = b;
        } else {
            break;
        }
    }
    let tail_page = tail_block * bp;
    let sel_pages: Vec<u32> = blocks
        .iter()
        .take_while(|&&b| b < tail_block)
        .flat_map(|&b| b * bp..(b + 1) * bp)
        .collect();
    if tail_page == 0 || sel_pages.len() as u32 == tail_page {
        return None;
    }
    Some(ViewShape {
        sel_pages,
        tail_page,
    })
}

/// The blocks a query span touches (mandatory in every pick).
pub fn mandatory_blocks(query: Range<u32>, block_tokens: u32) -> Vec<u32> {
    if query.is_empty() {
        return Vec::new();
    }
    let bt = block_tokens.max(1);
    (query.start / bt..(query.end - 1) / bt + 1).collect()
}

/// A view shape bound to one sequence by its anchors (one device descriptor).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Desc {
    pub shape: ViewShape,
    pub a0: u32,
    pub a_tail: u32,
    pub a_last: u32,
    pub last_page: u32,
}

impl Desc {
    /// Bind `shape` to the sequence whose real table is `row` and length
    /// `len` at selection. `None` when the row does not reach the tail page.
    pub fn bind(shape: ViewShape, row: &[u32], len: u32, block_size: u32) -> Option<Self> {
        let pages = pages_of(len, block_size);
        if pages == 0 || pages as usize > row.len() || shape.tail_page >= pages {
            return None;
        }
        let last_page = pages - 1;
        Some(Self {
            a0: row[0],
            a_tail: row[shape.tail_page as usize],
            a_last: row[last_page as usize],
            last_page,
            shape,
        })
    }

    /// Whether this descriptor applies to a row of `len` tokens.
    pub fn matches(&self, row: &[u32], len: u32, block_size: u32) -> bool {
        let pages = pages_of(len, block_size);
        self.last_page < pages
            && (pages as usize) <= row.len()
            && row[0] == self.a0
            && row[self.shape.tail_page as usize] == self.a_tail
            && row[self.last_page as usize] == self.a_last
    }

    /// The device words (see [`DESC_WORDS`]). Block ids and page indices fit
    /// `i32` (the device tables are `i32`).
    pub fn words(&self) -> [i32; DESC_WORDS] {
        [
            DESC_ACTIVE,
            self.a0 as i32,
            self.a_tail as i32,
            self.a_last as i32,
            self.last_page as i32,
            self.shape.tail_page as i32,
            self.shape.sel_pages.len() as i32,
            0,
        ]
    }
}

/// The descriptor that applies to a row: the matching one with the largest
/// `last_page`, the lowest index on a tie (the kernel's rule).
pub fn best_match<'a>(
    descs: &[Option<&'a Desc>],
    row: &[u32],
    len: u32,
    block_size: u32,
) -> Option<&'a Desc> {
    let mut best: Option<&'a Desc> = None;
    for d in descs.iter().flatten().copied() {
        if d.matches(row, len, block_size) && best.is_none_or(|b| d.last_page > b.last_page) {
            best = Some(d);
        }
    }
    best
}

/// The view table and length of one attention row: the host twin of
/// `kv_view_build_tables`. A row with no matching descriptor (or one too
/// short for it) keeps its real pages and length; a padding row (`len` 0)
/// yields nothing.
pub fn build_row(
    row: &[u32],
    len: u32,
    block_size: u32,
    descs: &[Option<&Desc>],
) -> (Vec<u32>, u32) {
    let pages = (pages_of(len, block_size) as usize).min(row.len());
    if let Some(d) = best_match(descs, row, len, block_size)
        && let Some(v) = d.shape.compose(row, len, block_size)
    {
        return v;
    }
    (row[..pages].to_vec(), len)
}

/// A stable 64-bit fingerprint of a token prefix (conversation matching).
pub fn prefix_hash(tokens: &[u32]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    tokens.len().hash(&mut h);
    for t in tokens {
        t.hash(&mut h);
    }
    h.finish()
}
