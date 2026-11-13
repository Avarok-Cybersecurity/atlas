// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm: RetroInfer, arXiv 2505.02922 (VLDB 2026).

//! Responsibility: the device buffers of the RetroInfer port and their exact
//! layouts: one attention layer's index ([`LayerIndex`]), the zone lists a
//! selection writes ([`ZoneLists`]) and the exact-zone buffer the gather fills
//! ([`ExactBuf`]). These are the types the L34 kernels read and write; the
//! contracts below are binding for both sides.
//!
//! # Index layout (per attention layer, per sequence)
//! One arena region ([`IndexLayout`]), 256-byte aligned sections:
//! - `header`: [`HEADER_WORDS`] i32, see [`IndexCounts`]. The host writes it
//!   after every build or update; kernels read counts from it (so a selection
//!   never needs a host value that changes between steps).
//! - `centroids`: `[kv_heads][cap_centroids][head_dim]` BF16, the plain mean
//!   of each cluster's member keys (the build's centering mean added back).
//! - `value_sum`: `[kv_heads][cap_centroids][head_dim]` BF16, the sum (not the
//!   mean) of each cluster's member values.
//! - `cluster_size`: `[kv_heads][cap_centroids]` i32 member count; 0 = empty,
//!   never selected.
//! - `list_offsets`: `[kv_heads][cap_centroids + 1]` i32, CSR offsets into the
//!   same head's `list_tokens`; cluster `c`'s members are
//!   `list_tokens[off[c]..off[c + 1]]`.
//! - `list_tokens`: `[kv_heads][cap_tokens]` i32 absolute token positions, each
//!   cluster's members ascending.
//!
//! Clusters cover positions `[sink, indexed_end)`; positions `[0, sink)` and
//! `[indexed_end, len)` are always exact.

use spark_runtime::gpu::DevicePtr;

/// Words in an index header.
pub const HEADER_WORDS: usize = 8;
/// Header word indexes.
pub mod header {
    pub const N_CENTROIDS: usize = 0;
    pub const NPROBE: usize = 1;
    pub const ES: usize = 2;
    pub const INDEXED_END: usize = 3;
    pub const SINK: usize = 4;
    pub const LISTED: usize = 5;
}

/// Section alignment inside every retro region.
pub const ALIGN: usize = 256;

pub(crate) fn align(v: usize) -> usize {
    v.div_ceil(ALIGN) * ALIGN
}

/// Bytes of one BF16 element.
pub const BF16: usize = 2;

/// The counts an index header carries; the host keeps this mirror.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IndexCounts {
    /// Centroids in use (empty ones included).
    pub n_centroids: u32,
    /// Retrieval-zone clusters a selection takes.
    pub nprobe: u32,
    /// Estimation-zone clusters a selection takes after them.
    pub es: u32,
    /// First position not in the index (the exact tail starts here).
    pub indexed_end: u32,
    /// Positions `[0, sink)` are the exact attention sink.
    pub sink: u32,
    /// Entries used in every head's `list_tokens` (`indexed_end - sink`).
    pub listed: u32,
}

impl IndexCounts {
    /// The header words, in [`header`] order.
    pub fn to_words(&self) -> [i32; HEADER_WORDS] {
        let w = |v: u32| i32::try_from(v).unwrap_or(i32::MAX);
        [
            w(self.n_centroids),
            w(self.nprobe),
            w(self.es),
            w(self.indexed_end),
            w(self.sink),
            w(self.listed),
            0,
            0,
        ]
    }

    /// The header as bytes for a host-to-device copy.
    pub fn to_bytes(&self) -> [u8; HEADER_WORDS * 4] {
        let mut out = [0u8; HEADER_WORDS * 4];
        for (i, w) in self.to_words().iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        out
    }

    /// Parse header bytes (tests and the shadow path read them back).
    pub fn from_bytes(b: &[u8]) -> Self {
        let w = |i: usize| -> u32 {
            let v = i32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]);
            u32::try_from(v.max(0)).unwrap_or(0)
        };
        Self {
            n_centroids: w(header::N_CENTROIDS),
            nprobe: w(header::NPROBE),
            es: w(header::ES),
            indexed_end: w(header::INDEXED_END),
            sink: w(header::SINK),
            listed: w(header::LISTED),
        }
    }
}

/// Section offsets of one layer index for given capacities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexLayout {
    pub kv_heads: u32,
    pub head_dim: u32,
    pub cap_centroids: u32,
    pub cap_tokens: u32,
}

impl IndexLayout {
    fn vec_bytes(&self) -> usize {
        self.kv_heads as usize * self.cap_centroids as usize * self.head_dim as usize * BF16
    }
    pub fn header_off(&self) -> usize {
        0
    }
    pub fn centroids_off(&self) -> usize {
        align(HEADER_WORDS * 4)
    }
    pub fn value_sum_off(&self) -> usize {
        self.centroids_off() + align(self.vec_bytes())
    }
    pub fn sizes_off(&self) -> usize {
        self.value_sum_off() + align(self.vec_bytes())
    }
    pub fn offsets_off(&self) -> usize {
        self.sizes_off() + align(self.kv_heads as usize * self.cap_centroids as usize * 4)
    }
    pub fn tokens_off(&self) -> usize {
        self.offsets_off() + align(self.kv_heads as usize * (self.cap_centroids as usize + 1) * 4)
    }
    /// Bytes of the whole region.
    pub fn total_bytes(&self) -> usize {
        self.tokens_off() + align(self.kv_heads as usize * self.cap_tokens as usize * 4)
    }
    /// Per-head element strides: (centroid rows, offsets, tokens).
    pub fn head_strides(&self) -> (usize, usize, usize) {
        (
            self.cap_centroids as usize,
            self.cap_centroids as usize + 1,
            self.cap_tokens as usize,
        )
    }
}

/// What the next `build::build_layer_index` call builds. Set by the host,
/// consumed by the kernel side.
///
/// Contract: per KV head, cluster the keys at positions
/// `[first_pos, first_pos + tokens)` (read from the paged pool through the
/// `SeqBatch` the caller passes: one row, a device block table, BF16 or FP8
/// K and V per `KvLayerView`) into `centroids` new centroids with `segments`
/// k-means segments and `iters` iterations (the oracle in `oracle.rs` is the
/// reference: centering by the range mean, uniform-stride initialization,
/// `iters - 1` normalized training passes per segment, one global plain-mean
/// pass, the mean added back). Write the new clusters at centroid indexes
/// `[centroid_offset, centroid_offset + centroids)`: centroids, value sums,
/// sizes, and the CSR entries with `list_offsets[centroid_offset] ==
/// token_offset` and `list_offsets[centroid_offset + centroids] ==
/// token_offset + tokens` (every token of the range listed once, each cluster's
/// members ascending). Entries below the offsets are never touched (an update
/// appends). `scratch` holds `RetroCfg::build_scratch_bytes(tokens,
/// centroids)` bytes; nothing else is allocated. The header is the host's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingBuild {
    pub first_pos: u32,
    pub tokens: u32,
    pub segments: u32,
    pub centroids: u32,
    pub centroid_offset: u32,
    pub token_offset: u32,
    pub iters: u32,
    pub scratch: DevicePtr,
    pub scratch_bytes: usize,
}

/// One attention layer's index for one sequence (see the module doc).
#[derive(Clone, Debug)]
pub struct LayerIndex {
    /// The arena region holding every section.
    pub base: DevicePtr,
    pub layout: IndexLayout,
    pub header: DevicePtr,
    pub centroids: DevicePtr,
    pub value_sum: DevicePtr,
    pub cluster_size: DevicePtr,
    pub list_offsets: DevicePtr,
    pub list_tokens: DevicePtr,
    /// Host mirror of the header.
    pub counts: IndexCounts,
    /// The build the next `build_layer_index` call performs.
    pub pending: Option<PendingBuild>,
    /// Query heads of the layer (`RetroCfg::num_q_heads`), for kernels that
    /// take only the index (selection). Set by the host when it allocates.
    pub num_q_heads: u32,
}

impl LayerIndex {
    /// The index whose region starts at `base` (allocated by the caller;
    /// `num_q_heads` is the caller's to set).
    pub fn at(base: DevicePtr, layout: IndexLayout) -> Self {
        Self {
            base,
            layout,
            header: base.offset(layout.header_off()),
            centroids: base.offset(layout.centroids_off()),
            value_sum: base.offset(layout.value_sum_off()),
            cluster_size: base.offset(layout.sizes_off()),
            list_offsets: base.offset(layout.offsets_off()),
            list_tokens: base.offset(layout.tokens_off()),
            counts: IndexCounts::default(),
            pending: None,
            num_q_heads: 0,
        }
    }

    /// The list pointers a zone list carries to the gather.
    pub fn list_ref(&self) -> IndexListRef {
        IndexListRef {
            header: self.header,
            cluster_size: self.cluster_size,
            list_offsets: self.list_offsets,
            list_tokens: self.list_tokens,
            layout: self.layout,
        }
    }
}

/// The part of an index the exact-zone gather reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexListRef {
    pub header: DevicePtr,
    pub cluster_size: DevicePtr,
    pub list_offsets: DevicePtr,
    pub list_tokens: DevicePtr,
    pub layout: IndexLayout,
}

/// One selection's output, written by `select::select_zones`.
///
/// Contract: per KV head `h`, score every non-empty centroid of the index with
/// the group's query heads (softmax over centroids per query head of
/// `q . c * inv_sqrt_d`, summed over the group; empty clusters never chosen),
/// rank by descending score with ties toward the lower centroid index, and
/// write the first `nprobe` (header) ids to `retrieval_ids[h][..]` in rank
/// order and the next `es` (header) ids to `estimation_ids[h][..]`, both
/// capped by the capacities, with the two counts at `counts[h * 2]` and
/// `counts[h * 2 + 1]`. Fixed capacities, counts on the device, no D2H.
/// `scores` and `scratch` are the kernel's (`RetroCfg::select_scratch_bytes`).
/// The host sets `lists` (`bind`) before the gather.
#[derive(Clone, Debug)]
pub struct ZoneLists {
    pub retrieval_ids: DevicePtr,
    pub estimation_ids: DevicePtr,
    pub counts: DevicePtr,
    pub retrieval_cap: u32,
    pub es_cap: u32,
    pub scores: DevicePtr,
    pub scratch: DevicePtr,
    pub scratch_bytes: usize,
    pub lists: Option<IndexListRef>,
}

/// Section sizes of one zone list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZoneLayout {
    pub kv_heads: u32,
    pub retrieval_cap: u32,
    pub es_cap: u32,
}

impl ZoneLayout {
    /// Bytes of the id and count sections (scores and scratch are separate).
    pub fn bytes(&self) -> usize {
        let h = self.kv_heads as usize;
        align(h * self.retrieval_cap as usize * 4)
            + align(h * self.es_cap as usize * 4)
            + align(h * 8)
    }
}

impl ZoneLists {
    /// The lists in a region of `ZoneLayout::bytes()` at `base`, sharing the
    /// selection scratch at `scores` / `scratch`.
    pub fn at(
        base: DevicePtr,
        l: ZoneLayout,
        scores: DevicePtr,
        scratch: DevicePtr,
        scratch_bytes: usize,
    ) -> Self {
        let h = l.kv_heads as usize;
        let r = align(h * l.retrieval_cap as usize * 4);
        let e = align(h * l.es_cap as usize * 4);
        Self {
            retrieval_ids: base,
            estimation_ids: base.offset(r),
            counts: base.offset(r + e),
            retrieval_cap: l.retrieval_cap,
            es_cap: l.es_cap,
            scores,
            scratch,
            scratch_bytes,
            lists: None,
        }
    }

    /// Point the gather at `index`'s cluster lists.
    pub fn bind(&mut self, index: &LayerIndex) {
        self.lists = Some(index.list_ref());
    }
}

/// The exact zone of one sequence for one layer, written by
/// `gather::gather_exact` and read by `attn::estimation_attention`.
///
/// Contract: per KV head `h`, entries `[0, valid_len[h])` of `k[h]`, `v[h]`
/// (BF16, FP8 KV dequantized with `KvLayerView::{kv_scale, v_scale}`) and
/// `positions[h]` in this order: the sink `[0, min(sink, indexed_end))`, then
/// the members of `retrieval_ids[h]` in rank order (each cluster ascending),
/// then the tail `[indexed_end, len)` ascending, where `len` is the gather
/// `SeqBatch`'s one row length (`seq_lens[0]` on the device). A retrieval
/// cluster that does not fit in whole is not stored and its members are added
/// to `dropped[h]`; the tail and the sink always fit (the host sizes
/// `capacity` with `exact_capacity`). The attention ignores the last
/// `row_trim` entries of every head (the later verify rows' own positions,
/// which are the newest tail entries), so row `j` of `R` verify rows passes
/// `row_trim = R - 1 - j`.
#[derive(Clone, Debug)]
pub struct ExactBuf {
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub positions: DevicePtr,
    pub valid_len: DevicePtr,
    pub dropped: DevicePtr,
    pub capacity: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub row_trim: u32,
    /// Query heads of the layer: the attention's GQA group is
    /// `num_q_heads / kv_heads` (`RetroCfg::num_q_heads`; the host sets it once
    /// after [`ExactBuf::at`]).
    pub num_q_heads: u32,
    /// The attention scale (`RetroCfg::inv_sqrt_d`); 0.0 means `1 / sqrt(head_dim)`.
    pub inv_sqrt_d: f32,
}

impl ExactBuf {
    /// Bytes of an exact buffer.
    pub fn bytes(kv_heads: u32, head_dim: u32, capacity: u32) -> usize {
        let h = kv_heads as usize;
        let rows = h * capacity as usize;
        2 * align(rows * head_dim as usize * BF16) + align(rows * 4) + 2 * align(h * 4)
    }

    /// The buffer in a region of [`ExactBuf::bytes`] at `base`.
    pub fn at(base: DevicePtr, kv_heads: u32, head_dim: u32, capacity: u32) -> Self {
        let h = kv_heads as usize;
        let rows = h * capacity as usize;
        let kv = align(rows * head_dim as usize * BF16);
        let pos = align(rows * 4);
        Self {
            k: base,
            v: base.offset(kv),
            positions: base.offset(2 * kv),
            valid_len: base.offset(2 * kv + pos),
            dropped: base.offset(2 * kv + pos + align(h * 4)),
            capacity,
            kv_heads,
            head_dim,
            row_trim: 0,
            num_q_heads: 0,
            inv_sqrt_d: 0.0,
        }
    }
}
