// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm after Quest (page min/max upper-bound selection) and the page-sparse prototype in
// arczhi/llama-tierkv scripts-5060ti/proto/proto_page_sparse.cu (MIT, Copyright (c) 2023-2026 The
// ggml authors; file has no header of its own). Independent CPU reference for Atlas's kernels; the
// formula is the one written in the port sprint's report 77: score_p = sum_c max(q_c * kmin_c,
// q_c * kmax_c). No prototype source was read.

//! Responsibility: the CPU reference of the Quest kernels (port lane L30),
//! used by the unit tests and by `examples/page_sparse_microtest.rs`: f32 to
//! f16 rounding toward minus and plus infinity (the kernels' `__float2half_rd`
//! and `__float2half_ru`), the page summary from scratch and incrementally
//! (the update kernel's fold, reset and mask rules), the Quest bound and page
//! scores in f64, the recent-window and identity rules, an exact top-k by
//! sorting and an emulation of the kernel's radix select, and the compacted
//! block table. Only `std` and `half`: nothing here touches a GPU.

use std::cmp::Reverse;

/// Tokens-per-block, blocks-per-page and the head geometry of one layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RefGeom {
    pub kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    pub page_blocks: usize,
}

impl RefGeom {
    pub fn page_tokens(&self) -> usize {
        self.block_size * self.page_blocks
    }
}

// ── f16 directed rounding ───────────────────────────────────────────────────

fn next_up(bits: u16) -> u16 {
    match bits {
        0x7C00 => bits,                                   // +inf
        b if b & 0x7C00 == 0x7C00 && b & 0x3FF != 0 => b, // NaN
        0x8000 => 0x0001,                                 // -0 -> smallest positive
        b if b & 0x8000 == 0 => b + 1,
        b => b - 1,
    }
}

fn next_down(bits: u16) -> u16 {
    match bits {
        0xFC00 => bits, // -inf
        b if b & 0x7C00 == 0x7C00 && b & 0x3FF != 0 => b,
        0x0000 => 0x8001, // +0 -> smallest negative
        b if b & 0x8000 == 0 => b - 1,
        b => b + 1,
    }
}

/// `__float2half_rd`: the largest f16 not above `x`.
pub fn f16_rd(x: f32) -> u16 {
    let h = half::f16::from_f32(x);
    if x.is_nan() {
        return h.to_bits();
    }
    if h.to_f32() > x {
        next_down(h.to_bits())
    } else {
        h.to_bits()
    }
}

/// `__float2half_ru`: the smallest f16 not below `x`.
pub fn f16_ru(x: f32) -> u16 {
    let h = half::f16::from_f32(x);
    if x.is_nan() {
        return h.to_bits();
    }
    if h.to_f32() < x {
        next_up(h.to_bits())
    } else {
        h.to_bits()
    }
}

pub fn f16_val(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// BF16 bits to f32.
pub fn bf16_val(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// FP8 E4M3 (OCP, finite-only with one NaN encoding) to f32.
pub fn e4m3_val(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = (b >> 3) & 0xF;
    let m = f32::from(b & 0x7);
    if e == 0xF && b & 0x7 == 0x7 {
        return f32::NAN;
    }
    if e == 0 {
        s * m / 8.0 * 2f32.powi(-6)
    } else {
        s * (1.0 + m / 8.0) * 2f32.powi(i32::from(e) - 7)
    }
}

// ── The K pool ──────────────────────────────────────────────────────────────

/// A host copy of one layer's K pool in the paged layout
/// `[num_blocks][block_size][kv_heads][head_dim]`.
pub struct KPool<'a> {
    pub data: &'a [u8],
    /// 2 = BF16, 1 = FP8 E4M3.
    pub elem_bytes: usize,
    pub block_stride: usize,
    pub geom: RefGeom,
}

impl KPool<'_> {
    fn offset(&self, phys: usize, off: usize, h: usize, c: usize) -> usize {
        let g = &self.geom;
        phys * self.block_stride + ((off * g.kv_heads + h) * g.head_dim + c) * self.elem_bytes
    }

    pub fn k(&self, phys: usize, off: usize, h: usize, c: usize) -> f32 {
        let o = self.offset(phys, off, h, c);
        if self.elem_bytes == 2 {
            bf16_val(u16::from_le_bytes([self.data[o], self.data[o + 1]]))
        } else {
            e4m3_val(self.data[o])
        }
    }

    /// The kernels' fingerprint: 8 bytes of head `h`'s key at the block's first token.
    pub fn fingerprint(&self, phys: usize, h: usize) -> u64 {
        let o = self.offset(phys, 0, h, 0);
        u64::from_le_bytes(self.data[o..o + 8].try_into().expect("8 bytes"))
    }

    /// Key of the token at logical position `pos` of a sequence with `table`.
    pub fn k_at(&self, table: &[i32], pos: usize, h: usize, c: usize) -> f32 {
        let bs = self.geom.block_size;
        self.k(table[pos / bs] as usize, pos % bs, h, c)
    }
}

// ── Summaries ───────────────────────────────────────────────────────────────

/// One (slot, head) record as the kernels store it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slot {
    /// f16 bits of kmin then kmax, `head_dim` each.
    pub kmin: Vec<u16>,
    pub kmax: Vec<u16>,
    pub mask: u64,
    pub fp: u64,
}

impl Slot {
    pub fn empty(head_dim: usize) -> Self {
        Self {
            kmin: vec![0x7C00; head_dim],
            kmax: vec![0xFC00; head_dim],
            mask: 0,
            fp: 0,
        }
    }
}

/// The mask of a page whose `pt` tokens were all folded.
pub fn full_mask(pt: usize) -> u64 {
    if pt >= 64 { u64::MAX } else { (1u64 << pt) - 1 }
}

/// Channel min/max of head `h` over positions `[lo, hi)`, directed-rounded.
pub fn fold_range(
    pool: &KPool,
    table: &[i32],
    h: usize,
    lo: usize,
    hi: usize,
) -> (Vec<u16>, Vec<u16>) {
    let hd = pool.geom.head_dim;
    let mut mn = vec![f32::INFINITY; hd];
    let mut mx = vec![f32::NEG_INFINITY; hd];
    for pos in lo..hi {
        for (c, (a, b)) in mn.iter_mut().zip(mx.iter_mut()).enumerate() {
            let v = pool.k_at(table, pos, h, c);
            *a = a.min(v);
            *b = b.max(v);
        }
    }
    (
        mn.iter().map(|&v| f16_rd(v)).collect(),
        mx.iter().map(|&v| f16_ru(v)).collect(),
    )
}

/// A full page's record built from scratch (what the score kernel's rebuild writes).
pub fn page_slot(pool: &KPool, table: &[i32], page: usize, h: usize) -> Slot {
    let g = pool.geom;
    let pt = g.page_tokens();
    let (kmin, kmax) = fold_range(pool, table, h, page * pt, (page + 1) * pt);
    Slot {
        kmin,
        kmax,
        mask: full_mask(pt),
        fp: pool.fingerprint(table[page * g.page_blocks] as usize, h),
    }
}

/// The update kernel's fold of the written range `[first, first + count)` into
/// the slot of page `page` (whose leading block is `table[page * pb]`).
pub fn update_slot(
    slot: &mut Slot,
    pool: &KPool,
    table: &[i32],
    page: usize,
    h: usize,
    first: usize,
    count: usize,
) {
    let g = pool.geom;
    let pt = g.page_tokens();
    let page_lo = page * pt;
    let end = first + count;
    if page_lo >= end || page_lo + pt <= first {
        return;
    }
    let lo = first.max(page_lo);
    let hi = end.min(page_lo + pt);
    let cur = pool.fingerprint(table[page * g.page_blocks] as usize, h);
    let reset = lo == page_lo || slot.fp != cur;
    let span = hi - lo;
    let bits = (if span >= 64 {
        u64::MAX
    } else {
        (1u64 << span) - 1
    }) << (lo - page_lo);
    slot.mask = if reset { bits } else { slot.mask | bits };
    slot.fp = cur;
    let (nmin, nmax) = fold_range(pool, table, h, lo, hi);
    for (smin, nmin) in slot.kmin.iter_mut().zip(&nmin) {
        let old = if reset { f32::INFINITY } else { f16_val(*smin) };
        *smin = f16_rd(old.min(f16_val(*nmin)));
    }
    for (smax, nmax) in slot.kmax.iter_mut().zip(&nmax) {
        let old = if reset {
            f32::NEG_INFINITY
        } else {
            f16_val(*smax)
        };
        *smax = f16_ru(old.max(f16_val(*nmax)));
    }
}

/// True when the score kernel trusts a stored record.
pub fn trusted(slot: &Slot, pool: &KPool, table: &[i32], page: usize, h: usize) -> bool {
    let g = pool.geom;
    slot.mask == full_mask(g.page_tokens())
        && slot.fp == pool.fingerprint(table[page * g.page_blocks] as usize, h)
}

// ── Scores ──────────────────────────────────────────────────────────────────

/// The Quest bound of one query head against one summary, in f64.
pub fn bound(q: &[f32], kmin: &[u16], kmax: &[u16]) -> f64 {
    q.iter()
        .zip(kmin.iter().zip(kmax))
        .filter(|(qc, _)| **qc != 0.0)
        .map(|(&qc, (&lo, &hi))| {
            let a = f64::from(qc) * f64::from(f16_val(lo));
            let b = f64::from(qc) * f64::from(f16_val(hi));
            a.max(b)
        })
        .sum()
}

/// One page's score: per query head the max over the group's rows, then the
/// sum (`gqa_sum`) or max over every query head of every KV head.
/// `q_rows[r]` is `[nq * head_dim]`; `slots[h]` the page's record of head `h`.
pub fn page_score(q_rows: &[&[f32]], nq: usize, slots: &[Slot], hd: usize, gqa_sum: bool) -> f64 {
    let kvh = slots.len();
    let group = nq / kvh;
    let mut page = if gqa_sum { 0.0 } else { f64::NEG_INFINITY };
    for (h, s) in slots.iter().enumerate() {
        for j in 0..group {
            let qh = h * group + j;
            let head = q_rows
                .iter()
                .map(|q| bound(&q[qh * hd..(qh + 1) * hd], &s.kmin, &s.kmax))
                .fold(f64::NEG_INFINITY, f64::max);
            page = if gqa_sum { page + head } else { page.max(head) };
        }
    }
    page
}

/// The first page of the recent window (the kernels' `recent_page`).
pub fn recent_page(e_lo: usize, recent: usize, pt: usize) -> usize {
    let start = e_lo.saturating_sub(recent);
    (start / pt).min((e_lo - 1) / pt)
}

/// The kernels' order-preserving map (NaN ranks as +inf).
pub fn sortable(s: f32) -> u32 {
    let s = if s.is_nan() { f32::INFINITY } else { s };
    let u = s.to_bits();
    if u & 0x8000_0000 != 0 {
        !u
    } else {
        u | 0x8000_0000
    }
}

/// Selection key: higher score first, ties to the newer page.
pub fn key(score: f32, page: usize) -> u64 {
    (u64::from(sortable(score)) << 32) | page as u64
}

/// Exact top-k by sorting: the candidate pages kept (ascending).
pub fn topk_sorted(scores: &[f32], sink: usize, rp: usize, k: usize) -> Vec<usize> {
    let mut c: Vec<usize> = (sink..rp).collect();
    c.sort_by_key(|&p| Reverse(key(scores[p], p)));
    c.truncate(k);
    c.sort_unstable();
    c
}

/// The compaction kernel's 8-pass radix select: the `k`-th largest key.
pub fn radix_threshold(keys: &[u64], k: usize) -> u64 {
    let (mut prefix, mut mask, mut need) = (0u64, 0u64, k);
    for d in (0..8).rev() {
        let shift = d * 8;
        let mut hist = [0usize; 256];
        for &key in keys {
            if key & mask == prefix {
                hist[((key >> shift) & 255) as usize] += 1;
            }
        }
        let mut acc = 0;
        let mut b = 255usize;
        while b > 0 {
            if acc + hist[b] >= need {
                break;
            }
            acc += hist[b];
            b -= 1;
        }
        need -= acc;
        prefix |= (b as u64) << shift;
        mask |= 255u64 << shift;
    }
    prefix
}

// ── Compaction ──────────────────────────────────────────────────────────────

/// The selection budget (the kernels' `sink_pages`, `recent_tokens`, `topk`, `min_ctx`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RefBudget {
    pub sink_pages: usize,
    pub recent_tokens: usize,
    pub topk_pages: usize,
    pub min_ctx: usize,
}

/// One group's compaction: the rows' View tables (the attended prefix only)
/// and lengths, and whether the group was sparse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Compacted {
    pub tables: Vec<Vec<i32>>,
    pub lens: Vec<i32>,
    pub sparse: bool,
    /// Pages kept (sparse groups only).
    pub pages: usize,
}

/// Mirror of `attn_ps_compact` for one group. `tables[r]` / `lens[r]` are the
/// group's dense rows; `scores` is indexed by page; `row_stride` the dense
/// row width and `score_stride` the score slots per group.
#[allow(clippy::too_many_arguments)]
pub fn compact(
    tables: &[&[i32]],
    lens: &[i32],
    scores: &[f32],
    g: RefGeom,
    b: RefBudget,
    row_stride: usize,
    score_stride: usize,
    max_candidates: usize,
) -> Compacted {
    let bs = g.block_size;
    let pb = g.page_blocks;
    let pt = g.page_tokens();
    let (e_lo, e_hi) = (lens[0], *lens.last().expect("rows"));
    let (mut rp, mut cand, mut nb_hi) = (0, 0, 0);
    if e_lo > 0 && e_hi > 0 {
        rp = recent_page(e_lo as usize, b.recent_tokens, pt);
        cand = rp.saturating_sub(b.sink_pages);
        nb_hi = (e_hi as usize).div_ceil(bs);
    }
    let sparse = e_lo > 0
        && e_hi as i64 > b.min_ctx as i64
        && cand > b.topk_pages
        && cand <= max_candidates
        && rp <= score_stride
        && nb_hi <= row_stride;
    if !sparse {
        let tables = tables
            .iter()
            .zip(lens)
            .map(|(t, &e)| {
                let n = (e.max(1) as usize).div_ceil(bs).min(row_stride);
                t[..n].to_vec()
            })
            .collect();
        return Compacted {
            tables,
            lens: lens.to_vec(),
            sparse: false,
            pages: 0,
        };
    }
    let chosen = topk_sorted(scores, b.sink_pages, rp, b.topk_pages);
    let kept_before = b.sink_pages + b.topk_pages;
    let row0 = tables[0];
    let mut out: Vec<i32> = Vec::new();
    out.extend_from_slice(&row0[..b.sink_pages * pb]);
    for &p in &chosen {
        out.extend_from_slice(&row0[p * pb..(p + 1) * pb]);
    }
    out.extend_from_slice(&row0[rp * pb..nb_hi]);
    let mut tabs = Vec::new();
    let mut out_lens = Vec::new();
    for &e in lens {
        let len = kept_before * pt + (e as usize - rp * pt);
        out_lens.push(len as i32);
        tabs.push(out[..len.div_ceil(bs)].to_vec());
    }
    Compacted {
        tables: tabs,
        lens: out_lens,
        sparse: true,
        pages: kept_before + (nb_hi.div_ceil(pb) - rp),
    }
}

/// Map each attended position of a compacted row back to its logical
/// position in the dense row (the test that the view keeps chronology and
/// never invents tokens): `None` when a compacted block is not in the dense row.
pub fn compacted_positions(
    dense: &[i32],
    view: &[i32],
    view_len: usize,
    bs: usize,
) -> Option<Vec<usize>> {
    let mut out = Vec::with_capacity(view_len);
    for pos in 0..view_len {
        let blk = view[pos / bs];
        let logical = dense.iter().position(|&b| b == blk)?;
        out.push(logical * bs + pos % bs);
    }
    Some(out)
}
