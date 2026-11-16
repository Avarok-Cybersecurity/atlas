// SPDX-License-Identifier: AGPL-3.0-only
// Translated for Atlas from microsoft/RetrievalAttention (RetroInfer), MIT License,
// Copyright (c) Microsoft Corporation. Upstream: cache_hub/retroinfer_cache.py:754-815
// (the execution buffer of steady zone plus retrieval clusters). Changes: a host model of
// `kernels/ada/common/retro_gather_ada.cu` on the device layouts of `state.rs`.

//! Responsibility: the CPU reference of the exact-zone gather on the device layouts (paged
//! pool, index header, CSR member lists, zone lists), written to be bit-equal to
//! `retro_gather_ada.cu`. The unit tests check it against L33's oracle
//! (`oracle_eval::exact_positions`); the GPU microtest checks the kernel against it.

use super::HEAD_DIM;
use crate::layers::qwen3_attention::sparse_seam::KvDtypeTag;
use crate::port::retro::state::header;

/// One sequence's paged KV on the host, in the device layout.
pub struct HostPaged<'a> {
    pub k_pool: &'a [u8],
    pub v_pool: &'a [u8],
    pub dtype: KvDtypeTag,
    pub block_size: u32,
    pub kv_heads: u32,
    /// Bytes from one block to the next.
    pub block_stride: usize,
    pub k_scale: f32,
    pub v_scale: f32,
    /// `[max_blocks_per_seq]` physical ids (negative = unreadable).
    pub block_table: &'a [i32],
    /// The rows' lengths; the gather covers the longest.
    pub row_lens: &'a [i32],
}

/// The index sections the gather reads: header words, CSR offsets and member tokens.
pub struct HostLists<'a> {
    pub header: &'a [i32],
    /// `[kv_heads][cap_centroids + 1]`.
    pub offsets: &'a [i32],
    /// `[kv_heads][cap_tokens]`.
    pub tokens: &'a [i32],
    pub cap_centroids: u32,
    pub cap_tokens: u32,
}

/// The retrieval zone: `ids[h * cap ..]` in rank order, `counts[h * 2]` its count.
pub struct HostZones<'a> {
    pub ids: &'a [i32],
    pub counts: &'a [i32],
    pub cap: u32,
}

/// The exact buffer on the host (BF16 as raw bits).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostExact {
    pub cap: u32,
    pub k: Vec<u16>,
    pub v: Vec<u16>,
    pub pos: Vec<i32>,
    pub valid: Vec<i32>,
    pub dropped: Vec<i32>,
}

impl HostExact {
    /// A buffer pre-filled with sentinels, so a test can see which entries were written.
    pub fn filled(kv_heads: u32, cap: u32, kv_bits: u16, word: i32) -> Self {
        let rows = kv_heads as usize * cap as usize;
        Self {
            cap,
            k: vec![kv_bits; rows * HEAD_DIM as usize],
            v: vec![kv_bits; rows * HEAD_DIM as usize],
            pos: vec![word; rows],
            valid: vec![word; kv_heads as usize],
            dropped: vec![word; kv_heads as usize],
        }
    }
}

/// The positions the gather writes for head `h` with capacity `cap`, and its drop count.
pub fn exact_positions(
    p: &HostPaged,
    l: &HostLists,
    z: &HostZones,
    h: usize,
    cap: u32,
) -> (Vec<i32>, i64) {
    let len = p.row_lens.iter().copied().max().unwrap_or(0).max(0);
    let ie = l.header[header::INDEXED_END].max(0);
    let sink = l.header[header::SINK].max(0);
    let sink_end = sink.min(ie).min(len);
    let tail_start = ie.min(len);
    let fixed = i64::from(sink_end) + i64::from(len - tail_start);
    let room = (i64::from(cap) - fixed).max(0);
    let nrc = z.counts[h * 2].clamp(0, z.cap as i32) as usize;
    let off = &l.offsets[h * (l.cap_centroids as usize + 1)..][..l.cap_centroids as usize + 1];
    let tok = &l.tokens[h * l.cap_tokens as usize..][..l.cap_tokens as usize];
    let mut out: Vec<i32> = (0..sink_end).collect();
    let (mut used, mut dropped) = (0i64, 0i64);
    for &id in &z.ids[h * z.cap as usize..][..nrc] {
        if id < 0 || id as u32 >= l.cap_centroids {
            continue;
        }
        let b = off[id as usize].clamp(0, l.cap_tokens as i32) as usize;
        let e = off[id as usize + 1].clamp(b as i32, l.cap_tokens as i32) as usize;
        let full = (e - b) as i64;
        if used + full > room {
            dropped += full;
            continue;
        }
        used += full;
        out.extend(tok[b..e].iter().copied().filter(|&t| t < len));
    }
    out.extend(tail_start..len);
    let n_out = out.len().min(cap as usize);
    dropped += (out.len() - n_out) as i64;
    out.truncate(n_out);
    (out, dropped)
}

fn host_row(p: &HostPaged, h: usize, pos: i32, value: bool) -> Option<Vec<u16>> {
    let bs = p.block_size as usize;
    if pos < 0 || pos as usize / bs >= p.block_table.len() {
        return None;
    }
    let phys = p.block_table[pos as usize / bs];
    if phys < 0 {
        return None;
    }
    let (hd, elem) = (HEAD_DIM as usize, p.dtype.bytes_per_elem());
    let at = phys as usize * p.block_stride
        + ((pos as usize % bs) * p.kv_heads as usize + h) * hd * elem;
    let raw = &(if value { p.v_pool } else { p.k_pool })[at..at + hd * elem];
    Some(match p.dtype {
        KvDtypeTag::Bf16 => raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect(),
        KvDtypeTag::Fp8 => {
            let mut row = vec![half::bf16::ZERO; hd];
            let scale = if value { p.v_scale } else { p.k_scale };
            spark_runtime::kv_dequant::dequant_fp8_to_bf16(raw, scale, &mut row);
            row.iter().map(|b| b.to_bits()).collect()
        }
    })
}

/// What the gather kernel writes, computed on the host. Entries at or after `valid[h]` keep
/// the values `out` held (the kernel never writes them).
pub fn reference_gather(p: &HostPaged, l: &HostLists, z: &HostZones, out: &mut HostExact) {
    let (hd, cap) = (HEAD_DIM as usize, out.cap as usize);
    for h in 0..p.kv_heads as usize {
        let (positions, dropped) = exact_positions(p, l, z, h, out.cap);
        out.valid[h] = positions.len() as i32;
        out.dropped[h] = dropped.min(i64::from(i32::MAX)) as i32;
        for (e, &pos) in positions.iter().enumerate() {
            let at = (h * cap + e) * hd;
            match (host_row(p, h, pos, false), host_row(p, h, pos, true)) {
                (Some(k), Some(v)) => {
                    out.k[at..at + hd].copy_from_slice(&k);
                    out.v[at..at + hd].copy_from_slice(&v);
                    out.pos[h * cap + e] = pos;
                }
                _ => {
                    out.k[at..at + hd].fill(0);
                    out.v[at..at + hd].fill(0);
                    out.pos[h * cap + e] = -1;
                }
            }
        }
    }
}
