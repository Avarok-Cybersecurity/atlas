// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm after Quest (page min/max upper-bound selection) and the page-sparse prototype in
// arczhi/llama-tierkv scripts-5060ti/proto/proto_page_sparse.cu (MIT, Copyright (c) 2023-2026 The
// ggml authors; file has no header of its own). Atlas GPU oracle; no prototype source was read.

//! Responsibility: the page-sparse microtest's fixture: geometry constants, the generator, host
//! codecs, the device `Case`, the summary fold and the CPU reference checks.

use anyhow::Result;
use spark_model::port::sparse_quest::launch::{
    self, Kernels, LayerBufs, StepRows, TableRef, UpdateRange,
};
use spark_model::port::sparse_quest::reference::{self as r, KPool, RefGeom};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::page_summary::{LayerGeom, PageSummaryBuffers, SummaryLayout};

pub(super) const NQ: usize = 16;
pub(super) const KVH: usize = 4;
pub(super) const HD: usize = 256;
pub(super) const BS: usize = 16;
pub(super) const PB: usize = 4;
pub(super) const R: usize = 4;
pub(super) const GROUPS: usize = 2;
pub(super) const GEOM: RefGeom = RefGeom {
    kv_heads: KVH,
    head_dim: HD,
    block_size: BS,
    page_blocks: PB,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

pub(super) fn bf16(v: f32) -> u16 {
    let u = v.to_bits();
    ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}

pub(super) fn up(gpu: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(b.len().max(256))?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}

pub(super) fn i32s(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub(super) fn down(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

pub(super) fn as_i32(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub(super) fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// One problem on the device: pool, tables (rows repeat the group's table), lengths, q.
pub(super) struct Case {
    pub(super) eb: usize,
    pub(super) pool_host: Vec<u8>,
    pub(super) tables: Vec<Vec<i32>>,
    pub(super) row_stride: usize,
    pub(super) lens: Vec<i32>,
    pub(super) q_host: Vec<f32>,
    pub(super) pool: DevicePtr,
    pub(super) bt: DevicePtr,
    pub(super) sl: DevicePtr,
    pub(super) q: DevicePtr,
    pub(super) num_blocks: usize,
}

pub(super) fn build(gpu: &dyn GpuBackend, len: usize, eb: usize, seed: u64) -> Result<Case> {
    let mut rng = Rng(seed | 1);
    let per = (len + R).div_ceil(BS);
    let num_blocks = GROUPS * per + 3;
    let stride = BS * KVH * HD * eb;
    let mut pool_host = vec![0u8; num_blocks * stride];
    if eb == 2 {
        for c in pool_host.chunks_exact_mut(2) {
            c.copy_from_slice(&bf16(rng.unit() * 3.0).to_le_bytes());
        }
    } else {
        for b in pool_host.iter_mut() {
            let v = (rng.next() & 0xFF) as u8;
            *b = if v & 0x7F == 0x7F { v & 0xFE } else { v };
        }
    }
    let mut ids: Vec<i32> = (0..num_blocks as i32).collect();
    for i in (1..ids.len()).rev() {
        ids.swap(i, (rng.next() % (i as u64 + 1)) as usize);
    }
    let row_stride = per + 5;
    let mut tables = Vec::new();
    let mut flat = Vec::new();
    let mut lens = Vec::new();
    for g in 0..GROUPS {
        let mut t = ids[g * per..(g + 1) * per].to_vec();
        t.resize(row_stride, 0);
        for rr in 0..R {
            flat.extend_from_slice(&t);
            lens.push((len + rr) as i32);
        }
        tables.push(t);
    }
    let q_host: Vec<f32> = (0..GROUPS * R * NQ * HD).map(|_| rng.unit()).collect();
    let q_bits: Vec<u8> = q_host.iter().flat_map(|&v| bf16(v).to_le_bytes()).collect();
    let q_host = q_host.iter().map(|&v| r::bf16_val(bf16(v))).collect();
    Ok(Case {
        eb,
        pool: up(gpu, &pool_host)?,
        pool_host,
        bt: up(gpu, &i32s(&flat))?,
        sl: up(gpu, &i32s(&lens))?,
        q: up(gpu, &q_bits)?,
        tables,
        row_stride,
        lens,
        q_host,
        num_blocks,
    })
}

pub(super) fn kpool(c: &Case) -> KPool<'_> {
    KPool {
        data: &c.pool_host,
        elem_bytes: c.eb,
        block_stride: BS * KVH * HD * c.eb,
        geom: GEOM,
    }
}

pub(super) fn layout(c: &Case) -> Result<SummaryLayout> {
    SummaryLayout::new(
        c.num_blocks as u32,
        BS as u32,
        vec![LayerGeom {
            kv_heads: KVH as u32,
            head_dim: HD as u32,
        }],
    )
}

pub(super) fn bufs(c: &Case, l: &SummaryLayout, s: &PageSummaryBuffers) -> LayerBufs {
    LayerBufs::new(
        l,
        s.base,
        0,
        c.pool,
        (BS * KVH * HD * c.eb) as u64,
        c.eb as u32,
    )
}

pub(super) fn rows(c: &Case) -> StepRows {
    StepRows {
        block_table: c.bt,
        seq_lens: c.sl,
        rows_per_seq: R as u32,
        groups: GROUPS as u32,
        row_stride: c.row_stride as u32,
    }
}

/// Fold the prefill (host ranges, 1,000-token chunks) then the verify rows (device ranges).
pub(super) fn fold_all(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    c: &Case,
    lb: &LayerBufs,
    len: usize,
    s: u64,
) -> Result<()> {
    for g in 0..GROUPS {
        let table = c.bt.offset(g * R * c.row_stride * 4);
        let mut first = 0;
        while first < len - 1 {
            let count = (len - 1 - first).min(1000);
            let range = UpdateRange::Host {
                first_pos: first as u32,
                count: count as u32,
            };
            launch::launch_update(
                gpu,
                k,
                lb,
                TableRef::Flat(table),
                c.row_stride as u32,
                range,
                s,
            )?;
            first += count;
        }
    }
    let dev = UpdateRange::Device {
        seq_lens: c.sl,
        rows_per_seq: R as u32,
        groups: GROUPS as u32,
    };
    launch::launch_update(
        gpu,
        k,
        lb,
        TableRef::Flat(c.bt),
        c.row_stride as u32,
        dev,
        s,
    )?;
    gpu.synchronize(s)
}

/// Check the records of pages `[first, end)` against the reference full build.
pub(super) fn check_summaries(
    gpu: &dyn GpuBackend,
    c: &Case,
    l: &SummaryLayout,
    s: &PageSummaryBuffers,
    first: usize,
    end: usize,
) -> Result<usize> {
    let all = down(gpu, s.base, l.total_bytes() as usize)?;
    let p = kpool(c);
    let mut bad = 0;
    for t in &c.tables {
        for page in first..end {
            let slot = t[page * PB] as u32;
            for h in 0..KVH {
                let want = r::page_slot(&p, t, page, h);
                let vo = l.vals_offset(0, slot, h as u32) as usize;
                let half = |i: usize| u16::from_le_bytes([all[vo + 2 * i], all[vo + 2 * i + 1]]);
                let kmin: Vec<u16> = (0..HD).map(half).collect();
                let kmax: Vec<u16> = (HD..2 * HD).map(half).collect();
                let mo = l.meta_offset(0, slot, h as u32) as usize;
                let word = |i: usize| {
                    u64::from_le_bytes(all[mo + 8 * i..mo + 8 * i + 8].try_into().unwrap())
                };
                if kmin != want.kmin
                    || kmax != want.kmax
                    || word(0) != want.mask
                    || word(1) != want.fp
                    || word(2) != 0
                {
                    bad += 1;
                }
            }
        }
    }
    Ok(bad)
}

/// The reference scores of group g's candidate pages (f64).
pub(super) fn ref_scores(c: &Case, g: usize, rp: usize, sink: usize) -> Vec<f64> {
    let p = kpool(c);
    let q: Vec<&[f32]> = (0..R)
        .map(|rr| &c.q_host[(g * R + rr) * NQ * HD..(g * R + rr + 1) * NQ * HD])
        .collect();
    let mut out = vec![f64::NAN; rp];
    for (page, o) in out.iter_mut().enumerate().skip(sink) {
        let slots: Vec<_> = (0..KVH)
            .map(|h| r::page_slot(&p, &c.tables[g], page, h))
            .collect();
        *o = r::page_score(&q, NQ, &slots, HD, false);
    }
    out
}
