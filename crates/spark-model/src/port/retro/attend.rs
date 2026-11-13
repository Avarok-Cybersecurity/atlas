// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm: RetroInfer, arXiv 2505.02922 (VLDB 2026).

//! Responsibility: the retro attention a `SparsePlan::Custom` plan runs in
//! place of the paged decode: per logical sequence of the step, select the
//! zones (one selection per row, merged into the verify union for K+1 rows),
//! gather the exact zone, and run the merged exact plus estimation attention
//! for every row with its own causal trim, through the L34 kernels. Host code
//! runs on every step (eager), so `graph_safe()` is false: a capture gets the
//! dense tables instead (the seam's bypass), and the selector vetoes decode
//! graphs while it is registered.

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::metrics::Metrics;
use super::oracle_eval::Zones;
use super::runtime::Runtime;
use super::state::ZoneLists;
use super::verify;
use crate::layers::qwen3_attention::sparse_seam::{CustomAttention, SeamCtx, SeqBatch};

/// The custom attention of the retro selector.
pub struct RetroAttention {
    pub(crate) rt: Arc<Mutex<Runtime>>,
    pub(crate) metrics: &'static Metrics,
    pub(crate) failed: Arc<std::sync::atomic::AtomicBool>,
}

fn read_i32s(gpu: &dyn GpuBackend, src: DevicePtr, n: usize, stream: u64) -> Result<Vec<i32>> {
    let mut b = vec![0u8; n * 4];
    gpu.copy_d2h_on_stream(src, &mut b, stream)?;
    Ok(b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn i32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|&x| (x as i32).to_le_bytes()).collect()
}

/// Per KV head, the zones a selection wrote into `z`.
fn read_zones(
    gpu: &dyn GpuBackend,
    z: &ZoneLists,
    kv_heads: usize,
    stream: u64,
) -> Result<Vec<Zones>> {
    let r = read_i32s(
        gpu,
        z.retrieval_ids,
        kv_heads * z.retrieval_cap as usize,
        stream,
    )?;
    let e = read_i32s(gpu, z.estimation_ids, kv_heads * z.es_cap as usize, stream)?;
    let n = read_i32s(gpu, z.counts, kv_heads * 2, stream)?;
    let take = |v: &[i32], h: usize, cap: usize, cnt: i32| -> Vec<u32> {
        let k = usize::try_from(cnt.max(0)).unwrap_or(0).min(cap);
        v[h * cap..h * cap + k]
            .iter()
            .map(|&x| u32::try_from(x.max(0)).unwrap_or(0))
            .collect()
    };
    Ok((0..kv_heads)
        .map(|h| Zones {
            retrieval: take(&r, h, z.retrieval_cap as usize, n[h * 2]),
            estimation: take(&e, h, z.es_cap as usize, n[h * 2 + 1]),
        })
        .collect())
}

/// Write per-head zones into `z` (ids padded to capacity, then the counts).
fn write_zones(gpu: &dyn GpuBackend, z: &ZoneLists, zones: &[Zones], stream: u64) -> Result<()> {
    let (rc, ec) = (z.retrieval_cap as usize, z.es_cap as usize);
    let mut r = vec![0u32; zones.len() * rc];
    let mut e = vec![0u32; zones.len() * ec];
    let mut n = Vec::with_capacity(zones.len() * 2);
    for (h, zh) in zones.iter().enumerate() {
        let nr = zh.retrieval.len().min(rc);
        let ne = zh.estimation.len().min(ec);
        r[h * rc..h * rc + nr].copy_from_slice(&zh.retrieval[..nr]);
        e[h * ec..h * ec + ne].copy_from_slice(&zh.estimation[..ne]);
        n.push(nr as u32);
        n.push(ne as u32);
    }
    gpu.copy_h2d_async(&i32_bytes(&r), z.retrieval_ids, stream)?;
    gpu.copy_h2d_async(&i32_bytes(&e), z.estimation_ids, stream)?;
    gpu.copy_h2d_async(&i32_bytes(&n), z.counts, stream)
}

/// Add the blocks of positions `[lo, hi)` to `blocks`.
fn add_span(blocks: &mut std::collections::BTreeSet<u32>, lo: u32, hi: u32, block_size: u32) {
    if hi > lo {
        blocks.extend(lo / block_size..=(hi - 1) / block_size);
    }
}

/// Debug (`AVAROK_RETRO_SELECTION_LOG`, off by default): one log line per
/// (decode step, layer) naming the KV blocks of the exact window the selection
/// chose (sink, the members of every retrieval cluster of every KV head, and
/// the exact tail), in the shape `niah_grid.py --selection-log` reads:
/// `seq=<id> layer=<n> ... blocks=[b0,b1,...]`. The estimation zone is not part
/// of the window (it is summarized, not read). A failed read-back is logged and
/// never fails the step.
#[allow(clippy::too_many_arguments)]
fn log_selection(
    gpu: &dyn GpuBackend,
    index: &super::state::LayerIndex,
    z: &ZoneLists,
    cfg: &super::config::RetroCfg,
    uid: u64,
    layer: u32,
    len: u32,
    stream: u64,
) {
    let run = || -> Result<String> {
        let zones = read_zones(gpu, z, cfg.kv_heads as usize, stream)?;
        let (_, off_stride, tok_stride) = index.layout.head_strides();
        let cn = index.counts;
        let bs = cfg.block_size.max(1);
        let mut blocks = std::collections::BTreeSet::new();
        add_span(&mut blocks, 0, cn.sink.min(cn.indexed_end), bs);
        add_span(&mut blocks, cn.indexed_end, len, bs);
        let n_off = cn.n_centroids as usize + 1;
        let listed = cn.listed as usize;
        for (h, zh) in zones.iter().enumerate() {
            if zh.retrieval.is_empty() {
                continue;
            }
            let offs = read_i32s(
                gpu,
                index.list_offsets.offset(h * off_stride * 4),
                n_off,
                stream,
            )?;
            let toks = read_i32s(
                gpu,
                index.list_tokens.offset(h * tok_stride * 4),
                listed,
                stream,
            )?;
            for &cl in &zh.retrieval {
                let cl = cl as usize;
                let (Some(&a), Some(&b)) = (offs.get(cl), offs.get(cl + 1)) else {
                    continue;
                };
                let (a, b) = (
                    usize::try_from(a.max(0)).unwrap_or(0),
                    usize::try_from(b.max(0)).unwrap_or(0),
                );
                for &t in toks.get(a..b.min(toks.len())).unwrap_or(&[]) {
                    blocks.insert(u32::try_from(t.max(0)).unwrap_or(0) / bs);
                }
            }
        }
        Ok(blocks
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(","))
    };
    match run() {
        Ok(list) => {
            tracing::info!("retro selection: seq={uid} layer={layer} len={len} blocks=[{list}]")
        }
        Err(e) => tracing::warn!(
            "retro selection log: attn layer {layer} read-back failed ({e:#}); line skipped"
        ),
    }
}

impl RetroAttention {
    fn run_inner(&self, c: &SeamCtx, s: &SeqBatch, q: DevicePtr, out: DevicePtr) -> Result<()> {
        let mut guard = self
            .rt
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let rt = &mut *guard;
        let layer = u32::try_from(c.attn_layer).unwrap_or(u32::MAX);
        let groups = rt.groups.clone();
        let Runtime { dev, seqs, .. } = rt;
        let dev = dev
            .as_mut()
            .context("retro: no device state at attention")?;
        let cfg = dev.cfg.clone();
        let kvh = cfg.kv_heads as usize;
        let q_row = |r: usize| q.offset(r * s.q_row_stride as usize * 2);
        let out_row = |r: usize| out.offset(r * (cfg.num_q_heads * cfg.head_dim) as usize * 2);
        let bufs = dev.bufs.as_mut().context("retro: step buffers missing")?;
        let shared = super::config::MAX_VERIFY_ROWS as usize;
        for g in &groups {
            let i = seqs
                .by_uid(g.uid)
                .context("retro: a planned sequence vanished")?;
            let index = seqs
                .get(i)
                .layers
                .get(&layer)
                .and_then(|l| l.index())
                .context("retro: a planned index vanished")?;
            let zi = if g.rows == 1 {
                bufs.zones[0].bind(index);
                super::select::select_zones(
                    c.gpu,
                    index,
                    q_row(g.first_row),
                    &cfg,
                    &mut bufs.zones[0],
                    c.stream,
                )?;
                0
            } else {
                let mut per_row: Vec<Vec<Zones>> = Vec::with_capacity(g.rows);
                for (j, z) in bufs.zones.iter_mut().enumerate().take(g.rows) {
                    z.bind(index);
                    super::select::select_zones(
                        c.gpu,
                        index,
                        q_row(g.first_row + j),
                        &cfg,
                        z,
                        c.stream,
                    )?;
                    per_row.push(read_zones(c.gpu, z, kvh, c.stream)?);
                }
                let mut union = Vec::with_capacity(kvh);
                let mut largest = 0usize;
                for h in 0..kvh {
                    let rows: Vec<Zones> = per_row.iter().map(|r| r[h].clone()).collect();
                    let u = verify::union_select(
                        &rows,
                        cfg.retrieval_cap as usize,
                        cfg.es_cap as usize,
                    );
                    largest = largest.max(u.zones.retrieval.len());
                    union.push(u.zones);
                }
                self.metrics.set_union_size(largest as u64);
                write_zones(c.gpu, &bufs.zones[shared], &union, c.stream)?;
                bufs.zones[shared].bind(index);
                shared
            };
            if super::api::selection_log() {
                log_selection(
                    c.gpu,
                    index,
                    &bufs.zones[zi],
                    &cfg,
                    g.uid,
                    layer,
                    g.len,
                    c.stream,
                );
            }
            let last = g.first_row + g.rows - 1;
            let lens = [g.len];
            let tables = [g.table.as_slice()];
            let first = [g.len.saturating_sub(1)];
            let one = [1u32];
            let gather_batch = SeqBatch {
                n: 1,
                block_table: s
                    .block_table
                    .offset(last * s.max_blocks_per_seq as usize * 4),
                seq_lens: s.seq_lens.offset(last * 4),
                max_blocks_per_seq: s.max_blocks_per_seq,
                host_seq_lens: &lens,
                seq_ids: &[],
                rows_per_seq: 1,
                q_row_stride: s.q_row_stride,
                q_rows: 1,
                first_pos: &first,
                new_tokens: &one,
                host_block_tables: &tables,
                block_table_ptrs: DevicePtr::NULL,
            };
            super::gather::gather_exact(
                c.gpu,
                &c.kv,
                &gather_batch,
                &bufs.zones[zi],
                &mut bufs.exact,
                c.stream,
            )?;
            for j in 0..g.rows {
                bufs.exact.row_trim = verify::row_trim(g.rows, j);
                super::attn::estimation_attention(
                    c.gpu,
                    q_row(g.first_row + j),
                    &bufs.exact,
                    index,
                    &bufs.zones[zi],
                    out_row(g.first_row + j),
                    None,
                    c.stream,
                )?;
            }
            bufs.exact.row_trim = 0;
            let cn = index.counts;
            let exact = cn.sink.min(cn.indexed_end)
                + g.len.saturating_sub(cn.indexed_end)
                + cn.nprobe * cfg.params.avg_cluster;
            self.metrics
                .set_touched_ratio(f64::from(exact + cn.es) / f64::from(g.len.max(1)));
            self.metrics.count_decode(1);
        }
        Ok(())
    }
}

impl CustomAttention for RetroAttention {
    fn run(&self, c: &SeamCtx, s: &SeqBatch, q: DevicePtr, out: DevicePtr) -> Result<()> {
        let res = self.run_inner(c, s, q, out);
        if let Err(e) = &res {
            if !self.failed.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::error!(
                    "retro: attention failed on attn layer {} ({e:#}); retro attention is off for the rest of the serve",
                    c.attn_layer
                );
            }
            self.metrics
                .count_fallback(super::metrics::Fallback::Kernel);
        }
        res
    }

    fn graph_safe(&self) -> bool {
        false
    }
}
