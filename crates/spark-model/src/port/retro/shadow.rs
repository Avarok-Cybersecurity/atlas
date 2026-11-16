// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm: RetroInfer, arXiv 2505.02922 (VLDB 2026).

//! Responsibility: shadow mode (`--retro-shadow`): decide go or no-go for
//! RetroInfer on this model before trusting any kernel. Every
//! `AVAROK_RETRO_SHADOW_EVERY`-th decode step, for the attention layers in
//! `AVAROK_RETRO_SHADOW_LAYERS`, the layer's K and V for the context and the
//! step's query are copied to the host, and a worker thread runs the CPU
//! oracle: it builds the layer's index once per sequence (then grows it with
//! the same update policy as the device path), selects the zones, runs the
//! merged attention and compares it with dense attention. Results are the
//! `atlas_retro_shadow_*` series and one log line per sample. The served output
//! never changes: shadow mode reads, it does not plan.
//!
//! One sample is in flight at a time; a step that comes while the worker is
//! busy is not sampled. The copies are large (a 64K-token layer in BF16 is
//! 256 MiB of K and V), which is why it is opt-in and sampled.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::config::RetroParams;
use super::metrics::Metrics;
use super::oracle::{HeadKv, OracleLayer};
use super::oracle_eval;
use super::update;
use crate::layers::qwen3_attention::sparse_seam::{KvDtypeTag, KvLayerView};

/// One sample: a layer's K/V and one query row.
pub struct Sample {
    pub uid: u64,
    pub attn_layer: u32,
    pub len: u32,
    pub block_size: u32,
    pub head_dim: usize,
    pub group: usize,
    pub scale: f32,
    /// `[kv_heads][len][head_dim]`.
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    /// `[num_q_heads][head_dim]`.
    pub q: Vec<f32>,
}

enum Msg {
    Sample(Box<Sample>),
    Forget(u64),
}

/// The shadow sampler and its worker.
pub struct Shadow {
    params: RetroParams,
    tx: Mutex<Option<Sender<Msg>>>,
    busy: Arc<AtomicBool>,
    metrics: &'static Metrics,
}

impl Shadow {
    pub fn new(params: RetroParams, metrics: &'static Metrics) -> Self {
        Self {
            params,
            tx: Mutex::new(None),
            busy: Arc::new(AtomicBool::new(false)),
            metrics,
        }
    }

    /// Whether `step` of attention layer `layer` is sampled at context `len`.
    pub fn due(&self, step: u64, layer: u32, len: u32) -> bool {
        self.params.shadow_layer(layer)
            && step.is_multiple_of(u64::from(self.params.shadow_every))
            && update::initial_end(len, &self.params, 1).is_some()
            && !self.busy.load(Ordering::Acquire)
    }

    fn sender(&self) -> Sender<Msg> {
        let mut tx = self
            .tx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        tx.get_or_insert_with(|| {
            let (tx, rx) = channel();
            let params = self.params.clone();
            let busy = self.busy.clone();
            let metrics = self.metrics;
            std::thread::Builder::new()
                .name("retro-shadow".into())
                .spawn(move || worker(rx, params, busy, metrics))
                .expect("retro: spawning the shadow worker");
            tx
        })
        .clone()
    }

    /// Copy one sample from the device and hand it to the worker.
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &self,
        gpu: &dyn GpuBackend,
        kv: &KvLayerView,
        table: &[u32],
        len: u32,
        q: DevicePtr,
        num_q_heads: u32,
        scale: f32,
        uid: u64,
        attn_layer: u32,
        stream: u64,
    ) -> Result<()> {
        let (k, v) = read_kv(gpu, kv, table, len, stream)?;
        let hd = kv.head_dim as usize;
        let mut qb = vec![0u8; num_q_heads as usize * hd * 2];
        gpu.copy_d2h_on_stream(q, &mut qb, stream)?;
        let q: Vec<f32> = qb
            .chunks_exact(2)
            .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect();
        let s = Sample {
            uid,
            attn_layer,
            len,
            block_size: kv.block_size,
            head_dim: hd,
            group: (num_q_heads / kv.kv_heads.max(1)) as usize,
            scale,
            k,
            v,
            q,
        };
        self.busy.store(true, Ordering::Release);
        if self.sender().send(Msg::Sample(Box::new(s))).is_err() {
            self.busy.store(false, Ordering::Release);
        }
        Ok(())
    }

    /// Drop a released sequence's oracle indexes.
    pub fn forget(&self, uid: u64) {
        let tx = self
            .tx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(tx) = tx.as_ref() {
            let _ = tx.send(Msg::Forget(uid));
        }
    }
}

/// Decode one stored K or V element.
fn elem(dtype: KvDtypeTag, b: &[u8], i: usize, scale: f32) -> f32 {
    match dtype {
        KvDtypeTag::Bf16 => {
            half::bf16::from_bits(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])).to_f32()
        }
        KvDtypeTag::Fp8 => avarok_core::numeric::fp8_e4m3_to_f32(b[i]) * scale,
    }
}

/// Read positions `[0, len)` of one layer's K and V through a host block
/// table into `[kv_heads][len][head_dim]` f32 (the paged layout is
/// `[block][slot][kv_head][head_dim]`; FP8 is dequantized with the layer's scales).
pub fn read_kv(
    gpu: &dyn GpuBackend,
    kv: &KvLayerView,
    table: &[u32],
    len: u32,
    stream: u64,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let bs = kv.block_size.max(1) as usize;
    let h = kv.kv_heads as usize;
    let hd = kv.head_dim as usize;
    let eb = kv.dtype.bytes_per_elem();
    let block_bytes = bs * h * hd * eb;
    let len = len as usize;
    let blocks = len.div_ceil(bs);
    anyhow::ensure!(
        table.len() >= blocks,
        "retro shadow: {len} tokens need {blocks} blocks, the table has {}",
        table.len()
    );
    let mut k = vec![vec![0f32; len * hd]; h];
    let mut v = vec![vec![0f32; len * hd]; h];
    let mut buf = vec![0u8; block_bytes];
    for (pool, out, scale) in [
        (kv.k_pool, &mut k, kv.kv_scale),
        (kv.v_pool, &mut v, kv.v_scale),
    ] {
        for (lb, &blk) in table[..blocks].iter().enumerate() {
            let src = pool.offset(blk as usize * kv.block_stride_bytes as usize);
            gpu.copy_d2h_on_stream(src, &mut buf, stream)
                .context("retro shadow: reading a KV block")?;
            for slot in 0..bs {
                let pos = lb * bs + slot;
                if pos >= len {
                    break;
                }
                for (head, dst) in out.iter_mut().enumerate() {
                    let base = (slot * h + head) * hd;
                    for (d, o) in dst[pos * hd..(pos + 1) * hd].iter_mut().enumerate() {
                        *o = elem(kv.dtype, &buf, base + d, scale);
                    }
                }
            }
        }
    }
    Ok((k, v))
}

/// Measure one sample against the sequence's oracle index (built on first use).
pub fn measure(
    s: &Sample,
    p: &RetroParams,
    idx: &mut Option<OracleLayer>,
) -> Option<oracle_eval::Quality> {
    let heads: Vec<HeadKv<'_>> =
        s.k.iter()
            .zip(&s.v)
            .map(|(k, v)| HeadKv {
                k,
                v,
                head_dim: s.head_dim,
            })
            .collect();
    if idx.is_none() {
        let end = update::initial_end(s.len, p, s.block_size)?;
        *idx = Some(OracleLayer::build(&heads, p, end));
    }
    let layer = idx.as_mut()?;
    if layer.counts.indexed_end > s.len {
        // The sequence rolled back below its index; rebuild.
        let end = update::initial_end(s.len, p, s.block_size)?;
        *layer = OracleLayer::build(&heads, p, end);
    }
    layer.catch_up(&heads, p, s.len, s.block_size);
    Some(oracle_eval::measure_layer(
        &heads,
        &layer.heads,
        &layer.counts,
        &s.q,
        s.group,
        s.len,
        s.scale,
        None,
    ))
}

fn worker(rx: Receiver<Msg>, p: RetroParams, busy: Arc<AtomicBool>, metrics: &'static Metrics) {
    let mut indexes: HashMap<(u64, u32), OracleLayer> = HashMap::new();
    while let Ok(msg) = rx.recv() {
        match msg {
            Msg::Forget(uid) => indexes.retain(|(u, _), _| *u != uid),
            Msg::Sample(s) => {
                let key = (s.uid, s.attn_layer);
                let mut slot = indexes.remove(&key);
                let built_before = slot.as_ref().map_or(0.0, |l| l.build_secs);
                if let Some(q) = measure(&s, &p, &mut slot) {
                    metrics.record_shadow(
                        s.attn_layer,
                        q.mass_exact,
                        q.mass_total,
                        q.out_rel_err,
                        q.jaccard,
                    );
                    let l = slot.as_ref().map_or(0.0, |l| l.build_secs);
                    metrics.add_build_secs(l - built_before);
                    tracing::info!(
                        "retro shadow: attn layer {} ctx {} mass recall {:.4} (exact {:.4}) out rel err {:.4} jaccard {:.3} ({} centroids, index cpu time {:.1} s)",
                        s.attn_layer,
                        s.len,
                        q.mass_total,
                        q.mass_exact,
                        q.out_rel_err,
                        q.jaccard,
                        slot.as_ref().map_or(0, |l| l.counts.n_centroids),
                        l,
                    );
                }
                if let Some(l) = slot {
                    indexes.insert(key, l);
                }
                busy.store(false, Ordering::Release);
            }
        }
    }
}
