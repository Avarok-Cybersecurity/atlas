// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the fixtures of `retro_attn_microtest` (port lane L34b): a deterministic
//! generator, device upload and download, a paged sequence with a CSR cluster index and a
//! retrieval zone for the gather cells, an exact buffer with an estimation zone over a random
//! index for the attention cells, and the launch descriptors both kinds of cell run.

use anyhow::Result;
use spark_model::layers::qwen3_attention::sparse_seam::{KvDtypeTag, KvLayerView};
use spark_model::port::retro::attn::{AttnLaunch, IndexView};
use spark_model::port::retro::gather::{ExactOut, GatherLaunch, HostExact};
use spark_model::port::retro::state::{IndexCounts, IndexLayout, IndexListRef};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

pub const HD: usize = 256;
pub const BS: u32 = 16;
pub const SENT: u8 = 0xAB;
pub const SCALE: f32 = 1.0 / 16.0;

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
    pub fn below(&mut self, n: u32) -> u32 {
        self.next() % n.max(1)
    }
    pub fn unit(&mut self) -> f32 {
        self.next() as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
    }
}

pub fn bits(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}

pub fn up<T: Copy>(gpu: &dyn GpuBackend, v: &[T]) -> Result<DevicePtr> {
    let bytes = std::mem::size_of_val(v);
    let p = gpu.alloc(bytes.max(256))?;
    // SAFETY: a plain-old-data slice viewed as bytes for the upload.
    let raw = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, bytes) };
    if !raw.is_empty() {
        gpu.copy_h2d(raw, p)?;
    }
    Ok(p)
}

pub fn down<T: Copy + Default>(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<T>> {
    let mut out = vec![T::default(); n];
    let bytes = std::mem::size_of_val(&out[..]);
    // SAFETY: a plain-old-data destination viewed as bytes.
    let raw = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, bytes) };
    gpu.copy_d2h(p, raw)?;
    Ok(out)
}

/// A paged sequence, its clustered index (CSR lists) and a retrieval zone.
pub struct GatherFx {
    pub kvh: u32,
    pub dtype: KvDtypeTag,
    pub k: Vec<u8>,
    pub v: Vec<u8>,
    pub table: Vec<i32>,
    pub lens: Vec<i32>,
    pub header: Vec<i32>,
    pub offsets: Vec<i32>,
    pub tokens: Vec<i32>,
    pub capc: u32,
    pub capt: u32,
    pub ids: Vec<i32>,
    pub counts: Vec<i32>,
    pub rcap: u32,
}

impl GatherFx {
    pub fn stride(&self) -> usize {
        BS as usize * self.kvh as usize * HD * self.dtype.bytes_per_elem()
    }
}

/// `rows` rows ending at `len + rows - 1`; tokens `[4, len - 64)` clustered at random into
/// `capc - 8` clusters (the last 8 stay empty); `ret` retrieval ids per head with an invalid
/// id, an out-of-range id and an empty cluster among them.
pub fn gather_fx(
    dtype: KvDtypeTag,
    kvh: u32,
    len: i32,
    rows: i32,
    ret: u32,
    seed: u64,
) -> GatherFx {
    let mut r = Rng(seed);
    let blocks = ((len + rows - 1) as u32).div_ceil(BS) as usize + 2;
    let stride = BS as usize * kvh as usize * HD * dtype.bytes_per_elem();
    let mut fill = |n: usize| -> Vec<u8> {
        // E4M3 NaN bytes are replaced so every compared value is a number.
        (0..n)
            .map(|_| match r.next() as u8 {
                b if b & 0x7f == 0x7f => 0x38,
                b => b,
            })
            .collect()
    };
    let (k, v) = (fill(blocks * stride), fill(blocks * stride));
    let mut table: Vec<i32> = (0..blocks as i32).collect();
    for i in (1..table.len()).rev() {
        let j = r.below(i as u32 + 1) as usize;
        table.swap(i, j);
    }
    let (sink, ie) = (4u32, (len - 64) as u32);
    let capc = (ie / 16).max(8) + 8;
    let capt = ie;
    let (mut offsets, mut tokens) = (Vec::new(), vec![0i32; (kvh * capt) as usize]);
    for h in 0..kvh as usize {
        let mut lists = vec![Vec::new(); capc as usize];
        for t in sink..ie {
            lists[r.below(capc - 8) as usize].push(t as i32);
        }
        let mut at = 0usize;
        offsets.push(0);
        for l in &lists {
            tokens[h * capt as usize + at..][..l.len()].copy_from_slice(l);
            at += l.len();
            offsets.push(at as i32);
        }
    }
    let header = IndexCounts {
        indexed_end: ie,
        sink,
        n_centroids: capc,
        ..IndexCounts::default()
    };
    let rcap = ret + 8;
    let mut ids = vec![0i32; (kvh * rcap) as usize];
    let mut counts = vec![0i32; kvh as usize * 2];
    for h in 0..kvh as usize {
        counts[h * 2] = ret as i32;
        for i in 0..ret as usize {
            ids[h * rcap as usize + i] = match i {
                3 => -1,
                7 => capc as i32 + 5,
                11 => capc as i32 - 1, // empty
                _ => r.below(capc) as i32,
            };
        }
    }
    GatherFx {
        kvh,
        dtype,
        k,
        v,
        table,
        lens: (0..rows).map(|i| len + i).collect(),
        header: header.to_words().to_vec(),
        offsets,
        tokens,
        capc,
        capt,
        ids,
        counts,
        rcap,
    }
}

/// Upload a gather fixture; the pool uses K scale 0.75 and V scale 1.5.
pub fn gather_launch(gpu: &dyn GpuBackend, f: &GatherFx, out: ExactOut) -> Result<GatherLaunch> {
    let stride = f.stride();
    let kv = KvLayerView {
        k_pool: up(gpu, &f.k)?,
        v_pool: up(gpu, &f.v)?,
        dtype: f.dtype,
        block_size: BS,
        kv_heads: f.kvh,
        head_dim: HD as u32,
        kv_scale: 0.75,
        v_scale: 1.5,
        block_stride_bytes: stride as u64,
        cache_stride: (stride / f.dtype.bytes_per_elem()) as u64,
    };
    let layout = IndexLayout {
        kv_heads: f.kvh,
        head_dim: HD as u32,
        cap_centroids: f.capc,
        cap_tokens: f.capt,
    };
    Ok(GatherLaunch {
        kv,
        block_table: up(gpu, &f.table)?,
        row_lens: up(gpu, &f.lens)?,
        rows: f.lens.len() as u32,
        max_blocks_per_seq: f.table.len() as u32,
        retrieval_ids: up(gpu, &f.ids)?,
        counts: up(gpu, &f.counts)?,
        retrieval_cap: f.rcap,
        lists: IndexListRef {
            header: up(gpu, &f.header)?,
            cluster_size: DevicePtr::NULL, // the gather reads the CSR offsets
            list_offsets: up(gpu, &f.offsets)?,
            list_tokens: up(gpu, &f.tokens)?,
            layout,
        },
        out,
    })
}

/// An exact buffer of `cap` entries per head, every byte set to [`SENT`].
pub fn exact_out(gpu: &dyn GpuBackend, kvh: u32, cap: u32) -> Result<(DevicePtr, ExactOut)> {
    let rows = (kvh * cap) as usize;
    let bytes = rows * HD * 4 + rows * 4 + 4096;
    let base = gpu.alloc(bytes)?;
    gpu.memset(base, SENT, bytes)?;
    let v = base.offset(rows * HD * 2);
    let pos = v.offset(rows * HD * 2);
    let valid = pos.offset((rows * 4).div_ceil(256) * 256);
    let dropped = valid.offset(256);
    Ok((
        base,
        ExactOut {
            k: base,
            v,
            pos,
            valid,
            dropped,
            kv_heads: kvh,
            cap,
        },
    ))
}

/// Attention inputs: an exact buffer and an estimation zone over a random index.
pub struct AttnFx {
    pub kvh: u32,
    pub qh: u32,
    pub rows: u32,
    pub row_trim: u32,
    pub q: Vec<u16>,
    pub exact: HostExact,
    pub ids: Vec<i32>,
    pub counts: Vec<i32>,
    pub es_cap: u32,
    pub cent: Vec<u16>,
    pub vsum: Vec<u16>,
    pub size: Vec<i32>,
    pub capc: u32,
}

/// `n_exact` entries per head (every 97th unreadable), `n_est` estimation ids per head (one
/// out of range; every 13th cluster empty), queries scaled by `gain`.
#[allow(clippy::too_many_arguments)]
pub fn attn_fx(
    kvh: u32,
    qh: u32,
    rows: u32,
    trim: u32,
    n_exact: usize,
    n_est: usize,
    gain: f32,
    seed: u64,
) -> AttnFx {
    let mut r = Rng(seed);
    let capc = 8192u32;
    let cap = (n_exact as u32).max(1);
    let mut exact = HostExact::filled(kvh, cap, 0, -1);
    for h in 0..kvh as usize {
        exact.valid[h] = n_exact as i32;
        exact.dropped[h] = 0;
        for e in 0..n_exact {
            let at = (h * cap as usize + e) * HD;
            for d in 0..HD {
                exact.k[at + d] = bits(r.unit());
                exact.v[at + d] = bits(r.unit());
            }
            exact.pos[h * cap as usize + e] = if e % 97 == 5 { -1 } else { e as i32 };
        }
    }
    let es_cap = (n_est as u32).max(1);
    let mut ids = vec![-1; (kvh * es_cap) as usize];
    let size: Vec<i32> = (0..kvh * capc)
        .map(|i| if i % 13 == 0 { 0 } else { 1 + (i % 31) as i32 })
        .collect();
    let mut counts = vec![0i32; kvh as usize * 2];
    for h in 0..kvh as usize {
        counts[h * 2 + 1] = n_est as i32;
        for i in 0..n_est {
            ids[h * es_cap as usize + i] = if i == 2 {
                capc as i32 + 3
            } else {
                r.below(capc) as i32
            };
        }
    }
    let n = (kvh * capc) as usize * HD;
    let cent = (0..n).map(|_| bits(r.unit())).collect();
    let vsum = (0..n)
        .map(|i| bits(r.unit() * size[i / HD] as f32))
        .collect();
    let q = (0..rows as usize * qh as usize * HD)
        .map(|_| bits(r.unit() * gain))
        .collect();
    AttnFx {
        kvh,
        qh,
        rows,
        row_trim: trim,
        q,
        exact,
        ids,
        counts,
        es_cap,
        cent,
        vsum,
        size,
        capc,
    }
}

/// Upload an attention fixture with fresh output and LSE buffers.
pub fn attn_launch(gpu: &dyn GpuBackend, f: &AttnFx) -> Result<AttnLaunch> {
    let ex = &f.exact;
    let (rows, qh) = (f.rows as usize, f.qh as usize);
    Ok(AttnLaunch {
        q: up(gpu, &f.q)?,
        q_row_stride: f.qh * HD as u32,
        rows: f.rows,
        row_trim: f.row_trim,
        q_heads: f.qh,
        scale: SCALE,
        exact: ExactOut {
            k: up(gpu, &ex.k)?,
            v: up(gpu, &ex.v)?,
            pos: up(gpu, &ex.pos)?,
            valid: up(gpu, &ex.valid)?,
            dropped: up(gpu, &ex.dropped)?,
            kv_heads: f.kvh,
            cap: ex.cap,
        },
        est_ids: up(gpu, &f.ids)?,
        counts: up(gpu, &f.counts)?,
        es_cap: f.es_cap,
        index: IndexView {
            centroids: up(gpu, &f.cent)?,
            value_sum: up(gpu, &f.vsum)?,
            cluster_size: up(gpu, &f.size)?,
            cap_centroids: f.capc,
            kv_heads: f.kvh,
            head_dim: HD as u32,
        },
        out: gpu.alloc(rows * qh * HD * 2)?,
        lse: gpu.alloc(rows * qh * 4)?,
    })
}
