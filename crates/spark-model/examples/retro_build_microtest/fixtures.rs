// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the fixtures of `retro_build_microtest`: synthetic paged KV layers (BF16
//! Gaussian, BF16 uniform, FP8) with a scrambled block table, the device state of one layer (index,
//! build scratch, zone lists), the build driver, device read-back helpers, and the check report.

use anyhow::Result;
use spark_model::layers::qwen3_attention::sparse_seam::{KvDtypeTag, KvLayerView, SeqBatch};
use spark_model::port::retro::build::{self, Job, reference};
use spark_model::port::retro::config::{self, RetroCfg, RetroParams};
use spark_model::port::retro::state::{
    IndexLayout, LayerIndex, PendingBuild, ZoneLayout, ZoneLists,
};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

pub const D: usize = 256;
pub const BS: u32 = 16;
pub const H: u32 = 4;
pub const QH: u32 = 16;

/// SplitMix64, so the example needs no RNG crate.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [-1, 1).
    pub fn uni(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }

    /// A random FP8 E4M3 byte of magnitude below 2^5 (no NaN), random sign.
    fn fp8(&mut self) -> u8 {
        let mag = (self.next() % 0x60) as u8;
        if self.next() & 1 == 1 {
            mag | 0x80
        } else {
            mag
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Gaussian,
    Uniform,
    Fp8,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Gaussian => "bf16 gaussian",
            Kind::Uniform => "bf16 uniform",
            Kind::Fp8 => "fp8 uniform",
        }
    }
}

/// Host keys and values per head (`[len][256]`, dequantized) and the device pools.
pub struct Layer {
    pub len: u32,
    pub keys: Vec<Vec<f32>>,
    pub vals: Vec<Vec<f32>>,
    pub table: Vec<u32>,
    pub kv: KvLayerView,
}

pub fn make_layer(gpu: &dyn GpuBackend, len: u32, kind: Kind, seed: u64) -> Result<Layer> {
    let mut r = Rng(seed);
    let fp8 = kind == Kind::Fp8;
    let es = if fp8 { 1 } else { 2 };
    let (h, n) = (H as usize, len as usize);
    let blocks = len.div_ceil(BS);
    let phys = blocks + 7;
    let table: Vec<u32> = (0..blocks).map(|b| (b * 7 + 3) % phys).collect();
    let stride = BS as usize * h * D * es;
    let mut kb = vec![0u8; phys as usize * stride];
    let mut vb = vec![0u8; phys as usize * stride];
    let (ksc, vsc) = (0.05f32, 0.03f32);
    let centers: Vec<f32> = (0..64 * D).map(|_| 3.0 * r.uni()).collect();
    let mut keys = vec![vec![0.0f32; n * D]; h];
    let mut vals = vec![vec![0.0f32; n * D]; h];
    for t in 0..n {
        let off = table[t / BS as usize] as usize * stride + (t % BS as usize) * h * D * es;
        for hh in 0..h {
            let lbl = (r.next() % 64) as usize;
            for d in 0..D {
                let at = off + (hh * D + d) * es;
                let i = t * D + d;
                if fp8 {
                    let (kbyte, vbyte) = (r.fp8(), r.fp8());
                    kb[at] = kbyte;
                    vb[at] = vbyte;
                    keys[hh][i] = reference::from_fp8(kbyte, ksc);
                    vals[hh][i] = reference::from_fp8(vbyte, vsc);
                } else {
                    let k = match kind {
                        Kind::Gaussian => 1.0 + centers[lbl * D + d] + 0.05 * r.uni(),
                        _ => r.uni(),
                    };
                    let kbits = reference::bf16_bits(k);
                    let vbits = reference::bf16_bits(r.uni());
                    kb[at..at + 2].copy_from_slice(&kbits.to_le_bytes());
                    vb[at..at + 2].copy_from_slice(&vbits.to_le_bytes());
                    keys[hh][i] = reference::from_bf16_bits(kbits);
                    vals[hh][i] = reference::from_bf16_bits(vbits);
                }
            }
        }
    }
    let k_pool = gpu.alloc(kb.len())?;
    let v_pool = gpu.alloc(vb.len())?;
    gpu.copy_h2d(&kb, k_pool)?;
    gpu.copy_h2d(&vb, v_pool)?;
    let kv = KvLayerView {
        k_pool,
        v_pool,
        dtype: if fp8 {
            KvDtypeTag::Fp8
        } else {
            KvDtypeTag::Bf16
        },
        block_size: BS,
        kv_heads: H,
        head_dim: 256,
        kv_scale: if fp8 { ksc } else { 1.0 },
        v_scale: if fp8 { vsc } else { 1.0 },
        block_stride_bytes: stride as u64,
        cache_stride: (BS as usize * h * D) as u64,
    };
    Ok(Layer {
        len,
        keys,
        vals,
        table,
        kv,
    })
}

pub fn d2h(gpu: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; bytes];
    gpu.copy_d2h(p, &mut v)?;
    Ok(v)
}

pub fn i32s(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub fn bf16s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| reference::from_bf16_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

pub fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// BF16 bytes of host values.
pub fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| reference::bf16_bits(x).to_le_bytes())
        .collect()
}

/// Max-abs error of `x` against `y`, relative to `y`'s max-abs value.
pub fn rel_max_abs(x: &[f32], y: &[f32]) -> f32 {
    let err = x
        .iter()
        .zip(y)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    err / y.iter().map(|b| b.abs()).fold(0.0f32, f32::max).max(1e-6)
}

/// The failed-check count.
pub struct Report(pub usize);

impl Report {
    pub fn check(&mut self, ok: bool, what: &str) {
        println!("  {} {what}", if ok { "ok  " } else { "FAIL" });
        self.0 += usize::from(!ok);
    }
}

/// The device state of one layer: config, index, build scratch, zone lists.
pub struct Dev {
    pub cfg: RetroCfg,
    pub idx: LayerIndex,
    pub scratch: DevicePtr,
    pub scratch_bytes: usize,
    pub zl: ZoneLists,
    pub bufs: Vec<DevicePtr>,
}

pub fn dev(gpu: &dyn GpuBackend, l: &Layer) -> Result<Dev> {
    let p = RetroParams::default();
    let cfg = RetroCfg::new(p, H, QH, 256, 0.0625, l.kv.dtype, l.len, BS)?;
    let layout = IndexLayout {
        kv_heads: H,
        head_dim: 256,
        cap_centroids: cfg.cap_centroids,
        cap_tokens: l.len,
    };
    let base = gpu.alloc(layout.total_bytes())?;
    gpu.memset(base, 0, layout.total_bytes())?;
    let scratch_bytes = cfg.build_scratch_bytes(l.len, l.len / 8);
    let scratch = gpu.alloc(scratch_bytes)?;
    let zlay = ZoneLayout {
        kv_heads: H,
        retrieval_cap: cfg.retrieval_cap,
        es_cap: cfg.es_cap,
    };
    let zb = gpu.alloc(zlay.bytes())?;
    let scores = gpu.alloc(H as usize * cfg.cap_centroids as usize * 4)?;
    let ss = gpu.alloc(cfg.select_scratch_bytes())?;
    let zl = ZoneLists::at(zb, zlay, scores, ss, cfg.select_scratch_bytes());
    let idx = LayerIndex::at(base, layout);
    let bufs = vec![base, scratch, zb, scores, ss];
    Ok(Dev {
        cfg,
        idx,
        scratch,
        scratch_bytes,
        zl,
        bufs,
    })
}

/// A one-row batch with a host block table only (the build uploads it into its scratch).
pub fn batch<'a>(lens: &'a [u32], tables: &'a [&'a [u32]]) -> SeqBatch<'a> {
    SeqBatch {
        n: 1,
        block_table: DevicePtr::NULL,
        seq_lens: DevicePtr::NULL,
        max_blocks_per_seq: 0,
        host_seq_lens: lens,
        seq_ids: &[],
        rows_per_seq: 1,
        q_row_stride: 0,
        q_rows: 1,
        first_pos: &[],
        new_tokens: &[],
        host_block_tables: tables,
        block_table_ptrs: DevicePtr::NULL,
    }
}

/// The initial build's job for `len` tokens (sink to len - window), sized as `update.rs` does.
pub fn initial_job(len: u32) -> Result<Job> {
    let p = RetroParams::default();
    let tokens = len - p.window - p.sink;
    let segs = config::n_segments(tokens, p.segment_tokens);
    let nc = config::n_centroids(tokens, p.avg_cluster, segs);
    Job::new(p.sink, tokens, nc, segs, p.kmeans_iters, 0, 0)
}

/// Run `job` through the public entry point (`PendingBuild` on the index) and wait for it.
pub fn run_build(gpu: &dyn GpuBackend, l: &Layer, d: &mut Dev, job: &Job) -> Result<()> {
    d.idx.pending = Some(PendingBuild {
        first_pos: job.first_pos,
        tokens: job.n_tokens,
        segments: job.n_segments,
        centroids: job.n_centroids,
        centroid_offset: job.centroid_offset,
        token_offset: job.token_offset,
        iters: job.iters,
        scratch: d.scratch,
        scratch_bytes: d.scratch_bytes,
    });
    let lens = [l.len];
    let tables = [l.table.as_slice()];
    let cfg = d.cfg.clone();
    let stream = gpu.default_stream();
    build::build_layer_index(gpu, &l.kv, &batch(&lens, &tables), &cfg, &mut d.idx, stream)?;
    d.idx.pending = None;
    gpu.synchronize(stream)
}

/// One head's clusters: centroids, value sums, sizes, member lists.
pub struct Clusters {
    pub cent: Vec<f32>,
    pub vsum: Vec<f32>,
    pub size: Vec<i32>,
    pub lists: Vec<Vec<i32>>,
}

/// Head `h`'s clusters `[c0, c0 + nc)` as the device holds them.
pub fn read_clusters(
    gpu: &dyn GpuBackend,
    idx: &LayerIndex,
    h: usize,
    c0: usize,
    nc: usize,
) -> Result<Clusters> {
    let (cs, os, ts) = idx.layout.head_strides();
    let row = D * 2;
    let cent = bf16s(&d2h(
        gpu,
        idx.centroids.offset((h * cs + c0) * row),
        nc * row,
    )?);
    let vsum = bf16s(&d2h(
        gpu,
        idx.value_sum.offset((h * cs + c0) * row),
        nc * row,
    )?);
    let size = i32s(&d2h(
        gpu,
        idx.cluster_size.offset((h * cs + c0) * 4),
        nc * 4,
    )?);
    let off = i32s(&d2h(
        gpu,
        idx.list_offsets.offset((h * os + c0) * 4),
        (nc + 1) * 4,
    )?);
    let toks = i32s(&d2h(gpu, idx.list_tokens.offset(h * ts * 4), ts * 4)?);
    let lists = (0..nc)
        .map(|c| toks[off[c].max(0) as usize..off[c + 1].max(off[c]).max(0) as usize].to_vec())
        .collect();
    Ok(Clusters {
        cent,
        vsum,
        size,
        lists,
    })
}

/// Every head's clusters `[0, nc)`.
pub fn read_all(gpu: &dyn GpuBackend, idx: &LayerIndex, nc: usize) -> Result<Vec<Clusters>> {
    (0..H as usize)
        .map(|h| read_clusters(gpu, idx, h, 0, nc))
        .collect()
}

/// The job's span of a head's `[len][256]` host array.
pub fn span(v: &[f32], j: &Job) -> Vec<f32> {
    v[j.first_pos as usize * D..j.end_pos() as usize * D].to_vec()
}

/// A mirror cluster's member list.
pub fn members(r: &reference::HeadIndex, c: usize) -> &[i32] {
    let s = r.cluster_start[c] as usize;
    &r.cluster_tokens[s..s + r.cluster_size[c] as usize]
}
