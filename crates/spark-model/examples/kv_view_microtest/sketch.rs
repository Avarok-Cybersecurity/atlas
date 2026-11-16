// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: kv_view_microtest gates S1 to S4: keys written through the real fused
//! K write kernels, sketched on the card, compared with the f64 pre-RoPE page means; the
//! tail path against the job path bit for bit; the query unrope against its input.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::kv_cache::sketch::{
    DEFAULT_MAX_JOBS, KERNEL_MODULE, SketchBuffers, SketchConfig, SketchJob, SketchLayer,
    TailBatch, UNROPE_KERNEL,
};
use spark_runtime::kv_select::oracle::{mean_rows, rms_norm, rope_forward};

use crate::{Gates, Rng, as_u16, bf16_bits, bf16_f64, download, rel_l2, u16_bytes, upload};

const KVH: usize = 4;
const HD: usize = 256;
const ROT: usize = 64;
const BS: usize = 16;
const THETA: f32 = 1.0e7;
const LAYERS: usize = 2;
const LAYER: usize = 1;
const NUM_BLOCKS: usize = 64;
const EPS: f32 = 1e-6;
const FP8_SCALE: f32 = 1.0 / 64.0;

/// (logical page, physical page, stored tokens): positions up to 199,992, a partial tail.
const PAGES: [(u32, u32, u32); 4] = [(0, 5, 16), (1, 40, 16), (777, 17, 16), (12_499, 63, 9)];

struct Written {
    pool: DevicePtr,
    stride: u64,
    /// `[page * KVH + head]` f64 means of the normed pre-RoPE keys.
    want: Vec<Vec<f64>>,
}

fn write_keys(g: &dyn GpuBackend, rng: &mut Rng, dtype: KvCacheDtype) -> Result<Written> {
    let n_tokens: usize = PAGES.iter().map(|p| p.2 as usize).sum();
    let bias: Vec<f64> = (0..KVH * HD).map(|_| rng.normal()).collect();
    let k_in: Vec<u16> = (0..n_tokens * KVH * HD)
        .map(|i| bf16_bits(bias[i % (KVH * HD)] + 0.5 * rng.normal()))
        .collect();
    let mut positions = Vec::with_capacity(n_tokens);
    let mut slots = Vec::with_capacity(n_tokens);
    for &(lp, phys, fill) in &PAGES {
        for o in 0..fill {
            positions.push(lp * BS as u32 + o);
            slots.push(i64::from(phys) * BS as i64 + i64::from(o));
        }
    }
    let elem = if dtype == KvCacheDtype::Fp8 { 1 } else { 2 };
    let stride = (BS * KVH * HD * elem) as u64;
    let pool = g.alloc(NUM_BLOCKS * stride as usize)?;
    g.memset(pool, 0, NUM_BLOCKS * stride as usize)?;
    let d_k = upload(g, &u16_bytes(&k_in))?;
    let d_w = upload(g, &u16_bytes(&[0u16; HD]))?;
    let d_pos = upload(
        g,
        &positions
            .iter()
            .flat_map(|p: &u32| p.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    let d_slot = upload(
        g,
        &slots
            .iter()
            .flat_map(|s: &i64| s.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    let name = if dtype == KvCacheDtype::Fp8 {
        "fused_k_norm_rope_cache_write_fp8"
    } else {
        "fused_k_norm_rope_cache_write_bf16"
    };
    let k = g.kernel("fused_k_norm_rope_cache", name)?;
    let launch = KernelLaunch::new(g, k)
        .grid([n_tokens as u32, KVH as u32, 1])
        .block([HD as u32, 1, 1])
        .arg_ptr(d_k)
        .arg_ptr(d_w)
        .arg_ptr(d_pos)
        .arg_ptr(pool)
        .arg_ptr(d_slot)
        .arg_u32(KVH as u32)
        .arg_u32(HD as u32)
        .arg_u32(ROT as u32)
        .arg_u32(BS as u32)
        .arg_f32(EPS)
        .arg_f32(THETA);
    let launch = if dtype == KvCacheDtype::Fp8 {
        launch.arg_f32(1.0 / FP8_SCALE)
    } else {
        launch
    };
    launch.launch(g.default_stream())?;
    let mut want = Vec::new();
    let mut t0 = 0usize;
    for &(_, _, fill) in &PAGES {
        for h in 0..KVH {
            let mut rows = Vec::with_capacity(fill as usize * HD);
            for t in t0..t0 + fill as usize {
                let at = (t * KVH + h) * HD;
                let x: Vec<f64> = k_in[at..at + HD].iter().map(|&b| bf16_f64(b)).collect();
                rows.extend(rms_norm(&x, f64::from(EPS)));
            }
            want.push(mean_rows(&rows, fill as usize, HD));
        }
        t0 += fill as usize;
    }
    g.synchronize(g.default_stream())?;
    for p in [d_k, d_w, d_pos, d_slot] {
        g.free(p)?;
    }
    Ok(Written { pool, stride, want })
}

fn sketch_cfg() -> SketchConfig {
    SketchConfig {
        num_blocks: NUM_BLOCKS,
        attn_layers: LAYERS,
        kv_heads: KVH,
        head_dim: HD,
        block_size: BS,
        rotary_dim: ROT,
        rope_theta: THETA,
        max_jobs: DEFAULT_MAX_JOBS,
    }
}

/// `(means of one physical page at LAYER for every head, its count)`.
fn read_page(g: &dyn GpuBackend, sk: &SketchBuffers, phys: u32) -> Result<(Vec<u16>, u8)> {
    let c = sk.config();
    let per = c.attn_layers * KVH * HD * 2;
    let means = as_u16(&download(
        g,
        sk.means_ptr()
            .offset(phys as usize * per + LAYER * KVH * HD * 2),
        KVH * HD * 2,
    )?);
    let count = download(g, sk.counts_ptr().offset(phys as usize), 1)?[0];
    Ok((means, count))
}

fn sketch_gate(
    g: &dyn GpuBackend,
    rng: &mut Rng,
    dtype: KvCacheDtype,
    gates: &mut Gates,
) -> Result<()> {
    let w = write_keys(g, rng, dtype)?;
    let sk = SketchBuffers::new(g, sketch_cfg())?;
    let layer = SketchLayer {
        attn_layer: LAYER,
        k_pool: w.pool,
        dtype,
        block_stride_bytes: w.stride,
        kv_scale: if dtype == KvCacheDtype::Fp8 {
            FP8_SCALE
        } else {
            1.0
        },
    };
    let jobs: Vec<SketchJob> = PAGES
        .iter()
        .map(|&(lp, phys, fill)| SketchJob {
            block: phys as i32,
            stream: 0,
            logical_page: lp,
            fill,
        })
        .collect();
    sk.update_jobs(g, &layer, &jobs, DevicePtr::NULL, 0, g.default_stream())?;
    let (mut worst, mut counts_ok) = (0.0f64, true);
    for (p, &(_, phys, fill)) in PAGES.iter().enumerate() {
        let (means, count) = read_page(g, &sk, phys)?;
        counts_ok &= u32::from(count) == fill;
        for h in 0..KVH {
            let got: Vec<f64> = means[h * HD..(h + 1) * HD]
                .iter()
                .map(|&b| bf16_f64(b))
                .collect();
            worst = worst.max(rel_l2(&got, &w.want[p * KVH + h]));
        }
    }
    let (name, bound) = if dtype == KvCacheDtype::Fp8 {
        ("S2 fp8 sketch", 0.1)
    } else {
        ("S1 bf16 sketch", 1e-2)
    };
    gates.check(
        name,
        worst <= bound && counts_ok,
        &format!("worst relative L2 {worst:.3e} (bound {bound:e}), counts {counts_ok}"),
    );
    if dtype == KvCacheDtype::Bf16 {
        tail_gate(g, &sk, &layer, gates)?;
    }
    sk.free(g)?;
    g.free(w.pool)?;
    Ok(())
}

/// S3: page 777 (physical 17) rebuilt through the tail path equals the job path's bits.
fn tail_gate(
    g: &dyn GpuBackend,
    sk: &SketchBuffers,
    layer: &SketchLayer,
    gates: &mut Gates,
) -> Result<()> {
    let (lp, phys) = (777usize, 17u32);
    let (want, _) = read_page(g, sk, phys)?;
    let max_blocks = lp + 1;
    let mut table = vec![0i32; 4 * max_blocks];
    for r in 0..4 {
        table[r * max_blocks + lp] = phys as i32;
    }
    let d_table = upload(
        g,
        &table
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    let d_lens = g.alloc(16)?;
    let stream = g.default_stream();
    sk.invalidate(g, &[phys], stream)?;
    let (_, cleared) = read_page(g, sk, phys)?;
    let base = (lp * BS) as i32;
    for t in 0..BS as i32 {
        // The previous launch reads the lengths: let it finish before they change.
        g.synchronize(stream)?;
        g.copy_h2d(&(base + t + 1).to_le_bytes(), d_lens)?;
        let tb = TailBatch {
            block_table: d_table,
            seq_lens: d_lens,
            max_blocks_per_seq: max_blocks as u32,
            n_rows: 1,
            rows_per_seq: 1,
            n_new: 1,
        };
        sk.update_tail(g, layer, &tb, stream)?;
    }
    let (single, count) = read_page(g, sk, phys)?;
    let ext = base + BS as i32;
    let lens: Vec<u8> = [ext - 3, ext - 2, ext - 1, ext]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    g.synchronize(stream)?;
    g.copy_h2d(&lens, d_lens)?;
    sk.invalidate(g, &[phys], stream)?;
    let tb = TailBatch {
        block_table: d_table,
        seq_lens: d_lens,
        max_blocks_per_seq: max_blocks as u32,
        n_rows: 4,
        rows_per_seq: 4,
        n_new: 4,
    };
    sk.update_tail(g, layer, &tb, stream)?;
    let (verify, _) = read_page(g, sk, phys)?;
    gates.check(
        "S3 tail path == job path",
        cleared == 0 && count == 16 && single == want && verify == want,
        &format!(
            "cleared count {cleared}, token-by-token equal {}, verify-group equal {}",
            single == want,
            verify == want
        ),
    );
    g.free(d_table)?;
    g.free(d_lens)?;
    Ok(())
}

/// S4: RoPE'd query rows at positions 150,000.. rotated back by kv_query_unrope_bf16.
fn unrope_gate(g: &dyn GpuBackend, rng: &mut Rng, gates: &mut Gates) -> Result<()> {
    let (rows, heads, pos0) = (8usize, 16usize, 150_000u32);
    let pre: Vec<f64> = (0..rows * heads * HD).map(|_| rng.normal()).collect();
    let mut post = pre.clone();
    for r in 0..rows {
        for h in 0..heads {
            let at = (r * heads + h) * HD;
            rope_forward(
                &mut post[at..at + HD],
                pos0 + r as u32,
                ROT,
                f64::from(THETA),
            );
        }
    }
    let bits: Vec<u16> = post.iter().map(|&v| bf16_bits(v)).collect();
    let d = upload(g, &u16_bytes(&bits))?;
    let k = g.kernel(KERNEL_MODULE, UNROPE_KERNEL)?;
    KernelLaunch::new(g, k)
        .grid([rows as u32, heads as u32, 1])
        .block([32, 1, 1])
        .arg_ptr(d)
        .arg_u32(rows as u32)
        .arg_u32((heads * HD) as u32)
        .arg_u32(heads as u32)
        .arg_u32(HD as u32)
        .arg_u32(ROT as u32)
        .arg_u32(pos0)
        .arg_f32(THETA)
        .launch(g.default_stream())?;
    let got: Vec<f64> = as_u16(&download(g, d, bits.len() * 2)?)
        .iter()
        .map(|&b| bf16_f64(b))
        .collect();
    let err = rel_l2(&got, &pre);
    gates.check(
        "S4 query unrope",
        err <= 1e-2,
        &format!("relative L2 {err:.3e}"),
    );
    g.free(d)?;
    Ok(())
}

pub fn run(g: &dyn GpuBackend, gates: &mut Gates) -> Result<()> {
    let mut rng = Rng(0x5eed_0031);
    sketch_gate(g, &mut rng, KvCacheDtype::Bf16, gates)?;
    sketch_gate(g, &mut rng, KvCacheDtype::Fp8, gates)?;
    unrope_gate(g, &mut rng, gates)
}
