// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the data side of `sparse_prefill_microtest`: seeded Q/K/V with optional planted
//! attention structure, BF16 and FP8 E4M3 encodings, a paged KV pool with a shuffled block table, the
//! production dense paged prefill kernel on the same cache, mask decoding and the error measures.

use anyhow::Result;
use half::bf16;
use spark_model::layers::ops;
use spark_model::port::sparse_prefill::reference::Problem;
use spark_model::port::sparse_prefill::{BLOCK, Call, Geometry, KvKind};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Tokens per KV block of the test cache.
pub const BS: usize = 16;

#[derive(Clone, Copy)]
pub struct Case {
    pub name: &'static str,
    pub kv: KvKind,
    pub nq: usize,
    pub nkv: usize,
    pub hd: usize,
    pub q_offset: usize,
    pub q_len: usize,
    pub stride: u32,
    pub threshold: f32,
    /// Each query block's queries and one earlier key block share a spike dimension.
    pub planted: bool,
    /// Run the FP64 oracle (off for the timing cases, where it would take many minutes).
    pub oracle: bool,
}

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
    }
}

fn round_bf16(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

/// FP8 E4M3 (finite-only, NaN at 0x7F) decode, as the kernel's `__nv_cvt_fp8_to_halfraw`.
fn e4m3(b: u8) -> f32 {
    let (s, e, m) = ((b >> 7) & 1, (b >> 3) & 0xF, b & 7);
    let v = if e == 0 {
        m as f32 / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7)
    };
    if s == 1 { -v } else { v }
}

/// Nearest E4M3 code (ties to the even code), by binary search over the sorted positive codes.
fn encode_e4m3(table: &[(f32, u8)], x: f32) -> u8 {
    let a = x.abs().min(448.0);
    let i = table.partition_point(|(v, _)| *v < a);
    let pick = if i == 0 {
        table[0]
    } else if i == table.len() {
        table[i - 1]
    } else {
        let (lo, hi) = (table[i - 1], table[i]);
        let (dl, dh) = (a - lo.0, hi.0 - a);
        if dl < dh || (dl == dh && lo.1 % 2 == 0) {
            lo
        } else {
            hi
        }
    };
    if x < 0.0 && pick.1 != 0 {
        pick.1 | 0x80
    } else {
        pick.1
    }
}

pub fn up<T: Copy>(gpu: &dyn GpuBackend, v: &[T]) -> Result<DevicePtr> {
    let bytes = std::mem::size_of_val(v);
    let p = gpu.alloc(bytes.max(1))?;
    // SAFETY: a plain-old-data slice viewed as bytes for the upload.
    let raw = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, bytes) };
    gpu.copy_h2d(raw, p)?;
    Ok(p)
}

pub fn down_u32(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u32>> {
    let mut b = vec![0u8; n * 4];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub fn down_u16(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Host data of one case: the values the reference reads (BF16-exact, or the FP8 code times its scale
/// rounded to BF16 as the kernel dequantizes) and the device images.
pub struct Built {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub dq: DevicePtr,
    pub dk: DevicePtr,
    pub dv: DevicePtr,
    pub dbt: DevicePtr,
    pub kv_len: usize,
    pub k_scale: f32,
    pub v_scale: f32,
}

impl Built {
    pub fn problem(&self, c: &Case) -> Problem<'_> {
        Problem {
            q: &self.q,
            k: &self.k,
            v: &self.v,
            q_len: c.q_len,
            kv_len: self.kv_len,
            q_offset: c.q_offset,
            nq: c.nq,
            nkv: c.nkv,
            hd: c.hd,
            inv_sqrt_d: 1.0 / (c.hd as f64).sqrt(),
        }
    }

    pub fn free(&self, gpu: &dyn GpuBackend) {
        for x in [self.dq, self.dk, self.dv, self.dbt] {
            gpu.free(x).ok();
        }
    }
}

fn plant(c: &Case, q: &mut [f32], k: &mut [f32], kv_len: usize) {
    let (bl, nq, nkv, hd) = (BLOCK as usize, c.nq, c.nkv, c.hd);
    for row in 0..c.q_len {
        let i = (c.q_offset + row) / bl;
        let jp = (i * 5 + 3) % i.saturating_sub(2).max(1);
        for h in 0..nq {
            let g = h / (nq / nkv);
            let dim = (i * 31 + g * 7) % hd;
            q[(row * nq + h) * hd + dim] += 8.0;
            for pos in jp * bl..((jp + 1) * bl).min(kv_len) {
                k[(pos * nkv + g) * hd + dim] = 8.0;
            }
        }
    }
}

pub fn build(gpu: &dyn GpuBackend, c: &Case, seed: u64) -> Result<Built> {
    let kv_len = c.q_offset + c.q_len;
    let row = c.nkv * c.hd;
    let mut rnd = lcg(seed);
    let mut q: Vec<f32> = (0..c.q_len * c.nq * c.hd).map(|_| rnd()).collect();
    let mut k: Vec<f32> = (0..kv_len * row).map(|_| rnd()).collect();
    let mut v: Vec<f32> = (0..kv_len * row).map(|_| rnd()).collect();
    if c.planted {
        plant(c, &mut q, &mut k, kv_len);
    }
    q.iter_mut().for_each(|x| *x = round_bf16(*x));
    let fp8 = c.kv == KvKind::Fp8;
    let (k_scale, v_scale) = if fp8 {
        (8.0 / 448.0, 1.0 / 448.0)
    } else {
        (1.0, 1.0)
    };
    let mut table: Vec<(f32, u8)> = (0u8..0x7F).map(|b| (e4m3(b), b)).collect();
    table.sort_by(|a, b| a.0.total_cmp(&b.0));
    // A shuffled physical layout with spare blocks, as a fragmented pool would have.
    let n_blocks = kv_len.div_ceil(BS);
    let n_phys = n_blocks + 3;
    let mut perm: Vec<u32> = (0..n_phys as u32).collect();
    let mut r2 = lcg(seed ^ 0xB10C);
    for i in (1..n_phys).rev() {
        let j = ((r2() * 0.5 + 0.5) * (i + 1) as f32) as usize % (i + 1);
        perm.swap(i, j);
    }
    let bt: Vec<u32> = perm[..n_blocks].to_vec();
    let place = |pos: usize| (bt[pos / BS] as usize * BS + pos % BS) * row;
    let width = if fp8 { 1 } else { 2 };
    let (mut pk, mut pv) = (
        vec![0u8; n_phys * BS * row * width],
        vec![0u8; n_phys * BS * row * width],
    );
    for pos in 0..kv_len {
        for x in 0..row {
            let (at, i) = ((place(pos) + x) * width, pos * row + x);
            if fp8 {
                let bk = encode_e4m3(&table, k[i] / k_scale);
                let bv = encode_e4m3(&table, v[i] / v_scale);
                (pk[at], pv[at]) = (bk, bv);
                k[i] = round_bf16(e4m3(bk) * k_scale);
                v[i] = round_bf16(e4m3(bv) * v_scale);
            } else {
                let (bk, bv) = (bf16::from_f32(k[i]), bf16::from_f32(v[i]));
                pk[at..at + 2].copy_from_slice(&bk.to_bits().to_le_bytes());
                pv[at..at + 2].copy_from_slice(&bv.to_bits().to_le_bytes());
                (k[i], v[i]) = (bk.to_f32(), bv.to_f32());
            }
        }
    }
    let qb: Vec<u16> = q.iter().map(|x| bf16::from_f32(*x).to_bits()).collect();
    Ok(Built {
        dq: up(gpu, &qb)?,
        dk: up(gpu, &pk)?,
        dv: up(gpu, &pv)?,
        dbt: up(gpu, &bt)?,
        q,
        k,
        v,
        kv_len,
        k_scale,
        v_scale,
    })
}

/// The kernels' plain-value call for a case, writing `out`.
pub fn call(c: &Case, b: &Built, out: DevicePtr) -> Call {
    Call {
        q: b.dq,
        out,
        k_pool: b.dk,
        v_pool: b.dv,
        block_table: b.dbt,
        q_len: c.q_len as u32,
        kv_len: b.kv_len as u32,
        q_offset: c.q_offset as u32,
        nq: c.nq as u32,
        nkv: c.nkv as u32,
        hd: c.hd as u32,
        block_size: BS as u32,
        cache_stride: (BS * c.nkv * c.hd) as u64,
        kv: c.kv,
        k_scale: b.k_scale,
        v_scale: b.v_scale,
        inv_sqrt_d: 1.0 / (c.hd as f32).sqrt(),
        sliding_window: 0,
        graph_capture: false,
        seam_active: false,
        layer: 0,
    }
}

/// (normwise relative error, max pointwise error) of BF16 bits against FP64 values.
pub fn compare(got: &[u16], want: &[f64]) -> (f64, f64) {
    let (mut num, mut den, mut max) = (0.0f64, 0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        let d = bf16::from_bits(*g).to_f32() as f64 - w;
        if !d.is_finite() {
            return (f64::INFINITY, f64::INFINITY);
        }
        num += d * d;
        den += w * w;
        max = max.max(d.abs());
    }
    ((num / den.max(1e-30)).sqrt(), max)
}

/// BF16 bits widened to FP64.
pub fn widen(bits: &[u16]) -> Vec<f64> {
    bits.iter()
        .map(|x| bf16::from_bits(*x).to_f32() as f64)
        .collect()
}

/// The kernel's mask words, as one `Vec<bool>` of candidates per (head, query block).
pub fn decode_masks(words: &[u32], g: &Geometry, p: &Problem) -> Vec<Vec<bool>> {
    let mw = g.mask_words as usize;
    (0..g.items as usize)
        .map(|item| {
            let i = g.q_block_lo as usize + item % g.n_qb as usize;
            (0..p.candidates(i))
                .map(|j| ((words[item * mw + j / 32] >> (j % 32)) & 1) == 1)
                .collect()
        })
        .collect()
}

/// Mask words with every candidate set (the identity masks), in the kernel's layout.
pub fn all_ones(masks: &[Vec<bool>], mask_words: u32) -> Vec<u32> {
    masks
        .iter()
        .flat_map(|m| {
            let mut w = vec![0u32; mask_words as usize];
            (0..m.len()).for_each(|j| w[j / 32] |= 1 << (j % 32));
            w
        })
        .collect()
}

/// The production dense paged prefill kernel on the same cache.
pub fn dense(gpu: &dyn GpuBackend, c: &Call, out: DevicePtr, stream: u64) -> Result<()> {
    if c.kv == KvKind::Fp8 {
        let kern = gpu.kernel("prefill_paged_fp8", "inferspark_prefill_paged_fp8")?;
        ops::prefill_attention_paged_fp8(
            gpu,
            kern,
            c.q,
            c.k_pool,
            c.v_pool,
            out,
            c.block_table,
            c.q_len,
            c.kv_len,
            c.q_offset,
            c.nq,
            c.nkv,
            c.hd,
            c.block_size,
            0,
            c.inv_sqrt_d,
            c.k_scale,
            c.v_scale,
            c.cache_stride,
            stream,
        )
    } else {
        let kern = gpu.kernel("prefill_paged", "inferspark_prefill_paged")?;
        ops::prefill_attention_paged(
            gpu,
            kern,
            c.q,
            c.k_pool,
            c.v_pool,
            out,
            c.block_table,
            c.q_len,
            c.kv_len,
            c.q_offset,
            c.nq,
            c.nkv,
            c.hd,
            c.block_size,
            0,
            c.inv_sqrt_d,
            stream,
        )
    }
}
