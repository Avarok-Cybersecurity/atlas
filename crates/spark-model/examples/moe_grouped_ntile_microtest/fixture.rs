// SPDX-License-Identifier: AGPL-3.0-only

//! Fixture for the MoE grouped-GEMM N-tile oracle: deterministic NVFP4 weight
//! and activation generation at the real Lightning-30B shapes, the device
//! buffer set for one (shape, expert-load) case, and the leg launcher.
//!
//! Split out of `main.rs` only to keep both files under the 500-LoC cap.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub const MODULE: &str = "moe_w4a16";
/// Lightning-30B's kernel target — one of only two that resolve
/// `moe_w4a16_grouped_gemm.cu` from `kernels/gb10/common/`.
pub const LIGHTNING_TARGET: &str = "nemotron-3-nano-30b-a3b";
const NUM_EXPERTS: usize = 128; // Lightning-30B A3B
const GROUP_SIZE: usize = 16; // NVFP4 scale group
pub const M_TILE: usize = 64; // kernel-side constant (all three entries)
const BLOCK: u32 = 128; // the ptrtable entries hardcode 128 threads
/// Per-tensor NVFP4 global scale. Deliberately not a power of two so the
/// dequant rounding is exercised rather than exact.
const SCALE2: f32 = 0.013_4;

// ───────────────────────── deterministic PRNG (splitmix64) ─────────────────
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn fill(&mut self, buf: &mut [u8]) {
        for chunk in buf.chunks_mut(8) {
            let w = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&w[..chunk.len()]);
        }
    }
}

fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

// ───────────────────────── device helpers ─────────────────────────
fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}
fn u16s_le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32s_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn i32s_le(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u64s_le(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Per-column "zero in EVERY row" mask — the direct signature of "no CTA ever
/// owned this column" (the caller memsets the output, so it stays at zero).
pub fn zero_col_mask(out: &[u16], rows: usize, n: usize) -> Vec<bool> {
    let mut mask = vec![true; n];
    for r in 0..rows {
        for c in 0..n {
            if out[r * n + c] != 0 {
                mask[c] = false;
            }
        }
    }
    mask
}

pub struct Shape {
    pub name: &'static str,
    pub n: usize,
    pub k: usize,
}

/// Everything one (shape, scenario) case needs on the device. Freed as a unit.
pub struct Case {
    a: DevicePtr,
    b_all: DevicePtr,
    s_all: DevicePtr,
    b_all_t: DevicePtr,
    s_all_t: DevicePtr,
    pub b_tbl: DevicePtr,
    pub s_tbl: DevicePtr,
    pub b_tbl_t: DevicePtr,
    pub s_tbl_t: DevicePtr,
    pub scale2_tbl: DevicePtr,
    offsets: DevicePtr,
    pub sorted_ids: DevicePtr,
    out: DevicePtr,
}

impl Case {
    pub fn free(self, gpu: &dyn GpuBackend) {
        for p in [
            self.a,
            self.b_all,
            self.s_all,
            self.b_all_t,
            self.s_all_t,
            self.b_tbl,
            self.s_tbl,
            self.b_tbl_t,
            self.s_tbl_t,
            self.scale2_tbl,
            self.offsets,
            self.sorted_ids,
            self.out,
        ] {
            let _ = gpu.free(p);
        }
    }
}

pub fn build_case(
    gpu: &dyn GpuBackend,
    shape: &Shape,
    counts: &[usize],
    total: usize,
    rng: &mut Rng,
) -> Result<Case> {
    let (n, k) = (shape.n, shape.k);
    let half_k = k / 2;
    let num_groups = k / GROUP_SIZE;
    let w_bytes = n * half_k; // packed FP4 nibbles, per expert
    let s_bytes = n * num_groups; // FP8 E4M3 scale bytes, per expert

    // A[total, K] BF16
    let a_host: Vec<u16> = (0..total * k)
        .map(|_| f32_to_bf16_bits(rng.unit() * 2.0 - 1.0))
        .collect();
    let a = upload(gpu, &u16s_le(&a_host))?;

    // Stacked per-expert weights. Generated expert-by-expert into small host
    // scratch buffers and uploaded at an offset, so host peak stays ~5 MB even
    // though the device side is ~360 MB per layout.
    let b_all = gpu.alloc(NUM_EXPERTS * w_bytes)?;
    let s_all = gpu.alloc(NUM_EXPERTS * s_bytes)?;
    let b_all_t = gpu.alloc(NUM_EXPERTS * w_bytes)?;
    let s_all_t = gpu.alloc(NUM_EXPERTS * s_bytes)?;

    let mut w = vec![0u8; w_bytes];
    let mut wt = vec![0u8; w_bytes];
    let mut s = vec![0u8; s_bytes];
    let mut st = vec![0u8; s_bytes];
    for e in 0..NUM_EXPERTS {
        rng.fill(&mut w);
        // FP8 E4M3 scales: exponent 4..=10, no NaN/inf, non-negative.
        for b in s.iter_mut() {
            let r = rng.next_u64();
            let exp = 4 + (r % 7) as u8;
            let mant = ((r >> 8) % 8) as u8;
            *b = (exp << 3) | mant;
        }
        // N-major [N, K/2] → transposed [K/2, N]
        for gn in 0..n {
            let row = &w[gn * half_k..(gn + 1) * half_k];
            for (kp, &byte) in row.iter().enumerate() {
                wt[kp * n + gn] = byte;
            }
        }
        // N-major [N, K/GROUP] → transposed [K/GROUP, N]
        for gn in 0..n {
            let row = &s[gn * num_groups..(gn + 1) * num_groups];
            for (g, &byte) in row.iter().enumerate() {
                st[g * n + gn] = byte;
            }
        }
        gpu.copy_h2d(&w, b_all.offset(e * w_bytes))?;
        gpu.copy_h2d(&s, s_all.offset(e * s_bytes))?;
        gpu.copy_h2d(&wt, b_all_t.offset(e * w_bytes))?;
        gpu.copy_h2d(&st, s_all_t.offset(e * s_bytes))?;
    }

    let b_ptrs: Vec<u64> = (0..NUM_EXPERTS)
        .map(|e| b_all.0 + (e * w_bytes) as u64)
        .collect();
    let s_ptrs: Vec<u64> = (0..NUM_EXPERTS)
        .map(|e| s_all.0 + (e * s_bytes) as u64)
        .collect();
    let b_ptrs_t: Vec<u64> = (0..NUM_EXPERTS)
        .map(|e| b_all_t.0 + (e * w_bytes) as u64)
        .collect();
    let s_ptrs_t: Vec<u64> = (0..NUM_EXPERTS)
        .map(|e| s_all_t.0 + (e * s_bytes) as u64)
        .collect();

    let mut offs = Vec::with_capacity(NUM_EXPERTS + 1);
    let mut acc = 0i32;
    offs.push(0);
    for &c in counts {
        acc += c as i32;
        offs.push(acc);
    }

    Ok(Case {
        a,
        b_all,
        s_all,
        b_all_t,
        s_all_t,
        b_tbl: upload(gpu, &u64s_le(&b_ptrs))?,
        s_tbl: upload(gpu, &u64s_le(&s_ptrs))?,
        b_tbl_t: upload(gpu, &u64s_le(&b_ptrs_t))?,
        s_tbl_t: upload(gpu, &u64s_le(&s_ptrs_t))?,
        scale2_tbl: upload(gpu, &f32s_le(&[SCALE2; NUM_EXPERTS]))?,
        offsets: upload(gpu, &i32s_le(&offs))?,
        sorted_ids: upload(gpu, &i32s_le(&(0..total as i32).collect::<Vec<_>>()))?,
        out: gpu.alloc((total * shape.n * 2).max(1))?,
    })
}

/// One leg: `(kernel, optional pointer-table args, grid.x)`. `None` selects the
/// legacy stacked-buffer signature, `Some(..)` the pointer-table signature.
#[derive(Clone, Copy)]
pub struct Leg {
    pub kernel: KernelHandle,
    pub ptrtable: Option<(DevicePtr, DevicePtr, DevicePtr, DevicePtr)>,
    pub grid_x: u32,
}

fn issue(
    gpu: &dyn GpuBackend,
    stream: u64,
    leg: Leg,
    case: &Case,
    shape: &Shape,
    max_m_tiles: u32,
) -> Result<()> {
    let Leg {
        kernel,
        ptrtable,
        grid_x,
    } = leg;
    let mut l = KernelLaunch::new(gpu, kernel)
        .grid([grid_x, max_m_tiles, NUM_EXPERTS as u32])
        .block([BLOCK, 1, 1])
        .arg_ptr(case.a);
    match ptrtable {
        Some((b_tbl, s_tbl, scale2_tbl, sorted)) => {
            l = l
                .arg_ptr(b_tbl)
                .arg_ptr(s_tbl)
                .arg_ptr(scale2_tbl)
                .arg_ptr(case.out)
                .arg_ptr(case.offsets)
                .arg_ptr(sorted);
        }
        None => {
            l = l
                .arg_ptr(case.b_all)
                .arg_ptr(case.s_all)
                .arg_f32(SCALE2)
                .arg_ptr(case.out)
                .arg_ptr(case.offsets);
        }
    }
    l.arg_u32(NUM_EXPERTS as u32)
        .arg_u32(shape.n as u32)
        .arg_u32(shape.k as u32)
        .launch(stream)
}

/// Zero the output (exactly as the production caller memsets it), run one leg,
/// and read back `[total, N]` BF16.
pub fn run_leg(
    gpu: &dyn GpuBackend,
    stream: u64,
    leg: Leg,
    case: &Case,
    shape: &Shape,
    max_m_tiles: u32,
    total: usize,
) -> Result<Vec<u16>> {
    let bytes = total * shape.n * 2;
    gpu.memset_async(case.out, 0, bytes, stream)?;
    issue(gpu, stream, leg, case, shape, max_m_tiles)?;
    gpu.synchronize(stream)?;
    let mut raw = vec![0u8; bytes];
    gpu.copy_d2h(case.out, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Mean wall-clock ms per launch over `ITERS` back-to-back launches (one host
/// sync at the end), after a short warm-up.
pub fn time_leg(
    gpu: &dyn GpuBackend,
    stream: u64,
    leg: Leg,
    case: &Case,
    shape: &Shape,
    max_m_tiles: u32,
) -> Result<f64> {
    const WARMUP: usize = 3;
    const ITERS: usize = 20;
    for _ in 0..WARMUP {
        issue(gpu, stream, leg, case, shape, max_m_tiles)?;
    }
    gpu.synchronize(stream)?;
    let t0 = std::time::Instant::now();
    for _ in 0..ITERS {
        issue(gpu, stream, leg, case, shape, max_m_tiles)?;
    }
    gpu.synchronize(stream)?;
    Ok(t0.elapsed().as_secs_f64() * 1e3 / ITERS as f64)
}

pub fn scenario_counts(kind: &str, rng: &mut Rng) -> Vec<usize> {
    match kind {
        // Rung-8 decode: 8 seqs x top_k 6 = 48 expanded rows over 128 experts,
        // so per-expert M lands in {0,1,2,3} and max_m_tiles == 1.
        "decode" => {
            let mut c = vec![0usize; NUM_EXPERTS];
            for _ in 0..48 {
                c[(rng.next_u64() as usize) % NUM_EXPERTS] += 1;
            }
            c
        }
        // Prefill-ish: a handful of experts spill past M_TILE so the
        // multi-m-tile path (grid.y > 1) is covered too.
        _ => {
            let mut c = vec![0usize; NUM_EXPERTS];
            for _ in 0..200 {
                c[(rng.next_u64() as usize) % NUM_EXPERTS] += 1;
            }
            for e in 0..4 {
                c[e] += 66;
            }
            c
        }
    }
}
