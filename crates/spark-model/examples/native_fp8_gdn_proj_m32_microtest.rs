// SPDX-License-Identifier: AGPL-3.0-only
//! GPU oracle for `w8a16_gemm_pipelined_m32` (G18 lever B), the 32-row M-tile
//! twin of `w8a16_gemm_pipelined`, at the block-scaled FP8 GDN projection
//! shapes the batched verify dispatches at R = 17..=32:
//!
//!   Qwen3.6-35B-A3B-FP8:  in_proj_qkvz [12288 x 2048], out_proj [2048 x 4096]
//!   Qwen3.8-27B (FP8 GDN): in_proj_qkvz [12288 x 5120], out_proj [5120 x 4096]
//!
//! Three claims, each with a known-bad control so a green run cannot be a
//! vacuous one:
//!
//!   1. BITS vs the 128-tile kernel it replaces: the twin's `[M, N]` output is
//!      BYTE-identical to `w8a16_gemm_pipelined` at every M in
//!      {1, 5, 7, 16, 17, 24, 32} — same sub-MMA windows, same fold points
//!      (kernel header) — and neither writes a row past M or a guard byte.
//!   2. STRIDES: the `_strided` launch (A pitch K+8, C pitch N+64) reproduces
//!      the contiguous output row for row and leaves the pitch gaps intact —
//!      the contract the multi-seq attention Q/K/V tier depends on.
//!   3. BUDGET vs the scalar `w8a16_gemv` (what the M=1 decode runs): within
//!      the tensor-core tiers' shared tolerance
//!      (`layers::dense_ffn::m16_tc::oracle`) — the twin inherits the 128
//!      tile's numerics and this pins that they are the accepted ones.
//!
//! And the number the lever exists for: sync'd wall time over 20 reps of
//! each kernel, printed as us/launch and effective weight GB/s (the whole
//! budget at these widths), 128 tile vs 32 tile. The analysis projected
//! in_proj 296 -> ~140 us and out_proj 166 -> ~60 us at M=32 on GB10.
//!
//! Run (GB10):
//!   cargo run --release -p spark-model --features cuda,gpu-examples \
//!     --example native_fp8_gdn_proj_m32_microtest
use anyhow::{Result, ensure};
use half::bf16;
use spark_model::layers::dense_ffn::m16_tc::oracle::{M16_TC_MAX_ULP, compare_m16_tc_block};
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::time::Instant;

const MAX_M: usize = 32;
const GUARD: usize = 64;
const SENTINEL: u8 = 0x5a;
const A_PAD: usize = 8; // elements, keeps the strided A rows 16 B-aligned
const C_PAD: usize = 64; // elements
const REPS: usize = 20;
/// Block-level gate, the `m16` oracle's value (`common/m16_tc_compare.rs`).
const REL_RMS_GATE: f64 = 1e-3;
const MS: [usize; 7] = [1, 5, 7, 16, 17, 24, 32];
/// (name, N, K)
const SHAPES: [(&str, usize, usize); 4] = [
    ("35B in_proj_qkvz", 12288, 2048),
    ("35B out_proj", 2048, 4096),
    ("27B in_proj_qkvz", 12288, 5120),
    ("27B out_proj", 5120, 4096),
];

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn values(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(2)
        .map(|x| f64::from(bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32()))
        .collect()
}

/// A guarded, sentinel-filled output `[rows, pitch]` BF16 and its expected
/// image; `live` extents are byte offsets past the leading guard.
struct Out {
    base: DevicePtr,
    image: Vec<u8>,
}

impl Out {
    fn new(gpu: &dyn GpuBackend, rows: usize, pitch: usize) -> Result<Self> {
        let image = vec![SENTINEL; rows * pitch * 2 + 2 * GUARD];
        Ok(Self {
            base: upload(gpu, &image)?,
            image,
        })
    }
    fn ptr(&self) -> DevicePtr {
        self.base.offset(GUARD)
    }
    fn reset(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.copy_h2d(&self.image, self.base)
    }
    fn read(&self, gpu: &dyn GpuBackend) -> Result<Vec<u8>> {
        let mut v = vec![0_u8; self.image.len()];
        gpu.copy_d2h(self.base, &mut v)?;
        Ok(v)
    }
}

/// Bytes outside `live` must be sentinel; bytes inside must be finite.
fn check_extent(
    observed: &[u8],
    sentinel: &[u8],
    live: &[(usize, usize)],
    who: &str,
) -> Result<()> {
    ensure!(observed.len() == sentinel.len(), "{who}: extent mismatch");
    let mut mask = vec![false; sentinel.len()];
    for &(start, len) in live {
        mask[GUARD + start..GUARD + start + len].fill(true);
    }
    for i in 0..sentinel.len() {
        if !mask[i] {
            ensure!(
                observed[i] == sentinel[i],
                "{who} wrote outside its extent at byte {i}"
            );
        }
    }
    for &(start, len) in live {
        ensure!(
            values(&observed[GUARD + start..GUARD + start + len])
                .iter()
                .all(|x| x.is_finite()),
            "{who}: nonfinite output"
        );
    }
    Ok(())
}

/// Claim 1/2: byte-identical live extents (the extents may sit at different
/// pitches on the two sides — `live_a[i]` corresponds to `live_b[i]`).
fn check_bits(
    a: &[u8],
    live_a: &[(usize, usize)],
    b: &[u8],
    live_b: &[(usize, usize)],
    what: &str,
) -> Result<()> {
    ensure!(live_a.len() == live_b.len(), "{what}: row-count mismatch");
    for (&(sa, la), &(sb, lb)) in live_a.iter().zip(live_b) {
        ensure!(la == lb, "{what}: extent length mismatch");
        let (x, y) = (
            &a[GUARD + sa..GUARD + sa + la],
            &b[GUARD + sb..GUARD + sb + lb],
        );
        if x != y {
            let diff = x
                .chunks_exact(2)
                .zip(y.chunks_exact(2))
                .filter(|(p, q)| p != q)
                .count();
            anyhow::bail!("{what}: {diff} BF16 element(s) differ (row at byte {sa})");
        }
    }
    Ok(())
}

/// Claim 3: the tensor-core budget vs the scalar reference over `[m, n]`.
fn check_budget(
    actual: &[u8],
    reference: &[u8],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(i32, f64)> {
    let a = &actual[GUARD..GUARD + m * n * 2];
    let r = &reference[GUARD..GUARD + m * n * 2];
    let d = compare_m16_tc_block(a, r, n, k);
    if let Some(o) = d.over_budget.first() {
        anyhow::bail!(
            "vs scalar: {} element(s) over the {M16_TC_MAX_ULP}-ULP / floor budget, first at \
             ({}, {}) scalar {:e} tile {:e} {} ULP",
            d.over_budget.len(),
            o.row,
            o.col,
            o.reference,
            o.actual,
            o.ulp
        );
    }
    ensure!(
        d.rel_rms <= REL_RMS_GATE,
        "vs scalar: rel_rms {:.3e} > {REL_RMS_GATE:.0e}",
        d.rel_rms
    );
    Ok((d.max_ulp, d.rel_rms))
}

fn rows(m: usize, n: usize, pitch: usize) -> Vec<(usize, usize)> {
    (0..m).map(|r| (r * pitch * 2, n * 2)).collect()
}

fn time_launch(gpu: &dyn GpuBackend, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    f()?;
    gpu.synchronize(0)?;
    let t = Instant::now();
    for _ in 0..REPS {
        f()?;
    }
    gpu.synchronize(0)?;
    Ok(t.elapsed().as_secs_f64() * 1e6 / REPS as f64)
}

fn main() -> Result<()> {
    let gpu = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let scalar = gpu.kernel("w8a16_gemv", "w8a16_gemv")?;
    let full = gpu.kernel("w8a16_gemm_pipelined", "w8a16_gemm_pipelined")?;
    let m32 = gpu.kernel("w8a16_gemm_pipelined_m32", "w8a16_gemm_pipelined_m32")?;
    let mut rng = Lcg(0x0323_8a16_2026_u64);
    let mut failures = 0usize;
    let mut controls_done = false;

    for (name, n, k) in SHAPES {
        let weight: Vec<u8> = (0..n * k)
            .map(|_| {
                let x = rng.next();
                ((x % 127) as u8) | (((x >> 7) & 1) as u8 * 128)
            })
            .collect();
        let scale: Vec<u8> = (0..(n / 128) * (k / 128))
            .flat_map(|_| (((rng.next() % 16 + 1) as f32) / 1024.0).to_le_bytes())
            .collect();
        let acts: Vec<u16> = (0..MAX_M * k)
            .map(|_| bf16::from_f32(((rng.next() % 2049) as f32 - 1024.0) / 1024.0).to_bits())
            .collect();
        let acts_bytes: Vec<u8> = acts.iter().flat_map(|v| v.to_le_bytes()).collect();
        // Strided A: the same rows at pitch K + A_PAD, pad elements poisoned
        // so a kernel that reads past K shows up in the bits.
        let acts_strided: Vec<u8> = (0..MAX_M)
            .flat_map(|r| {
                let mut row: Vec<u16> = acts[r * k..(r + 1) * k].to_vec();
                row.extend(std::iter::repeat_n(0x7f80_u16, A_PAD)); // +inf
                row
            })
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let w = upload(&gpu, &weight)?;
        let s = upload(&gpu, &scale)?;
        let a = upload(&gpu, &acts_bytes)?;
        let a_s = upload(&gpu, &acts_strided)?;
        let out_full = Out::new(&gpu, MAX_M, n)?;
        let out_m32 = Out::new(&gpu, MAX_M, n)?;
        let out_scalar = Out::new(&gpu, MAX_M, n)?;
        let out_strided = Out::new(&gpu, MAX_M, n + C_PAD)?;
        let weight_bytes = (n * k) as f64;

        for m in MS {
            for o in [&out_full, &out_m32, &out_scalar, &out_strided] {
                o.reset(&gpu)?;
            }
            let (mu, nu, ku) = (m as u32, n as u32, k as u32);
            ops::w8a16_gemm_pipelined(&gpu, full, a, w, s, out_full.ptr(), mu, nu, ku, 0)?;
            ops::w8a16_gemm_pipelined_m32(&gpu, m32, a, w, s, out_m32.ptr(), mu, nu, ku, 0)?;
            for r in 0..m {
                ops::w8a16_gemv(
                    &gpu,
                    scalar,
                    a.offset(r * k * 2),
                    w,
                    s,
                    out_scalar.ptr().offset(r * n * 2),
                    nu,
                    ku,
                    0,
                )?;
            }
            ops::w8a16_gemm_pipelined_m32_strided(
                &gpu,
                m32,
                a_s,
                w,
                s,
                out_strided.ptr(),
                mu,
                nu,
                ku,
                (k + A_PAD) as u32,
                (n + C_PAD) as u32,
                0,
            )?;
            gpu.synchronize(0)?;
            let (full_o, m32_o, sc_o, st_o) = (
                out_full.read(&gpu)?,
                out_m32.read(&gpu)?,
                out_scalar.read(&gpu)?,
                out_strided.read(&gpu)?,
            );
            let live = rows(m, n, n);
            let live_s = rows(m, n, n + C_PAD);

            if !controls_done {
                controls(&m32_o, &out_m32.image, &live, &sc_o, m, n, k)?;
                controls_done = true;
            }

            let verdict = (|| -> Result<(i32, f64)> {
                check_extent(&full_o, &out_full.image, &live, "128-tile")?;
                check_extent(&m32_o, &out_m32.image, &live, "32-tile")?;
                check_extent(&st_o, &out_strided.image, &live_s, "32-tile strided")?;
                check_bits(&m32_o, &live, &full_o, &live, "32-tile vs 128-tile")?;
                check_bits(
                    &st_o,
                    &live_s,
                    &m32_o,
                    &live,
                    "strided vs contiguous 32-tile",
                )?;
                check_budget(&m32_o, &sc_o, m, n, k)
            })();
            match verdict {
                Ok((max_ulp, rel_rms)) => println!(
                    "PASS {name} M={m} N={n} K={k}: bits==128-tile, strided==contiguous, \
                     vs scalar max_ulp={max_ulp} rel_rms={rel_rms:.3e}"
                ),
                Err(e) => {
                    println!("FAIL {name} M={m} N={n} K={k}: {e}");
                    failures += 1;
                }
            }
        }

        // The lever's number: both tiles at M=32 (and M=16 for the C=8 rung
        // question the analysis left open).
        for m in [16usize, 32] {
            let (mu, nu, ku) = (m as u32, n as u32, k as u32);
            let t_full = time_launch(&gpu, || {
                ops::w8a16_gemm_pipelined(&gpu, full, a, w, s, out_full.ptr(), mu, nu, ku, 0)
            })?;
            let t_m32 = time_launch(&gpu, || {
                ops::w8a16_gemm_pipelined_m32(&gpu, m32, a, w, s, out_m32.ptr(), mu, nu, ku, 0)
            })?;
            println!(
                "TIME {name} M={m}: 128-tile {t_full:.1} us ({:.0} GB/s)  32-tile {t_m32:.1} us \
                 ({:.0} GB/s)  speedup {:.2}x",
                weight_bytes / t_full / 1e3,
                weight_bytes / t_m32 / 1e3,
                t_full / t_m32
            );
        }
    }

    ensure!(failures == 0, "{failures} case(s) failed");
    println!(
        "ALL PASS: w8a16_gemm_pipelined_m32 byte-identical to w8a16_gemm_pipelined at M in \
         {MS:?} on {} shapes, strided == contiguous, within the tensor-core budget vs scalar",
        SHAPES.len()
    );
    Ok(())
}

/// Known-bad controls, run once against the real checks: a one-bit output
/// flip must fail the byte comparison, a row past M must fail the extent
/// check, and an exponent-bit flip (~2x, hundreds of ULP) must fail the
/// scalar budget. A control that the oracle admits is a broken oracle.
fn controls(
    m32_o: &[u8],
    image: &[u8],
    live: &[(usize, usize)],
    sc_o: &[u8],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let mut flipped = m32_o.to_vec();
    flipped[GUARD] ^= 1;
    let e = check_bits(&flipped, live, m32_o, live, "control").expect_err("bit flip admitted");
    println!("KNOWN_BAD output-bit: refused: {e}");

    let mut past_m = m32_o.to_vec();
    if m < MAX_M {
        past_m[GUARD + m * n * 2] ^= 1;
        let e = check_extent(&past_m, image, live, "control").expect_err("row past M admitted");
        println!("KNOWN_BAD row-past-M: refused: {e}");
    }
    let mut guard = m32_o.to_vec();
    guard[0] ^= 1;
    let e = check_extent(&guard, image, live, "control").expect_err("guard write admitted");
    println!("KNOWN_BAD guard: refused: {e}");

    // Largest |value| of row 0, so the exponent flip cannot land near zero.
    let row0 = values(&sc_o[GUARD..GUARD + n * 2]);
    let idx = (0..n)
        .max_by(|&i, &j| row0[i].abs().total_cmp(&row0[j].abs()))
        .unwrap_or(0);
    let mut exp = m32_o.to_vec();
    exp[GUARD + idx * 2 + 1] ^= 0x40;
    let e = check_budget(&exp, sc_o, m, n, k).expect_err("exponent flip admitted");
    println!("KNOWN_BAD exponent-bit: refused: {e}");
    Ok(())
}
