// SPDX-License-Identifier: AGPL-3.0-only
//! GPU oracle and timings for the 2/3/5/6/8-bit MLX affine path on sm_89 (lane L17c of the port
//! sprint): `kernels/ada/common/affine_n_ada.cu` against the CPU oracle in
//! `spark_model::port::affine::widths::reference`, at the Holo 3.1 4B FFN shapes, the Qwen3.8-27B FFN
//! shapes and an odd shape (n = 129).
//!
//! WHAT THIS PINS (each gate prints PASS/FAIL; any FAIL exits non-zero):
//!  * **The quantizer**: words, scales and biases byte-equal to `reference::quantize_mlx`, with and
//!    without the clip search; a planted NaN is refused with its group index.
//!  * **The packers**: tiles and transposed tables byte-equal to `reference::pack`.
//!  * **The dequant**: `dequant_bf16` byte-equal to the host's `bf16(s * q + b)`.
//!  * **The GEMV** at M = 1, 2, 4, 5, 8, 16: every output within the float64 bound of
//!    `reference::gemv_f64`; row r at M = k bit-equal to row r decoded alone (row invariance, the MTP
//!    verify property); the split-K form bit-equal to the unsplit form; five runs byte-equal.
//!  * **Planted controls**: the same outputs must FAIL the bound against a reference that drops the
//!    last group and against one without the bias term, or the bound is too loose to mean anything.
//!
//! TIMINGS: M = 1 and M = 4 decode over enough weight copies to defeat the 32 MB L2, sync'd
//! `Instant` over 20 reps after 3 warm-ups; GB/s of packed weight bytes against the card's 288 GB/s.
//!
//! Run on the Ada card, in a build of the ada target:
//!     AVAROK_TARGET_HW=ada cargo run --release -p spark-model \
//!       --example affine_n_microtest --features cuda,gpu-examples
use anyhow::{Result, anyhow, bail};
use spark_model::port::affine::widths::{self, AffineNWeight, MlxDevice, layout, reference as r};
use spark_model::port::w4::{W4Linear, W4Scratch};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::time::Instant;

/// `(label, n, k)`: the 4B gate/up and down, the 27B gate/up and down, and an odd n.
const SHAPES: &[(&str, usize, usize)] = &[
    ("4b-gate", 9216, 2560),
    ("4b-down", 2560, 9216),
    ("27b-gate", 17408, 5120),
    ("27b-down", 5120, 17408),
    ("odd-129", 129, 2560),
];
const ROWS: &[usize] = &[1, 2, 4, 5, 8, 16];
const REPEATS: usize = 5;
const REPS: u32 = 20;
const WARMUP: u32 = 3;
/// Weight copies cycle until they span this many bytes (three times the RTX 4060 Ti's L2).
const COLD_BYTES: usize = 96 << 20;
const PEAK_GBS: f64 = 288.0;

struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn bf16s(&mut self, len: usize, sigma: f32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                r::round_bf16(
                    (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.7 * sigma,
                )
            })
            .collect()
    }
}

struct Report {
    fails: usize,
}

impl Report {
    fn gate(&mut self, name: &str, ok: bool, detail: &str) {
        println!("{} {name}: {detail}", if ok { "PASS" } else { "FAIL" });
        self.fails += usize::from(!ok);
    }
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| r::bf16_bits(*x).to_le_bytes())
        .collect()
}

fn le_bytes<T: Copy, const N: usize>(v: &[T], f: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    v.iter().flat_map(|x| f(*x)).collect()
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(4))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn download(gpu: &dyn GpuBackend, p: DevicePtr, len: usize) -> Result<Vec<u8>> {
    gpu.synchronize(0)?;
    let mut host = vec![0u8; len];
    gpu.copy_d2h(p, &mut host)?;
    Ok(host)
}

fn bf16_out(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| r::bf16_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Outputs (bf16 values) outside the float64 bound `(want, tol)` (rows are independent, so the
/// first `got.len()` entries of a 16-row reference serve every M), and the worst error / bound.
fn violations(want: &[f64], tol: &[f64], got: &[f32]) -> (usize, f64) {
    let mut worst = 0f64;
    let bad = got
        .iter()
        .zip(want.iter().zip(tol))
        .filter(|(g, (y, t))| {
            let e = (f64::from(**g) - **y).abs();
            worst = worst.max(e / **t);
            e > **t
        })
        .count();
    (bad, worst)
}

struct Gpu<'a> {
    gpu: &'a dyn GpuBackend,
    scratch: W4Scratch,
    unsplit: W4Scratch,
}

impl Gpu<'_> {
    fn decode(&self, w: &dyn W4Linear, x: DevicePtr, rows: usize, split: bool) -> Result<Vec<f32>> {
        let out = self.gpu.alloc(rows * w.n() * 2)?;
        w.decode(
            self.gpu,
            x,
            rows,
            out,
            if split { &self.scratch } else { &self.unsplit },
            0,
        )?;
        let got = bf16_out(&download(self.gpu, out, rows * w.n() * 2)?);
        self.gpu.free(out)?;
        Ok(got)
    }
}

fn time_us(gpu: &dyn GpuBackend, mut run: impl FnMut(usize) -> Result<()>) -> Result<f64> {
    for i in 0..WARMUP {
        run(i as usize)?;
    }
    gpu.synchronize(0)?;
    let t0 = Instant::now();
    for i in 0..REPS {
        run(i as usize)?;
    }
    gpu.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / f64::from(REPS))
}

/// The quantizer gates at `clip`; returns the device MLX weight and the host oracle.
fn quantizer(
    g: &Gpu,
    rep: &mut Report,
    tag: &str,
    v: &[f32],
    src: DevicePtr,
    n: usize,
    k: usize,
    bits: u8,
    clip: u8,
) -> Result<(MlxDevice, r::MlxWeight)> {
    let want = r::quantize_mlx(v, n, k, bits, clip)
        .map_err(|gi| anyhow!("host quantizer refused group {gi}"))?;
    let m = widths::quantize_mlx_device(g.gpu, src, n, k, bits, clip, 0)?;
    let words = download(g.gpu, m.words, want.words.len() * 4)?;
    let scales = download(g.gpu, m.scales, want.scales.len() * 2)?;
    let biases = download(g.gpu, m.biases, want.biases.len() * 2)?;
    let ok = words == le_bytes(&want.words, u32::to_le_bytes)
        && scales == le_bytes(&want.scales, u16::to_le_bytes)
        && biases == le_bytes(&want.biases, u16::to_le_bytes);
    rep.gate(
        &format!("{tag} quantizer clip={clip}"),
        ok,
        "words, scales, biases byte-equal to the host",
    );
    Ok((m, want))
}

fn check(
    g: &Gpu,
    rep: &mut Report,
    rng: &mut Lcg,
    label: &str,
    n: usize,
    k: usize,
    bits: u8,
) -> Result<()> {
    let tag = format!("{bits}b {label} [{n}x{k}]");
    let v = rng.bf16s(n * k, 0.02);
    let src = upload(g.gpu, &bf16_bytes(&v))?;
    if label.starts_with("odd") || label == "4b-gate" {
        let (m, _) = quantizer(g, rep, &tag, &v, src, n, k, bits, 4)?;
        m.free(g.gpu)?;
    }
    let (m, host) = quantizer(g, rep, &tag, &v, src, n, k, bits, 0)?;
    g.gpu.free(src)?;
    let w = AffineNWeight::from_mlx_device(g.gpu, &m, 0)?;
    let t = r::pack(&host);
    let tiles = download(g.gpu, w.tiles_ptr(), t.words.len() * 4)?;
    let st = download(g.gpu, w.scales_ptr(), t.scales_t.len() * 2)?;
    let bt = download(g.gpu, w.biases_ptr(), t.biases_t.len() * 2)?;
    let ok = tiles == le_bytes(&t.words, u32::to_le_bytes)
        && st == le_bytes(&t.scales_t, u16::to_le_bytes)
        && bt == le_bytes(&t.biases_t, u16::to_le_bytes);
    rep.gate(
        &format!("{tag} pack"),
        ok,
        "tiles and transposed tables byte-equal to the host",
    );

    let deq = g.gpu.alloc(n * k * 2)?;
    w.dequant_bf16(g.gpu, deq, 0)?;
    let want_deq = le_bytes(&r::dequant_mlx(&host), |x| r::bf16_bits(x).to_le_bytes());
    rep.gate(
        &format!("{tag} dequant"),
        download(g.gpu, deq, n * k * 2)? == want_deq,
        "bf16(s * q + b) byte-equal",
    );
    g.gpu.free(deq)?;

    let x = rng.bf16s(16 * k, 1.0);
    let xd = upload(g.gpu, &bf16_bytes(&x))?;
    let alone: Vec<Vec<f32>> = (0..16)
        .map(|row| g.decode(&w, xd.offset(row * k * 2), 1, true))
        .collect::<Result<_>>()?;
    let (want, tol) = r::gemv_f64(&host, &x, 16);
    let (mut gemv_ok, mut inv_ok, mut worst) = (true, true, 0f64);
    for &rows in ROWS {
        let got = g.decode(&w, xd, rows, true)?;
        let (bad, wr) = violations(&want, &tol, &got);
        worst = worst.max(wr);
        gemv_ok &= bad == 0;
        for (row, solo) in alone.iter().enumerate().take(rows) {
            inv_ok &= got[row * n..(row + 1) * n]
                .iter()
                .zip(solo)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        }
    }
    rep.gate(
        &format!("{tag} gemv bound"),
        gemv_ok,
        &format!("M in {ROWS:?}, worst error {worst:.3} of the bound"),
    );
    rep.gate(
        &format!("{tag} row invariance"),
        inv_ok,
        "row r at M = k bit-equal to row r alone",
    );
    let full = g.decode(&w, xd, 16, true)?;
    let same = |o: &[f32]| o.iter().zip(&full).all(|(a, b)| a.to_bits() == b.to_bits());
    let unsplit = g.decode(&w, xd, 16, false)?;
    rep.gate(
        &format!("{tag} split"),
        same(&unsplit),
        &format!("sk = {} split vs unsplit bit-equal", layout::split_k(n, k)),
    );
    let mut rep_ok = true;
    for _ in 0..REPEATS {
        rep_ok &= same(&g.decode(&w, xd, 16, true)?);
    }
    rep.gate(
        &format!("{tag} repeat"),
        rep_ok,
        &format!("{REPEATS} runs byte-equal"),
    );

    if label == "4b-gate" {
        let kg = k / 64;
        let mut dropped = host.clone();
        for c in 0..n {
            dropped.scales[c * kg + kg - 1] = 0;
            dropped.biases[c * kg + kg - 1] = 0;
        }
        let (dw, dt) = r::gemv_f64(&dropped, &x, 16);
        let (bad, _) = violations(&dw, &dt, &full);
        rep.gate(
            &format!("{tag} control: dropped group"),
            bad > n * 8,
            &format!("{bad} of {} outside the bound", n * 16),
        );
        let mut nobias = host.clone();
        nobias.biases.iter_mut().for_each(|b| *b = 0);
        let (nw, nt) = r::gemv_f64(&nobias, &x, 16);
        let (bad, _) = violations(&nw, &nt, &full);
        rep.gate(
            &format!("{tag} control: no bias term"),
            bad > n * 8,
            &format!("{bad} of {} outside the bound", n * 16),
        );
    }

    if !label.starts_with("odd") {
        let bytes = w.device_bytes();
        let copies = COLD_BYTES.div_ceil(bytes).max(2);
        let mut ws = vec![w];
        for _ in 1..copies {
            ws.push(AffineNWeight::from_mlx_device(g.gpu, &m, 0)?);
        }
        let out = g.gpu.alloc(16 * n * 2)?;
        for rows in [1usize, 4] {
            let us = time_us(g.gpu, |i| {
                ws[i % copies].decode(g.gpu, xd, rows, out, &g.scratch, 0)
            })?;
            let gbs = bytes as f64 / us / 1e3;
            println!(
                "info {tag} M={rows}: {us:.1} us, {gbs:.1} GB/s ({:.0}% of {PEAK_GBS} GB/s), {copies} copies",
                100.0 * gbs / PEAK_GBS
            );
        }
        g.gpu.free(out)?;
        for c in &ws {
            c.free_device(g.gpu)?;
        }
    } else {
        w.free_device(g.gpu)?;
    }
    m.free(g.gpu)?;
    g.gpu.free(xd)?;
    Ok(())
}

fn main() -> Result<()> {
    let gpu = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    if !widths::available(&gpu) {
        bail!("affine_n_ada is not built: build with AVAROK_TARGET_HW=ada");
    }
    let max_k = SHAPES.iter().map(|s| s.2).max().unwrap_or(0);
    let max_ws = SHAPES
        .iter()
        .map(|s| layout::split_work_bytes(s.1, s.2))
        .max()
        .unwrap_or(0);
    let tickets = gpu.alloc(spark_model::port::w4::scratch::TICKET_BYTES)?;
    gpu.memset(tickets, 0, spark_model::port::w4::scratch::TICKET_BYTES)?;
    let scratch = W4Scratch {
        ws: gpu.alloc(max_ws.max(4))?,
        ws_bytes: max_ws,
        tickets,
        xq: DevicePtr::NULL,
        xs: gpu.alloc(layout::sums_bytes(16, max_k))?,
    };
    let unsplit = W4Scratch {
        ws: DevicePtr::NULL,
        ws_bytes: 0,
        ..scratch
    };
    let g = Gpu {
        gpu: &gpu,
        scratch,
        unsplit,
    };
    let mut rep = Report { fails: 0 };
    let mut rng = Lcg(0x5e89_2026_a77e_0003);

    let mut v = rng.bf16s(64 * 256, 0.02);
    v[3 * 256 + 2 * 64 + 9] = f32::NAN;
    let src = upload(&gpu, &bf16_bytes(&v))?;
    let refused = widths::quantize_mlx_device(&gpu, src, 64, 256, 3, 0, 0);
    let named = matches!(&refused, Err(e) if e.to_string().contains("row 3 group 2"));
    rep.gate("planted NaN", named, "the quantizer refuses row 3 group 2");
    if let Ok(m) = refused {
        m.free(&gpu)?;
    }
    gpu.free(src)?;

    for &bits in &layout::BITS {
        for &(label, n, k) in SHAPES {
            check(&g, &mut rep, &mut rng, label, n, k, bits)?;
        }
    }
    let tickets_after = download(&gpu, tickets, spark_model::port::w4::scratch::TICKET_BYTES)?;
    rep.gate(
        "tickets",
        tickets_after.iter().all(|b| *b == 0),
        "the split kernel left every ticket zero",
    );
    println!("{} gate(s) failed", rep.fails);
    if rep.fails > 0 {
        bail!("{} affine_n gate(s) failed", rep.fails);
    }
    Ok(())
}
