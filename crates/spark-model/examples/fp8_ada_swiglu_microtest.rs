// SPDX-License-Identifier: AGPL-3.0-only
//! GPU oracle + timings for the sm_89 fused SwiGLU + per-token-group FP8
//! quantizer (`kernels/ada/common/swiglu_quant_fp8_ada.cu`) against the pair
//! it replaces in the dense FFN's W8A8 prefill, `moe_silu_mul` then
//! `per_token_group_quant_fp8`, at the Holo 3.1 0.8B and 4B intermediates.
//!
//! WHAT THIS PINS (each row prints PASS/FAIL; any FAIL exits non-zero):
//!  * **fused == pair**: every E4M3 byte and every FP32 scale bit equal, at M
//!    from 1 to a full 3,921-token chunk and K from one 128-wide group to the
//!    4B intermediate, with a zero group (the 1e-12 scale floor), a +inf and a
//!    NaN gate element, and groups whose products sit near the top (about
//!    1e30) and the bottom (about 1e-40) of BF16's range.
//!  * **fused == host**: the bytes and scales equal
//!    `fp8_swiglu_ref::quant_of_product` of the pair's BF16 product read back
//!    from the device (exact; the host SwiGLU itself is reported, not gated,
//!    because `ex2.approx` has no bit-exact host model).
//!  * **the extent**: every output carries 64 sentinel bytes either side.
//!  * **a control** that must differ (the pair on different `up` values), so a
//!    comparator that cannot fail does not read as a pass.
//!
//! TIMINGS: sync'd `Instant` over 20 reps after 3 warm-ups. GB/s counts the
//! compulsory traffic of each side: 9 bytes per element for the pair (read
//! gate and up, write the product, read it, write the byte), 5 for the fused
//! kernel; scales are ignored.
//!
//! Run on the Ada card, in a build of the ada target:
//!     AVAROK_TARGET_HW=ada cargo run --release -p spark-model \
//!       --example fp8_ada_swiglu_microtest --features cuda,gpu-examples
use anyhow::{Context, Result, ensure};
use spark_model::layers::ops;
use spark_model::weight_map::fp8_blockscale_ref::round_bf16;
use spark_model::weight_map::fp8_swiglu_ref::{quant_of_product, swiglu_product};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::time::Instant;

/// `(model, intermediate)`: K of the down projection's input, plus the
/// smallest legal K (one 128-wide group per row) as an edge case.
const MODELS: &[(&str, u32)] = &[("holo-0.8b", 3584), ("holo-4b", 9216), ("one-group", 128)];
/// Rows: decode-adjacent widths, partial tiles, a full prefill chunk.
const ROWS: &[u32] = &[1, 5, 63, 65, 512, 2048, 3921];
const GUARD: usize = 64;
const SENTINEL: u8 = 0x5a;
const REPS: u32 = 20;
const WARMUP: u32 = 3;

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
                let x = (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.7 * sigma;
                round_bf16(x)
            })
            .collect()
    }
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect()
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn time_us(gpu: &dyn GpuBackend, mut run: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        run()?;
    }
    gpu.synchronize(0)?;
    let t0 = Instant::now();
    for _ in 0..REPS {
        run()?;
    }
    gpu.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / f64::from(REPS))
}

/// A device buffer with sentinel guards either side.
struct Guarded {
    base: DevicePtr,
    bytes: usize,
}

impl Guarded {
    fn new(gpu: &dyn GpuBackend, bytes: usize) -> Result<Self> {
        let base = gpu.alloc(bytes + 2 * GUARD)?;
        gpu.copy_h2d(&vec![SENTINEL; bytes + 2 * GUARD], base)?;
        Ok(Self { base, bytes })
    }
    fn ptr(&self) -> DevicePtr {
        self.base.offset(GUARD)
    }
    /// The payload, after checking both guards.
    fn read(&self, gpu: &dyn GpuBackend) -> Result<Vec<u8>> {
        gpu.synchronize(0)?;
        let mut host = vec![0u8; self.bytes + 2 * GUARD];
        gpu.copy_d2h(self.base, &mut host)?;
        let guards = host[..GUARD].iter().chain(&host[GUARD + self.bytes..]);
        ensure!(
            guards.into_iter().all(|b| *b == SENTINEL),
            "a kernel wrote outside its output"
        );
        Ok(host[GUARD..GUARD + self.bytes].to_vec())
    }
    fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.base)
    }
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bf16_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

fn unequal_bytes(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

fn unequal_scales(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

fn verdict(fails: usize) -> &'static str {
    if fails == 0 { "PASS" } else { "FAIL" }
}

struct Kernels {
    silu: KernelHandle,
    quant: ops::Fp8ActQuant,
    fused: KernelHandle,
}

/// The inputs of one row: random gate and up, plus the edge cases.
fn inputs(rng: &mut Lcg, m: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut g = rng.bf16s(m * k, 2.0);
    let mut u = rng.bf16s(m * k, 1.0);
    // Row 0, group 0: all zero, so its scale is the 1e-12 floor.
    g[..128].fill(0.0);
    // Last row: a +inf and a NaN gate element, in two different groups where
    // the row has them, in its one group otherwise (K = 128).
    let last = (m - 1) * k;
    let (inf_at, nan_at) = if k >= 384 { (130, 300) } else { (5, 9) };
    g[last + inf_at] = f32::INFINITY;
    g[last + nan_at] = f32::NAN;
    // Rows 1 and 2, group 0: products near the top of BF16's range (about
    // 1e30) and near its bottom (about 1e-40, BF16 subnormals), so the group
    // scale runs to both extremes.
    for (row, factor) in [(1usize, 1e15f32), (2, 1e-20)] {
        if m > row {
            for i in row * k..row * k + 128 {
                g[i] = round_bf16(g[i] * factor);
                u[i] = round_bf16(u[i] * factor);
            }
        }
    }
    (g, u)
}

/// The pair, as the W8A8 arm launches it: SwiGLU into `prod`, then quantize.
#[allow(clippy::too_many_arguments)]
fn pair(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    g: DevicePtr,
    u: DevicePtr,
    prod: DevicePtr,
    q: DevicePtr,
    s: DevicePtr,
    m: u32,
    kk: u32,
) -> Result<()> {
    ops::silu_mul(gpu, k.silu, g, u, prod, m * kk, 0)?;
    ops::per_token_group_quant_fp8(gpu, k.quant, prod, q, s, m, kk, 0)
}

fn row(gpu: &dyn GpuBackend, k: &Kernels, rng: &mut Lcg, m: u32, kk: u32) -> Result<usize> {
    let (mu, kd) = (m as usize, kk as usize);
    let (gh, uh) = inputs(rng, mu, kd);
    let g = upload(gpu, &bf16_bytes(&gh))?;
    let u = upload(gpu, &bf16_bytes(&uh))?;
    let prod = Guarded::new(gpu, mu * kd * 2)?;
    let (q_pair, s_pair) = (
        Guarded::new(gpu, mu * kd)?,
        Guarded::new(gpu, mu * kd / 32)?,
    );
    let (q_fused, s_fused) = (
        Guarded::new(gpu, mu * kd)?,
        Guarded::new(gpu, mu * kd / 32)?,
    );

    let run_pair = || pair(gpu, k, g, u, prod.ptr(), q_pair.ptr(), s_pair.ptr(), m, kk);
    let run_fused =
        || ops::swiglu_quant_fp8_ada(gpu, k.fused, g, u, q_fused.ptr(), s_fused.ptr(), m, kk, 0);
    run_pair()?;
    run_fused()?;
    let us_pair = time_us(gpu, run_pair)?;
    let us_fused = time_us(gpu, run_fused)?;

    let (qp, sp) = (q_pair.read(gpu)?, f32s(&s_pair.read(gpu)?));
    let (qf, sf) = (q_fused.read(gpu)?, f32s(&s_fused.read(gpu)?));
    let dev_prod = bf16_to_f32s(&prod.read(gpu)?);
    let (bad_b, bad_s) = (unequal_bytes(&qf, &qp), unequal_scales(&sf, &sp));
    let (hq, hs) = quant_of_product(&dev_prod, mu, kd);
    let (host_b, host_s) = (unequal_bytes(&qf, &hq), unequal_scales(&sf, &hs));
    let host_swiglu = swiglu_product(&gh, &uh)
        .iter()
        .zip(&dev_prod)
        .filter(|(a, b)| a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()))
        .count();
    let bad = bad_b + bad_s + host_b + host_s;
    let gbs = |bytes_per: f64, us: f64| bytes_per * mu as f64 * kd as f64 / (us * 1e-6) / 1e9;
    println!(
        "  [{}] M={m:<5} K={kk}: fused vs pair bytes={bad_b} scales={bad_s} | vs host bytes={host_b} \
         scales={host_s} (host SwiGLU off by an ulp on {host_swiglu}, info)  pair {us_pair:.1} us \
         {:.0} GB/s | fused {us_fused:.1} us {:.0} GB/s | {:.2}x",
        verdict(bad),
        gbs(9.0, us_pair),
        gbs(5.0, us_fused),
        us_pair / us_fused
    );
    for b in [prod, q_pair, s_pair, q_fused, s_fused] {
        b.free(gpu)?;
    }
    gpu.free(g)?;
    gpu.free(u)?;
    Ok(usize::from(bad != 0))
}

/// The comparator must be able to fail: the pair on a different `up` has to
/// disagree with the fused kernel on the original one.
fn control(gpu: &dyn GpuBackend, k: &Kernels, rng: &mut Lcg, kk: u32) -> Result<usize> {
    let (m, kd) = (64u32, kk as usize);
    let (gh, uh) = inputs(rng, 64, kd);
    let other = rng.bf16s(64 * kd, 1.0);
    let (g, u, u2) = (
        upload(gpu, &bf16_bytes(&gh))?,
        upload(gpu, &bf16_bytes(&uh))?,
        upload(gpu, &bf16_bytes(&other))?,
    );
    let prod = Guarded::new(gpu, 64 * kd * 2)?;
    let (qp, sp) = (
        Guarded::new(gpu, 64 * kd)?,
        Guarded::new(gpu, 64 * kd / 32)?,
    );
    let (qf, sf) = (
        Guarded::new(gpu, 64 * kd)?,
        Guarded::new(gpu, 64 * kd / 32)?,
    );
    pair(gpu, k, g, u2, prod.ptr(), qp.ptr(), sp.ptr(), m, kk)?;
    ops::swiglu_quant_fp8_ada(gpu, k.fused, g, u, qf.ptr(), sf.ptr(), m, kk, 0)?;
    let differ = unequal_bytes(&qf.read(gpu)?, &qp.read(gpu)?);
    let ok = differ > 0;
    println!(
        "  [{}] control (pair on other up values) K={kk}: {differ} bytes differ, must be > 0",
        if ok { "PASS" } else { "FAIL" }
    );
    for b in [prod, qp, sp, qf, sf] {
        b.free(gpu)?;
    }
    for p in [g, u, u2] {
        gpu.free(p)?;
    }
    Ok(usize::from(!ok))
}

fn main() -> Result<()> {
    let gpu = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let k = Kernels {
        silu: gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
        quant: ops::Fp8ActQuant::shared_only(
            gpu.kernel("per_token_group_quant_fp8", "per_token_group_quant_fp8")?,
        ),
        fused: gpu
            .kernel("swiglu_quant_fp8_ada", "swiglu_quant_fp8_ada")
            .context("swiglu_quant_fp8_ada absent: build with AVAROK_TARGET_HW=ada")?,
    };
    let mut rng = Lcg(0x5e89_2026_5a1f_0002);
    let mut fails = 0;
    for &(model, inter) in MODELS {
        println!("{model} (intermediate {inter}):");
        for &m in ROWS {
            fails += row(&gpu, &k, &mut rng, m, inter)?;
        }
        fails += control(&gpu, &k, &mut rng, inter)?;
    }
    ensure!(fails == 0, "{fails} row(s) failed");
    println!("all rows PASS");
    Ok(())
}
