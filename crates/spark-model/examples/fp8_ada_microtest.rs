// SPDX-License-Identifier: AGPL-3.0-only
//! GPU oracle + timings for the sm_89 FP8 path: the load-time block quantizer,
//! the register-dequant decode GEMVs (`kernels/ada/common/fp8_gemv_ada.cu`) and
//! the pipelined W8A8 prefill GEMM (`kernels/ada/common/fp8_gemm_ada.cu`), at
//! the dense Holo 3.1 0.8B and 4B FFN shapes.
//!
//! WHAT THIS PINS (each row prints PASS/FAIL; any FAIL exits non-zero):
//!  * **The quantizer** (`quantize_bf16_to_fp8_blockscaled`) against its host
//!    mirror `weight_map::fp8_blockscale_ref::quantize_blockscaled`: every byte
//!    and every scale equal.
//!  * **The GEMVs**: the Ada drop-in against the gb10 kernel it replaces AND
//!    against the exact host model `gemv_w8a16_exact`, `unequal == 0` on the
//!    BF16 outputs. The SwiGLU-input GEMV is compared to the gb10 kernel only
//!    (its `__expf` has no bit-exact host model).
//!  * **The W8A8 GEMM**: the Ada drop-in against `fp8_gemm_t_blockscaled`,
//!    `unequal == 0` at M from 1 to 2048 (partial 64-row tiles included), and
//!    on the small M rows the activation quantizer byte for byte and the GEMM
//!    against `gemm_w8a8_ref` within 2 BF16 ulps plus `ACC_TOL` of each
//!    output's term magnitude.
//!  * **The extent**: every output carries 64 sentinel bytes either side.
//!
//! TIMINGS: sync'd `Instant` over 20 reps after 3 warm-ups, the house pattern.
//! GEMV rows print GB/s of FP8 weight bytes (the decode budget, against the
//! RTX 4060 Ti's 288 GB/s); GEMM rows print TFLOP/s for both kernels.
//!
//! Run on the Ada card, in a build of the ada target:
//!     AVAROK_TARGET_HW=ada cargo run --release -p spark-model \
//!       --example fp8_ada_microtest --features cuda,gpu-examples
use anyhow::{Context, Result, ensure};
use spark_model::layers::ops;
use spark_model::weight_map::fp8_blockscale_ref::{
    self as fref, BlockScaledFp8, gemm_w8a8_ref, gemm_w8a8_term_mag, gemv_w8a16_exact,
    per_token_group_quant, quantize_blockscaled,
};
use spark_model::weight_map::{DenseWeight, quantize_to_fp8_blockscaled};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::time::Instant;

/// `(model, hidden, intermediate)` of the dense FFNs this path is for.
const MODELS: &[(&str, u32, u32)] = &[("holo-0.8b", 1024, 3584), ("holo-4b", 2560, 9216)];
/// Prefill row counts for the GEMM: a short prompt to a full chunk.
const GEMM_M: &[u32] = &[1, 16, 17, 63, 65, 128, 512, 2048];
/// GEMM rows up to this M are also checked against the CPU reference.
const HOST_CHECK_MAX_M: u32 = 17;
/// BF16's machine epsilon, 2^-7.
const BF16_EPS: f32 = 1.0 / 128.0;
/// Accumulation error allowed per unit of term magnitude (see `acc_err`):
/// 2^-13. The FP8 `mma.sync` sums its products below FP32 precision; on an
/// RTX 4060 Ti the worst output at these shapes is 2^-15 of its term
/// magnitude, with the Ada and gb10 kernels bit-identical, so the margin is
/// 4x and the error is the instruction's, not the tiling's.
const ACC_TOL: f32 = 1.0 / 8192.0;
const GUARD: usize = 64;
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
    /// BF16 values, roughly normal with standard deviation `sigma`.
    fn bf16s(&mut self, len: usize, sigma: f32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                let x = (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.7 * sigma;
                fref::round_bf16(x)
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

fn download(gpu: &dyn GpuBackend, ptr: DevicePtr, len: usize) -> Result<Vec<u8>> {
    let mut host = vec![0u8; len];
    gpu.copy_d2h(ptr, &mut host)?;
    Ok(host)
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

/// A BF16 output buffer with sentinel guards either side.
struct Out {
    base: DevicePtr,
    bytes: usize,
}

impl Out {
    fn new(gpu: &dyn GpuBackend, elems: usize) -> Result<Self> {
        let o = Self {
            base: gpu.alloc(elems * 2 + 2 * GUARD)?,
            bytes: elems * 2,
        };
        o.reset(gpu)?;
        Ok(o)
    }
    fn ptr(&self) -> DevicePtr {
        self.base.offset(GUARD)
    }
    fn reset(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.copy_h2d(&vec![0x5a; self.bytes + 2 * GUARD], self.base)
    }
    /// The payload as BF16 bits, after checking both guards.
    fn read(&self, gpu: &dyn GpuBackend) -> Result<Vec<u16>> {
        gpu.synchronize(0)?;
        let host = download(gpu, self.base, self.bytes + 2 * GUARD)?;
        let guards = host[..GUARD].iter().chain(&host[GUARD + self.bytes..]);
        ensure!(
            guards.into_iter().all(|b| *b == 0x5a),
            "a kernel wrote outside its output"
        );
        Ok(host[GUARD..GUARD + self.bytes]
            .chunks_exact(2)
            .map(|p| u16::from_le_bytes([p[0], p[1]]))
            .collect())
    }
}

fn unequal(a: &[u16], b: &[u16]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

fn host_bits(v: &[f32]) -> Vec<u16> {
    v.iter()
        .map(|x| half::bf16::from_f32(*x).to_bits())
        .collect()
}

/// A GEMM output's accumulation error: its distance from the reference after
/// allowing 2 BF16 ulps for the final round, as a fraction of the output's
/// term magnitude (`gemm_w8a8_term_mag`). The error of a long dot product
/// scales with the size of its terms, not of its result, so a bound relative
/// to the result alone fails every output whose terms cancel.
fn acc_err(got: u16, want: f32, mag: f32) -> f32 {
    let g = half::bf16::from_bits(got).to_f32();
    let round = 2.0 * BF16_EPS * want.abs();
    ((g - want).abs() - round).max(0.0) / mag.max(f32::MIN_POSITIVE)
}

fn verdict(fails: usize) -> &'static str {
    if fails == 0 { "PASS" } else { "FAIL" }
}

struct Kernels {
    quant: KernelHandle,
    act_quant: ops::Fp8ActQuant,
    gemv: [KernelHandle; 2],
    dual: [KernelHandle; 2],
    silu: [KernelHandle; 2],
    gemm: [KernelHandle; 2],
}

/// A weight quantized on the device, checked against the host mirror.
struct DevWeight {
    host: BlockScaledFp8,
    weight: DevicePtr,
    scale: DevicePtr,
}

fn quantize(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    rng: &mut Lcg,
    n: u32,
    kk: u32,
    fails: &mut usize,
) -> Result<DevWeight> {
    let (nn, kd) = (n as usize, kk as usize);
    let x = rng.bf16s(nn * kd, 0.02);
    let src = upload(gpu, &bf16_bytes(&x))?;
    let w = quantize_to_fp8_blockscaled(&DenseWeight { weight: src }, nn, kd, gpu, k.quant, 0)?;
    gpu.synchronize(0)?;
    let host = quantize_blockscaled(&x, nn, kd);
    let dev_bytes = download(gpu, w.weight, nn * kd)?;
    let dev_scales: Vec<f32> = download(gpu, w.row_scale, host.scales.len() * 4)?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let bad = dev_bytes
        .iter()
        .zip(&host.bytes)
        .filter(|(a, b)| a != b)
        .count()
        + dev_scales
            .iter()
            .zip(&host.scales)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
    *fails += usize::from(bad != 0);
    println!(
        "  [{}] quantize [{n}x{kk}]: {bad} byte/scale mismatches",
        verdict(bad)
    );
    gpu.free(src)?;
    Ok(DevWeight {
        host,
        weight: w.weight,
        scale: w.row_scale,
    })
}

fn gemv_rows(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    rng: &mut Lcg,
    hidden: u32,
    inter: u32,
) -> Result<usize> {
    let mut fails = 0;
    let gate = quantize(gpu, k, rng, inter, hidden, &mut fails)?;
    let up = quantize(gpu, k, rng, inter, hidden, &mut fails)?;
    let down = quantize(gpu, k, rng, hidden, inter, &mut fails)?;
    let a = rng.bf16s(hidden as usize, 1.0);
    let a_dev = upload(gpu, &bf16_bytes(&a))?;

    // gate+up dual: both kernels, both against the exact host model.
    let outs: Vec<Out> = (0..4)
        .map(|_| Out::new(gpu, inter as usize))
        .collect::<Result<_>>()?;
    let mut us = [0.0; 2];
    for (i, h) in k.dual.iter().enumerate() {
        let (o0, o1) = (outs[2 * i].ptr(), outs[2 * i + 1].ptr());
        let run = || {
            ops::w8a16_gemv_dual(
                gpu,
                *h,
                a_dev,
                gate.weight,
                gate.scale,
                o0,
                up.weight,
                up.scale,
                o1,
                inter,
                hidden,
                0,
            )
        };
        run()?;
        us[i] = time_us(gpu, run)?;
    }
    let want = [
        host_bits(&gemv_w8a16_exact(&a, &gate.host)),
        host_bits(&gemv_w8a16_exact(&a, &up.host)),
    ];
    let mut bad = 0;
    for (j, w) in want.iter().enumerate() {
        let (base, ada) = (outs[j].read(gpu)?, outs[2 + j].read(gpu)?);
        bad += unequal(&ada, &base) + unequal(&ada, w) + unequal(&base, w);
    }
    fails += usize::from(bad != 0);
    let gbs = |us: f64| 2.0 * f64::from(inter) * f64::from(hidden) / (us * 1e-6) / 1e9;
    println!(
        "  [{}] gate+up dual N=2x{inter} K={hidden}: unequal={bad}  gb10 {:.1} us {:.0} GB/s | ada {:.1} us {:.0} GB/s",
        verdict(bad),
        us[0],
        gbs(us[0]),
        us[1],
        gbs(us[1])
    );

    // down, plain and with the SwiGLU input computed inline.
    let g = rng.bf16s(inter as usize, 1.0);
    let u = rng.bf16s(inter as usize, 1.0);
    let (g_dev, u_dev) = (upload(gpu, &bf16_bytes(&g))?, upload(gpu, &bf16_bytes(&u))?);
    let outs: Vec<Out> = (0..4)
        .map(|_| Out::new(gpu, hidden as usize))
        .collect::<Result<_>>()?;
    let (mut us_plain, mut us_silu) = ([0.0; 2], [0.0; 2]);
    for i in 0..2 {
        let (op, os) = (outs[i].ptr(), outs[2 + i].ptr());
        let plain = || {
            ops::w8a16_gemv(
                gpu,
                k.gemv[i],
                g_dev,
                down.weight,
                down.scale,
                op,
                hidden,
                inter,
                0,
            )
        };
        let silu = || {
            ops::w8a16_gemv_silu_input(
                gpu,
                k.silu[i],
                g_dev,
                u_dev,
                down.weight,
                down.scale,
                os,
                hidden,
                inter,
                0,
            )
        };
        plain()?;
        silu()?;
        us_plain[i] = time_us(gpu, plain)?;
        us_silu[i] = time_us(gpu, silu)?;
    }
    let want = host_bits(&gemv_w8a16_exact(&g, &down.host));
    let (pb, pa) = (outs[0].read(gpu)?, outs[1].read(gpu)?);
    let bad_plain = unequal(&pa, &pb) + unequal(&pa, &want);
    let bad_silu = unequal(&outs[3].read(gpu)?, &outs[2].read(gpu)?);
    fails += usize::from(bad_plain != 0) + usize::from(bad_silu != 0);
    let gbs = |us: f64| f64::from(inter) * f64::from(hidden) / (us * 1e-6) / 1e9;
    println!(
        "  [{}] down N={hidden} K={inter}: unequal={bad_plain}  gb10 {:.1} us {:.0} GB/s | ada {:.1} us {:.0} GB/s",
        verdict(bad_plain),
        us_plain[0],
        gbs(us_plain[0]),
        us_plain[1],
        gbs(us_plain[1])
    );
    println!(
        "  [{}] down+SwiGLU N={hidden} K={inter}: unequal(ada vs gb10)={bad_silu}  gb10 {:.1} us | ada {:.1} us",
        verdict(bad_silu),
        us_silu[0],
        us_silu[1]
    );
    Ok(fails)
}

fn gemm_rows(gpu: &dyn GpuBackend, k: &Kernels, rng: &mut Lcg, n: u32, kk: u32) -> Result<usize> {
    let mut fails = 0;
    let w = quantize(gpu, k, rng, n, kk, &mut fails)?;
    for &m in GEMM_M {
        let (mu, kd) = (m as usize, kk as usize);
        let x = rng.bf16s(mu * kd, 1.0);
        let x_dev = upload(gpu, &bf16_bytes(&x))?;
        let a_fp8 = gpu.alloc(mu * kd)?;
        let a_scale = gpu.alloc(mu * (kd / 128) * 4)?;
        ops::per_token_group_quant_fp8(gpu, k.act_quant, x_dev, a_fp8, a_scale, m, kk, 0)?;
        let outs: Vec<Out> = (0..2)
            .map(|_| Out::new(gpu, mu * n as usize))
            .collect::<Result<_>>()?;
        let mut us = [0.0; 2];
        for (i, h) in k.gemm.iter().enumerate() {
            let o = outs[i].ptr();
            let run = || {
                ops::fp8_gemm_t_blockscaled(
                    gpu, *h, a_fp8, a_scale, w.weight, w.scale, o, m, n, kk, 0,
                )
            };
            run()?;
            us[i] = time_us(gpu, run)?;
        }
        let (base, ada) = (outs[0].read(gpu)?, outs[1].read(gpu)?);
        let vs_gb10 = unequal(&ada, &base);
        let mut host = String::new();
        let mut bad = vs_gb10;
        if m <= HOST_CHECK_MAX_M {
            // The activation quantizer and the GEMM against the host, on the
            // small M rows only (the reference is O(M*N*K) on the CPU).
            let (hb, hs) = per_token_group_quant(&x, mu, kd);
            let dev_b = download(gpu, a_fp8, mu * kd)?;
            let act = dev_b.iter().zip(&hb).filter(|(a, b)| a != b).count();
            let want = gemm_w8a8_ref(&hb, &hs, mu, &w.host);
            let mag = gemm_w8a8_term_mag(&hb, &hs, mu, &w.host);
            let errs: Vec<f32> = ada
                .iter()
                .zip(want.iter().zip(&mag))
                .map(|(got, (want, mag))| acc_err(*got, *want, *mag))
                .collect();
            let off_ref = errs.iter().filter(|e| **e > ACC_TOL).count();
            let worst = errs.iter().fold(0f32, |a, e| a.max(*e));
            bad += act + off_ref;
            host = format!(
                " act-quant={act} off-ref={off_ref} worst={:.1}u",
                worst / f32::EPSILON
            );
        }
        fails += usize::from(bad != 0);
        let tf = |us: f64| 2.0 * f64::from(m) * f64::from(n) * f64::from(kk) / (us * 1e-6) / 1e12;
        println!(
            "  [{}] W8A8 M={m:<5} N={n} K={kk}: unequal={vs_gb10}{host}  gb10 {:.1} us {:.1} TFLOP/s | ada {:.1} us {:.1} TFLOP/s",
            verdict(bad),
            us[0],
            tf(us[0]),
            us[1],
            tf(us[1])
        );
        for p in [x_dev, a_fp8, a_scale] {
            gpu.free(p)?;
        }
    }
    Ok(fails)
}

fn main() -> Result<()> {
    let gpu = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let pair = |base: (&str, &str), ada: (&str, &str)| -> Result<[KernelHandle; 2]> {
        Ok([
            gpu.kernel(base.0, base.1)?,
            gpu.kernel(ada.0, ada.1).with_context(|| {
                format!(
                    "{}::{} absent: build with AVAROK_TARGET_HW=ada",
                    ada.0, ada.1
                )
            })?,
        ])
    };
    let k = Kernels {
        quant: gpu.kernel(
            "quantize_bf16_to_fp8_blockscaled",
            "quantize_bf16_to_fp8_blockscaled",
        )?,
        act_quant: ops::Fp8ActQuant::shared_only(
            gpu.kernel("per_token_group_quant_fp8", "per_token_group_quant_fp8")?,
        ),
        gemv: pair(
            ("w8a16_gemv", "w8a16_gemv"),
            ("fp8_gemv_ada", "fp8_gemv_ada"),
        )?,
        dual: pair(
            ("w8a16_gemv_fused", "w8a16_gemv_dual"),
            ("fp8_gemv_ada", "fp8_gemv_ada_dual"),
        )?,
        silu: pair(
            ("w8a16_gemv_fused", "w8a16_gemv_silu_input"),
            ("fp8_gemv_ada", "fp8_gemv_ada_silu_input"),
        )?,
        gemm: pair(
            ("fp8_gemm_t_blockscaled", "fp8_gemm_t_blockscaled"),
            ("fp8_gemm_ada", "fp8_gemm_ada_w8a8"),
        )?,
    };
    let mut rng = Lcg(0x5e89_2026_f8f8_0001);
    let mut fails = 0;
    for &(model, hidden, inter) in MODELS {
        println!("{model} (hidden {hidden}, intermediate {inter}):");
        fails += gemv_rows(&gpu, &k, &mut rng, hidden, inter)?;
        fails += gemm_rows(&gpu, &k, &mut rng, inter, hidden)?;
        fails += gemm_rows(&gpu, &k, &mut rng, hidden, inter)?;
    }
    ensure!(fails == 0, "{fails} row(s) failed");
    println!("all rows PASS");
    Ok(())
}
