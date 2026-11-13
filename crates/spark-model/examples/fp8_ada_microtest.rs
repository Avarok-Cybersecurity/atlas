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
//! TIMINGS (L07 retrofit, master plan queue row Q1): every row is DRAM-true.
//! The weights rotate through byte-identical copies totaling more than twice
//! the L2 (the old rows re-read one copy 20 times, which is how the 4B down
//! projection showed 927 GB/s on a 288 GB/s card), the two kernels are timed
//! interleaved in rotated order (five rounds, 3 warm-ups, 20 calls per sample,
//! CUDA events), and each GEMV row also prints the HOT one-copy figure beside
//! the cold one so the cache artifact is visible. GEMV rows print GB/s of FP8
//! weight bytes against the measured ceiling of `hbm_bandwidth_probe`; GEMM rows
//! print cold TFLOP/s for both kernels. Method: TensorFold's distinct weight
//! copies, Cinference's and TensorFold's interleaved rounds (see
//! `common/cold_l2.rs`).
//!
//! Run on the Ada card, in a build of the ada target:
//!     AVAROK_TARGET_HW=ada cargo run --release -p spark-model \
//!       --example fp8_ada_microtest --features cuda,gpu-examples
use anyhow::{Context, Result, ensure};
use spark_model::layers::ops;
use spark_model::weight_map::fp8_blockscale_ref::{
    BlockScaledFp8, gemm_w8a8_ref, gemm_w8a8_term_mag, gemv_w8a16_exact, per_token_group_quant,
    quantize_blockscaled,
};
use spark_model::weight_map::{DenseWeight, quantize_to_fp8_blockscaled};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

#[path = "common/bench_args.rs"]
mod bench_args;
#[path = "common/cold_l2.rs"]
mod cold_l2;
#[path = "common/profiler.rs"]
mod profiler;
#[path = "common/fp8_ada_support.rs"]
mod support;

use cold_l2::BenchCfg;
use support::{
    ACC_TOL, Lcg, Out, acc_err, bf16_bytes, clone_buf, download, host_bits, make_rings, pair_us,
    unequal, upload, verdict,
};

/// `(model, hidden, intermediate)` of the dense FFNs this path is for.
const MODELS: &[(&str, u32, u32)] = &[("holo-0.8b", 1024, 3584), ("holo-4b", 2560, 9216)];
/// Prefill row counts for the GEMM: a short prompt to a full chunk.
const GEMM_M: &[u32] = &[1, 16, 17, 63, 65, 128, 512, 2048];
/// GEMM rows up to this M are also checked against the CPU reference.
const HOST_CHECK_MAX_M: u32 = 17;

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

/// Bytes of the FP8 weight and of its 128 x 128 FP32 scale grid.
fn sizes(n: u32, k: u32) -> (usize, usize) {
    let (n, k) = (n as usize, k as usize);
    (n * k, n.div_ceil(128) * k.div_ceil(128) * 4)
}

/// `gb10 .. us .. GB/s | ada .. us .. GB/s` for FP8 weight `bytes` per call.
fn pair_str(us: [f64; 2], bytes: f64) -> String {
    let g = |u: f64| bytes / (u * 1e-6) / 1e9;
    format!(
        "gb10 {:.1} us {:.0} GB/s | ada {:.1} us {:.0} GB/s",
        us[0],
        g(us[0]),
        us[1],
        g(us[1])
    )
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
    (hidden, inter): (u32, u32),
    (l2, cfg): (usize, &BenchCfg),
) -> Result<usize> {
    let mut fails = 0;
    let gate = quantize(gpu, k, rng, inter, hidden, &mut fails)?;
    let up = quantize(gpu, k, rng, inter, hidden, &mut fails)?;
    let down = quantize(gpu, k, rng, hidden, inter, &mut fails)?;
    let a = rng.bf16s(hidden as usize, 1.0);
    let a_dev = upload(gpu, &bf16_bytes(&a))?;

    // gate+up dual: both kernels, both against the exact host model.
    let (wb, sb) = sizes(inter, hidden);
    let outs: Vec<Out> = (0..4)
        .map(|_| Out::new(gpu, inter as usize))
        .collect::<Result<_>>()?;
    let rings = make_rings(l2, 2 * (wb + sb), |c| {
        if c == 0 {
            return Ok([gate.weight, gate.scale, up.weight, up.scale]);
        }
        Ok([
            clone_buf(gpu, gate.weight, wb)?,
            clone_buf(gpu, gate.scale, sb)?,
            clone_buf(gpu, up.weight, wb)?,
            clone_buf(gpu, up.scale, sb)?,
        ])
    })?;
    let dual = |i: usize, c: [DevicePtr; 4]| {
        ops::w8a16_gemv_dual(
            gpu,
            k.dual[i],
            a_dev,
            c[0],
            c[1],
            outs[2 * i].ptr(),
            c[2],
            c[3],
            outs[2 * i + 1].ptr(),
            inter,
            hidden,
            0,
        )
    };
    let cold = pair_us(gpu, cfg, &rings.cold, &dual)?;
    let hot = pair_us(gpu, cfg, &rings.hot, &dual)?;
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
    let bytes = 2.0 * wb as f64;
    println!(
        "  [{}] gate+up dual N=2x{inter} K={hidden}: unequal={bad}  cold x{} copies: {}\n        HOT (one copy, L2-resident, not a bandwidth number): {}",
        verdict(bad),
        rings.plan.copies,
        pair_str(cold, bytes),
        pair_str(hot, bytes)
    );

    // down, plain and with the SwiGLU input computed inline.
    let g = rng.bf16s(inter as usize, 1.0);
    let u = rng.bf16s(inter as usize, 1.0);
    let (g_dev, u_dev) = (upload(gpu, &bf16_bytes(&g))?, upload(gpu, &bf16_bytes(&u))?);
    let outs: Vec<Out> = (0..4)
        .map(|_| Out::new(gpu, hidden as usize))
        .collect::<Result<_>>()?;
    let (wb, sb) = sizes(hidden, inter);
    let rings = make_rings(l2, wb + sb, |c| {
        if c == 0 {
            return Ok((down.weight, down.scale));
        }
        Ok((
            clone_buf(gpu, down.weight, wb)?,
            clone_buf(gpu, down.scale, sb)?,
        ))
    })?;
    let plain = |i: usize, (w, s): (DevicePtr, DevicePtr)| {
        ops::w8a16_gemv(gpu, k.gemv[i], g_dev, w, s, outs[i].ptr(), hidden, inter, 0)
    };
    let silu = |i: usize, (w, s): (DevicePtr, DevicePtr)| {
        ops::w8a16_gemv_silu_input(
            gpu,
            k.silu[i],
            g_dev,
            u_dev,
            w,
            s,
            outs[2 + i].ptr(),
            hidden,
            inter,
            0,
        )
    };
    let (plain_cold, plain_hot) = (
        pair_us(gpu, cfg, &rings.cold, &plain)?,
        pair_us(gpu, cfg, &rings.hot, &plain)?,
    );
    let (silu_cold, silu_hot) = (
        pair_us(gpu, cfg, &rings.cold, &silu)?,
        pair_us(gpu, cfg, &rings.hot, &silu)?,
    );
    let want = host_bits(&gemv_w8a16_exact(&g, &down.host));
    let (pb, pa) = (outs[0].read(gpu)?, outs[1].read(gpu)?);
    let bad_plain = unequal(&pa, &pb) + unequal(&pa, &want);
    let bad_silu = unequal(&outs[3].read(gpu)?, &outs[2].read(gpu)?);
    fails += usize::from(bad_plain != 0) + usize::from(bad_silu != 0);
    let bytes = wb as f64;
    println!(
        "  [{}] down N={hidden} K={inter}: unequal={bad_plain}  cold x{} copies: {}\n        HOT (L2-resident): {}",
        verdict(bad_plain),
        rings.plan.copies,
        pair_str(plain_cold, bytes),
        pair_str(plain_hot, bytes)
    );
    println!(
        "  [{}] down+SwiGLU N={hidden} K={inter}: unequal(ada vs gb10)={bad_silu}  cold gb10 {:.1} us | ada {:.1} us   HOT gb10 {:.1} us | ada {:.1} us",
        verdict(bad_silu),
        silu_cold[0],
        silu_cold[1],
        silu_hot[0],
        silu_hot[1]
    );
    Ok(fails)
}

fn gemm_rows(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    rng: &mut Lcg,
    (n, kk): (u32, u32),
    (l2, cfg): (usize, &BenchCfg),
) -> Result<usize> {
    let mut fails = 0;
    let w = quantize(gpu, k, rng, n, kk, &mut fails)?;
    let (wb, sb) = sizes(n, kk);
    let rings = make_rings(l2, wb + sb, |c| {
        if c == 0 {
            return Ok((w.weight, w.scale));
        }
        Ok((clone_buf(gpu, w.weight, wb)?, clone_buf(gpu, w.scale, sb)?))
    })?;
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
        let launch = |i: usize, (wp, sp): (DevicePtr, DevicePtr)| {
            ops::fp8_gemm_t_blockscaled(
                gpu,
                k.gemm[i],
                a_fp8,
                a_scale,
                wp,
                sp,
                outs[i].ptr(),
                m,
                n,
                kk,
                0,
            )
        };
        let us = pair_us(gpu, cfg, &rings.cold, &launch)?;
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
            "  [{}] W8A8 M={m:<5} N={n} K={kk}: unequal={vs_gb10}{host}  cold gb10 {:.1} us {:.1} TFLOP/s | ada {:.1} us {:.1} TFLOP/s",
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
    let info = profiler::device_info();
    let l2 = if info.l2_bytes > 0 {
        info.l2_bytes
    } else {
        cold_l2::FALLBACK_L2_BYTES
    };
    let cfg = bench_args::bench_cfg(0)?;
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
    println!(
        "L2 {} MiB: rows are cold (rotated copies > 2x L2); HOT is the one-copy contrast",
        l2 >> 20
    );
    let mut rng = Lcg(0x5e89_2026_f8f8_0001);
    let mut fails = 0;
    for &(model, hidden, inter) in MODELS {
        println!("{model} (hidden {hidden}, intermediate {inter}):");
        fails += gemv_rows(&gpu, &k, &mut rng, (hidden, inter), (l2, &cfg))?;
        fails += gemm_rows(&gpu, &k, &mut rng, (inter, hidden), (l2, &cfg))?;
        fails += gemm_rows(&gpu, &k, &mut rng, (hidden, inter), (l2, &cfg))?;
    }
    ensure!(fails == 0, "{fails} row(s) failed");
    println!("all rows PASS");
    Ok(())
}
