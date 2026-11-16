// SPDX-License-Identifier: AGPL-3.0-only
//! GPU oracle + timings for the sm_89 fused residual add + RMSNorm + FP8 group
//! quantizer (`kernels/ada/common/rmsnorm_quant_fp8_ada.cu`, port row
//! `rmsnorm_fp8_quant_fused`) against the pair it replaces at the prefill
//! FFN-norm: `residual_add_rms_norm` then `per_token_group_quant_fp8`.
//!
//! WHAT THIS PINS (PASS/FAIL; any FAIL exits non-zero):
//!  * **fused == pair** on ALL FIVE outputs, every byte: the updated hidden,
//!    the residual copy, the BF16 normed output, the E4M3 codes and the FP32
//!    scales, at hidden 128 (one group), 1024 (0.8B), 2560 (4B), 5120 (27B)
//!    and rows from 1 to a full 3,921-token chunk, with edge rows: a zero row
//!    (the 1e-12 scale floor), a NaN in `src` (a NaN row), and a row near the
//!    top of BF16's range (its FP32 sum of squares overflows);
//!  * **fused == host**: codes and scales equal
//!    `fp8_blockscale_ref::per_token_group_quant` of the pair's BF16 output read
//!    back from the device (exact; the host norm itself is not modelled,
//!    because `rsqrtf` has no bit-exact host twin);
//!  * **the extent**: sentinel guards either side of every output;
//!  * **a control**: the pair with a different norm weight must differ.
//!
//! TIMINGS (informational): warm microseconds of the pair and of the fused
//! kernel, and GB/s over each side's compulsory traffic (pair: read hidden and
//! src, write hidden, residual and output, read output, write codes; fused:
//! the same without the output re-read).
//!
//! Run on the Ada card, in a build of the ada target:
//!     AVAROK_TARGET_HW=ada cargo run --release -p spark-model \
//!       --example rmsnorm_quant_fp8_microtest --features cuda,gpu-examples
use anyhow::{Context, Result, ensure};
use spark_model::layers::ops;
use spark_model::weight_map::DenseWeight;
use spark_model::weight_map::fp8_blockscale_ref::per_token_group_quant;
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

#[allow(dead_code)]
#[path = "common/l15_fp8.rs"]
mod l15;
use l15::*;

const HIDDEN: &[u32] = &[128, 1024, 2560, 5120];
const ROWS: &[u32] = &[1, 5, 63, 512, 3921];
const EPS: f32 = 1e-6;

struct Kernels {
    norm: KernelHandle,
    quant: ops::Fp8ActQuant,
    fused: KernelHandle,
}

/// The fused kernel's launch, as `norm_fused::rmsnorm_quant_fp8` makes it.
#[allow(clippy::too_many_arguments)]
fn fused(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    hidden: DevicePtr,
    src: DevicePtr,
    w: DevicePtr,
    out: DevicePtr,
    res: DevicePtr,
    q: DevicePtr,
    s: DevicePtr,
    m: u32,
    h: u32,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([m, 1, 1])
        .block([h.min(1024), 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(src)
        .arg_ptr(w)
        .arg_ptr(out)
        .arg_ptr(res)
        .arg_ptr(q)
        .arg_ptr(s)
        .arg_u32(h)
        .arg_f32(EPS)
        .launch(0)
}

/// One side's five outputs.
struct Side {
    hidden: Guarded,
    res: Guarded,
    out: Guarded,
    q: Guarded,
    s: Guarded,
}

impl Side {
    fn new(gpu: &dyn GpuBackend, hidden0: &[u8], m: usize, h: usize) -> Result<Self> {
        Ok(Self {
            hidden: Guarded::with(gpu, hidden0)?,
            res: Guarded::new(gpu, m * h * 2)?,
            out: Guarded::new(gpu, m * h * 2)?,
            q: Guarded::new(gpu, m * h)?,
            s: Guarded::new(gpu, m * (h / 128) * 4)?,
        })
    }
    fn reset(&self, gpu: &dyn GpuBackend, hidden0: &[u8]) -> Result<()> {
        gpu.copy_h2d(hidden0, self.hidden.ptr())
    }
    fn read(&self, gpu: &dyn GpuBackend) -> Result<[Vec<u8>; 5]> {
        Ok([
            self.hidden.read(gpu)?,
            self.res.read(gpu)?,
            self.out.read(gpu)?,
            self.q.read(gpu)?,
            self.s.read(gpu)?,
        ])
    }
    fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        for g in [self.hidden, self.res, self.out, self.q, self.s] {
            g.free(gpu)?;
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn pair(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    p: &Side,
    src: DevicePtr,
    w: DevicePtr,
    m: u32,
    h: u32,
) -> Result<()> {
    ops::residual_add_rms_norm(
        gpu,
        k.norm,
        p.hidden.ptr(),
        src,
        &DenseWeight { weight: w },
        p.out.ptr(),
        p.res.ptr(),
        m,
        h,
        EPS,
        0,
    )?;
    ops::per_token_group_quant_fp8(gpu, k.quant, p.out.ptr(), p.q.ptr(), p.s.ptr(), m, h, 0)
}

fn run_fused(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    f: &Side,
    src: DevicePtr,
    w: DevicePtr,
    m: u32,
    h: u32,
) -> Result<()> {
    fused(
        gpu,
        k.fused,
        f.hidden.ptr(),
        src,
        w,
        f.out.ptr(),
        f.res.ptr(),
        f.q.ptr(),
        f.s.ptr(),
        m,
        h,
    )
}

/// hidden0, src and the norm weight for `m` rows, with the edge rows.
fn inputs(rng: &mut Lcg, m: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut hid = rng.bf16s(m * h, 1.0);
    let mut src = rng.bf16s(m * h, 0.5);
    let weight = rng.bf16s(h, 0.1); // offset-from-one weights near 0
    // Row 0: hidden + src == 0 everywhere, so the output is all zero and
    // every group's scale is the floor.
    hid[..h].fill(0.0);
    src[..h].fill(0.0);
    if m > 1 {
        // Row 1: one NaN in src, so the whole row is NaN.
        src[h + 3] = f32::NAN;
    }
    if m > 2 {
        // Row 2: magnitudes near the top of BF16's range; the FP32 sum of
        // squares overflows to inf, so rms is 0 and the row normalizes to 0.
        for x in &mut hid[2 * h..3 * h] {
            *x = half::bf16::from_f32(*x * 1e36).to_f32();
        }
    }
    (hid, src, weight)
}

const OUTPUTS: [&str; 5] = ["hidden", "residual", "output", "codes", "scales"];

fn row(gpu: &dyn GpuBackend, k: &Kernels, rng: &mut Lcg, m: u32, h: u32) -> Result<usize> {
    let (mu, hu) = (m as usize, h as usize);
    let (hid, src_h, w_h) = inputs(rng, mu, hu);
    let hidden0 = bf16_bytes(&hid);
    let src = upload(gpu, &bf16_bytes(&src_h))?;
    let w = upload(gpu, &bf16_bytes(&w_h))?;
    let (p, f) = (
        Side::new(gpu, &hidden0, mu, hu)?,
        Side::new(gpu, &hidden0, mu, hu)?,
    );
    pair(gpu, k, &p, src, w, m, h)?;
    run_fused(gpu, k, &f, src, w, m, h)?;
    let (pv, fv) = (p.read(gpu)?, f.read(gpu)?);
    let bad: Vec<usize> = pv
        .iter()
        .zip(&fv)
        .map(|(a, b)| unequal_bytes(a, b))
        .collect();
    // Host: the shared quantizer's rule over the pair's BF16 output.
    let (hq, hs) = per_token_group_quant(&bf16_to_f32s(&pv[2]), mu, hu);
    let host_q = unequal_bytes(&fv[3], &hq);
    let host_s = unequal_bytes(&fv[4], &f32_bytes(&hs));
    let fails = bad.iter().sum::<usize>() + host_q + host_s;
    // Timing: the in/out hidden is restored before every rep, outside it.
    let us_pair = time_warm_us(gpu, || {
        p.reset(gpu, &hidden0)?;
        pair(gpu, k, &p, src, w, m, h)
    })?;
    let us_fused = time_warm_us(gpu, || {
        f.reset(gpu, &hidden0)?;
        run_fused(gpu, k, &f, src, w, m, h)
    })?;
    let per = |b: f64, us: f64| gbs(b * mu as f64 * hu as f64, us);
    let detail: Vec<String> = OUTPUTS
        .iter()
        .zip(&bad)
        .map(|(n, b)| format!("{n}={b}"))
        .collect();
    println!(
        "  [{}] M={m:<5} H={h}: fused vs pair {} | vs host codes={host_q} scales={host_s} | pair {us_pair:.1} us \
         ({:.0} GB/s) fused {us_fused:.1} us ({:.0} GB/s) | {:.2}x (both include the hidden reset)",
        verdict(fails),
        detail.join(" "),
        per(13.0, us_pair),
        per(11.0, us_fused),
        us_pair / us_fused
    );
    p.free(gpu)?;
    f.free(gpu)?;
    gpu.free(src)?;
    gpu.free(w)?;
    Ok(usize::from(fails != 0))
}

/// The comparator must be able to fail: the pair with another weight.
fn control(gpu: &dyn GpuBackend, k: &Kernels, rng: &mut Lcg, h: u32) -> Result<usize> {
    let (m, hu) = (64u32, h as usize);
    let (hid, src_h, w_h) = inputs(rng, 64, hu);
    let hidden0 = bf16_bytes(&hid);
    let src = upload(gpu, &bf16_bytes(&src_h))?;
    let w = upload(gpu, &bf16_bytes(&w_h))?;
    let w2 = upload(gpu, &bf16_bytes(&rng.bf16s(hu, 0.1)))?;
    let (p, f) = (
        Side::new(gpu, &hidden0, 64, hu)?,
        Side::new(gpu, &hidden0, 64, hu)?,
    );
    pair(gpu, k, &p, src, w2, m, h)?;
    run_fused(gpu, k, &f, src, w, m, h)?;
    let differ = unequal_bytes(&p.q.read(gpu)?, &f.q.read(gpu)?);
    let ok = differ > 0;
    println!(
        "  [{}] control (pair on another weight) H={h}: {differ} code bytes differ, must be > 0",
        verdict(usize::from(!ok))
    );
    p.free(gpu)?;
    f.free(gpu)?;
    for x in [src, w, w2] {
        gpu.free(x)?;
    }
    Ok(usize::from(!ok))
}

fn main() -> Result<()> {
    let gpu = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let k = Kernels {
        norm: gpu.kernel("norm", "residual_add_rms_norm")?,
        quant: ops::Fp8ActQuant::shared_only(
            gpu.kernel("per_token_group_quant_fp8", "per_token_group_quant_fp8")?,
        ),
        fused: gpu
            .kernel("rmsnorm_quant_fp8_ada", "rmsnorm_quant_fp8_ada")
            .context("rmsnorm_quant_fp8_ada absent: build with AVAROK_TARGET_HW=ada")?,
    };
    let mut rng = Lcg(0x15_0815_2026_0003);
    let mut fails = 0;
    for &h in HIDDEN {
        println!("hidden {h}:");
        for &m in ROWS {
            fails += row(&gpu, &k, &mut rng, m, h)?;
        }
        fails += control(&gpu, &k, &mut rng, h)?;
    }
    ensure!(fails == 0, "{fails} row(s) failed");
    println!("all rows PASS");
    Ok(())
}
