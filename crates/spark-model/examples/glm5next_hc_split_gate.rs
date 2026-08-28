// SPDX-License-Identifier: AGPL-3.0-only
//! BIT-IDENTITY gate for the split mHC pre-pass.
//!
//! `glm5next_hc_mix` + `glm5next_hc_finish` replace the fused single-block `glm5next_hc_pre` on
//! the serve path. The fused kernel is KEPT and is the oracle here: the split must reproduce it
//! **byte for byte**, not "to a tolerance". Nothing about the split is an approximation — the
//! reduction width, the strided accumulation order and the tree reduce are unchanged; only the
//! grid is. A tolerance-based gate would let a real reordering through.
//!
//!   cargo run -p spark-model --release --example glm5next_hc_split_gate \
//!       --features cuda,gpu-examples
//!
//! 🪤 Every rank/H/hc combination gated here is driven with the SAME pseudo-random stream on
//! both arms, and the outputs are poisoned before each launch, so a kernel that fails to write
//! shows up as garbage rather than as a stale pass.

use anyhow::{Result, bail};
use spark_model::layers::ops::{
    Glm5NextMhcKernels, Glm5NextMhcSiteWeights, MHC_MIX_MAX_TOKENS, glm_hc_pre,
};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

/// `(hidden, hc_mult, tokens)`. GLM's own shape first, then the register-bound edge.
/// GLM-5.3-Flash is `hidden_size = 4096, hc_mult = 4` — that case leads, and it is the one
/// the timing block below reports. The rest exercise the grid arithmetic and the register bound.
const CASES: [(usize, usize, usize); 5] = [
    (4096, 4, 1),
    (5120, 4, 1),
    (5120, 4, 7),
    (1024, 2, 3),
    (256, 4, 1),
];
const SINKHORN_ITERS: u32 = 20;
const HC_EPS: f32 = 1e-6;
const NORM_EPS: f32 = 1e-5;

fn mix_hc(hc: usize) -> usize {
    (2 + hc) * hc
}

/// A cheap deterministic LCG — the two arms must see the SAME bytes, so this must not depend on
/// hashing order or thread scheduling.
fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

fn dn(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// Fill `p` with 0xAB so an unwritten output cannot pass as a match with the other arm.
fn poison(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<()> {
    g.copy_h2d(&vec![0xABu8; bytes], p)
}

/// Mean wall microseconds per launch, synchronised once at each end (not per launch, which
/// would charge the sync instead of the kernel).
fn time_us(g: &dyn GpuBackend, reps: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..10 {
        f()?;
    }
    g.synchronize(0)?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        f()?;
    }
    g.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / reps as f64)
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let k = Glm5NextMhcKernels::resolve(&gpu)?;
    println!(
        "glm5next hc split gate — fused glm5next_hc_pre is the ORACLE, split must be byte-identical\n"
    );

    for (hid, hc, t) in CASES {
        if t > MHC_MIX_MAX_TOKENS {
            bail!("case T={t} exceeds MHC_MIX_MAX_TOKENS");
        }
        let m = mix_hc(hc);
        let hc_dim = hc * hid;
        let mut seed = 0x5eed_1234_u64 ^ (hid as u64) << 20 ^ (hc as u64) << 8 ^ t as u64;
        let streams: Vec<f32> = (0..t * hc_dim).map(|_| lcg(&mut seed)).collect();
        // Scale `fn` down: the dot product runs over hc*H terms and the split must match the
        // fused sum bit for bit, which it does at any magnitude — this only keeps the sigmoids
        // off their saturated tails so the Sinkhorn is exercised rather than clamped.
        let hc_fn: Vec<f32> = (0..m * hc_dim).map(|_| lcg(&mut seed) * 0.02).collect();
        let hc_base: Vec<f32> = (0..m).map(|_| lcg(&mut seed)).collect();
        let hc_scale: Vec<f32> = vec![0.7, 1.3, 0.9];

        let d_streams = up_f32(&gpu, &streams)?;
        let d_fn = up_f32(&gpu, &hc_fn)?;
        let d_base = up_f32(&gpu, &hc_base)?;
        let d_scale = up_f32(&gpu, &hc_scale)?;

        let y_b = t * hid * 2;
        let post_b = t * hc * 4;
        let comb_b = t * hc * hc * 4;

        // ── arm A: the fused oracle ──
        let (ya, pa, ca) = (gpu.alloc(y_b)?, gpu.alloc(post_b)?, gpu.alloc(comb_b)?);
        poison(&gpu, ya, y_b)?;
        poison(&gpu, pa, post_b)?;
        poison(&gpu, ca, comb_b)?;
        KernelLaunch::new(&gpu, k.hc_pre)
            .grid([t as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(d_streams)
            .arg_ptr(d_fn)
            .arg_ptr(d_scale)
            .arg_ptr(d_base)
            .arg_ptr(ya)
            .arg_ptr(pa)
            .arg_ptr(ca)
            .arg_u32(hid as u32)
            .arg_u32(hc as u32)
            .arg_u32(SINKHORN_ITERS)
            .arg_f32(NORM_EPS)
            .arg_f32(HC_EPS)
            .launch(0)?;
        gpu.synchronize(0)?;
        let (ya_h, pa_h, ca_h) = (
            dn(&gpu, ya, y_b)?,
            dn(&gpu, pa, post_b)?,
            dn(&gpu, ca, comb_b)?,
        );

        // ── arm B: the split pair, through the production launcher ──
        let w = Glm5NextMhcSiteWeights {
            hc_fn: d_fn,
            hc_scale: d_scale,
            hc_base: d_base,
            mix: gpu.alloc(MHC_MIX_MAX_TOKENS * m * 4)?,
        };
        let (yb, pb, cb) = (gpu.alloc(y_b)?, gpu.alloc(post_b)?, gpu.alloc(comb_b)?);
        poison(&gpu, yb, y_b)?;
        poison(&gpu, pb, post_b)?;
        poison(&gpu, cb, comb_b)?;
        glm_hc_pre(
            &gpu,
            &k,
            d_streams,
            &w,
            yb,
            pb,
            cb,
            t as u32,
            hid as u32,
            hc as u32,
            SINKHORN_ITERS,
            NORM_EPS,
            HC_EPS,
            0,
        )?;
        gpu.synchronize(0)?;
        let (yb_h, pb_h, cb_h) = (
            dn(&gpu, yb, y_b)?,
            dn(&gpu, pb, post_b)?,
            dn(&gpu, cb, comb_b)?,
        );

        for (name, a, b) in [
            ("y", &ya_h, &yb_h),
            ("post", &pa_h, &pb_h),
            ("comb", &ca_h, &cb_h),
        ] {
            if a.iter().all(|&x| x == 0xAB) {
                bail!(
                    "H={hid} hc={hc} T={t}: `{name}` is still poison — the ORACLE never wrote it"
                );
            }
            if a != b {
                let i = a.iter().zip(b).position(|(x, y)| x != y).unwrap();
                bail!(
                    "H={hid} hc={hc} T={t}: `{name}` differs at byte {i}: oracle {:#04x} split {:#04x}",
                    a[i],
                    b[i]
                );
            }
        }
        println!("  H={hid:>5} hc={hc} T={t}  mix_hc={m:>2}  y/post/comb BYTE-IDENTICAL");

        // ── timing, GLM's own shape only: which half of the split actually costs ──
        if (hid, hc, t) == (4096, 4, 1) {
            let reps = 300;
            let t_fused = time_us(&gpu, reps, || {
                KernelLaunch::new(&gpu, k.hc_pre)
                    .grid([t as u32, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(d_streams)
                    .arg_ptr(d_fn)
                    .arg_ptr(d_scale)
                    .arg_ptr(d_base)
                    .arg_ptr(ya)
                    .arg_ptr(pa)
                    .arg_ptr(ca)
                    .arg_u32(hid as u32)
                    .arg_u32(hc as u32)
                    .arg_u32(SINKHORN_ITERS)
                    .arg_f32(NORM_EPS)
                    .arg_f32(HC_EPS)
                    .launch(0)
            })?;
            let t_mix = time_us(&gpu, reps, || {
                KernelLaunch::new(&gpu, k.hc_mix)
                    .grid([t as u32, m as u32, 1])
                    .block([256, 1, 1])
                    .arg_ptr(d_streams)
                    .arg_ptr(d_fn)
                    .arg_ptr(w.mix)
                    .arg_u32(hid as u32)
                    .arg_u32(hc as u32)
                    .arg_f32(NORM_EPS)
                    .launch(0)
            })?;
            let t_fin = time_us(&gpu, reps, || {
                KernelLaunch::new(&gpu, k.hc_finish)
                    .grid([t as u32, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(d_streams)
                    .arg_ptr(w.mix)
                    .arg_ptr(d_scale)
                    .arg_ptr(d_base)
                    .arg_ptr(yb)
                    .arg_ptr(pb)
                    .arg_ptr(cb)
                    .arg_u32(hid as u32)
                    .arg_u32(hc as u32)
                    .arg_u32(SINKHORN_ITERS)
                    .arg_f32(HC_EPS)
                    .launch(0)
            })?;
            let t_fin1 = time_us(&gpu, reps, || {
                KernelLaunch::new(&gpu, k.hc_finish)
                    .grid([t as u32, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(d_streams)
                    .arg_ptr(w.mix)
                    .arg_ptr(d_scale)
                    .arg_ptr(d_base)
                    .arg_ptr(yb)
                    .arg_ptr(pb)
                    .arg_ptr(cb)
                    .arg_u32(hid as u32)
                    .arg_u32(hc as u32)
                    .arg_u32(1)
                    .arg_f32(HC_EPS)
                    .launch(0)
            })?;
            // `t_fin` above launches `hc_finish` on the OLD grid (T,1,1) directly, so it is
            // the before-picture. This one goes through the production launcher, which grids it
            // (T, 1 + ceil(H/256)) and spreads the collapse.
            let t_prod = time_us(&gpu, reps, || {
                glm_hc_pre(
                    &gpu,
                    &k,
                    d_streams,
                    &w,
                    yb,
                    pb,
                    cb,
                    t as u32,
                    hid as u32,
                    hc as u32,
                    SINKHORN_ITERS,
                    NORM_EPS,
                    HC_EPS,
                    0,
                )
            })?;
            println!("    hc_finish@iters=1 {t_fin1:8.1}  (Sinkhorn cost = (fin - fin1) * 20/19)");
            println!(
                "    PRODUCTION mix+finish (wide collapse) {t_prod:8.1} us/call = {:8.3} ms/token over 90 sites",
                t_prod * 90.0 / 1000.0
            );
            println!(
                "\n  TIMING (GLM shape, {reps} reps, us/call):\n    \
                 fused hc_pre {t_fused:8.1}\n    hc_mix       {t_mix:8.1}\n    \
                 hc_finish    {t_fin:8.1}\n    split total  {:8.1}\n    \
                 per token (90 sites) {:8.3} ms\n",
                t_mix + t_fin,
                (t_mix + t_fin) * 90.0 / 1000.0
            );
        }
    }

    println!(
        "\nsplit gate PASS — glm5next_hc_mix + glm5next_hc_finish == glm5next_hc_pre, byte for byte"
    );
    Ok(())
}
