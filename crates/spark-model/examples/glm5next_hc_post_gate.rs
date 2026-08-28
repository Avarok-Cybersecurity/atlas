// SPDX-License-Identifier: AGPL-3.0-only
//! BIT-IDENTITY gate for the widened mHC post-pass.
//!
//! `glm5next_hc_post` (compile-time trip counts, grid `(T, ceil(H/256))`) replaces
//! `glm5next_hc_post_ref` (runtime trip counts, grid `(T,1)`) on the serve path. The reference
//! kernel is KEPT and is the oracle here: the widened kernel must reproduce it **byte for
//! byte**, not "to a tolerance". Nothing about the change is an approximation — the executed
//! arithmetic and its order are unchanged, only which thread runs it and where `rv` lives — so
//! a tolerance gate would let a real reordering through.
//!
//!   cargo run -p spark-model --release --example glm5next_hc_post_gate \
//!       --features cuda,gpu-examples
//!
//! 🔴 The ALIASED case is the one that matters. On the serve path `out` IS the highway
//! `streams` tensor, so `hc_post` reads and writes the same allocation. That is safe only
//! because a block touching column `d` reads exactly `res[i*H+d]` and writes exactly
//! `o[j*H+d]` — columns never cross blocks. `ALIAS` below drives that configuration directly
//! rather than trusting the argument.

use anyhow::{Result, bail};
use spark_model::layers::ops::{Glm5NextMhcKernels, glm_hc_post};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

/// `(hidden, hc_mult, tokens)`. GLM-5.3's own shape first, then shapes that exercise the
/// grid arithmetic: H not a multiple of the 256-wide block, and a single-block H.
const CASES: [(usize, usize, usize); 5] = [
    (4096, 4, 1),
    (4096, 4, 5),
    (5120, 4, 1),
    (1000, 2, 3),
    (256, 4, 1),
];

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

/// Upload as BF16 by truncating the f32 pattern — this is only the `block_out` operand, and
/// both arms read the identical bytes, so the rounding mode is irrelevant to the comparison.
fn up_bf16(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d
        .iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

fn dn(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

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
    let k_ref = gpu.kernel("glm5next_mhc", "glm5next_hc_post_ref")?;
    println!(
        "glm5next hc_post gate — glm5next_hc_post_ref is the ORACLE, the widened kernel must be byte-identical\n"
    );

    for (hid, hc, t) in CASES {
        for alias in [false, true] {
            let mut seed = 0x9051_0000_u64 ^ (hid as u64) << 20 ^ (hc as u64) << 8 ^ t as u64;
            let block_out: Vec<f32> = (0..t * hid).map(|_| lcg(&mut seed)).collect();
            let residual: Vec<f32> = (0..t * hc * hid).map(|_| lcg(&mut seed)).collect();
            let post: Vec<f32> = (0..t * hc).map(|_| lcg(&mut seed) + 1.0).collect();
            let comb: Vec<f32> = (0..t * hc * hc).map(|_| lcg(&mut seed)).collect();

            let d_block = up_bf16(&gpu, &block_out)?;
            let d_post = up_f32(&gpu, &post)?;
            let d_comb = up_f32(&gpu, &comb)?;
            let out_b = t * hc * hid * 4;

            // Each arm gets its OWN residual buffer: under `alias` the kernel writes into it,
            // so a shared one would feed arm B what arm A left behind.
            let ra = up_f32(&gpu, &residual)?;
            let rb = up_f32(&gpu, &residual)?;
            let (oa, ob) = if alias {
                (ra, rb)
            } else {
                let (a, b) = (gpu.alloc(out_b)?, gpu.alloc(out_b)?);
                poison(&gpu, a, out_b)?;
                poison(&gpu, b, out_b)?;
                (a, b)
            };

            // ── arm A: the reference, one block, runtime trip counts ──
            KernelLaunch::new(&gpu, k_ref)
                .grid([t as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(d_block)
                .arg_ptr(ra)
                .arg_ptr(d_post)
                .arg_ptr(d_comb)
                .arg_ptr(oa)
                .arg_u32(hid as u32)
                .arg_u32(hc as u32)
                .launch(0)?;
            gpu.synchronize(0)?;
            let a_h = dn(&gpu, oa, out_b)?;

            // ── arm B: the widened kernel, through the production launcher ──
            glm_hc_post(
                &gpu, k.hc_post, d_block, rb, d_post, d_comb, ob, t as u32, hid as u32, hc as u32,
                0,
            )?;
            gpu.synchronize(0)?;
            let b_h = dn(&gpu, ob, out_b)?;

            if a_h.iter().all(|&x| x == 0xAB) {
                bail!("H={hid} hc={hc} T={t} alias={alias}: the ORACLE never wrote `out`");
            }
            if a_h != b_h {
                let i = a_h.iter().zip(&b_h).position(|(x, y)| x != y).unwrap();
                bail!(
                    "H={hid} hc={hc} T={t} alias={alias}: `out` differs at byte {i}: \
                     oracle {:#04x} widened {:#04x}",
                    a_h[i],
                    b_h[i]
                );
            }
            println!("  H={hid:>5} hc={hc} T={t} alias={alias:<5}  out BYTE-IDENTICAL");

            // ── timing, GLM's own decode shape only ──
            if (hid, hc, t, alias) == (4096, 4, 1, true) {
                let reps = 500;
                let t_ref = time_us(&gpu, reps, || {
                    KernelLaunch::new(&gpu, k_ref)
                        .grid([t as u32, 1, 1])
                        .block([256, 1, 1])
                        .arg_ptr(d_block)
                        .arg_ptr(ra)
                        .arg_ptr(d_post)
                        .arg_ptr(d_comb)
                        .arg_ptr(oa)
                        .arg_u32(hid as u32)
                        .arg_u32(hc as u32)
                        .launch(0)
                })?;
                let t_new = time_us(&gpu, reps, || {
                    glm_hc_post(
                        &gpu,
                        k.hc_post,
                        d_block,
                        rb,
                        d_post,
                        d_comb,
                        ob,
                        t as u32,
                        hid as u32,
                        hc as u32,
                        0,
                    )
                })?;
                println!(
                    "\n  TIMING (GLM decode shape H=4096 hc=4 T=1, {reps} reps, us/call):\n    \
                     hc_post_ref (1 block) {t_ref:8.1}\n    hc_post     (16 blocks) {t_new:8.1}\n    \
                     speedup                {:8.2}x\n    \
                     per token (90 sites)   {:8.3} ms -> {:8.3} ms\n",
                    t_ref / t_new,
                    t_ref * 90.0 / 1000.0,
                    t_new * 90.0 / 1000.0
                );
            }
        }
    }

    println!("\nhc_post gate PASS — glm5next_hc_post == glm5next_hc_post_ref, byte for byte");
    Ok(())
}
