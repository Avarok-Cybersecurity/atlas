// SPDX-License-Identifier: AGPL-3.0-only
//! BYTE-parity gate for the NVFP4 batched SSM projections.
//!
//! Every NVFP4 model that decodes more than one sequence at a time routes its
//! projections through these kernels — the batched LM head
//! (`model/trait_impl/lm_head_batched.rs`), `dense_ffn`, the qwen3 multi-seq
//! `qkv`/`o_proj`/`ffn` paths, the MTP head, and Nemotron-H's SSM
//! `in_proj`/`out_proj` (whose batched-projection rung is only sound if this
//! gate passes). All of that is legitimate only if
//! `w4a16_gemv_batch2/3/4/8/16` is byte-identical to M separate `w4a16_gemv`
//! calls. The kernel header claimed exactly that ("Per-row accumulation
//! order is IDENTICAL to `w4a16_gemv`") — this test is what decides it.
//!
//! Mirrors `w8a16_batch_bitparity_microtest.rs`: RAW BF16 BYTES, not a
//! cosine, at PRODUCTION projection shapes, over several seeds and every M
//! each tier can serve. A cosine gate is exactly what hid the
//! `w8a16_gemv_batch4` fused-add defect.
//!
//! Tiers 5/6/7 (exact-M, added 2026-08-17) are checked at EVERY M they can be
//! dispatched at, because `w4a16_gemv_tiers::select_tier` widens to the next
//! resolved tier when one is absent. They are optional: an older target PTX
//! set without them still runs the gate on 4/8/16 + the fixed-M kernels.
//!
//! Exit: 0 all legs byte-identical, 1 any leg differs,
//! 2 kernels absent from this target's module set.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=nemotron-3-nano-30b-a3b \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example w4a16_batch_bitparity_microtest

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const GROUP_SIZE: usize = 16;
const MAX_M: usize = 16;
const SCALE2: f32 = 0.0123_f32;

/// The two `nano` rows are the EXACT shapes Nano-30B-A3B / Lightning-30B-A3B
/// dispatch (hidden 2688, d_inner 4096, in_proj_size 10304). The two `super`
/// rows cover the hidden-4096 Super/Puzzle backbone at the natural doubling
/// (d_inner 8192); they are shape COVERAGE for a larger N and a deeper K, not
/// a claim about that checkpoint's exact `in_proj_size`.
const SHAPES: [(&str, usize, usize); 10] = [
    ("nano  in_proj  [10304 x 2688]", 10304, 2688),
    ("nano  out_proj [ 2688 x 4096]", 2688, 4096),
    ("super in_proj  [18560 x 4096]", 18560, 4096),
    ("super out_proj [ 4096 x 8192]", 4096, 8192),
    // qwen3.8-27b dense (hidden 5120, 24q/4kv x head_dim 256, ffn 17408,
    // vocab 248320): the EXACT projection + LM-head shapes the MTP verify
    // forward dispatches at M=K. Added 2026-08-19 while hunting the temp-0
    // spec-decode divergence — the nemotron rows above all PASS on the
    // qwen3.8 target, but the prior defects in this family were
    // SHAPE-specific, so parity must be proven at these dims too.
    ("qwen3.8 q_proj  [ 6144 x  5120]", 6144, 5120),
    ("qwen3.8 kv_proj [ 1024 x  5120]", 1024, 5120),
    ("qwen3.8 o_proj  [ 5120 x  6144]", 5120, 6144),
    ("qwen3.8 gate/up [17408 x  5120]", 17408, 5120),
    ("qwen3.8 down    [ 5120 x 17408]", 5120, 17408),
    ("qwen3.8 lm_head [248320 x 5120]", 248320, 5120),
];

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn r(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn down(g: &dyn GpuBackend, p: DevicePtr, n_bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n_bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn worst_delta(a: &[u8], b: &[u8]) -> (usize, f32) {
    let mut n_diff = 0usize;
    let mut worst = 0f32;
    for (x, y) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
        if x != y {
            n_diff += 1;
            let fx = bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32();
            let fy = bf16::from_bits(u16::from_le_bytes([y[0], y[1]])).to_f32();
            worst = worst.max((fx - fy).abs());
        }
    }
    (n_diff, worst)
}

/// `w4a16_gemv(A, B_packed, B_scale, scale2, C, [M,] N, K)`.
#[allow(clippy::too_many_arguments)]
fn launch(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    a: DevicePtr,
    w: DevicePtr,
    ws: DevicePtr,
    c: DevicePtr,
    m: Option<u32>,
    n: u32,
    k: u32,
) -> Result<()> {
    let mut l = KernelLaunch::new(g, kh)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w)
        .arg_ptr(ws)
        .arg_f32(SCALE2)
        .arg_ptr(c);
    if let Some(m) = m {
        l = l.arg_u32(m);
    }
    l.arg_u32(n).arg_u32(k).launch(0)
}

/// M x single-row `w4a16_gemv` into `c` — the canonical reference the
/// per-seq default decode loop produces.
#[allow(clippy::too_many_arguments)]
fn reference(
    g: &dyn GpuBackend,
    m1: KernelHandle,
    a: DevicePtr,
    w: DevicePtr,
    ws: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    for t in 0..m {
        launch(
            g,
            m1,
            a.offset(t * k * 2),
            w,
            ws,
            c.offset(t * n * 2),
            None,
            n as u32,
            k as u32,
        )?;
    }
    Ok(())
}

struct Inputs {
    a: Vec<u8>,
    w: Vec<u8>,
    ws: Vec<u8>,
}

/// Random NVFP4 operands. Block-scale bytes are held in 0x30..=0x47 — finite
/// positive E4M3 (0.5 .. 3.5); 0x7F/0xFF are NaN and a zero scale would blank
/// the output and hide any reordering.
fn gen_inputs(seed: u64, n: usize, k: usize) -> Inputs {
    let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xB4B4);
    let a = (0..MAX_M * k)
        .flat_map(|_| bf16::from_f32(rng.r(-1.5, 1.5)).to_bits().to_le_bytes())
        .collect();
    let w = (0..n * k / 2).map(|_| rng.r(0.0, 256.0) as u8).collect();
    let ws = (0..n * k / GROUP_SIZE)
        .map(|_| 0x30u8 + (rng.r(0.0, 24.0) as u8))
        .collect();
    Inputs { a, w, ws }
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;

    let m1_k = g.kernel("w4a16_gemv", "w4a16_gemv");
    // Every M a tier can be DISPATCHED at, not just the M its width names:
    // `w4a16_gemv_tiers::select_tier` widens to the next resolved tier when
    // one is missing, so batch7 can legitimately serve M=5 and batch8 M=5..8.
    // Checking a tier only at its own width would leave the widened legs
    // unproven.
    let tier_specs: [(&str, Vec<usize>); 6] = [
        ("batch4", (2..=4).collect()),
        ("batch5", (2..=5).collect()),
        ("batch6", (2..=6).collect()),
        ("batch7", (2..=7).collect()),
        ("batch8", (2..=8).collect()),
        ("batch16", (2..=16).collect()),
    ];
    let tiers: Vec<(&str, KernelHandle, Vec<usize>)> = tier_specs
        .iter()
        .filter_map(|(t, ms)| {
            let kh = g.kernel("w4a16_gemv", &format!("w4a16_gemv_{t}")).ok()?;
            Some((*t, kh, ms.clone()))
        })
        .collect();

    // batch2/batch3 are SEPARATE fixed-M kernels (no `M` argument), not
    // instantiations of the batchm template, and serve K=2/K=3 spec verify
    // plus C=2/C=3 multi-seq decode (dense_ffn, qwen3 qkv/o_proj, MoE
    // forward_k3). They had never been byte-checked against M x w4a16_gemv.
    let fixed_tiers: Vec<(&str, KernelHandle, usize)> = [("batch2", 2usize), ("batch3", 3usize)]
        .iter()
        .filter_map(|(t, m)| {
            let kh = g.kernel("w4a16_gemv", &format!("w4a16_gemv_{t}")).ok()?;
            Some((*t, kh, *m))
        })
        .collect();
    let m1_k = match m1_k {
        // 4/8/16 are the tiers every NVFP4 target has carried; 5/6/7 are newer
        // and legitimately absent from an older target PTX set, so they are
        // checked when present but never required to run the gate.
        Ok(kh)
            if tiers
                .iter()
                .filter(|(t, ..)| matches!(*t, "batch4" | "batch8" | "batch16"))
                .count()
                == 3
                && fixed_tiers.len() == 2 =>
        {
            kh
        }
        _ => {
            println!("w4a16 GEMV kernels absent from this target set — SKIP");
            std::process::exit(2);
        }
    };

    let mut clean = true;
    let mut control_ok = true;
    for seed in [1u64, 99, 12345] {
        for (label, n, k) in SHAPES {
            let inp = gen_inputs(seed, n, k);
            let a_d = up(g, &inp.a)?;
            let w_d = up(g, &inp.w)?;
            let ws_d = up(g, &inp.ws)?;
            let c_batch = g.alloc(MAX_M * n * 2)?;
            let c_ref = g.alloc(MAX_M * n * 2)?;

            for (tier, kh, ms) in &tiers {
                for &m in ms {
                    g.memset(c_batch, 0, MAX_M * n * 2)?;
                    g.memset(c_ref, 0, MAX_M * n * 2)?;
                    launch(
                        g,
                        *kh,
                        a_d,
                        w_d,
                        ws_d,
                        c_batch,
                        Some(m as u32),
                        n as u32,
                        k as u32,
                    )?;
                    reference(g, m1_k, a_d, w_d, ws_d, c_ref, m, n, k)?;
                    g.synchronize(0)?;
                    let cb = down(g, c_batch, m * n * 2)?;
                    let cr = down(g, c_ref, m * n * 2)?;
                    let identical = cb == cr;
                    let (n_diff, worst) = worst_delta(&cb, &cr);
                    clean &= identical;
                    println!(
                        "seed {seed:>5}  {label}  {tier:<7} M={m:<3} \
                         byte-identical={identical:<5} diff_elems={n_diff:<7} \
                         max|delta|={worst:.6}"
                    );
                }
            }

            for (tier, kh, m) in &fixed_tiers {
                let m = *m;
                g.memset(c_batch, 0, MAX_M * n * 2)?;
                g.memset(c_ref, 0, MAX_M * n * 2)?;
                launch(g, *kh, a_d, w_d, ws_d, c_batch, None, n as u32, k as u32)?;
                reference(g, m1_k, a_d, w_d, ws_d, c_ref, m, n, k)?;
                g.synchronize(0)?;
                let cb = down(g, c_batch, m * n * 2)?;
                let cr = down(g, c_ref, m * n * 2)?;
                let identical = cb == cr;
                let (n_diff, worst) = worst_delta(&cb, &cr);
                clean &= identical;
                println!(
                    "seed {seed:>5}  {label}  {tier:<7} M={m:<3} \
                     byte-identical={identical:<5} diff_elems={n_diff:<7} \
                     max|delta|={worst:.6}"
                );
            }

            // ── Negative control: the harness MUST see a 1-ULP activation
            // perturbation on row 1. Without this a byte compare that
            // silently compares two zeroed buffers would "pass".
            let (tier_kh, m) = (tiers[0].1, 4usize);
            let mut pert = inp.a.clone();
            pert[2 * (k + 7)] ^= 1;
            let a_pert = up(g, &pert)?;
            g.memset(c_batch, 0, MAX_M * n * 2)?;
            g.memset(c_ref, 0, MAX_M * n * 2)?;
            launch(
                g,
                tier_kh,
                a_d,
                w_d,
                ws_d,
                c_batch,
                Some(m as u32),
                n as u32,
                k as u32,
            )?;
            reference(g, m1_k, a_pert, w_d, ws_d, c_ref, m, n, k)?;
            g.synchronize(0)?;
            let differs = down(g, c_batch, m * n * 2)? != down(g, c_ref, m * n * 2)?;
            control_ok &= differs;
            println!("seed {seed:>5}  {label}  CONTROL 1-ULP perturbation detected={differs}");
            g.free(a_pert).ok();

            for p in [a_d, w_d, ws_d, c_batch, c_ref] {
                g.free(p).ok();
            }
        }
    }

    if !control_ok {
        println!("FAIL — negative control did not mismatch; this harness is VACUOUS.");
        std::process::exit(1);
    }
    if clean {
        println!(
            "PASS — the w4a16 batched GEMV tiers (batch2/3/4/8/16) are byte-identical to \
             M x w4a16_gemv at every projection shape and every M they serve."
        );
        Ok(())
    } else {
        println!(
            "FAIL — a w4a16 batched tier is NOT byte-identical, so NVFP4 decode output \
             depends on how many sequences happen to share the step."
        );
        std::process::exit(1);
    }
}
