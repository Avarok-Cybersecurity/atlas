// SPDX-License-Identifier: AGPL-3.0-only
//! GPU oracle + timings for the Ada affine 4-bit prompt GEMMs (port lane L17b:
//! `kernels/ada/common/affine_prefill_ada.cu`, `affine_swiglu_sums_ada.cu`; host
//! `crates/spark-model/src/port/affine/prefill.rs`), against the CPU oracles of
//! `port/affine/prefill_oracle.rs`.
//!
//! WHAT THIS PINS (each row prints PASS/FAIL; any FAIL exits non-zero):
//!  * **quantizer == host model**: every int8 code byte, scale bit and sum bit.
//!  * **int8 GEMM == host model, byte for byte** (`oracle::int8_gemm`), at M =
//!    17, 64, 65, 128, 512, 2048 on the 4B gate (9216 x 2560), down (2560 x 9216)
//!    and a ragged 129 x 2560, on sampled rows; and within the derived bound of
//!    the float64 affine product (`oracle::int8_bound`).
//!  * **BF16 GEMM within its float64 bound** (`oracle::bf16_reference`) at the same
//!    shapes and rows.
//!  * **row invariance**: the first 17 rows at M = 2048 equal M = 17, both routes.
//!  * **five-run byte repeat**, and **_many == separate calls** (gate + up).
//!  * **exceptional groups**: a NaN row and an overflowing group make the int8
//!    route's output equal the BF16 route's, and the fallback counter moves by one.
//!  * **fusion == pair**: `affine4_swiglu_act_quant_ada` against `moe_silu_mul`
//!    then `affine4_act_quant_ada` (h, codes, scales, sums), and the fused down
//!    projection against the unfused one, clean and with a NaN gate element.
//!  * **controls that must fail**: the int8 bytes against a model with the last
//!    group dropped, and the int8 route against the BF16 route on clean inputs.
//!  * every output carries 64 sentinel bytes either side.
//!
//! TIMINGS: sync'd `Instant`, 3 warm-ups, 20 reps, TFLOP/s = 2 M N K / t, for the
//! 4B FFN shapes at M = 2048 and 3921, BF16 against int8 (quantizer included).
//!
//! Run on the Ada card, in a build of the ada target:
//!     AVAROK_TARGET_HW=ada cargo run --release -p spark-model \
//!       --example affine_prefill_microtest --features cuda,gpu-examples
use anyhow::{Context, Result, ensure};
use spark_model::layers::ops;
use spark_model::port::affine::prefill::oracle::{self, Lcg};
use spark_model::port::affine::prefill::{self, Int8Act};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::GpuBackend;

#[path = "common/affine_prefill_support.rs"]
mod support;
use support::{
    Guarded, MAX_M, bf16_bytes, free_scratch, gemm, pick, read, sample_rows, scratch, time_us,
    u16s, upload, verdict, weight,
};

const MS: &[usize] = &[17, 64, 65, 128, 512, 2048];

/// Gates 1 to 4 and the controls on one `n x k` shape.
fn shape(gpu: &dyn GpuBackend, rng: &mut Lcg, n: usize, k: usize) -> Result<usize> {
    let w = weight(gpu, rng, n, k)?;
    let m_max = MS[MS.len() - 1];
    let x: Vec<f32> = (0..m_max * k)
        .map(|j| rng.bf16_normal(if (j / k) % 7 == 3 { 8.0 } else { 1.0 }))
        .collect();
    let xd = upload(gpu, &bf16_bytes(&x))?;
    let s = scratch(gpu, k)?;
    let q = oracle::quantize_rows(&x, k, m_max, k);
    let kg = k / oracle::GROUP;
    let mut fails = 0;

    // Gate 1: the quantizer, bytes and bits.
    let act = Int8Act::carve(&s, m_max, k, prefill::flags(gpu, 0)?).context("carve")?;
    prefill::quantize_int8_rows(gpu, xd, k, m_max, k, &act, 0)?;
    let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
    let ok = read(gpu, act.xq, m_max * k)? == q.xq
        && read(gpu, act.xa, m_max * kg * 4)? == f32_bytes(&q.xa)
        && read(gpu, act.xs, m_max * kg * 4)? == f32_bytes(&q.xs);
    println!(
        "  [{}] {n}x{k} quantizer codes, scales and sums == host model",
        verdict(ok)
    );
    fails += usize::from(!ok);

    // Host references on the sampled rows of the largest M (prompt rows are row-invariant).
    let rows = sample_rows(m_max);
    let model = oracle::int8_gemm(&w.host, &q, &rows);
    let want = oracle::exact(&w.host, &x, k, &rows);
    let ibound = oracle::int8_bound(&w.host, &x, k, &q, &rows, &want);
    let (bwant, bbound) = oracle::bf16_reference(&w.host, &x, k, &rows);
    // Indices into `rows` of the rows below m.
    let below = |m: usize| -> Vec<usize> { (0..rows.len()).filter(|&i| rows[i] < m).collect() };

    // Gates 2 and 3, and row invariance.
    let mut first = (Vec::new(), Vec::new());
    for &m in MS {
        let idx = below(m);
        let local: Vec<usize> = idx.iter().map(|&i| rows[i]).collect();
        let (i8, bf) = (
            gemm(gpu, &w, xd, m, &s, true)?,
            gemm(gpu, &w, xd, m, &s, false)?,
        );
        let (gi, gb) = (pick(&i8, n, &local), pick(&bf, n, &local));
        let bytes_bad = gi
            .iter()
            .zip(pick(&model, n, &idx))
            .filter(|(a, b)| **a != *b)
            .count();
        let (iw, ib) = (pick(&want, n, &idx), pick(&ibound, n, &idx));
        let (bw, bb) = (pick(&bwant, n, &idx), pick(&bbound, n, &idx));
        let (iv, bv) = (
            oracle::violations(&gi, &iw, &ib),
            oracle::violations(&gb, &bw, &bb),
        );
        let ok = bytes_bad == 0 && iv == 0 && bv == 0;
        println!(
            "  [{}] {n}x{k} M={m:<5} int8 bytes != model {bytes_bad}, int8 out of bound {iv}, \
             bf16 out of bound {bv}",
            verdict(ok)
        );
        fails += usize::from(!ok);
        if m == MS[0] {
            first = (i8, bf);
        } else if m == m_max {
            let ok = i8[..17 * n] == first.0[..] && bf[..17 * n] == first.1[..];
            println!(
                "  [{}] {n}x{k} rows 0..17 at M={m} == M=17, both routes",
                verdict(ok)
            );
            fails += usize::from(!ok);
        }
    }

    // Gate 4: five-run byte repeat at M = 512.
    let base = (
        gemm(gpu, &w, xd, 512, &s, true)?,
        gemm(gpu, &w, xd, 512, &s, false)?,
    );
    let mut same = true;
    for _ in 0..4 {
        same &= gemm(gpu, &w, xd, 512, &s, true)? == base.0;
        same &= gemm(gpu, &w, xd, 512, &s, false)? == base.1;
    }
    println!("  [{}] {n}x{k} five runs byte-equal", verdict(same));
    fails += usize::from(!same);

    // Controls: a dropped last group changes the int8 bytes; int8 and BF16 differ on clean inputs.
    let mut dropped = w.host.clone();
    for col in 0..n {
        dropped.scales[col * kg + kg - 1] = 0.0;
        dropped.biases[col * kg + kg - 1] = 0.0;
    }
    let bad = oracle::int8_gemm(&dropped, &q, &rows);
    let idx = below(512);
    let local: Vec<usize> = idx.iter().map(|&i| rows[i]).collect();
    let got = pick(&base.0, n, &local);
    let differ = got
        .iter()
        .zip(pick(&bad, n, &idx))
        .filter(|(a, b)| **a != *b)
        .count();
    let routes = base.0.iter().zip(&base.1).filter(|(a, b)| a != b).count();
    let ok = differ > 0 && routes > 0;
    println!(
        "  [{}] {n}x{k} controls: dropped group differs on {differ}, int8 vs bf16 on {routes}",
        verdict(ok)
    );
    fails += usize::from(!ok);
    w.free(gpu)?;
    gpu.free(xd)?;
    free_scratch(gpu, &s)?;
    Ok(fails)
}

/// `_many` against separate calls, and exceptional groups, on the 4B gate/up shape.
fn many_and_exceptional(gpu: &dyn GpuBackend, rng: &mut Lcg) -> Result<usize> {
    let (n, k, m) = (9216, 2560, 300);
    let (a, b) = (weight(gpu, rng, n, k)?, weight(gpu, rng, n, k)?);
    let mut x: Vec<f32> = (0..m * k).map(|_| rng.bf16_normal(1.0)).collect();
    let xd = upload(gpu, &bf16_bytes(&x))?;
    let s = scratch(gpu, k)?;
    let mut fails = 0;
    for int8 in [true, false] {
        let (oa, ob) = (Guarded::new(gpu, m * n * 2)?, Guarded::new(gpu, m * n * 2)?);
        let members = [(a.dev, oa.ptr()), (b.dev, ob.ptr())];
        ensure!(
            prefill::run_packed(&members, gpu, xd, m, &s, int8, 0)?,
            "declined"
        );
        let ok = u16s(&oa.read(gpu)?) == gemm(gpu, &a, xd, m, &s, int8)?
            && u16s(&ob.read(gpu)?) == gemm(gpu, &b, xd, m, &s, int8)?;
        let route = if int8 { "int8" } else { "bf16" };
        println!("  [{}] gate+up _many == two calls ({route})", verdict(ok));
        fails += usize::from(!ok);
        gpu.free(oa.base)?;
        gpu.free(ob.base)?;
    }
    // A NaN in row 3 and an overflowing group in row 5: the int8 route gives the BF16 route's bytes.
    x[3 * k + 100] = f32::NAN;
    x[5 * k + 128..5 * k + 192].fill(3.0e38);
    let xe = upload(gpu, &bf16_bytes(&x))?;
    let stale = prefill::take_fallbacks(gpu, 0)?;
    let (i8, bf) = (
        gemm(gpu, &a, xe, m, &s, true)?,
        gemm(gpu, &a, xe, m, &s, false)?,
    );
    let counted = prefill::take_fallbacks(gpu, 0)?;
    let ok = i8 == bf && counted == 1 && stale == 0;
    println!(
        "  [{}] exceptional input: int8 route == bf16 route, fallbacks {counted} (want 1, stale {stale})",
        verdict(ok)
    );
    fails += usize::from(!ok);
    a.free(gpu)?;
    b.free(gpu)?;
    gpu.free(xd)?;
    gpu.free(xe)?;
    free_scratch(gpu, &s)?;
    Ok(fails)
}

/// The fusion against the pair, and the fused down projection against the unfused one.
fn fusion(gpu: &dyn GpuBackend, rng: &mut Lcg) -> Result<usize> {
    let (n, k, m) = (2560, 9216, 257);
    let down = weight(gpu, rng, n, k)?;
    let mut g: Vec<f32> = (0..m * k).map(|_| rng.bf16_normal(2.0)).collect();
    let u: Vec<f32> = (0..m * k).map(|_| rng.bf16_normal(1.0)).collect();
    let ud = upload(gpu, &bf16_bytes(&u))?;
    let (prod, h) = (gpu.alloc(m * k * 2)?, gpu.alloc(m * k * 2)?);
    let silu = gpu.kernel("moe_silu_mul", "moe_silu_mul")?;
    let flags = prefill::flags(gpu, 0)?;
    let mut fails = 0;
    // Clean inputs (the int8 path end to end), then a NaN gate element (the exceptional path).
    for (case, want_fb) in [("clean", 0), ("NaN gate", 2)] {
        if want_fb > 0 {
            g[7 * k + 5] = f32::NAN;
        }
        let gd = upload(gpu, &bf16_bytes(&g))?;
        let (s1, s2) = (scratch(gpu, k)?, scratch(gpu, k)?);
        let a1 = Int8Act::carve(&s1, m, k, flags).context("carve")?;
        let a2 = Int8Act::carve(&s2, m, k, flags).context("carve")?;
        ops::silu_mul(gpu, silu, gd, ud, prod, u32::try_from(m * k)?, 0)?;
        prefill::quantize_int8_rows(gpu, prod, k, m, k, &a1, 0)?;
        let pair = [
            read(gpu, prod, m * k * 2)?,
            read(gpu, a1.xq, m * k)?,
            read(gpu, a1.xa, m * k / 8)?,
        ];
        prefill::swiglu_quantize(gpu, gd, ud, k, h, k, m, k, &a2, 0)?;
        let fused = [
            read(gpu, h, m * k * 2)?,
            read(gpu, a2.xq, m * k)?,
            read(gpu, a2.xa, m * k / 8)?,
        ];
        let ok = pair == fused;
        println!(
            "  [{}] {case}: fused SwiGLU + sums + int8 == the pair",
            verdict(ok)
        );
        fails += usize::from(!ok);
        let (o1, o2) = (Guarded::new(gpu, m * n * 2)?, Guarded::new(gpu, m * n * 2)?);
        prefill::run_packed(&[(down.dev, o1.ptr())], gpu, prod, m, &s1, true, 0)?;
        let ran = prefill::run_down_packed(&down.dev, gpu, gd, ud, k, h, k, m, o2.ptr(), &s2, 0)?;
        let same = ran && o1.read(gpu)? == o2.read(gpu)?;
        let fell_back = prefill::take_fallbacks(gpu, 0)?;
        let ok = same && fell_back == want_fb;
        println!(
            "  [{}] {case}: fused down projection == SwiGLU then the prompt GEMM \
             (fallbacks {fell_back}, want {want_fb})",
            verdict(ok)
        );
        fails += usize::from(!ok);
        for p in [gd, o1.base, o2.base] {
            gpu.free(p)?;
        }
        free_scratch(gpu, &s1)?;
        free_scratch(gpu, &s2)?;
    }
    down.free(gpu)?;
    for p in [ud, prod, h] {
        gpu.free(p)?;
    }
    Ok(fails)
}

fn timings(gpu: &dyn GpuBackend, rng: &mut Lcg) -> Result<()> {
    for (role, n, k) in [("gate", 9216, 2560), ("down", 2560, 9216)] {
        let w = weight(gpu, rng, n, k)?;
        let x: Vec<f32> = (0..MAX_M * k).map(|_| rng.bf16_normal(1.0)).collect();
        let xd = upload(gpu, &bf16_bytes(&x))?;
        let s = scratch(gpu, k)?;
        let out = gpu.alloc(MAX_M * n * 2)?;
        for m in [2048, MAX_M] {
            let tf = |us: f64| 2.0 * (m * n * k) as f64 / (us * 1e-6) / 1e12;
            let run = |int8: bool| {
                time_us(gpu, || {
                    prefill::run_packed(&[(w.dev, out)], gpu, xd, m, &s, int8, 0).map(|_| ())
                })
            };
            let (bf, i8) = (run(false)?, run(true)?);
            println!(
                "  timing {role} {n}x{k} M={m}: bf16 {bf:.0} us {:.1} TFLOP/s | int8 {i8:.0} us \
                 {:.1} TFLOP/s | {:.2}x",
                tf(bf),
                tf(i8),
                bf / i8
            );
        }
        w.free(gpu)?;
        gpu.free(xd)?;
        gpu.free(out)?;
        free_scratch(gpu, &s)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let gpu = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    ensure!(
        prefill::prepare(&gpu, 0)?,
        "affine prefill kernels absent: build with AVAROK_TARGET_HW=ada"
    );
    let mut rng = Lcg(0x17b0_2026_0000_0001);
    let mut fails = 0;
    for (n, k) in [(9216, 2560), (2560, 9216), (129, 2560)] {
        println!("shape {n}x{k}:");
        fails += shape(&gpu, &mut rng, n, k)?;
    }
    println!("many and exceptional:");
    fails += many_and_exceptional(&gpu, &mut rng)?;
    println!("fusion:");
    fails += fusion(&gpu, &mut rng)?;
    println!("timings:");
    timings(&gpu, &mut rng)?;
    ensure!(fails == 0, "{fails} row(s) failed");
    println!("all rows PASS");
    Ok(())
}
