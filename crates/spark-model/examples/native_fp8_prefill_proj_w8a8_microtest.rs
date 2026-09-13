// SPDX-License-Identifier: AGPL-3.0-only
//! The PREFILL projections that round 9 left on W8A16, at Qwen3.8-27B shapes:
//! the shipped `w8a16_gemm_pipelined` / `w8a16_gemm_t_m128` kernels against the
//! W8A8 block-scaled cuBLASLt arm this change adds (#917/#928).
//!
//! WHY. nsys on 1xH100, 2026-09-11 round 9, `Qwen/Qwen3.8-27B-FP8`, recipe
//! `ATLAS_CUBLAS_GEMM=ffn,ssm,attn` (`nsys-r9-prefill`). A 1193-token prefill
//! is **368.263 ms** of GPU-busy union. `w8a16_gemm_pipelined` is **100.582 ms
//! = 27.31%** over **112 launches** and `w8a16_gemm_t_m128` a further
//! **12.012 ms** over 32. The kernels' launchers make the grid a shape decoder
//! (`(ceil(N/32), ceil(M/128))` and `(ceil(N/128), ceil(M/128))`), and the
//! 1193-token prompt is served as two chunks, 1168 + 25:
//!
//! | launches | grid | shape | site | µs |
//! |---|---|---|---|---|
//! | 48 | 160x10 | N=5120 K=6144 M=1168 | SSM `out_proj`, chunk 0 | 54 119.7 |
//! | 16 | 384x10 | N=12288 K=5120 M=1168 | attn `q_proj`, chunk 0 | 32 466.8 |
//! | 48 | 160x1 | N=5120 K=6144 M=25 | SSM `out_proj`, chunk 1 | 13 995.4 |
//! | 32 | 8x10 | N=1024 K=5120 M=1168 | attn `k_proj`+`v_proj`, chunk 0 | 12 012.2 |
//!
//! Those are the four shapes below. (The dense FFN is NOT among them: the same
//! trace has gate/up/down on `nvjet…` cuBLASLt lines at every M.)
//!
//! It answers three questions per projection and guesses none:
//!
//!   1. NUMBERS. W8A8 quantizes the ACTIVATION to E4M3 per 128-wide K group,
//!      which the W8A16 kernels do not, so the two are NOT bit-identical and no
//!      bit gate would be honest. The gate is cosine >= 0.999 and relative RMS
//!      <= 3e-2 against the W8A16 reference, plus: nothing outside the route's
//!      extent (sentinel), nothing non-finite.
//!
//!      ⚠ THE FLOOR, so a marginal `rel_rms` is read correctly: E4M3 carries
//!      3 stored mantissa bits, so round-to-nearest costs ~2.5% RMS relative
//!      error per element and that error does NOT average down over a dot
//!      product of independent terms. The EXPECTED `rel_rms` is ~2-2.6e-2 —
//!      3e-2 is one notch of headroom over the floor, not a loose tolerance,
//!      and cosine is the robust metric. Same floor the FFN and decode-proj
//!      microtests state. `ATLAS_W8A8_REL_RMS_GATE` overrides it.
//!
//!   2. THE PHANTOM ROWS. cuBLASLt is handed `ceil16(M)` and WRITES rows
//!      `M..ceil16(M)`. At M=1193 that is 1200, i.e. 7 rows past the real
//!      output — the extent every capacity clause in
//!      `qwen3_ssm/prefill_out_w8a8.rs` and
//!      `qwen3_attention/prefill_qkv_w8a8.rs` is written against. This pins
//!      that they land THERE and nowhere else: finite, inside the padded
//!      extent, sentinel intact beyond it.
//!
//!   3. TIME. Per projection, sync'd over `REPS`: µs per launch and the
//!      TFLOP/s it implies, for both routes.
//!
//! Run (H100): `cargo run --release -p spark-model --features cuda,gpu-examples
//! --example native_fp8_prefill_proj_w8a8_microtest`. The example calls
//! cuBLASLt directly, so `ATLAS_CUBLAS_GEMM` is not required here — the serve
//! spelling is printed at the end for copy-paste.

use anyhow::{Result, ensure};
use half::bf16;
use spark_model::layers::ops;
use spark_model::weight_map::{Fp8Weight, WeightQuantFormat};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::time::Instant;

const H: usize = 5120; // hidden
const VALUE_DIM: usize = 6144; // GDN out_proj contract width
const Q_PROJ_DIM: usize = 12288; // gated [Q|gate]
const KV_DIM: usize = 1024;
/// Chunk 0 of the round-9 trace prompt is 1168; 1193 is the whole prompt and is
/// deliberately NOT a multiple of 16, so the phantom rows are exercised.
const ROWS: [usize; 2] = [64, 1193];
const REPS: usize = 10;
const GUARD: usize = 64;
const SENTINEL: u8 = 0x5a;
const COSINE_GATE: f64 = 0.999;
const REL_RMS_GATE: f64 = 3e-2;

/// Which W8A16 kernel a projection falls back to in the serve — the reference
/// this compares and times against.
#[derive(Clone, Copy, PartialEq)]
enum Ref {
    /// `w8a16_gemm_pipelined` over the row-major `[N, K]` weight.
    Pipelined,
    /// `w8a16_gemm_n128_m128` over the transposed `[K, N]` twin.
    TransposedM128,
}

struct Proj {
    name: &'static str,
    n: usize,
    k: usize,
    reference: Ref,
    weight: DevicePtr,
    scale: DevicePtr,
    weight_t: DevicePtr,
    scale_t: DevicePtr,
    act: DevicePtr,
}

impl Proj {
    fn flops(&self, m: usize) -> f64 {
        2.0 * m as f64 * self.n as f64 * self.k as f64
    }
    fn fp8w(&self) -> Fp8Weight {
        Fp8Weight {
            weight: self.weight,
            row_scale: self.scale,
            n: self.n as u32,
            k: self.k as u32,
            scale_format: WeightQuantFormat::Fp8BlockScaled,
        }
    }
}

struct Kernels {
    pipelined: KernelHandle,
    t_m128: KernelHandle,
    quant: KernelHandle,
    kmajor: KernelHandle,
}

struct Scratch {
    act_fp8: DevicePtr,
    act_scale: DevicePtr,
    act_scale_kmajor: DevicePtr,
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn values(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(2)
        .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32() as f64)
        .collect()
}

fn metrics(observed: &[u8], baseline: &[u8], live: usize) -> (f64, f64) {
    let a = values(&observed[GUARD..GUARD + live]);
    let b = values(&baseline[GUARD..GUARD + live]);
    let (mut dot, mut na, mut nb, mut se, mut ss) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
        se += (x - y) * (x - y);
        ss += y * y;
    }
    let cosine = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    };
    let rel_rms = if ss > 0.0 { (se / ss).sqrt() } else { 0.0 };
    (cosine, rel_rms)
}

/// `live` bytes are what the routes are COMPARED on (rows 0..M); `written`
/// bytes are everything the W8A8 route may TOUCH (rows 0..ceil16(M)). Beyond
/// `written` the sentinel must survive.
fn check(
    observed: &[u8],
    baseline: &[u8],
    sentinel: &[u8],
    live: usize,
    written: usize,
    gate: f64,
) -> Result<()> {
    ensure!(
        observed.len() == sentinel.len() && baseline.len() == sentinel.len(),
        "output extent mismatch"
    );
    for i in 0..sentinel.len() {
        if !(GUARD..GUARD + written).contains(&i) {
            ensure!(
                observed[i] == sentinel[i],
                "route wrote outside its extent at byte {i}"
            );
        }
    }
    ensure!(
        values(&observed[GUARD..GUARD + written])
            .iter()
            .all(|x| x.is_finite()),
        "nonfinite projection output (phantom rows included)"
    );
    let (cosine, rel_rms) = metrics(observed, baseline, live);
    ensure!(cosine >= COSINE_GATE, "cosine {cosine:.6} < {COSINE_GATE}");
    ensure!(rel_rms <= gate, "rel_rms {rel_rms:.4e} > {gate:.1e}");
    Ok(())
}

/// The shipped W8A16 arm for this projection — BF16 activations straight in.
fn run_w8a16(gpu: &dyn GpuBackend, k: &Kernels, p: &Proj, out: DevicePtr, m: usize) -> Result<()> {
    match p.reference {
        Ref::Pipelined => ops::w8a16_gemm_pipelined(
            gpu,
            k.pipelined,
            p.act,
            p.weight,
            p.scale,
            out,
            m as u32,
            p.n as u32,
            p.k as u32,
            0,
        ),
        Ref::TransposedM128 => ops::w8a16_gemm_n128_m128(
            gpu, k.t_m128, p.act, p.weight_t, p.scale_t, out, m as u32, p.n as u32, p.k as u32, 0,
        ),
    }
}

/// The arm under test, exactly as the two new prefill sites compose it:
/// `per_token_group_quant_fp8` once, then `cublas_fp8_proj_prequant`.
fn run_w8a8(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    s: &Scratch,
    p: &Proj,
    out: DevicePtr,
    m: usize,
) -> Result<()> {
    ops::per_token_group_quant_fp8(
        gpu,
        k.quant,
        p.act,
        s.act_fp8,
        s.act_scale,
        m as u32,
        p.k as u32,
        0,
    )?;
    ops::cublas_fp8_proj_prequant(
        gpu,
        k.kmajor,
        s.act_fp8,
        s.act_scale,
        s.act_scale_kmajor,
        &p.fp8w(),
        out,
        m as u32,
        p.n as u32,
        p.k as u32,
        0,
    )
}

/// `[N, K]` FP8 -> `[K, N]`, and `[N/128, K/128]` FP32 -> `[K/128, N/128]`.
/// The serve builds these once at load (`prefill_weights.rs`); here they exist
/// so the `k_proj`/`v_proj` timing is against the kernel the serve really runs.
fn transpose(weight: &[u8], scale: &[u8], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let mut wt = vec![0u8; n * k];
    for row in 0..n {
        for col in 0..k {
            wt[col * n + row] = weight[row * k + col];
        }
    }
    let (sn, sk) = (n / 128, k / 128);
    let mut st = vec![0u8; sn * sk * 4];
    for row in 0..sn {
        for col in 0..sk {
            let src = (row * sk + col) * 4;
            let dst = (col * sn + row) * 4;
            st[dst..dst + 4].copy_from_slice(&scale[src..src + 4]);
        }
    }
    (wt, st)
}

struct Rng(u64);
impl Rng {
    fn bits(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
    /// FP8 E4M3 bytes, sign kept, exponents clipped off the NaN encodings.
    fn fp8(&mut self, n: usize, depth: usize) -> Vec<u8> {
        (0..n * depth)
            .map(|_| {
                let x = self.bits();
                ((x % 127) as u8) | (((x >> 7) & 1) as u8 * 128)
            })
            .collect()
    }
    /// One FP32 scale per 128x128 weight block.
    fn scales(&mut self, n: usize, depth: usize) -> Vec<u8> {
        (0..(n / 128) * (depth / 128))
            .flat_map(|_| (((self.bits() % 16 + 1) as f32) / 1024.0).to_le_bytes())
            .collect()
    }
    fn acts(&mut self, elems: usize) -> Vec<u8> {
        (0..elems)
            .flat_map(|_| {
                bf16::from_f32(((self.bits() % 2049) as f32 - 1024.0) / 1024.0)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect()
    }
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let k = Kernels {
        pipelined: gpu.kernel("w8a16_gemm_pipelined", "w8a16_gemm_pipelined")?,
        t_m128: gpu.kernel("w8a16_gemm_t_m128", "w8a16_gemm_t_m128")?,
        quant: gpu.kernel("per_token_group_quant_fp8", "per_token_group_quant_fp8")?,
        kmajor: gpu.kernel("fp8_scale_transpose", "fp8_act_scale_to_kmajor")?,
    };
    let gate: f64 = std::env::var("ATLAS_W8A8_REL_RMS_GATE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(REL_RMS_GATE);
    let max_m = ops::cublas_fp8_m_pad(*ROWS.iter().max().unwrap() as u32) as usize;
    println!(
        "gates: cosine >= {COSINE_GATE}, rel_rms <= {gate:.1e} \
         (E4M3 activation-quant floor is ~2-2.6e-2; see the module header)\n\
         rows {ROWS:?}, cuBLASLt M pad reaches {max_m}"
    );

    let mut rng = Rng(0x0928_2026_5A5A_0001);
    let act_h = upload(&gpu, &rng.acts(max_m * H))?;
    let act_v = upload(&gpu, &rng.acts(max_m * VALUE_DIM))?;

    let mut mk = |name, n: usize, kk: usize, reference, act| -> Result<Proj> {
        let w = rng.fp8(n, kk);
        let s = rng.scales(n, kk);
        let (wt, st) = if reference == Ref::TransposedM128 {
            transpose(&w, &s, n, kk)
        } else {
            (vec![0u8; 1], vec![0u8; 4])
        };
        Ok(Proj {
            name,
            n,
            k: kk,
            reference,
            weight: upload(&gpu, &w)?,
            scale: upload(&gpu, &s)?,
            weight_t: upload(&gpu, &wt)?,
            scale_t: upload(&gpu, &st)?,
            act,
        })
    };
    let projections = [
        mk("ssm out_proj", H, VALUE_DIM, Ref::Pipelined, act_v)?,
        mk("attn q_proj", Q_PROJ_DIM, H, Ref::Pipelined, act_h)?,
        mk("attn k_proj", KV_DIM, H, Ref::TransposedM128, act_h)?,
        mk("attn v_proj", KV_DIM, H, Ref::TransposedM128, act_h)?,
    ];

    let kg = H.max(VALUE_DIM) / 128;
    let scratch = Scratch {
        act_fp8: gpu.alloc(max_m * H.max(VALUE_DIM))?,
        act_scale: gpu.alloc(max_m * kg * 4)?,
        act_scale_kmajor: gpu.alloc(max_m * kg * 4)?,
    };

    let mut failures = 0usize;
    let mut controls_done = false;
    for m in ROWS {
        let m_pad = ops::cublas_fp8_m_pad(m as u32) as usize;
        println!("\n=== M = {m} (cuBLASLt pad = {m_pad}) ===");
        for p in &projections {
            let written = m_pad * p.n * 2;
            let live = m * p.n * 2;
            let sentinel = vec![SENTINEL; written + 2 * GUARD];
            let base = upload(&gpu, &sentinel)?;
            let out = base.offset(GUARD);

            let capture = |w8a8: bool| -> Result<Vec<u8>> {
                gpu.copy_h2d(&sentinel, base)?;
                if w8a8 {
                    run_w8a8(&gpu, &k, &scratch, p, out, m)?;
                } else {
                    run_w8a16(&gpu, &k, p, out, m)?;
                }
                gpu.synchronize(0)?;
                let mut host = vec![0_u8; sentinel.len()];
                gpu.copy_d2h(base, &mut host)?;
                Ok(host)
            };
            let reference = capture(false)?;
            let observed = capture(true)?;

            if !controls_done {
                // A green run has to be able to go red: three corruptions of
                // the OBSERVED buffer, all refused by the real oracle.
                for control in ["guard", "nonfinite", "value"] {
                    let mut bad = observed.clone();
                    match control {
                        "guard" => bad[0] ^= 1,
                        "nonfinite" => {
                            bad[GUARD..GUARD + 2].copy_from_slice(&0x7fc0_u16.to_le_bytes())
                        }
                        _ => {
                            for x in bad[GUARD..GUARD + p.n * 2].chunks_exact_mut(2) {
                                x.copy_from_slice(&bf16::from_f32(1.0e3).to_bits().to_le_bytes());
                            }
                        }
                    }
                    let err = check(&bad, &reference, &sentinel, live, written, gate)
                        .expect_err("known-bad output was admitted by the real oracle");
                    println!("KNOWN_BAD {control}: refused: {err}");
                }
                controls_done = true;
            }

            let (cosine, rel_rms) = metrics(&observed, &reference, live);
            let phantom = if m_pad > m {
                let vals = values(&observed[GUARD + live..GUARD + written]);
                format!(
                    " phantom_rows={m}..{m_pad} finite={} max_abs={:.3}",
                    vals.iter().all(|x| x.is_finite()),
                    vals.iter().fold(0.0_f64, |a, x| a.max(x.abs()))
                )
            } else {
                String::new()
            };
            println!(
                "  {:<14} N={:<5} K={:<5} ref={:<10} cosine={cosine:.6} rel_rms={rel_rms:.4e}{phantom}",
                p.name,
                p.n,
                p.k,
                match p.reference {
                    Ref::Pipelined => "pipelined",
                    Ref::TransposedM128 => "t_m128",
                },
            );
            if let Err(e) = check(&observed, &reference, &sentinel, live, written, gate) {
                println!("  FAIL {} M={m}: {e}", p.name);
                failures += 1;
            }

            println!("  --- timing, {REPS} reps ---");
            for (label, w8a8) in [("W8A16 (shipped)", false), ("W8A8-cuBLASLt", true)] {
                if w8a8 {
                    run_w8a8(&gpu, &k, &scratch, p, out, m)?;
                } else {
                    run_w8a16(&gpu, &k, p, out, m)?;
                }
                gpu.synchronize(0)?;
                let t0 = Instant::now();
                for _ in 0..REPS {
                    if w8a8 {
                        run_w8a8(&gpu, &k, &scratch, p, out, m)?;
                    } else {
                        run_w8a16(&gpu, &k, p, out, m)?;
                    }
                }
                gpu.synchronize(0)?;
                let us = t0.elapsed().as_secs_f64() * 1e6 / REPS as f64;
                println!(
                    "  {:<14} {:<16} {us:>9.1} us  {:>7.1} TFLOP/s",
                    p.name,
                    label,
                    p.flops(m) / (us / 1e6) / 1e12
                );
            }
        }
    }

    ensure!(failures == 0, "{failures} case(s) failed");
    println!(
        "\nALL PASS: Qwen3.8-27B prefill projections at M in {ROWS:?} — W8A8 cuBLASLt within \
         cosine {COSINE_GATE} / rel_rms {gate:.1e} of the W8A16 reference, phantom rows finite \
         and confined to ceil16(M).\n\
         Serve spelling: ATLAS_CUBLAS_GEMM=ffn,ssm,attn \
         (ATLAS_SSM_OUT_W8A16_ONLY / ATLAS_ATTN_QKV_W8A16_ONLY revert per site)."
    );
    Ok(())
}
