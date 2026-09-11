// SPDX-License-Identifier: AGPL-3.0-only
//! The 5..16-row DECODE projections at Qwen3.8-27B shapes: the shipped
//! `w8a16_gemv_batch16[_strided]` tier against the W8A8 block-scaled cuBLASLt
//! arm this change adds (#927).
//!
//! WHY. H100, 2026-09-11 round 7, batch 16, steady-state n=16 decode step
//! **43.595 ms** (idle 4.2%), nsys `--cuda-graph-trace=node`. Resolved by grid
//! shape, these four projections are **21.85 ms = 50.1% of the step**: SSM
//! `in_proj_qkvz` (N=16384 K=5120) 48 x 235.3 µs = 11 294 µs at **357 GB/s**;
//! SSM `out_proj` + attn `o_proj` (N=5120) 64 x 106.5 µs = 6 814 µs; attn
//! `q_proj` (N=12288, strided) 16 x 180.8 µs = 2 893 µs; attn `k_proj`+
//! `v_proj` (N=1024, strided) 32 x 26.4 µs = 846 µs. The dense FFN, at the SAME
//! 16 rows in the SAME step, runs cuBLASLt W8A8 at ~128 µs/layer for 267 MB of
//! weights — ~2 100 GB/s-equivalent, which is the receipt for this change.
//!
//! It answers three questions per projection and guesses none:
//!
//!   1. NUMBERS. W8A8 quantizes the ACTIVATION to E4M3 per 128-wide K group,
//!      which the W8A16 GEMV does not, so the two are NOT bit-identical and no
//!      bit gate would be honest. The gate is cosine >= 0.999 and relative RMS
//!      <= 3e-2 against the GEMV, plus: nothing outside the route's extent
//!      (sentinel), nothing non-finite, gaps between Q|K|V untouched.
//!
//!      ⚠ THE FLOOR, so a marginal `rel_rms` is read correctly: E4M3's 3
//!      stored mantissa bits cost ~2.5% RMS relative error per element, and
//!      over a dot product of independent terms that does NOT average down
//!      relative to the signal. EXPECTED `rel_rms` is therefore ~2-2.6%: 3e-2
//!      is one notch of headroom over the floor, not a loose tolerance, and
//!      cosine (~0.9997 there) is the robust metric. Same floor as
//!      `native_fp8_ffn_w8a8_microtest`; `ATLAS_W8A8_REL_RMS_GATE` overrides.
//!
//!   2. THE PHANTOM ROWS. cuBLASLt is handed `ceil16(M) = 16` at every rung of
//!      this band and WRITES rows `m..16`; with the strided Q/K/V output those
//!      land in decode slots not in the step. This pins that they land THERE
//!      and nowhere else: finite, inside their own projection's columns,
//!      gaps still holding the sentinel.
//!
//!   3. TIME. Per projection, sync'd over `REPS`: µs per launch and the
//!      weight-bytes-per-second it implies, for both routes.
//!
//! Shapes: hidden 5120, head_dim 256, 24 q-heads / 4 kv-heads, output gate on
//! (`q_proj` = 12288 interleaved `[Q|gate]`, kv 1024, slot 14336 elements,
//! `o_proj` N=5120 over K=6144). SSM: fused QKVZ 16384 over K=5120, `out_proj`
//! N=5120 over K=6144 — `value_dim` off the round-7 trace, where SSM
//! `out_proj` and attn `o_proj` share the GrdX=1280 group at ~31.5 MB/launch.
//!
//! Run (H100): `ATLAS_TARGET_HW=hopper cargo run --release -p spark-model
//! --features cuda,gpu-examples --example native_fp8_decode_proj_w8a8_microtest`.
//! ★ The target matters: `fp8_scale_transpose.cu` is a HOPPER-owned source
//! (`[kernels] overrides`; b200 links it, gb10 has no copy), so a GB10 build
//! fails the `gpu.kernel` lookup by name — the tier does not exist there.
//! cuBLASLt is called directly, so `ATLAS_CUBLAS_GEMM` is not needed; the
//! serve spelling is printed at the end.

use anyhow::{Result, ensure};
use half::bf16;
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::time::Instant;

const H: usize = 5120;
const QKVZ_N: usize = 16384; // SSM fused in_proj_qkvz
const VALUE_DIM: usize = 6144; // SSM out_proj contract width
const Q_PROJ_DIM: usize = 12288; // gated [Q|gate]
const KV_DIM: usize = 1024;
const PER_SEQ_QKV: usize = Q_PROJ_DIM + 2 * KV_DIM; // 14336 BF16 elements
const O_K: usize = 6144; // q_heads * head_dim
const MAX_M: usize = 16; // the cuBLASLt M pad across the whole band
const ROWS: [usize; 3] = [5, 8, 16];
const REPS: usize = 20;
const GUARD: usize = 64; // sentinel bytes either side of every buffer
const SENTINEL: u8 = 0x5a;
const COSINE_GATE: f64 = 0.999;
const REL_RMS_GATE: f64 = 3e-2;

/// One projection under test. `ldc` is the BF16 elements between output ROWS
/// (equal to `n` when contiguous) and `offset` this projection's BF16 element
/// offset inside a row; `act` is `[MAX_M, k]` BF16.
struct Proj {
    name: &'static str,
    n: usize,
    k: usize,
    ldc: usize,
    offset: usize,
    weight: DevicePtr,
    scale: DevicePtr,
    act: DevicePtr,
}

impl Proj {
    fn strided(&self) -> bool {
        self.ldc != self.n
    }
    /// Live extents (byte offset, byte length) for `rows` output rows.
    fn live(&self, rows: usize) -> Vec<(usize, usize)> {
        (0..rows)
            .map(|r| ((r * self.ldc + self.offset) * 2, self.n * 2))
            .collect()
    }
    fn weight_bytes(&self) -> f64 {
        (self.n * self.k) as f64
    }
}

struct Kernels {
    batch16: KernelHandle,
    batch16_strided: KernelHandle,
    quant: KernelHandle,
    kmajor: KernelHandle,
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

/// Cosine and relative RMS over the live extents.
fn metrics(observed: &[u8], baseline: &[u8], live: &[(usize, usize)]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut se, mut ss) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(start, len) in live {
        let a = values(&observed[GUARD + start..GUARD + start + len]);
        let b = values(&baseline[GUARD + start..GUARD + start + len]);
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            na += x * x;
            nb += y * y;
            se += (x - y) * (x - y);
            ss += y * y;
        }
    }
    let cosine = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    };
    let rel_rms = if ss > 0.0 { (se / ss).sqrt() } else { 0.0 };
    (cosine, rel_rms)
}

/// The oracle. `live` is what the route is COMPARED on (rows 0..m); `written`
/// is everything it may TOUCH (rows 0..m_pad — cuBLASLt writes the phantom
/// rows). Outside `written` the sentinel must survive (the ldc/extent check);
/// inside it every value must be finite, phantom rows included.
fn check(
    observed: &[u8],
    baseline: &[u8],
    sentinel: &[u8],
    live: &[(usize, usize)],
    written: &[(usize, usize)],
    gate: f64,
) -> Result<()> {
    ensure!(
        observed.len() == sentinel.len() && baseline.len() == sentinel.len(),
        "output extent mismatch"
    );
    let mut mask = vec![false; sentinel.len()];
    for &(start, len) in written {
        mask[GUARD + start..GUARD + start + len].fill(true);
    }
    for i in 0..sentinel.len() {
        if !mask[i] {
            ensure!(
                observed[i] == sentinel[i],
                "route wrote outside its extent at byte {i}"
            );
        }
    }
    for &(start, len) in written {
        ensure!(
            values(&observed[GUARD + start..GUARD + start + len])
                .iter()
                .all(|x| x.is_finite()),
            "nonfinite projection output (phantom rows included)"
        );
    }
    let (cosine, rel_rms) = metrics(observed, baseline, live);
    ensure!(cosine >= COSINE_GATE, "cosine {cosine:.6} < {COSINE_GATE}");
    ensure!(rel_rms <= gate, "rel_rms {rel_rms:.4e} > {gate:.1e}");
    Ok(())
}

/// The shipped tier: ONE `w8a16_gemv_batch16[_strided]` launch, weights read
/// once, BF16 activations.
fn run_batch16(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    p: &Proj,
    out: DevicePtr,
    m: usize,
) -> Result<()> {
    if p.strided() {
        return ops::w8a16_gemv_batch16_strided(
            gpu,
            k.batch16_strided,
            p.act,
            p.weight,
            p.scale,
            out.offset(p.offset * 2),
            m as u32,
            p.n as u32,
            p.k as u32,
            p.k as u32,
            p.ldc as u32,
            0,
        );
    }
    ops::w8a16_gemv_batch16(
        gpu, k.batch16, p.act, p.weight, p.scale, out, m as u32, p.n as u32, p.k as u32, 0,
    )
}

/// The arm under test, exactly as `ops::decode_w8a8_quant_act` +
/// `ops::decode_w8a8_gemm` compose it in the serve.
fn run_w8a8(
    gpu: &dyn GpuBackend,
    scratch: &ops::DecodeW8a8Scratch,
    p: &Proj,
    out: DevicePtr,
    m: usize,
) -> Result<()> {
    ops::decode_w8a8_quant_act(gpu, scratch, p.act, m as u32, p.k as u32, 0)?;
    let plan = if p.strided() {
        ops::DecodeW8a8Plan::strided(m, p.n as u32, p.k as u32, p.ldc as u32, usize::MAX)
    } else {
        ops::DecodeW8a8Plan::contiguous(m, p.n as u32, p.k as u32, usize::MAX)
    };
    let w = spark_model::weight_map::Fp8Weight {
        weight: p.weight,
        row_scale: p.scale,
        n: p.n as u32,
        k: p.k as u32,
        scale_format: spark_model::weight_map::WeightQuantFormat::Fp8BlockScaled,
    };
    ops::decode_w8a8_gemm(scratch, &w, out.offset(p.offset * 2), &plan, 0)
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let k = Kernels {
        batch16: gpu.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch16")?,
        batch16_strided: gpu.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch16_strided")?,
        quant: gpu.kernel("per_token_group_quant_fp8", "per_token_group_quant_fp8")?,
        kmajor: gpu.kernel("fp8_scale_transpose", "fp8_act_scale_to_kmajor")?,
    };
    let gate: f64 = std::env::var("ATLAS_W8A8_REL_RMS_GATE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(REL_RMS_GATE);
    println!(
        "gates: cosine >= {COSINE_GATE}, rel_rms <= {gate:.1e} \
         (E4M3 activation-quant floor is ~2-2.6e-2; see the module header)"
    );

    // A struct, not a nest of closures: three generators sharing one `random`
    // closure would each hold a mutable borrow of it, and this file should not
    // depend on the order in which they happen to be last-used.
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
    let mut rng = Rng(0x0927_2026_5A5A_0001);

    // Two activations: `[MAX_M, H]` feeds qkvz and q/k/v, `[MAX_M, 6144]`
    // feeds the SSM out_proj and the attention o_proj.
    let act_h = upload(&gpu, &rng.acts(MAX_M * H))?;
    let act_v = upload(&gpu, &rng.acts(MAX_M * O_K))?;

    let mk = |rng: &mut Rng, name, n: usize, kk: usize, ldc: usize, offset, act| -> Result<Proj> {
        Ok(Proj {
            name,
            n,
            k: kk,
            ldc,
            offset,
            weight: upload(&gpu, &rng.fp8(n, kk))?,
            scale: upload(&gpu, &rng.scales(n, kk))?,
            act,
        })
    };
    // Each group shares one output buffer, so the strided gaps and the
    // neighbouring projections' slots are real neighbours and not padding.
    let ssm_qkvz = mk(&mut rng, "ssm in_proj_qkvz", QKVZ_N, H, QKVZ_N, 0, act_h)?;
    let ssm_out = mk(&mut rng, "ssm out_proj", H, VALUE_DIM, H, 0, act_v)?;
    let q = mk(
        &mut rng,
        "attn q_proj",
        Q_PROJ_DIM,
        H,
        PER_SEQ_QKV,
        0,
        act_h,
    )?;
    let kp = mk(
        &mut rng,
        "attn k_proj",
        KV_DIM,
        H,
        PER_SEQ_QKV,
        Q_PROJ_DIM,
        act_h,
    )?;
    let v = mk(
        &mut rng,
        "attn v_proj",
        KV_DIM,
        H,
        PER_SEQ_QKV,
        Q_PROJ_DIM + KV_DIM,
        act_h,
    )?;
    let o = mk(&mut rng, "attn o_proj", H, O_K, H, 0, act_v)?;
    let projections = [&ssm_qkvz, &ssm_out, &q, &kp, &v, &o];

    // One output arena per row-pitch, sized for the FULL padded row count.
    let mut arenas: Vec<(usize, Vec<u8>, DevicePtr)> = Vec::new();
    for ldc in [QKVZ_N, H, PER_SEQ_QKV] {
        let sentinel = vec![SENTINEL; MAX_M * ldc * 2 + 2 * GUARD];
        let base = upload(&gpu, &sentinel)?;
        arenas.push((ldc, sentinel, base));
    }
    let arena = |ldc: usize| arenas.iter().find(|a| a.0 == ldc).expect("arena");

    // The activation-quant scratch, sized exactly as the serve's arena is.
    let kg = H.max(O_K) / 128;
    let scratch = ops::DecodeW8a8Scratch {
        act_fp8: gpu.alloc(MAX_M * H.max(O_K))?,
        act_fp8_bytes: MAX_M * H.max(O_K),
        act_scale: gpu.alloc(MAX_M * kg * 4)?,
        act_scale_bytes: MAX_M * kg * 4,
        act_scale_kmajor: gpu.alloc(MAX_M * kg * 4)?,
        act_scale_kmajor_bytes: MAX_M * kg * 4,
        quant_k: k.quant,
        scale_kmajor_k: k.kmajor,
    };

    let mut failures = 0usize;
    let mut controls_done = false;
    for m in ROWS {
        println!("\n=== M = {m} (cuBLASLt pad = {MAX_M}) ===");
        for p in projections {
            let (ldc, sentinel, base) = arena(p.ldc);
            let (ldc, base) = (*ldc, *base);
            let out = base.offset(GUARD);
            let live = p.live(m);
            // The W8A8 route may touch rows 0..MAX_M; the GEMV only 0..m.
            let padded = p.live(MAX_M);

            let capture = |w8a8: bool| -> Result<Vec<u8>> {
                gpu.copy_h2d(sentinel, base)?;
                if w8a8 {
                    run_w8a8(&gpu, &scratch, p, out, m)?;
                } else {
                    run_batch16(&gpu, &k, p, out, m)?;
                }
                gpu.synchronize(0)?;
                let mut host = vec![0_u8; sentinel.len()];
                gpu.copy_d2h(base, &mut host)?;
                Ok(host)
            };
            let reference = capture(false)?;
            let observed = capture(true)?;

            if !controls_done && p.strided() {
                // A green run has to be able to go red: four corruptions of
                // the OBSERVED buffer, all refused by the real oracle.
                for control in ["gap", "guard", "nonfinite", "value"] {
                    let mut bad = observed.clone();
                    match control {
                        // A byte in the last slot's tail, past every live
                        // column of a strided row — an ldc that is too large.
                        "gap" => bad[GUARD + MAX_M * ldc * 2 - 2] ^= 1,
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
                    let err = check(&bad, &reference, sentinel, &live, &padded, gate)
                        .expect_err("known-bad output was admitted by the real oracle");
                    println!("KNOWN_BAD {control}: refused: {err}");
                }
                controls_done = true;
            }

            let (cosine, rel_rms) = metrics(&observed, &reference, &live);
            let phantom = if m < MAX_M {
                let rows: Vec<_> = (m..MAX_M)
                    .map(|r| ((r * p.ldc + p.offset) * 2, p.n * 2))
                    .collect();
                let vals: Vec<f64> = rows
                    .iter()
                    .flat_map(|&(s, l)| values(&observed[GUARD + s..GUARD + s + l]))
                    .collect();
                format!(
                    " phantom_rows={}..{} finite={} max_abs={:.3}",
                    m,
                    MAX_M,
                    vals.iter().all(|x| x.is_finite()),
                    vals.iter().fold(0.0_f64, |a, x| a.max(x.abs()))
                )
            } else {
                String::new()
            };
            println!(
                "  {:<18} N={:<5} K={:<5} {:<10} cosine={cosine:.6} rel_rms={rel_rms:.4e}{phantom}",
                p.name,
                p.n,
                p.k,
                if p.strided() { "strided" } else { "contiguous" },
            );
            if let Err(e) = check(&observed, &reference, sentinel, &live, &padded, gate) {
                println!("  FAIL {} M={m}: {e}", p.name);
                failures += 1;
            }
        }

        println!("  --- timing, {REPS} reps per projection ---");
        for p in projections {
            let out = arena(p.ldc).2.offset(GUARD);
            for (label, w8a8) in [("batch16", false), ("W8A8-cuBLASLt", true)] {
                if w8a8 {
                    run_w8a8(&gpu, &scratch, p, out, m)?;
                } else {
                    run_batch16(&gpu, &k, p, out, m)?;
                }
                gpu.synchronize(0)?;
                let t0 = Instant::now();
                for _ in 0..REPS {
                    if w8a8 {
                        run_w8a8(&gpu, &scratch, p, out, m)?;
                    } else {
                        run_batch16(&gpu, &k, p, out, m)?;
                    }
                }
                gpu.synchronize(0)?;
                let us = t0.elapsed().as_secs_f64() * 1e6 / REPS as f64;
                // "GB/s-equivalent": weight bytes read once over the wall
                // time. NOT a bandwidth claim for the W8A8 route — that is
                // compute-bound tensor-core work and can exceed HBM peak. It
                // is the unit the round-7 table uses (357 GB/s for
                // `in_proj_qkvz`), so both routes are reported in it.
                println!(
                    "  {:<18} {:<14} {us:>9.1} us  {:>7.0} GB/s-equiv",
                    p.name,
                    label,
                    p.weight_bytes() / (us / 1e6) / 1e9
                );
            }
        }
    }

    ensure!(failures == 0, "{failures} case(s) failed");
    println!(
        "\nALL PASS: Qwen3.8-27B decode projections at M in {ROWS:?} — W8A8 cuBLASLt within \
         cosine {COSINE_GATE} / rel_rms {gate:.1e} of the batch16 GEMV, strided gaps intact, \
         phantom rows finite and confined to their own slots.\n\
         Serve spelling: ATLAS_CUBLAS_GEMM=ffn,ssm,attn (ATLAS_NO_W8A8_DECODE_PROJ reverts)."
    );
    Ok(())
}
