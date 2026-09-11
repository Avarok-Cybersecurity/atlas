// SPDX-License-Identifier: AGPL-3.0-only
//! Dense-FFN W8A8 block-scaled prefill at the real Qwen3.8-27B shapes (#917/#928).
//!
//! WHY. The native-FP8 dense FFN ran its prefill GEMMs as W8A16 — BF16
//! activations against E4M3 weights, so the MMA is the BF16 tensor-core path.
//! On H100 (2026-09-11, 1193-token prompt) that put TTFT at 1075 ms against
//! vLLM's 287 ms, with `w8a16_gemm_pipelined` turning ~12 TFLOP/s on these
//! shapes. This microtest measures the replacement: the same W8A8 block-scaled
//! arithmetic the attention projections already use, with cuBLASLt as the
//! Hopper fast path.
//!
//! What it reports, per shape and per M:
//!   * W8A8 (kernel) vs W8A16 — max_abs, cosine, relative RMS. W8A8 quantizes
//!     the ACTIVATION to E4M3 per 128-wide K group, which W8A16 does not; the
//!     difference is vLLM's dynamic W8A8 arithmetic and a deliberate precision
//!     trade, not a defect.
//!   * cuBLASLt vs the W8A8 kernel — same quantized inputs, same FP32
//!     epilogue, so the only licensed difference is FP32 accumulation ORDER.
//!     Accepted per element at one BF16 rounding step (see
//!     `CUBLAS_SMALL_MAGNITUDE`); `over_1ulp`, `sign_flips`, `unequal_bf16` and
//!     the ordinal `max_ulp` are all printed so "close" is a number.
//!   * TFLOP/s for each path (CUDA events, 10 iterations after warm-up).
//!
//! NUMERICS FLOOR — read before judging a marginal `rel_rms`. E4M3 carries 3
//! stored mantissa bits, so round-to-nearest costs ~2.5% RMS relative error per
//! element, and for a dot product of independent terms that error does NOT
//! average down relative to the signal: the expected `rel_rms` of this
//! comparison on random inputs is ~2-2.6%, sitting right on the 2% gate.
//! Cosine is the robust metric (~0.9997 at that error). `ATLAS_W8A8_REL_RMS_GATE`
//! overrides the bound for a measurement run; the value used is always printed.
//!
//! SCALE LAYOUT — `ATLAS_CUBLAS_SCALE_LAYOUT=kmajor|rowmajor` (default
//! `kmajor`). cuBLASLt reads the VEC128 activation scales with the TOKEN index
//! contiguous, the transpose of the `[M, K/128]` the quantizer writes; the
//! `rowmajor` setting feeds the untransposed buffer, which is the reading that
//! measured rel_rms 7.7e-2 / ~33 000 BF16 ULP on H100 on 2026-09-11. It is kept
//! so both readings can be shown on one box, and it is EXPECTED TO FAIL.
//!
//! Run (H100):
//!   cargo run --release -p spark-model --features cuda,gpu-examples \
//!     --example native_fp8_ffn_w8a8_microtest
//!   ATLAS_CUBLAS_GEMM=1 cargo run --release -p spark-model \
//!     --features cuda,gpu-examples --example native_fp8_ffn_w8a8_microtest
//!   ATLAS_CUBLAS_GEMM=1 ATLAS_CUBLAS_SCALE_LAYOUT=rowmajor cargo run --release \
//!     -p spark-model --features cuda,gpu-examples \
//!     --example native_fp8_ffn_w8a8_microtest

use anyhow::{Result, bail};
use half::bf16;
use spark_model::layers::ops;
use spark_model::weight_map::{Fp8Weight, WeightQuantFormat};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

// CUDA driver event API — kernel-only timing. Wall-clock `Instant` carries a
// ~0.3 ms per-launch host floor that swamps the signal on these shapes.
// Signatures mirror `examples/w8a16_microtest.rs` (SSOT for these decls).
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// Qwen3.8-27B dense FFN: hidden 5120, intermediate 17408.
const H: usize = 5120;
const INTER: usize = 17408;
/// 64 = a short prefill chunk; 1193 = the prompt length in the #917 H100 trace.
const BATCHES: [usize; 2] = [64, 1193];
const BLOCK: usize = 128;
const ITERS: u32 = 10;
const WARMUP: u32 = 3;
const COSINE_GATE: f64 = 0.999;
const REL_RMS_GATE: f64 = 0.02;
/// cuBLASLt-vs-kernel acceptance. The two implementations consume the SAME FP8
/// bytes and the SAME FP32 scales, so the only licensed difference between them
/// is the ORDER of the FP32 accumulation (tile shape, split-K, epilogue
/// association). That is a real difference and it has a bounded consequence:
/// each FP32 partial sum moves by a few parts in 1e7, which is far under one
/// BF16 step (2^-8 relative), so an output lands on a different BF16 value only
/// when the exact sum sits within that sliver of a rounding boundary — and then
/// it moves by exactly ONE step, never two. A layout bug does not look like
/// that; it permutes scales, which moves elements by whole factors.
///
/// Hence the acceptance below is a **tolerance on every element**, not a
/// bit-identity check:
///
///   * `|a - b| <= 1 BF16 ULP at max(|a|, |b|)` — one rounding step, measured
///     at the larger operand's magnitude so a pair straddling a binade boundary
///     is judged by the coarser grid it actually shares.
///   * OR both values are under [`CUBLAS_SMALL_MAGNITUDE`]. Near zero the
///     output is what is left after cancellation between O(K) terms of much
///     larger magnitude, so its remaining bits — including its SIGN — are set
///     by accumulation order and carry no information. 0.05 is ~1.7x the
///     largest sign-flipped magnitude measured on H100 (0.0148, 0.0167, 0.0237,
///     0.0297 on 2026-09-11) and ~2e-4 of the shapes' output range, i.e. below
///     anything the downstream SiLU/BF16 store can distinguish.
///
/// The count of elements outside that bound is the gate and must be **0**.
/// `sign_flips`, `unequal_bf16` and `max_ulp` are REPORTED, not gated.
///
/// WHY `max_ulp` IS NO LONGER A GATE. The previous bound was `max_ulp <= 2` on
/// an ORDINAL ULP distance — `|ord(a) - ord(b)|` over sign-magnitude BF16 — and
/// that metric scores a sign flip on a value of magnitude `v` as `2 * ord(v)`,
/// a number in the tens of thousands however tiny `v` is. The 2026-09-11 H100
/// run measured exactly that failure mode: cosine 0.9999996, rel_rms
/// 9.1e-4-9.3e-4, `max_abs` equal to one BF16 ULP at the largest outputs
/// (1.000 and 2.000), and `max_ulp ~= 31 000` decoding to sign flips at
/// |v| <= 0.03. A bound no correctly-ordered FP32 accumulation can satisfy is
/// a bit-identity gate wearing a tolerance's clothes; this is the tolerance it
/// was pretending to be.
///
/// `rel_rms` moves 1e-3 -> 2e-3 for headroom over the same run's 9.29e-4 while
/// staying an order of magnitude under the ~2.5e-2 E4M3 floor the W8A16
/// comparison sits on — which is what keeps these a LAYOUT test and not a
/// precision test. The row-major control fails all three: rel_rms 1.1e-2-8.8e-2,
/// cosine down to 0.9961, and `max_abs` 55.6 against a one-ULP bound of 2.
const CUBLAS_SMALL_MAGNITUDE: f64 = 0.05;
const CUBLAS_COSINE_GATE: f64 = 0.99999;
const CUBLAS_REL_RMS_GATE: f64 = 2e-3;

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next_u32() % 2049) as f32 / 1024.0 - 1.0
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn download_bf16(gpu: &dyn GpuBackend, ptr: DevicePtr, elems: usize) -> Result<Vec<u16>> {
    let mut raw = vec![0u8; elems * 2];
    gpu.copy_d2h(ptr, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn to_f64(bits: &[u16]) -> Vec<f64> {
    bits.iter()
        .map(|b| bf16::from_bits(*b).to_f32() as f64)
        .collect()
}

/// BF16 bits → a monotonically ordered integer, so `|ord(a) - ord(b)|` is the
/// ULP distance (the standard sign-magnitude → two's-complement remap).
fn ord(bits: u16) -> i32 {
    if bits & 0x8000 != 0 {
        -((bits & 0x7fff) as i32)
    } else {
        bits as i32
    }
}

/// One BF16 ULP at `bits`' magnitude — the gap between it and its neighbour.
/// BF16 stores 7 mantissa bits, so a normal value with unbiased exponent `e`
/// steps by `2^(e - 7)`; every subnormal steps by the smallest normal's step,
/// `2^-133`. Inf/NaN report an infinite step so nothing is judged "close" to
/// them by arithmetic accident (the `<=` below still rejects a NaN difference).
fn bf16_ulp(bits: u16) -> f64 {
    let biased_exp = ((bits >> 7) & 0xff) as i32;
    match biased_exp {
        0xff => f64::INFINITY,
        0 => (-133.0_f64).exp2(),
        e => ((e - 127 - 7) as f64).exp2(),
    }
}

/// The per-element acceptance described on [`CUBLAS_SMALL_MAGNITUDE`]: one BF16
/// rounding step at the larger magnitude, or both values under the small-value
/// escape.
fn within_one_bf16_ulp(a_bits: u16, b_bits: u16) -> bool {
    let a = bf16::from_bits(a_bits).to_f32() as f64;
    let b = bf16::from_bits(b_bits).to_f32() as f64;
    if a.abs() < CUBLAS_SMALL_MAGNITUDE && b.abs() < CUBLAS_SMALL_MAGNITUDE {
        return true;
    }
    let ulp = if a.abs() >= b.abs() {
        bf16_ulp(a_bits)
    } else {
        bf16_ulp(b_bits)
    };
    (a - b).abs() <= ulp
}

/// Opposite signs with neither value zero. Reported for diagnosis: on these
/// shapes every flip seen so far has been a cancellation residue under the
/// small-magnitude escape, and a flip on a LARGE output would fail the
/// one-ULP bound anyway.
fn is_sign_flip(a_bits: u16, b_bits: u16) -> bool {
    let a = bf16::from_bits(a_bits).to_f32();
    let b = bf16::from_bits(b_bits).to_f32();
    a != 0.0 && b != 0.0 && (a < 0.0) != (b < 0.0)
}

struct Compare {
    max_abs: f64,
    cosine: f64,
    rel_rms: f64,
    max_ulp: i32,
    unequal: usize,
    /// Elements failing [`within_one_bf16_ulp`] — the cuBLASLt-vs-kernel gate.
    over_bound: usize,
    sign_flips: usize,
}

fn compare(a_bits: &[u16], b_bits: &[u16]) -> Compare {
    let (a, b) = (to_f64(a_bits), to_f64(b_bits));
    let (mut dot, mut na, mut nb, mut max_abs, mut sq_diff, mut sq_ref) =
        (0.0, 0.0, 0.0, 0.0_f64, 0.0, 0.0);
    for (x, y) in a.iter().zip(&b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
        max_abs = max_abs.max((x - y).abs());
        sq_diff += (x - y) * (x - y);
        sq_ref += y * y;
    }
    Compare {
        max_abs,
        cosine: dot / (na.sqrt() * nb.sqrt()),
        rel_rms: (sq_diff / sq_ref.max(f64::MIN_POSITIVE)).sqrt(),
        max_ulp: a_bits
            .iter()
            .zip(b_bits)
            .map(|(x, y)| (ord(*x) - ord(*y)).abs())
            .max()
            .unwrap_or(0),
        unequal: a_bits.iter().zip(b_bits).filter(|(x, y)| x != y).count(),
        over_bound: a_bits
            .iter()
            .zip(b_bits)
            .filter(|(x, y)| !within_one_bf16_ulp(**x, **y))
            .count(),
        sign_flips: a_bits
            .iter()
            .zip(b_bits)
            .filter(|(x, y)| is_sign_flip(**x, **y))
            .count(),
    }
}

/// GPU time per iteration for `launch`, in seconds (CUDA events, no host sync
/// between iterations).
fn time_gpu(
    gpu: &dyn GpuBackend,
    stream: u64,
    mut launch: impl FnMut() -> Result<()>,
) -> Result<f64> {
    for _ in 0..WARMUP {
        launch()?;
    }
    gpu.synchronize(stream)?;
    let (mut ev_start, mut ev_end) = (0u64, 0u64);
    for (ev, what) in [(&mut ev_start, "start"), (&mut ev_end, "end")] {
        let rc = unsafe { cuEventCreate(ev, 0) };
        if rc != 0 {
            bail!("cuEventCreate({what}) failed: status {rc}");
        }
    }
    if unsafe { cuEventRecord(ev_start, stream) } != 0 {
        bail!("cuEventRecord(start) failed");
    }
    for _ in 0..ITERS {
        launch()?;
    }
    if unsafe { cuEventRecord(ev_end, stream) } != 0 {
        bail!("cuEventRecord(end) failed");
    }
    if unsafe { cuEventSynchronize(ev_end) } != 0 {
        bail!("cuEventSynchronize failed");
    }
    let mut ms: f32 = 0.0;
    if unsafe { cuEventElapsedTime(&mut ms, ev_start, ev_end) } != 0 {
        bail!("cuEventElapsedTime failed");
    }
    unsafe {
        cuEventDestroy_v2(ev_start);
        cuEventDestroy_v2(ev_end);
    }
    Ok((ms as f64 / 1e3) / ITERS as f64)
}

fn tflops(m: usize, n: usize, k: usize, secs: f64) -> f64 {
    2.0 * m as f64 * n as f64 * k as f64 / secs / 1e12
}

struct Shape {
    label: &'static str,
    n: usize,
    k: usize,
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let stream = 0u64;
    let w8a16_k = gpu.kernel("w8a16_gemm_pipelined", "w8a16_gemm_pipelined")?;
    let quant_k = gpu.kernel("per_token_group_quant_fp8", "per_token_group_quant_fp8")?;
    let w8a8_k = gpu.kernel("fp8_gemm_t_blockscaled", "fp8_gemm_t_blockscaled")?;
    let scale_kmajor_k = gpu.kernel("fp8_scale_transpose", "fp8_act_scale_to_kmajor")?;
    let want_cublas = std::env::var("ATLAS_CUBLAS_GEMM").as_deref() == Ok("1");
    let kmajor = ops::cublas_scale_layout_kmajor();
    let rel_rms_gate = std::env::var("ATLAS_W8A8_REL_RMS_GATE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(REL_RMS_GATE);

    println!(
        "dense-FFN W8A8 microtest — H={H} INTER={INTER}  cuBLASLt={}  \
         scale_layout={}  gates: cosine>={COSINE_GATE} rel_rms<={rel_rms_gate}  \
         cuBLASLt-vs-kernel: over_1ulp==0 (small-value escape \
         |v|<{CUBLAS_SMALL_MAGNITUDE}) cosine>={CUBLAS_COSINE_GATE} \
         rel_rms<={CUBLAS_REL_RMS_GATE}",
        if want_cublas {
            "on (ATLAS_CUBLAS_GEMM=1)"
        } else {
            "off"
        },
        if kmajor {
            "kmajor [K/128,M_pad] (documented)"
        } else {
            "rowmajor [M,K/128] (pre-fix control, expected to FAIL)"
        }
    );

    let mut rng = Rng(0x9_17_09_28_2026);
    let max_m_pad = BATCHES
        .iter()
        .map(|m| m.div_ceil(16) * 16)
        .max()
        .expect("BATCHES is non-empty");
    let max_k = H.max(INTER);
    let max_n = H.max(INTER);

    // Activations [max_m_pad, max_k] BF16 — one buffer, sliced per shape. The
    // padded rows exist because the cuBLASLt arm reads ceil16(M) rows.
    let act_host: Vec<u8> = (0..max_m_pad * max_k)
        .flat_map(|_| bf16::from_f32(rng.unit()).to_bits().to_le_bytes())
        .collect();
    let act = upload(&gpu, &act_host)?;
    let a_fp8 = gpu.alloc(max_m_pad * max_k)?;
    let a_scale = gpu.alloc(max_m_pad * (max_k / BLOCK) * 4)?;
    // `[K/128, ceil16(M)]` transposed scales for the cuBLASLt arm — the layout
    // adapter's destination, same element count as `a_scale`.
    let a_scale_kmajor = gpu.alloc(max_m_pad * (max_k / BLOCK) * 4)?;
    let out_ref = gpu.alloc(max_m_pad * max_n * 2)?;
    let out_w8a8 = gpu.alloc(max_m_pad * max_n * 2)?;
    let out_cublas = gpu.alloc(max_m_pad * max_n * 2)?;

    let shapes = [
        Shape {
            label: "gate/up",
            n: INTER,
            k: H,
        },
        Shape {
            label: "down",
            n: H,
            k: INTER,
        },
    ];
    let mut failures: Vec<String> = Vec::new();

    for shape in shapes {
        let (n, k) = (shape.n, shape.k);
        // E4M3 bytes with the 0x7F/0xFF NaN encodings excluded (mantissa capped
        // at 6 for the all-ones exponent is what the quantizer emits; staying
        // under 127 in the magnitude avoids the encoding entirely).
        let w_host: Vec<u8> = (0..n * k)
            .map(|_| {
                let x = rng.next_u32();
                ((x % 127) as u8) | (((x >> 7) & 1) as u8) << 7
            })
            .collect();
        // Block scales [N/128, K/128] FP32 — the checkpoint layout both the
        // W8A16 kernel and the W8A8 FP32 epilogue index as `scale[n/128][k/128]`.
        let s_host: Vec<u8> = (0..(n / BLOCK) * (k / BLOCK))
            .flat_map(|_| ((rng.next_u32() % 16 + 1) as f32 / 1024.0).to_le_bytes())
            .collect();
        let weight = upload(&gpu, &w_host)?;
        let scale = upload(&gpu, &s_host)?;
        let fp8w = Fp8Weight {
            weight,
            row_scale: scale,
            n: n as u32,
            k: k as u32,
            scale_format: WeightQuantFormat::Fp8BlockScaled,
        };

        for m in BATCHES {
            let (mu, nu, ku) = (m as u32, n as u32, k as u32);
            // ── W8A16 reference (today's production path) ──
            let w8a16 = || {
                ops::w8a16_gemm_pipelined(
                    &gpu, w8a16_k, act, weight, scale, out_ref, mu, nu, ku, stream,
                )
            };
            w8a16()?;
            gpu.synchronize(stream)?;
            let ref_bits = download_bf16(&gpu, out_ref, m * n)?;
            let t_w8a16 = time_gpu(&gpu, stream, w8a16)?;

            // ── W8A8: quantize once, then the in-tree GEMM ──
            ops::per_token_group_quant_fp8(&gpu, quant_k, act, a_fp8, a_scale, mu, ku, stream)?;
            gpu.synchronize(stream)?;
            let w8a8 = || {
                ops::fp8_gemm_t_blockscaled(
                    &gpu, w8a8_k, a_fp8, a_scale, weight, scale, out_w8a8, mu, nu, ku, stream,
                )
            };
            w8a8()?;
            gpu.synchronize(stream)?;
            let w8a8_bits = download_bf16(&gpu, out_w8a8, m * n)?;
            let t_w8a8 = time_gpu(&gpu, stream, w8a8)?;

            let c = compare(&w8a8_bits, &ref_bits);
            println!(
                "[{}] M={m} N={n} K={k}\n  W8A16 ref : {:>8.3} ms  {:>7.2} TFLOP/s\n  \
                 W8A8 kern : {:>8.3} ms  {:>7.2} TFLOP/s  ({:.2}x)  \
                 max_abs={:.6} cosine={:.6} rel_rms={:.4}",
                shape.label,
                t_w8a16 * 1e3,
                tflops(m, n, k, t_w8a16),
                t_w8a8 * 1e3,
                tflops(m, n, k, t_w8a8),
                t_w8a16 / t_w8a8,
                c.max_abs,
                c.cosine,
                c.rel_rms,
            );
            if !(c.cosine >= COSINE_GATE) || !c.cosine.is_finite() {
                failures.push(format!(
                    "{} M={m}: cosine {:.6} < {COSINE_GATE}",
                    shape.label, c.cosine
                ));
            }
            if !(c.rel_rms <= rel_rms_gate) || !c.rel_rms.is_finite() {
                failures.push(format!(
                    "{} M={m}: rel_rms {:.4} > {rel_rms_gate}",
                    shape.label, c.rel_rms
                ));
            }

            // ── cuBLASLt on the SAME quantized activation ──
            if want_cublas {
                let cublas = || {
                    ops::cublas_fp8_proj_prequant(
                        &gpu,
                        scale_kmajor_k,
                        a_fp8,
                        a_scale,
                        a_scale_kmajor,
                        &fp8w,
                        out_cublas,
                        mu,
                        nu,
                        ku,
                        stream,
                    )
                };
                cublas()?;
                gpu.synchronize(stream)?;
                let cub_bits = download_bf16(&gpu, out_cublas, m * n)?;
                let t_cub = time_gpu(&gpu, stream, cublas)?;
                let d = compare(&cub_bits, &w8a8_bits);
                let r = compare(&cub_bits, &ref_bits);
                println!(
                    "  cuBLASLt  : {:>8.3} ms  {:>7.2} TFLOP/s  ({:.2}x vs W8A16, {:.2}x vs kernel)\n    \
                     vs kernel: over_1ulp={}/{} sign_flips={} unequal_bf16={} max_ulp={} \
                     max_abs={:.6} cosine={:.9} rel_rms={:.2e}\n    \
                     vs W8A16 : max_abs={:.6} cosine={:.6} rel_rms={:.4}",
                    t_cub * 1e3,
                    tflops(m, n, k, t_cub),
                    t_w8a16 / t_cub,
                    t_w8a8 / t_cub,
                    d.over_bound,
                    m * n,
                    d.sign_flips,
                    d.unequal,
                    d.max_ulp,
                    d.max_abs,
                    d.cosine,
                    d.rel_rms,
                    r.max_abs,
                    r.cosine,
                    r.rel_rms,
                );
                // Same quantized inputs and the same FP32 epilogue: a real
                // disagreement here is a layout bug (scale order, transpose),
                // not precision — see the gate constants for why the bounds are
                // where they are.
                if d.over_bound > 0 {
                    failures.push(format!(
                        "{} M={m}: cuBLASLt vs kernel {} of {} elements outside one BF16 ULP \
                         (small-value escape |v|<{CUBLAS_SMALL_MAGNITUDE}) — check the VEC128 \
                         act-scale / BLK128x128 weight-scale layouts",
                        shape.label,
                        d.over_bound,
                        m * n
                    ));
                }
                if !(d.cosine >= CUBLAS_COSINE_GATE) || !d.cosine.is_finite() {
                    failures.push(format!(
                        "{} M={m}: cuBLASLt vs kernel cosine {:.9} < {CUBLAS_COSINE_GATE}",
                        shape.label, d.cosine
                    ));
                }
                if !(d.rel_rms <= CUBLAS_REL_RMS_GATE) || !d.rel_rms.is_finite() {
                    failures.push(format!(
                        "{} M={m}: cuBLASLt vs kernel rel_rms {:.2e} > {CUBLAS_REL_RMS_GATE:.0e}",
                        shape.label, d.rel_rms
                    ));
                }
                if !(r.cosine >= COSINE_GATE) {
                    failures.push(format!(
                        "{} M={m}: cuBLASLt vs W8A16 cosine {:.6} < {COSINE_GATE}",
                        shape.label, r.cosine
                    ));
                }
            }
        }
        gpu.free(weight).ok();
        gpu.free(scale).ok();
    }

    for p in [
        act,
        a_fp8,
        a_scale,
        a_scale_kmajor,
        out_ref,
        out_w8a8,
        out_cublas,
    ] {
        gpu.free(p).ok();
    }
    if failures.is_empty() {
        println!("RESULT: PASS (all shapes within cosine/rel_rms/one-BF16-ULP gates)");
        Ok(())
    } else {
        for f in &failures {
            eprintln!("FAIL: {f}");
        }
        bail!("{} gate(s) failed", failures.len())
    }
}

/// CPU tests for the cuBLASLt-vs-kernel acceptance. They need no GPU, no
/// kernels and no model: the point of pulling the bound out into
/// [`within_one_bf16_ulp`] is that the rule can be pinned on synthetic vectors
/// instead of only being exercised by the H100 run it gates.
///
/// Enabled by `test = true` on this example's `[[example]]` stanza, so
/// `cargo test -p spark-model --example native_fp8_ffn_w8a8_microtest
/// --features cuda,gpu-examples` runs them.
#[cfg(test)]
mod tests {
    use super::*;

    /// The BF16 value one representable step above `x`, by construction rather
    /// than by arithmetic: increment the significand of the stored bits.
    fn next_bf16_up(x: f32) -> u16 {
        let b = bf16::from_f32(x).to_bits();
        assert!(b & 0x8000 == 0, "helper is for positive values");
        b + 1
    }

    #[test]
    fn ulp_is_the_gap_to_the_next_bf16() {
        // 1.0 sits at the bottom of its binade: 7 stored mantissa bits -> 2^-7.
        let one = bf16::from_f32(1.0).to_bits();
        assert_eq!(bf16_ulp(one), 2.0_f64.powi(-7));
        // And the step really is the distance to the neighbour.
        let up = bf16::from_bits(next_bf16_up(1.0)).to_f32() as f64;
        assert!((up - 1.0 - bf16_ulp(one)).abs() < 1e-12);
        // The H100 run's max_abs values are exactly one ULP at their magnitude:
        // 1.000 in [128, 256) and 2.000 in [256, 512).
        assert_eq!(bf16_ulp(bf16::from_f32(200.0).to_bits()), 1.0);
        assert_eq!(bf16_ulp(bf16::from_f32(400.0).to_bits()), 2.0);
    }

    #[test]
    fn one_ulp_apart_passes() {
        for v in [1.0_f32, 3.5, 128.0, 200.0, 17408.0, 0.0625] {
            let a = bf16::from_f32(v).to_bits();
            let b = next_bf16_up(v);
            assert!(
                within_one_bf16_ulp(a, b),
                "one step at {v} should be accepted"
            );
            assert!(within_one_bf16_ulp(b, a), "the bound is symmetric at {v}");
        }
    }

    #[test]
    fn two_ulps_apart_fails() {
        for v in [1.0_f32, 3.5, 128.0, 200.0, 17408.0, 0.0625] {
            let a = bf16::from_f32(v).to_bits();
            let b = next_bf16_up(v) + 1;
            assert!(
                !within_one_bf16_ulp(a, b),
                "two steps at {v} must be rejected — this is the resolution the \
                 gate exists to have"
            );
        }
    }

    #[test]
    fn near_zero_sign_flip_passes() {
        // The H100 residual: cancellation noise whose sign is set by the
        // accumulation order. Magnitudes are the four decoded from the
        // 2026-09-11 run, plus the escape's own boundary.
        for v in [0.0148_f32, 0.0167, 0.0237, 0.0297, 0.049] {
            let a = bf16::from_f32(v).to_bits();
            let b = bf16::from_f32(-v).to_bits();
            assert!(
                within_one_bf16_ulp(a, b),
                "a sign flip at |v|={v} is cancellation residue, not a layout bug"
            );
            assert!(is_sign_flip(a, b), "and it is still COUNTED as a flip");
        }
    }

    #[test]
    fn sign_flip_above_the_escape_fails() {
        // Nothing large gets the escape: a flip at 0.06 is 0.12 apart against a
        // one-ULP bound of 2^-11, and a flip at 200.0 is 400 apart against 1.0.
        for v in [0.06_f32, 1.0, 200.0] {
            let a = bf16::from_f32(v).to_bits();
            let b = bf16::from_f32(-v).to_bits();
            assert!(
                !within_one_bf16_ulp(a, b),
                "a sign flip at |v|={v} is outside the small-value escape"
            );
        }
    }

    #[test]
    fn equal_values_and_zero_pass_and_nan_does_not() {
        let x = bf16::from_f32(7.25).to_bits();
        assert!(within_one_bf16_ulp(x, x));
        assert!(!is_sign_flip(x, x));
        let zero = bf16::from_f32(0.0).to_bits();
        let neg_zero = bf16::from_f32(-0.0).to_bits();
        assert!(within_one_bf16_ulp(zero, neg_zero));
        assert!(!is_sign_flip(zero, neg_zero), "+-0 is not a sign flip");
        let nan = bf16::from_f32(f32::NAN).to_bits();
        assert!(!within_one_bf16_ulp(nan, x), "NaN must never be accepted");
        assert!(!within_one_bf16_ulp(nan, nan));
    }

    /// A pair straddling a binade boundary is judged at the COARSER grid, which
    /// is the larger magnitude's — otherwise 127.5 and 128.0, genuine BF16
    /// neighbours, would be scored as a violation. The deliberate cost is that
    /// just under a boundary the rule admits two steps of the finer grid below
    /// it (127.0 vs 128.0); at the boundary the two values' shared resolution
    /// IS the coarse one, and a layout bug is never a boundary-sized error.
    #[test]
    fn straddling_a_binade_uses_the_larger_magnitude() {
        let hi = bf16::from_f32(128.0).to_bits(); // ULP 1.0
        let lo = hi - 1; // 127.5, the largest BF16 below 128 (ULP 0.5)
        assert_eq!(bf16::from_bits(lo).to_f32(), 127.5);
        assert!(within_one_bf16_ulp(lo, hi));
        assert!(within_one_bf16_ulp(hi, lo));
        assert!(
            within_one_bf16_ulp(lo - 1, hi),
            "127.0 vs 128.0 is one step of the coarser grid — accepted by design"
        );
        assert!(
            !within_one_bf16_ulp(lo - 2, hi),
            "126.5 vs 128.0 is 1.5 ULP even at the coarser grid"
        );
    }

    /// The whole-vector `compare` must agree with the per-element rule, and
    /// count rather than gate the flips.
    #[test]
    fn compare_counts_over_bound_and_sign_flips() {
        let a: Vec<u16> = vec![
            bf16::from_f32(1.0).to_bits(),    // equal
            bf16::from_f32(3.5).to_bits(),    // one ulp apart
            bf16::from_f32(0.0148).to_bits(), // near-zero sign flip
            bf16::from_f32(64.0).to_bits(),   // two ulps apart
        ];
        let b: Vec<u16> = vec![
            bf16::from_f32(1.0).to_bits(),
            next_bf16_up(3.5),
            bf16::from_f32(-0.0148).to_bits(),
            next_bf16_up(64.0) + 1,
        ];
        let c = compare(&a, &b);
        assert_eq!(c.over_bound, 1, "only the two-ULP element is out of bound");
        assert_eq!(c.sign_flips, 1);
        assert_eq!(c.unequal, 3);
    }
}
