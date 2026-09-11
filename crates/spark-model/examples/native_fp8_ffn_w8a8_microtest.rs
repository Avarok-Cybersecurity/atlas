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
//!
//! ★ REQUIRES A KERNEL SET THAT CARRIES `fp8_scale_transpose.cu`. Since the
//! 2026-09-11 arch separation that source is a HOPPER-TUNED file
//! (`kernels/hopper/common`, declared in that target's `[kernels] overrides`)
//! and `kernels/gb10` does not carry it, so this oracle is for a binary built
//! with `ATLAS_TARGET_HW=hopper` (or `b200`, which symlinks the same source).
//! On a GB10 build the `gpu.kernel(...)` lookup below fails by name, which is
//! the honest answer: the tier this grades does not exist there.

use anyhow::{Result, bail};
use half::bf16;
use spark_model::layers::ops;
use spark_model::weight_map::{Fp8Weight, WeightQuantFormat};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

#[path = "common/native_fp8_bf16_compare.rs"]
pub(crate) mod native_fp8_bf16_compare;
use native_fp8_bf16_compare::{
    CUBLAS_COSINE_GATE, CUBLAS_REL_RMS_GATE, CUBLAS_SMALL_MAGNITUDE, compare,
};

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
