// SPDX-License-Identifier: AGPL-3.0-only
//! The FUSED dense-FFN gate+up decode GEMM against the two-GEMM pair it
//! replaces, at Qwen3.8-27B shapes (#927).
//!
//! WHY. nsys `--cuda-graph-trace=node`, 1xH100 80GB HBM3,
//! `Qwen/Qwen3.8-27B-FP8` @ `3717cb05e`, round 13 cell V, median `n = 16`
//! decode step **19.887 ms** busy (`h100-r13-attribution.md` §§C.2–C.4): the
//! dense FFN's gate and up projections are **128 graph nodes, 5 730.5 µs =
//! 44.77 µs/node**, `K = 5120` `N = 17408`, 89.1 MB of E4M3 per node —
//! **1 991 GB/s = 59.4 % of HBM**. In the SAME step `down` moves the same
//! 89.1 MB in ONE launch per layer at 71.4 %, and SSM `in_proj_qkvz` at 73.2 %.
//! The weight bytes are read once either way; the 12-point gap is the second
//! launch. At an 80 % target the pair is worth **1 476 µs/step (7.4 %)**.
//!
//! It answers three questions and guesses none:
//!
//!   1. BITS. Concatenating along N produces INDEPENDENT output columns over
//!      the same K with the same block scales, so this is a BYTE-equality gate
//!      and not a tolerance: fused column `j` must equal gate column `j` for
//!      `j < N`, and up column `j - N` above it, byte for byte. Same for the
//!      SiLU consumer — `ops::silu_mul_strided` over the fused rows against
//!      `moe_silu_mul` over the pair. Four KNOWN_BAD controls prove the
//!      comparator can go red, AT EVERY `M` — round 16 ran them once, at the
//!      first `M`, with one of them zeroing a cuBLASLt PAD row that was
//!      already zero, so a green kernel failed the example and took M=8, M=16,
//!      two further controls and every timing arm down with it (§2.2 of the
//!      round-16 receipt). The controls and their arming proof now live in
//!      `common/gateup_fused_controls.rs`.
//!
//!   2. THE PHANTOM ROWS. cuBLASLt is handed `ceil16(M) = 16` at every rung of
//!      this band and WRITES rows `m..16`. Both arms are checked over the full
//!      padded extent, and the guard bands either side of every buffer must
//!      survive untouched — an `N = 34816` call that overran its output would
//!      otherwise land in the neighbouring arena buffer in a serve.
//!
//!   3. TIME. Per `M`, sync'd over `REPS`: µs per gate+up unit and the
//!      weight-bytes-per-second it implies, for both arms. That number is the
//!      lever's whole claim.
//!
//! Run (H100): `ATLAS_TARGET_HW=hopper cargo run --release -p spark-model
//! --features cuda,gpu-examples --example native_fp8_ffn_gateup_fused_microtest`.
//! ★ The target matters: `fp8_scale_transpose.cu` is a HOPPER-owned source
//! (`[kernels] overrides`), so a GB10 build fails the lookup by name. cuBLASLt
//! is called directly, so `ATLAS_CUBLAS_GEMM` is not needed; the serve
//! spelling is printed at the end.

use anyhow::{Result, ensure};
use half::bf16;
use spark_model::layers::ops;
use spark_model::weight_map::{Fp8Weight, WeightQuantFormat};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::time::Instant;

// The KNOWN_BAD controls, the spans they index and the host tests that prove
// each one moves a graded byte at every `M` in `ROWS` (#927, round 16 §2.2).
#[path = "common/gateup_fused_controls.rs"]
mod controls;
use controls::{BF16, GATEUP_CONTROLS, apply_control, half_spans, moved_bytes};

const H: usize = 5120; // hidden / contraction width
const INTER: usize = 17408; // intermediate — one half of the fused N
const MAX_M: usize = 16; // the cuBLASLt M pad across the whole band
const ROWS: [usize; 3] = [5, 8, 16];
const REPS: usize = 20;
const GUARD: usize = 64; // sentinel bytes either side of every buffer
const SENTINEL: u8 = 0x5a;

struct Rng(u64);

impl Rng {
    fn bits(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
    /// FP8 E4M3 bytes, sign kept, exponents clipped off the NaN encodings.
    fn fp8(&mut self, elems: usize) -> Vec<u8> {
        (0..elems)
            .map(|_| {
                let x = self.bits();
                ((x % 127) as u8) | (((x >> 7) & 1) as u8 * 128)
            })
            .collect()
    }
    /// One FP32 scale per 128x128 weight block.
    fn scales(&mut self, blocks: usize) -> Vec<u8> {
        (0..blocks)
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

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

/// A guarded device buffer: `GUARD` sentinel bytes, `len` payload, `GUARD`
/// sentinel bytes. `ptr` is the payload; `base` is what gets re-armed.
struct Guarded {
    base: DevicePtr,
    ptr: DevicePtr,
    sentinel: Vec<u8>,
    len: usize,
}

impl Guarded {
    fn new(gpu: &dyn GpuBackend, len: usize) -> Result<Self> {
        let sentinel = vec![SENTINEL; len + 2 * GUARD];
        let base = upload(gpu, &sentinel)?;
        Ok(Self {
            base,
            ptr: base.offset(GUARD),
            sentinel,
            len,
        })
    }
    fn arm(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.copy_h2d(&self.sentinel, self.base)
    }
    /// Payload bytes, after checking both guard bands survived.
    fn read(&self, gpu: &dyn GpuBackend) -> Result<Vec<u8>> {
        let mut host = vec![0_u8; self.sentinel.len()];
        gpu.copy_d2h(self.base, &mut host)?;
        ensure!(
            host[..GUARD].iter().all(|&b| b == SENTINEL)
                && host[GUARD + self.len..].iter().all(|&b| b == SENTINEL),
            "a launch wrote outside its output extent (guard band clobbered)"
        );
        Ok(host[GUARD..GUARD + self.len].to_vec())
    }
}

/// The oracle: `observed` must equal `reference` byte for byte over `spans`
/// (byte offset, byte length) of `observed`, read against `ref_spans` of
/// `reference`. Also refuses anything non-finite inside those spans.
fn equal_bytes(
    observed: &[u8],
    reference: &[u8],
    spans: &[(usize, usize)],
    ref_spans: &[(usize, usize)],
) -> Result<()> {
    ensure!(spans.len() == ref_spans.len(), "span count mismatch");
    for (&(o, len), &(r, rlen)) in spans.iter().zip(ref_spans) {
        ensure!(len == rlen, "span length mismatch");
        let a = &observed[o..o + len];
        let b = &reference[r..r + rlen];
        ensure!(
            a.chunks_exact(2)
                .all(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).is_finite()),
            "nonfinite value in the fused output"
        );
        if a != b {
            let bad = a.iter().zip(b).filter(|(x, y)| x != y).count();
            anyhow::bail!("{bad} of {len} bytes differ (byte equality is the gate)");
        }
    }
    Ok(())
}

/// `out[m, n] = a_fp8[m, k] @ w[n, k]ᵀ`, the exact call the serve's
/// `w8a8_gemm` cuBLASLt arm makes.
#[allow(clippy::too_many_arguments)]
fn gemm(
    gpu: &dyn GpuBackend,
    kmajor_k: KernelHandle,
    a_fp8: DevicePtr,
    a_scale: DevicePtr,
    a_scale_kmajor: DevicePtr,
    w: &Fp8Weight,
    out: DevicePtr,
    m: usize,
) -> Result<()> {
    ops::cublas_fp8_proj_prequant(
        gpu,
        kmajor_k,
        a_fp8,
        a_scale,
        a_scale_kmajor,
        w,
        out,
        m as u32,
        w.n,
        w.k,
        0,
    )
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let quant = ops::Fp8ActQuant::resolve(&gpu);
    let kmajor = gpu.kernel("fp8_scale_transpose", "fp8_act_scale_to_kmajor")?;
    let silu = gpu.kernel("moe_silu_mul", "moe_silu_mul")?;
    let silu_strided = gpu.kernel("silu_mul_strided", "silu_mul_strided")?;
    println!(
        "gate: BYTE EQUALITY (not a tolerance). Splitting N gives independent \
         output columns over the same K with the same block scales, so the \
         fused halves must reproduce the pair exactly."
    );
    println!("shapes: K={H} N={INTER} per half, fused N={}", 2 * INTER);

    let mut rng = Rng(0x0927_2026_F05E_D001);

    // ONE weight allocation holding `[gate | up]` along N, with the two halves
    // as VIEWS inside it — the loader's contract reproduced byte for byte
    // (`weight_loader/qwen35_dense.rs`), so any difference the comparison sees
    // is the GEMM's and not the operands'.
    let w_bytes = rng.fp8(2 * INTER * H);
    let s_blocks = (2 * INTER / 128) * (H / 128);
    let s_bytes = rng.scales(s_blocks);
    let fused_w = upload(&gpu, &w_bytes)?;
    let fused_s = upload(&gpu, &s_bytes)?;
    let half_scale_bytes = (INTER / 128) * (H / 128) * 4;
    let view = |w_off: usize, s_off: usize, n: usize| Fp8Weight {
        weight: fused_w.offset(w_off),
        row_scale: fused_s.offset(s_off),
        n: n as u32,
        k: H as u32,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    };
    let gate_w = view(0, 0, INTER);
    let up_w = view(INTER * H, half_scale_bytes, INTER);
    let fused = view(0, 0, 2 * INTER);

    let act = upload(&gpu, &rng.acts(MAX_M * H))?;

    // Activation-quant scratch, sized exactly as the serve's arena is.
    let kg = H / 128;
    let a_fp8 = gpu.alloc(MAX_M * H)?;
    let a_scale = gpu.alloc(MAX_M * kg * 4)?;
    let a_kmajor = gpu.alloc(MAX_M * kg * 4)?;

    let gate_out = Guarded::new(&gpu, MAX_M * INTER * BF16)?;
    let up_out = Guarded::new(&gpu, MAX_M * INTER * BF16)?;
    let fused_out = Guarded::new(&gpu, MAX_M * 2 * INTER * BF16)?;
    let silu_ref = Guarded::new(&gpu, MAX_M * INTER * BF16)?;
    let silu_obs = Guarded::new(&gpu, MAX_M * INTER * BF16)?;

    // The whole padded extent, per half. cuBLASLt writes rows `m..16`, so
    // every row of the buffer is compared, phantom rows included: a phantom
    // row that differed between the arms would be a real difference in a
    // serve, where those rows land in decode slots not in the step.
    let contiguous = half_spans(MAX_M, INTER, 0, INTER);
    let fused_gate = half_spans(MAX_M, 2 * INTER, 0, INTER);
    let fused_up = half_spans(MAX_M, 2 * INTER, INTER, INTER);

    let mut failures = 0usize;

    for m in ROWS {
        println!("\n=== M = {m} (cuBLASLt pad = {MAX_M}) ===");
        // Quantize ONCE — gate and up share the input, which is why the serve
        // splits `cublas_fp8_proj` from its prequant sibling in the first place.
        ops::per_token_group_quant_fp8(&gpu, quant, act, a_fp8, a_scale, m as u32, H as u32, 0)?;

        // ── the pair ──
        gate_out.arm(&gpu)?;
        up_out.arm(&gpu)?;
        gemm(
            &gpu,
            kmajor,
            a_fp8,
            a_scale,
            a_kmajor,
            &gate_w,
            gate_out.ptr,
            m,
        )?;
        gemm(&gpu, kmajor, a_fp8, a_scale, a_kmajor, &up_w, up_out.ptr, m)?;
        gpu.synchronize(0)?;
        let gate_host = gate_out.read(&gpu)?;
        let up_host = up_out.read(&gpu)?;

        // ── the fused arm ──
        fused_out.arm(&gpu)?;
        gemm(
            &gpu,
            kmajor,
            a_fp8,
            a_scale,
            a_kmajor,
            &fused,
            fused_out.ptr,
            m,
        )?;
        gpu.synchronize(0)?;
        let fused_host = fused_out.read(&gpu)?;

        for (name, spans, reference) in [
            ("gate half", &fused_gate, &gate_host),
            ("up half", &fused_up, &up_host),
        ] {
            match equal_bytes(&fused_host, reference, spans, &contiguous) {
                Ok(()) => println!("  {name}: BYTE-IDENTICAL over {MAX_M} padded rows"),
                Err(e) => {
                    failures += 1;
                    println!("  {name}: FAIL {e}");
                }
            }
        }

        // ── the consumer ──
        silu_ref.arm(&gpu)?;
        silu_obs.arm(&gpu)?;
        ops::silu_mul(
            &gpu,
            silu,
            gate_out.ptr,
            up_out.ptr,
            silu_ref.ptr,
            (m * INTER) as u32,
            0,
        )?;
        ops::silu_mul_strided(
            &gpu,
            silu_strided,
            fused_out.ptr,
            fused_out.ptr.offset(INTER * BF16),
            silu_obs.ptr,
            m as u32,
            INTER as u32,
            (2 * INTER) as u32,
            INTER as u32,
            0,
        )?;
        gpu.synchronize(0)?;
        let (r, o) = (silu_ref.read(&gpu)?, silu_obs.read(&gpu)?);
        let live: Vec<(usize, usize)> = (0..m)
            .map(|row| (row * INTER * BF16, INTER * BF16))
            .collect();
        match equal_bytes(&o, &r, &live, &live) {
            Ok(()) => println!("  silu_mul_strided: BYTE-IDENTICAL to moe_silu_mul over {m} rows"),
            Err(e) => {
                failures += 1;
                println!("  silu_mul_strided: FAIL {e}");
            }
        }

        // A green run has to be able to go red — at EVERY M, not once at the
        // first one. The controls are cheap (host-side, on a clone of an
        // already-downloaded buffer) and the thing they are sensitive to is
        // exactly M: rows `m..MAX_M` are cuBLASLt pad, and a control that
        // lands there grades nothing. `moved_bytes` is checked FIRST, so an
        // inert perturbation now fails saying it perturbed nothing, instead of
        // saying the oracle admitted a known-bad output (§2.2).
        for control in GATEUP_CONTROLS {
            let mut bad = fused_host.clone();
            apply_control(control, &mut bad, m, &fused_gate, &fused_up);
            let moved = moved_bytes(&bad, &fused_host, &fused_gate);
            ensure!(
                moved > 0,
                "KNOWN_BAD {control} changed none of the bytes the gate half \
                 grades at M={m} (pad rows are {m}..{MAX_M}) — the control is \
                 inert and the refusal below would prove nothing"
            );
            let err = equal_bytes(&bad, &gate_host, &fused_gate, &contiguous)
                .expect_err("known-bad output was admitted by the real oracle");
            println!("  KNOWN_BAD {control}: moved {moved} graded byte(s), refused: {err}");
        }

        // ── time ──
        let bytes = (2 * INTER * H) as f64; // both halves, read once either way
        let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
            f()?;
            gpu.synchronize(0)?;
            let t = Instant::now();
            for _ in 0..REPS {
                f()?;
            }
            gpu.synchronize(0)?;
            Ok(t.elapsed().as_secs_f64() / REPS as f64)
        };
        let pair = time(&|| {
            gemm(
                &gpu,
                kmajor,
                a_fp8,
                a_scale,
                a_kmajor,
                &gate_w,
                gate_out.ptr,
                m,
            )?;
            gemm(&gpu, kmajor, a_fp8, a_scale, a_kmajor, &up_w, up_out.ptr, m)
        })?;
        let one = time(&|| {
            gemm(
                &gpu,
                kmajor,
                a_fp8,
                a_scale,
                a_kmajor,
                &fused,
                fused_out.ptr,
                m,
            )
        })?;
        println!(
            "  two GEMMs: {:8.2} us  {:7.0} GB/s   |  fused: {:8.2} us  {:7.0} GB/s   \
             ({:.2}x)",
            pair * 1e6,
            bytes / pair / 1e9,
            one * 1e6,
            bytes / one / 1e9,
            pair / one,
        );
    }

    println!(
        "\nserve spelling: the arm is `[defaults] ffn_gateup_fused` \
         (hopper `true`); `ATLAS_FFN_GATEUP_FUSED=0` restores the two-GEMM \
         pair. Round-13 receipt and the round-16 prediction: \
         FFN-GATEUP-FUSION-ATTRIBUTION.md"
    );
    ensure!(failures == 0, "{failures} byte-equality gate(s) failed");
    println!("OK");
    Ok(())
}
