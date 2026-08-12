// SPDX-License-Identifier: AGPL-3.0-only
//
// Launcher for the native Ternary-Bonsai Q2_0 MMQ prefill GEMM (Tier-2).
// Kernel: kernels/gb10/qwen3.6-27b/nvfp4/q2_0_mmq.cu (module `q2_0_mmq`,
// entries `atlas_q2_0_mmq128_nc/_wc`). Keeps the 2-bit weight PACKED and does
// the prefill matmul as a tensor-core int8 MMA with dequant-in-register
// (`(code-1)*d`) against a q8_1-quantized activation, producing BF16 — no BF16
// weight scratch, no dequant tax, no co-dispatch race.
//
// The q8_1 activation quantize is SHARED with Q4_K: reuse
// `super::quantize_act_q8_1` (kernel `atlas_q8_1_quantize_ds4_bf16`, DS4 layout)
// and `super::q8_1_scratch_bytes` — Q2_0 also uses DS4 (the `(code-1)*d` dequant
// never reads q8_1's `s` term). The only Q2_0-specific launch difference vs
// `q4k_mmq_gemm` is `stride_row_x = k/QK2_0` (K/128, not K/256).
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::PackedQ2Weight;

/// Q2_0 MMQ block size: 128 weights per `block_q2_0`.
pub const QK2_0: u32 = 128;
/// sizeof(block_q2_0) bytes: fp16 scale d (2) + 128 codes @ 4/byte (32) = 34.
pub const Q2_0_BLOCK_BYTES: usize = 34;

/// Sub-flag gating the native Q2_0 MMQ prefill path (`ATLAS_GGUF_NATIVE_Q2_MMQ=1`).
/// Default off: keep the transient-dequant stopgap so the two can be A/B'd on GPU.
/// (`ATLAS_GGUF_NATIVE_Q2` still gates keep-packing overall — this only chooses
/// how the kept-packed weight is consumed in PREFILL.)
pub fn native_q2_mmq_enabled() -> bool {
    std::env::var("ATLAS_GGUF_NATIVE_Q2_MMQ").ok().as_deref() == Some("1")
}

/// Bytes for the packed `block_q2_0` form of an `[n, k]` weight (`k % 128 == 0`).
pub fn q2_0_weight_bytes(n: u32, k: u32) -> usize {
    (n as usize) * (k as usize / QK2_0 as usize) * Q2_0_BLOCK_BYTES
}

/// Q2_0 MMQ GEMM: `C[m,n]` (bf16) = `A_q8[m,k]` x `W_q2_0[n,k]`. Fused bf16 store.
///
/// `a_q8` is the q8_1_mmq (DS4) activation produced by
/// [`super::quantize_act_q8_1`]; `w_q2_0` is the packed `block_q2_0` weight
/// `[n, k]` (the same buffer resident for the decode GEMV — no repack). Grid /
/// block / smem mirror the Q4_K MMQ (same tile geometry, mmq_x=mmq_y=128).
#[allow(clippy::too_many_arguments)]
pub fn q2_0_mmq_gemm(
    gpu: &dyn GpuBackend,
    kernel_nc: KernelHandle, // atlas_q2_0_mmq128_nc
    kernel_wc: KernelHandle, // atlas_q2_0_mmq128_wc
    a_q8: DevicePtr,         // q8_1_mmq activations
    w_q2_0: DevicePtr,       // block_q2_0 weights [n, k]
    out_bf16: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let kernel = if !n.is_multiple_of(128) {
        kernel_wc
    } else {
        kernel_nc
    };
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([32, 8, 1])
        .shared_mem(super::q4k_mmq::Q4K_MMQ_SMEM)
        .arg_ptr(w_q2_0) // x = weights
        .arg_ptr(a_q8) // y = q8_1 activations
        .arg_ptr(out_bf16) // dst
        .arg_u32(n) // nrows_x
        .arg_u32(m) // ncols_dst
        .arg_u32(k) // ncols_x
        .arg_u32(k / QK2_0) // stride_row_x = K/128
        .arg_u32(m) // ncols_y
        .arg_u32(n) // stride_col_dst
        .launch(stream)
}

/// Q2_0 MMQ GEMM against a [`PackedQ2Weight`] (asserts `group == 128`, the only
/// group the MMQ block layout supports — callers fall back to transient-dequant
/// for group 64). Convenience over [`q2_0_mmq_gemm`].
pub fn q2_0_mmq_gemm_packed(
    gpu: &dyn GpuBackend,
    kernel_nc: KernelHandle,
    kernel_wc: KernelHandle,
    a_q8: DevicePtr,
    w: &PackedQ2Weight,
    out_bf16: DevicePtr,
    m: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        w.group == 128,
        "Q2_0 MMQ requires group 128 (got {}); use the transient-dequant path for group 64",
        w.group
    );
    q2_0_mmq_gemm(
        gpu, kernel_nc, kernel_wc, a_q8, w.weight, out_bf16, m, w.n, w.k, stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use half::f16;

    // fp16 round-trip identical to the kernel's inline scale read (fp16 store at
    // load, fp16->f32 in-kernel): the weight scale `d` is only ever fp16-precise.
    fn f32_to_f16_bits(x: f32) -> u16 {
        f16::from_f32(x).to_bits()
    }
    fn f16_bits_to_f32(bits: u16) -> f32 {
        f16::from_bits(bits).to_f32()
    }

    /// Pack an `[n, k]` ternary code matrix (values in {0,1,2}, dequant (code-1))
    /// with per-(row, group-of-128) fp16 scale `d` into `block_q2_0` bytes,
    /// exactly matching the on-disk layout the kernel consumes.
    fn pack_q2_0(codes: &[u8], scales: &[f32], n: usize, k: usize) -> Vec<u8> {
        assert_eq!(k % 128, 0);
        let blocks_per_row = k / 128;
        let mut out = vec![0u8; n * blocks_per_row * Q2_0_BLOCK_BYTES];
        for row in 0..n {
            for b in 0..blocks_per_row {
                let blk = (row * blocks_per_row + b) * Q2_0_BLOCK_BYTES;
                let dbits = f32_to_f16_bits(scales[row * blocks_per_row + b]);
                out[blk] = (dbits & 0xff) as u8;
                out[blk + 1] = (dbits >> 8) as u8;
                for j in 0..128 {
                    let c = codes[row * k + b * 128 + j] & 0x3;
                    let byte = blk + 2 + j / 4;
                    out[byte] |= c << (2 * (j % 4));
                }
            }
        }
        out
    }

    /// CPU model of the kernel's Q2_0 MMQ arithmetic: per-32 q8_1 activation
    /// quantize (d = absmax/127, int8 round-to-nearest), int8 MAC, fold
    /// `(code-1)*d_w * a_int8*d_a`. Mirrors `load_tiles_q2_0` +
    /// `vec_dot_q8_0_q8_1_mma`. Output row-major `[m, n]`.
    fn mmq_cpu(
        act: &[f32],
        codes: &[u8],
        scales: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let bpr = k / 128; // blocks per weight row
        // Quantize activation per 32-lane group: d_a = absmax/127, qs = round(a/d_a).
        let ng = k / 32;
        let mut a_q = vec![0i8; m * k];
        let mut a_d = vec![0f32; m * ng];
        for r in 0..m {
            for g in 0..ng {
                let mut amax = 0f32;
                for t in 0..32 {
                    amax = amax.max(act[r * k + g * 32 + t].abs());
                }
                let d = amax / 127.0;
                a_d[r * ng + g] = d;
                for t in 0..32 {
                    let q = if d > 0.0 {
                        (act[r * k + g * 32 + t] / d).round().clamp(-127.0, 127.0)
                    } else {
                        0.0
                    };
                    a_q[r * k + g * 32 + t] = q as i8;
                }
            }
        }
        let mut out = vec![0f32; m * n];
        for r in 0..m {
            for col in 0..n {
                let mut acc = 0f32;
                for g in 0..ng {
                    let b = g / 4; // which 128-block
                    let dw = f16_bits_to_f32(f32_to_f16_bits(scales[col * bpr + b]));
                    let da = a_d[r * ng + g];
                    let mut isum = 0i32;
                    for t in 0..32 {
                        let ki = g * 32 + t;
                        let w = (codes[col * k + ki] & 0x3) as i32 - 1;
                        isum += w * a_q[r * k + ki] as i32;
                    }
                    acc += isum as f32 * dw * da;
                }
                out[r * n + col] = acc;
            }
        }
        out
    }

    /// FP oracle: direct dequant `(code-1)*d` × full-precision activation.
    fn oracle(act: &[f32], codes: &[u8], scales: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let bpr = k / 128;
        let mut out = vec![0f32; m * n];
        for r in 0..m {
            for col in 0..n {
                let mut acc = 0f32;
                for ki in 0..k {
                    let b = ki / 128;
                    let dw = f16_bits_to_f32(f32_to_f16_bits(scales[col * bpr + b]));
                    let w = ((codes[col * k + ki] & 0x3) as i32 - 1) as f32 * dw;
                    acc += w * act[r * k + ki];
                }
                out[r * n + col] = acc;
            }
        }
        out
    }

    #[test]
    fn q2_0_block_layout_matches_spec() {
        // Ternary-Bonsai spec: 34-byte block, 128 weights, 4 codes/byte.
        assert_eq!(Q2_0_BLOCK_BYTES, 2 + 128 / 4);
        assert_eq!(QK2_0, 128);
        assert_eq!(q2_0_weight_bytes(256, 512), 256 * (512 / 128) * 34);
        // Byte-packing round-trips: code j lands at bits [2*(j%4)] of byte 2+j/4.
        let codes: Vec<u8> = (0..128).map(|j| (j % 3) as u8).collect();
        let packed = pack_q2_0(&codes, &[0.5], 1, 128);
        assert_eq!(packed.len(), 34);
        for j in 0..128usize {
            let got = (packed[2 + j / 4] >> (2 * (j % 4))) & 0x3;
            assert_eq!(got, (j % 3) as u8, "code {j} mispacked");
        }
    }

    #[test]
    fn q2_0_mmq_math_matches_oracle() {
        // Small [M,N,K], K multiple of 128 (two groups per block boundary check).
        let (m, n, k) = (3usize, 5usize, 256usize);
        let bpr = k / 128;
        // Deterministic pseudo-random codes in {0,1,2} and per-block scales.
        let mut codes = vec![0u8; n * k];
        for (i, c) in codes.iter_mut().enumerate() {
            *c = (((i * 2654435761usize) >> 5) % 3) as u8; // {0,1,2}
        }
        let mut scales = vec![0f32; n * bpr];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = 0.015 + 0.01 * ((i % 7) as f32); // small fp16-friendly magnitudes
        }
        let mut act = vec![0f32; m * k];
        for (i, a) in act.iter_mut().enumerate() {
            let x = (i as f32 * 0.12345).sin();
            *a = x * 1.7;
        }

        let mmq = mmq_cpu(&act, &codes, &scales, m, n, k);
        let orc = oracle(&act, &codes, &scales, m, n, k);

        // Relative error is bounded by the q8_1 activation quantization (int8,
        // per-32 absmax) — expect the same ~6-7e-3 band as the verified Q4_K MMQ.
        let mut max_rel = 0f32;
        let mut denom = 0f32;
        let mut num = 0f32;
        for i in 0..m * n {
            let e = (mmq[i] - orc[i]).abs();
            num += e * e;
            denom += orc[i] * orc[i];
            let r = e / orc[i].abs().max(1e-3);
            max_rel = max_rel.max(r);
        }
        let l2_rel = (num / denom.max(1e-12)).sqrt();
        assert!(
            l2_rel < 1e-2,
            "Q2_0 MMQ L2 rel_err {l2_rel:.4e} exceeds 1e-2 (max pointwise {max_rel:.4e})"
        );
    }

    #[test]
    fn q2_0_mmq_ternary_only_codes() {
        // True ternary {-1,0,1}: code alphabet {0,1,2}; code=3 (value +2) never
        // emitted. Confirm the math is exact-to-oracle-within-quant for that set.
        let (m, n, k) = (2usize, 4usize, 128usize);
        let codes: Vec<u8> = (0..n * k).map(|i| (i % 3) as u8).collect();
        let scales = vec![0.02f32; n * (k / 128)];
        let act: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.3).collect();
        let mmq = mmq_cpu(&act, &codes, &scales, m, n, k);
        let orc = oracle(&act, &codes, &scales, m, n, k);
        for i in 0..m * n {
            assert!(
                (mmq[i] - orc[i]).abs() <= 0.02 * orc[i].abs().max(0.5),
                "idx {i}: mmq {} vs oracle {}",
                mmq[i],
                orc[i]
            );
        }
    }
}
