// SPDX-License-Identifier: AGPL-3.0-only
//! NVFP4 op-level numeric comparator + packed-weight transpose bit-exactness.
//
// Goal: decide whether the corrupt-output native-NVFP4 prefill path is a
// wrapper layout/scale BUG (fixable) or inherent W4A4 quantization LOSS
// (abandon on the sensitive projections). For each Holo projection shape we
// compare three GEMM results over the SAME inputs:
//   - out_cutlass: the native CUTLASS NVFP4 kernel (act packed in-wrapper,
//     weight = our packed_t/scale_t).
//   - out_ref:     a host W4A4 dequant reference. The weight side reads the
//     EXACT packed_t nibbles + scale_t (e4m3) bytes the kernel consumes, so
//     it is bit-faithful to the kernel's weight operand; the activation side
//     replicates the wrapper's per-16-group max/6 -> e2m1 quantizer.
//   - out_true:    the full unquantized BF16 GEMM.
// Interpretation:
//   cos(cutlass,ref) ~ 1  -> kernel + layouts are CORRECT; any divergence
//       from out_true is inherent W4A4 loss (see cos(ref,true)).
//   cos(cutlass,ref) low  -> wrapper layout/scale BUG; the kernel is not
//       computing what its packed operands imply.

use super::super::*;
use super::*;

#[test]
#[ignore = "requires a free CUDA device and CUTLASS_HOME build"]
fn cutlass_nvfp4_projection_numeric_comparator() {
    // Small M faithfully exercises the N/K-dependent weight scale-factor
    // layout (the suspected bug correlates with large N) while keeping the
    // host reference GEMM cheap.
    const M: usize = 128;
    let shapes = [
        ("ssm_qkvz", 12288usize, 2048usize),
        ("attn_q", 8192, 2048),
        ("attn_kv", 512, 2048),
        ("attn_o", 2048, 4096),
    ];

    for (name, n, k) in shapes {
        assert_eq!(k % 16, 0, "{name}: K must be a multiple of 16");

        // Host inputs (bf16, as the device sees them).
        let weight_bf16: Vec<u16> = (0..n * k)
            .map(|i| f32_to_bf16(gen_val(i as u64) * 0.2))
            .collect();
        let act_bf16: Vec<u16> = (0..M * k)
            .map(|i| f32_to_bf16(gen_val((i as u64) ^ 0xA5A5_0000_0000) * 2.0))
            .collect();

        let packed_len = (k / 2) * n; // [K/2, N] u8
        let scale_len = (k / 16) * n; // [K/16, N] u8

        let weight_dev;
        let act_dev;
        let packed_dev;
        let scale_dev;
        let out_dev;
        unsafe {
            weight_dev = device_alloc(weight_bf16.len() * 2);
            act_dev = device_alloc(act_bf16.len() * 2);
            packed_dev = device_alloc(packed_len);
            scale_dev = device_alloc(scale_len);
            out_dev = device_alloc(M * n * 2);
            copy_h2d(weight_dev, &weight_bf16);
            copy_h2d(act_dev, &act_bf16);
        }

        // Pack the weight to Atlas transposed NVFP4 exactly as the runtime does.
        pack_bf16_weight_to_nvfp4_t(
            weight_dev as u64,
            packed_dev as u64,
            scale_dev as u64,
            n as u32,
            k as u32,
            0,
        )
        .unwrap();
        unsafe {
            cuda_check(cudaDeviceSynchronize(), "pack synchronize");
        }

        // Native CUTLASS NVFP4 GEMM (weight_scale_2 = 1.0 for this pack).
        nvfp4_gemm_bf16_act_weight_t(
            act_dev as u64,
            packed_dev as u64,
            scale_dev as u64,
            1.0,
            out_dev as u64,
            M as u32,
            n as u32,
            k as u32,
            0,
        )
        .unwrap();
        unsafe {
            cuda_check(cudaDeviceSynchronize(), "cutlass gemm synchronize");
        }

        let mut out_cutlass_bf16 = vec![0u16; M * n];
        let mut packed = vec![0u8; packed_len];
        let mut scale = vec![0u8; scale_len];
        unsafe {
            copy_d2h(&mut out_cutlass_bf16, out_dev);
            copy_d2h(&mut packed, packed_dev);
            copy_d2h(&mut scale, scale_dev);
            cuda_check(cudaFree(weight_dev), "free weight");
            cuda_check(cudaFree(act_dev), "free act");
            cuda_check(cudaFree(packed_dev), "free packed");
            cuda_check(cudaFree(scale_dev), "free scale");
            cuda_check(cudaFree(out_dev), "free out");
        }

        // Weight dequant, bit-faithful to the kernel's operand: read the same
        // packed nibbles + e4m3 group scales the kernel consumes. The pack
        // kernel emits CUTLASS [N,K/2] layout (K-contiguous): byte for
        // (n=col,k) is col*(K/2) + k/2, nibble = k&1. Scales stay [K/16,N].
        let mut w_q = vec![0f32; n * k];
        let mut w_true = vec![0f32; n * k];
        for col in 0..n {
            for kk in 0..k {
                let g = kk / 16;
                let byte = packed[col * (k / 2) + kk / 2];
                let nib = if kk % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                let s = e4m3_to_f32(scale[g * n + col]);
                w_q[col * k + kk] = decode_e2m1(nib) * s;
                w_true[col * k + kk] = bf16_to_f32(weight_bf16[col * k + kk]);
            }
        }

        // Activation dequant, replicating the wrapper's per-16-group quantizer.
        let mut a_q = vec![0f32; M * k];
        let mut a_true = vec![0f32; M * k];
        for m in 0..M {
            for g in 0..(k / 16) {
                let base = g * 16;
                let mut max_abs = 0.0f32;
                for i in 0..16 {
                    let v = bf16_to_f32(act_bf16[m * k + base + i]);
                    max_abs = max_abs.max(v.abs());
                }
                let s = if max_abs > 0.0 { max_abs / 6.0 } else { 1.0 };
                let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
                for i in 0..16 {
                    let v = bf16_to_f32(act_bf16[m * k + base + i]);
                    let nib = f32_to_e2m1(v * inv);
                    a_q[m * k + base + i] = decode_e2m1(nib) * s;
                    a_true[m * k + base + i] = v;
                }
            }
        }

        // Reference GEMMs: out[m,n] = sum_k a[m,k] * w[n,k].
        let mut out_ref = vec![0f32; M * n];
        let mut out_true = vec![0f32; M * n];
        let mut out_cutlass = vec![0f32; M * n];
        for m in 0..M {
            for col in 0..n {
                let mut acc_ref = 0.0f32;
                let mut acc_true = 0.0f32;
                for kk in 0..k {
                    acc_ref += a_q[m * k + kk] * w_q[col * k + kk];
                    acc_true += a_true[m * k + kk] * w_true[col * k + kk];
                }
                out_ref[m * n + col] = acc_ref;
                out_true[m * n + col] = acc_true;
                out_cutlass[m * n + col] = bf16_to_f32(out_cutlass_bf16[m * n + col]);
            }
        }

        let cos_cr = cosine(&out_cutlass, &out_ref);
        let cos_ct = cosine(&out_cutlass, &out_true);
        let cos_rt = cosine(&out_ref, &out_true);
        let max_abs_cr = out_cutlass
            .iter()
            .zip(&out_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let ref_rms =
            (out_ref.iter().map(|x| (x * x) as f64).sum::<f64>() / out_ref.len() as f64).sqrt();

        let verdict = if cos_cr > 0.999 {
            "KERNEL OK (cutlass matches W4A4 ref) -> divergence from true is inherent W4A4 loss"
        } else if cos_cr > 0.95 {
            "SUSPECT (minor cutlass<->ref drift; check scale rounding)"
        } else {
            "BUG (cutlass does NOT match its own packed operands -> layout/scale wrong)"
        };

        eprintln!(
            "NVFP4_COMPARATOR {name} M={M} N={n} K={k} \
             cos(cutlass,ref)={cos_cr:.6} cos(cutlass,true)={cos_ct:.6} \
             cos(ref,true)={cos_rt:.6} max_abs(cutlass-ref)={max_abs_cr:.5} \
             ref_rms={ref_rms:.5} => {verdict}"
        );
    }
}

#[test]
#[ignore = "requires a free CUDA device and CUTLASS_HOME build"]
fn cutlass_nvfp4_transpose_is_bit_exact() {
    // Validate the [K/2,N] -> [N,K/2] packed-weight transpose used by the
    // native-checkpoint path (cutlass_nvfp4_proj). Build a golden [N,K/2]
    // pack from a bf16 weight, derive the [K/2,N] "checkpoint" form by
    // transposing on host, run the device transpose, and require it to
    // reproduce the golden pack byte-for-byte.
    const N: usize = 512;
    const K: usize = 256;
    let half = K / 2;

    let weight_bf16: Vec<u16> = (0..N * K)
        .map(|i| f32_to_bf16(gen_val(i as u64) * 0.2))
        .collect();

    // Golden [N,K/2] pack via the (fixed) pack kernel.
    let weight_dev;
    let golden_dev;
    let scale_dev;
    unsafe {
        weight_dev = device_alloc(weight_bf16.len() * 2);
        golden_dev = device_alloc(N * half);
        scale_dev = device_alloc((K / 16) * N);
        copy_h2d(weight_dev, &weight_bf16);
    }
    pack_bf16_weight_to_nvfp4_t(
        weight_dev as u64,
        golden_dev as u64,
        scale_dev as u64,
        N as u32,
        K as u32,
        0,
    )
    .unwrap();
    unsafe {
        cuda_check(cudaDeviceSynchronize(), "pack synchronize");
    }
    let mut golden = vec![0u8; N * half];
    unsafe {
        copy_d2h(&mut golden, golden_dev);
    }

    // Host-derived [K/2,N] checkpoint form: src[h*N + c] = golden[c*half + h].
    let mut checkpoint = vec![0u8; half * N];
    for c in 0..N {
        for h in 0..half {
            checkpoint[h * N + c] = golden[c * half + h];
        }
    }

    let src_dev;
    let dst_dev;
    unsafe {
        src_dev = device_alloc(checkpoint.len());
        dst_dev = device_alloc(N * half);
        copy_h2d(src_dev, &checkpoint);
    }
    transpose_nvfp4_packed_kton(src_dev as u64, dst_dev as u64, N as u32, K as u32, 0).unwrap();
    unsafe {
        cuda_check(cudaDeviceSynchronize(), "transpose synchronize");
    }
    let mut got = vec![0u8; N * half];
    unsafe {
        copy_d2h(&mut got, dst_dev);
        cuda_check(cudaFree(weight_dev), "free weight");
        cuda_check(cudaFree(golden_dev), "free golden");
        cuda_check(cudaFree(scale_dev), "free scale");
        cuda_check(cudaFree(src_dev), "free src");
        cuda_check(cudaFree(dst_dev), "free dst");
    }

    assert_eq!(
        got, golden,
        "device transpose must reproduce the golden [N,K/2] pack"
    );
    eprintln!(
        "NVFP4_TRANSPOSE bit-exact over N={N} K={K} ({} bytes) OK",
        N * half
    );
}
