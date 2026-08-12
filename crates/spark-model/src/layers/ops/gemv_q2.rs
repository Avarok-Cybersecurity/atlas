// SPDX-License-Identifier: AGPL-3.0-only

//! Launch wrappers for the native keep-packed ternary Q2_0 decode GEMV.
//!
//! Mirrors the `w8a16_gemv` / `w4a16_gemv` wrapper shape (`KernelLaunch`,
//! grid `(ceil(N/4),1,1)`, block `(256,1,1)`), but the Q2_0 weight carries its
//! fp16 scale INLINE in each `block_q2_0`, so there is no separate scale-pointer
//! argument — the kernel reads `d` from every block. Kernel source:
//! `kernels/gb10/common/q2_0_gemv.cu` (module stem `q2_0_gemv`).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::PackedQ2Weight;

/// Q2_0 GEMV (M=1 decode): `C[1,N] = A[1,K] @ dequant(B)`, weights kept packed.
///
/// `A` is BF16 `[1, K]`; `B` is the raw `block_q2_0` buffer; `C` is BF16
/// `[1, N]`. Dequant `(code-1)*d` happens inside the dot-product.
///
/// Kernel: `q2_0_gemv(A, B, C, N, K, group)`  Grid: (ceil(N/4),1,1) Block: (256,1,1)
pub fn q2_0_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &PackedQ2Weight,
    output: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(weight.n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(weight.n)
        .arg_u32(weight.k)
        .arg_u32(weight.group as u32)
        .launch(stream)
}

/// Q2_0 batched GEMV (M=1..8 decode): `C[M,N] = A[M,K] @ dequant(B)`.
///
/// Reads each weight block once and accumulates across all `m` activation rows,
/// amortizing the (2-bit) weight-byte read across the batch. `A` is BF16
/// `[M, K]` row-major, `C` is BF16 `[M, N]` row-major.
///
/// Kernel: `q2_0_gemv_batchm(A, B, C, N, K, group, M)`.
#[allow(clippy::too_many_arguments)]
pub fn q2_0_gemv_batchm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &PackedQ2Weight,
    output: DevicePtr,
    m: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(weight.n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(weight.n)
        .arg_u32(weight.k)
        .arg_u32(weight.group as u32)
        .arg_u32(m)
        .launch(stream)
}

/// Dequant a packed Q2_0 weight `[N, K]` (contiguous `block_q2_0` blocks) into a
/// pre-allocated BF16 scratch buffer `[N, K]` on `stream`, IN PLACE (no alloc,
/// no host sync). Reuses the load-time `dequant_q2_0_gn_to_bf16` kernel
/// (`dequant_gguf_bf16` module). Used by packed-Q2 PREFILL: dequant → transient
/// BF16 → normal BF16 GEMM → free scratch (the resident weight stays 2-bit).
///
/// `n_blocks = n * (k / group)`; each block is `2 + group/4` bytes and expands
/// to `group` BF16 elements. Kernel: grid `(n_blocks,1,1)` block `(256,1,1)`.
#[allow(clippy::too_many_arguments)]
pub fn dequant_q2_0_gn_to_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    blocks: DevicePtr,
    out: DevicePtr,
    n: u32,
    k: u32,
    group: u32,
    stream: u64,
) -> Result<()> {
    let n_blocks = n * (k / group);
    let block_bytes = 2 + group / 4;
    KernelLaunch::new(gpu, kernel)
        .grid([n_blocks, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(blocks)
        .arg_ptr(out)
        .arg_u32(n_blocks)
        .arg_u32(group)
        .arg_u32(block_bytes)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use half::f16;

    /// Build one `block_q2_0` (group @ 34/18 bytes) from `group` codes in
    /// {0,1,2,3} plus an fp16 scale — the exact on-disk PrismML layout the
    /// `q2_0_gemv` kernel reads: `[fp16 d @ front][group/4 bytes, 4 codes/byte,
    /// low-bits-first]`.
    fn pack_block(d: f32, codes: &[u8]) -> Vec<u8> {
        let group = codes.len();
        let mut b = Vec::with_capacity(2 + group / 4);
        b.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for chunk in codes.chunks(4) {
            let mut byte = 0u8;
            for (t, &c) in chunk.iter().enumerate() {
                debug_assert!(c < 4);
                byte |= (c & 3) << (2 * t as u8);
            }
            b.push(byte);
        }
        b
    }

    /// Pure Rust mirror of the kernel's dequant-in-dot-product:
    /// `out = sum_k a[k] * (code(k)-1) * d(k/group)`, reading blocks exactly as
    /// `q2_0_gemv.cu` does. One weight row = `k/group` contiguous blocks.
    fn packed_dot(row_bytes: &[u8], a: &[f32], group: usize) -> f32 {
        let block_bytes = 2 + group / 4;
        let blocks = a.len() / group;
        let mut acc = 0.0f32;
        for b in 0..blocks {
            let blk = &row_bytes[b * block_bytes..(b + 1) * block_bytes];
            let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            let qs = &blk[2..];
            for j in 0..group {
                let code = (qs[j >> 2] >> (2 * (j & 3))) & 3;
                acc += a[b * group + j] * ((code as i32 - 1) as f32) * d;
            }
        }
        acc
    }

    /// Golden: dequant each code to `(code-1)*d` first, then a plain dense dot.
    /// This is the numeric oracle the on-device kernel must match; here it locks
    /// the CPU-side bit layout + symbol mapping the kernel spec depends on.
    fn dense_dot(codes: &[u8], d: &[f32], a: &[f32], group: usize) -> f32 {
        (0..a.len())
            .map(|k| a[k] * ((codes[k] as i32 - 1) as f32) * d[k / group])
            .sum()
    }

    #[test]
    fn packed_gemv_math_matches_dense_g128() {
        let group = 128usize;
        let k = 256usize; // 2 groups
        // Deterministic pseudo-random codes {0,1,2,3} and activations.
        let codes: Vec<u8> = (0..k).map(|i| ((i * 7 + 3) % 4) as u8).collect();
        let a: Vec<f32> = (0..k).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect();
        let d = [0.0123f32, -0.0456f32];

        let mut row = Vec::new();
        row.extend(pack_block(d[0], &codes[0..group]));
        row.extend(pack_block(d[1], &codes[group..2 * group]));
        assert_eq!(row.len(), 2 * (2 + group / 4));

        let got = packed_dot(&row, &a, group);
        let want = dense_dot(&codes, &d, &a, group);
        assert!((got - want).abs() < 1e-3, "packed {got} vs dense {want}");
    }

    #[test]
    fn ternary_symbols_are_code_minus_one() {
        // code {0,1,2,3} → {-1, 0, +1, +2}: asymmetric "ternary+" per the spec.
        let group = 4usize;
        let codes = [0u8, 1, 2, 3];
        let a = [1.0f32, 1.0, 1.0, 1.0];
        let d = [2.0f32];
        let row = pack_block(d[0], &codes);
        // sum a*(code-1)*d = (-1 + 0 + 1 + 2) * 2 = 4.
        assert!((packed_dot(&row, &a, group) - 4.0).abs() < 1e-4);
        // low-bits-first: byte 0 packs codes[0..4] = 0|1<<2|2<<4|3<<6 = 0xE4.
        assert_eq!(row[2], 0xE4);
    }

    #[test]
    fn packed_gemv_math_matches_dense_g64() {
        let group = 64usize;
        let k = 128usize;
        let codes: Vec<u8> = (0..k).map(|i| ((i * 3 + 1) % 4) as u8).collect();
        let a: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
        let d = [0.01f32, 0.02f32];
        let mut row = Vec::new();
        row.extend(pack_block(d[0], &codes[0..group]));
        row.extend(pack_block(d[1], &codes[group..]));
        let got = packed_dot(&row, &a, group);
        let want = dense_dot(&codes, &d, &a, group);
        assert!(
            (got - want).abs() < 1e-3,
            "g64 packed {got} vs dense {want}"
        );
    }
}
