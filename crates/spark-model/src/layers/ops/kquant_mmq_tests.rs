// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! GPU parity for the DeepSeek-V4.1 K-quant expert kernels against a CPU oracle:
//! the weights are dequantised by spark-runtime's CPU decoders (the loader's own
//! reference), the activation is quantised to q8_1 in Rust exactly as the kernels
//! do it (per-block `d = amax / 127`, round half away from zero), and the product
//! is an f32 GEMM. Decode (`kquant_mmvq`, M = 1 and M = 5) and prefill
//! (`kquant_mmq_gemm`, M = 200) are both held to that number for Q2_K and Q3_K.
//!
//! `#[ignore]` per repo convention (CI is CPU-only). Run on Blackbird:
//! ```text
//! cargo test -p spark-model --release kquant -- --ignored --nocapture
//! ```

use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::dequant_cpu::{GgmlType, dequant_to_f32};

use super::kquant_mmq::*;
use crate::layers::ops;

const N: u32 = 320; // not a multiple of 128: exercises the _wc MMQ entry
const K: u32 = 1024;
const SCALES_F16: [u16; 4] = [0x3C00, 0x3800, 0x4000, 0x3400]; // 1.0, 0.5, 2.0, 0.25

fn lcg_bytes(n: usize, mut state: u32) -> Vec<u8> {
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (state >> 16) as u8
        })
        .collect()
}

/// `[N][K]` raw super-blocks with sane per-block fp16 scales at `f16_offsets`.
fn build_weight(block_bytes: usize, f16_offsets: &[usize], seed: u32) -> Vec<u8> {
    let n_blocks = (N as usize) * (K as usize / 256);
    let mut raw = lcg_bytes(n_blocks * block_bytes, seed);
    for b in 0..n_blocks {
        for (i, &off) in f16_offsets.iter().enumerate() {
            let s = SCALES_F16[(b + i) % SCALES_F16.len()];
            raw[b * block_bytes + off] = (s & 0xFF) as u8;
            raw[b * block_bytes + off + 1] = (s >> 8) as u8;
        }
    }
    raw
}

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let lsb = (b >> 16) & 1;
    (b.wrapping_add(0x7FFF + lsb) >> 16) as u16
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Activations `[M][K]` in [-1, 1), as bf16 bits and as the f32 the kernel sees.
fn build_act(m: usize, seed: u32) -> (Vec<u16>, Vec<f32>) {
    let bytes = lcg_bytes(m * K as usize * 2, seed);
    let bits: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| bf16_bits((u16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0) - 1.0))
        .collect();
    let f: Vec<f32> = bits.iter().map(|&b| bf16_to_f32(b)).collect();
    (bits, f)
}

/// q8_1 emulation: per `block` values, d = amax/127, q = round(x/d), back to f32.
fn q8_emulate(x: &[f32], block: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for (i, chunk) in x.chunks(block).enumerate() {
        let amax = chunk.iter().fold(0f32, |a, v| a.max(v.abs()));
        let d = amax / 127.0;
        for (j, &v) in chunk.iter().enumerate() {
            out[i * block + j] = if amax == 0.0 {
                0.0
            } else {
                (v / d).round() * d
            };
        }
    }
    out
}

fn cpu_gemm(w: &[f32], xq: &[f32], m: usize) -> Vec<f32> {
    let (n, k) = (N as usize, K as usize);
    let mut c = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = xq[i * k..(i + 1) * k]
                .iter()
                .zip(&w[j * k..(j + 1) * k])
                .map(|(a, b)| a * b)
                .sum();
        }
    }
    c
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = g.alloc(bytes.len()).unwrap();
    g.copy_h2d(bytes, p).unwrap();
    p
}

fn download_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut buf = vec![0u8; n * 2];
    g.copy_d2h(p, &mut buf).unwrap();
    buf.chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn assert_close(label: &str, got: &[f32], want: &[f32]) {
    let max = want.iter().fold(0f32, |a, v| a.max(v.abs()));
    // bf16 output rounding (half an ulp at the max magnitude) plus f32 order noise
    let tol = max * 2f32.powi(-7) + 1e-3;
    let mut worst = 0f32;
    let mut at = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    assert!(
        worst <= tol,
        "{label}: max abs diff {worst} > tol {tol} at {at} (got {}, want {}; max |want| {max})",
        got[at],
        want[at]
    );
    println!(
        "  {label}: {} values within {tol:.4} (worst {worst:.5}, max |want| {max:.2})",
        got.len()
    );
}

struct Ty {
    name: &'static str,
    id: u32,
    block_bytes: usize,
    f16_offsets: &'static [usize],
    mmvq: &'static str,
    mmq_nc: &'static str,
    mmq_wc: &'static str,
    quant: &'static str,
    smem: u32,
    /// values per activation scale in the MMQ q8_1 layout this type uses
    mmq_act_block: usize,
}

const TYPES: [Ty; 2] = [
    Ty {
        name: "Q2_K",
        id: 10,
        block_bytes: Q2K_BLOCK_BYTES,
        f16_offsets: &[80, 82],
        mmvq: "kquant_mmvq_q2_k",
        mmq_nc: "atlas_q2_k_mmq128_nc",
        mmq_wc: "atlas_q2_k_mmq128_wc",
        quant: "atlas_q8_1_quantize_d2s6_bf16",
        smem: Q2K_MMQ_SMEM,
        mmq_act_block: 64,
    },
    Ty {
        name: "Q3_K",
        id: 11,
        block_bytes: Q3K_BLOCK_BYTES,
        f16_offsets: &[108],
        mmvq: "kquant_mmvq_q3_k",
        mmq_nc: "atlas_q3_k_mmq128_nc",
        mmq_wc: "atlas_q3_k_mmq128_wc",
        quant: "atlas_q8_1_quantize_d4_bf16",
        smem: Q3K_MMQ_SMEM,
        mmq_act_block: 32,
    },
];

fn backend() -> spark_runtime::cuda_backend::AvarokCudaBackend {
    let set = avarok_kernels::ptx_for_exact_target("deepseek-v4-flash", "nvfp4")
        .expect("deepseek-v4-flash/nvfp4 not in this build");
    spark_runtime::cuda_backend::AvarokCudaBackend::new(0, &set.modules).expect("CUDA backend")
}

#[test]
#[ignore = "requires a CUDA GB10 + the deepseek-v4-flash kernel target"]
fn kquant_decode_mmvq_matches_cpu_oracle() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    for ty in &TYPES {
        let raw = build_weight(ty.block_bytes, ty.f16_offsets, 0x5EED_0001 + ty.id);
        let t = GgmlType::from_id(ty.id, 128).unwrap();
        let mut w = vec![0f32; (N * K) as usize];
        dequant_to_f32(t, &raw, w.len(), &mut w).unwrap();
        let w_dev = upload(g, &raw);
        let k_rows = g.kernel(KQUANT_MODULE, "kquant_q8_1_rows_bf16").unwrap();
        let k_mmvq = g.kernel(KQUANT_MODULE, ty.mmvq).unwrap();
        for m in [1u32, 5] {
            let (bits, xf) = build_act(m as usize, 0xA5A5 + m);
            let x_bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
            let x_dev = upload(g, &x_bytes);
            let y_dev = g.alloc(kquant_q8_1_rows_bytes(m, K)).unwrap();
            let out_dev = g.alloc((m * N) as usize * 2).unwrap();
            kquant_q8_1_rows(g, k_rows, x_dev, y_dev, m, K, stream).unwrap();
            kquant_mmvq(g, k_mmvq, w_dev, y_dev, out_dev, N, K, m, stream).unwrap();
            g.synchronize(stream).unwrap();
            let got = download_bf16(g, out_dev, (m * N) as usize);
            let want = cpu_gemm(&w, &q8_emulate(&xf, 32), m as usize);
            assert_close(&format!("{} mmvq M={m}", ty.name), &got, &want);
            for p in [x_dev, y_dev, out_dev] {
                g.free(p).unwrap();
            }
        }
        g.free(w_dev).unwrap();
    }
}

#[test]
#[ignore = "requires a CUDA GB10 + the deepseek-v4-flash kernel target"]
fn kquant_prefill_mmq_matches_cpu_oracle() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let m = 200u32;
    for ty in &TYPES {
        let raw = build_weight(ty.block_bytes, ty.f16_offsets, 0x5EED_0100 + ty.id);
        let t = GgmlType::from_id(ty.id, 128).unwrap();
        let mut w = vec![0f32; (N * K) as usize];
        dequant_to_f32(t, &raw, w.len(), &mut w).unwrap();
        let w_dev = upload(g, &raw);
        let (bits, xf) = build_act(m as usize, 0xBEEF + ty.id);
        let x_bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let x_dev = upload(g, &x_bytes);
        let a_q8 = g.alloc(kquant_mmq_act_bytes(m, K)).unwrap();
        let out_dev = g.alloc((m * N) as usize * 2).unwrap();
        let k_quant = g.kernel(KQUANT_MODULE, ty.quant).unwrap();
        let k_nc = g.kernel(KQUANT_MODULE, ty.mmq_nc).unwrap();
        let k_wc = g.kernel(KQUANT_MODULE, ty.mmq_wc).unwrap();
        ops::quantize_act_q8_1(g, k_quant, x_dev, a_q8, m, K, stream).unwrap();
        kquant_mmq_gemm(
            g, k_nc, k_wc, a_q8, w_dev, out_dev, m, N, K, ty.smem, stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        let got = download_bf16(g, out_dev, (m * N) as usize);
        let want = cpu_gemm(&w, &q8_emulate(&xf, ty.mmq_act_block), m as usize);
        assert_close(&format!("{} mmq M={m}", ty.name), &got, &want);
        for p in [x_dev, a_q8, out_dev, w_dev] {
            g.free(p).unwrap();
        }
    }
}

/// The Q6_K output head (`kquant_mmvq_q6_k_w`, warp per row) against the CPU
/// dequant + q8_1-emulated GEMM, M = 1 (decode) and 5 (a batched head).
#[test]
#[ignore = "requires a CUDA GB10 + the deepseek-v4-flash kernel target"]
fn kquant_q6k_head_mmvq_w_matches_cpu_oracle() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    // Q6_K: { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d } = 210 B, d at 208.
    let raw = build_weight(Q6K_BLOCK_BYTES, &[208], 0x5EED_000E);
    let t = GgmlType::from_id(14, 128).unwrap();
    let mut w = vec![0f32; (N * K) as usize];
    dequant_to_f32(t, &raw, w.len(), &mut w).unwrap();
    let w_dev = upload(g, &raw);
    let k_rows = g.kernel(KQUANT_MODULE, "kquant_q8_1_rows_bf16").unwrap();
    let k_mmvq = g.kernel(KQUANT_MODULE, "kquant_mmvq_q6_k_w").unwrap();
    for m in [1u32, 5] {
        let (bits, xf) = build_act(m as usize, 0x6A6A + m);
        let x_bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let x_dev = upload(g, &x_bytes);
        let y_dev = g.alloc(kquant_q8_1_rows_bytes(m, K)).unwrap();
        let out_dev = g.alloc((m * N) as usize * 2).unwrap();
        kquant_q8_1_rows(g, k_rows, x_dev, y_dev, m, K, stream).unwrap();
        kquant_mmvq_w(g, k_mmvq, w_dev, y_dev, out_dev, N, K, m, stream).unwrap();
        g.synchronize(stream).unwrap();
        let got = download_bf16(g, out_dev, (m * N) as usize);
        let want = cpu_gemm(&w, &q8_emulate(&xf, 32), m as usize);
        assert_close(&format!("Q6_K mmvq_w M={m}"), &got, &want);
        for p in [x_dev, y_dev, out_dev] {
            g.free(p).unwrap();
        }
    }
    g.free(w_dev).unwrap();
}
