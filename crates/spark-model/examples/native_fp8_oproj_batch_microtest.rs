// SPDX-License-Identifier: AGPL-3.0-only
//! Actual Qwen3.8 O-projection shape, existing scalar versus chunked batch4.
//! No new kernel math: require identical BF16 output bytes and intact guards.
use anyhow::{Result, ensure};
use half::bf16;
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const N: usize = 5120;
const K: usize = 6144;
const MAX_M: usize = 16;
const ORIGINAL_M: usize = 5;
const GUARD: usize = 64;

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
fn check_output(observed: &[u8], baseline: &[u8], sentinel: &[u8], bytes: usize) -> Result<()> {
    ensure!(
        observed.len() == sentinel.len() && baseline.len() == sentinel.len(),
        "output extent mismatch"
    );
    let actual = &observed[GUARD..GUARD + bytes];
    let expected = &baseline[GUARD..GUARD + bytes];
    ensure!(
        values(actual)
            .iter()
            .chain(values(expected).iter())
            .all(|x| x.is_finite()),
        "nonfinite projection output"
    );
    ensure!(
        observed[..GUARD] == sentinel[..GUARD]
            && observed[GUARD + bytes..] == sentinel[GUARD + bytes..],
        "batched row offset wrote outside output extent"
    );
    ensure!(
        baseline[..GUARD] == sentinel[..GUARD]
            && baseline[GUARD + bytes..] == sentinel[GUARD + bytes..],
        "scalar row offset wrote outside output extent"
    );
    ensure!(
        actual == expected,
        "batch output differs from production scalar BF16 bits"
    );
    Ok(())
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let scalar = gpu.kernel("w8a16_gemv", "w8a16_gemv")?;
    let batch4 = gpu.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch4")?;
    let batch16 = gpu.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch16")?;
    let mut state = 0x51a7_8a16_2026_u64;
    let mut random = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 32) as u32
    };
    let weights: Vec<u8> = (0..N * K)
        .map(|_| {
            let x = random();
            ((x % 127) as u8) | (((x >> 7) & 1) as u8 * 128)
        })
        .collect();
    let mut acts: Vec<u8> = (0..ORIGINAL_M * K)
        .flat_map(|_| {
            bf16::from_f32(((random() % 2049) as f32 - 1024.0) / 1024.0)
                .to_bits()
                .to_le_bytes()
        })
        .collect();
    let scales: Vec<u8> = (0..N / 128 * (K / 128))
        .flat_map(|_| (((random() % 16 + 1) as f32) / 1024.0).to_le_bytes())
        .collect();
    // Keep the observed M4 red fixture's first five rows and scales unchanged.
    // Additional rows are generated only after the original scale draws.
    acts.extend((0..(MAX_M - ORIGINAL_M) * K).flat_map(|_| {
        bf16::from_f32(((random() % 2049) as f32 - 1024.0) / 1024.0)
            .to_bits()
            .to_le_bytes()
    }));
    let weight = upload(&gpu, &weights)?;
    let input_base = upload(&gpu, &[vec![0x5a; GUARD], acts, vec![0x5a; GUARD]].concat())?;
    let input = input_base.offset(GUARD);
    let scale = upload(&gpu, &scales)?;
    let output_bytes = MAX_M * N * 2;
    let sentinel = vec![0x5a; output_bytes + 2 * GUARD];
    let scalar_base = upload(&gpu, &sentinel)?;
    let batch_base = upload(&gpu, &sentinel)?;
    let scalar_out = scalar_base.offset(GUARD);
    let batch_out = batch_base.offset(GUARD);
    let mut first_oracle = true;
    for m in [2_usize, 4, 5, 8, 16] {
        gpu.copy_h2d(&sentinel, scalar_base)?;
        gpu.copy_h2d(&sentinel, batch_base)?;
        for row in 0..m {
            ops::w8a16_gemv(
                &gpu,
                scalar,
                input.offset(row * K * 2),
                weight,
                scale,
                scalar_out.offset(row * N * 2),
                N as u32,
                K as u32,
                0,
            )?;
        }
        if m > 5 {
            // The same CUDA template also exports M<=16; exercise it directly.
            KernelLaunch::new(&gpu, batch16)
                .grid([N.div_ceil(4) as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(input)
                .arg_ptr(weight)
                .arg_ptr(scale)
                .arg_ptr(batch_out)
                .arg_u32(m as u32)
                .arg_u32(N as u32)
                .arg_u32(K as u32)
                .launch(0)?;
        } else {
            for first in (0..m).step_by(4) {
                ops::w8a16_gemv_batch4(
                    &gpu,
                    batch4,
                    input.offset(first * K * 2),
                    weight,
                    scale,
                    batch_out.offset(first * N * 2),
                    (m - first).min(4) as u32,
                    N as u32,
                    K as u32,
                    0,
                )?;
            }
        }
        gpu.synchronize(0)?;
        let mut baseline = vec![0_u8; sentinel.len()];
        let mut observed = vec![0_u8; sentinel.len()];
        gpu.copy_d2h(scalar_base, &mut baseline)?;
        gpu.copy_d2h(batch_base, &mut observed)?;
        let bytes = m * N * 2;
        let expected = &baseline[GUARD..GUARD + bytes];
        let actual = &observed[GUARD..GUARD + bytes];
        if first_oracle {
            for mutation in ["output-bit", "guard", "nonfinite"] {
                let mut bad = baseline.clone();
                match mutation {
                    "output-bit" => bad[GUARD] ^= 1,
                    "guard" => bad[0] ^= 1,
                    _ => bad[GUARD..GUARD + 2].copy_from_slice(&0x7fc0_u16.to_le_bytes()),
                }
                let error = check_output(&bad, &baseline, &sentinel, bytes)
                    .expect_err("known-bad output was admitted by the real comparison oracle");
                println!("KNOWN_BAD {mutation}: refused: {error}");
            }
            first_oracle = false;
        }
        let a = values(actual);
        let b = values(expected);
        let max_abs = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        let dot: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum();
        let norm_b: f64 = b.iter().map(|x| x * x).sum();
        let cosine = dot / (norm_a * norm_b).sqrt();
        let mismatches = actual
            .chunks_exact(2)
            .zip(expected.chunks_exact(2))
            .filter(|(x, y)| x != y)
            .count();
        println!(
            "M={m} N={N} K={K} unequal_bf16={mismatches} max_abs={max_abs:.9} cosine={cosine:.12}"
        );
        check_output(&observed, &baseline, &sentinel, bytes)?;
    }
    println!(
        "ALL PASS: actual O shape batch4 M2/M4/M5 and batch16 M8/M16, exact scalar equivalence and guards"
    );
    Ok(())
}
