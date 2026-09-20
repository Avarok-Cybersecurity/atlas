// SPDX-License-Identifier: AGPL-3.0-only
//! Independent f64 reference for the opt-in FP32-I/O K3 dense/shared MLP.
#![cfg(feature = "cuda")]
use anyhow::{Context, Result, ensure};
use half::bf16;
use spark_model::kimi_k3::dense_cuda::launch_dense_matrices;
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

struct Memory<'a> {
    gpu: &'a dyn GpuBackend,
    allocations: Vec<DevicePtr>,
}
impl Memory<'_> {
    fn upload(&mut self, values: &[f32], dtype: u32) -> Result<DevicePtr> {
        let bytes: Vec<u8> = if dtype == 0 {
            values.iter().flat_map(|v| v.to_le_bytes()).collect()
        } else {
            values
                .iter()
                .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
                .collect()
        };
        let ptr = self.gpu.alloc(bytes.len())?;
        self.allocations.push(ptr);
        self.gpu.copy_h2d(&bytes, ptr)?;
        Ok(ptr)
    }
}
impl Drop for Memory<'_> {
    fn drop(&mut self) {
        for p in self.allocations.drain(..).rev() {
            let _ = self.gpu.free(p);
        }
    }
}
fn values(n: usize, salt: u64, dtype: u32) -> Vec<f32> {
    let mut state = salt;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let x = ((state % 1009) as f32 - 504.0) / 2048.0;
            if dtype == 0 {
                x
            } else {
                bf16::from_f32(x).to_f32()
            }
        })
        .collect()
}

#[test]
#[ignore = "requires explicitly selected idle CUDA GPU and B200 K3 kernels"]
fn k3_resident_dense_mlp_matches_independent_f64() -> Result<()> {
    let ordinal = std::env::var("K3_ORACLE_GPU_ORDINAL")?.parse()?;
    let target = avarok_kernels::ptx_for_exact_target("kimi-k3", "mxfp4").context("K3 target")?;
    let gpu = AvarokCudaBackend::new(ordinal, &target.modules)?;
    let stream = gpu.create_stream()?;
    for (hidden, inter) in [(47, 33), (96, 64), (1024, 2048), (4096, 512)] {
        for dtype in [0, 1] {
            let x = values(hidden, 9357, 0);
            let gate = values(hidden * inter, 74401, dtype);
            let up = values(hidden * inter, 22879, dtype);
            let down = values(hidden * inter, 53407, dtype);
            let mut mid = vec![0f64; inter];
            for row in 0..inter {
                let g: f64 = (0..hidden)
                    .map(|i| f64::from(x[i]) * f64::from(gate[row * hidden + i]))
                    .sum();
                let u: f64 = (0..hidden)
                    .map(|i| f64::from(x[i]) * f64::from(up[row * hidden + i]))
                    .sum();
                mid[row] = 4.0 * (g / 4.0).tanh() / (1.0 + (-g).exp()) * (25.0 * (u / 25.0).tanh());
            }
            let expected: Vec<f64> = (0..hidden)
                .map(|row| {
                    (0..inter)
                        .map(|i| mid[i] * f64::from(down[row * inter + i]))
                        .sum()
                })
                .collect();
            let mut memory = Memory {
                gpu: &gpu,
                allocations: Vec::new(),
            };
            let g = memory.upload(&gate, dtype)?;
            let u = memory.upload(&up, dtype)?;
            let d = memory.upload(&down, dtype)?;
            let actual = launch_dense_matrices(
                &gpu, &x, g, u, d, dtype, dtype, hidden, inter, 4.0, 25.0, stream,
            )?;
            let mut max_error = 0.0f64;
            for (i, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
                let error = (f64::from(got) - want).abs();
                max_error = max_error.max(error);
                ensure!(
                    got.is_finite() && error <= 2e-4 + want.abs() * 2e-4,
                    "dense dtype={dtype} shape={hidden}x{inter} index={i}: got={got} expected={want} error={error}"
                );
            }
            println!(
                "PASS dense dtype={dtype} hidden={hidden} inter={inter} max_abs_error={max_error}"
            );
        }
    }
    Ok(())
}
