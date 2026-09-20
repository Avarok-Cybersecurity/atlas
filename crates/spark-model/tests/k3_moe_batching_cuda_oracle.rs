// SPDX-License-Identifier: AGPL-3.0-only

//! K3 selected-expert regression against the pre-batching three-GEMM schedule.
//! Deployment preparation, not a benchmark certification or full-model test.
//! Run on an idle GB10 with an external deadline (CUDA synchronization may hang):
//! ```text
//! AVAROK_TARGET_HW=gb10 AVAROK_TARGET_MODEL=kimi-k3 AVAROK_TARGET_QUANT=mxfp4 \
//! cargo test -p spark-model --test k3_moe_batching_cuda_oracle --no-run
//! K3_ORACLE_GPU_ORDINAL=0 timeout 180s cargo test -p spark-model \
//! --test k3_moe_batching_cuda_oracle -- --ignored --nocapture --test-threads=1
//! ```
//! Keep the same AVAROK_TARGET_* values and target directory for both commands.

#![cfg(feature = "cuda")]

use anyhow::{Context, Result, ensure};
use half::bf16;
use spark_model::kimi_k3::moe_cuda::{K3MoeGemmKernels, launch_k3_moe_e8m0_ptrtable};
use spark_model::weight_map::QuantizedWeight;
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

struct AllocationScope<'a> {
    gpu: &'a dyn GpuBackend,
    pointers: Vec<DevicePtr>,
}

impl AllocationScope<'_> {
    fn upload(&mut self, bytes: &[u8]) -> Result<DevicePtr> {
        let p = self.gpu.alloc(bytes.len())?;
        self.pointers.push(p);
        self.gpu.copy_h2d(bytes, p)?;
        Ok(p)
    }
}

impl Drop for AllocationScope<'_> {
    fn drop(&mut self) {
        for p in self.pointers.drain(..).rev() {
            let _ = self.gpu.free(p);
        }
    }
}

struct Weight {
    packed: Vec<u8>,
    scales: Vec<u8>,
}

fn weight(n: usize, k: usize, salt: usize) -> Weight {
    let mut packed = vec![0; n * k / 2];
    let mut state = 0x1234_5678_9abc_def0u64 ^ salt as u64;
    for byte in &mut packed {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *byte = (state >> 32) as u8;
    }
    // Different consecutive scale groups and expert/output rows. Every scale
    // is finite, normal and exactly representable: 2^-1, 2^0 and 2^1.
    let scales = (0..n * (k / 32))
        .map(|i| 126 + ((i / (k / 32) * 2 + i % (k / 32) + salt) % 3) as u8)
        .collect();
    Weight { packed, scales }
}

/// Independent launch schedule: three GEMMs, as before the gate/up batching.
/// Compares the production batched schedule against separate gate/up/down launches.
fn separate_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    weights: &[QuantizedWeight],
    input: &[f32],
    n: usize,
    k: usize,
    stream: u64,
) -> Result<Vec<f32>> {
    let m = weights.len();
    let mut mem = AllocationScope {
        gpu,
        pointers: Vec::new(),
    };
    let a = mem.upload(
        &input
            .iter()
            .flat_map(|&x| bf16::from_f32(x).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let c = mem.upload(&vec![0xff; m * n * 2])?;
    let off = mem.upload(
        &(0..=m as i32)
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>(),
    )?;
    let ids = mem.upload(&(0..m as i32).flat_map(i32::to_le_bytes).collect::<Vec<_>>())?;
    launch_k3_moe_e8m0_ptrtable(
        gpu, kernel, a, weights, c, off, ids, m as u32, n as u32, k as u32, stream,
    )?;
    gpu.synchronize(stream)?;
    let mut raw = vec![0; m * n * 2];
    gpu.copy_d2h(c, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect())
}

#[test]
#[ignore = "requires an explicitly selected idle CUDA device and compiled K3 MXFP4 kernels"]
fn k3_gate_up_batching_preserves_selected_expert_pipeline() -> Result<()> {
    use avarok_core::kimi_k3::{LatentMoeConfig, situ_glu_vec};
    use spark_model::kimi_k3::moe_cuda::launch_k3_latent_moe_experts;
    let ordinal = std::env::var("K3_ORACLE_GPU_ORDINAL")?.parse()?;
    let target = avarok_kernels::ptx_for_exact_target("kimi-k3", "mxfp4").context("K3 target")?;
    let gpu = AvarokCudaBackend::new(ordinal, &target.modules)?;
    let kernels = K3MoeGemmKernels::resolve(&gpu)?;
    let stream = gpu.create_stream()?;
    let cfg = LatentMoeConfig {
        hidden: 96,
        latent: 96,
        expert_hidden: 64,
        n_routed: 4,
        top_k: 3,
        n_shared: 0,
        situ_beta: 4.0,
        situ_linear_beta: 25.0,
        use_norm: false,
        renormalize: true,
    };
    let mut mem = AllocationScope {
        gpu: &gpu,
        pointers: Vec::new(),
    };
    let mut all = Vec::new();
    for e in 0..4 {
        for (p, n, k) in [("w1", 64, 96), ("w2", 96, 64), ("w3", 64, 96)] {
            let salt = e * 7
                + match p {
                    "w1" => 1,
                    "w2" => 2,
                    _ => 3,
                };
            let w = weight(n, k, salt);
            all.push((
                format!("model.layers.1.block_sparse_moe.experts.{e}.{p}"),
                QuantizedWeight {
                    weight: mem.upload(&w.packed)?,
                    weight_scale: mem.upload(&w.scales)?,
                    weight_scale_2: 1.0,
                    ..QuantizedWeight::null()
                },
            ));
        }
    }
    let input: Vec<f32> = (0..96)
        .map(|i| ((i * 13 % 31) as f32 - 15.0) / 64.0)
        .collect();
    for selected in [vec![3], vec![2, 0], vec![3, 1, 0]] {
        let m = selected.len();
        let mixed: Vec<f32> = (0..m).map(|i| (i + 1) as f32 / 6.0).collect();
        let select = |p: usize| {
            selected
                .iter()
                .map(|&e| all[e * 3 + p].1)
                .collect::<Vec<_>>()
        };
        let repeated: Vec<f32> = input.iter().copied().cycle().take(m * 96).collect();
        let gate = separate_rows(
            &gpu,
            kernels.ptrtable,
            &select(0),
            &repeated,
            64,
            96,
            stream,
        )?;
        let up = separate_rows(
            &gpu,
            kernels.ptrtable,
            &select(2),
            &repeated,
            64,
            96,
            stream,
        )?;
        let mut middle = Vec::new();
        for e in 0..m {
            middle.extend(situ_glu_vec(
                &gate[e * 64..(e + 1) * 64],
                &up[e * 64..(e + 1) * 64],
                4.0,
                25.0,
            ));
        }
        let down = separate_rows(&gpu, kernels.ptrtable, &select(1), &middle, 96, 64, stream)?;
        let mut expected = vec![0.0f32; 96];
        for e in 0..m {
            for i in 0..96 {
                expected[i] += mixed[e] * down[e * 96 + i];
            }
        }
        let actual = launch_k3_latent_moe_experts(
            &gpu, &kernels, &all, &input, &selected, &mixed, &cfg, stream,
        )?;
        ensure!(
            actual.iter().all(|x| x.is_finite()),
            "nonfinite selected-expert output"
        );
        ensure!(
            actual == expected,
            "batched gate/up changed selected-expert output: {selected:?}"
        );
        println!(
            "PASS selected={selected:?}: gate/up batching matches separate three-GEMM pipeline exactly"
        );
    }
    Ok(())
}
