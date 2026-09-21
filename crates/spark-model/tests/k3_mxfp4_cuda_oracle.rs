// SPDX-License-Identifier: AGPL-3.0-only

//! Real K3 non-transposed E8M0 grouped GEMM against independent CPU arithmetic.
//! Deployment preparation, not a benchmark certification or full-model test.
//! Run on an idle GB10 with an external deadline (CUDA synchronization may hang):
//! ```text
//! AVAROK_TARGET_HW=gb10 AVAROK_TARGET_MODEL=kimi-k3 AVAROK_TARGET_QUANT=mxfp4 \
//! cargo test -p spark-model --test k3_mxfp4_cuda_oracle --no-run
//! K3_ORACLE_GPU_ORDINAL=0 timeout 180s cargo test -p spark-model \
//! --test k3_mxfp4_cuda_oracle -- --ignored --nocapture --test-threads=1
//! ```
//! Keep the same AVAROK_TARGET_* values and target directory for both commands.

#![cfg(feature = "cuda")]

use anyhow::{Context, Result, ensure};
use avarok_core::config::parse_config;
use half::bf16;
use spark_model::kimi_k3::moe_cuda::{K3MoeGemmKernels, launch_k3_moe_e8m0_ptrtable};
use spark_model::layers::ops::moe_w4a16_grouped_gemm_ptrtable;
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

fn decoded(w: &Weight, row: usize, col: usize, k: usize) -> f64 {
    // Independent scalar E2M1 decode; no production quantization helpers.
    let byte = w.packed[row * (k / 2) + col / 2];
    let nibble = (byte >> (4 * (col % 2))) & 15;
    let magnitude = match nibble & 7 {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        3 => 1.5,
        4 => 2.0,
        5 => 3.0,
        6 => 4.0,
        _ => 6.0,
    };
    let sign = if nibble & 8 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from(w.scales[row * (k / 32) + col / 32]) - 127;
    sign * magnitude * 2.0f64.powi(exponent)
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    stream: u64,
    label: &str,
    n: usize,
    k: usize,
    counts: &[usize],
    salt: usize,
) -> Result<()> {
    ensure!(
        n > 0 && k > 0 && k.is_multiple_of(32),
        "packed group geometry"
    );
    let m: usize = counts.iter().sum();
    let mut memory = AllocationScope {
        gpu,
        pointers: Vec::new(),
    };
    let mut offsets = vec![0i32];
    for count in counts {
        offsets.push(offsets.last().unwrap() + *count as i32);
    }
    // Reversed token IDs prove A indirection while C stays in grouped order.
    let ids: Vec<i32> = (0..m as i32).rev().collect();
    let a: Vec<bf16> = (0..m * k)
        .map(|i| bf16::from_f32((((i * 17 + i / k * 11 + salt) % 33) as f32 - 16.0) / 16.0))
        .collect();
    let ap = memory.upload(
        &a.iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let off = memory.upload(
        &offsets
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let sti = memory.upload(&ids.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())?;
    // NaN sentinel catches any unwritten output, including edge tiles.
    let cp = memory.upload(&vec![0xff; m * n * 2])?;
    let weights: Vec<_> = (0..counts.len()).map(|e| weight(n, k, salt + e)).collect();
    let mut packed = Vec::new();
    for w in &weights {
        packed.push(QuantizedWeight {
            weight: memory.upload(&w.packed)?,
            weight_scale: memory.upload(&w.scales)?,
            weight_scale_2: 1.0,
            ..QuantizedWeight::null()
        });
    }
    if counts.iter().all(|&c| c == 1) {
        launch_k3_moe_e8m0_ptrtable(
            gpu,
            kernel,
            ap,
            &packed,
            cp,
            off,
            sti,
            counts.len() as u32,
            n as u32,
            k as u32,
            stream,
        )?;
    } else {
        let bp = memory.upload(
            &packed
                .iter()
                .flat_map(|w| w.weight.0.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let bs = memory.upload(
            &packed
                .iter()
                .flat_map(|w| w.weight_scale.0.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let s2 = memory.upload(
            &vec![1.0f32; counts.len()]
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        moe_w4a16_grouped_gemm_ptrtable(
            gpu,
            kernel,
            ap,
            bp,
            bs,
            s2,
            cp,
            off,
            sti,
            counts.len() as u32,
            n as u32,
            k as u32,
            counts.iter().copied().max().unwrap().div_ceil(64) as u32,
            stream,
        )?;
    }
    gpu.synchronize(stream)?;
    let mut raw = vec![0; m * n * 2];
    gpu.copy_d2h(cp, &mut raw)?;
    let mut max_error = 0.0f64;
    for (expert, w) in weights.iter().enumerate() {
        for row in offsets[expert] as usize..offsets[expert + 1] as usize {
            for col in 0..n {
                let sum: f64 = (0..k)
                    .map(|j| {
                        f64::from(a[ids[row] as usize * k + j].to_f32()) * decoded(w, col, j, k)
                    })
                    .sum();
                let index = 2 * (row * n + col);
                let actual = bf16::from_bits(u16::from_le_bytes([raw[index], raw[index + 1]]));
                let expected = bf16::from_f32(sum as f32);
                max_error = max_error.max((f64::from(actual.to_f32()) - sum).abs());
                // All products are multiples of 1/64, |A|<=1, |B|<=12,
                // K<=3584. Thus every FP32 partial sum is exact (absolute sum
                // < 2^24/64). Only final BF16 RNE rounding remains: require
                // identical values, treating +/- zero as equivalent.
                ensure!(
                    actual.to_f32() == expected.to_f32(),
                    "{label} N={n} K={k} row={row} col={col}: actual={actual} expected={expected} f64={sum}"
                );
            }
        }
    }
    println!(
        "PASS {label}: N={n} K={k} expert_rows={counts:?} outputs={} BF16 exact; max_rounding_error={max_error}",
        m * n
    );
    Ok(())
}

#[test]
#[ignore = "requires an explicitly selected idle CUDA device and compiled K3 MXFP4 kernels"]
fn k3_e8m0_production_shapes_match_cpu() -> Result<()> {
    let ordinal: usize = std::env::var("K3_ORACLE_GPU_ORDINAL")
        .context("set K3_ORACLE_GPU_ORDINAL explicitly")?
        .parse()?;
    let target = avarok_kernels::ptx_for_exact_target("kimi-k3", "mxfp4")
        .context("build AVAROK_TARGET_MODEL=kimi-k3 AVAROK_TARGET_QUANT=mxfp4")?;
    println!("K3 oracle compiled PTX architecture: {}", target.ptx_arch);
    let gpu = AvarokCudaBackend::new(ordinal, &target.modules)?;
    // Resolve through the production mandatory-handle path before any launch.
    let kernel = K3MoeGemmKernels::resolve(&gpu)?.ptrtable;
    let stream = gpu.create_stream()?;
    let config = parse_config(include_str!(
        "../../../docs/k3/fixtures/moonshotai-Kimi-K3-config.json"
    ))?;
    ensure!(
        config.moe_intermediate_size.is_multiple_of(8),
        "TP8 geometry"
    );
    let intermediate = config.moe_intermediate_size / 8;
    let latent = config.moe_latent_size;
    ensure!(
        latent <= 3584 && intermediate <= 3584,
        "update exact-sum bound before changing fixture"
    );
    for (salt, label, n, k) in [
        (1, "TP8 w1", intermediate, latent),
        (2, "TP8 w3", intermediate, latent),
        (3, "TP8 w2", latent, intermediate),
    ] {
        run_case(&gpu, kernel, stream, label, n, k, &[1, 1], salt)?;
        run_case(&gpu, kernel, stream, label, n, k, &[1, 0, 3], salt + 7)?;
    }
    run_case(
        &gpu,
        kernel,
        stream,
        "irregular M/N and second M tile",
        70,
        96,
        &[65, 0, 2],
        17,
    )
}

/// Independent launch schedule: three GEMMs, as before the gate/up batching.
/// Existing tests above establish each GEMM's independent CPU arithmetic.
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
