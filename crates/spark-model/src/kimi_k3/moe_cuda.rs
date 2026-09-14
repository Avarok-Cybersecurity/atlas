// SPDX-License-Identifier: AGPL-3.0-only

//! Host launch for K3 packed LatentMoE experts (`moe_w4a16` E8M0 ptrtable).
//!
//! Required handle: [`PTRTABLE_E8M0`] from DSV4 extra_cu. Lookup-fail bails;
//! packed tensors must not silently dequant on the host F32 twin path.

use std::collections::HashMap;

use anyhow::{Context, Result, bail, ensure};
use atlas_core::kimi_k3::{LatentMoeConfig, situ_glu_vec};
use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layers::ops::moe_w4a16_grouped_gemm_ptrtable;
use crate::weight_map::QuantizedWeight;

/// PTX module from kimi-k3 extra_cu of DSV4 `moe_w4a16_grouped_gemm.cu`.
pub const MODULE: &str = "moe_w4a16";
pub const PTRTABLE_E8M0: &str = "moe_w4a16_grouped_gemm_ptrtable_e8m0";
pub const E8M0_ENTRY: &str = PTRTABLE_E8M0;

#[derive(Clone, Copy, Debug)]
pub struct K3MoeGemmKernels {
    pub ptrtable: KernelHandle,
}

impl K3MoeGemmKernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            ptrtable: gpu.kernel(MODULE, PTRTABLE_E8M0).with_context(|| {
                format!(
                    "K3 MXFP4: {MODULE}::{PTRTABLE_E8M0} missing; \
                     packed experts cannot silently run host F32"
                )
            })?,
        })
    }
}

/// One grouped E8M0 GEMM: `C[M, N] = A[M, K] @ B_packed[N, K/2]` via ptrtable.
///
/// Each expert in `packed` owns one output row (`expert_offsets` 0..=num_experts).
#[allow(clippy::too_many_arguments)]
pub fn launch_k3_moe_e8m0_ptrtable(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    packed: &[QuantizedWeight],
    c: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        packed.len() == num_experts as usize && num_experts > 0,
        "K3 MXFP4: ptrtable length {} != num_experts {num_experts}",
        packed.len()
    );
    let mut hold = Vec::new();
    let run = (|| {
        let (packed_ptrs, scale_ptrs, scale2_vals) = upload_ptr_table(gpu, packed, &mut hold)?;
        moe_w4a16_grouped_gemm_ptrtable(
            gpu,
            kernel,
            a,
            packed_ptrs,
            scale_ptrs,
            scale2_vals,
            c,
            expert_offsets,
            sorted_token_ids,
            num_experts,
            n_out,
            k,
            1, // decode: one token per selected expert
            stream,
        )
        .context("moe_w4a16_grouped_gemm_ptrtable_e8m0")
    })();
    for p in hold {
        let _ = gpu.free(p);
    }
    run
}

/// Packed w1/w2/w3 SiTU mix. Same contract as [`atlas_core::kimi_k3::mix_routed_experts`].
#[allow(clippy::too_many_arguments)]
pub fn launch_k3_latent_moe_experts(
    gpu: &dyn GpuBackend,
    kernels: &K3MoeGemmKernels,
    packed: &[(String, QuantizedWeight)],
    latent: &[f32],
    ids: &[usize],
    mix_w: &[f32],
    cfg: &LatentMoeConfig,
    stream: u64,
) -> Result<Vec<f32>> {
    ensure!(
        latent.len() == cfg.latent,
        "K3 MXFP4: latent {} != {}",
        latent.len(),
        cfg.latent
    );
    ensure!(ids.len() == mix_w.len(), "K3 MXFP4: ids/weights rank");
    if ids.is_empty() {
        return Ok(vec![0.0; cfg.latent]);
    }
    let table = index_packed(packed)?;
    let w1 = gather_proj(&table, ids, 0, "w1")?;
    let w2 = gather_proj(&table, ids, 1, "w2")?;
    let w3 = gather_proj(&table, ids, 2, "w3")?;
    let m = ids.len();
    let a_w1: Vec<f32> = latent
        .iter()
        .copied()
        .cycle()
        .take(m * cfg.latent)
        .collect();
    let gate = gemm_rows(
        gpu,
        kernels,
        &a_w1,
        m,
        &w1,
        cfg.expert_hidden as u32,
        cfg.latent as u32,
        stream,
    )?;
    let up = gemm_rows(
        gpu,
        kernels,
        &a_w1,
        m,
        &w3,
        cfg.expert_hidden as u32,
        cfg.latent as u32,
        stream,
    )?;
    let mut mid = Vec::with_capacity(m * cfg.expert_hidden);
    for e in 0..m {
        let g = &gate[e * cfg.expert_hidden..(e + 1) * cfg.expert_hidden];
        let u = &up[e * cfg.expert_hidden..(e + 1) * cfg.expert_hidden];
        mid.extend(situ_glu_vec(g, u, cfg.situ_beta, cfg.situ_linear_beta));
    }
    let down_rows = gemm_rows(
        gpu,
        kernels,
        &mid,
        m,
        &w2,
        cfg.latent as u32,
        cfg.expert_hidden as u32,
        stream,
    )?;
    let mut mixed = vec![0.0f32; cfg.latent];
    for (e, &w) in mix_w.iter().enumerate() {
        let y = &down_rows[e * cfg.latent..(e + 1) * cfg.latent];
        for (acc, yy) in mixed.iter_mut().zip(y) {
            *acc += w * *yy;
        }
    }
    Ok(mixed)
}

fn gemm_rows(
    gpu: &dyn GpuBackend,
    kernels: &K3MoeGemmKernels,
    a_f32: &[f32],
    m: usize,
    packed: &[QuantizedWeight],
    n_out: u32,
    k: u32,
    stream: u64,
) -> Result<Vec<f32>> {
    ensure!(
        a_f32.len() == m * k as usize,
        "K3 MXFP4: A {} vs {m}x{k}",
        a_f32.len()
    );
    let mut hold = Vec::new();
    let run = (|| {
        let a = up_bf16(gpu, a_f32, &mut hold)?;
        let c = gpu.alloc((m * n_out as usize * 2).max(1))?;
        hold.push(c);
        let off: Vec<i32> = (0..=m as i32).collect();
        let ids: Vec<i32> = (0..m as i32).collect();
        let offsets = up_i32(gpu, &off, &mut hold)?;
        let sorted = up_i32(gpu, &ids, &mut hold)?;
        launch_k3_moe_e8m0_ptrtable(
            gpu,
            kernels.ptrtable,
            a,
            packed,
            c,
            offsets,
            sorted,
            m as u32,
            n_out,
            k,
            stream,
        )?;
        gpu.synchronize(stream)?;
        let mut raw = vec![0u8; m * n_out as usize * 2];
        gpu.copy_d2h(c, &mut raw)?;
        Ok(bf16_to_f32(&raw))
    })();
    for p in hold {
        let _ = gpu.free(p);
    }
    run
}

fn upload_ptr_table(
    gpu: &dyn GpuBackend,
    packed: &[QuantizedWeight],
    hold: &mut Vec<DevicePtr>,
) -> Result<(DevicePtr, DevicePtr, DevicePtr)> {
    let n = packed.len();
    let packed_bytes: Vec<u8> = packed
        .iter()
        .flat_map(|w| w.weight.0.to_le_bytes())
        .collect();
    let scale_bytes: Vec<u8> = packed
        .iter()
        .flat_map(|w| w.weight_scale.0.to_le_bytes())
        .collect();
    let scale2_bytes: Vec<u8> = packed
        .iter()
        .flat_map(|w| w.weight_scale_2.to_le_bytes())
        .collect();
    let packed_ptrs = gpu.alloc((n * 8).max(1))?;
    hold.push(packed_ptrs);
    gpu.copy_h2d(&packed_bytes, packed_ptrs)?;
    let scale_ptrs = gpu.alloc((n * 8).max(1))?;
    hold.push(scale_ptrs);
    gpu.copy_h2d(&scale_bytes, scale_ptrs)?;
    let scale2_vals = gpu.alloc((n * 4).max(1))?;
    hold.push(scale2_vals);
    gpu.copy_h2d(&scale2_bytes, scale2_vals)?;
    Ok((packed_ptrs, scale_ptrs, scale2_vals))
}

fn index_packed(
    packed: &[(String, QuantizedWeight)],
) -> Result<HashMap<(usize, usize), QuantizedWeight>> {
    let mut t = HashMap::new();
    for (prefix, w) in packed {
        let (id, proj) = parse_expert_proj(prefix)?;
        if t.insert((id, proj), *w).is_some() {
            bail!("K3 MXFP4: duplicate packed {prefix}");
        }
    }
    Ok(t)
}

fn parse_expert_proj(prefix: &str) -> Result<(usize, usize)> {
    let (head, proj) = prefix
        .rsplit_once('.')
        .with_context(|| format!("K3 MXFP4: expert prefix {prefix}"))?;
    let proj_i = match proj {
        "w1" => 0,
        "w2" => 1,
        "w3" => 2,
        _ => bail!("K3 MXFP4: expected w1|w2|w3 in {prefix}"),
    };
    let id_s = head
        .rsplit_once(".experts.")
        .map(|(_, id)| id)
        .with_context(|| format!("K3 MXFP4: experts.id in {prefix}"))?;
    let id: usize = id_s
        .parse()
        .with_context(|| format!("K3 MXFP4: expert id {id_s}"))?;
    Ok((id, proj_i))
}

fn gather_proj(
    table: &HashMap<(usize, usize), QuantizedWeight>,
    ids: &[usize],
    proj: usize,
    name: &str,
) -> Result<Vec<QuantizedWeight>> {
    ids.iter()
        .map(|&id| {
            table.get(&(id, proj)).copied().with_context(|| {
                format!(
                    "K3 MXFP4: expert {id} {name} missing; packed experts cannot silently run host F32"
                )
            })
        })
        .collect()
}

fn up_bf16(gpu: &dyn GpuBackend, v: &[f32], hold: &mut Vec<DevicePtr>) -> Result<DevicePtr> {
    let b: Vec<u8> = v
        .iter()
        .flat_map(|&f| bf16::from_f32(f).to_le_bytes())
        .collect();
    let p = gpu.alloc(b.len().max(1))?;
    hold.push(p);
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}

fn up_i32(gpu: &dyn GpuBackend, v: &[i32], hold: &mut Vec<DevicePtr>) -> Result<DevicePtr> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(b.len().max(1))?;
    hold.push(p);
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}

fn bf16_to_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};
    use spark_runtime::kernel_args::div_ceil;

    fn dummy_qw(gpu: &MockGpuBackend) -> QuantizedWeight {
        QuantizedWeight {
            weight: gpu.alloc(16).unwrap(),
            weight_scale: gpu.alloc(1).unwrap(),
            weight_scale_2: 1.0,
            input_scale: DevicePtr::NULL,
            weight_scale_2_vec: DevicePtr::NULL,
        }
    }

    #[test]
    fn resolve_looks_up_e8m0_ptrtable() {
        let gpu = MockGpuBackend::new();
        let _ = K3MoeGemmKernels::resolve(&gpu).unwrap();
        assert_eq!(
            gpu.kernel_lookups_snapshot(),
            vec![(MODULE.to_string(), PTRTABLE_E8M0.to_string())]
        );
    }

    #[test]
    fn deny_kernel_resolve_bails_not_silent_cpu() {
        let gpu = MockGpuBackend::new();
        gpu.deny_kernel(MODULE, PTRTABLE_E8M0);
        let err = K3MoeGemmKernels::resolve(&gpu).unwrap_err().to_string();
        assert!(
            err.contains(PTRTABLE_E8M0) && err.contains("cannot silently run host F32"),
            "{err}"
        );
        assert_eq!(gpu.launch_count(), 0);
    }

    #[test]
    fn mock_launch_contract_one_expert() {
        let gpu = MockGpuBackend::new();
        let k = K3MoeGemmKernels::resolve(&gpu).unwrap();
        let packed = dummy_qw(&gpu);
        let a = gpu.alloc(64).unwrap();
        let c = gpu.alloc(128).unwrap();
        let off = gpu.alloc(8).unwrap();
        let ids = gpu.alloc(4).unwrap();
        let n_out = 64u32;
        let kk = 32u32;
        launch_k3_moe_e8m0_ptrtable(&gpu, k.ptrtable, a, &[packed], c, off, ids, 1, n_out, kk, 3)
            .unwrap();
        let launches = gpu.launches_snapshot();
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].grid, [div_ceil(n_out, 64), 1, 1]);
        assert_eq!(launches[0].block, [128, 1, 1]);
        assert_eq!(launches[0].stream, 3);
        assert_eq!(launches[0].args.len(), 10);
        assert_eq!(
            launches[0].args[7],
            MockArg::Bytes(1u32.to_le_bytes().to_vec())
        );
        assert_eq!(
            launches[0].args[8],
            MockArg::Bytes(n_out.to_le_bytes().to_vec())
        );
        assert_eq!(
            launches[0].args[9],
            MockArg::Bytes(kk.to_le_bytes().to_vec())
        );
    }

    #[test]
    fn parse_k3_expert_prefix() {
        let p = "language_model.model.layers.12.block_sparse_moe.experts.7.w1";
        assert_eq!(parse_expert_proj(p).unwrap(), (7, 0));
        assert_eq!(
            parse_expert_proj("model.layers.1.block_sparse_moe.experts.0.w3").unwrap(),
            (0, 2)
        );
    }

    #[test]
    fn empty_ids_does_not_launch() {
        let gpu = MockGpuBackend::new();
        let k = K3MoeGemmKernels::resolve(&gpu).unwrap();
        let cfg = LatentMoeConfig {
            hidden: 2,
            latent: 2,
            expert_hidden: 2,
            n_routed: 1,
            top_k: 1,
            n_shared: 0,
            situ_beta: 4.0,
            situ_linear_beta: 25.0,
            use_norm: false,
            renormalize: true,
        };
        let y =
            launch_k3_latent_moe_experts(&gpu, &k, &[], &[1.0, 0.0], &[], &[], &cfg, 0).unwrap();
        assert_eq!(y, vec![0.0, 0.0]);
        assert_eq!(gpu.launch_count(), 0);
    }

    #[test]
    fn missing_packed_proj_bails_not_cpu() {
        let gpu = MockGpuBackend::new();
        let k = K3MoeGemmKernels::resolve(&gpu).unwrap();
        let cfg = LatentMoeConfig {
            hidden: 2,
            latent: 2,
            expert_hidden: 2,
            n_routed: 1,
            top_k: 1,
            n_shared: 0,
            situ_beta: 4.0,
            situ_linear_beta: 25.0,
            use_norm: false,
            renormalize: true,
        };
        let packed = [(
            "model.layers.1.block_sparse_moe.experts.0.w1".to_string(),
            dummy_qw(&gpu),
        )];
        let before = gpu.launch_count();
        let err =
            launch_k3_latent_moe_experts(&gpu, &k, &packed, &[1.0, 0.0], &[0], &[1.0], &cfg, 0)
                .unwrap_err()
                .to_string();
        assert!(
            err.contains("w2") && err.contains("cannot silently run host F32"),
            "{err}"
        );
        assert_eq!(gpu.launch_count(), before);
    }
}
