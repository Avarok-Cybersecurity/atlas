// SPDX-License-Identifier: AGPL-3.0-only

//! Host launch for K3 CUDA KDA decode (`kda_decode` PTX module).
//!
//! Two kernels, one token: conv-4 + SiLU, then L2 q/k + delta-rule.
//! CPU oracle: [`atlas_core::kimi_k3::kda_decode_token`]. Spark2 still
//! compiles the `.cu`; this path is the compare harness, not serve.

use anyhow::{Context, Result, bail};
use atlas_core::kimi_k3::{KDA_L2_EPS, KdaConfig, KdaState};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// PTX module stem = `kernels/gb10/kimi-k3/bf16/kda_decode.cu`.
pub const MODULE: &str = "kda_decode";
pub const CONV_ENTRY: &str = "k3_kda_conv_update_f32";
pub const RECURRENT_ENTRY: &str = "k3_kda_recurrent_step_f32";
const CONV_BLOCK: u32 = 128;

#[derive(Clone, Copy, Debug)]
pub struct K3KdaDecodeKernels {
    pub conv: KernelHandle,
    pub recurrent: KernelHandle,
}

impl K3KdaDecodeKernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            conv: gpu.kernel(MODULE, CONV_ENTRY)?,
            recurrent: gpu.kernel(MODULE, RECURRENT_ENTRY)?,
        })
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn up(gpu: &dyn GpuBackend, v: &[f32], hold: &mut Vec<DevicePtr>) -> Result<DevicePtr> {
    let b = f32_bytes(v);
    let p = gpu.alloc(b.len().max(1))?;
    hold.push(p);
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}

/// One-token CUDA KDA decode. Updates `state` in place. Same buffers as CPU.
pub fn launch_k3_kda_decode_token(
    gpu: &dyn GpuBackend,
    kernels: &K3KdaDecodeKernels,
    x_qkv: &[f32],
    conv_w: &[f32],
    gate: &[f32],
    beta: &[f32],
    cfg: &KdaConfig,
    state: &mut KdaState,
    stream: u64,
) -> Result<Vec<f32>> {
    let (h, d, k) = (cfg.heads, cfg.head_dim, cfg.conv_kernel);
    let c = cfg.conv_dim();
    if x_qkv.len() != c {
        bail!("k3 kda: x_qkv {} != conv_dim {c}", x_qkv.len());
    }
    if conv_w.len() != cfg.conv_elems() {
        bail!("k3 kda: conv_w {} != {}", conv_w.len(), cfg.conv_elems());
    }
    if gate.len() != cfg.qkv_dim() || beta.len() != h {
        bail!("k3 kda: gate/beta rank");
    }
    if state.conv.len() != cfg.conv_elems() || state.recurrent.len() != cfg.recurrent_elems() {
        bail!("k3 kda: state rank");
    }
    if k == 0 || d == 0 {
        bail!("k3 kda: D and conv_kernel must be > 0");
    }

    let mut hold = Vec::new();
    let run = (|| {
        let dx = up(gpu, x_qkv, &mut hold)?;
        let dw = up(gpu, conv_w, &mut hold)?;
        let dgate = up(gpu, gate, &mut hold)?;
        let dbeta = up(gpu, beta, &mut hold)?;
        let dconv = up(gpu, &state.conv, &mut hold)?;
        let drec = up(gpu, &state.recurrent, &mut hold)?;
        let dy = gpu.alloc((c * 4).max(1))?;
        hold.push(dy);
        let dout = gpu.alloc((cfg.qkv_dim() * 4).max(1))?;
        hold.push(dout);

        KernelLaunch::new(gpu, kernels.conv)
            .grid([div_ceil(c as u32, CONV_BLOCK), 1, 1])
            .block([CONV_BLOCK, 1, 1])
            .arg_ptr(dx)
            .arg_ptr(dw)
            .arg_ptr(dconv)
            .arg_ptr(dy)
            .arg_u32(c as u32)
            .arg_u32(k as u32)
            .launch(stream)
            .context("k3_kda_conv_update_f32")?;

        let rec_block = (d as u32).min(128);
        KernelLaunch::new(gpu, kernels.recurrent)
            .grid([h as u32, 1, 1])
            .block([rec_block, 1, 1])
            .shared_mem((3 * d * 4) as u32)
            .arg_ptr(dy)
            .arg_ptr(dgate)
            .arg_ptr(dbeta)
            .arg_ptr(drec)
            .arg_ptr(dout)
            .arg_u32(h as u32)
            .arg_u32(d as u32)
            .arg_f32(KDA_L2_EPS)
            .launch(stream)
            .context("k3_kda_recurrent_step_f32")?;
        gpu.synchronize(stream)?;

        let mut out_b = vec![0u8; cfg.qkv_dim() * 4];
        let mut conv_b = vec![0u8; cfg.conv_elems() * 4];
        let mut rec_b = vec![0u8; cfg.recurrent_elems() * 4];
        gpu.copy_d2h(dout, &mut out_b)?;
        gpu.copy_d2h(dconv, &mut conv_b)?;
        gpu.copy_d2h(drec, &mut rec_b)?;
        state.conv = bytes_f32(&conv_b);
        state.recurrent = bytes_f32(&rec_b);
        Ok(bytes_f32(&out_b))
    })();
    for p in hold {
        let _ = gpu.free(p);
    }
    run
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_core::kimi_k3::kda_decode_token;
    use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};

    #[test]
    fn resolve_looks_up_k3_entries() {
        let gpu = MockGpuBackend::new();
        let _ = K3KdaDecodeKernels::resolve(&gpu).unwrap();
        assert_eq!(
            gpu.kernel_lookups_snapshot(),
            vec![
                (MODULE.to_string(), CONV_ENTRY.to_string()),
                (MODULE.to_string(), RECURRENT_ENTRY.to_string()),
            ]
        );
    }

    #[test]
    fn mock_launch_contract_twin_geometry() {
        let gpu = MockGpuBackend::new();
        let k = K3KdaDecodeKernels::resolve(&gpu).unwrap();
        let cfg = KdaConfig::twin_0_40b();
        let mut state = KdaState::new(&cfg);
        let x = vec![0.1f32; cfg.conv_dim()];
        let w = vec![0.2f32; cfg.conv_elems()];
        let gate = vec![-0.5f32; cfg.qkv_dim()];
        let beta = vec![0.25f32; cfg.heads];
        let _ = launch_k3_kda_decode_token(&gpu, &k, &x, &w, &gate, &beta, &cfg, &mut state, 3)
            .unwrap();
        let launches = gpu.launches_snapshot();
        assert_eq!(launches.len(), 2, "conv then recurrent");
        let c = cfg.conv_dim() as u32;
        assert_eq!(launches[0].grid, [div_ceil(c, CONV_BLOCK), 1, 1]);
        assert_eq!(launches[0].block, [CONV_BLOCK, 1, 1]);
        assert_eq!(launches[0].shared_mem, 0);
        assert_eq!(launches[0].stream, 3);
        assert_eq!(launches[0].args.len(), 6);
        assert_eq!(
            launches[0].args[4],
            MockArg::Bytes(c.to_le_bytes().to_vec())
        );
        assert_eq!(
            launches[0].args[5],
            MockArg::Bytes((cfg.conv_kernel as u32).to_le_bytes().to_vec())
        );

        let rec = &launches[1];
        assert_eq!(rec.grid, [cfg.heads as u32, 1, 1]);
        assert_eq!(rec.block, [cfg.head_dim as u32, 1, 1]);
        assert_eq!(rec.shared_mem, (3 * cfg.head_dim * 4) as u32);
        assert_eq!(rec.args.len(), 8);
        assert_eq!(
            rec.args[7],
            MockArg::Bytes(KDA_L2_EPS.to_le_bytes().to_vec())
        );
    }

    #[test]
    fn cpu_oracle_is_the_compare_target() {
        let cfg = KdaConfig::twin_0_40b();
        let mut state = KdaState::new(&cfg);
        let x = vec![0.1f32; cfg.conv_dim()];
        let w = vec![0.2f32; cfg.conv_elems()];
        let gate = vec![-0.5f32; cfg.qkv_dim()];
        let beta = vec![0.25f32; cfg.heads];
        let y = kda_decode_token(&x, &w, &gate, &beta, &cfg, &mut state);
        assert_eq!(y.len(), cfg.qkv_dim());
        assert!(y.iter().any(|v| v.abs() > 1e-8));
    }
}
