// SPDX-License-Identifier: AGPL-3.0-only

//! Host launch for K3 CUDA gated-NoPE MLA decode (`mla_decode` PTX module).
//!
//! Two kernels, one token: maybe_rope (NoPE skips rotate), then SDPA + gate.
//! CPU oracle: [`avarok_core::kimi_k3::mla_decode_token`]. BoundLayer serve
//! FullAttention default is this launch (`K3_CUDA_MLA=0` keeps CPU).

use anyhow::{Context, Result, bail, ensure};
use avarok_core::kimi_k3::{MlaConfig, MlaKv};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// PTX module stem = `kernels/gb10/kimi-k3/bf16/mla_decode.cu`.
pub const MODULE: &str = "mla_decode";
pub const ROPE_ENTRY: &str = "k3_mla_maybe_rope_f32";
pub const SDPA_ENTRY: &str = "k3_mla_sdpa_gate_f32";
const ROPE_BLOCK: u32 = 128;
const SDPA_BLOCK: u32 = 32;

#[derive(Clone, Copy, Debug)]
pub struct K3MlaDecodeKernels {
    pub rope: KernelHandle,
    pub sdpa: KernelHandle,
}

impl K3MlaDecodeKernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            rope: gpu.kernel(MODULE, ROPE_ENTRY)?,
            sdpa: gpu.kernel(MODULE, SDPA_ENTRY)?,
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

/// Device-resident MLA KV. Append is one-row D2D/H2D; SDPA reads the buffer.
/// Host `MlaKv` remains the CPU oracle. Do not re-upload `[0..T]` each token.
pub struct MlaDeviceKv {
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub seq_len: usize,
    pub cap: usize,
    k_row: usize,
    v_row: usize,
}

impl MlaDeviceKv {
    pub fn k_row(&self) -> usize {
        self.k_row
    }

    pub fn v_row(&self) -> usize {
        self.v_row
    }

    pub fn alloc(gpu: &dyn GpuBackend, cap: usize, k_row: usize, v_row: usize) -> Result<Self> {
        ensure!(k_row > 0 && v_row > 0, "k3 mla: resident row rank");
        let cap = cap.max(1);
        let k = gpu.alloc((cap * k_row * 4).max(1))?;
        let v = match gpu.alloc((cap * v_row * 4).max(1)) {
            Ok(ptr) => ptr,
            Err(error) => {
                let _ = gpu.free(k);
                return Err(error);
            }
        };
        Ok(Self {
            k,
            v,
            seq_len: 0,
            cap,
            k_row,
            v_row,
        })
    }

    pub fn alloc_and_upload(
        gpu: &dyn GpuBackend,
        host: &MlaKv,
        cap: usize,
        k_row: usize,
        v_row: usize,
    ) -> Result<Self> {
        ensure!(
            host.k.len() == host.seq_len.saturating_mul(k_row)
                && host.v.len() == host.seq_len.saturating_mul(v_row),
            "k3 mla: host KV rank"
        );
        ensure!(
            cap >= host.seq_len.max(1),
            "k3 mla: device KV cap {cap} < seq {}",
            host.seq_len
        );
        let mut kv = Self::alloc(gpu, cap, k_row, v_row)?;
        if host.seq_len == 0 {
            return Ok(kv);
        }
        let upload = gpu
            .copy_h2d(&f32_bytes(&host.k), kv.k)
            .and_then(|()| gpu.copy_h2d(&f32_bytes(&host.v), kv.v));
        if let Err(error) = upload {
            let _ = kv.free(gpu);
            return Err(error);
        }
        kv.seq_len = host.seq_len;
        Ok(kv)
    }

    pub fn validate_cfg(&self, cfg: &MlaConfig) -> Result<()> {
        let k_row = cfg.heads * cfg.qk_head_dim();
        let v_row = cfg.heads * cfg.v_head_dim;
        ensure!(
            self.k_row == k_row && self.v_row == v_row,
            "k3 mla: resident KV rank"
        );
        Ok(())
    }

    pub fn download(&self, gpu: &dyn GpuBackend, host: &mut MlaKv) -> Result<()> {
        let n = self.seq_len;
        let mut kb = vec![0u8; n.saturating_mul(self.k_row) * 4];
        let mut vb = vec![0u8; n.saturating_mul(self.v_row) * 4];
        if n > 0 {
            gpu.copy_d2h(self.k, &mut kb)?;
            gpu.copy_d2h(self.v, &mut vb)?;
        }
        host.k = bytes_f32(&kb);
        host.v = bytes_f32(&vb);
        host.seq_len = n;
        Ok(())
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        let k = gpu.free(self.k);
        let v = gpu.free(self.v);
        k.and(v)
    }

    fn append_from_device(
        &mut self,
        gpu: &dyn GpuBackend,
        k_new: DevicePtr,
        v_host: &[f32],
    ) -> Result<()> {
        if self.seq_len >= self.cap {
            bail!("k3 mla: device KV cap {} full", self.cap);
        }
        ensure!(
            v_host.len() == self.v_row,
            "k3 mla: V row rank {} != {}",
            v_host.len(),
            self.v_row
        );
        let k_off = DevicePtr(self.k.0 + (self.seq_len * self.k_row * 4) as u64);
        gpu.copy_d2d(k_new, k_off, self.k_row * 4)?;
        let v_off = DevicePtr(self.v.0 + (self.seq_len * self.v_row * 4) as u64);
        gpu.copy_h2d(&f32_bytes(v_host), v_off)?;
        self.seq_len += 1;
        Ok(())
    }
}

/// One-token CUDA gated-NoPE MLA. Updates `q`/`k` (rope) and `kv` (append).
#[allow(clippy::too_many_arguments)]
pub fn launch_k3_mla_decode_token(
    gpu: &dyn GpuBackend,
    kernels: &K3MlaDecodeKernels,
    q: &mut [f32],
    k: &mut [f32],
    v: &[f32],
    g: &[f32],
    kv: &mut MlaKv,
    cfg: &MlaConfig,
    pos: usize,
    theta: f32,
    stream: u64,
) -> Result<Vec<f32>> {
    let (h, dq, dv) = (cfg.heads, cfg.qk_head_dim(), cfg.v_head_dim);
    if q.len() != h * dq || k.len() != h * dq {
        bail!("k3 mla: q/k rank");
    }
    if v.len() != h * dv || g.len() != h * dv {
        bail!("k3 mla: v/g rank");
    }
    if dq == 0 {
        bail!("k3 mla: dq must be > 0");
    }

    let mut hold = Vec::new();
    let run = (|| {
        let dq_ptr = up(gpu, q, &mut hold)?;
        let dk_new = up(gpu, k, &mut hold)?;
        KernelLaunch::new(gpu, kernels.rope)
            .grid([div_ceil(h as u32, ROPE_BLOCK), 1, 1])
            .block([ROPE_BLOCK, 1, 1])
            .arg_ptr(dq_ptr)
            .arg_ptr(dk_new)
            .arg_u32(h as u32)
            .arg_u32(cfg.qk_nope_head_dim as u32)
            .arg_u32(cfg.qk_rope_head_dim as u32)
            .arg_u32(pos as u32)
            .arg_f32(theta)
            .arg_u32(u32::from(cfg.mla_use_nope))
            .launch(stream)
            .context("k3_mla_maybe_rope_f32")?;
        gpu.synchronize(stream)?;

        let mut qb = vec![0u8; q.len() * 4];
        let mut kb = vec![0u8; k.len() * 4];
        gpu.copy_d2h(dq_ptr, &mut qb)?;
        gpu.copy_d2h(dk_new, &mut kb)?;
        q.copy_from_slice(&bytes_f32(&qb));
        k.copy_from_slice(&bytes_f32(&kb));
        kv.append(k, v);

        let dq2 = up(gpu, q, &mut hold)?;
        let dk = up(gpu, &kv.k, &mut hold)?;
        let dv_ptr = up(gpu, &kv.v, &mut hold)?;
        let dg = up(gpu, g, &mut hold)?;
        let dout = gpu.alloc((h * dv * 4).max(1))?;
        hold.push(dout);
        KernelLaunch::new(gpu, kernels.sdpa)
            .grid([h as u32, 1, 1])
            .block([SDPA_BLOCK, 1, 1])
            .arg_ptr(dq2)
            .arg_ptr(dk)
            .arg_ptr(dv_ptr)
            .arg_ptr(dg)
            .arg_ptr(dout)
            .arg_u32(kv.seq_len as u32)
            .arg_u32(h as u32)
            .arg_u32(dq as u32)
            .arg_u32(dv as u32)
            .arg_u32(u32::from(cfg.mla_use_output_gate))
            .launch(stream)
            .context("k3_mla_sdpa_gate_f32")?;
        gpu.synchronize(stream)?;

        let mut out_b = vec![0u8; h * dv * 4];
        gpu.copy_d2h(dout, &mut out_b)?;
        Ok(bytes_f32(&out_b))
    })();
    for p in hold {
        let _ = gpu.free(p);
    }
    run
}

/// Device-resident KV: rope stays on device, append is one row, one D2H (output).
#[allow(clippy::too_many_arguments)]
pub fn launch_k3_mla_decode_token_on_device(
    gpu: &dyn GpuBackend,
    kernels: &K3MlaDecodeKernels,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    kv: &mut MlaDeviceKv,
    cfg: &MlaConfig,
    pos: usize,
    theta: f32,
    stream: u64,
) -> Result<Vec<f32>> {
    let (h, dq, dv) = (cfg.heads, cfg.qk_head_dim(), cfg.v_head_dim);
    if q.len() != h * dq || k.len() != h * dq || v.len() != h * dv || g.len() != h * dv {
        bail!("k3 mla: q/k/v/g rank");
    }
    let mut hold = Vec::new();
    let run = (|| {
        let dq_ptr = up(gpu, q, &mut hold)?;
        let dk_new = up(gpu, k, &mut hold)?;
        KernelLaunch::new(gpu, kernels.rope)
            .grid([div_ceil(h as u32, ROPE_BLOCK), 1, 1])
            .block([ROPE_BLOCK, 1, 1])
            .arg_ptr(dq_ptr)
            .arg_ptr(dk_new)
            .arg_u32(h as u32)
            .arg_u32(cfg.qk_nope_head_dim as u32)
            .arg_u32(cfg.qk_rope_head_dim as u32)
            .arg_u32(pos as u32)
            .arg_f32(theta)
            .arg_u32(u32::from(cfg.mla_use_nope))
            .launch(stream)
            .context("k3_mla_maybe_rope_f32")?;
        kv.append_from_device(gpu, dk_new, v)?;
        let dg = up(gpu, g, &mut hold)?;
        let dout = gpu.alloc((h * dv * 4).max(1))?;
        hold.push(dout);
        KernelLaunch::new(gpu, kernels.sdpa)
            .grid([h as u32, 1, 1])
            .block([SDPA_BLOCK, 1, 1])
            .arg_ptr(dq_ptr)
            .arg_ptr(kv.k)
            .arg_ptr(kv.v)
            .arg_ptr(dg)
            .arg_ptr(dout)
            .arg_u32(kv.seq_len as u32)
            .arg_u32(h as u32)
            .arg_u32(dq as u32)
            .arg_u32(dv as u32)
            .arg_u32(u32::from(cfg.mla_use_output_gate))
            .launch(stream)
            .context("k3_mla_sdpa_gate_f32")?;
        gpu.synchronize(stream)?;
        let mut out_b = vec![0u8; h * dv * 4];
        gpu.copy_d2h(dout, &mut out_b)?;
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
    use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};

    #[test]
    fn resolve_looks_up_k3_entries() {
        let gpu = MockGpuBackend::new();
        let _ = K3MlaDecodeKernels::resolve(&gpu).unwrap();
        assert_eq!(
            gpu.kernel_lookups_snapshot(),
            vec![
                (MODULE.to_string(), ROPE_ENTRY.to_string()),
                (MODULE.to_string(), SDPA_ENTRY.to_string()),
            ]
        );
    }

    #[test]
    fn mock_launch_contract_twin_geometry() {
        let gpu = MockGpuBackend::new();
        let kernels = K3MlaDecodeKernels::resolve(&gpu).unwrap();
        let cfg = MlaConfig::twin_0_40b();
        let mut q = vec![0.1f32; cfg.heads * cfg.qk_head_dim()];
        let mut k = vec![0.2f32; cfg.heads * cfg.qk_head_dim()];
        let v = vec![0.3f32; cfg.heads * cfg.v_head_dim];
        let g = vec![0.0f32; cfg.heads * cfg.v_head_dim];
        let mut kv = MlaKv::default();
        let _ = launch_k3_mla_decode_token(
            &gpu, &kernels, &mut q, &mut k, &v, &g, &mut kv, &cfg, 3, 10000.0, 3,
        )
        .unwrap();
        assert_eq!(kv.seq_len, 1);
        let launches = gpu.launches_snapshot();
        assert_eq!(launches.len(), 2, "rope then sdpa_gate");
        assert_eq!(
            launches[0].grid,
            [div_ceil(cfg.heads as u32, ROPE_BLOCK), 1, 1]
        );
        assert_eq!(launches[0].block, [ROPE_BLOCK, 1, 1]);
        assert_eq!(launches[0].args.len(), 8);
        assert_eq!(
            launches[0].args[7],
            MockArg::Bytes(1u32.to_le_bytes().to_vec()),
            "twin is NoPE"
        );
        let sdpa = &launches[1];
        assert_eq!(sdpa.grid, [cfg.heads as u32, 1, 1]);
        assert_eq!(sdpa.block, [SDPA_BLOCK, 1, 1]);
        assert_eq!(sdpa.shared_mem, 0);
        assert_eq!(sdpa.stream, 3);
        assert_eq!(sdpa.args.len(), 10);
        assert_eq!(
            sdpa.args[9],
            MockArg::Bytes(1u32.to_le_bytes().to_vec()),
            "twin uses output gate"
        );
    }

    #[test]
    fn device_kv_second_token_d2h_is_output_only() {
        let gpu = MockGpuBackend::new();
        let kernels = K3MlaDecodeKernels::resolve(&gpu).unwrap();
        let cfg = MlaConfig::twin_0_40b();
        let q = vec![0.1f32; cfg.heads * cfg.qk_head_dim()];
        let k = vec![0.2f32; cfg.heads * cfg.qk_head_dim()];
        let v = vec![0.3f32; cfg.heads * cfg.v_head_dim];
        let g = vec![0.0f32; cfg.heads * cfg.v_head_dim];
        let mut kv = MlaDeviceKv::alloc(
            &gpu,
            8,
            cfg.heads * cfg.qk_head_dim(),
            cfg.heads * cfg.v_head_dim,
        )
        .unwrap();
        let _ = launch_k3_mla_decode_token_on_device(
            &gpu, &kernels, &q, &k, &v, &g, &mut kv, &cfg, 0, 10000.0, 0,
        )
        .unwrap();
        let before = gpu.d2h_blocking_count();
        let _ = launch_k3_mla_decode_token_on_device(
            &gpu, &kernels, &q, &k, &v, &g, &mut kv, &cfg, 1, 10000.0, 0,
        )
        .unwrap();
        let pulled = gpu.d2h_blocking_count() - before;
        assert_eq!(
            pulled, 1,
            "resident MLA must not re-download KV; D2H is output only (got {pulled})"
        );
        assert_eq!(kv.seq_len, 2);
    }

    #[test]
    fn host_kv_wrapper_h2d_grows_with_history_known_bad() {
        // Oracle: the old serving wrapper re-uploads [0..T] each token.
        // This known-bad case must keep failing so the resident path is not
        // confused with it.
        let gpu = MockGpuBackend::new();
        let kernels = K3MlaDecodeKernels::resolve(&gpu).unwrap();
        let cfg = MlaConfig::twin_0_40b();
        let mut q = vec![0.1f32; cfg.heads * cfg.qk_head_dim()];
        let mut k = vec![0.2f32; cfg.heads * cfg.qk_head_dim()];
        let v = vec![0.3f32; cfg.heads * cfg.v_head_dim];
        let g = vec![0.0f32; cfg.heads * cfg.v_head_dim];
        let mut kv = MlaKv::default();
        let _ = launch_k3_mla_decode_token(
            &gpu, &kernels, &mut q, &mut k, &v, &g, &mut kv, &cfg, 0, 10000.0, 0,
        )
        .unwrap();
        let after_t0 = gpu.h2d_bytes();
        let _ = launch_k3_mla_decode_token(
            &gpu, &kernels, &mut q, &mut k, &v, &g, &mut kv, &cfg, 1, 10000.0, 0,
        )
        .unwrap();
        let t1 = gpu.h2d_bytes() - after_t0;
        assert!(
            t1 > after_t0,
            "host-KV wrapper must re-upload history (t0={after_t0} t1={t1})"
        );
    }

    #[test]
    fn resident_kv_h2d_does_not_grow_with_history() {
        let gpu = MockGpuBackend::new();
        let kernels = K3MlaDecodeKernels::resolve(&gpu).unwrap();
        let cfg = MlaConfig::twin_0_40b();
        let q = vec![0.1f32; cfg.heads * cfg.qk_head_dim()];
        let k = vec![0.2f32; cfg.heads * cfg.qk_head_dim()];
        let v = vec![0.3f32; cfg.heads * cfg.v_head_dim];
        let g = vec![0.0f32; cfg.heads * cfg.v_head_dim];
        let mut kv = MlaDeviceKv::alloc(
            &gpu,
            8,
            cfg.heads * cfg.qk_head_dim(),
            cfg.heads * cfg.v_head_dim,
        )
        .unwrap();
        let _ = launch_k3_mla_decode_token_on_device(
            &gpu, &kernels, &q, &k, &v, &g, &mut kv, &cfg, 0, 10000.0, 0,
        )
        .unwrap();
        let t0 = gpu.h2d_bytes();
        let _ = launch_k3_mla_decode_token_on_device(
            &gpu, &kernels, &q, &k, &v, &g, &mut kv, &cfg, 1, 10000.0, 0,
        )
        .unwrap();
        let t1 = gpu.h2d_bytes() - t0;
        assert_eq!(
            t1, t0,
            "resident MLA token-append H2D must stay O(1) in T (t0={t0} t1={t1})"
        );
    }

    #[test]
    fn failed_second_buffer_allocation_frees_k() {
        let gpu = MockGpuBackend::new();
        // K fits the cap; V is larger so the second alloc fails.
        gpu.set_max_allocation_bytes(8 * 4);
        assert!(MlaDeviceKv::alloc(&gpu, 1, 8, 16).is_err());
        assert_eq!(gpu.alloc_count(), 0);
    }
}
