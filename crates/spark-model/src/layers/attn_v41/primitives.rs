// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! The per-stage launch wrappers every [`AttnV41`] path is built from: the
//! projection GEMM/GEMV (bf16 or `Q2_K`), RMSNorm, RoPE and the fp8/fp4
//! quantisers. Split from `attn_v41.rs` (500-LoC cap).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::{AttnMat, AttnV41, QUANT_BLOCKS_PER_LAUNCH};
use crate::layers::ops;
use crate::layers::ops::{Q2K_MMQ_SMEM, kquant_mmq_gemm, kquant_mmvq_w, kquant_q8_1_rows};
use crate::weight_map::DenseWeight;

impl AttnV41 {
    pub(super) fn gemm(
        &self,
        gpu: &dyn GpuBackend,
        a: DevicePtr,
        w: AttnMat,
        c: DevicePtr,
        m: usize,
        n: usize,
        kk: usize,
        stream: u64,
    ) -> Result<()> {
        let w = match w {
            AttnMat::Bf16(p) => p,
            AttnMat::Q3K(_) => anyhow::bail!("attention projections are never Q3_K"),
            AttnMat::Q2K(blocks) => {
                let (m, n, kk) = (m as u32, n as u32, kk as u32);
                if m <= 8 {
                    kquant_q8_1_rows(gpu, self.k.q8_rows, a, self.a_q8, m, kk, stream)?;
                    return kquant_mmvq_w(
                        gpu,
                        self.k.mmvq_q2k_w,
                        blocks,
                        self.a_q8,
                        c,
                        n,
                        kk,
                        m,
                        stream,
                    );
                }
                ops::quantize_act_q8_1(gpu, self.k.quant_d2s6, a, self.a_q8, m, kk, stream)?;
                return kquant_mmq_gemm(
                    gpu,
                    self.k.mmq_q2k_nc,
                    self.k.mmq_q2k_wc,
                    self.a_q8,
                    blocks,
                    c,
                    m,
                    n,
                    kk,
                    Q2K_MMQ_SMEM,
                    stream,
                );
            }
        };
        if m == 1 {
            return ops::dense_gemv(
                gpu,
                self.k.gemv,
                a,
                &DenseWeight { weight: w },
                c,
                n as u32,
                kk as u32,
                stream,
            );
        }
        ops::dense_gemm(
            gpu,
            self.k.gemm,
            a,
            &DenseWeight { weight: w },
            c,
            m as u32,
            n as u32,
            kk as u32,
            stream,
        )
    }

    pub(super) fn rmsnorm(
        &self,
        gpu: &dyn GpuBackend,
        f32_in: bool,
        x: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        rows: usize,
        dim: usize,
        stream: u64,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        KernelLaunch::new(
            gpu,
            if f32_in {
                self.k.rmsnorm_f32
            } else {
                self.k.rmsnorm_bf16
            },
        )
        .grid([rows as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(w)
        .arg_ptr(out)
        .arg_u32(dim as u32)
        .arg_f32(self.cfg.eps)
        .launch(stream)
    }

    pub(super) fn rope(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        pos: DevicePtr,
        rows: usize,
        row_len: usize,
        yarn: bool,
        inverse: bool,
        stream: u64,
    ) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        KernelLaunch::new(gpu, self.k.rope)
            .grid([rows as u32, 1, 1])
            .block([(self.cfg.rope_dim / 2).max(32) as u32, 1, 1])
            .arg_ptr(x)
            .arg_ptr(pos)
            .arg_ptr(if yarn { self.fc_yarn } else { self.fc_plain })
            .arg_u32(row_len as u32)
            .arg_u32(self.cfg.rope_dim as u32)
            .arg_u32(inverse as u32)
            .launch(stream)
    }

    pub(super) fn act_quant(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        n_values: usize,
        stream: u64,
    ) -> Result<()> {
        let n_blocks = n_values / 32;
        if n_blocks == 0 {
            return Ok(());
        }
        KernelLaunch::new(gpu, self.k.act_quant)
            .grid([(n_blocks as u32).div_ceil(QUANT_BLOCKS_PER_LAUNCH), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_u32(n_blocks as u32)
            .launch(stream)
    }

    pub(super) fn fp4_quant(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        n_values: usize,
        block: usize,
        e4m3_scale: bool,
        stream: u64,
    ) -> Result<()> {
        let n_blocks = n_values / block;
        if n_blocks == 0 {
            return Ok(());
        }
        KernelLaunch::new(gpu, self.k.fp4_quant)
            .grid([(n_blocks as u32).div_ceil(QUANT_BLOCKS_PER_LAUNCH), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_u32(n_blocks as u32)
            .arg_u32(block as u32)
            .arg_u32(e4m3_scale as u32)
            .launch(stream)
    }
}
