// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! The shared expert's projections (`shared_expert_body`), split out of
//! `forward.rs` in phase 7 with the quantisation folds: an input whose q8_1
//! rows are already in the scratch given is not quantised again (the routed
//! experts and both shared projections read the token's rows once), and on
//! the GEMV arm the SwiGLU writes `sh` and its q8_1 rows in one launch
//! (`kquant_swiglu_q8_1_rows_bf16`). Same bytes as the separate launches.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{MoeV41, MoeV41LayerWeights};
use crate::layers::ops::{
    self, Q2K_MMQ_SMEM, Q3K_MMQ_SMEM, ResidentMat, kquant_mmq_gemm, kquant_mmvq_w,
    kquant_q8_1_rows, kquant_swiglu_q8_1_rows,
};
use crate::weight_map::DenseWeight;

fn kquant(wt: ResidentMat) -> bool {
    matches!(wt, ResidentMat::Q2K(_) | ResidentMat::Q3K(_))
}

impl MoeV41 {
    /// The shared expert up to its output `sd` (`[m, dim]` bf16), with the
    /// q8_1 scratch buffers given (the side stream has its own `h_q8`), no
    /// `acc`. `x_pre`: `a_q8` already holds the q8_1 rows of `x` (`m <= 8`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn shared_expert_body(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        x: DevicePtr,
        m: usize,
        a_q8: DevicePtr,
        h_q8: DevicePtr,
        x_pre: bool,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        // shared expert: bf16 (GEMV / tiled GEMM) or the GGUF's K-quant
        // blocks on the routed experts' kernels (GEMV at m <= 8, MMQ above).
        // `ready`: the q8_1 rows of `a` are already in `a_q8` (GEMV arms).
        let kq = |a: DevicePtr,
                  a_q8: DevicePtr,
                  ready: bool,
                  wt: ResidentMat,
                  out: DevicePtr,
                  n: usize,
                  kdim: usize|
         -> Result<()> {
            let (mu, nu, ku) = (m as u32, n as u32, kdim as u32);
            match wt {
                ResidentMat::Bf16(p) if m == 1 => ops::dense_gemv(
                    gpu,
                    self.k.gemv,
                    a,
                    &DenseWeight { weight: p },
                    out,
                    nu,
                    ku,
                    stream,
                ),
                ResidentMat::Bf16(p) => ops::dense_gemm(
                    gpu,
                    self.k.gemm,
                    a,
                    &DenseWeight { weight: p },
                    out,
                    mu,
                    nu,
                    ku,
                    stream,
                ),
                ResidentMat::Q2K(b) if m <= 8 => {
                    if !ready {
                        kquant_q8_1_rows(gpu, self.k.q8_rows, a, a_q8, mu, ku, stream)?;
                    }
                    kquant_mmvq_w(gpu, self.k.mmvq_q2k, b, a_q8, out, nu, ku, mu, stream)
                }
                ResidentMat::Q3K(b) if m <= 8 => {
                    if !ready {
                        kquant_q8_1_rows(gpu, self.k.q8_rows, a, a_q8, mu, ku, stream)?;
                    }
                    kquant_mmvq_w(gpu, self.k.mmvq_q3k, b, a_q8, out, nu, ku, mu, stream)
                }
                ResidentMat::Q2K(b) => {
                    ops::quantize_act_q8_1(gpu, self.k.quant_d2s6, a, a_q8, mu, ku, stream)?;
                    kquant_mmq_gemm(
                        gpu,
                        self.k.mmq_q2k_nc,
                        self.k.mmq_q2k_wc,
                        a_q8,
                        b,
                        out,
                        mu,
                        nu,
                        ku,
                        Q2K_MMQ_SMEM,
                        stream,
                    )
                }
                ResidentMat::Q3K(b) => {
                    ops::quantize_act_q8_1(gpu, self.k.quant_d4, a, a_q8, mu, ku, stream)?;
                    kquant_mmq_gemm(
                        gpu,
                        self.k.mmq_q3k_nc,
                        self.k.mmq_q3k_wc,
                        a_q8,
                        b,
                        out,
                        mu,
                        nu,
                        ku,
                        Q3K_MMQ_SMEM,
                        stream,
                    )
                }
            }
        };
        kq(x, a_q8, x_pre, w.shared_w1, self.sg, c.inter, c.dim)?;
        // the w1 GEMV arm left the rows of `x` in `a_q8`
        let x_ready = x_pre || (m <= 8 && kquant(w.shared_w1));
        kq(x, a_q8, x_ready, w.shared_w3, self.su, c.inter, c.dim)?;
        let fuse = m <= 8 && kquant(w.shared_w2);
        if fuse {
            kquant_swiglu_q8_1_rows(
                gpu,
                self.k.swiglu_q8,
                self.sg,
                self.su,
                DevicePtr(0),
                self.sh,
                h_q8,
                m as u32,
                c.inter as u32,
                c.swiglu_limit,
                stream,
            )?;
        } else {
            self.launch_n(gpu, self.k.swiglu, m * c.inter, stream, |l| {
                l.arg_ptr(self.sg)
                    .arg_ptr(self.su)
                    .arg_ptr(DevicePtr(0))
                    .arg_ptr(self.sh)
                    .arg_u32(m as u32)
                    .arg_u32(c.inter as u32)
                    .arg_f32(c.swiglu_limit)
            })?;
        }
        kq(self.sh, h_q8, fuse, w.shared_w2, self.sd, c.dim, c.inter)
    }
}
