// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the two helpers behind `w4_proj.rs`'s `w4_project`: the
//! BF16 GEMM of a declined prefill chunk over the dequantized weight, and the
//! once-per-model log line naming the route.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::GdnW4Proj;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::layers::qwen3_ssm::Qwen3SsmLayer;
use crate::weight_map::DenseWeight;

impl Qwen3SsmLayer {
    /// The BF16 GEMM of a declined prefill chunk over the dequantized
    /// `[n, k]` weight: the layer's tensor-core `dense_gemm_bf16_pipelined`
    /// (the BF16 `out_proj` prefill arm's kernel) when resolved, else
    /// cuBLASLt (the BF16 QKVZ prefill arm's choice), else the scalar
    /// `dense_gemm` (that arm's own fallback).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn w4_bf16_gemm(
        &self,
        ctx: &ForwardContext,
        x: DevicePtr,
        w_bf16: DevicePtr,
        out: DevicePtr,
        m: usize,
        n: usize,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        let (m, n, k) = (m as u32, n as u32, k as u32);
        let dw = DenseWeight { weight: w_bf16 };
        if self.dense_gemm_pipelined_k.0 != 0 {
            return ops::dense_gemm_bf16_pipelined(
                ctx.gpu,
                self.dense_gemm_pipelined_k,
                x,
                &dw,
                out,
                m,
                n,
                k,
                stream,
            );
        }
        if ops::cublas_bf16_proj_dense(x, w_bf16, out, m, n, k, stream).is_ok() {
            return Ok(());
        }
        ensure!(
            self.dense_gemm_k.0 != 0,
            "GDN W4: a declined prefill needs a BF16 GEMM, and this target resolved neither \
             `dense_gemm_bf16_pipelined` nor cuBLASLt nor `dense_gemm`"
        );
        ops::dense_gemm(ctx.gpu, self.dense_gemm_k, x, &dw, out, m, n, k, stream)
    }

    /// Name the route once per model: formats, shapes, the decode window and
    /// how wider calls are served. The GPU pass greps this line.
    pub(super) fn w4_log_route(&self, ctx: &ForwardContext, p: &GdnW4Proj) {
        if ctx.stats.once("log:gdn_w4_proj_route") {
            let (q, o) = (&*p.qkvz, &*p.out);
            tracing::info!(
                "[avarok] GDN W4 projections: in_proj_qkvz {} [{}x{}], out_proj {} [{}x{}] \
                 ({} MiB/layer); decode rows <= {} per launch, wider decode-side calls as decode \
                 chunks (M=1 bits per row), prompts through W4Linear::prefill with the BF16 \
                 dequant fallback",
                q.format(),
                q.n(),
                q.k(),
                o.format(),
                o.n(),
                o.k(),
                (q.device_bytes() + o.device_bytes()) >> 20,
                q.max_decode_rows().min(o.max_decode_rows()),
            );
        }
    }
}
