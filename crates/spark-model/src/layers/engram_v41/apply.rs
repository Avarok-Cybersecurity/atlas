// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b
//! [`EngramV41::apply`]: the `wkv` projection of the rows in the workspace
//! (single token: off the raw Q2_K blocks when the layer carries them, else
//! the bf16 GEMV; several tokens: the bf16 GEMM) and the gate onto the
//! highway. Split from `engram_v41.rs` (500-LoC cap).

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::EngramV41;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

impl EngramV41 {
    /// Project the rows already in the workspace and gate `tokens` positions
    /// of `streams` (`[T, hc, dim]` f32) in place, for model layer `layer`.
    pub fn apply(
        &self,
        gpu: &dyn GpuBackend,
        layer: usize,
        streams: DevicePtr,
        tokens: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            tokens <= self.max_tokens,
            "engram: {tokens} tokens exceeds the {} workspace",
            self.max_tokens
        );
        let w = self
            .layer(layer)
            .with_context(|| format!("engram: layer {layer} has no weights"))?;
        let wkv = DenseWeight { weight: w.wkv };
        if tokens == 1 && !w.wkv_q2k.is_null() {
            let (n, k) = (self.out_features() as u32, self.in_features() as u32);
            ensure!(
                k.is_multiple_of(256),
                "engram: in_features {k} is not Q2_K blocks"
            );
            KernelLaunch::new(gpu, self.gemv_q2k_k)
                .grid([n.div_ceil(4), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.rows_of(w))
                .arg_ptr(w.wkv_q2k)
                .arg_ptr(self.kv)
                .arg_u32(n)
                .arg_u32(k)
                .launch(stream)?;
        } else if tokens == 1 {
            ops::dense_gemv(
                gpu,
                self.gemv_k,
                self.rows_of(w),
                &wkv,
                self.kv,
                self.out_features() as u32,
                self.in_features() as u32,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                gpu,
                self.gemm_k,
                self.rows_of(w),
                &wkv,
                self.kv,
                tokens as u32,
                self.out_features() as u32,
                self.in_features() as u32,
                stream,
            )?;
        }
        KernelLaunch::new(gpu, self.gate_k)
            .grid([tokens as u32, self.hc as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(streams)
            .arg_ptr(self.kv)
            .arg_ptr(w.qk)
            .arg_u32(self.dim as u32)
            .arg_u32(self.hc as u32)
            .arg_f32(self.eps)
            .launch(stream)
    }
}
