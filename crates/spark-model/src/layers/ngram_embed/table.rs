// SPDX-License-Identifier: AGPL-3.0-only

//! One n-gram lookup table on device: BF16 as shipped, FP8-quantized at
//! load, or NVMe-backed by a bounded row cache.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use crate::weight_map::{DenseWeight, Fp8DenseWeight};

/// One n-gram lookup table on device: BF16 as shipped, or FP8-quantized at
/// load (per-row E4M3 + f32 scale — halves the ~63 GB table footprint;
/// embeddings tolerate this well and the gather dequantizes on read).
pub enum NgramTable {
    Bf16(DenseWeight),
    Fp8(Fp8DenseWeight),
    /// NVMe-backed: only a bounded set of ROWS is resident, in a pinned
    /// GPU-addressable arena. The host resolves `row_id -> slot` (the ids are
    /// a pure function of token ids, so this is host-side anyway) and the
    /// SAME gather kernels then read the arena by slot index — no kernel
    /// change, no `cuMemcpyHtoD` on the fault path.
    ///
    /// This is what makes a 51 B-parameter embedding table serveable on a
    /// 121 GB box: the tables are the model's largest tensors and its least
    /// bandwidth-hungry (12 rows ~ 3 KB per token), so demoting them buys
    /// back tens of GB for KV.
    #[cfg(feature = "cuda")]
    Cached(Box<spark_storage::NgramRowCache>),
}

impl NgramTable {
    /// Quantize a BF16 table to FP8 on the GPU (per-row E4M3 absmax +
    /// f32 scale via `quantize_bf16_to_fp8`) — the quantize-on-load
    /// lever. The caller frees the BF16 source afterwards; tables are
    /// loaded one at a time so peak overhead is a single table.
    pub fn quantize_bf16(
        w: &DenseWeight,
        rows: usize,
        dim: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Self> {
        let k = gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;
        Ok(Self::Fp8(crate::weight_map::quantize_to_fp8(
            w, rows, dim, gpu, k, stream,
        )?))
    }
}
