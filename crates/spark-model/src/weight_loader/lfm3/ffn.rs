// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from HuggingFace transformers (models/lfm3, Apache-2.0, Copyright 2026 The HuggingFace Team) and vLLM (vllm/model_executor/models/lfm3.py, Apache-2.0, Copyright contributors to the vLLM project). Changes: ported from Python/Triton to Rust/CUDA for Atlas.

//! Responsibility: the dense SwiGLU FFN of one LFM3 layer (`w2(silu(w1 x) *
//! w3 x)`: `w1` is the gate, `w3` the up, `w2` the down projection), shared by
//! the conv layers and the attention layers.
//!
//! BF16 weights install through `set_bf16_weights` (cuBLASLt prefill on Ada);
//! an all-E4M3 group (the vendor checkpoint, or the load-time route) installs
//! through `set_fp8_weights`. The NVFP4 fallback slots are NULL either way:
//! the FFN never touches them once an overlay is set.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::fp8::{Class, Fp8Loader, Group};
use super::names::Geometry;
use crate::layers::dense_ffn::DenseFfnWeights;
use crate::layers::{DenseFfnLayer, FfnComponent};
use crate::weight_map::QuantizedWeight;

fn null_nvfp4_weights() -> DenseFfnWeights {
    DenseFfnWeights {
        gate_proj: QuantizedWeight::null(),
        up_proj: QuantizedWeight::null(),
        down_proj: QuantizedWeight::null(),
        gate_proj_t: None,
        up_proj_t: None,
        down_proj_t: None,
    }
}

/// Build the FFN of layer prefix `lp`.
pub(super) fn build(
    fp8: &Fp8Loader<'_>,
    gpu: &dyn GpuBackend,
    g: &Geometry,
    lp: &str,
) -> Result<FfnComponent> {
    let ff = format!("{lp}.feed_forward");
    let (w1, w3, w2) = (format!("{ff}.w1"), format!("{ff}.w3"), format!("{ff}.w2"));
    let group = fp8.load_group(
        Class::Ffn,
        [
            (w1.as_str(), g.inter, g.hidden),
            (w3.as_str(), g.inter, g.hidden),
            (w2.as_str(), g.hidden, g.inter),
        ],
    )?;
    let mut layer = DenseFfnLayer::new(null_nvfp4_weights(), gpu)?;
    match group {
        Group::Bf16([gate, up, down]) => layer.set_bf16_weights(gate, up, down),
        Group::Fp8([gate, up, down]) => layer.set_fp8_weights(gate, up, down),
    }
    Ok(FfnComponent::Dense(layer))
}
