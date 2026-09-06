// SPDX-License-Identifier: AGPL-3.0-only

//! Weight name mapping from HuggingFace safetensors to typed layer structures.
//!
//! Maps the 72 unique weight patterns from Qwen3-Next-80B-A3B-Instruct-NVFP4
//! into structured per-layer weight references.
//!
//! Refactor wave 4a (2026-05-03): split into `weight_map/` sub-modules.

#[path = "weight_map/exl3_materialize.rs"]
mod exl3_materialize;
#[path = "weight_map/exl3_materialize_dense.rs"]
mod exl3_materialize_dense;
#[path = "weight_map/exl3_materialize_moe.rs"]
mod exl3_materialize_moe;
#[path = "weight_map/expert.rs"]
mod expert;
#[path = "weight_map/fp8_dequant.rs"]
mod fp8_dequant;
#[path = "weight_map/fp8_lut.rs"]
mod fp8_lut;
#[path = "weight_map/loaders_fp8.rs"]
mod loaders_fp8;
#[path = "weight_map/loaders_moe.rs"]
mod loaders_moe;
#[path = "weight_map/loaders_mtp.rs"]
mod loaders_mtp;
#[path = "weight_map/model_a.rs"]
mod model_a;
#[path = "weight_map/model_b.rs"]
mod model_b;
#[path = "weight_map/moe.rs"]
mod moe;
#[path = "weight_map/moe_exl3.rs"]
mod moe_exl3;
#[path = "weight_map/nemotron.rs"]
mod nemotron;
#[path = "weight_map/nvfp4_detect.rs"]
mod nvfp4_detect;
#[path = "weight_map/quant_helpers.rs"]
mod quant_helpers;
#[path = "weight_map/quantize_fns.rs"]
mod quantize_fns;
#[path = "weight_map/quantize_fp8_bs.rs"]
mod quantize_fp8_bs;
#[path = "weight_map/quantized.rs"]
mod quantized;
#[path = "weight_map/ssm_qwen35.rs"]
mod ssm_qwen35;
#[path = "weight_map/ssm_qwen35_more.rs"]
mod ssm_qwen35_more;

pub use exl3_materialize::*;
pub use exl3_materialize_dense::*;
pub use exl3_materialize_moe::*;
pub use expert::*;
pub(crate) use fp8_dequant::*;
pub use loaders_fp8::*;
pub use loaders_mtp::*;
pub use model_a::*;
pub use moe::*;
pub use moe_exl3::*;
pub use nemotron::*;
pub use nvfp4_detect::*;
pub use quantize_fns::*;
pub use quantize_fp8_bs::*;
pub use quantized::*;
pub use ssm_qwen35::*;

// Modules whose only exports are `pub(crate)` / `pub(super)` helpers.
#[allow(unused_imports)]
pub(crate) use {fp8_lut::*, loaders_moe::*, model_b::*, quant_helpers::*, ssm_qwen35_more::*};
