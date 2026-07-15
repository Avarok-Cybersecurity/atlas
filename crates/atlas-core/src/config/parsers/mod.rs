// SPDX-License-Identifier: AGPL-3.0-only

//! Per-model-family JSON parsers, split out of `config.rs` for file-size
//! budget.

mod deepseek_v4;
mod gemma4;
mod lora;
mod minimax;
mod mistral;
mod quantization;
mod step3p7;
mod vision;

pub(crate) use deepseek_v4::parse_deepseek_v4;
pub(crate) use gemma4::parse_gemma4_params;
pub use lora::{PEFT_SUPPORTED_TARGET_MODULES, PeftAdapterConfig, parse_peft_adapter_config};
pub(crate) use minimax::parse_minimax_m2;
pub use mistral::parse_mistral_params;
pub use quantization::parse_quantization_config;
pub(crate) use step3p7::parse_step3p7;
pub(crate) use vision::parse_vision_config;
