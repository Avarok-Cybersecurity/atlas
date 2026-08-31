// SPDX-License-Identifier: AGPL-3.0-only

//! Per-model-family JSON parsers, split out of `config.rs` for file-size
//! budget.

mod deepseek_v4;
mod gemma4;
mod laguna;
mod longcat;
mod lora;
mod minimax;
mod mistral;
mod quantization;
mod qwen4_exp;
mod step3p7;
mod vision;

pub(crate) use deepseek_v4::parse_deepseek_v4;
pub(crate) use gemma4::parse_gemma4_params;
pub(crate) use laguna::parse_laguna;
pub(crate) use longcat::parse_longcat_ngram;
pub use lora::{
    PEFT_SUPPORTED_TARGET_MODULES, PeftAdapterConfig, allow_partial_targets,
    parse_peft_adapter_config,
};
pub(crate) use minimax::parse_minimax_m2;
pub use mistral::parse_mistral_params;
pub use quantization::parse_quantization_config;
pub(crate) use qwen4_exp::parse_qwen4_exp;
pub(crate) use step3p7::parse_step3p7;
pub(crate) use vision::parse_vision_config;
