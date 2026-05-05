// SPDX-License-Identifier: AGPL-3.0-only

//! Per-model-family JSON parsers, split out of `config.rs` for file-size
//! budget.

mod gemma4;
mod minimax;
mod mistral;
mod quantization;
mod vision;

pub(crate) use gemma4::parse_gemma4_params;
pub(crate) use minimax::parse_minimax_m2;
pub use mistral::parse_mistral_params;
pub use quantization::parse_quantization_config;
pub(crate) use vision::parse_vision_config;
