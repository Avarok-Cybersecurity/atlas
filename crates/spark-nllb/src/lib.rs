// SPDX-License-Identifier: AGPL-3.0-only

//! Self-contained CPU reference runtime for NLLB-200 / M2M-100 encoder-decoder
//! (seq2seq) translation.
//!
//! Atlas's production engine is decoder-only and GPU-only, so it cannot run an
//! encoder-decoder model — the marker loader in `spark-model` fails fast for
//! `model_type = "m2m_100" | "nllb"`. This crate provides a dependency-light,
//! CPU-only path that actually *loads and runs* the model from the safetensors
//! conversion (`MonumentalSystems/nllb-200-3.3B`), so the checkpoint can be
//! validated end-to-end without the CUDA/Metal stack.

pub mod config;
pub mod lora;
pub mod model;
pub mod ops;
pub mod weights;

pub use config::NllbConfig;
pub use lora::LoraSet;
pub use model::NllbModel;
