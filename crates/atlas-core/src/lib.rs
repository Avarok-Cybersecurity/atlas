// SPDX-License-Identifier: AGPL-3.0-only

#![deny(warnings)]
#![deny(clippy::all)]

pub mod arch;
pub mod capabilities;
pub mod compute;
pub mod config;
pub mod dtype;
pub mod error;
pub mod fault;
pub mod numeric;
pub mod safetensors;
pub mod scope;
pub mod target;
pub mod tensor;

// `device` always compiles — its `sm121` submodule is pure constants
// that spark-model's launch heuristics consume on every backend.
// The `AtlasDevice` cudarc wrapper inside `device` is itself gated
// behind the `cuda` feature.
pub mod device;

// CUDA-only modules: rely on `cudarc` and the NVIDIA driver. Gated so the
// crate compiles on hosts without a CUDA toolchain (e.g. Apple Silicon).
#[cfg(feature = "cuda")]
pub mod cuda_host;
#[cfg(feature = "cuda")]
pub mod kernel;
#[cfg(feature = "cuda")]
pub mod registry;
#[cfg(feature = "cuda")]
pub mod stream;
