// SPDX-License-Identifier: AGPL-3.0-only

#![deny(warnings)]
#![deny(clippy::all)]
// Kernel-launch helpers and trait-impl wide signatures legitimately exceed
// clippy's 7-argument default. The same goes for the indexing-loop patterns
// that mirror the kernel grids we dispatch.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
// Some FP/integer special-case branches return the same value but have
// distinct semantic meanings (NaN vs zero, etc.). Audit shows these are
// intentional.
#![allow(clippy::if_same_then_else)]
// The HSS / disk-spill plumbing threads `Vec<u32>` through trait methods so
// callers can grow them in place; converting to slices breaks the contract.
#![allow(clippy::ptr_arg)]
// HF safetensors index tuples are wide on purpose.
#![allow(clippy::type_complexity)]

pub mod engine;
pub mod factory;
pub mod forward;
pub mod layer;
pub mod layers;
pub mod mistral_loader;
pub mod model;
pub mod precision_schedule;
pub mod preflight;
pub mod quant_format;
pub mod speculative;
pub mod tp_shard;
pub mod traits;
pub mod vision_preprocess;
pub mod weight_loader;
pub mod weight_map;
