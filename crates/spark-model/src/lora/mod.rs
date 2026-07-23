// SPDX-License-Identifier: AGPL-3.0-only

//! Startup-static PEFT LoRA adapter: remap/validate/pack into the
//! fixed-address rank-padded pool. v0 = one adapter, slot 0, always on.
//!
//! NAMING: everything here is `Peft*`/`adapter_*`/`Lora*` (adapter sense) —
//! `kv_lora_rank`/`q_lora_rank` (atlas-core/src/config.rs:182-207) are MLA
//! vocabulary, not this.
//!
//! NOTE on leaks: the intermediate `WeightStore` device copies of the
//! unpadded A/B tensors become garbage after pool packing and are never
//! freed (no dealloc on weight structs anywhere in Atlas). Accepted at
//! holo adapter scale (~tens of MiB).
//!
//! SDD facade: the surface is split by functional seam into `types` (the
//! module/AB enums + weight/slot structs + `LoraWeights` impl), `slot_math`
//! (pure slot/offset placement + routing), `key` (classify + adapter identity),
//! `env` (the `$ATLAS_LORA_*` hatches + `validate_peft_config`), and `loading`
//! (audit/pack + the load entry points). Every public name re-exports at its
//! own visibility so `crate::lora::X` / `spark_model::lora::X` paths are stable.

mod env;
mod key;
mod loading;
mod slot_math;
mod types;

pub use env::*;
pub use key::*;
pub use loading::*;
pub use slot_math::*;
pub use types::*;

// RDMA LoRA staging pulls `spark_storage::{LoraAbKind, LoraLandTarget}`, which
// spark-storage only exports under `cuda`; its sole caller
// (`swap_lora_slot_from_peer`) is already `cfg(feature = "cuda")`. Gate the
// module so the non-cuda (metal) build doesn't try to resolve those imports.
#[cfg(feature = "cuda")]
// RDMA LoRA staging lands adapter tensors via spark-storage's RDMA weight
// loader; RDMA needs rdma-core, so this stays unix-only even though the NVMe
// tier itself is now portable.
#[cfg(unix)]
pub mod rdma_stage;

#[cfg(test)]
#[path = "test_support.rs"]
mod test_support;
