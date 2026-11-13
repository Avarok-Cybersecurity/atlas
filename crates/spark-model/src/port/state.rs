// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the per-model state the port sprint's (`port/consumer-opt`)
//! lanes add, held as ONE field (`port`) on `TransformerModel`.
//!
//! One field per lane; each field's type is defined in the lane's own module
//! and implements `Default`, so the model's constructor never changes again
//! (`port: Default::default()` in `model/impl_a1.rs`). Lines are inserted
//! immediately BEFORE the marker below by the coordinator's fragment apply
//! pass.

/// The port lanes' per-model state. Empty until a lane adds a field; a row-off
/// lane's field is its type's `Default` and is never read.
#[derive(Default)]
pub struct ModelPortState {
    /// Lane L25: staging of the batched sequence swap (`kv_batched_swap`).
    pub kv_swap: spark_runtime::kv_cache::block_io::swap::SwapStaging,
    // @port:fields
}
