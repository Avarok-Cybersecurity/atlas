// SPDX-License-Identifier: AGPL-3.0-only

//! `ModelDims` — model-shape descriptor threaded through every layer's
//! forward signature. Lives in spark-storage because the high-speed-
//! swap orchestrator was the original consumer, but the type itself
//! has no GPU state and must remain compilable on macOS / no-cuda
//! builds where the swap orchestrator isn't reachable.

/// Per-call dimensions describing the model the orchestrator serves.
#[derive(Clone, Copy, Debug)]
pub struct ModelDims {
    pub num_layers: u32,
    pub max_blocks_per_layer: u32,
    pub num_q_heads: u16,
    pub num_kv_heads: u16,
    pub head_dim: u16,
    pub block_size: u16,
    /// Config-derived model fingerprint (spark-model's `ModelFingerprint`,
    /// KV convention `derive_kv`: blob_bytes = 0) — the per-model identity
    /// the KV paging namespace folds (`kv_paging::ns::derive_kv_ns`) so two
    /// models sharing one paging peer can never collide. `None` when the
    /// loader could not derive one (or in geometry-only tests/benches);
    /// `ATLAS_KV_PAGING=1` then fails fast at connect unless
    /// `ATLAS_KV_PAGING_NS` is set explicitly. Unread on every other path.
    pub model_fp: Option<std::num::NonZeroU64>,
}
