// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the port sprint's (`port/consumer-opt`) SERVING rows of
//! `kernels/<hw>/HARDWARE.toml` `[defaults]`, as the runtime type baked into
//! [`crate::TargetDefaults::port`].
//!
//! A sub-table of [`crate::TargetDefaults`] rather than more fields on it, so
//! the sprint's rows land in files of their own: `build_defaults_port.rs`
//! parses them, `build.rs` bakes them into the same generated
//! `TARGET_DEFAULTS` constant (one generated file, one `include!`, one content
//! hash), and `spark_model::layers::ops::target_defaults_port` resolves them
//! against the environment.
//!
//! Every row is OFF (or its baseline string or number) on every target when it
//! lands. A row-off path is byte-identical to the behavior before the row
//! existed, so GB10, Hopper, B200, B300, Metal and Strix serve exactly as they
//! did.
//!
//! Rows arrive through the coordinator's fragment apply pass: each `@port:`
//! marker below is an insertion point, and text is inserted immediately BEFORE
//! the marker line.

/// The port sprint's rows of one compiled target's `[defaults]` table.
///
/// `Copy` and entirely `'static`, like [`crate::TargetDefaults`], of which it
/// is the `port` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortDefaults {
    /// When a 4-bit BF16 load route is on (dense_affine4_from_bf16,
    /// gdn_affine4_from_bf16, attn_affine4_from_bf16), every projection the
    /// Marlin kernels serve (N a multiple of 64, K of 128, more than one group)
    /// is quantized once at load by round-to-nearest int4 at group 128 (GPTQ's
    /// symmetric uint4b8 grid, --w4-clip-search shrink candidates per group)
    /// instead of the MLX affine codes, and served through W4Linear on vLLM's
    /// Marlin W4A16 GEMM (kernels/ada/common/marlin_ada.cu, host
    /// crates/spark-model/src/port/marlin/): decode and verify rows 1..64 in one
    /// launch on one thread configuration fixed per weight (row invariant),
    /// prompts on vLLM's M split.
    pub marlin_w4a16: bool,
    /// Batched sequence swap
    /// (`spark-model/src/model/trait_impl/sequence/state_io.rs`,
    /// `spark-runtime/src/kv_cache/block_io.rs`): a swapped-out sequence's image
    /// (each block's K and V for every attention layer, then each GDN layer's h
    /// and conv state) is gathered with batched async copies
    /// (`cuMemcpyBatchAsync` where it resolved, else one async copy per region
    /// and one event wait per chunk) into one reused page-locked staging buffer
    /// of at most one `--pinned-pool-chunk-mb` chunk, and the file write of one
    /// chunk overlaps the copy of the next; restore mirrors it.
    pub kv_batched_swap: bool,
    /// SSM tier blob store eviction order
    /// (`spark-model/src/model/ssm_tier/store.rs`): a capped host-RAM
    /// `MemBlobStore` gives up the least recently used blob (a fault-in hit and a
    /// put both make a key the most recent) instead of the oldest insert, and the
    /// model removes the tier blobs the snapshot index orphans when a fresh save
    /// re-homes a spilled prefix or the tail sweep drops one
    /// (`spark-runtime/src/radix_tree/snapshot_insert.rs`).
    pub ssm_store_lru: bool,
    /// Radix prefix-cache eviction takes its victim from an ordered (last_access,
    /// node id) index of evictable leaves, maintained at every transition of the
    /// victim predicate, instead of scanning every node once per evicted block
    /// (`spark-runtime/src/radix_tree/inner/lru_index.rs`).
    pub radix_lru_index: bool,
    /// A prefix-cache hit may end inside a KV block: the chunked prefill copies
    /// that block into a private block (every attention layer's K and V, D2D on
    /// the prefill stream before the block table is used) instead of sharing the
    /// donor's writable tail, which is the issue #1193 aliasing bug
    /// (`radix_tree/inner/cow_walk.rs`, `prefill_b/prefix_cow.rs`).
    pub prefix_cow_tail: bool,
    // @port:fields
}

/// What a target that declares no port row gets: every row's baseline.
///
/// Kept in agreement with `build_defaults_port.rs`'s `Default for PortFields`
/// by the row templates (one baseline per row, written into both files in the
/// same apply pass).
pub const PORT_BASELINE: PortDefaults = PortDefaults {
    marlin_w4a16: false,
    kv_batched_swap: false,
    ssm_store_lru: false,
    radix_lru_index: false,
    prefix_cow_tail: false,
    // @port:baseline
};
