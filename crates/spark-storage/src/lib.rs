// SPDX-License-Identifier: AGPL-3.0-only

#![deny(warnings)]
#![deny(clippy::all)]

// Atlas spark-storage: high-speed NVMe-backed KV cache offload.
//
// Phase 0 of `--high-speed-swap` (see plan at
// /workspace/.claude/plans/i-want-to-ensure-valiant-bunny.md): runtime probe
// that decides whether the production backend should be cuFile/GDS or
// io_uring + pinned-host bounce. Later phases add the predictor, scratch
// pool, eviction, and I/O thread.
//
// Feature gating: every module that touches the CUDA driver (raw FFI in
// `cuda_min`, the module/event helpers in `cuda_module`, anything that
// holds a `DeviceBuffer`) is gated behind the `cuda` feature so the
// crate compiles on Apple Silicon (`--no-default-features --features
// metal`) where the high-speed-swap path won't be reachable anyway.

#[cfg(feature = "cuda")]
pub mod cuda_graph;
#[cfg(feature = "cuda")]
pub mod cuda_min;
#[cfg(feature = "cuda")]
pub mod cuda_module;

// Re-export the module/event/launch helpers from their new home so existing
// `use spark_storage::cuda_min::{CudaModule, CudaEvent, launch_kernel}` paths
// keep working.
#[cfg(feature = "cuda")]
pub use cuda_module::{CudaEvent, CudaModule, launch_kernel};

// Pure CPU-side modules — types, configs, references. Always compiled.
pub mod attention_ref;
pub mod cascade_policy;
pub mod config;
pub mod eviction;
pub mod expert;
pub mod expert_pack;
pub mod expert_peer;
pub mod group;
pub mod kv_paging;
pub mod model_dims;
pub mod predictor_ref;
pub mod projection;
// The one-sided RDMA KV transport backend — a `StorageBackend` impl that
// offloads/restores KV groups to a `cache_peer` blade over verbs. cuda (for the
// pinned-host bounce + copy_h2d) + the verbs shim.
#[cfg(all(feature = "cuda", atlas_rdma_verbs))]
pub mod rdma_kv_backend;
pub mod rdma_snapshot;
pub mod snapshot_swap;
// RDMA weight-staging peer (RO): serves a model's safetensors shards to the
// weight loader over one-sided READ for fast model swaps. `manifest`/`wire` are
// un-gated; `serve`/`shard` are `cfg(unix)` internally.
pub mod weight_peer;

// KV overflow blade (cache-peer): a passive remote-RAM RW tier for the
// high-speed-swap KV cache, served over one-sided RDMA. `cache_peer` is the
// server; `blade_cap` is the process-global commit ledger it (and the
// weight/expert peers) reserve against — CUDA-free arithmetic, so it's
// `#[allow(dead_code)]` on verbs-OFF builds where the handshake that consumes
// it is compiled out.
#[cfg(unix)]
#[allow(dead_code)]
pub(crate) mod blade_cap;
pub mod cache_peer;

// `ModelDims` is a plain POD struct (no GPU state) that
// `spark-model`'s public surface threads through every layer's
// forward signature; it must stay reachable on metal builds even
// though the high-speed-swap orchestrator that consumes it is
// CUDA-gated.
pub use model_dims::ModelDims;

// `layout` opens disk files with `O_DIRECT` and pre-allocates via
// `posix_fallocate` — both Linux-specific. Only the cuda-side modules
// (high_speed_swap, backend/io_uring, backend/posix) consume it, so
// gating it on the cuda feature is sufficient.
#[cfg(feature = "cuda")]
pub mod layout;

// CUDA-only modules: each holds raw `cu*` FFI calls or a `DeviceBuffer`,
// or transitively imports from the cuda_* modules above. Gated together
// because separating them would just smear the boundary.
#[cfg(feature = "cuda")]
pub mod backend;
#[cfg(feature = "cuda")]
pub mod bench;
// T1 write-back cache composite (wraps any StorageBackend). cuda but not verbs.
#[cfg(feature = "cuda")]
pub mod cascade_backend;
#[cfg(feature = "cuda")]
pub mod expert_arena;
#[cfg(feature = "cuda")]
pub mod expert_tier;
#[cfg(feature = "cuda")]
pub mod expert_tier_rdma;
#[cfg(feature = "cuda")]
pub mod high_speed_swap;
#[cfg(feature = "cuda")]
pub mod predictor;
#[cfg(feature = "cuda")]
pub mod probe;
#[cfg(feature = "cuda")]
pub mod scratch_pool;
#[cfg(feature = "cuda")]
pub mod tiled_attention;
// RDMA weight loader — the cuda client of `weight_peer` that one-sided-READs a
// model's tensors into a `spark_runtime::weights::WeightStore` for fast swaps.
// RDMA-stage a PEFT adapter's A/B tensors straight into a resident LoRA pool
// slot (reuses the weight_peer manifest + wire; landing byte-identical to the
// disk pack).
#[cfg(feature = "cuda")]
pub mod weight_lora_rdma;
#[cfg(feature = "cuda")]
pub mod weight_tier_rdma;

#[cfg(feature = "cuda")]
pub use backend::{IoUringBackend, PosixBackend, ReadRequest, StorageBackend};
pub use config::HighSpeedSwapConfig;
pub use eviction::EvictionPolicy;
pub use expert::{
    ExpertKey, ExpertLayout, ExpertRecordHeader, ExpertRecordId, ExpertRecordSpec, Proj, ProjBytes,
};
#[cfg(feature = "cuda")]
pub use expert_arena::ExpertArena;
#[cfg(unix)]
pub use expert_pack::{ExpertFileReader, ExpertFileWriter};
pub use expert_pack::{ExpertIndex, ProjData, ProjView, pack_record, unpack_record};
#[cfg(feature = "cuda")]
pub use expert_tier::{
    ArenaSlot, ExpertResidency, ExpertTier, PosixTier, TierKind, UmaArenaTier, open_tier,
};
#[cfg(feature = "cuda")]
pub use expert_tier_rdma::RdmaTier;
#[cfg(feature = "cuda")]
pub use high_speed_swap::{HighSpeedSwap, install_local, local_installed, with_local};
#[cfg(all(feature = "cuda", atlas_rdma_verbs))]
pub use kv_paging::KvPagingBackend;
#[cfg(all(feature = "cuda", atlas_rdma_verbs))]
pub use rdma_kv_backend::RdmaKvBackend;
pub use rdma_snapshot::RdmaSnapshotArena;

/// `true` iff `atlas_rdma_verbs` was re-emitted for this crate by build.rs (the
/// one-sided verbs shim lives in the CUDA-free `atlas-rdma` crate; `rustc-cfg`
/// doesn't cross crates, so build.rs re-emits it off atlas-rdma's `links`
/// metadata). `rdma_verbs_probe_tests` asserts it, so a silent cfg evaporation
/// fails `cargo test -p spark-storage --lib` on verbs hosts instead of
/// green-building with the gated modules compiled out.
pub const fn rdma_verbs_enabled() -> bool {
    cfg!(atlas_rdma_verbs)
}

#[cfg(test)]
mod rdma_verbs_probe_tests;

// Non-cuda stub surface — same names as the real CUDA orchestrator
// above so spark-model's call sites compile unchanged. `with_local`
// always returns None (orchestrator absent), `local_installed` is
// false, and `install_local` bails — see `stubs.rs` for rationale.
#[cfg(not(feature = "cuda"))]
mod stubs;
#[cfg(not(feature = "cuda"))]
pub use stubs::{HighSpeedSwap, install_local, local_installed, with_local};

#[cfg(feature = "cuda")]
pub use predictor::{Predictor, PredictorDims};
#[cfg(feature = "cuda")]
pub use probe::{Backend, ProbeConfig, ProbeResult, run_probe};
pub use projection::{PredictorShape, build_projection};
#[cfg(feature = "cuda")]
pub use tiled_attention::{TiledAttention, TiledAttentionDims};
#[cfg(feature = "cuda")]
pub use weight_lora_rdma::{LoraAbKind, LoraLandTarget, RdmaLoraLoader};
pub use weight_peer::{WeightManifest, WeightTensorRecord};
#[cfg(feature = "cuda")]
pub use weight_tier_rdma::RdmaWeightLoader;
