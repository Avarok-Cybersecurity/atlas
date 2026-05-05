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

pub mod cuda_graph;
pub mod cuda_min;
pub mod cuda_module;

// Re-export the module/event/launch helpers from their new home so existing
// `use spark_storage::cuda_min::{CudaModule, CudaEvent, launch_kernel}` paths
// keep working.
pub use cuda_module::{CudaEvent, CudaModule, launch_kernel};
pub mod attention_ref;
pub mod backend;
pub mod bench;
pub mod config;
pub mod eviction;
pub mod group;
pub mod high_speed_swap;
pub mod layout;
pub mod predictor;
pub mod predictor_ref;
pub mod probe;
pub mod projection;
pub mod scratch_pool;
pub mod tiled_attention;

pub use backend::{IoUringBackend, PosixBackend, ReadRequest, StorageBackend};
pub use config::HighSpeedSwapConfig;
pub use eviction::EvictionPolicy;
pub use high_speed_swap::{HighSpeedSwap, ModelDims, install_local, local_installed, with_local};

pub use predictor::{Predictor, PredictorDims};
pub use probe::{Backend, ProbeConfig, ProbeResult, run_probe};
pub use projection::{PredictorShape, build_projection};
pub use tiled_attention::{TiledAttention, TiledAttentionDims};
