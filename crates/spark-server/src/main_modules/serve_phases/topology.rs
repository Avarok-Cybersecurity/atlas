// SPDX-License-Identifier: AGPL-3.0-only

//! TP/EP topology resolution + NCCL communicator init.

// `Context` only used by the nccl-feature `init_nccl_comm` to wrap
// NCCL bootstrap errors; cuda-without-nccl and metal builds don't
// reach that path.
#[cfg(feature = "nccl")]
use anyhow::Context;
use anyhow::Result;

use atlas_core::config::ModelConfig;

use crate::cli;

pub(crate) struct Topology {
    pub(crate) world_size: usize,
    pub(crate) tp_size: usize,
    pub(crate) ep_size: usize,
    pub(crate) tp_rank: usize,
    pub(crate) ep_rank: usize,
}

pub(crate) fn resolve_topology(
    args: &cli::ServeArgs,
    config: &mut ModelConfig,
) -> Result<Topology> {
    let (tp_size, ep_size) = if args.tp_size == 1 && args.ep_size == 1 && args.world_size > 1 {
        (1usize, args.world_size)
    } else {
        (args.tp_size.max(1), args.ep_size.max(1))
    };
    let derived_world = if tp_size == ep_size {
        tp_size
    } else {
        tp_size * ep_size
    };
    let world_size = if args.world_size <= 1 && (tp_size > 1 || ep_size > 1) {
        tracing::info!(
            "Auto-derived world_size={} from --tp-size {} --ep-size {} (rule: \
             tp==ep → overlapping = tp; else orthogonal = tp×ep). Pass \
             --world-size to override.",
            derived_world,
            tp_size,
            ep_size,
        );
        derived_world
    } else {
        args.world_size
    };
    let (tp_rank, ep_rank) = if tp_size == ep_size && tp_size == world_size && tp_size > 1 {
        (args.rank, args.rank)
    } else if world_size == tp_size * ep_size {
        (args.rank % tp_size, args.rank / tp_size)
    } else {
        anyhow::bail!(
            "Invalid parallelism topology: world_size={} but tp_size={} × ep_size={} = {}. \
             Either use orthogonal mesh (world = tp × ep) or overlapping groups \
             (world = tp = ep, used for 2-GPU TP+EP composition).",
            world_size,
            tp_size,
            ep_size,
            tp_size * ep_size,
        );
    };
    config.tp_rank = tp_rank;
    config.tp_world_size = tp_size;
    config.ep_rank = ep_rank;
    config.ep_world_size = ep_size;
    if tp_size > 1 {
        let loader = spark_model::factory::loader_for_config(config)?;
        if !loader.supports_tp() {
            anyhow::bail!(
                "TP (--tp-size > 1) is not supported by the {} weight loader. \
                 Run with --tp-size 1 (EP-only). To extend TP to this architecture, \
                 wire `crate::tp_shard::slice_for_rank` per attention/MoE/SSM \
                 tensor in the loader and override `ModelWeightLoader::supports_tp()` \
                 to return true. See `weight_loader/minimax.rs` as the reference.",
                config.model_type,
            );
        }
        drop(loader);
        if !config.num_attention_heads.is_multiple_of(tp_size) {
            anyhow::bail!(
                "TP requires num_attention_heads ({}) divisible by tp_size ({})",
                config.num_attention_heads,
                tp_size,
            );
        }
        if !config.num_key_value_heads.is_multiple_of(tp_size) {
            anyhow::bail!(
                "TP requires num_key_value_heads ({}) divisible by tp_size ({})",
                config.num_key_value_heads,
                tp_size,
            );
        }
        config.num_attention_heads /= tp_size;
        config.num_key_value_heads /= tp_size;
        // GDN HeadParallel: linear-attention (SSM) key/value head counts are
        // sharded exactly like attention heads — each rank owns a contiguous
        // head range; the recurrence is head-parallel with one all-reduce after
        // out_proj. Only relevant for SSM-hybrid models (linear_*_heads > 0);
        // pure-attention configs leave these at 0 and skip the divide.
        if config.linear_num_key_heads > 0 || config.linear_num_value_heads > 0 {
            if !config.linear_num_key_heads.is_multiple_of(tp_size) {
                anyhow::bail!(
                    "TP requires linear_num_key_heads ({}) divisible by tp_size ({})",
                    config.linear_num_key_heads,
                    tp_size,
                );
            }
            if !config.linear_num_value_heads.is_multiple_of(tp_size) {
                anyhow::bail!(
                    "TP requires linear_num_value_heads ({}) divisible by tp_size ({})",
                    config.linear_num_value_heads,
                    tp_size,
                );
            }
            config.linear_num_key_heads /= tp_size;
            config.linear_num_value_heads /= tp_size;
        }
        tracing::info!(
            "TP-local head counts: num_attention_heads={}, num_key_value_heads={}, \
             linear_num_key_heads={}, linear_num_value_heads={}",
            config.num_attention_heads,
            config.num_key_value_heads,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
        );
    }
    if world_size > 1 {
        let (start, end) = config.local_expert_range();
        tracing::info!(
            "Parallelism: global rank {}/{} (tp_rank={}/{}, ep_rank={}/{}), local experts [{}, {})",
            args.rank,
            world_size,
            tp_rank,
            tp_size,
            ep_rank,
            ep_size,
            start,
            end,
        );
    }
    Ok(Topology {
        world_size,
        tp_size,
        ep_size,
        tp_rank,
        ep_rank,
    })
}

/// `max_batch_tokens` and `hidden_size` size the 2-rank all-reduce receive
/// buffer. Together they bound the largest payload any caller can hand a
/// collective: prefill MoE, prefill attention and prefill SSM all reduce a
/// `[num_tokens, hidden_size]` BF16 tensor, and `num_tokens` is capped by
/// `max_batch_tokens` (the same bound the `moe_output` arena buffer is sized
/// on). Deriving the capacity here is what keeps a wider model or a larger
/// `--max-prefill-tokens` from overrunning a fixed allocation.
#[cfg(feature = "nccl")]
pub(crate) fn init_nccl_comm(
    args: &cli::ServeArgs,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    world_size: usize,
    max_batch_tokens: usize,
    hidden_size: usize,
) -> Result<Option<std::sync::Arc<dyn spark_comm::CommBackend>>> {
    use spark_comm::CommBackend;
    if world_size <= 1 {
        return Ok(None);
    }
    let recv_capacity = spark_comm::nccl_backend::required_recv_bytes(
        max_batch_tokens,
        hidden_size,
        spark_comm::nccl_backend::ALL_REDUCE_DTYPE_BYTES,
    )
    .context("Failed to size the NCCL receive buffer")?;
    tracing::info!(
        "Initializing NCCL: rank {}/{}, master {}:{}, recv_buffer {} MiB \
         (max_batch_tokens={} × hidden_size={} × {} B)",
        args.rank,
        world_size,
        args.master_addr,
        args.master_port,
        recv_capacity / (1024 * 1024),
        max_batch_tokens,
        hidden_size,
        spark_comm::nccl_backend::ALL_REDUCE_DTYPE_BYTES,
    );
    let cuda_stream = gpu.default_stream();
    let backend = spark_comm::NcclBackend::new(
        args.rank,
        world_size,
        &args.master_addr,
        args.master_port,
        cuda_stream,
        recv_capacity,
    )
    .context("Failed to initialize NCCL")?;
    tracing::info!("NCCL initialized: rank {}", backend.rank());
    Ok(Some(
        std::sync::Arc::new(backend) as std::sync::Arc<dyn spark_comm::CommBackend>
    ))
}

/// CUDA-without-NCCL variant (SCALE/AMD gfx1151): the CUDA compute
/// backend is active but no NCCL library is linked, so multi-GPU
/// collectives are unavailable. `world_size > 1` is rejected explicitly
/// so a misconfigured `--rank > 0` invocation fails fast instead of
/// silently degrading to single-rank.
#[cfg(all(feature = "cuda", not(feature = "nccl")))]
pub(crate) fn init_nccl_comm(
    _args: &cli::ServeArgs,
    _gpu: &dyn spark_runtime::gpu::GpuBackend,
    world_size: usize,
    _max_batch_tokens: usize,
    _hidden_size: usize,
) -> Result<Option<std::sync::Arc<dyn spark_comm::CommBackend>>> {
    if world_size > 1 {
        anyhow::bail!(
            "multi-rank NCCL is not available in this build (cuda feature \
             without nccl — SCALE/AMD gfx1151 has no NCCL library); \
             single-device only"
        );
    }
    Ok(None)
}

/// Metal-feature variant: NCCL multi-GPU isn't reachable on a single
/// Apple Silicon device, so collective ops fall back to the no-op
/// `SingleGpuBackend`. `world_size > 1` is rejected explicitly so a
/// misconfigured `--rank > 0` invocation fails fast instead of
/// silently degrading to single-rank.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(crate) fn init_nccl_comm(
    _args: &cli::ServeArgs,
    _gpu: &dyn spark_runtime::gpu::GpuBackend,
    world_size: usize,
    _max_batch_tokens: usize,
    _hidden_size: usize,
) -> Result<Option<std::sync::Arc<dyn spark_comm::CommBackend>>> {
    if world_size > 1 {
        anyhow::bail!(
            "multi-rank NCCL is not available on Apple Silicon (metal feature); \
             single-device only"
        );
    }
    Ok(None)
}
