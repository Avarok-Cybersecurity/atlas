// SPDX-License-Identifier: AGPL-3.0-only

//! Fast safetensors loader (InstantTensor-style) — pure Rust.
//!
//! Two wins over the mmap-based loader in [`crate::weights`]:
//!
//! 1. **`O_DIRECT`** reads. Bypasses the OS page cache, so the bytes never
//!    compete with GPU allocations on GB10 unified memory. The mmap path
//!    already works around this with `POSIX_FADV_DONTNEED` post-load; here
//!    we avoid the pollution in the first place.
//! 2. **Pipelined read/copy**. One background reader thread fetches the
//!    next tensor while the main thread does `copy_h2d` for the current
//!    one. Overlaps disk I/O with the host→device memcpy.
//!
//! Behavioural parity with [`crate::weights::SafetensorsLoader`] is
//! preserved — same EP filtering, same OOM pre-flight, same UVM fallback
//! on GPU allocation failure, same extra-weights handling.
//!
//! One deliberate divergence: with a [`FastSafetensorsLoader::pool_predicate`]
//! installed, the packed-EXL3 quartets it selects are uploaded into ONE
//! allocation per (shard, class) — `pool.rs` — instead of one per tensor,
//! because `cuMemAlloc_v2` on GB10 charges sub-2 MiB requests a 2 MiB
//! chunk-tail tax (~17.9 GiB on the 4.05bpw Qwen3.8-Flash-Next export).
//! Every other tensor still takes the per-tensor path, byte for byte.

use crate::gpu::GpuBackend;
use crate::weights::{
    WeightArena, WeightLoader, WeightStore, WeightTensor, check_oom_guard, estimate_has_fp8,
    estimate_load_bytes,
};
use anyhow::{Result, bail};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

mod direct_io;
mod fs_probe;
mod header;
mod pool;
mod shard;

use header::resolve_shards;
use shard::{load_shard_fast, select_pooled_prefixes};

/// `(prefix, trellis_shape) -> pool?` — decides which EXL3 linears' four
/// tensors are pooled. Lives in spark-model (it needs the native-serving
/// gates and kernel envelopes), injected here.
pub type PoolPredicate = Arc<dyn Fn(&str, &[usize]) -> bool + Send + Sync>;

/// Pure-Rust InstantTensor-style loader. Same public shape as
/// [`crate::weights::SafetensorsLoader`].
pub struct FastSafetensorsLoader {
    pub ep_rank: usize,
    pub ep_world_size: usize,
    pub num_experts: usize,
    pub peak_memory_multiplier: Option<f64>,
    /// Skip the W4A4 `*.input_scale` activation scales at load.
    ///
    /// ModelOpt NVFP4 checkpoints ship one 0-dim F32 scalar per quantized
    /// projection. On a 512-expert model that is ~74k four-byte allocations,
    /// each taking a full allocation granule — GBs of padding for values
    /// Atlas never reads, because it serves w4a16 (BF16 activations) and the
    /// NVFP4 loader already treats the key as optional.
    ///
    /// OPT-IN: `step3p7` reads this key on its own path, so it must stay off
    /// unless the model's loader is known not to need it.
    pub skip_activation_scales: bool,
    /// Skip `mtp.*` tensors at load.
    ///
    /// For models whose loader deliberately does not build an MTP head,
    /// uploading its weights is pure waste — on Qwen3.8-Flash-Next that is a
    /// 1.49 GB expert shard plus the MTP backbone, held resident while the KV
    /// cache goes without.
    ///
    /// OPT-IN: a model that DOES build an MTP head must keep them, so this is
    /// set only where `load_mtp_weights` is known to return `None`.
    pub skip_mtp: bool,
    /// When true (default), attempt `O_DIRECT`; fall back to buffered reads if
    /// the filesystem rejects it (tmpfs, overlayfs, some FUSE backends).
    pub try_direct_io: bool,
    /// Per-shard heuristic cap: if a shard's tensor count exceeds this,
    /// we skip `O_DIRECT` for that shard and fall back to buffered +
    /// pipelined reads even when [`Self::try_direct_io`] is `true`.
    ///
    /// Motivation: `O_DIRECT`'s 4 KiB-aligned per-tensor `pread` has a
    /// fixed syscall + copy overhead that kernel readahead amortises for
    /// free on the buffered path. Benchmarks on GB10 showed buffered wins
    /// above ~5k tensors/shard; O_DIRECT wins below. Set to [`usize::MAX`]
    /// to disable.
    pub direct_io_tensor_cap: usize,
    /// When true, advise the kernel to read a whole buffered shard
    /// sequentially before the per-tensor copy loop starts. This helps NFS
    /// mounts where many small tensor reads defeat normal readahead.
    pub prefetch_shards: bool,
    /// Pool the `.trellis/.suh/.svh/.mul1` quartets of every EXL3 prefix
    /// this predicate admits into one arena per (shard, class) — see
    /// `pool.rs`. `None` (the default, and every non-EXL3 checkpoint):
    /// per-tensor allocation for everything. The serve path installs
    /// `spark_model::weight_map::exl3_fast_load_pool_predicate()`, which
    /// admits exactly the prefixes the materialize pass will keep packed
    /// and honours the `ATLAS_EXL3_WEIGHT_POOL=0` kill switch.
    pub pool_predicate: Option<PoolPredicate>,
    /// Skip a multimodal checkpoint's vision tower.
    ///
    /// Set by the caller from `ModelWeightLoader::binds_vision_encoder()`:
    /// false by default, true only when the model's loader is a text-only
    /// port that will never bind the tower. Reading it anyway costs the full
    /// tower in unified memory (1.05 GiB/rank on GLM-5.3's checkpoint) from
    /// load time until `build_model` frees it — which is after the inference
    /// -buffer preflight has already refused the serve.
    pub skip_vision: bool,
}

/// Per-shard loader knobs, read-only across the shard loop.
pub(crate) struct ShardOpts<'a> {
    pub(crate) try_direct_io: bool,
    pub(crate) direct_io_tensor_cap: usize,
    pub(crate) prefetch_shards: bool,
    /// EXL3 prefixes whose quartets are pooled (empty: nothing pooled).
    pub(crate) pooled_prefixes: &'a HashSet<String>,
}

/// Where a shard load writes: the tensor map, deferred locators, the
/// arenas it allocated, and the two once-only log latches.
pub(crate) struct ShardSink<'a> {
    pub(crate) out: &'a mut HashMap<String, WeightTensor>,
    pub(crate) deferred: &'a mut HashMap<String, crate::weights::DeferredTensor>,
    pub(crate) arenas: &'a mut Vec<WeightArena>,
    pub(crate) offload_logged: &'a mut bool,
    pub(crate) pool_fallback_logged: &'a mut bool,
}

/// Is this tensor part of a multimodal checkpoint's vision tower?
///
/// Same three spellings `build_model`'s unbound-tower reclaim matches, kept
/// here so the load-time skip and the post-bind free can never disagree.
pub fn is_vision_tensor(name: &str) -> bool {
    name.starts_with("model.visual.")
        || name.starts_with("model.vision")
        || name.starts_with("visual.")
}

/// Default tensor-count cap for per-shard `O_DIRECT`. Above this, the fast
/// loader uses buffered reads even when `try_direct_io = true`. See the
/// field doc on [`FastSafetensorsLoader::direct_io_tensor_cap`].
pub const DEFAULT_DIRECT_IO_TENSOR_CAP: usize = 5000;

impl Default for FastSafetensorsLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[path = "skip.rs"]
mod skip;

impl FastSafetensorsLoader {
    pub fn new() -> Self {
        Self {
            ep_rank: 0,
            ep_world_size: 1,
            num_experts: 0,
            peak_memory_multiplier: None,
            skip_activation_scales: false,
            skip_mtp: false,
            try_direct_io: true,
            direct_io_tensor_cap: DEFAULT_DIRECT_IO_TENSOR_CAP,
            prefetch_shards: false,
            pool_predicate: None,
            skip_vision: false,
        }
    }

    pub fn with_ep(ep_rank: usize, ep_world_size: usize, num_experts: usize) -> Self {
        Self {
            ep_rank,
            ep_world_size,
            num_experts,
            peak_memory_multiplier: None,
            skip_activation_scales: false,
            skip_mtp: false,
            try_direct_io: true,
            direct_io_tensor_cap: DEFAULT_DIRECT_IO_TENSOR_CAP,
            prefetch_shards: false,
            pool_predicate: None,
            skip_vision: false,
        }
    }
}

impl WeightLoader for FastSafetensorsLoader {
    fn load(
        &self,
        model_dir: &Path,
        gpu: &dyn GpuBackend,
        oom_reserve_bytes: usize,
    ) -> Result<WeightStore> {
        let skip_fn = |name: &str| self.should_skip_tensor(name);

        // Resolve shard list (sharded index, single file, or unindexed shards).
        let (shard_files, tensor_to_shard): (Vec<PathBuf>, Option<HashMap<String, String>>) =
            resolve_shards(model_dir)?;

        // Pre-flight OOM estimate (identical to SafetensorsLoader).
        //
        // The n-gram tables are DEFERRED further down — they are never
        // uploaded, so counting them here refuses a model that fits. On
        // LongCat-Flash-Lite they are 62.8 of the checkpoint's 138 GB, which
        // is the difference between a 167 GB "peak" and a 98 GB one.
        let preflight_skip = |name: &str| skip_fn(name) || crate::weights::is_ngram_table(name);
        {
            let estimated = estimate_load_bytes(&shard_files, &preflight_skip)?;
            let has_fp8 = estimate_has_fp8(&shard_files, &preflight_skip)?;
            let mult = self
                .peak_memory_multiplier
                .unwrap_or(if has_fp8 { 1.5 } else { 1.3 });
            let peak = (estimated as f64 * mult) as usize;
            let free = gpu.free_memory()?;
            let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
            tracing::info!(
                "Fast-load pre-flight: {:.2} GB on-disk, {:.1}x overhead = {:.2} GB peak, \
                 {:.2} GB free, {:.1} GB reserve (FP8: {})",
                gib(estimated),
                mult,
                gib(peak),
                gib(free),
                gib(oom_reserve_bytes),
                has_fp8,
            );
            crate::progress::preflight(gib(estimated), gib(free));
            if peak + oom_reserve_bytes > free {
                bail!(
                    "OOM pre-flight: peak {:.2} GB + {:.2} GB reserve exceeds {:.2} GB free. \
                     Use a smaller quantization or add more GPUs for EP.",
                    gib(peak),
                    gib(oom_reserve_bytes),
                    gib(free),
                );
            }
        }

        // Pooling pre-pass: the EXL3 prefixes whose quartets go into arenas,
        // decided model-wide from the headers (already parsed for the
        // pre-flight; a few ms) so a layer straddling two shards resolves
        // its aux tensors' class in either shard.
        let pooled_prefixes: HashSet<String> = match &self.pool_predicate {
            Some(pred) => {
                let set = select_pooled_prefixes(
                    &shard_files,
                    tensor_to_shard.as_ref(),
                    &skip_fn,
                    pred.as_ref(),
                )?;
                tracing::info!(
                    "EXL3 weight pool: {} kept-packed prefixes ({} tensors) will be pooled \
                     into one arena per (shard, class); everything else per tensor",
                    set.len(),
                    set.len() * 4,
                );
                set
            }
            None => HashSet::new(),
        };
        let shard_opts = ShardOpts {
            try_direct_io: self.try_direct_io,
            direct_io_tensor_cap: self.direct_io_tensor_cap,
            prefetch_shards: self.prefetch_shards,
            pooled_prefixes: &pooled_prefixes,
        };

        // Load each shard. Loaded tensors filtered by EP rules upstream.
        let mut weights: HashMap<String, WeightTensor> = HashMap::new();
        // Locations of tensors deliberately NOT uploaded (the n-gram tables).
        let mut deferred: HashMap<String, crate::weights::DeferredTensor> = HashMap::new();
        // Arenas the shards allocated; adopted by the store below.
        let mut arenas: Vec<WeightArena> = Vec::new();
        let total_shards = shard_files.len();
        let initial_free = gpu.free_memory()?;
        let mut offload_logged = false;
        let mut pool_fallback_logged = false;

        for (i, shard_path) in shard_files.iter().enumerate() {
            // When an index is present, only load the tensors it routes here;
            // otherwise load everything in the shard. `None` means "load all".
            let shard_name = shard_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            let tensor_filter: Option<Vec<String>> = tensor_to_shard.as_ref().map(|map| {
                map.iter()
                    .filter(|(_, s)| *s == shard_name)
                    .map(|(t, _)| t.clone())
                    .collect()
            });

            tracing::info!(
                "Fast-loading shard {}/{}: {}{}",
                i + 1,
                total_shards,
                shard_name,
                tensor_filter
                    .as_ref()
                    .map(|v| format!(" ({} tensors)", v.len()))
                    .unwrap_or_default(),
            );
            crate::progress::shard_start(i + 1, total_shards, shard_name);

            load_shard_fast(
                shard_path,
                tensor_filter.as_deref(),
                gpu,
                &skip_fn,
                &shard_opts,
                &mut ShardSink {
                    out: &mut weights,
                    deferred: &mut deferred,
                    arenas: &mut arenas,
                    offload_logged: &mut offload_logged,
                    pool_fallback_logged: &mut pool_fallback_logged,
                },
            )?;

            let free_now = gpu.free_memory().unwrap_or(0);
            let used = initial_free.saturating_sub(free_now);
            tracing::info!(
                "  Shard {}/{} done — GPU memory: {:.2} GB used, {:.2} GB free",
                i + 1,
                total_shards,
                used as f64 / (1024.0 * 1024.0 * 1024.0),
                free_now as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            crate::progress::shard_done(
                i + 1,
                total_shards,
                used as f64 / (1024.0 * 1024.0 * 1024.0),
                free_now as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            if !offload_logged {
                check_oom_guard(
                    gpu,
                    oom_reserve_bytes,
                    &format!("fast weight loading (shard {}/{})", i + 1, total_shards),
                )?;
            }
        }

        // Extra weights (e.g. MTP grafted from another quantization).
        let no_skip = |_: &str| false;
        let extra = model_dir.join("extra_weights.safetensors");
        if extra.exists() {
            tracing::info!("Fast-loading extra_weights.safetensors");
            let mut extra_offload = false;
            load_shard_fast(
                &extra,
                None,
                gpu,
                &no_skip,
                &shard_opts,
                &mut ShardSink {
                    out: &mut weights,
                    deferred: &mut deferred,
                    arenas: &mut arenas,
                    offload_logged: &mut extra_offload,
                    pool_fallback_logged: &mut pool_fallback_logged,
                },
            )?;
        }

        tracing::info!("Fast-loaded {} weight tensors", weights.len());
        let mut store = WeightStore::from_map(weights);
        for (name, d) in deferred {
            store.defer(name, d);
        }
        if !arenas.is_empty() {
            let bytes: usize = arenas.iter().map(|a| a.bytes).sum();
            tracing::info!(
                "EXL3 weight pool: {} arena(s), {:.2} GB owned by the weight store",
                arenas.len(),
                bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        }
        for a in arenas {
            store.adopt_arena(a);
        }
        Ok(store)
    }
}


#[cfg(test)]
mod skip_vision_tests {
    use super::{FastSafetensorsLoader, is_vision_tensor};

    fn loader(skip_vision: bool, ep: usize) -> FastSafetensorsLoader {
        let mut l = FastSafetensorsLoader::with_ep(0, ep, 288);
        l.skip_vision = skip_vision;
        l
    }

    #[test]
    fn vision_names_are_recognised() {
        assert!(is_vision_tensor("model.visual.blocks.0.attn.proj.weight"));
        assert!(is_vision_tensor(
            "model.vision_tower.encoder.layer.0.weight"
        ));
        assert!(is_vision_tensor("visual.merger.proj.weight"));
        assert!(!is_vision_tensor(
            "model.language_model.layers.45.eh_proj.weight"
        ));
        // The trap: a text tensor whose name merely CONTAINS "vision".
        assert!(!is_vision_tensor(
            "model.language_model.layers.3.mlp.revision.weight"
        ));
    }

    #[test]
    fn skip_vision_drops_only_the_tower() {
        let l = loader(true, 2);
        assert!(l.should_skip_tensor("model.visual.blocks.0.attn.proj.weight"));
        assert!(!l.should_skip_tensor("model.language_model.layers.45.eh_proj.weight"));
        assert!(!l.should_skip_tensor("lm_head.weight"));
    }

    #[test]
    fn skip_vision_applies_without_ep() {
        // The EP short-circuit must not swallow the vision rule at ep=1.
        let l = loader(true, 1);
        assert!(l.should_skip_tensor("model.visual.blocks.0.attn.proj.weight"));
        assert!(!l.should_skip_tensor("model.layers.0.self_attn.q_proj.weight"));
    }

    #[test]
    fn default_loader_keeps_the_tower() {
        let l = loader(false, 2);
        assert!(!l.should_skip_tensor("model.visual.blocks.0.attn.proj.weight"));
        assert!(!FastSafetensorsLoader::new().skip_vision);
    }

    #[test]
    fn ep_expert_filtering_is_unchanged_by_the_vision_rule() {
        let l = loader(true, 2); // ep_rank 0 of 2, 288 experts -> keeps 0..143
        assert!(
            !l.should_skip_tensor("model.language_model.layers.4.mlp.experts.7.up_proj.weight")
        );
        assert!(
            l.should_skip_tensor("model.language_model.layers.4.mlp.experts.200.up_proj.weight")
        );
    }
}
