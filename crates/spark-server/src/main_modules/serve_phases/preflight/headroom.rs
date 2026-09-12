// SPDX-License-Identifier: AGPL-3.0-only

//! The yardstick the #915 SSM decode-ring auto-fit is measured against.
//!
//! # Why the first pass fitted against the wrong number
//!
//! The first pass (`2224f5f11`) sized the ring from PRE-LOAD free memory.
//! Measured on 1xH100 2026-09-11 (`h100-round2-report.md`, stages 3 and 5,
//! plus the `preflight_probe.sh` sweep), that yardstick is ~6x too large and
//! the fitter therefore never fired at the batch sizes that matter:
//!
//! | `--max-batch-size` | ring chosen | fired? | serve |
//! |---|---|---|---|
//! | 4 | 8 | no | boots |
//! | 16 | 8 (18.94 GB) | no | **refused** by the KV stage |
//! | 32 | 8 (37.88 GB) | no | **refused** by the KV stage |
//! | 64 | 8 -> 4 | yes | still refused |
//!
//! The fitter's own WARN said it: `reserve 54.13 of 78.57 GB` — 78.57 GB is
//! free memory before a single weight byte is resident. The quantity that has
//! to accommodate the reserve is the POST-load KV headroom, and it is a
//! different stage (`spark_model::factory::build`) that enforces it, four
//! minutes later, with no shrink path — only a refusal:
//!
//! ```text
//! KV cache: 79.2 GB total x 90% util = 71.3 GB budget;
//!           33.2 GB pre-KV + 45.8 GB reserve -> 0 GB for KV
//! ```
//!
//! # The yardstick this module computes
//!
//! The SAME formula that stage uses, evaluated from what preflight can
//! already see:
//!
//! ```text
//! budget          = total_mem x --gpu-memory-utilization
//! pre_kv_estimate = weights + derived + buffer_arena + fixed_reserve
//! headroom        = budget - pre_kv_estimate
//! ```
//!
//! and the ring is then the largest `DECODE_RING_FIT_LADDER` depth for which
//! `ring_bytes + kv_floor <= headroom`. `fixed_reserve` is inside `pre_kv`
//! rather than beside it because `headroom` is defined as exactly the two
//! terms the fit trades off — ring against KV.
//!
//! Round-6 receipt (`h100-round6-report.md`, native-FP8 residency fix in):
//! weights 28.75 GB, derived 4.24 GB, arena 3.57 GiB, `fixed` = reserve minus
//! ring. At `--max-batch-size 16` that leaves ~29 GiB of headroom, the full
//! 8-slot ring (18.94 GiB) fits beside the KV floor, and the serve boots with
//! 13.6 GB of real KV. At 32 the 8-slot ring wants 37.88 GiB against ~27 GiB
//! of headroom, so the fit picks 4 — the depth the operator had to pass by
//! hand (`--ssm-decode-ring-slots 4`) to get that boot at all.
//!
//! # What it refuses to estimate
//!
//! Both new terms are ESTIMATES and neither is allowed to be a guess:
//!
//! * `weights` is the checkpoint's on-disk byte count. That equals resident
//!   bytes only for checkpoints Atlas loads verbatim; a BF16 checkpoint
//!   requantised to NVFP4 at load is a different number entirely.
//! * `derived` is `predicted_residency::predicted_derived_bytes`, which
//!   answers only for the native-FP8 dense route.
//!
//! So the post-load yardstick is used ONLY when the derived prediction is
//! available — the same route on which the on-disk count is a sound upper
//! bound on residency (the loader prunes, never grows: round 6's 28.75 GB
//! checkpoint settles at ~24.7 GB resident, so this over-states pre-KV by
//! ~4 GB and can only choose a SMALLER ring, never a larger one). Every other
//! model falls back to the pre-load free-memory yardstick and the log says
//! which one was used and why.

use atlas_core::config::ModelConfig;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype};

use crate::cli;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The default `kv_floor` context length, in tokens per sequence.
///
/// The fit must not hand the whole headroom to the ring and leave a KV cache
/// too small to admit `--max-batch-size` sequences: rollback depth is a
/// nice-to-have, a KV cache is the serve. 4096 is a deliberate FLOOR and not
/// a target — round 6 booted at bs=16 with 13.6 GB of KV (~415k tokens), far
/// above this — chosen so the reservation stays a guard rail rather than a
/// second, hidden concurrency cap.
///
/// Lever: `ATLAS_KV_FLOOR_TOKENS=<n>` (0 disables the floor entirely).
pub(super) const DEFAULT_KV_FLOOR_TOKENS: usize = 4096;

/// `ATLAS_KV_FLOOR_TOKENS`, or [`DEFAULT_KV_FLOOR_TOKENS`].
///
/// An unparseable value takes the default rather than 0: silently removing
/// the floor because of a typo is the failure this guard exists to prevent.
pub(super) fn kv_floor_tokens() -> usize {
    match std::env::var("ATLAS_KV_FLOOR_TOKENS") {
        Ok(v) => v.trim().parse().unwrap_or_else(|_| {
            tracing::warn!(
                "ATLAS_KV_FLOOR_TOKENS='{v}' is not a token count — using the default {}",
                DEFAULT_KV_FLOOR_TOKENS,
            );
            DEFAULT_KV_FLOOR_TOKENS
        }),
        Err(_) => DEFAULT_KV_FLOOR_TOKENS,
    }
}

/// What `preflight_reserve` needs from its caller to build the post-load
/// yardstick, and cannot derive from `args` + `config` alone.
pub(crate) struct PostLoadInputs<'a> {
    /// `gpu.total_memory()` — the same total the KV budget stage multiplies
    /// by `--gpu-memory-utilization`.
    pub(crate) total_mem: usize,
    /// The resolved checkpoint directory, for the on-disk weight estimate.
    pub(crate) model_dir: &'a std::path::Path,
    /// The KV dtype the cache will actually be built with, resolved through
    /// `serve_phases::kv_cache::resolve_kv_dtype_str` so preflight and the
    /// cache cannot disagree about bytes per token.
    pub(crate) kv_dtype: KvCacheDtype,
    /// Both `qwen3_attention::W8A8_PREFILL_KERNELS` are loaded for this
    /// target. Decides whether the loader builds the Q and O FP8 prefill
    /// twins (~1.5 GB on the 27B).
    pub(crate) w8a8_prefill_kernels: bool,
}

/// Every term of the fit decision, kept so the INFO line can print all of
/// them rather than a single derived number nobody can check.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Headroom {
    /// `total_mem x --gpu-memory-utilization`.
    pub(super) budget: usize,
    /// Checkpoint bytes on disk.
    pub(super) weights: usize,
    /// Derived copies the loader will build on top of them.
    pub(super) derived: usize,
    /// `BufferSizes::from_config(..).total_bytes()`.
    pub(super) arena: usize,
    /// The reserve terms that do NOT scale with ring depth.
    pub(super) fixed: usize,
    /// KV bytes for `--max-batch-size` sequences at the floor context.
    pub(super) kv_floor: usize,
    /// `budget - (weights + derived + arena + fixed)`, saturating.
    pub(super) headroom: usize,
}

/// What the ring is fitted against.
pub(super) enum Yardstick {
    /// Pre-load free memory — the first pass's behaviour, kept for every
    /// route whose post-load residency cannot be predicted. The `&'static
    /// str` is why, for the log.
    PreLoadFree(&'static str),
    /// The predicted post-load KV headroom.
    PostLoad(Headroom),
}

impl Yardstick {
    /// `(bytes that must fit BESIDE the ring, the limit both must fit in)` —
    /// the two arguments `ssm_reserve::fit_decode_ring_slots` takes.
    ///
    /// The two modes differ only in what the ring is competing with: free
    /// memory against the rest of the reserve, or predicted headroom against
    /// the KV floor. One ladder, two bases.
    pub(super) fn ladder_basis(
        &self,
        reserve_without_ring: usize,
        free_mem: usize,
    ) -> (usize, usize) {
        match self {
            Self::PreLoadFree(_) => (reserve_without_ring, free_mem),
            Self::PostLoad(h) => (h.kv_floor, h.headroom),
        }
    }

    /// What the shrink WARN calls the thing it sized from.
    pub(super) fn name(&self) -> &'static str {
        match self {
            Self::PreLoadFree(_) => "pre-load free memory",
            Self::PostLoad(_) => "the predicted post-load KV headroom",
        }
    }

    /// What the shrink WARN calls the sum it checked against that yardstick:
    /// the whole reserve in pre-load mode, the ring plus the KV floor in
    /// post-load mode.
    pub(super) fn total_label(&self) -> &'static str {
        match self {
            Self::PreLoadFree(_) => "reserve",
            Self::PostLoad(_) => "ring + KV floor",
        }
    }
}

/// Build the post-load yardstick, or say why it is unavailable.
///
/// `fixed_reserve` and `arena` come from the caller because they are already
/// computed there and are the reserve's own arithmetic; everything else is
/// estimated here.
pub(super) fn post_load_yardstick(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    inputs: &PostLoadInputs<'_>,
    fixed_reserve: usize,
    arena: usize,
) -> Yardstick {
    let route = spark_model::weight_loader::predicted_residency::Fp8RouteInputs::from_env(
        config,
        inputs.w8a8_prefill_kernels,
    );
    let estimate =
        spark_model::weight_loader::predicted_residency::predicted_derived_bytes(config, &route);
    let Some(derived) = estimate.bytes() else {
        return Yardstick::PreLoadFree(
            estimate
                .reason()
                .unwrap_or("the derived-residency prediction declined"),
        );
    };
    let Some(weights) = checkpoint_bytes(inputs.model_dir) else {
        return Yardstick::PreLoadFree("the checkpoint's on-disk size could not be read");
    };
    let budget = (inputs.total_mem as f64 * args.gpu_memory_utilization) as usize;
    let weights = weights as usize;
    let derived = derived as usize;
    let pre_kv = weights
        .saturating_add(derived)
        .saturating_add(arena)
        .saturating_add(fixed_reserve);
    Yardstick::PostLoad(Headroom {
        budget,
        weights,
        derived,
        arena,
        fixed: fixed_reserve,
        kv_floor: kv_floor_bytes(args, config, inputs.kv_dtype),
        headroom: budget.saturating_sub(pre_kv),
    })
}

/// KV bytes for `--max-batch-size` sequences at the floor context length.
///
/// Bytes per token come from the KV cache's OWN helper
/// (`KvCacheConfig::block_bytes_kv_all_layers`, which
/// `PagedKvCache::compute_num_blocks` divides the budget by), so the floor
/// and the real pool are priced by one function. MLA is mirrored from
/// `factory/build.rs`'s step 5: an absorbed cache stores one head of
/// `kv_lora_rank + qk_rope_head_dim`, not `num_key_value_heads x head_dim`.
///
/// `layer_dims` is left empty because `config.kv_layer_dims` is populated by
/// the LOADER and does not exist yet; `dims_for_layer` then falls back to the
/// global pair, which is exact for every homogeneous model — and this
/// yardstick is only used on the native-FP8 dense route, which is one.
pub(super) fn kv_floor_bytes(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    kv_dtype: KvCacheDtype,
) -> usize {
    let tokens = kv_floor_tokens().min(args.max_seq_len);
    if tokens == 0 || args.max_batch_size == 0 {
        return 0;
    }
    args.max_batch_size * tokens * kv_bytes_per_token(args, config, kv_dtype)
}

/// Bytes one cached token costs across every attention layer, K and V.
pub(super) fn kv_bytes_per_token(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    kv_dtype: KvCacheDtype,
) -> usize {
    let (num_kv_heads, head_dim) = if config.kv_lora_rank > 0 {
        (1, config.kv_lora_rank + config.qk_rope_head_dim)
    } else {
        (config.num_key_value_heads, config.head_dim)
    };
    let block_size = args.block_size.max(1);
    let kv = KvCacheConfig {
        block_size,
        num_kv_heads,
        head_dim,
        num_layers: config.num_attention_layers(),
        dtype: kv_dtype,
        layer_dtypes: Vec::new(),
        layer_dims: Vec::new(),
        cache_blocks_per_seq: None,
    };
    kv.block_bytes_kv_all_layers() / block_size
}

/// The checkpoint's on-disk byte count, from the safetensors index when there
/// is one and from the directory listing otherwise.
///
/// This is the pre-load stand-in for the `Weights: X GB` the post-load audit
/// prints from `store.resident_bytes()`. It is an UPPER bound on residency
/// for a checkpoint loaded verbatim: the loader prunes tensors it has folded
/// into derived copies (round 6: 28.75 GB of checkpoint settles at ~24.7 GB
/// resident once the fused `[QKV|Z]` sources are dropped), and it never
/// invents weights. Over-stating pre-KV can only shrink the fitted ring,
/// which is the safe direction.
///
/// `None` — no index, no shard files, unreadable directory — takes the caller
/// back to the pre-load yardstick rather than fitting against a zero.
pub(super) fn checkpoint_bytes(model_dir: &std::path::Path) -> Option<u64> {
    if let Some(total) = indexed_shard_bytes(model_dir) {
        return Some(total);
    }
    // No index: sum the weight files present. Safetensors first, GGUF only
    // when there are none, so a directory carrying both is not double-counted.
    for ext in ["safetensors", "gguf"] {
        let total: u64 = std::fs::read_dir(model_dir)
            .ok()?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|x| x.to_str())
                    .is_some_and(|x| x == ext)
            })
            .filter_map(|e| e.metadata().ok().map(|m| m.len()))
            .sum();
        if total > 0 {
            return Some(total);
        }
    }
    None
}

/// Sum of the DISTINCT shard files a `*.safetensors.index.json` references.
///
/// Read from the index rather than from `metadata.total_size` (which many
/// re-quants omit or compute differently) and de-duplicated, because the
/// weight map names one shard per tensor and a 66-shard checkpoint would
/// otherwise be counted 1,606 times. Resolution order mirrors
/// `spark_runtime::fast_weights::header::resolve_shards`.
fn indexed_shard_bytes(model_dir: &std::path::Path) -> Option<u64> {
    let index = [
        "model.safetensors.index.json",
        "consolidated.safetensors.index.json",
    ]
    .into_iter()
    .map(|n| model_dir.join(n))
    .find(|p| p.exists())?;
    let raw = std::fs::read_to_string(&index).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let map = v.get("weight_map")?.as_object()?;
    let shards: std::collections::HashSet<&str> = map.values().filter_map(|s| s.as_str()).collect();
    let total: u64 = shards
        .iter()
        .filter_map(|s| std::fs::metadata(model_dir.join(s)).ok().map(|m| m.len()))
        .sum();
    (total > 0).then_some(total)
}

impl Headroom {
    /// Every term of the decision, in the order the arithmetic runs, so an
    /// operator reading one INFO line can redo the subtraction.
    pub(super) fn describe(&self, util: f64, ring_bytes: usize, slots: usize) -> String {
        let gb = |b: usize| b as f64 / GIB;
        format!(
            "budget {:.2} GB ({:.0}% util) = weights {:.2} + derived {:.2} + arena {:.2} + \
             fixed reserve {:.2} -> headroom {:.2} GB; ring({}) {:.2} + KV floor {:.2} = {:.2} GB",
            gb(self.budget),
            util * 100.0,
            gb(self.weights),
            gb(self.derived),
            gb(self.arena),
            gb(self.fixed),
            gb(self.headroom),
            slots,
            gb(ring_bytes),
            gb(self.kv_floor),
            gb(ring_bytes + self.kv_floor),
        )
    }
}

#[cfg(test)]
#[path = "headroom_tests.rs"]
mod tests;
