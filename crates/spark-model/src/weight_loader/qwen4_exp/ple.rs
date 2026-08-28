// SPDX-License-Identifier: AGPL-3.0-only

//! PLE weights: the projections, the three norms, the dilated conv, and the
//! 320M-row n-gram table served off NVMe.
//!
//! ```text
//! {lp}.ple.key_proj.weight                       [hc*H, ple_embed_dim]
//! {lp}.ple.value_proj.weight                     [H,    ple_embed_dim]
//! {lp}.ple.norm_key/norm_query/norm_conv.weight  [hc*H]
//! {lp}.ple.conv1d.weight                         [hc*H, 1, K]
//! {lp}.ple.ple_embedding.layer_multipliers       [ngram_size]   I64
//! {lp}.ple.ple_embedding.ngram_heads_offsets     [ngram_heads]  I64
//! {lp}.ple.ple_embedding.ngram_heads_vocab_sizes [ngram_heads]  I64
//! {lp}.ple.ple_embedding.ngram_embedding.shard_{0..127}.weight  [R, 160] BF16
//! ```
//!
//! The 128 shards are ONE logical table of `128 * R` rows. They live in a
//! single safetensors file but are NOT laid out consecutively — other weights
//! interleave — so the row cache is opened SEGMENTED, with each shard's own
//! base offset. A single-offset open would read the wrong rows for every
//! shard past the first and, since the rows are all valid embeddings, would
//! do it silently.

#[cfg(feature = "cuda")]
use anyhow::Context;
use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

#[cfg(feature = "cuda")]
use crate::layers::ngram_embed::NgramTable;
use crate::layers::ple::PleLayer;
#[cfg(feature = "cuda")]
use crate::layers::ple::{PleIdDims, PleWeights};
#[cfg(feature = "cuda")]
use crate::weight_map::dense;

/// Resident rows in the pinned arena. A prefill pins `tokens * ngram_heads`
/// rows at once (2048 x 16 = 32,768), so the default leaves headroom over the
/// largest batch this model currently fits; at 320 B/row it costs ~21 MB.
#[cfg(feature = "cuda")]
fn slots_from_env() -> usize {
    std::env::var("ATLAS_PLE_CACHE_SLOTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(65536)
}

/// Read a small I64 device tensor back to the host.
///
/// `layer_multipliers` and the two per-head tables are 3 and 16 elements —
/// they are uploaded like any other weight, and the id hash needs them on the
/// host. Reading them back beats adding a host-side path to `WeightStore` for
/// 280 bytes.
#[cfg(feature = "cuda")]
fn i64_host(store: &WeightStore, name: &str, gpu: &dyn GpuBackend) -> Result<Vec<u64>> {
    let t = store.get(name).with_context(|| format!("PLE: {name}"))?;
    let n = t.num_elements();
    let mut raw = vec![0u8; n * 8];
    gpu.copy_d2h(t.ptr, &mut raw)
        .with_context(|| format!("PLE: reading {name} back to host"))?;
    Ok(raw
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

/// Build the PLE layer for `layer_idx`, or `None` if this model has none.
#[cfg(feature = "cuda")]
pub(super) fn load(
    store: &WeightStore,
    config: &ModelConfig,
    layer_idx: usize,
    max_tokens: usize,
    gpu: &dyn GpuBackend,
) -> Result<Option<PleLayer>> {
    if config.ple_layer_ids.is_empty() {
        return Ok(None);
    }
    // `ple_layer_ids` is 1-INDEXED — the reference selects with
    // `ple_layer_ids.index(layer_idx + 1)` — so `[2]` means MODEL LAYER 1.
    if !config.ple_layer_ids.contains(&(layer_idx + 1)) {
        return Ok(None);
    }
    let lp = format!("{}.ple", config.layer_prefix(layer_idx));
    let h = config.hidden_size;
    let hc = config.hc_mult;
    let eos = config.eos_token_id;

    let dims = PleIdDims {
        ngram_size: config.emb_neighbor_num,
        heads_per_ngram: config.emb_split_num,
        multipliers: i64_host(store, &format!("{lp}.ple_embedding.layer_multipliers"), gpu)?,
        head_vocab_sizes: i64_host(
            store,
            &format!("{lp}.ple_embedding.ngram_heads_vocab_sizes"),
            gpu,
        )?,
        head_offsets: i64_host(
            store,
            &format!("{lp}.ple_embedding.ngram_heads_offsets"),
            gpu,
        )?,
        eos_token_id: eos,
    };
    dims.validate().context("PLE: checkpoint id geometry")?;
    let heads = dims.ngram_heads();

    // ── the segmented table ──
    let mut bases = Vec::new();
    let mut rows_per = 0usize;
    let mut head_dim = 0usize;
    let mut path = None;
    for i in 0.. {
        let name = format!("{lp}.ple_embedding.ngram_embedding.shard_{i}.weight");
        let Some(d) = store.deferred(&name) else {
            break;
        };
        anyhow::ensure!(
            d.shape.len() == 2,
            "PLE: shard {i} has shape {:?}, expected 2-D",
            d.shape
        );
        if i == 0 {
            rows_per = d.shape[0];
            head_dim = d.shape[1];
            path = Some(d.path.clone());
        } else {
            anyhow::ensure!(
                d.shape[0] == rows_per && d.shape[1] == head_dim,
                "PLE: shard {i} is {:?} but shard 0 is [{rows_per}, {head_dim}]. \
                 The segmented row cache maps a global id with one divide, which \
                 requires every shard to hold the same number of rows.",
                d.shape
            );
            anyhow::ensure!(
                Some(&d.path) == path.as_ref(),
                "PLE: shard {i} lives in a different file from shard 0; the row \
                 cache opens ONE backing file"
            );
        }
        bases.push(d.offset);
    }
    anyhow::ensure!(
        !bases.is_empty(),
        "PLE: no `{lp}.ple_embedding.ngram_embedding.shard_*` was deferred. Either \
         the checkpoint has none, or they were UPLOADED whole — which for this \
         table is 102 GB of BF16 and would not have fit."
    );
    let path = path.expect("checked non-empty");
    let slots = slots_from_env();
    let cache = spark_storage::NgramRowCache::open_segmented(
        &path,
        bases.clone(),
        rows_per as u64,
        None, // BF16 rows, no per-row scale file
        head_dim * 2,
        slots,
    )
    .context("PLE: n-gram row cache")?;

    let weights = PleWeights {
        key_proj: dense(store, &format!("{lp}.key_proj.weight"))?,
        value_proj: dense(store, &format!("{lp}.value_proj.weight"))?,
        norm_key: dense(store, &format!("{lp}.norm_key.weight"))?,
        norm_query: dense(store, &format!("{lp}.norm_query.weight"))?,
        norm_conv: dense(store, &format!("{lp}.norm_conv.weight"))?,
        conv1d: dense(store, &format!("{lp}.conv1d.weight"))?,
    };

    let dilation = config.emb_neighbor_num; // conv dilation IS ngram_size
    tracing::info!(
        "PLE at MODEL LAYER {layer_idx} (ple_layer_ids={:?}, 1-indexed): \
         {} shards x {rows_per} rows x {head_dim} dims = {} rows ({:.1} GB BF16) \
         served off NVMe with {slots} cached slots ({:.1} MB); {heads} heads, \
         conv k={} dilation={dilation} (state {} steps)",
        config.ple_layer_ids,
        bases.len(),
        bases.len() * rows_per,
        (bases.len() * rows_per * head_dim * 2) as f64 / 1e9,
        (slots * head_dim * 2) as f64 / 1e6,
        config.ple_conv_kernel_size,
        (config.ple_conv_kernel_size - 1) * dilation,
    );

    PleLayer::new(
        dims,
        head_dim,
        h,
        hc,
        config.ple_conv_kernel_size,
        dilation,
        config.rms_norm_eps as f32,
        weights,
        NgramTable::Cached(Box::new(cache)),
        max_tokens,
        gpu,
    )
    .map(Some)
    .context("PLE: layer construction")
}

/// Non-CUDA builds have no NVMe row cache — it serves rows out of a pinned,
/// GPU-addressable arena — so a PLE model cannot be served here. REFUSE
/// rather than return `None` (same rationale as `longcat/ngram.rs`): `None`
/// means "this model has no PLE", and quietly answering that for a model
/// that does have one silently drops the n-gram injection.
#[cfg(not(feature = "cuda"))]
pub(super) fn load(
    _store: &WeightStore,
    config: &ModelConfig,
    _layer_idx: usize,
    _max_tokens: usize,
    _gpu: &dyn GpuBackend,
) -> Result<Option<PleLayer>> {
    if config.ple_layer_ids.is_empty() {
        return Ok(None);
    }
    anyhow::bail!(
        "qwen4_exp PLE: this checkpoint has n-gram embeddings, but the row \
         cache that serves them needs the `cuda` feature; this build cannot \
         serve it"
    )
}
