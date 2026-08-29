// SPDX-License-Identifier: AGPL-3.0-only

//! Builds the LongCat n-gram embedding from a loaded `WeightStore`.
//!
//! The 12 lookup tables are the model's largest tensors by far — 62.8 GB of
//! the checkpoint's 138 GB — and the weight loaders DEFER them: they are
//! never uploaded, only recorded as `(shard path, absolute offset, shape)`.
//! This module turns those records into `NgramTable::Cached` row caches that
//! read rows straight out of the safetensors shards on demand.
//!
//! Why cached rather than resident: a row is 512 bytes (256 BF16), and a
//! token touches exactly 12 of them. The tables are simultaneously the
//! biggest tensors in the model and the least bandwidth-hungry, so keeping a
//! bounded slot window costs a few hundred MB instead of 62.8 GB (BF16) or
//! 31.4 GB (FP8) — and that memory goes to KV instead.

#[cfg(feature = "cuda")]
use anyhow::Context;
use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

#[cfg(feature = "cuda")]
use crate::layers::ngram_embed::NgramTable;
use crate::layers::ngram_embed::{NgramDims, NgramEmbedding};
#[cfg(feature = "cuda")]
use crate::weight_map::dense;

/// Resident rows per table. 65536 slots x 512 B = 33.5 MB per table, so all
/// 12 cost ~402 MB — against 62.8 GB for the same tables held BF16-resident.
#[cfg(feature = "cuda")]
const DEFAULT_SLOTS: usize = 65536;

#[cfg(feature = "cuda")]
fn slots_from_env() -> usize {
    std::env::var("ATLAS_NGRAM_CACHE_SLOTS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_SLOTS)
}

/// Build the n-gram embedding, or `None` when this checkpoint has no n-gram
/// trio in its config (every non-LongCat model).
#[cfg(feature = "cuda")]
pub(super) fn build(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    max_tokens: usize,
) -> Result<Option<NgramEmbedding>> {
    // Bisection lever: ATLAS_NGRAM_DISABLE=1 serves the plain `embed_tokens`
    // gather instead of the fused embedding. Output is WRONG (12/13 of the
    // signal is missing) but deterministic, which is exactly what is needed to
    // ask "is this concurrency bug mine, or does it predate the n-gram path?"
    if std::env::var("ATLAS_NGRAM_DISABLE").is_ok() {
        tracing::warn!(
            "ATLAS_NGRAM_DISABLE set — n-gram embedding NOT installed;              output will be incorrect. Diagnostic use only."
        );
        return Ok(None);
    }
    let Some(dims) = NgramDims::from_config(config) else {
        return Ok(None);
    };
    let n_tables = dims.num_tables();
    let slots = slots_from_env();

    let word = dense(store, "model.embed_tokens.weight").context("ngram: base embedding")?;

    let mut tables = Vec::with_capacity(n_tables);
    let mut projs = Vec::with_capacity(n_tables);
    for i in 0..n_tables {
        let tname = format!("model.ngram_embeddings.embedders.{i}.weight");
        let d = store.deferred(&tname).ok_or_else(|| {
            anyhow::anyhow!(
                "ngram: table {tname} was not deferred by the loader — it is either \
                 missing from the checkpoint or was uploaded whole (62.8 GB of BF16)"
            )
        })?;
        anyhow::ensure!(
            d.shape.len() == 2,
            "ngram: table {tname} has shape {:?}, expected 2-D",
            d.shape
        );
        let rows_total = d.shape[0] as u64;
        let expected = dims.table_rows(i);
        anyhow::ensure!(
            rows_total == expected,
            "ngram: table {i} has {rows_total} rows but the config's hash geometry \
             wants {expected} (ratio*vocab + 2*i + 1) — the id space and the table \
             would disagree and every lookup would be wrong"
        );
        anyhow::ensure!(
            d.shape[1] == dims.table_dim(),
            "ngram: table {i} dim {} != hidden/num_tables {}",
            d.shape[1],
            dims.table_dim()
        );
        // BF16 rows, no per-row scale file.
        let row_stride = d.shape[1] * 2;
        let cache = spark_storage::NgramRowCache::open_at(
            &d.path, d.offset, None, rows_total, row_stride, slots,
        )
        .with_context(|| format!("ngram: row cache for table {i}"))?;
        tables.push(NgramTable::Cached(Box::new(cache)));

        let pname = format!("model.ngram_embeddings.post_projs.{i}.weight");
        projs.push(dense(store, &pname).with_context(|| format!("ngram: proj {i}"))?);
    }

    tracing::info!(
        "LongCat n-gram embedding: {n_tables} tables x {} rows x {} dims, NVMe row cache \
         ({slots} slots = {:.0} MB total, vs {:.1} GB BF16-resident); fusion = \
         (base + sum proj_i(table_i)) / {}",
        dims.table_rows(0),
        dims.table_dim(),
        (n_tables * slots * dims.table_dim() * 2) as f64 / 1e6,
        (n_tables as f64 * dims.table_rows(0) as f64 * dims.table_dim() as f64 * 2.0) / 1e9,
        n_tables + 1,
    );

    Ok(Some(NgramEmbedding::new(
        dims, word, tables, projs, max_tokens, gpu,
    )?))
}

/// Non-CUDA builds have no row cache — it serves rows out of a pinned,
/// GPU-addressable arena — so an n-gram model cannot be served at all here.
/// REFUSE rather than return `None`: `None` means "this architecture has no
/// n-gram embedding", and quietly answering that for a model that does have
/// one is how you end up serving a plain gather and wondering why the output
/// is fluent nonsense.
#[cfg(not(feature = "cuda"))]
pub(super) fn build(
    _store: &WeightStore,
    config: &ModelConfig,
    _gpu: &dyn GpuBackend,
    _max_tokens: usize,
) -> Result<Option<NgramEmbedding>> {
    if NgramDims::from_config(config).is_none() {
        return Ok(None);
    }
    anyhow::bail!(
        "ngram: this checkpoint has n-gram embeddings, but the row cache that \
         serves them needs the `cuda` feature; this build cannot serve it"
    )
}
