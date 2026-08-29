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

/// Read a one-element scale tensor back to the host as f32.
///
/// The same trick as `i64_host` and for the same reason: it is four bytes, and
/// the weight is already on the device. It accepts BF16 or FP32 because the
/// checkpoints differ on which they use for a scalar.
#[cfg(feature = "cuda")]
fn f32_scalar(store: &WeightStore, name: &str, gpu: &dyn GpuBackend) -> Result<f32> {
    let t = store.get(name).with_context(|| format!("PLE: {name}"))?;
    anyhow::ensure!(
        t.num_elements() == 1,
        "PLE: {name} has {} elements; a table-wide scale is one",
        t.num_elements()
    );
    match t.dtype {
        spark_runtime::weights::WeightDtype::FP32 => {
            let mut raw = [0u8; 4];
            gpu.copy_d2h(t.ptr, &mut raw)?;
            Ok(f32::from_le_bytes(raw))
        }
        spark_runtime::weights::WeightDtype::BF16 => {
            let mut raw = [0u8; 2];
            gpu.copy_d2h(t.ptr, &mut raw)?;
            // BF16 is the top 16 bits of an f32.
            Ok(f32::from_bits(u32::from(u16::from_le_bytes(raw)) << 16))
        }
        other => anyhow::bail!("PLE: {name} is {other:?}; expected a BF16 or FP32 scalar"),
    }
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
    // (path, byte offset) per shard. The path is carried PER SHARD because the
    // released NVFP4 checkpoint spreads these 128 shards across ten
    // `model-plefp8-*.safetensors` files; requiring one file refused the model
    // outright at shard 2.
    let mut shards: Vec<(std::path::PathBuf, u64)> = Vec::new();
    let mut rows_per = 0usize;
    let mut head_dim = 0usize;
    let mut dtype = None;
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
            dtype = Some(d.dtype);
        } else {
            anyhow::ensure!(
                Some(d.dtype) == dtype,
                "PLE: shard {i} is {:?} but shard 0 is {:?}; one row stride covers \
                 the whole table",
                d.dtype,
                dtype
            );
            // Equal row counts are still required: the cache maps a global id
            // to its shard with one divide, and that is only valid when every
            // shard holds the same number of rows. Differing FILES are fine —
            // each shard names its own.
            anyhow::ensure!(
                d.shape[0] == rows_per && d.shape[1] == head_dim,
                "PLE: shard {i} is {:?} but shard 0 is [{rows_per}, {head_dim}]. \
                 The segmented row cache maps a global id with one divide, which \
                 requires every shard to hold the same number of rows.",
                d.shape
            );
        }
        shards.push((d.path.clone(), d.offset));
    }
    anyhow::ensure!(
        !shards.is_empty(),
        "PLE: no `{lp}.ple_embedding.ngram_embedding.shard_*` was deferred. Either \
         the checkpoint has none, or they were UPLOADED whole — which for this \
         table is 102 GB of BF16 and would not have fit."
    );
    // The stride comes from the DTYPE, not from an assumption. LongCat ships
    // this table as BF16; the released Qwen3.8-Flash-Next NVFP4 checkpoint
    // ships it as FP8 E4M3 in files literally named `model-plefp8-*`. Hardcoding
    // `head_dim * 2` read two bytes per element out of a one-byte table: every
    // row came back as the wrong bytes, and once the doubled stride walked past
    // the end of the last file it failed outright.
    // The shard walk above stops at the FIRST gap, so a checkpoint that lost
    // `shard_5` in transfer loads as a 5-shard table and serves — until a token
    // hashes past the rows that exist and `resolve` bails mid-request, at an
    // arbitrary later time, on a machine nobody is watching. The shipped id
    // tables say exactly how many rows the hash can produce, so compare them
    // here, where the answer is a refusal instead of a 500 an hour from now.
    let rows_total = rows_per as u64 * shards.len() as u64;
    let highest_id = dims
        .head_offsets
        .iter()
        .zip(dims.head_vocab_sizes.iter())
        .map(|(off, vocab)| off + vocab)
        .max()
        .unwrap_or(0);
    anyhow::ensure!(
        highest_id <= rows_total,
        "PLE: the checkpoint's id tables reach row {highest_id}, but only \
         {rows_total} rows are present ({} shards x {rows_per}). Either a \
         `shard_*` tensor is missing from this checkpoint — the walk stops at the \
         first gap — or the id tables belong to a different conversion.",
        shards.len()
    );

    let dtype = dtype.context("PLE: no shard dtype")?;
    let elem = match dtype {
        spark_runtime::weights::WeightDtype::BF16 => 2,
        spark_runtime::weights::WeightDtype::FP8E4M3 => 1,
        other => anyhow::bail!(
            "PLE: n-gram table is {other:?}; the row cache reads raw rows and the \
             gather kernel has a path for BF16 and FP8 E4M3 only"
        ),
    };
    let slots = slots_from_env();
    // No longer `mut`: the only mutation was the constant scale, which the
    // gather cannot use and which is now a refusal (below).
    let cache = spark_storage::NgramRowCache::open_segmented(
        &shards,
        rows_per as u64,
        None, // scales are not per-row here; see the constant below
        head_dim * elem,
        slots,
    )
    .context("PLE: n-gram row cache")?;

    // An FP8 table needs a scale for the gather to dequantize with. This
    // checkpoint carries ONE for the whole table
    // (`ngram_embedding.weight_scale`, BF16, shape [1]) rather than one per
    // row, so every slot gets the same value and nothing is faulted for it.
    if elem == 1 {
        let name = format!("{lp}.ple_embedding.ngram_embedding.weight_scale");
        let scale = f32_scalar(store, &name, gpu)
            .with_context(|| format!("PLE: FP8 table needs {name}"))?;
        anyhow::ensure!(
            scale.is_finite() && scale > 0.0,
            "PLE: n-gram weight_scale is {scale}, which cannot dequantize anything"
        );
        // ...and the gather cannot use it, so REFUSE rather than answer wrongly.
        //
        // `PleLayer` binds `embed_from_argmax::batched_embed`, whose table
        // pointer is `__nv_bfloat16*`. Pointed at an arena packed one byte per
        // element it strides twice as far as a row, so every row but the first
        // is read from the middle of another row, those bytes are reinterpreted
        // as BF16, this scale is multiplied in nowhere, and slots past the
        // halfway mark read beyond the allocation. The model loads, serves, and
        // is fluently wrong -- the one failure no operator can attribute to the
        // checkpoint they chose, which is why it is a refusal and not a warning.
        //
        // `batched_embed_fp8` already exists in that same .cu file; wiring it to
        // `PleLayer` and checking its numerics on a GPU is the fix. Until that is
        // measured, an honest error at load beats a model that answers.
        anyhow::bail!(
            "PLE: this checkpoint stores the n-gram table in FP8 (1 byte/element, \
             scale {scale:.6e}), but the gather kernel wired to PleLayer reads BF16 \
             (2 bytes/element) and applies no scale -- loading it would produce \
             silently wrong output rather than an error. Use a BF16 conversion of \
             this model, or wire `batched_embed_fp8` into PleLayer first."
        );
    }

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
        shards.len(),
        shards.len() * rows_per,
        (shards.len() * rows_per * head_dim * 2) as f64 / 1e9,
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
