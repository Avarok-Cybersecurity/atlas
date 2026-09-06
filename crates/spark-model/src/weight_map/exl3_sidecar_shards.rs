// SPDX-License-Identifier: AGPL-3.0-only

//! `register_exl3_sidecar_shards` — safetensors files an EXL3 export keeps
//! OUTSIDE `model.safetensors.index.json`, registered into the store before
//! `materialize_exl3` so their trellis linears materialize (or are kept
//! packed) exactly like the index-listed ones. Child of `exl3_materialize.rs`
//! (split for the 500-LoC cap).
//!
//! The motivating file is the 4.05bpw_h6_ng6 branch of
//! `turboderp/Qwen3.8-Flash-Next-exl3`: its whole ViT tower (987
//! `model.visual.*` tensors, 561 MB, K=6 trellis + F16 fused `attn.qkv`,
//! `pos_embed`, biases, norms) ships in `vision_k6.safetensors`, which the
//! index does not mention. Atlas's loaders read index-listed shards plus
//! `extra_weights.safetensors`, so that branch booted text-only. The other
//! branches keep the tower in the last index-listed shard, and ship the MTP
//! hyper-connection mixer in an un-indexed
//! `mtp_hyper_connection_mixer_patch.safetensors` instead.
//!
//! Discovery rule ([`select_exl3_sidecar_shards`]): every `*.safetensors`
//! file in the model directory that is NOT an index-listed shard, NOT one of
//! the main loader's own un-indexed patterns (`model.safetensors`,
//! `model.safetensors-*`, `consolidated*`), NOT `extra_weights.safetensors`
//! (the main loader already reads it) and NOT `ngram_embedding.safetensors`
//! (`register_exl3_ngram_sidecar` owns it — its trellis must be DEFERRED,
//! not uploaded). This mirrors ExLlamaV3's own loader, which globs
//! `*.safetensors` in the directory and never consults the index (that is
//! how the un-indexed mixer patch reaches it at all); the vendored reference
//! snapshot (`.research/exllamav3_ref/`) does not include that loader
//! module, so the rule is reconstructed from the export layout rather than
//! copied. Within a sidecar, a tensor whose name the store already holds is
//! SKIPPED (the index wins; logged) — and the caller's skip policy (EP
//! sharding, `skip_mtp`, `.input_scale`) applies exactly as it did to the
//! main shards, so a qwen4_exp serve never uploads the mixer patch.
//!
//! Dtypes go through the standard ingest (`load_safetensors_file`): F16 ->
//! BF16 on the host (the fused qkv, pos_embed and biases the ViT loader reads
//! as BF16 — on 4.05 they are F16 where the other branches ship BF16),
//! `.suh`/`.svh` keep exact f16 bits, I16 trellis -> `UInt16`, I32 `.mul1`
//! -> `Int32`. Nothing here is EXL3-specific except the reason the files
//! exist: a BF16 `vision_*.safetensors` would register just the same.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

/// Files another registration owns; never treated as a sidecar here.
const OWNED_ELSEWHERE: [&str; 2] = ["extra_weights.safetensors", "ngram_embedding.safetensors"];

/// The file-name shape of the EXL3 export's separate vision tower
/// (`vision.safetensors` / `vision_<tag>.safetensors`, e.g. `vision_k6`).
/// Used for the load log only — discovery does not depend on it.
pub fn is_exl3_vision_sidecar_name(file_name: &str) -> bool {
    file_name == "vision.safetensors"
        || (file_name.starts_with("vision_") && file_name.ends_with(".safetensors"))
}

/// Pure discovery rule: which of the directory's file names are sidecar
/// shards to register. `index_shards` is the set of shard file names the
/// index maps tensors to (empty for an un-indexed checkpoint). Sorted, so
/// registration order — and the "first wins" rule for a name two sidecars
/// both carry — is deterministic.
pub fn select_exl3_sidecar_shards<'a>(
    dir_listing: impl IntoIterator<Item = &'a str>,
    index_shards: &HashSet<String>,
) -> Vec<String> {
    let mut out: Vec<String> = dir_listing
        .into_iter()
        .filter(|n| n.ends_with(".safetensors"))
        .filter(|n| !index_shards.contains(*n))
        .filter(|n| !OWNED_ELSEWHERE.contains(n))
        // The main loader's un-indexed fallbacks (single file / bare shards).
        .filter(|n| {
            *n != "model.safetensors"
                && !n.starts_with("model.safetensors-")
                && !n.starts_with("consolidated")
        })
        .map(str::to_string)
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Shard file names the index maps tensors to (the VALUES of `weight_map`).
/// Empty when there is no index or it does not parse — an unreadable index
/// is the main loader's error to raise, not this pass's.
pub fn index_listed_shards(model_dir: &Path) -> HashSet<String> {
    let mut out = HashSet::new();
    for index_name in [
        "model.safetensors.index.json",
        "consolidated.safetensors.index.json",
    ] {
        let Ok(raw) = std::fs::read(model_dir.join(index_name)) else {
            continue;
        };
        let Ok(json) = serde_json::from_slice::<serde_json::Value>(&raw) else {
            continue;
        };
        if let Some(map) = json.get("weight_map").and_then(|m| m.as_object()) {
            out.extend(map.values().filter_map(|v| v.as_str()).map(str::to_string));
        }
    }
    out
}

/// What the registration did — for the load log and the tests.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Exl3SidecarStats {
    /// Sidecar files found by the discovery rule (registered or not).
    pub files: usize,
    /// Tensors inserted into the store.
    pub tensors: usize,
    /// The `model.visual.*` / `*.visual.*` subset of `tensors`.
    pub vision_tensors: usize,
    /// Tensors present in a sidecar but already in the store (index wins).
    pub already_present: usize,
    /// Tensors the caller's skip policy rejected (EP / skip_mtp / scales).
    pub skipped_by_policy: usize,
}

/// Register every sidecar shard of `model_dir` (see the module doc) into
/// `store`. `skip` is the caller's tensor-skip policy — pass the SAME rule
/// the main loader used (EP sharding, `skip_mtp`, `.input_scale`) so a
/// sidecar can never smuggle in a tensor the main shards were told to omit.
/// Call BEFORE `materialize_exl3`: the sidecar's trellis linears must be in
/// the store when the pass walks it, or they are neither materialized nor
/// kept and the ViT loader sees no tower. No-op (Ok, zero stats) when the
/// directory has no sidecar shards.
pub fn register_exl3_sidecar_shards(
    gpu: &dyn GpuBackend,
    store: &mut WeightStore,
    model_dir: &Path,
    oom_reserve_bytes: usize,
    skip: &dyn Fn(&str) -> bool,
) -> Result<Exl3SidecarStats> {
    let mut stats = Exl3SidecarStats::default();
    let listing: Vec<String> = std::fs::read_dir(model_dir)
        .with_context(|| format!("listing {}", model_dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .collect();
    let index_shards = index_listed_shards(model_dir);
    let sidecars = select_exl3_sidecar_shards(listing.iter().map(String::as_str), &index_shards);
    stats.files = sidecars.len();

    for file_name in &sidecars {
        let path = model_dir.join(file_name);
        // Both counters are per-file decisions taken inside the loader's
        // skip callback; `Cell`s keep the closure `Fn`.
        let present = std::cell::Cell::new(0usize);
        let policy = std::cell::Cell::new(0usize);
        let loaded = {
            let store_ref: &WeightStore = store;
            let skip_fn = |name: &str| {
                if skip(name) {
                    policy.set(policy.get() + 1);
                    return true;
                }
                if store_ref.contains(name) {
                    present.set(present.get() + 1);
                    return true;
                }
                false
            };
            spark_runtime::weights::load_safetensors_file(&path, gpu, oom_reserve_bytes, &skip_fn)
                .with_context(|| format!("EXL3 sidecar shard {file_name}"))?
        };
        let n = loaded.len();
        let vision = loaded.keys().filter(|k| k.contains(".visual.")).count();
        let bytes: usize = loaded
            .values()
            .map(|t| t.num_elements() * t.dtype.byte_size())
            .sum();
        for (name, t) in loaded {
            // `skip_fn` refused every name already present, so this never
            // displaces a tensor (nothing to free).
            store.insert(name, t);
        }
        stats.tensors += n;
        stats.vision_tensors += vision;
        stats.already_present += present.get();
        stats.skipped_by_policy += policy.get();
        tracing::info!(
            "EXL3 sidecar shard {file_name}{}: {n} tensors registered ({vision} vision, \
             {:.2} GB), {} already in the store (index wins), {} skipped by the loader policy",
            if is_exl3_vision_sidecar_name(file_name) {
                " (vision tower outside the index)"
            } else {
                ""
            },
            bytes as f64 / 1e9,
            present.get(),
            policy.get(),
        );
        if present.get() > 0 {
            tracing::warn!(
                "EXL3 sidecar shard {file_name}: {} tensors duplicate index-listed names and \
                 were ignored — the index-listed copy is served",
                present.get()
            );
        }
    }
    Ok(stats)
}

#[cfg(test)]
#[path = "exl3_sidecar_shards_tests.rs"]
mod tests;
