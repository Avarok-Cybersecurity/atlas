// SPDX-License-Identifier: AGPL-3.0-only

//! EXL3 PLE n-gram sidecar registration (`ngram_embedding.safetensors`,
//! `exl3_ngram_trellis` v1). Child module of `exl3_materialize.rs`, split out
//! for the ≤500 LoC cap; re-exported from there so the public path
//! (`weight_map::register_exl3_ngram_sidecar`) is unchanged.

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{DeferredTensor, WeightDtype, WeightStore, WeightTensor};

/// Probe `model_dir/ngram_embedding.safetensors` — the EXL3 export's
/// standalone PLE n-gram file (`exl3_ngram_trellis` v1) — and register its
/// tensors in the store under the names the PLE loader expects:
///
///  * the big `.trellis` `[rows, 1 + 160*K/16]` I16 tensor is DEFERRED
///    (the NVMe row cache faults raw rows from it; uploading it whole
///    would be ~39 GB),
///  * `head_bias` `[heads, 160]` F16 is uploaded verbatim (exact f16 bits
///    are gather-kernel inputs),
///  * `head_offsets` / `head_vocab_sizes` are uploaded RENAMED to the
///    `ngram_heads_offsets` / `ngram_heads_vocab_sizes` names the standard
///    checkpoints use; `layer_multipliers` keeps its name.
///
/// Returns Ok(true) when the sidecar was found and registered, Ok(false)
/// when absent or not the exl3_ngram_trellis format (both are fine — the
/// standard PLE shard walk applies). Call before quant detection.
pub fn register_exl3_ngram_sidecar(
    gpu: &dyn GpuBackend,
    store: &mut WeightStore,
    model_dir: &std::path::Path,
) -> Result<bool> {
    use std::io::Read;

    let path = model_dir.join("ngram_embedding.safetensors");
    if !path.is_file() {
        return Ok(false);
    }
    let mut f =
        std::fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?;
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8)?;
    let header_len = u64::from_le_bytes(len8) as usize;
    ensure!(
        header_len <= 16 * 1024 * 1024,
        "ngram_embedding.safetensors header is {header_len} bytes; refusing"
    );
    let mut hdr = vec![0u8; header_len];
    f.read_exact(&mut hdr)?;
    let json: serde_json::Value =
        serde_json::from_slice(&hdr).context("ngram_embedding.safetensors header is not JSON")?;
    let meta = json.get("__metadata__");
    let format = meta
        .and_then(|m| m.get("format"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if format != "exl3_ngram_trellis" {
        tracing::warn!(
            "ngram_embedding.safetensors present but format is {format:?} (expected \
             exl3_ngram_trellis) — leaving it to the standard PLE path"
        );
        return Ok(false);
    }
    let version = meta
        .and_then(|m| m.get("version"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    ensure!(
        version == "1",
        "exl3_ngram_trellis version {version:?} is not the understood version 1"
    );

    let data_start = 8 + header_len as u64;
    let obj = json.as_object().context("header not an object")?;
    let mut registered_trellis = false;
    for (name, info) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype_str = info["dtype"].as_str().unwrap_or("");
        let shape: Vec<usize> = info["shape"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();
        let offs = info["data_offsets"]
            .as_array()
            .and_then(|a| {
                let s = a.first()?.as_u64()?;
                let e = a.get(1)?.as_u64()?;
                Some((s, e))
            })
            .with_context(|| format!("ngram sidecar {name}: bad data_offsets"))?;

        if name.ends_with(".trellis") {
            ensure!(
                dtype_str == "I16" && shape.len() == 2,
                "ngram sidecar trellis: dtype {dtype_str} shape {shape:?}, expected I16 2-D"
            );
            store.defer(
                name.clone(),
                DeferredTensor {
                    path: path.clone(),
                    offset: data_start + offs.0,
                    shape,
                    dtype: WeightDtype::UInt16,
                },
            );
            registered_trellis = true;
            continue;
        }

        // Small tensors: read the bytes and upload. Leaf renames map the
        // sidecar's names onto the standard checkpoint's.
        let store_name = if let Some(base) = name.strip_suffix(".ngram_embedding.head_offsets") {
            format!("{base}.ngram_heads_offsets")
        } else if let Some(base) = name.strip_suffix(".ngram_embedding.head_vocab_sizes") {
            format!("{base}.ngram_heads_vocab_sizes")
        } else if let Some(base) = name.strip_suffix(".ngram_embedding.layer_multipliers") {
            format!("{base}.layer_multipliers")
        } else {
            // head_bias (and anything future) keeps its own name.
            name.clone()
        };
        let dtype = match dtype_str {
            "I64" => WeightDtype::Int64,
            "F16" => WeightDtype::F16, // exact bits are gather inputs
            "BF16" => WeightDtype::BF16,
            other => bail!("ngram sidecar {name}: unsupported dtype {other}"),
        };
        let nbytes = (offs.1 - offs.0) as usize;
        ensure!(
            nbytes <= 1 << 20,
            "ngram sidecar {name}: {nbytes} bytes is not small"
        );
        let mut buf = vec![0u8; nbytes];
        use std::io::{Seek, SeekFrom};
        f.seek(SeekFrom::Start(data_start + offs.0))?;
        f.read_exact(&mut buf)?;
        let ptr = gpu.alloc(nbytes.max(4))?;
        gpu.copy_h2d(&buf, ptr)?;
        store.insert(store_name, WeightTensor { ptr, shape, dtype });
    }
    ensure!(
        registered_trellis,
        "exl3_ngram_trellis sidecar had no .trellis tensor"
    );
    tracing::info!(
        "EXL3 ngram sidecar registered from {} (trellis deferred for the NVMe row cache)",
        path.display()
    );
    Ok(true)
}
