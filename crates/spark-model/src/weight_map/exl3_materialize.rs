// SPDX-License-Identifier: AGPL-3.0-only

//! EXL3 checkpoint materialization: rewrite trellis-quantized linears into
//! the tensor layouts the existing model loaders already consume, so every
//! family's load path (qwen4_exp included) works on an EXL3 checkpoint with
//! ZERO per-arm changes.
//!
//! Runs ONCE, right after the weight store is loaded and before quant-format
//! detection / model construction. Idempotent: after the pass no `.trellis`
//! tensor remains, so a second call is a no-op.
//!
//! Per EXL3 linear (`{p}.trellis` + `.suh` + `.svh` [+ `.mul1`]), routed by
//! prefix:
//!
//!  * **MoE experts + shared experts** (`.mlp.experts.N.` /
//!    `.mlp.shared_expert.`) — reconstruct to BF16 (transient) and
//!    immediately runtime-quantize to the ModelOpt-style NVFP4 triplet
//!    (`.weight` U8 `[n, k/2]` + `.weight_scale` FP8 `[n, k/16]` +
//!    `.weight_scale_2` F32 scalar). `quantized_any` then takes its
//!    `Standard` arm verbatim. Quantizing INSIDE the pass is load-bearing:
//!    the routed experts are ~90% of the model's parameters, and holding
//!    them all as BF16 simultaneously (~4x the packed bytes) cannot fit —
//!    the transient here is one tensor at a time.
//!  * **Everything else** (attention, GDN, lm_head, MTP) — materialize as a
//!    plain BF16 `.weight` `[out, in]`. The Standard-variant arms read
//!    exactly that (`dense_auto` + runtime NVFP4 quantization per arm), and
//!    these tensors are small enough (~6 GB total on Qwen3.8-Flash-Next)
//!    that BF16 residency until construction is fine.
//!
//! The source trellis/suh/svh/mul1 tensors are freed as each linear lands,
//! so peak memory ≈ the packed checkpoint + one transient BF16 tensor.
//!
//! NOT covered here (documented gaps, each needs its own decode path):
//!  * `ngram_embedding.safetensors` (`exl3_ngram_trellis` row format) — the
//!    PLE n-gram tables. A qwen4_exp load will fail at the PLE loader's
//!    existing "no shard_* was deferred" check; this pass logs the reason
//!    up front.
//!  * `vision_k6.safetensors` — the EXL3-quantized vision encoder ships
//!    outside the weight index and is not loaded.
//!
//! Double-quantization note: EXL3(K bits) -> BF16 -> NVFP4 costs quality vs
//! a native-NVFP4 calibration AND vs native EXL3 serving. This pass is the
//! LOADING path (checkpoint compatibility); the native fused trellis-GEMM
//! (`.research/EXL3_DECODE_FINDINGS.md`) is the fidelity/memory path.

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::exl3::{Exl3Weight, store_has_exl3};
use spark_runtime::weights::{DeferredTensor, WeightDtype, WeightStore, WeightTensor};

use super::{DenseWeight, quantize_to_nvfp4};

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
    let mut f = std::fs::File::open(&path)
        .with_context(|| format!("opening {}", path.display()))?;
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8)?;
    let header_len = u64::from_le_bytes(len8) as usize;
    ensure!(
        header_len <= 16 * 1024 * 1024,
        "ngram_embedding.safetensors header is {header_len} bytes; refusing"
    );
    let mut hdr = vec![0u8; header_len];
    f.read_exact(&mut hdr)?;
    let json: serde_json::Value = serde_json::from_slice(&hdr)
        .context("ngram_embedding.safetensors header is not JSON")?;
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
            .map(|a| a.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
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
        ensure!(nbytes <= 1 << 20, "ngram sidecar {name}: {nbytes} bytes is not small");
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

/// What the pass did — for the load log.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Exl3MaterializeStats {
    /// Linears rewritten as NVFP4 triplets (experts).
    pub quantized: usize,
    /// Linears rewritten as dense BF16 `.weight`.
    pub bf16: usize,
}

/// Expert-family prefixes get the NVFP4 triplet; everything else BF16.
fn wants_nvfp4_triplet(prefix: &str) -> bool {
    prefix.contains(".mlp.experts.") || prefix.contains(".mlp.shared_expert.")
}

/// Rewrite every EXL3 linear in `store` into loader-consumable tensors.
/// No-op (Ok, zero stats) when the store has no EXL3 tensors.
pub fn materialize_exl3(
    gpu: &dyn GpuBackend,
    store: &mut WeightStore,
) -> Result<Exl3MaterializeStats> {
    let mut stats = Exl3MaterializeStats::default();
    if !store_has_exl3(store) {
        return Ok(stats);
    }

    let prefixes: Vec<String> = store
        .names()
        .filter_map(|n| n.strip_suffix(".trellis").map(str::to_string))
        .collect();
    tracing::info!(
        "EXL3 checkpoint: materializing {} trellis-quantized linears \
         (experts -> NVFP4 triplet, rest -> BF16 dense)",
        prefixes.len()
    );
    // The PLE n-gram tables ship as a separate `exl3_ngram_trellis`-format
    // file that never enters the store; a model that requires them
    // (qwen4_exp) will fail at the PLE loader with its missing-shard error.
    // Say why up front rather than letting that error stand alone.
    if !store
        .names()
        .any(|n| n.contains("ple_embedding.ngram_embedding.shard_"))
    {
        tracing::info!(
            "EXL3 checkpoint has no in-store PLE n-gram shards — the exl3 export keeps \
             them in ngram_embedding.safetensors; `register_exl3_ngram_sidecar` (called \
             right after this pass on the serve path) registers that file's trellis \
             tensors for the NVMe row cache. Models without PLE are unaffected either way."
        );
    }

    let absmax_k = gpu
        .kernel("quantize_nvfp4", "nvfp4_global_absmax")
        .context("EXL3 materialization needs the quantize_nvfp4 kernels")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();

    for p in &prefixes {
        let w = Exl3Weight::from_store(gpu, store, p)
            .with_context(|| format!("EXL3 materialization: resolving {p}"))?;
        let (n, k) = (w.out_dim, w.in_dim);
        let bf16 = w
            .to_bf16(gpu)
            .with_context(|| format!("EXL3 materialization: reconstructing {p}"))?;

        if wants_nvfp4_triplet(p) {
            let dense = DenseWeight { weight: bf16 };
            let q = quantize_to_nvfp4(&dense, n, k, gpu, absmax_k, quantize_k, stream)
                .with_context(|| format!("EXL3 materialization: quantizing {p}"))?;
            gpu.free(bf16)?;
            let scale2 = gpu.alloc(4)?;
            gpu.copy_h2d(&q.weight_scale_2.to_le_bytes(), scale2)?;
            store.insert(
                format!("{p}.weight"),
                WeightTensor {
                    ptr: q.weight,
                    shape: vec![n, k / 2],
                    dtype: WeightDtype::UInt8,
                },
            );
            store.insert(
                format!("{p}.weight_scale"),
                WeightTensor {
                    ptr: q.weight_scale,
                    shape: vec![n, k / 16],
                    dtype: WeightDtype::FP8E4M3,
                },
            );
            store.insert(
                format!("{p}.weight_scale_2"),
                WeightTensor {
                    ptr: scale2,
                    shape: vec![1],
                    dtype: WeightDtype::FP32,
                },
            );
            stats.quantized += 1;
        } else {
            store.insert(
                format!("{p}.weight"),
                WeightTensor {
                    ptr: bf16,
                    shape: vec![n, k],
                    dtype: WeightDtype::BF16,
                },
            );
            stats.bf16 += 1;
        }

        for suffix in ["trellis", "suh", "svh", "mul1"] {
            if let Some(t) = store.remove(&format!("{p}.{suffix}")) {
                gpu.free(t.ptr)?;
            }
        }
    }

    tracing::info!(
        "EXL3 materialization done: {} experts -> NVFP4 triplets, {} linears -> BF16 dense",
        stats.quantized,
        stats.bf16
    );
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::*;

    fn t(gpu: &MockGpuBackend, shape: Vec<usize>, dtype: WeightDtype) -> WeightTensor {
        let bytes: usize = shape.iter().product::<usize>() * dtype.byte_size().max(1);
        WeightTensor {
            ptr: gpu.alloc(bytes.max(4)).unwrap(),
            shape,
            dtype,
        }
    }

    fn exl3_linear(gpu: &MockGpuBackend, m: &mut HashMap<String, WeightTensor>, p: &str, k: u32) {
        // [2560 -> 640] geometry, K bits.
        m.insert(
            format!("{p}.trellis"),
            t(gpu, vec![160, 40, 16 * k as usize], WeightDtype::UInt16),
        );
        m.insert(format!("{p}.suh"), t(gpu, vec![2560], WeightDtype::F16));
        m.insert(format!("{p}.svh"), t(gpu, vec![640], WeightDtype::F16));
        m.insert(format!("{p}.mul1"), t(gpu, vec![], WeightDtype::Int32));
    }

    #[test]
    fn no_exl3_is_noop() {
        let gpu = MockGpuBackend::new();
        let mut m = HashMap::new();
        m.insert("a.weight".to_string(), t(&gpu, vec![8, 8], WeightDtype::BF16));
        let mut store = WeightStore::from_map(m);
        let stats = materialize_exl3(&gpu, &mut store).unwrap();
        assert_eq!(stats, Exl3MaterializeStats::default());
        assert!(store.contains("a.weight"));
    }

    #[test]
    fn routes_experts_to_triplet_and_attention_to_bf16() {
        let gpu = MockGpuBackend::new();
        let mut m = HashMap::new();
        exl3_linear(
            &gpu,
            &mut m,
            "model.layers.0.mlp.experts.3.gate_proj",
            4,
        );
        exl3_linear(
            &gpu,
            &mut m,
            "model.layers.0.mlp.shared_expert.up_proj",
            6,
        );
        exl3_linear(&gpu, &mut m, "model.layers.0.linear_attn.in_proj_qkv", 6);
        // Bystander that must survive untouched.
        m.insert(
            "model.layers.0.norm.weight".to_string(),
            t(&gpu, vec![2560], WeightDtype::BF16),
        );
        let mut store = WeightStore::from_map(m);

        let stats = materialize_exl3(&gpu, &mut store).unwrap();
        assert_eq!(stats.quantized, 2);
        assert_eq!(stats.bf16, 1);

        // Expert: ModelOpt-style NVFP4 triplet, [n=640, k=2560].
        let ep = "model.layers.0.mlp.experts.3.gate_proj";
        let w = store.get(&format!("{ep}.weight")).unwrap();
        assert_eq!(w.dtype, WeightDtype::UInt8);
        assert_eq!(w.shape, vec![640, 1280]); // [n, k/2]
        let s = store.get(&format!("{ep}.weight_scale")).unwrap();
        assert_eq!(s.dtype, WeightDtype::FP8E4M3);
        assert_eq!(s.shape, vec![640, 160]); // [n, k/16]
        let s2 = store.get(&format!("{ep}.weight_scale_2")).unwrap();
        assert_eq!(s2.dtype, WeightDtype::FP32);

        // Attention: dense BF16 [out, in].
        let ap = "model.layers.0.linear_attn.in_proj_qkv";
        let w = store.get(&format!("{ap}.weight")).unwrap();
        assert_eq!(w.dtype, WeightDtype::BF16);
        assert_eq!(w.shape, vec![640, 2560]);

        // Every EXL3 source tensor is gone; the bystander survived.
        for p in [ep, ap, "model.layers.0.mlp.shared_expert.up_proj"] {
            for sfx in ["trellis", "suh", "svh", "mul1"] {
                assert!(!store.contains(&format!("{p}.{sfx}")), "{p}.{sfx} remains");
            }
        }
        assert!(store.contains("model.layers.0.norm.weight"));

        // Idempotent: second call is a no-op.
        let again = materialize_exl3(&gpu, &mut store).unwrap();
        assert_eq!(again, Exl3MaterializeStats::default());
    }
}
