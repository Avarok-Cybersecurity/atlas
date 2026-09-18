// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! DeepSeek-V4.1 Flash: the loader that assembles [`DeepSeekV41Layer`](crate::layers::deepseek_v41_layer::DeepSeekV41Layer)s from
//! the seven-shard Q2_K GGUF.
//!
//! Resident (dequantized to bf16 once by the GGUF path, ~2.9 GiB): attention,
//! the compressor and indexer projections, the routers, the shared experts,
//! the engram projections, the norms, the embedding and the Q6_K head. Left
//! on disk and recorded as deferred by the GGUF loader: the 40 x 3 routed
//! expert stacks and the two engram tables, served by `expert_stream`.
//!
//! What the GGUF narrows to bf16 but the kernels want in f32 (norm weights,
//! the attention sinks, the ratio-2 compressor projections, the mHC mixes)
//! is widened here at load; what the kernels want as a product (`engram q * k`)
//! is formed here.
//!
//! The source-layer sets (`kv_source_layer_ids`, `index_source_layer_ids`)
//! are derived from which layers ship compressor / indexer tensors, which the
//! S1 real-file oracle proved equal to the published config. The candidate
//! settings are not in the GGUF metadata; the published `text_config` values
//! are the defaults (candidate source layer 20, 2048 blocks of 8).

mod load_layers;

use anyhow::{Context, Result, ensure};
use avarok_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layer::TransformerLayer;
use crate::layers::ops::ResidentMat;
use crate::layers::qwen3_attention::HcSiteWeights;
use crate::weight_loader::ModelWeightLoader;
use crate::weight_map::{DenseWeight, MtpWeights, dense_auto};

pub struct DeepSeekV41WeightLoader;

pub(super) const DEFAULT_CANDIDATE_SOURCE: usize = 20;
pub(super) const DEFAULT_CANDIDATE_TOPK_BLOCKS: usize = 2048;
pub(super) const DEFAULT_CANDIDATE_BLOCK: usize = 8;

pub(super) fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

pub(super) fn bf16_ptr(store: &WeightStore, name: &str) -> Result<DevicePtr> {
    let t = store.get(name)?;
    ensure!(
        t.dtype == WeightDtype::BF16,
        "{name}: expected bf16, got {:?}",
        t.dtype
    );
    Ok(t.ptr)
}

/// A resident projection as the GGUF path left it: bf16 (expanded on load)
/// or raw Q2_K / Q3_K blocks (`WeightDtype::Q2K` / `Q3K`, the K-quant path).
pub(super) fn resident_mat(store: &WeightStore, name: &str) -> Result<ResidentMat> {
    let t = store.get(name)?;
    match t.dtype {
        WeightDtype::BF16 => Ok(ResidentMat::Bf16(t.ptr)),
        WeightDtype::Q2K => Ok(ResidentMat::Q2K(t.ptr)),
        WeightDtype::Q3K => Ok(ResidentMat::Q3K(t.ptr)),
        d => anyhow::bail!("{name}: expected bf16, Q2_K or Q3_K, got {d:?}"),
    }
}

pub(super) fn download_f32(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    name: &str,
) -> Result<Vec<f32>> {
    let t = store.get(name)?;
    let n = t.num_elements();
    match t.dtype {
        WeightDtype::BF16 => {
            let mut b = vec![0u8; n * 2];
            gpu.copy_d2h(t.ptr, &mut b)?;
            Ok(b.chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect())
        }
        WeightDtype::FP32 => {
            let mut b = vec![0u8; n * 4];
            gpu.copy_d2h(t.ptr, &mut b)?;
            Ok(b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        WeightDtype::Q2K | WeightDtype::Q3K => {
            use spark_runtime::weights::dequant_cpu::{GgmlType, dequant_to_f32};
            let gt = if t.dtype == WeightDtype::Q2K {
                GgmlType::Q2K
            } else {
                GgmlType::Q3K
            };
            let mut b = vec![0u8; t.byte_size()];
            gpu.copy_d2h(t.ptr, &mut b)?;
            let mut out = vec![0f32; n];
            dequant_to_f32(gt, &b, n, &mut out)
                .with_context(|| format!("{name}: CPU dequant of the resident {gt:?} blocks"))?;
            Ok(out)
        }
        other => anyhow::bail!("{name}: cannot widen {other:?} to f32"),
    }
}

fn upload_f32(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(bytes.len().max(4))?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}

/// A tensor the kernels read as f32: the store's bf16 copy widened on the device.
pub(super) fn f32_ptr(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    name: &str,
    expect: usize,
) -> Result<DevicePtr> {
    let v = download_f32(gpu, store, name)?;
    ensure!(
        v.len() == expect,
        "{name}: {} elements, expected {expect}",
        v.len()
    );
    upload_f32(gpu, &v)
}

pub(super) fn hc_site(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    lp: &str,
    site: &str,
    c: &ModelConfig,
) -> Result<HcSiteWeights> {
    let hc = c.hc_mult;
    let mix_hc = (2 + hc) * hc;
    Ok(HcSiteWeights {
        hc_fn: f32_ptr(
            gpu,
            store,
            &format!("{lp}.hc_{site}_fn"),
            mix_hc * hc * c.hidden_size,
        )?,
        hc_base: f32_ptr(gpu, store, &format!("{lp}.hc_{site}_base"), mix_hc)?,
        hc_scale: f32_ptr(gpu, store, &format!("{lp}.hc_{site}_scale"), 3)?,
        lowrank: None,
    })
}

impl ModelWeightLoader for DeepSeekV41WeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[spark_runtime::kv_cache::KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        load_layers::load_layers(store, config, gpu, layer_kv_dtypes)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense_auto(store, "model.embed_tokens.weight", gpu)
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense_auto(store, "model.norm.weight", gpu)
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense_auto(store, "lm_head.weight", gpu)
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None)
    }
}

/// Layer 0 of the real model against the CPU reference, stage by stage: the
/// same five embedding rows through hc_mixes / collapse / norm / attention /
/// hc_post / MoE on the GPU and on the CPU (real weights downloaded from the
/// store, the six routed experts dequantised by the loader's decoders).
/// Ratio-0 layer: no engram, no shared runtime, so every mismatch is local.
#[cfg(all(test, feature = "cuda"))]
mod real_file_tests;
