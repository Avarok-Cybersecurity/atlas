// SPDX-License-Identifier: AGPL-3.0-only

//! Keep-packed GGUF Q4_K_M expert loading for Laguna-S-2.1.
//!
//! The GGUF loader (spark-runtime) uploads the routed-expert stacks as raw
//! Q4_K/Q6_K super-blocks and tags the per-expert views `PackedQ4K`/`PackedQ6K`.
//! This module turns those store tensors into the [`PackedExpertWeights`] the
//! MoE keep-packed prefill arm consumes — no dequant, the layer views alias the
//! store's block buffers.

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::weight_map::{
    DenseWeight, PackedExpertWeights, PackedQ4Weight, PackedQ6Weight, QuantWeight,
};

/// True when this layer's routed experts were stored keep-packed (Q4_K gate).
pub(super) fn experts_keep_packed(store: &WeightStore, mlp: &str) -> bool {
    store
        .get(&format!("{mlp}.experts.0.gate_proj.weight"))
        .map(|t| t.is_packed_q4k())
        .unwrap_or(false)
}

/// Build the per-expert packed views for one MoE layer: Q4_K gate/up, and a
/// per-layer Q4_K-or-Q6_K down (Q4_K_M mixes the down type PER LAYER — all
/// experts in a layer share one ggml type, the GGUF stores `ffn_down_exps` as
/// a single stacked tensor). Remote EP experts get NULL placeholders.
pub(super) fn load_packed_experts(
    store: &WeightStore,
    config: &ModelConfig,
    mlp: &str,
) -> Result<Vec<PackedExpertWeights>> {
    let mut packed = Vec::with_capacity(config.num_experts);
    for e in 0..config.num_experts {
        if !config.is_local_expert(e) {
            packed.push(PackedExpertWeights {
                gate: PackedQ4Weight::null_view(),
                up: PackedQ4Weight::null_view(),
                down: QuantWeight::PackedQ6(PackedQ6Weight::null_view()),
            });
            continue;
        }
        let ep = format!("{mlp}.experts.{e}");
        let down_prefix = format!("{ep}.down_proj");
        let down = if store.get(&format!("{down_prefix}.weight"))?.is_packed_q4k() {
            QuantWeight::PackedQ4(packed_q4_from_store(store, &down_prefix)?)
        } else {
            QuantWeight::PackedQ6(packed_q6_from_store(store, &down_prefix)?)
        };
        packed.push(PackedExpertWeights {
            gate: packed_q4_from_store(store, &format!("{ep}.gate_proj"))?,
            up: packed_q4_from_store(store, &format!("{ep}.up_proj"))?,
            down,
        });
    }
    Ok(packed)
}

/// Wrap a keep-packed Q4_K store tensor (`{prefix}.weight`, tagged
/// [`WeightDtype::PackedQ4K`] by the GGUF loader) into a [`PackedQ4Weight`]
/// layer view. The pointer aliases the store's block buffer (no copy).
fn packed_q4_from_store(store: &WeightStore, prefix: &str) -> Result<PackedQ4Weight> {
    let t = store.get(&format!("{prefix}.weight"))?;
    ensure!(t.is_packed_q4k(), "{prefix}.weight is not keep-packed Q4_K");
    ensure!(
        t.shape.len() == 2,
        "{prefix}.weight is not 2D ({:?})",
        t.shape
    );
    Ok(PackedQ4Weight {
        weight: t.ptr,
        n: t.shape[0] as u32,
        k: t.shape[1] as u32,
    })
}

/// Wrap a keep-packed Q6_K store tensor into a [`PackedQ6Weight`] layer view.
fn packed_q6_from_store(store: &WeightStore, prefix: &str) -> Result<PackedQ6Weight> {
    let t = store.get(&format!("{prefix}.weight"))?;
    ensure!(t.is_packed_q6k(), "{prefix}.weight is not keep-packed Q6_K");
    ensure!(
        t.shape.len() == 2,
        "{prefix}.weight is not 2D ({:?})",
        t.shape
    );
    Ok(PackedQ6Weight {
        weight: t.ptr,
        n: t.shape[0] as u32,
        k: t.shape[1] as u32,
    })
}

/// Materialise the sigmoid router's `e_score_correction_bias` as an F32 device
/// buffer for the keep-packed GGUF path.
///
/// The GGUF loader dequants this originally-F32 tensor to **BF16** on load (2
/// bytes/elem), but `moe_topk_sigmoid_batched` reads `bias` as `const float*`
/// (4 bytes/elem). Handing it the BF16 buffer makes the kernel over-read one
/// buffer-length past the end — a CUDA_ERROR_ILLEGAL_ADDRESS that surfaces at
/// whichever layer's trailing bytes land on an unmapped page (seen drifting
/// across layers with prompt length). Widen BF16→F32 here into a correctly
/// sized [num_experts] F32 device allocation. Safetensors already ships an F32
/// device pointer and keeps `dense()`, so this runs on the keep-packed path only.
pub(super) fn dense_bias_to_device(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let t = store
        .get(name)
        .with_context(|| format!("keep-packed bias {name} missing"))?;
    let n = t.num_elements();
    let src_bytes = t.byte_size();
    // `t.ptr` is a DEVICE buffer — the GGUF loader dequants this bias to BF16 on
    // GPU. Copy it down at its true (BF16) size, widen on the host, upload as F32.
    let mut host = vec![0u8; src_bytes];
    gpu.copy_d2h(t.ptr, &mut host)
        .with_context(|| format!("copy_d2h correction_bias {name} ({src_bytes}B)"))?;
    let f32_bytes: Vec<u8> = match t.dtype {
        WeightDtype::BF16 => host
            .chunks_exact(2)
            .flat_map(|c| {
                let bf = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((bf as u32) << 16).to_le_bytes()
            })
            .collect(),
        WeightDtype::FP32 => host,
        d => anyhow::bail!("unexpected correction_bias dtype {d:?} for {name}"),
    };
    let ptr = gpu
        .alloc(n * 4)
        .with_context(|| format!("allocate F32 device copy of {name}"))?;
    gpu.copy_h2d(&f32_bytes, ptr)?;
    Ok(DenseWeight { weight: ptr })
}
