// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side per-layer decode used by `K3BoundLayer` as an explicit
//! **CPU fallback GPU wrapper** (copy-out / CPU mixer+MLP+AttnRes / copy-in).
//! This is not CUDA KDA. GPU `.cu` is a later slice.

use super::cache::{HybridCache, LayerCache};
use super::cpu_forward::{AttnResStream, K3LayerCtx, forward_one_layer, forward_token};
use super::cpu_weights::{Ablation, K3CpuModel};
use super::greedy::greedy_decode;
use super::ops::argmax;

fn wrap_token(model: &K3CpuModel, token: u32, pos: usize, cache: &mut HybridCache) -> Vec<f32> {
    // Identity "GPU" copy: hidden is the embedding / previous mix. The spark
    // wrapper copies BF16 D2H, runs this, copies H2D.
    let embed = super::ops::embed_token(&model.embed, token, model.graph.hidden, model.vocab);
    let mut stream = AttnResStream::new(model.graph.hidden, model.graph.attn_res_block_size);
    stream.partial.clone_from(&embed);
    let ctx = K3LayerCtx::from_model(model);
    for layer in &model.layers {
        forward_one_layer(
            &ctx,
            layer,
            pos,
            &mut cache.layers[layer.spec.index],
            &mut stream,
            Ablation::default(),
        );
    }
    stream.mix(
        &model.output_res_proj,
        &model.output_res_norm,
        model.eps,
        1.0,
    )
}

#[test]
fn cpu_fallback_gpu_wrapper_matches_forward_token() {
    let model = K3CpuModel::synthetic_tiny();
    let mut a = HybridCache::from_graph(&model.graph, &model.kda);
    let mut b = HybridCache::from_graph(&model.graph, &model.kda);
    let want = forward_token(&model, 3, 0, &mut a, Ablation::default());
    let mix = wrap_token(&model, 3, 0, &mut b);
    let got = super::attnres::rms_norm(&mix, &model.final_norm, model.eps);
    assert_eq!(
        got, want,
        "CPU fallback GPU wrapper (copy-out identity) must match forward_token"
    );
}

#[test]
fn cpu_fallback_mix0_changes_greedy_tokens() {
    let model = K3CpuModel::synthetic_tiny();
    let prompt = [1u32, 2, 3];
    let mix1 = greedy_decode(&model, &prompt, 8, Ablation::default());
    let mix0 = greedy_decode(
        &model,
        &prompt,
        8,
        Ablation {
            attnres_mix: 0.0,
            ..Ablation::default()
        },
    );
    assert_ne!(
        mix0, mix1,
        "RST known-bad: mix=0 must change tokens vs mix=1 (CPU fallback GPU wrapper)"
    );
}

#[test]
fn cpu_fallback_prefix_cache_bytes_match_c3() {
    let model = K3CpuModel::synthetic_tiny();
    let prefix = [1u32, 2, 3, 4];
    let mut cold = HybridCache::from_graph(&model.graph, &model.kda);
    let mut h = Vec::new();
    for (pos, &tok) in prefix.iter().enumerate() {
        h = forward_token(&model, tok, pos, &mut cold, Ablation::default());
    }
    let blobs: Vec<Vec<u8>> = cold.layers.iter().map(LayerCache::to_bytes).collect();
    let mut hit = HybridCache::from_graph(&model.graph, &model.kda);
    for (slot, blob) in hit.layers.iter_mut().zip(&blobs) {
        *slot = LayerCache::from_bytes(blob).unwrap();
    }
    let next = argmax(&super::cpu_forward::logits(&model, &h));
    let h_hit = forward_token(&model, next, prefix.len(), &mut hit, Ablation::default());
    let h_cold = forward_token(&model, next, prefix.len(), &mut cold, Ablation::default());
    assert_eq!(
        h_hit, h_cold,
        "C3: restored LayerCache bytes must match in-place prefix cache"
    );
}
