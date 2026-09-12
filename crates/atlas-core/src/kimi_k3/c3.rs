// SPDX-License-Identifier: AGPL-3.0-only

//! C3: prefix-cache hit equals no-cache greedy tokens.

use super::cache::{HybridCache, LayerCache};
use super::cpu_forward::{forward_token, logits};
use super::cpu_weights::{Ablation, K3CpuModel};
use super::greedy::greedy_decode;
use super::ops::argmax;

const NEW_TOKENS: usize = 8;

fn prefill(model: &K3CpuModel, tokens: &[u32], cache: &mut HybridCache) -> Vec<f32> {
    let ablation = Ablation::default();
    let mut h = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        h = forward_token(model, tok, pos, cache, ablation);
    }
    h
}

fn greedy_from(
    model: &K3CpuModel,
    cache: &mut HybridCache,
    hidden: &mut Vec<f32>,
    tokens: &mut Vec<u32>,
    max_new: usize,
) {
    let ablation = Ablation::default();
    for _ in 0..max_new {
        let next = argmax(&logits(model, hidden));
        tokens.push(next);
        let pos = tokens.len() - 1;
        *hidden = forward_token(model, next, pos, cache, ablation);
    }
}

fn prefix_hit_tokens(model: &K3CpuModel, prefix: &[u32], max_new: usize) -> Vec<u32> {
    let mut cache = HybridCache::from_graph(&model.graph, &model.kda);
    let mut h = prefill(model, prefix, &mut cache);
    let mut tokens = prefix.to_vec();
    greedy_from(model, &mut cache, &mut h, &mut tokens, max_new);
    tokens
}

fn assert_c3(model: &K3CpuModel, prefix: &[u32], label: &str) {
    let hit = prefix_hit_tokens(model, prefix, NEW_TOKENS);
    let cold = greedy_decode(model, prefix, NEW_TOKENS, Ablation::default());
    assert_eq!(
        hit, cold,
        "C3 {label}: prefix-cache hit must match no-cache greedy"
    );
}

fn stomp_kda_conv_slot0(cache: &mut HybridCache, kernel: usize) {
    for slot in &mut cache.layers {
        if let LayerCache::Kda(state) = slot {
            assert!(!state.conv.is_empty(), "KDA conv empty");
            assert_eq!(state.conv.len() % kernel, 0);
            // Slot 0 is shifted out on the next conv_update, so it is dead
            // for the following token. Plant on slot 0 and on the tap that
            // becomes slot 0 after the shift (current slot 1).
            for c in 0..state.conv.len() / kernel {
                let row = c * kernel;
                state.conv[row] = 7.0;
                if kernel > 1 {
                    state.conv[row + 1] = 7.0;
                }
            }
            return;
        }
    }
    panic!("no KDA cache slot to stomp");
}

fn stomp_mla_kv(cache: &mut HybridCache) {
    for slot in &mut cache.layers {
        if let LayerCache::Mla(kv) = slot {
            assert!(kv.seq_len > 0, "MLA kv empty after prefix write");
            for x in &mut kv.k {
                *x = 7.0;
            }
            for x in &mut kv.v {
                *x = -7.0;
            }
            return;
        }
    }
    panic!("no MLA cache slot to stomp");
}

#[test]
fn c3_prefix_cache_hit_matches_nocache_tiny() {
    assert_c3(&K3CpuModel::synthetic_tiny(), &[1, 2, 3, 4], "tiny");
}

#[test]
fn c3_prefix_cache_hit_matches_nocache_small() {
    assert_c3(&K3CpuModel::synthetic_small(), &[1, 2, 3, 4, 5], "small");
}

#[test]
fn c3_stomp_kda_conv_slot0_changes_tokens() {
    let model = K3CpuModel::synthetic_tiny();
    let prefix = [1u32, 2, 3, 4];
    let clean = prefix_hit_tokens(&model, &prefix, NEW_TOKENS);

    let mut cache = HybridCache::from_graph(&model.graph, &model.kda);
    let mut h = prefill(&model, &prefix, &mut cache);
    stomp_kda_conv_slot0(&mut cache, model.kda.conv_kernel);
    let mut tokens = prefix.to_vec();
    greedy_from(&model, &mut cache, &mut h, &mut tokens, NEW_TOKENS);

    assert_ne!(
        tokens, clean,
        "RST known-bad: stomping KDA conv slot 0 after a cache write must change tokens"
    );
}

#[test]
fn c3_stomp_mla_kv_changes_tokens() {
    let model = K3CpuModel::synthetic_tiny();
    let prefix = [1u32, 2, 3, 4];
    let clean = prefix_hit_tokens(&model, &prefix, NEW_TOKENS);

    let mut cache = HybridCache::from_graph(&model.graph, &model.kda);
    let mut h = prefill(&model, &prefix, &mut cache);
    stomp_mla_kv(&mut cache);
    let mut tokens = prefix.to_vec();
    greedy_from(&model, &mut cache, &mut h, &mut tokens, NEW_TOKENS);

    assert_ne!(
        tokens, clean,
        "RST known-bad: stomping MLA kv after a cache write must change tokens"
    );
}

// TODO: GPU C3 — paged prefix-cache restore (KDA conv/recurrent + MLA KV) vs cold prefill.
