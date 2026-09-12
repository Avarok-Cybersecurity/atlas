// SPDX-License-Identifier: AGPL-3.0-only

//! C4: MLA KV and KDA state advance on the same positions.
//!
//! Self-consistency / fixture gate (not HF token-exact). After prefilling
//! `N` tokens, one decode step must leave KDA conv/recurrent and MLA kv
//! matching a cold prefill of `N+1`, including when the prefix cache is
//! cloned (a prefix hit).

use super::cache::{HybridCache, LayerCache, MlaKv};
use super::cpu_forward::forward_token;
use super::cpu_weights::{Ablation, K3CpuModel};
use super::kda::KdaState;

const ATOL: f32 = 1e-5;

fn close(a: &[f32], b: &[f32], atol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= atol)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn prefill(model: &K3CpuModel, tokens: &[u32], cache: &mut HybridCache) {
    let ablation = Ablation::default();
    for (pos, &tok) in tokens.iter().enumerate() {
        let _ = forward_token(model, tok, pos, cache, ablation);
    }
}

fn cold_cache(model: &K3CpuModel, tokens: &[u32]) -> HybridCache {
    let mut cache = HybridCache::from_graph(&model.graph, &model.kda);
    prefill(model, tokens, &mut cache);
    cache
}

/// Prefill `prefix`, clone the cache (prefix hit), then decode `next`.
fn prefix_hit_decode(model: &K3CpuModel, prefix: &[u32], next: u32) -> HybridCache {
    let mut cache = HybridCache::from_graph(&model.graph, &model.kda);
    prefill(model, prefix, &mut cache);
    let mut hit = cache.clone();
    let _ = forward_token(model, next, prefix.len(), &mut hit, Ablation::default());
    hit
}

fn caches_close(a: &HybridCache, b: &HybridCache, atol: f32) -> bool {
    a.layers.len() == b.layers.len()
        && a.layers.iter().zip(&b.layers).all(|(x, y)| match (x, y) {
            (LayerCache::Kda(s), LayerCache::Kda(t)) => {
                close(&s.conv, &t.conv, atol) && close(&s.recurrent, &t.recurrent, atol)
            }
            (LayerCache::Mla(s), LayerCache::Mla(t)) => {
                s.seq_len == t.seq_len && close(&s.k, &t.k, atol) && close(&s.v, &t.v, atol)
            }
            _ => false,
        })
}

fn cache_max_abs(a: &HybridCache, b: &HybridCache) -> f32 {
    a.layers
        .iter()
        .zip(&b.layers)
        .map(|(x, y)| match (x, y) {
            (LayerCache::Kda(s), LayerCache::Kda(t)) => {
                max_abs(&s.conv, &t.conv).max(max_abs(&s.recurrent, &t.recurrent))
            }
            (LayerCache::Mla(s), LayerCache::Mla(t)) => {
                max_abs(&s.k, &t.k).max(max_abs(&s.v, &t.v))
            }
            _ => f32::INFINITY,
        })
        .fold(0.0, f32::max)
}

fn assert_mla_len(cache: &HybridCache, want: usize) {
    for (i, slot) in cache.layers.iter().enumerate() {
        if let LayerCache::Mla(kv) = slot {
            assert_eq!(kv.seq_len, want, "MLA layer {i} seq_len");
            let k_stride = kv.k.len() / kv.seq_len.max(1);
            let v_stride = kv.v.len() / kv.seq_len.max(1);
            assert_eq!(kv.k.len(), want * k_stride);
            assert_eq!(kv.v.len(), want * v_stride);
        }
    }
}

fn assert_c4(model: &K3CpuModel, prefix: &[u32], next: u32, label: &str) {
    let hit = prefix_hit_decode(model, prefix, next);
    let mut full_tokens = prefix.to_vec();
    full_tokens.push(next);
    let cold = cold_cache(model, &full_tokens);
    assert_mla_len(&hit, full_tokens.len());
    assert_mla_len(&cold, full_tokens.len());
    assert!(
        caches_close(&hit, &cold, ATOL),
        "C4 {label}: prefix-hit decode vs cold prefill of N+1 max_abs={} atol={ATOL}",
        cache_max_abs(&hit, &cold)
    );
}

/// Newest conv sample written into slot 0 of the prefix window (no shift).
fn plant_kda_conv_wrong_slot(prefix: &KdaState, after: &mut KdaState, kernel: usize) {
    assert_eq!(prefix.conv.len(), after.conv.len());
    assert_eq!(after.conv.len() % kernel, 0);
    for c in 0..after.conv.len() / kernel {
        let row = c * kernel;
        let newest = after.conv[row + kernel - 1];
        after.conv[row..row + kernel].copy_from_slice(&prefix.conv[row..row + kernel]);
        after.conv[row] = newest;
    }
}

/// New token's packed K/V written over row 0; seq_len stays N+1.
fn plant_mla_kv_wrong_row(prefix: &MlaKv, after: &mut MlaKv) {
    assert!(after.seq_len == prefix.seq_len + 1, "need one appended row");
    assert!(prefix.seq_len > 0);
    let k_stride = after.k.len() / after.seq_len;
    let v_stride = after.v.len() / after.seq_len;
    let last = after.seq_len - 1;
    let swap = |buf: &mut [f32], stride: usize| {
        for i in 0..stride {
            buf.swap(i, last * stride + i);
        }
    };
    swap(&mut after.k, k_stride);
    swap(&mut after.v, v_stride);
}

fn prefix_and_hit(
    model: &K3CpuModel,
    prefix: &[u32],
    next: u32,
) -> (HybridCache, HybridCache, HybridCache) {
    let mut cache = HybridCache::from_graph(&model.graph, &model.kda);
    prefill(model, prefix, &mut cache);
    let prefix_cache = cache.clone();
    let hit = prefix_hit_decode(model, prefix, next);
    let mut full = prefix.to_vec();
    full.push(next);
    (prefix_cache, hit, cold_cache(model, &full))
}

#[test]
fn c4_hybrid_state_prefix_hit_matches_cold_prefill_tiny() {
    assert_c4(&K3CpuModel::synthetic_tiny(), &[1, 2, 3, 4], 5, "tiny");
}

#[test]
fn c4_hybrid_state_prefix_hit_matches_cold_prefill_small() {
    assert_c4(&K3CpuModel::synthetic_small(), &[1, 2, 3, 4, 5], 6, "small");
}

#[test]
fn c4_wrong_kda_conv_slot_after_prefix_hit_diverges() {
    let model = K3CpuModel::synthetic_tiny();
    let prefix = [1u32, 2, 3, 4];
    let (prefix_cache, mut hit, cold) = prefix_and_hit(&model, &prefix, 5);
    let kernel = model.kda.conv_kernel;
    let mut planted = false;
    for (slot, prefix_slot) in hit.layers.iter_mut().zip(&prefix_cache.layers) {
        if let (LayerCache::Kda(after), LayerCache::Kda(before)) = (slot, prefix_slot) {
            plant_kda_conv_wrong_slot(before, after, kernel);
            planted = true;
            break;
        }
    }
    assert!(planted, "no KDA slot to plant");
    assert!(
        !caches_close(&hit, &cold, ATOL),
        "RST known-bad: writing the new KDA conv sample into slot 0 must diverge from cold N+1 (max_abs={})",
        cache_max_abs(&hit, &cold)
    );
}

#[test]
fn c4_wrong_mla_kv_row_after_prefix_hit_diverges() {
    let model = K3CpuModel::synthetic_tiny();
    let prefix = [1u32, 2, 3, 4];
    let (prefix_cache, mut hit, cold) = prefix_and_hit(&model, &prefix, 5);
    let mut planted = false;
    for (slot, prefix_slot) in hit.layers.iter_mut().zip(&prefix_cache.layers) {
        if let (LayerCache::Mla(after), LayerCache::Mla(before)) = (slot, prefix_slot) {
            plant_mla_kv_wrong_row(before, after);
            planted = true;
            break;
        }
    }
    assert!(planted, "no MLA slot to plant");
    assert!(
        !caches_close(&hit, &cold, ATOL),
        "RST known-bad: writing the new MLA kv into row 0 must diverge from cold N+1 (max_abs={})",
        cache_max_abs(&hit, &cold)
    );
}

// TODO: GPU C4 — paged hybrid cache: prefix-hit KDA conv/recurrent + MLA KV vs cold prefill.
