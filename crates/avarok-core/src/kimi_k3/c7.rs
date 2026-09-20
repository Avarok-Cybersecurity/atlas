// SPDX-License-Identifier: AGPL-3.0-only

//! C7: dummy TP=2 column-split `o_proj` + allreduce == unsplit TP=1 tokens.
//!
//! In-process two-rank (thread + sum). This is **not** spark1+spark2 NCCL,
//! and `nccl_2rank_bench` is not token identity.

use super::cpu_weights::{Ablation, K3CpuModel};
use super::greedy::greedy_decode;
use super::layer::{MixerKind, MlpKind};
use super::ops::{ident, matvec, matvec_column_tp};

const NEW_TOKENS: usize = 8;
const PROMPT: [u32; 4] = [1, 2, 3, 4];

fn tp_ablation(world: usize, drop_rank: Option<usize>) -> Ablation {
    Ablation {
        o_proj_tp: world,
        drop_o_proj_rank: drop_rank,
        ..Ablation::default()
    }
}

fn dummy() -> &'static K3CpuModel {
    static M: std::sync::OnceLock<K3CpuModel> = std::sync::OnceLock::new();
    M.get_or_init(K3CpuModel::synthetic_prod_width_dummy)
}

fn greedy(model: &K3CpuModel, world: usize, drop: Option<usize>) -> Vec<u32> {
    greedy_decode(model, &PROMPT, NEW_TOKENS, tp_ablation(world, drop))
}

#[test]
fn c7_column_tp2_ident_allreduce_is_bit_exact() {
    let w = ident(8, 8);
    let x: Vec<f32> = (0..8).map(|i| (i as f32) * 0.25).collect();
    let y1 = matvec(&w, &x, 8, 8);
    let y2 = matvec_column_tp(&w, &x, 8, 8, 2, None);
    assert_eq!(y1, y2, "identity o_proj TP=2 allreduce must match unsplit");
    let dropped = matvec_column_tp(&w, &x, 8, 8, 2, Some(1));
    assert_ne!(dropped, y1, "drop rank-1 shard must change the hidden");
}

#[test]
fn c7_tiny_tp2_matches_tp1_tokens() {
    let model = K3CpuModel::synthetic_tiny();
    let tp1 = greedy(&model, 1, None);
    let tp2 = greedy(&model, 2, None);
    assert_eq!(tp1.len(), PROMPT.len() + NEW_TOKENS);
    assert_eq!(tp1, tp2, "C7 tiny: TP=2 tokens must match TP=1");
}

#[test]
fn c7_tiny_drop_rank1_o_proj_shard_changes_tokens() {
    let model = K3CpuModel::synthetic_tiny();
    let tp2 = greedy(&model, 2, None);
    let dropped = greedy(&model, 2, Some(1));
    assert_ne!(
        dropped, tp2,
        "RST known-bad: drop rank-1 o_proj shard must change tiny tokens"
    );
}

#[test]
fn c7_prod_width_dummy_shape() {
    let model = dummy();
    assert_eq!(model.graph.hidden, 7168);
    assert_eq!(model.kda.heads, 96);
    assert_eq!(model.mla.heads, 96);
    assert_eq!(model.vocab, 256);
    assert_eq!(model.graph.layers.len(), 2);
    assert_eq!(model.graph.layers[0].mixer, MixerKind::Kda);
    assert_eq!(model.graph.layers[0].mlp, MlpKind::Dense);
    assert_eq!(model.graph.layers[1].mixer, MixerKind::Mla);
    assert_eq!(model.graph.layers[1].mlp, MlpKind::LatentMoe);
    assert!(model.graph.last_is_mla());
    assert_eq!(model.moe.n_routed, 8);
    assert!(model.kda.qkv_dim().is_multiple_of(2));
    assert!((model.mla.heads * model.mla.v_head_dim).is_multiple_of(2));
}

#[test]
fn c7_dummy_tp2_matches_tp1_tokens() {
    let model = dummy();
    let tp1 = greedy(model, 1, None);
    let tp2 = greedy(model, 2, None);
    assert_eq!(tp1.len(), PROMPT.len() + NEW_TOKENS);
    assert!(
        tp1[PROMPT.len()..]
            .iter()
            .all(|&t| (t as usize) < model.vocab)
    );
    assert_eq!(
        tp1, tp2,
        "C7: TP=2 column-split o_proj + allreduce must match TP=1 tokens"
    );
}

#[test]
fn c7_drop_rank1_o_proj_shard_changes_tokens() {
    let model = dummy();
    let tp2 = greedy(model, 2, None);
    let dropped = greedy(model, 2, Some(1));
    assert_ne!(
        dropped, tp2,
        "RST known-bad: drop rank-1 o_proj shard must change tokens"
    );
}

// TODO: GPU C7 — spark1+spark2 NCCL TP=2 vs spark1 single-GPU tokens.
// `nccl_2rank_bench` is fabric, not this oracle.
