// SPDX-License-Identifier: AGPL-3.0-only

//! K3 Megatron TP plan. `supports_tp` is true — the umbrella loader used to
//! refuse `--tp-size 2`. Full `slice_for_rank` bind lives with the weight
//! loader; this slice publishes the plan + the fail-fast flag.
//!
//! Head counts on `config` are already per-rank. Full sizes = local * tp.

use atlas_core::config::ModelConfig;
use atlas_core::kimi_k3::{MixerKind, MlpKind};

use crate::tp_shard::TpShardKind;

/// K3 does tensor-parallel. Callers must not fail-fast `supports_tp` false.
pub fn supports_tp() -> bool {
    true
}

/// `(kind, full_out, full_in)` for one checkpoint key.
pub fn tensor_plan(
    name: &str,
    mixer: MixerKind,
    _mlp: MlpKind,
    config: &ModelConfig,
) -> (TpShardKind, usize, usize) {
    let tp = config.tp_world_size.max(1);
    let h = config.hidden_size;
    let kda_heads = config.linear_num_key_heads * tp;
    let kda_d = config.linear_key_head_dim;
    let kda_q = kda_heads * kda_d;
    let conv_k = config.linear_conv_kernel_dim.max(1);
    let mla_heads = config.num_attention_heads * tp;
    let qk = config.qk_nope_head_dim + config.qk_rope_head_dim;
    let dv = mla_heads * config.v_head_dim;
    let kv_b = mla_heads * (config.qk_nope_head_dim + config.v_head_dim);
    let inter = config.intermediate_size;
    let eh = config.moe_intermediate_size;
    let lat = config.moe_latent_size;

    if name.ends_with(".self_attn.q_proj.weight")
        || name.ends_with(".self_attn.k_proj.weight")
        || name.ends_with(".self_attn.v_proj.weight")
    {
        return (TpShardKind::ColumnParallel, kda_q, h);
    }
    if name.ends_with(".self_attn.q_conv1d.weight")
        || name.ends_with(".self_attn.k_conv1d.weight")
        || name.ends_with(".self_attn.v_conv1d.weight")
    {
        return (TpShardKind::ColumnParallel, kda_q, conv_k);
    }
    if name.ends_with(".self_attn.g_proj.weight") {
        let n = match mixer {
            MixerKind::Kda => kda_q,
            MixerKind::Mla => dv,
        };
        return (TpShardKind::ColumnParallel, n, h);
    }
    if name.ends_with(".self_attn.o_proj.weight") {
        let inn = match mixer {
            MixerKind::Kda => kda_q,
            MixerKind::Mla => dv,
        };
        return (TpShardKind::RowParallel, h, inn);
    }
    if name.ends_with(".self_attn.b_proj.weight") {
        return (TpShardKind::ColumnParallel, kda_heads, h);
    }
    if name.ends_with(".self_attn.A_log") {
        return (TpShardKind::ColumnParallel, kda_heads, 1);
    }
    if name.ends_with(".self_attn.dt_bias") {
        return (TpShardKind::ColumnParallel, kda_q, 1);
    }
    if name.ends_with(".self_attn.f_b_proj.weight") {
        return (TpShardKind::ColumnParallel, kda_q, kda_d);
    }
    if name.ends_with(".self_attn.q_b_proj.weight") {
        return (
            TpShardKind::ColumnParallel,
            mla_heads * qk,
            config.q_lora_rank,
        );
    }
    if name.ends_with(".self_attn.kv_b_proj.weight") {
        return (TpShardKind::ColumnParallel, kv_b, config.kv_lora_rank);
    }
    if name.ends_with(".mlp.gate_proj.weight") || name.ends_with(".mlp.up_proj.weight") {
        return (TpShardKind::ColumnParallel, inter, h);
    }
    if name.ends_with(".mlp.down_proj.weight") {
        return (TpShardKind::RowParallel, h, inter);
    }
    if name.contains(".block_sparse_moe.experts.") {
        if name.ends_with(".w1.weight") || name.ends_with(".w3.weight") {
            return (TpShardKind::ColumnParallel, eh, lat);
        }
        if name.ends_with(".w2.weight") {
            return (TpShardKind::RowParallel, lat, eh);
        }
    }
    (TpShardKind::Replicated, 1, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_core::config::parse_config;

    const TWIN: &str = include_str!("../../../../docs/k3/fixtures/Kimi-K3-0.40B-config.json");

    fn twin() -> ModelConfig {
        parse_config(TWIN).expect("0.40B twin")
    }

    fn divide_heads_for_tp(config: &mut ModelConfig, rank: usize, size: usize) {
        config.tp_rank = rank;
        config.tp_world_size = size;
        if size > 1 {
            config.num_attention_heads /= size;
            config.num_key_value_heads /= size;
            config.linear_num_key_heads /= size;
            config.linear_num_value_heads /= size;
        }
    }

    #[test]
    fn kimi_k3_supports_tp() {
        assert!(supports_tp(), "loader must not refuse --tp-size 2");
    }

    #[test]
    fn kda_q_is_column_o_is_row() {
        let mut c = twin();
        divide_heads_for_tp(&mut c, 0, 2);
        let q = tensor_plan(
            "language_model.model.layers.0.self_attn.q_proj.weight",
            MixerKind::Kda,
            MlpKind::Dense,
            &c,
        );
        let o = tensor_plan(
            "language_model.model.layers.0.self_attn.o_proj.weight",
            MixerKind::Kda,
            MlpKind::Dense,
            &c,
        );
        assert_eq!(q, (TpShardKind::ColumnParallel, 8 * 32, 1024));
        assert_eq!(o, (TpShardKind::RowParallel, 1024, 8 * 32));
        let embed = tensor_plan(
            "language_model.model.embed_tokens.weight",
            MixerKind::Kda,
            MlpKind::Dense,
            &c,
        );
        assert_eq!(embed.0, TpShardKind::Replicated);
    }

    #[test]
    fn rank0_and_rank1_q_proj_full_out_match() {
        let mut c0 = twin();
        divide_heads_for_tp(&mut c0, 0, 2);
        let mut c1 = twin();
        divide_heads_for_tp(&mut c1, 1, 2);
        let q0 = tensor_plan(
            "language_model.model.layers.0.self_attn.q_proj.weight",
            MixerKind::Kda,
            MlpKind::Dense,
            &c0,
        );
        let q1 = tensor_plan(
            "language_model.model.layers.0.self_attn.q_proj.weight",
            MixerKind::Kda,
            MlpKind::Dense,
            &c1,
        );
        assert_eq!(q0, q1, "full sizes reconstruct from local*tp on both ranks");
        assert_eq!(c0.tp_rank, 0);
        assert_eq!(c1.tp_rank, 1);
        assert_ne!(c0.tp_rank, c1.tp_rank);
    }
}
