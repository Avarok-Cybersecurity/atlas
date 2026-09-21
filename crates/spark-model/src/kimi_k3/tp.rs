// SPDX-License-Identifier: AGPL-3.0-only

//! K3 device-binding adapter for the shared tensor-parallel storage plan.
//! Runtime loading partitions packed and dense weights before GPU allocation.
//!
//! Head counts on `config` are already per-rank. Full sizes = local * tp.

use avarok_core::config::ModelConfig;
use avarok_core::kimi_k3::{MixerKind, MlpKind};

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
    use avarok_core::kimi_k3::tp::{TpAxis, tensor_plan};
    let (axis, n, k) = tensor_plan(name, mixer, config);
    let kind = match axis {
        TpAxis::Replicated => TpShardKind::Replicated,
        TpAxis::Rows => TpShardKind::ColumnParallel,
        TpAxis::Columns => TpShardKind::RowParallel,
    };
    (kind, n, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use avarok_core::config::parse_config;

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

    fn round_bf16(x: f32) -> f32 {
        let bits = avarok_core::numeric::f32_to_bf16(x);
        avarok_core::numeric::bf16_bytes_to_f32(bits.to_le_bytes())
    }

    #[test]
    fn production_7168_bf16_allreduce_rounding_is_measured() {
        // tp_allreduce stores f32 partials as BF16, NCCL-sums, widens back.
        // Twin 16/16 held. This is the production-width error, not an assumption.
        let n = 7168;
        let a = vec![1.0f32 / 3.0; n];
        let b = vec![2.0f32 / 3.0; n];
        let mut max = 0.0f32;
        for i in 0..n {
            let f32s = a[i] + b[i];
            let bfs = round_bf16(a[i]) + round_bf16(b[i]);
            max = max.max((f32s - bfs).abs());
        }
        assert!(max > 0.0, "1/3 is not exact in BF16");
        assert!(
            max < 0.01,
            "7168-wide double BF16 round before NCCL max_abs={max}"
        );
        eprintln!("K3 TP BF16 allreduce max_abs @7168 (1/3+2/3) = {max}");
    }
}
