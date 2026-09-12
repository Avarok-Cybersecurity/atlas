// SPDX-License-Identifier: AGPL-3.0-only

//! One K3 decoder layer as host graph data: mixer + MLP + AttnRes sites.
//!
//! ```text
//! h = AttnRes(blocks, partial, attn_res_*)
//! mix_out = KDA | gated-NoPE-MLA (on RMSNorm(h))
//! partial += mix_out
//! h = AttnRes(blocks, partial, mlp_res_*)
//! mlp_out = dense SiTU-GLU | LatentMoE (on RMSNorm(h))
//! partial += mlp_out
//! at block boundary: push partial, reset
//! ```

use crate::config::{LayerType, ModelConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixerKind {
    Kda,
    Mla,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpKind {
    Dense,
    LatentMoe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct K3LayerSpec {
    pub index: usize,
    pub mixer: MixerKind,
    pub mlp: MlpKind,
}

#[derive(Debug, Clone)]
pub struct K3Graph {
    pub layers: Vec<K3LayerSpec>,
    pub hidden: usize,
    pub attn_res_block_size: usize,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    pub use_full_rank_gate: bool,
    pub mla_use_nope: bool,
    pub mla_use_output_gate: bool,
}

impl K3Graph {
    pub fn from_config(c: &ModelConfig) -> Self {
        let dense_end = c.mlp_only_layers.len();
        let layers = c
            .layer_types
            .iter()
            .enumerate()
            .map(|(i, t)| K3LayerSpec {
                index: i,
                mixer: match t {
                    LayerType::LinearAttention => MixerKind::Kda,
                    _ => MixerKind::Mla,
                },
                mlp: if i < dense_end {
                    MlpKind::Dense
                } else {
                    MlpKind::LatentMoe
                },
            })
            .collect();
        Self {
            layers,
            hidden: c.hidden_size,
            attn_res_block_size: c.attn_res_block_size,
            situ_beta: c.activation_situ_beta,
            situ_linear_beta: c.activation_situ_linear_beta,
            use_full_rank_gate: c.use_full_rank_gate,
            mla_use_nope: c.mla_use_nope,
            mla_use_output_gate: c.mla_use_output_gate,
        }
    }

    pub fn kda_count(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| l.mixer == MixerKind::Kda)
            .count()
    }

    pub fn mla_count(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| l.mixer == MixerKind::Mla)
            .count()
    }

    pub fn last_is_mla(&self) -> bool {
        self.layers
            .last()
            .is_some_and(|l| l.mixer == MixerKind::Mla)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::parse_config;

    #[test]
    fn twin_0_40b_graph_is_six_kda_last_mla() {
        const TWIN: &str = include_str!("../../../../docs/k3/fixtures/Kimi-K3-0.40B-config.json");
        let c = parse_config(TWIN).expect("0.40B twin");
        let g = K3Graph::from_config(&c);
        assert_eq!(g.layers.len(), 8);
        assert_eq!(g.kda_count(), 6);
        assert_eq!(g.mla_count(), 2);
        assert!(g.last_is_mla());
        // 3:1 then trailing MLA: 0-2 KDA, 3 MLA, 4-6 KDA, 7 MLA.
        for i in [0, 1, 2, 4, 5, 6] {
            assert_eq!(g.layers[i].mixer, MixerKind::Kda, "layer {i}");
        }
        assert_eq!(g.layers[3].mixer, MixerKind::Mla);
        assert_eq!(g.layers[7].mixer, MixerKind::Mla);
        assert_eq!(g.layers[0].mlp, MlpKind::Dense);
        assert!(g.layers[1..].iter().all(|l| l.mlp == MlpKind::LatentMoe));
        assert_eq!(g.attn_res_block_size, 4);
        assert_eq!(g.hidden, 1024);
        assert!(g.mla_use_nope);
        assert!(g.mla_use_output_gate);
        assert!(g.use_full_rank_gate);
        assert_eq!(g.situ_beta, 4.0);
        assert_eq!(g.situ_linear_beta, 25.0);
    }

    #[test]
    fn official_graph_census() {
        const OFFICIAL: &str =
            include_str!("../../../../docs/k3/fixtures/moonshotai-Kimi-K3-config.json");
        let c = parse_config(OFFICIAL).expect("official");
        let g = K3Graph::from_config(&c);
        assert_eq!(g.layers.len(), 93);
        assert_eq!(g.kda_count(), 69);
        assert_eq!(g.mla_count(), 24);
        assert!(g.last_is_mla());
        assert_eq!(g.attn_res_block_size, 12);
        assert_eq!(g.layers[0].mlp, MlpKind::Dense);
    }
}
