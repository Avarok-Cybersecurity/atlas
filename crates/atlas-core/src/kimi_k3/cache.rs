// SPDX-License-Identifier: AGPL-3.0-only

//! Hybrid cache: paged MLA KV + KDA recurrent/conv state.
//!
//! Struct-only this slice. Forward restore is exercised by the KDA
//! prefix-hit test; GPU paging comes later.

use super::kda::{KdaConfig, KdaState};
use super::layer::{K3Graph, MixerKind};

/// One MLA layer's host KV (unpaged CPU stand-in).
#[derive(Clone, Debug, Default)]
pub struct MlaKv {
    /// Packed keys `[T, H, dq]`.
    pub k: Vec<f32>,
    /// Packed values `[T, H, dv]`.
    pub v: Vec<f32>,
    pub seq_len: usize,
}

#[derive(Clone, Debug)]
pub enum LayerCache {
    Kda(KdaState),
    Mla(MlaKv),
}

/// Per-sequence hybrid cache. Slot identity is the layer index; a prefix
/// hit that writes KDA state into the wrong slot is the C4 mutant.
#[derive(Clone, Debug)]
pub struct HybridCache {
    pub layers: Vec<LayerCache>,
}

impl HybridCache {
    pub fn from_graph(graph: &K3Graph, kda: &KdaConfig) -> Self {
        let layers = graph
            .layers
            .iter()
            .map(|l| match l.mixer {
                MixerKind::Kda => LayerCache::Kda(KdaState::new(kda)),
                MixerKind::Mla => LayerCache::Mla(MlaKv::default()),
            })
            .collect();
        Self { layers }
    }

    pub fn kda_mut(&mut self, layer: usize) -> Option<&mut KdaState> {
        match self.layers.get_mut(layer) {
            Some(LayerCache::Kda(s)) => Some(s),
            _ => None,
        }
    }

    pub fn mla_mut(&mut self, layer: usize) -> Option<&mut MlaKv> {
        match self.layers.get_mut(layer) {
            Some(LayerCache::Mla(s)) => Some(s),
            _ => None,
        }
    }
}

impl MlaKv {
    /// Append one token's packed K/V (`[H, dq]` / `[H, dv]`).
    pub fn append(&mut self, k: &[f32], v: &[f32]) {
        self.k.extend_from_slice(k);
        self.v.extend_from_slice(v);
        self.seq_len += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::parse_config;
    use crate::kimi_k3::layer::K3Graph;

    #[test]
    fn twin_cache_slots_follow_mixer() {
        const TWIN: &str = include_str!("../../../../docs/k3/fixtures/Kimi-K3-0.40B-config.json");
        let c = parse_config(TWIN).unwrap();
        let g = K3Graph::from_config(&c);
        let kda = KdaConfig {
            heads: 8,
            head_dim: 32,
            conv_kernel: 4,
            gate_lower_bound: -5.0,
            use_full_rank_gate: true,
        };
        let cache = HybridCache::from_graph(&g, &kda);
        assert_eq!(cache.layers.len(), 8);
        for i in [0, 1, 2, 4, 5, 6] {
            assert!(matches!(cache.layers[i], LayerCache::Kda(_)));
        }
        for i in [3, 7] {
            assert!(matches!(cache.layers[i], LayerCache::Mla(_)));
        }
    }
}
