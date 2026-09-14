// SPDX-License-Identifier: AGPL-3.0-only

//! Hybrid cache: paged MLA KV + KDA recurrent/conv state.
//!
//! Prefix-cache restore reuses C3 CPU semantics: snapshot/restore the
//! per-layer [`LayerCache`] (KDA conv/recurrent or MLA KV).
//!
//! Host bytes are the prefix snapshot. Decode hot path must use
//! `spark_model::kimi_k3::DeviceHybridCache` (device-resident conv/recurrent
//! and MLA KV). Per-token D2H/H2D of those buffers is not this cache.

use anyhow::{Context, Result, bail};

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

impl LayerCache {
    /// Host blob for Marconi aux / C3 prefix-cache restore.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        match self {
            LayerCache::Kda(s) => {
                b.push(1);
                push_f32s(&mut b, &s.conv);
                push_f32s(&mut b, &s.recurrent);
            }
            LayerCache::Mla(kv) => {
                b.push(2);
                b.extend_from_slice(&(kv.seq_len as u32).to_le_bytes());
                push_f32s(&mut b, &kv.k);
                push_f32s(&mut b, &kv.v);
            }
        }
        b
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let tag = bytes.first().copied().context("empty K3 LayerCache blob")?;
        let rest = &bytes[1..];
        match tag {
            1 => {
                let (conv, rest) = take_f32s(rest)?;
                let (recurrent, rest) = take_f32s(rest)?;
                if !rest.is_empty() {
                    bail!("KDA LayerCache blob has trailing bytes");
                }
                Ok(LayerCache::Kda(KdaState { conv, recurrent }))
            }
            2 => {
                if rest.len() < 4 {
                    bail!("MLA LayerCache blob truncated seq_len");
                }
                let seq_len = u32::from_le_bytes(rest[..4].try_into().unwrap()) as usize;
                let (k, rest) = take_f32s(&rest[4..])?;
                let (v, rest) = take_f32s(rest)?;
                if !rest.is_empty() {
                    bail!("MLA LayerCache blob has trailing bytes");
                }
                Ok(LayerCache::Mla(MlaKv { k, v, seq_len }))
            }
            t => bail!("unknown K3 LayerCache tag {t}"),
        }
    }
}

fn push_f32s(b: &mut Vec<u8>, xs: &[f32]) {
    b.extend_from_slice(&(xs.len() as u32).to_le_bytes());
    for x in xs {
        b.extend_from_slice(&x.to_le_bytes());
    }
}

fn take_f32s(bytes: &[u8]) -> Result<(Vec<f32>, &[u8])> {
    if bytes.len() < 4 {
        bail!("LayerCache f32 vec truncated length");
    }
    let n = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let need = 4 + n.checked_mul(4).context("LayerCache f32 overflow")?;
    if bytes.len() < need {
        bail!("LayerCache f32 vec truncated body");
    }
    let mut v = Vec::with_capacity(n);
    for chunk in bytes[4..need].chunks_exact(4) {
        v.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok((v, &bytes[need..]))
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
            gate_lower_bound: Some(-5.0),
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

    #[test]
    fn layer_cache_bytes_roundtrip_kda_and_mla() {
        let kda = KdaConfig {
            heads: 1,
            head_dim: 2,
            conv_kernel: 4,
            gate_lower_bound: Some(-5.0),
            use_full_rank_gate: true,
        };
        let mut k = LayerCache::Kda(KdaState::new(&kda));
        if let LayerCache::Kda(s) = &mut k {
            s.conv[0] = 1.25;
            s.recurrent[0] = -0.5;
        }
        let back = LayerCache::from_bytes(&k.to_bytes()).unwrap();
        match back {
            LayerCache::Kda(s) => {
                assert_eq!(s.conv[0], 1.25);
                assert_eq!(s.recurrent[0], -0.5);
            }
            LayerCache::Mla(_) => panic!("KDA roundtrip"),
        }
        let mut m = LayerCache::Mla(MlaKv::default());
        if let LayerCache::Mla(kv) = &mut m {
            kv.append(&[1.0, 2.0], &[3.0, 4.0]);
        }
        let back = LayerCache::from_bytes(&m.to_bytes()).unwrap();
        match back {
            LayerCache::Mla(kv) => {
                assert_eq!(kv.seq_len, 1);
                assert_eq!(kv.k, vec![1.0, 2.0]);
                assert_eq!(kv.v, vec![3.0, 4.0]);
            }
            LayerCache::Kda(_) => panic!("MLA roundtrip"),
        }
    }

    #[test]
    fn trash_kda_state_after_prefix_clone_diverges() {
        let kda = KdaConfig::twin_0_40b();
        let mut cache = HybridCache {
            layers: vec![LayerCache::Kda(KdaState::new(&kda))],
        };
        if let LayerCache::Kda(s) = &mut cache.layers[0] {
            s.conv[0] = 0.3;
            s.recurrent[1] = -0.2;
        }
        let prefix = cache.clone();
        if let LayerCache::Kda(s) = &mut cache.layers[0] {
            for x in &mut s.conv {
                *x = 7.0;
            }
            for x in &mut s.recurrent {
                *x = 7.0;
            }
        }
        assert_ne!(
            cache.layers[0].to_bytes(),
            prefix.layers[0].to_bytes(),
            "RST known-bad: trash-all KDA conv+recurrent after prefix clone must change the blob"
        );
    }

    #[test]
    fn wrong_mla_kv_row_after_append_diverges() {
        let mut kv = MlaKv::default();
        kv.append(&[1.0, 0.0], &[0.0, 1.0]);
        kv.append(&[2.0, 0.0], &[0.0, 2.0]);
        let clean = kv.clone();
        let k_stride = kv.k.len() / kv.seq_len;
        let v_stride = kv.v.len() / kv.seq_len;
        let last = kv.seq_len - 1;
        for i in 0..k_stride {
            kv.k.swap(i, last * k_stride + i);
        }
        for i in 0..v_stride {
            kv.v.swap(i, last * v_stride + i);
        }
        assert_ne!(
            kv.k, clean.k,
            "RST known-bad: swapped MLA K row must diverge"
        );
        assert_ne!(
            kv.v, clean.v,
            "RST known-bad: swapped MLA V row must diverge"
        );
    }
}
