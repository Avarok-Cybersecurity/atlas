// SPDX-License-Identifier: AGPL-3.0-only

//! DSA binding proofs against the measured layer-3 shapes of
//! `LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3`.

use std::collections::BTreeMap;

use super::*;
use crate::layers::glm5next_kda::binding::RawTensor;

fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 64,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: 16_384,
    }
}

/// A source built straight from the spec table — the "checkpoint is exactly right"
/// baseline that individual tests then perturb.
struct FakeSource {
    t: BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)>,
}

impl FakeSource {
    fn good() -> Self {
        let mut t = BTreeMap::new();
        for s in dsa_tensor_specs(&cfg(), 64) {
            let elems: usize = s.shape.iter().product();
            t.insert(
                s.name.clone(),
                (s.dtype, s.shape.clone(), vec![0u8; elems * 2]),
            );
        }
        Self { t }
    }
    fn drop(mut self, name: &str) -> Self {
        assert!(self.t.remove(name).is_some(), "{name} was not present");
        self
    }
    fn reshape(mut self, name: &str, shape: Vec<usize>) -> Self {
        let e = self.t.get_mut(name).expect("present");
        let elems: usize = shape.iter().product();
        e.1 = shape;
        e.2 = vec![0u8; elems * 2];
        self
    }
    fn add(mut self, name: &str, shape: Vec<usize>) -> Self {
        let elems: usize = shape.iter().product();
        self.t
            .insert(name.to_string(), (Dtype::Bf16, shape, vec![0u8; elems * 2]));
        self
    }
}

impl TensorSource for FakeSource {
    fn get(&self, name: &str) -> Option<RawTensor<'_>> {
        self.t.get(name).map(|(d, s, b)| RawTensor {
            dtype: *d,
            shape: s.clone(),
            bytes: b.as_slice(),
        })
    }
    fn names(&self) -> Vec<String> {
        self.t.keys().cloned().collect()
    }
}

/// The spec table reproduces the checkpoint's measured layer byte total exactly.
#[test]
fn spec_table_matches_the_measured_layer() {
    let specs = dsa_tensor_specs(&cfg(), 64);
    assert_eq!(specs.len(), 14, "DSA block tensor count");
    let bytes: usize = specs
        .iter()
        .map(|s| s.shape.iter().product::<usize>() * 2)
        .sum();
    // attn_DSA = 2,748,117,504 B over 11 text DSA layers.
    assert_eq!(bytes, 249_828_864);
    assert_eq!(bytes * 11, 2_748_117_504);
}

#[test]
fn a_correct_block_binds() {
    let r = verify_dsa_block(&cfg(), 64, &FakeSource::good()).expect("should bind");
    assert_eq!(r.bound, 14);
    assert_eq!(r.total_bytes, 249_828_864);
    assert!(r.unclaimed.is_empty());
}

/// 🪤 The LayerNorm bias is REQUIRED. Dropping it must be an error, never a
/// silent zero — that is the whole reason `k_norm` is not an RMSNorm.
#[test]
fn a_missing_k_norm_bias_is_an_error() {
    let src = FakeSource::good().drop("self_attn.indexer.k_norm.bias");
    let e = verify_dsa_block(&cfg(), 64, &src).unwrap_err();
    assert!(e.to_string().contains("k_norm.bias"), "unexpected: {e}");
    assert!(e.to_string().contains("missing"), "unexpected: {e}");
}

/// 🪤 Swapping q_b's and kv_b's per-head widths still yields a well-formed 2-D
/// tensor. The shape check is what catches it.
#[test]
fn swapped_per_head_widths_are_rejected() {
    // kv_b's width (512) applied to q_b.
    let src = FakeSource::good().reshape("self_attn.q_b_proj.weight", vec![64 * 512, 1536]);
    let e = verify_dsa_block(&cfg(), 64, &src).unwrap_err();
    assert!(e.to_string().contains("q_b_proj"), "unexpected: {e}");
    assert!(e.to_string().contains("shape"), "unexpected: {e}");
}

/// 🪤 NoPE: a 576-wide latent (DeepSeek-V4-Flash's 512+64) must be rejected.
#[test]
fn a_deepseek_v4_shaped_latent_is_rejected() {
    let src = FakeSource::good().reshape("self_attn.kv_a_proj_with_mqa.weight", vec![576, 4096]);
    let e = verify_dsa_block(&cfg(), 64, &src).unwrap_err();
    assert!(
        e.to_string().contains("kv_a_proj_with_mqa"),
        "unexpected: {e}"
    );
}

/// 🪤 The APE table is [kpool, index_head_dim]. The transpose has the same element
/// count, so only the shape check separates them.
#[test]
fn a_transposed_ape_table_is_rejected() {
    let src =
        FakeSource::good().reshape("self_attn.indexer.index_kpool_compress_ape", vec![128, 4]);
    let e = verify_dsa_block(&cfg(), 64, &src).unwrap_err();
    assert!(e.to_string().contains("compress_ape"), "unexpected: {e}");
}

/// An unexpected attention tensor means the architecture moved. Never skipped.
#[test]
fn an_unclaimed_self_attn_tensor_is_an_error() {
    let src = FakeSource::good().add("self_attn.wkv_a_rope.weight", vec![64, 4096]);
    let e = verify_dsa_block(&cfg(), 64, &src).unwrap_err();
    assert!(e.to_string().contains("unclaimed"), "unexpected: {e}");
}

/// Non-attention tensors in the same layer (mHC, norms, MLP) are not the DSA
/// binder's business and must not trip the unclaimed check.
#[test]
fn non_attention_tensors_are_ignored() {
    let src = FakeSource::good()
        .add("hc_attn_fn", vec![24, 16384])
        .add("mlp.gate.weight", vec![288, 4096])
        .add("input_layernorm.weight", vec![4096]);
    let r = verify_dsa_block(&cfg(), 64, &src).expect("should still bind");
    assert_eq!(r.bound, 14);
}

/// The spec table follows the config: a different head count changes exactly the
/// three head-shaped tensors and nothing else.
#[test]
fn specs_track_the_head_count() {
    let a = dsa_tensor_specs(&cfg(), 64);
    let b = dsa_tensor_specs(&cfg(), 32);
    let differing: Vec<&str> = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| x.shape != y.shape)
        .map(|(x, _)| x.name.as_str())
        .collect();
    assert_eq!(
        differing,
        vec![
            "self_attn.q_b_proj.weight",
            "self_attn.kv_b_proj.weight",
            "self_attn.o_proj.weight"
        ]
    );
}
