// SPDX-License-Identifier: AGPL-3.0-only
//! Additional GPU weight storage created by the marked K3 model binder.
//! Dense/packed layer binding aliases resident allocations. Only the three
//! engine-facing FP32 tensors allocate BF16 copies; source allocations remain.
//! This excludes engine workspace, KV, CUDA/NCCL and lazy inference allocations.
use anyhow::{Context, Result};

pub fn engine_bf16_weight(name: &str) -> bool {
    [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    .iter()
    .any(|suffix| name == *suffix || name.strip_suffix(suffix).is_some_and(|p| p.ends_with('.')))
}

pub fn extra_gpu_bytes(name: &str, is_fp32: bool, elements: usize) -> Result<usize> {
    if is_fp32 && engine_bf16_weight(name) {
        elements
            .checked_mul(2)
            .context("K3 BF16 binding allocation overflow")
    } else {
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fp32_engine_copies_are_counted_and_other_weights_alias() {
        for key in [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
        ] {
            for name in [key.to_string(), format!("language_model.{key}")] {
                assert_eq!(extra_gpu_bytes(&name, true, 64).unwrap(), 128);
                assert_eq!(extra_gpu_bytes(&name, false, 64).unwrap(), 0);
            }
        }
        assert_eq!(
            extra_gpu_bytes("model.layers.0.self_attn.q_proj.weight", true, 64).unwrap(),
            0
        );
        assert!(!engine_bf16_weight("not_lm_head.weight"));
        assert!(extra_gpu_bytes("lm_head.weight", true, usize::MAX).is_err());
    }
}
