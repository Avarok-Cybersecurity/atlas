// SPDX-License-Identifier: AGPL-3.0-only

//! K3 official MXFP4 names → DeepSeek-V4 E8M0 unpack.
//!
//! DSV4 native MXFP4: `{prefix}.weight` (E2M1 packed) + `{prefix}.scale` (E8M0).
//! K3 official TSV: `{prefix}.weight_packed` + `{prefix}.weight_scale`.
//! Math is [`crate::mxfp4_e8m0`]; this module only remaps names.

use std::collections::HashMap;

use anyhow::{Context, Result, bail, ensure};
use safetensors::tensor::Dtype;

use crate::mxfp4_e8m0::{dequant_nvfp4_e8m0_to_bf16, dequant_nvfp4_e8m0_to_f32};

/// Official TSV classes that are packed routed-expert MXFP4.
pub const PACKED_CLASSES: &[&str] = &[
    "language_model.model.layers.*.block_sparse_moe.experts.*.w1.weight_packed",
    "language_model.model.layers.*.block_sparse_moe.experts.*.w1.weight_scale",
    "language_model.model.layers.*.block_sparse_moe.experts.*.w2.weight_packed",
    "language_model.model.layers.*.block_sparse_moe.experts.*.w2.weight_scale",
    "language_model.model.layers.*.block_sparse_moe.experts.*.w3.weight_packed",
    "language_model.model.layers.*.block_sparse_moe.experts.*.w3.weight_scale",
];

/// Opt-in for CPU `from_pretrained` packed ingest. Default is refuse.
pub fn allow_mxfp4() -> bool {
    std::env::var("K3_ALLOW_MXFP4").as_deref() == Ok("1")
}

pub fn is_expert_packed(name: &str) -> bool {
    name.contains("block_sparse_moe.experts.") && name.ends_with(".weight_packed")
}

pub fn is_expert_scale(name: &str) -> bool {
    name.contains("block_sparse_moe.experts.") && name.ends_with(".weight_scale")
}

pub fn unpacked_name(packed_name: &str) -> Option<String> {
    packed_name
        .strip_suffix("weight_packed")
        .map(|p| format!("{p}weight"))
}

pub fn scale_name(packed_name: &str) -> Option<String> {
    packed_name
        .strip_suffix("weight_packed")
        .map(|p| format!("{p}weight_scale"))
}

/// Unpack a K3 expert pair through the DSV4 host function.
pub fn unpack_expert(packed: &[u8], scale: &[u8], n: usize, k: usize) -> Result<Vec<f32>> {
    dequant_nvfp4_e8m0_to_f32(packed, scale, n, k)
}

/// Same unpack as DSV4 GPU-upload host path (BF16 bits).
pub fn unpack_expert_bf16(packed: &[u8], scale: &[u8], n: usize, k: usize) -> Result<Vec<u16>> {
    dequant_nvfp4_e8m0_to_bf16(packed, scale, n, k)
}

#[derive(Default)]
pub(super) struct PackedSink {
    packed: HashMap<String, (Vec<usize>, Vec<u8>)>,
    scales: HashMap<String, (Vec<usize>, Vec<u8>)>,
}

impl PackedSink {
    pub(super) fn take(
        &mut self,
        name: &str,
        dtype: Dtype,
        shape: &[usize],
        data: &[u8],
    ) -> Result<bool> {
        if is_expert_packed(name) {
            if !allow_mxfp4() {
                bail!("S5 MXFP4 not this slice ({name})");
            }
            ensure!(
                dtype == Dtype::U8,
                "{name}: packed MXFP4 wants U8, got {dtype:?}"
            );
            self.packed
                .insert(name.to_string(), (shape.to_vec(), data.to_vec()));
            return Ok(true);
        }
        if is_expert_scale(name) {
            if !allow_mxfp4() {
                bail!("S5 MXFP4 not this slice ({name})");
            }
            ensure!(
                dtype == Dtype::U8,
                "{name}: E8M0 weight_scale wants U8, got {dtype:?}"
            );
            self.scales
                .insert(name.to_string(), (shape.to_vec(), data.to_vec()));
            return Ok(true);
        }
        Ok(false)
    }

    pub(super) fn finish(self, out: &mut HashMap<String, Vec<f32>>) -> Result<()> {
        for (name, (shape, data)) in &self.packed {
            let sname = scale_name(name).expect("weight_packed suffix");
            let (_sshape, sdata) = self
                .scales
                .get(&sname)
                .with_context(|| format!("{name}: missing DSV4-mapped E8M0 {sname}"))?;
            let (n, k) = packed_nk(shape)?;
            let vals = unpack_expert(data, sdata, n, k)?;
            let dest = unpacked_name(name).expect("weight_packed suffix");
            if out.insert(dest.clone(), vals).is_some() {
                bail!("duplicate unpacked {dest}");
            }
        }
        Ok(())
    }
}

fn packed_nk(shape: &[usize]) -> Result<(usize, usize)> {
    match *shape {
        [n, k2] => Ok((n, k2.checked_mul(2).expect("k packed"))),
        [len] => Ok((1, len.checked_mul(2).expect("k packed 1d"))),
        _ => bail!("MXFP4 packed shape {shape:?}: want [n, k/2] or [k/2]"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kimi_k3::cpu_weights::K3CpuModel;
    use crate::numeric::f32_to_bf16;
    use safetensors::tensor::{TensorView, serialize};
    use std::path::PathBuf;
    use std::process::Command;

    const TINY_CFG: &str = r#"{
      "model_type": "kimi_k3",
      "text_config": {
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "attn_res_block_size": 2,
        "first_k_dense_replace": 1,
        "head_dim": 2,
        "hidden_act": "situ",
        "hidden_size": 4,
        "intermediate_size": 4,
        "kv_lora_rank": 2,
        "latent_moe_use_norm": true,
        "linear_attn_config": {
          "full_attn_layers": [2],
          "head_dim": 2,
          "kda_layers": [1],
          "num_heads": 1,
          "short_conv_kernel_size": 4,
          "use_full_rank_gate": true
        },
        "mla_use_nope": true,
        "mla_use_output_gate": true,
        "model_type": "kimi_linear",
        "moe_intermediate_size": 4,
        "moe_renormalize": true,
        "moe_router_activation_func": "sigmoid",
        "num_attention_heads": 1,
        "num_experts": 2,
        "num_experts_per_token": 1,
        "num_hidden_layers": 2,
        "num_shared_experts": 1,
        "q_lora_rank": 4,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "routed_expert_hidden_size": 4,
        "topk_method": "noaux_tc",
        "v_head_dim": 2,
        "vocab_size": 8
      }
    }"#;

    fn pack_nibble(n: usize, k: usize, nib: u8) -> (Vec<u8>, Vec<usize>) {
        let total = n * k;
        let mut packed = vec![0u8; total / 2];
        for i in 0..total {
            if i.is_multiple_of(2) {
                packed[i / 2] |= nib & 0x0F;
            } else {
                packed[i / 2] |= (nib & 0x0F) << 4;
            }
        }
        (packed, vec![n, k / 2])
    }

    #[test]
    fn packed_classes_match_official_tsv() {
        let tsv = include_str!("../../../../docs/k3/official-weight-classes.tsv");
        let mut from_tsv: Vec<&str> = tsv
            .lines()
            .filter_map(|line| {
                let class = line.split('\t').next_back()?;
                (class.contains("experts.")
                    && (class.ends_with("weight_packed") || class.ends_with("weight_scale")))
                .then_some(class)
            })
            .collect();
        from_tsv.sort_unstable();
        let mut ours = PACKED_CLASSES.to_vec();
        ours.sort_unstable();
        assert_eq!(ours, from_tsv);
    }

    #[test]
    fn k3_names_map_to_dsv4_weight_and_scale() {
        let packed = "language_model.model.layers.12.block_sparse_moe.experts.7.w1.weight_packed";
        assert_eq!(
            unpacked_name(packed).as_deref(),
            Some("language_model.model.layers.12.block_sparse_moe.experts.7.w1.weight")
        );
        assert_eq!(
            scale_name(packed).as_deref(),
            Some("language_model.model.layers.12.block_sparse_moe.experts.7.w1.weight_scale")
        );
        // DSV4 pair is `.weight` + `.scale`; K3 uses `_packed` / `_scale`.
        assert_ne!(
            scale_name(packed).unwrap().as_str().rsplit('.').next(),
            Some("scale")
        );
    }

    #[test]
    fn synthetic_packed_round_trips_dsv4_e8m0() {
        // n=1, k=32 (DSV4 GROUP_SIZE). nibble 2 = E2M1 1.0; scale 127 = 2^0.
        let n = 1;
        let k = crate::mxfp4_e8m0::GROUP_SIZE;
        let (packed, _) = pack_nibble(n, k, 2);
        let scales = vec![127u8];
        let got = unpack_expert(&packed, &scales, n, k).unwrap();
        assert_eq!(got, vec![1.0; k]);
        let bf = unpack_expert_bf16(&packed, &scales, n, k).unwrap();
        let want: Vec<u16> = vec![1.0; k].into_iter().map(f32_to_bf16).collect();
        assert_eq!(bf, want);
    }

    #[test]
    fn flipped_e8m0_scale_diverges_from_bf16_reference() {
        let n = 1;
        let k = crate::mxfp4_e8m0::GROUP_SIZE;
        let (packed, _) = pack_nibble(n, k, 6); // E2M1 4.0
        let good_scale = vec![127u8];
        let mut bad_scale = good_scale.clone();
        bad_scale[0] ^= 0x01; // 127 → 126: 4.0 vs 2.0
        let good = unpack_expert_bf16(&packed, &good_scale, n, k).unwrap();
        let bad = unpack_expert_bf16(&packed, &bad_scale, n, k).unwrap();
        let ref_bf: Vec<u16> = unpack_expert(&packed, &good_scale, n, k)
            .unwrap()
            .into_iter()
            .map(f32_to_bf16)
            .collect();
        assert_eq!(good, ref_bf, "good unpack is the BF16 reference");
        assert_ne!(bad, ref_bf, "flipped E8M0 scale must diverge");
    }

    #[test]
    fn from_pretrained_allow_mxfp4_skips_s5_bail() {
        const THIS: &str = "kimi_k3::mxfp4::tests::from_pretrained_allow_mxfp4_skips_s5_bail";
        const MARKER: &str = "K3_MXFP4_ALLOW_CHILD";
        if std::env::var_os(MARKER).is_some() {
            let dir = scratch_packed_only();
            let err = K3CpuModel::from_pretrained(&dir).unwrap_err().to_string();
            let _ = std::fs::remove_dir_all(&dir);
            assert!(
                !err.contains("S5 MXFP4 not this slice"),
                "K3_ALLOW_MXFP4=1 must not use the S5 refuse: {err}"
            );
            return;
        }
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", THIS])
            .env(MARKER, "1")
            .env("K3_ALLOW_MXFP4", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "ALLOW child failed:\n{}",
            String::from_utf8_lossy(&output.stdout)
        );
    }

    fn scratch_packed_only() -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("k3-mxfp4-allow-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), TINY_CFG).unwrap();
        let (packed, _) = pack_nibble(1, 32, 2);
        let scale = vec![127u8];
        let pv = TensorView::new(Dtype::U8, vec![1, 16], &packed).unwrap();
        let sv = TensorView::new(Dtype::U8, vec![1], &scale).unwrap();
        let bytes = serialize(
            [
                (
                    "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed"
                        .to_string(),
                    pv,
                ),
                (
                    "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_scale"
                        .to_string(),
                    sv,
                ),
            ],
            None,
        )
        .unwrap();
        std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
        dir
    }
}
