// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: real-file oracle for the Qwen3.5-family GGUF `ModelConfig`
//! (`general.architecture = qwen35`, e.g. `Qwen3.8-27B-UD-IQ4_XS.gguf`).
//!
//! CPU only: it reads the header and the tensor table and never touches tensor
//! data, so it runs without a GPU. It is gated on an environment variable so CI
//! and machines without the file skip it (the tests pass without doing anything):
//!
//!   ATLAS_QWEN35_GGUF_DIR=~/models/Qwen3.8-27B-GGUF \
//!     cargo test -p spark-runtime qwen35_real_file -- --nocapture
//!
//! The checks are generic (any `qwen35` file) plus the facts of the 27B read out
//! of its header on 2026-11-13, applied when the file has 64 trunk layers.

use std::path::PathBuf;

use avarok_core::config::LayerType;

use super::container::GgufFile;
use super::names::{GgufName, mtp_layer, translate_with_mtp};
use super::value_transform::gdn_dims;
use crate::weights::{config_from_gguf_dir, find_gguf};

/// The directory named by `ATLAS_QWEN35_GGUF_DIR`, or `None` to skip.
fn dir() -> Option<PathBuf> {
    let d = PathBuf::from(std::env::var_os("ATLAS_QWEN35_GGUF_DIR")?);
    if !d.is_dir() {
        panic!("ATLAS_QWEN35_GGUF_DIR={} is not a directory", d.display());
    }
    Some(d)
}

fn dims(g: &GgufFile, name: &str) -> Vec<usize> {
    g.tensor(name)
        .unwrap_or_else(|| panic!("tensor {name} missing from the GGUF"))
        .dims
        .clone()
}

#[test]
fn qwen35_real_file_config_matches_the_tensor_table() {
    let Some(d) = dir() else {
        eprintln!("ATLAS_QWEN35_GGUF_DIR unset: skipping the real-file qwen35 config check");
        return;
    };
    let c = config_from_gguf_dir(&d).expect("build ModelConfig from the GGUF metadata");

    let path = find_gguf(&d).expect("find the .gguf");
    let file = std::fs::File::open(&path).expect("open the .gguf");
    // SAFETY: read-only mmap of an existing file for the duration of the test.
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file).expect("mmap") };
    let g = GgufFile::parse(&mmap).expect("parse the GGUF container");

    assert_eq!(g.get_str("general.architecture"), Some("qwen35"));
    assert_eq!(c.model_type, "qwen3_5");
    assert!(c.is_qwen35_dense());
    assert!(c.attn_gated);

    // Block layout: trunk plus the MTP block(s), in the order the name map expects.
    let mtp = c.mtp_num_hidden_layers;
    let block_count = g.get_u64("qwen35.block_count").unwrap() as usize;
    assert_eq!(c.num_hidden_layers + mtp, block_count);
    assert_eq!(c.layer_types.len(), c.num_hidden_layers);
    let head_layer = mtp_layer(g.tensors.iter().map(|t| t.name.as_str()));
    assert_eq!(
        head_layer,
        (mtp > 0).then_some(c.num_hidden_layers),
        "the MTP block must sit right after the trunk"
    );

    let h = c.hidden_size;
    let q_dim = c.num_attention_heads * c.head_dim;
    let kv_dim = c.num_key_value_heads * c.head_dim;
    let nk = c.linear_num_key_heads;
    let nv = c.linear_num_value_heads;
    let kd = c.linear_key_head_dim;
    let vd = c.linear_value_head_dim;
    let qkv_rows = 2 * nk * kd + nv * vd;

    // The GDN geometry the value transforms use must be the geometry the config carries.
    let gd = gdn_dims(&g, "qwen35").expect("gdn dims from the ssm keys");
    assert_eq!(
        (
            gd.num_k_heads,
            gd.num_v_heads,
            gd.key_head_dim,
            gd.value_head_dim
        ),
        (nk, nv, kd, vd)
    );

    for (i, ty) in c.layer_types.iter().enumerate() {
        let t = |s: &str| format!("blk.{i}.{s}");
        assert_eq!(dims(&g, &t("attn_norm.weight")), vec![h]);
        assert_eq!(dims(&g, &t("post_attention_norm.weight")), vec![h]);
        match ty {
            LayerType::FullAttention => {
                // Gated Q: Q and its output gate interleaved, twice the Q width.
                assert_eq!(dims(&g, &t("attn_q.weight")), vec![h, 2 * q_dim]);
                assert_eq!(dims(&g, &t("attn_k.weight")), vec![h, kv_dim]);
                assert_eq!(dims(&g, &t("attn_v.weight")), vec![h, kv_dim]);
                assert_eq!(dims(&g, &t("attn_output.weight")), vec![q_dim, h]);
                assert_eq!(dims(&g, &t("attn_q_norm.weight")), vec![c.head_dim]);
                assert_eq!(dims(&g, &t("attn_k_norm.weight")), vec![c.head_dim]);
                assert!(
                    g.tensor(&t("ssm_a")).is_none(),
                    "layer {i} also has GDN tensors"
                );
            }
            LayerType::LinearAttention => {
                assert_eq!(dims(&g, &t("attn_qkv.weight")), vec![h, qkv_rows]);
                assert_eq!(dims(&g, &t("attn_gate.weight")), vec![h, nv * vd]);
                assert_eq!(
                    dims(&g, &t("ssm_conv1d.weight")),
                    vec![c.linear_conv_kernel_dim, qkv_rows]
                );
                assert_eq!(dims(&g, &t("ssm_a")), vec![nv]);
                assert_eq!(dims(&g, &t("ssm_dt.bias")), vec![nv]);
                assert_eq!(dims(&g, &t("ssm_alpha.weight")), vec![h, nv]);
                assert_eq!(dims(&g, &t("ssm_beta.weight")), vec![h, nv]);
                assert_eq!(dims(&g, &t("ssm_norm.weight")), vec![vd]);
                assert_eq!(dims(&g, &t("ssm_out.weight")), vec![nv * vd, h]);
                assert!(
                    g.tensor(&t("attn_q.weight")).is_none(),
                    "layer {i} also has attention tensors"
                );
            }
            other => panic!("layer {i}: unexpected layer type {other:?}"),
        }
        assert_eq!(
            dims(&g, &t("ffn_gate.weight")),
            vec![h, c.intermediate_size]
        );
        assert_eq!(
            dims(&g, &t("ffn_down.weight")),
            vec![c.intermediate_size, h]
        );
    }

    // Embedding and head.
    assert_eq!(dims(&g, "token_embd.weight"), vec![h, c.vocab_size]);
    assert_eq!(g.tensor("output.weight").is_some(), !c.tie_word_embeddings);
    if let Some(n) = g.arr_len("tokenizer.ggml.tokens") {
        assert_eq!(n, c.vocab_size, "tokenizer count vs embedding rows");
    }

    // Every tensor of the file has a name the loader can place.
    let unmapped: Vec<&str> = g
        .tensors
        .iter()
        .filter(|t| translate_with_mtp(&t.name, "qwen35", head_layer).is_none())
        .map(|t| t.name.as_str())
        .collect();
    assert!(
        unmapped.is_empty(),
        "tensors with no name mapping: {unmapped:?}"
    );
    let dropped = g
        .tensors
        .iter()
        .filter(|t| {
            matches!(
                translate_with_mtp(&t.name, "qwen35", head_layer),
                Some(GgufName::Drop)
            )
        })
        .count();
    eprintln!(
        "qwen35 real file: {} tensors, {} dropped, trunk layers {}, mtp {}, vocab {}",
        g.tensors.len(),
        dropped,
        c.num_hidden_layers,
        mtp,
        c.vocab_size
    );

    // The 27B's own facts.
    if c.num_hidden_layers == 64 {
        assert_eq!(
            (h, c.intermediate_size, c.vocab_size),
            (5120, 17408, 248320)
        );
        assert_eq!(
            (c.num_attention_heads, c.num_key_value_heads, c.head_dim),
            (24, 4, 256)
        );
        assert_eq!(
            (nk, nv, kd, vd, c.linear_conv_kernel_dim),
            (16, 48, 128, 128, 4)
        );
        assert_eq!(c.full_attention_interval, 4);
        assert_eq!(
            c.layer_types
                .iter()
                .filter(|t| **t == LayerType::FullAttention)
                .count(),
            16
        );
        assert_eq!(c.mtp_num_hidden_layers, 1);
        assert_eq!(c.mrope_section, [11, 11, 10]);
        assert!(c.mrope_interleaved);
        assert!((c.partial_rotary_factor - 0.25).abs() < 1e-12);
        assert!((c.rope_theta - 1.0e7).abs() < 1e-3);
        assert_eq!((c.eos_token_id, c.bos_token_id), (248046, 248044));
        assert!(!c.tie_word_embeddings);
    }
}
