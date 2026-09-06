// SPDX-License-Identifier: AGPL-3.0-only

//! Tensor-NAME parsing: the numeric trailing-index sort, the n-gram table
//! predicate, and the expert-index extractor. Pure string work, which is
//! why it is unit-tested here rather than behind a loader.

pub(crate) fn split_trailing_index(name: &str) -> (String, u64) {
    let mut segs: Vec<&str> = name.split('.').collect();
    for i in (0..segs.len()).rev() {
        if let Ok(n) = segs[i].parse::<u64>() {
            segs.remove(i);
            return (segs.join("."), n);
        }
    }
    (name.to_string(), u64::MAX)
}

/// Whether a tensor is an n-gram embedding TABLE — the huge blobs that are
/// served off NVMe instead of being uploaded with the rest of the checkpoint.
///
/// TWO NAMING FAMILIES, because two model families ship this mechanism:
///
///   LongCat-Flash-Lite  `*.ngram_embeddings.embedders.{i}.weight`
///                       12 tables, ~5.2 GB each
///   Qwen3.8-Flash-Next  `*.ple.ple_embedding.ngram_embedding.shard_{i}.weight`
///                       128 shards of one logical table, 47.7 GB (FP8) to
///                       95.4 GB (BF16) in total
///
/// Matching is by TENSOR NAME rather than by file, deliberately: RadixArk
/// isolates the Qwen shards in dedicated `model-plefp8-*` files, but Inferact
/// buries all 128 inside one 95.4 GB `model-00001-of-00004.safetensors`. A
/// filename rule would work for one release and silently fail the other,
/// where "silently" means a 221 GB OOM pre-flight on a 121 GB box.
///
/// What must NOT match, because the loader needs these resident:
///   - LongCat's small `post_projs`
///   - Qwen's `ngram_embedding.weight_scale` (one BF16 scalar)
///   - Qwen's `ngram_heads_offsets` / `ngram_heads_vocab_sizes` (I64, 16 each)
pub fn is_ngram_table(name: &str) -> bool {
    if !name.ends_with(".weight") {
        return false;
    }
    name.contains("ngram_embeddings.embedders.") || name.contains("ngram_embedding.shard_")
}

#[cfg(test)]
mod ngram_defer_tests {
    use super::*;
    use crate::weights::{DeferredTensor, WeightDtype, WeightStore};

    #[test]
    fn ngram_table_predicate_matches_only_the_big_tables() {
        assert!(is_ngram_table("model.ngram_embeddings.embedders.0.weight"));
        assert!(is_ngram_table("model.ngram_embeddings.embedders.11.weight"));
        // The small projections are ordinary tensors and must still load.
        assert!(!is_ngram_table(
            "model.ngram_embeddings.post_projs.0.weight"
        ));
        assert!(!is_ngram_table("model.embed_tokens.weight"));
        assert!(!is_ngram_table(
            "model.layers.0.mlp.experts.3.gate_proj.weight"
        ));
    }

    /// Qwen3.8-Flash-Next stores ONE logical table as 128 `shard_{i}` tensors
    /// under a PLE block, rather than LongCat's 12 separate embedders.
    #[test]
    fn ngram_table_predicate_matches_the_qwen_ple_shards() {
        let base = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";
        assert!(is_ngram_table(&format!("{base}.shard_0.weight")));
        assert!(is_ngram_table(&format!("{base}.shard_127.weight")));

        // The per-table scalar scale (RadixArk's FP8 release) must stay
        // resident — the row cache needs it to dequantize.
        assert!(!is_ngram_table(&format!("{base}.weight_scale")));
        // The head range tables are 16 I64 values the loader reads directly.
        assert!(!is_ngram_table(
            "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets"
        ));
        assert!(!is_ngram_table(
            "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes"
        ));
        // The PLE block's own projections are ordinary tensors.
        assert!(!is_ngram_table(
            "model.language_model.layers.1.ple.key_proj.weight"
        ));
        assert!(!is_ngram_table(
            "model.language_model.layers.1.ple.value_proj.weight"
        ));
        assert!(!is_ngram_table(
            "model.language_model.layers.1.ple.conv1d.weight"
        ));
    }

    #[test]
    fn deferred_sorts_numerically_not_lexicographically() {
        let mut st = WeightStore::empty();
        for i in [0usize, 2, 10, 11, 9] {
            st.defer(
                format!("model.ngram_embeddings.embedders.{i}.weight"),
                DeferredTensor {
                    path: std::path::PathBuf::from("x"),
                    offset: i as u64,
                    shape: vec![1, 1],
                    dtype: WeightDtype::BF16,
                },
            );
        }
        let got: Vec<u64> = st.deferred_sorted().iter().map(|(_, d)| d.offset).collect();
        assert_eq!(got, vec![0, 2, 9, 10, 11], "table order must be numeric");
    }
}

/// Parse expert index from tensor name (e.g. "model.layers.3.mlp.experts.42.gate_proj.weight" → 42).
/// `(layer, expert)` from a routed-expert tensor name, e.g.
/// `model.layers.3.mlp.experts.42.gate_proj.weight` -> `(3, 42)`.
///
/// Both halves, because boot-time expert loading is PER LAYER: a category
/// keeps expert 42 in one layer and not in another, so the expert index
/// alone cannot decide whether a tensor is needed. `None` when either part
/// is absent, which is every non-routed-expert tensor — including `mtp.*`,
/// whose experts belong to the drafter and are always loaded.
pub fn parse_layer_expert(name: &str) -> Option<(usize, usize)> {
    if name.starts_with("mtp.") {
        return None;
    }
    let parts: Vec<&str> = name.split('.').collect();
    let mut layer = None;
    let mut expert = None;
    for (i, part) in parts.iter().enumerate() {
        if *part == "layers" && i + 1 < parts.len() {
            layer = parts[i + 1].parse::<usize>().ok();
        }
        if *part == "experts" && i + 1 < parts.len() {
            expert = parts[i + 1].parse::<usize>().ok();
        }
    }
    Some((layer?, expert?))
}

pub fn parse_expert_index(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "experts" && i + 1 < parts.len() {
            return parts[i + 1].parse().ok();
        }
    }
    None
}

#[cfg(test)]
mod from_str_tests {
    use crate::weights::WeightDtype;

    #[test]
    fn from_safetensors_str_matches_disk_mapping() {
        // The RDMA weight peer publishes these raw header strings; the client
        // must resolve them to the exact WeightDtype the disk loaders use, else
        // byte_size/shape diverge and logits break. Locks the closed mapping.
        use WeightDtype::*;
        for (s, want) in [
            ("F32", FP32),
            ("BF16", BF16),
            ("U8", UInt8),
            ("I8", UInt8), // packed NVFP4 raw container
            ("F8_E4M3", FP8E4M3),
            ("F8_E8M0", FP8E8M0),
            ("I64", Int64),
        ] {
            assert_eq!(
                WeightDtype::from_safetensors_str(s).unwrap(),
                want,
                "dtype {s}"
            );
        }
        // F16 is converted to BF16 at disk-load; a store (and therefore a
        // peer manifest) can never contain it, so the wire mapping rejects it.
        assert!(WeightDtype::from_safetensors_str("F16").is_err());
        assert!(WeightDtype::from_safetensors_str("bogus").is_err());
    }

    #[test]
    fn f16_bytes_convert_to_bf16_via_f32() {
        use half::{bf16, f16};
        // Cover sign, exact powers of two, a value needing mantissa rounding
        // (f16 has 10 mantissa bits, bf16 only 7), f16 max, and a subnormal.
        let vals = [0.0f32, 1.0, -1.5, 0.1, 65504.0, -6.1035156e-5];
        let src: Vec<u8> = vals
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect();
        let out = crate::weights::f16_to_bf16_bytes(&src);
        assert_eq!(out.len(), src.len());
        for (i, v) in vals.iter().enumerate() {
            let got = bf16::from_le_bytes([out[2 * i], out[2 * i + 1]]);
            let want = bf16::from_f32(f16::from_f32(*v).to_f32());
            assert_eq!(got, want, "value {v}");
        }
    }
}
