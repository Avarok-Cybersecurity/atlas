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

/// Whether a tensor is an n-gram embedding TABLE — the huge
/// `*.ngram_embeddings.embedders.{i}.weight` blobs (~5.2 GB each on
/// LongCat-Flash-Lite, 12 of them).
///
/// These are deliberately not uploaded with the rest of the checkpoint: they
/// are served either by streaming per-table quantize-on-load or straight off
/// NVMe by the row cache, both of which read them in place. The small
/// `post_projs` are NOT matched — those are ordinary tensors.
pub fn is_ngram_table(name: &str) -> bool {
    name.contains("ngram_embeddings.embedders.") && name.ends_with(".weight")
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
