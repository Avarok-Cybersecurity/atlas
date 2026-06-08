// SPDX-License-Identifier: AGPL-3.0-only
//
// HuggingFace metadata detection — port of `class HFTokenizerAnalyzer`
// and `TokenizerInfo::Impl::DetectMetadataFromHF` in
// `cpp/tokenizer_info.cc`.
//
// Given the JSON text of a HuggingFace `tokenizer.json`, infer:
//   * the `VocabType` (from the `decoder` field), and
//   * whether the tokenizer adds a prefix space (from the `normalizer`
//     or `pre_tokenizer` field).
//
// The C++ uses picojson; we use `serde_json`. On any structural failure
// the C++ logs a warning and falls back to `VocabType::RAW` — we mirror
// that fall-back, returning the metadata rather than an error.

use serde_json::Value;

use crate::tokenizer::vocab_type::VocabType;

/// Metadata inferred from a HuggingFace `tokenizer.json`.
///
/// Mirrors the JSON object produced by C++ `DetectMetadataFromHF`,
/// which serializes `{"vocab_type": <int>, "add_prefix_space": <bool>}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HfMetadata {
    /// The detected vocabulary type (`Raw` on detection failure).
    pub vocab_type: VocabType,
    /// Whether the tokenizer prepends a space to the first token.
    pub add_prefix_space: bool,
}

impl HfMetadata {
    /// Serialize to the compact JSON form C++ `DetectMetadataFromHF`
    /// produces: `{"vocab_type":N,"add_prefix_space":bool}`.
    pub fn to_json(self) -> String {
        format!(
            "{{\"vocab_type\":{},\"add_prefix_space\":{}}}",
            self.vocab_type.as_int(),
            self.add_prefix_space
        )
    }
}

/// Detect [`HfMetadata`] from the raw text of a `tokenizer.json`.
///
/// Returns `Err` only when `backend_str` is not parseable JSON or is not
/// a JSON object — faithful to the C++ `XGRAMMAR_CHECK(err.empty() &&
/// v.is<object>())`. A well-formed object that simply lacks the expected
/// fields yields `HfMetadata { vocab_type: Raw, add_prefix_space: false }`.
pub fn detect_metadata_from_hf(backend_str: &str) -> Result<HfMetadata, String> {
    let value: Value = serde_json::from_str(backend_str)
        .map_err(|e| format!("Failed to parse JSON object: {e}"))?;
    let obj = value
        .as_object()
        .ok_or_else(|| "Failed to parse JSON object: not an object".to_string())?;

    Ok(HfMetadata {
        vocab_type: detect_vocab_type(obj),
        add_prefix_space: detect_add_prefix_space(obj),
    })
}

type Obj = serde_json::Map<String, Value>;

/// Detect the vocabulary type from the `decoder` field.
///
/// Finds `{"type": "ByteFallback"}` or `{"type": "ByteLevel"}` in the
/// `decoder` field (descending into a `Sequence` decoder's `decoders`
/// array). Any structural mismatch falls back to `VocabType::Raw`.
fn detect_vocab_type(obj: &Obj) -> VocabType {
    let decoder = match obj.get("decoder").and_then(Value::as_object) {
        Some(d) => d,
        None => return VocabType::Raw,
    };
    let top_type = match decoder.get("type").and_then(Value::as_str) {
        Some(t) => t,
        None => return VocabType::Raw,
    };

    // A `Sequence` decoder nests a `decoders` array; otherwise the
    // single decoder object is examined directly.
    let decoders: Vec<&Value> = if top_type == "Sequence" {
        match decoder.get("decoders").and_then(Value::as_array) {
            Some(arr) => arr.iter().collect(),
            None => return VocabType::Raw,
        }
    } else {
        vec![obj.get("decoder").expect("decoder present above")]
    };

    for d in decoders {
        let d_obj = match d.as_object() {
            Some(o) => o,
            None => return VocabType::Raw,
        };
        let ty = match d_obj.get("type").and_then(Value::as_str) {
            Some(t) => t,
            None => return VocabType::Raw,
        };
        match ty {
            "ByteLevel" => return VocabType::ByteLevel,
            "ByteFallback" => return VocabType::ByteFallback,
            _ => {}
        }
    }
    VocabType::Raw
}

/// Detect whether the tokenizer adds a prefix space: true if either a
/// `Prepend` normalizer with `▁`, or a `Metaspace` pre-tokenizer with
/// `prepend_scheme` of `always`/`first` is present.
fn detect_add_prefix_space(obj: &Obj) -> bool {
    detect_prepend_normalizer(obj) || detect_metaspace_pretokenizer(obj)
}

/// Find `{"type": "Prepend", "prepend": "▁"}` in the `normalizer` field
/// (descending into a `Sequence` normalizer's `normalizers` array).
fn detect_prepend_normalizer(obj: &Obj) -> bool {
    let normalizer_value = match obj.get("normalizer") {
        Some(v) if v.is_object() => v,
        _ => return false,
    };
    let normalizer = normalizer_value.as_object().expect("checked is_object");
    let top_type = match normalizer.get("type").and_then(Value::as_str) {
        Some(t) => t,
        None => return false,
    };

    let normalizers: Vec<&Value> = if top_type == "Sequence" {
        match normalizer.get("normalizers").and_then(Value::as_array) {
            Some(arr) => arr.iter().collect(),
            None => return false,
        }
    } else {
        vec![normalizer_value]
    };

    for n in normalizers {
        let n_obj = match n.as_object() {
            Some(o) => o,
            None => continue,
        };
        if n_obj.get("type").and_then(Value::as_str) != Some("Prepend") {
            continue;
        }
        if n_obj.get("prepend").and_then(Value::as_str) == Some("\u{2581}") {
            return true;
        }
    }
    false
}

/// Find `pre_tokenizer: {"type": "Metaspace", "prepend_scheme":
/// "always" | "first"}`.
fn detect_metaspace_pretokenizer(obj: &Obj) -> bool {
    let pre = match obj.get("pre_tokenizer").and_then(Value::as_object) {
        Some(p) => p,
        None => return false,
    };
    let ty = pre.get("type").and_then(Value::as_str);
    let scheme = pre.get("prepend_scheme").and_then(Value::as_str);
    match (ty, scheme) {
        (Some("Metaspace"), Some(s)) => s == "always" || s == "first",
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_json_is_err() {
        assert!(detect_metadata_from_hf("{not json").is_err());
    }

    #[test]
    fn non_object_json_is_err() {
        assert!(detect_metadata_from_hf("[1,2,3]").is_err());
        assert!(detect_metadata_from_hf("42").is_err());
    }

    #[test]
    fn empty_object_falls_back_to_raw() {
        let m = detect_metadata_from_hf("{}").unwrap();
        assert_eq!(m.vocab_type, VocabType::Raw);
        assert!(!m.add_prefix_space);
    }

    #[test]
    fn byte_level_decoder_detected() {
        let json = r#"{"decoder":{"type":"ByteLevel"}}"#;
        let m = detect_metadata_from_hf(json).unwrap();
        assert_eq!(m.vocab_type, VocabType::ByteLevel);
    }

    #[test]
    fn byte_fallback_decoder_detected() {
        let json = r#"{"decoder":{"type":"ByteFallback"}}"#;
        let m = detect_metadata_from_hf(json).unwrap();
        assert_eq!(m.vocab_type, VocabType::ByteFallback);
    }

    #[test]
    fn sequence_decoder_descends() {
        let json = r#"{"decoder":{"type":"Sequence","decoders":[
            {"type":"Replace"},{"type":"ByteFallback"},{"type":"Fuse"}]}}"#;
        let m = detect_metadata_from_hf(json).unwrap();
        assert_eq!(m.vocab_type, VocabType::ByteFallback);
    }

    #[test]
    fn sequence_decoder_byte_level_inside() {
        let json = r#"{"decoder":{"type":"Sequence","decoders":[
            {"type":"ByteLevel","add_prefix_space":true}]}}"#;
        let m = detect_metadata_from_hf(json).unwrap();
        assert_eq!(m.vocab_type, VocabType::ByteLevel);
    }

    #[test]
    fn unknown_decoder_is_raw() {
        let json = r#"{"decoder":{"type":"Replace"}}"#;
        let m = detect_metadata_from_hf(json).unwrap();
        assert_eq!(m.vocab_type, VocabType::Raw);
    }

    #[test]
    fn sequence_missing_decoders_is_raw() {
        let json = r#"{"decoder":{"type":"Sequence"}}"#;
        let m = detect_metadata_from_hf(json).unwrap();
        assert_eq!(m.vocab_type, VocabType::Raw);
    }

    #[test]
    fn prepend_normalizer_sets_prefix_space() {
        let json = r#"{"normalizer":{"type":"Prepend","prepend":"▁"}}"#;
        let m = detect_metadata_from_hf(json).unwrap();
        assert!(m.add_prefix_space);
    }

    #[test]
    fn prepend_normalizer_wrong_marker_no_prefix() {
        let json = r#"{"normalizer":{"type":"Prepend","prepend":" "}}"#;
        let m = detect_metadata_from_hf(json).unwrap();
        assert!(!m.add_prefix_space);
    }

    #[test]
    fn sequence_normalizer_descends() {
        let json = r#"{"normalizer":{"type":"Sequence","normalizers":[
            {"type":"Strip"},{"type":"Prepend","prepend":"▁"}]}}"#;
        let m = detect_metadata_from_hf(json).unwrap();
        assert!(m.add_prefix_space);
    }

    #[test]
    fn metaspace_pretokenizer_always() {
        let json = r#"{"pre_tokenizer":{"type":"Metaspace","prepend_scheme":"always"}}"#;
        let m = detect_metadata_from_hf(json).unwrap();
        assert!(m.add_prefix_space);
    }

    #[test]
    fn metaspace_pretokenizer_first() {
        let json = r#"{"pre_tokenizer":{"type":"Metaspace","prepend_scheme":"first"}}"#;
        let m = detect_metadata_from_hf(json).unwrap();
        assert!(m.add_prefix_space);
    }

    #[test]
    fn metaspace_pretokenizer_never_no_prefix() {
        let json = r#"{"pre_tokenizer":{"type":"Metaspace","prepend_scheme":"never"}}"#;
        let m = detect_metadata_from_hf(json).unwrap();
        assert!(!m.add_prefix_space);
    }

    #[test]
    fn to_json_format() {
        let m = HfMetadata {
            vocab_type: VocabType::ByteLevel,
            add_prefix_space: false,
        };
        assert_eq!(m.to_json(), r#"{"vocab_type":2,"add_prefix_space":false}"#);
        let m2 = HfMetadata {
            vocab_type: VocabType::ByteFallback,
            add_prefix_space: true,
        };
        assert_eq!(m2.to_json(), r#"{"vocab_type":1,"add_prefix_space":true}"#);
    }

    #[test]
    fn qwen_style_byte_level_with_metaspace_absent() {
        // Qwen/MiniMax: ByteLevel decoder, no prefix space (F68-critical).
        let json = r#"{"decoder":{"type":"ByteLevel"},
            "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false}}"#;
        let m = detect_metadata_from_hf(json).unwrap();
        assert_eq!(m.vocab_type, VocabType::ByteLevel);
        assert!(!m.add_prefix_space);
    }
}
