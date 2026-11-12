// SPDX-License-Identifier: AGPL-3.0-only

//! llama.cpp's `/tokenize` and `/detokenize` request and response shapes.
//!
//! Atlas already serves both routes with its own shapes (`prompt` or
//! `messages` in, `{tokens, count}` out; `{tokens}` in, `{text}` out). The
//! handlers in `misc_handlers.rs` keep those, and switch to this module when a
//! `/tokenize` body carries `content`, the llama.cpp field. `/detokenize`
//! takes the same `tokens` field in both dialects, so its answer carries both
//! `content` (llama.cpp) and `text` (Atlas), and no client has to pick.

use serde_json::{Map, Value, json};

use super::codec::TextCodec;
use super::error::CompatError;

/// One element of a `/tokenize` `content` array: text, or a token id that is
/// passed through unchanged.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Segment {
    Text(String),
    Id(u32),
}

/// A parsed llama.cpp `/tokenize` request.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TokenizeRequest {
    pub segments: Vec<Segment>,
    pub add_special: bool,
    pub with_pieces: bool,
}

/// True when the body is a llama.cpp tokenize request (it has `content`).
pub(crate) fn is_llama_shape(body: &Map<String, Value>) -> bool {
    body.contains_key("content")
}

fn bool_field(body: &Map<String, Value>, key: &str, default: bool) -> Result<bool, CompatError> {
    match body.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => Err(CompatError::invalid(format!("'{key}' must be a boolean"))),
    }
}

/// Parse `content` (a string, or an array mixing strings and token ids),
/// `add_special`, `parse_special` and `with_pieces`.
pub(crate) fn parse_tokenize(body: &Map<String, Value>) -> Result<TokenizeRequest, CompatError> {
    // Atlas parses special-token text (`<|im_start|>`) into the special id
    // whatever the flag says, which is llama.cpp's default. A request that
    // asks for plain-text treatment would get different tokens, so refuse it.
    if !bool_field(body, "parse_special", true)? {
        return Err(CompatError::unsupported(
            "parse_special",
            "special-token text is always parsed into its token id; \
             only parse_special=true can be honored",
        ));
    }
    let add_special = bool_field(body, "add_special", false)?;
    let with_pieces = bool_field(body, "with_pieces", false)?;
    let segments = match body.get("content") {
        Some(Value::String(s)) => vec![Segment::Text(s.clone())],
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| match item {
                Value::String(s) => Ok(Segment::Text(s.clone())),
                Value::Number(n) => n
                    .as_u64()
                    .and_then(|v| u32::try_from(v).ok())
                    .map(Segment::Id)
                    .ok_or_else(|| {
                        CompatError::invalid("'content' token ids must be non-negative integers")
                    }),
                _ => Err(CompatError::invalid(
                    "'content' array elements must be strings or token ids",
                )),
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => {
            return Err(CompatError::invalid(
                "'content' must be a string or an array of strings and token ids",
            ));
        }
    };
    Ok(TokenizeRequest {
        segments,
        add_special,
        with_pieces,
    })
}

/// Tokenize the segments. As in llama.cpp, `add_special` applies to the first
/// segment only, and only when that segment is text.
pub(crate) fn tokenize_segments(
    codec: &dyn TextCodec,
    segments: &[Segment],
    add_special: bool,
) -> Result<Vec<u32>, CompatError> {
    let mut out = Vec::new();
    for (i, seg) in segments.iter().enumerate() {
        match seg {
            Segment::Text(t) => {
                let ids = codec
                    .encode_text(t, add_special && i == 0)
                    .map_err(|e| CompatError::invalid(format!("tokenization error: {e}")))?;
                out.extend(ids);
            }
            Segment::Id(id) => out.push(*id),
        }
    }
    Ok(out)
}

/// A token's piece as llama.cpp reports it: the text when the bytes are valid
/// UTF-8, otherwise the array of byte values (half of a multi-byte character).
pub(crate) fn piece_value(codec: &dyn TextCodec, id: u32) -> Value {
    let bytes = codec.token_bytes(id);
    match String::from_utf8(bytes) {
        Ok(s) => Value::String(s),
        Err(e) => Value::Array(e.into_bytes().into_iter().map(|b| json!(b)).collect()),
    }
}

/// The `/tokenize` response body: `{"tokens": [...]}`, where each element is
/// `{"id", "piece"}` when `with_pieces` is set.
pub(crate) fn tokenize_response(
    codec: &dyn TextCodec,
    req: &TokenizeRequest,
) -> Result<Value, CompatError> {
    let ids = tokenize_segments(codec, &req.segments, req.add_special)?;
    let tokens: Vec<Value> = if req.with_pieces {
        ids.iter()
            .map(|&id| json!({"id": id, "piece": piece_value(codec, id)}))
            .collect()
    } else {
        ids.iter().map(|&id| json!(id)).collect()
    };
    Ok(json!({ "tokens": tokens }))
}

/// The `/detokenize` response body. `content` is llama.cpp's field (special
/// tokens are written out, as `llama-server` does); `text` is the field Atlas
/// has always returned (special tokens dropped).
pub(crate) fn detokenize_response(
    codec: &dyn TextCodec,
    ids: &[u32],
) -> Result<Value, CompatError> {
    let content = codec
        .decode_text(ids, false)
        .map_err(|e| CompatError::invalid(format!("detokenization error: {e}")))?;
    let text = codec
        .decode_text(ids, true)
        .map_err(|e| CompatError::invalid(format!("detokenization error: {e}")))?;
    Ok(json!({ "content": content, "text": text }))
}

#[cfg(test)]
mod tests {
    use super::super::toy_codec::{BOS, EOT, ToyCodec};
    use super::*;

    fn body(v: Value) -> Map<String, Value> {
        v.as_object().cloned().expect("object")
    }

    #[test]
    fn content_key_selects_the_llama_dialect() {
        assert!(is_llama_shape(&body(json!({"content": "x"}))));
        assert!(!is_llama_shape(&body(json!({"prompt": "x"}))));
        assert!(!is_llama_shape(&body(json!({"messages": []}))));
    }

    #[test]
    fn plain_string_defaults() {
        let r = parse_tokenize(&body(json!({"content": "hi"}))).unwrap();
        assert_eq!(r.segments, vec![Segment::Text("hi".into())]);
        assert!(!r.add_special && !r.with_pieces);
        let out = tokenize_response(&ToyCodec, &r).unwrap();
        assert_eq!(out, json!({"tokens": [204, 205]}));
    }

    #[test]
    fn add_special_prepends_bos_to_a_leading_text_segment_only() {
        let r = parse_tokenize(&body(json!({"content": "a", "add_special": true}))).unwrap();
        assert_eq!(
            tokenize_response(&ToyCodec, &r).unwrap(),
            json!({"tokens": [BOS, 197]})
        );
        // Mixed: a leading token id means no BOS for the text after it.
        let r = parse_tokenize(&body(json!({"content": [7, "a"], "add_special": true}))).unwrap();
        assert_eq!(
            tokenize_response(&ToyCodec, &r).unwrap(),
            json!({"tokens": [7, 197]})
        );
        // Mixed: BOS goes on the first text segment, not on later ones.
        let r = parse_tokenize(&body(json!({"content": ["a", "b"], "add_special": true}))).unwrap();
        assert_eq!(
            tokenize_response(&ToyCodec, &r).unwrap(),
            json!({"tokens": [BOS, 197, 198]})
        );
    }

    #[test]
    fn special_token_text_is_parsed() {
        let r = parse_tokenize(&body(json!({"content": "<|eot|>"}))).unwrap();
        assert_eq!(
            tokenize_response(&ToyCodec, &r).unwrap(),
            json!({"tokens": [EOT]})
        );
    }

    #[test]
    fn parse_special_false_is_a_400_not_silently_ignored() {
        let err = parse_tokenize(&body(json!({"content": "x", "parse_special": false})))
            .expect_err("must reject");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("parse_special"));
        // True is the default and is fine.
        assert!(parse_tokenize(&body(json!({"content": "x", "parse_special": true}))).is_ok());
    }

    #[test]
    fn with_pieces_gives_text_or_byte_arrays() {
        // The euro sign is 3 byte tokens: each piece alone is not UTF-8.
        let r =
            parse_tokenize(&body(json!({"content": "a\u{20AC}", "with_pieces": true}))).unwrap();
        let out = tokenize_response(&ToyCodec, &r).unwrap();
        let toks = out["tokens"].as_array().unwrap();
        assert_eq!(toks[0], json!({"id": 197, "piece": "a"}));
        assert_eq!(toks[1], json!({"id": 100 + 0xE2, "piece": [0xE2]}));
        assert_eq!(toks[2]["piece"], json!([0x82]));
        assert_eq!(toks[3]["piece"], json!([0xAC]));
    }

    #[test]
    fn malformed_content_is_a_400() {
        for bad in [
            json!({"content": 5}),
            json!({"content": null}),
            json!({"content": [true]}),
            json!({"content": [-1]}),
            json!({"content": [1.5]}),
            json!({"content": [4294967296u64]}),
            json!({"content": "x", "add_special": "yes"}),
        ] {
            let err = parse_tokenize(&body(bad.clone())).expect_err(&bad.to_string());
            assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST, "{bad}");
        }
    }

    #[test]
    fn detokenize_carries_both_dialect_fields() {
        let ids = [197, EOT, 198];
        let out = detokenize_response(&ToyCodec, &ids).unwrap();
        assert_eq!(out["content"], "a<|eot|>b", "llama.cpp writes specials out");
        assert_eq!(out["text"], "ab", "Atlas's own field drops them");
    }

    #[test]
    fn detokenize_of_nothing_is_empty_text() {
        let out = detokenize_response(&ToyCodec, &[]).unwrap();
        assert_eq!(out, json!({"content": "", "text": ""}));
    }
}
