// SPDX-License-Identifier: AGPL-3.0-only

//! The two `/completion` inputs that need the tokenizer: `prompt` and
//! `logit_bias`.

use serde_json::Value;

use super::error::CompatError;
use super::fields::field_err;
use super::params::ParseContext;
use super::tokenize::{Segment, tokenize_segments};

/// `logit_bias`: `[[token, bias], ...]` or `{"token": bias}`, where a token is
/// an id or a string (every token of its tokenization) and a bias is a number
/// or `false` (never produce).
pub(super) fn parse_logit_bias(
    v: &Value,
    ctx: &ParseContext<'_>,
) -> Result<Vec<(u32, f32)>, CompatError> {
    let bias_of = |b: &Value| -> Result<f32, CompatError> {
        match b {
            Value::Number(n) => Ok(n.as_f64().unwrap_or(0.0) as f32),
            Value::Bool(false) => Ok(f32::NEG_INFINITY),
            _ => Err(field_err("logit_bias", "a bias must be a number or false")),
        }
    };
    let ids_of = |t: &str| -> Result<Vec<u32>, CompatError> {
        match t.parse::<i64>() {
            Ok(id) => token_in_vocab(id, ctx).map(|i| vec![i]),
            Err(_) => ctx
                .codec
                .encode_text(t, false)
                .map_err(|e| field_err("logit_bias", &e)),
        }
    };
    let mut out = Vec::new();
    match v {
        Value::Array(pairs) => {
            for p in pairs {
                let (tok, bias) = match p.as_array().map(Vec::as_slice) {
                    Some([t, b]) => (t, b),
                    _ => return Err(field_err("logit_bias", "each entry must be [token, bias]")),
                };
                let bias = bias_of(bias)?;
                let ids = match tok {
                    Value::Number(n) => {
                        let id = n.as_i64().ok_or_else(|| {
                            field_err("logit_bias", "a token id must be an integer")
                        })?;
                        vec![token_in_vocab(id, ctx)?]
                    }
                    Value::String(s) => ctx
                        .codec
                        .encode_text(s, false)
                        .map_err(|e| field_err("logit_bias", &e))?,
                    _ => return Err(field_err("logit_bias", "a token must be an id or a string")),
                };
                out.extend(ids.into_iter().map(|i| (i, bias)));
            }
        }
        Value::Object(map) => {
            for (k, b) in map {
                let bias = bias_of(b)?;
                out.extend(ids_of(k)?.into_iter().map(|i| (i, bias)));
            }
        }
        _ => return Err(field_err("logit_bias", "must be an array or an object")),
    }
    Ok(out)
}

fn token_in_vocab(id: i64, ctx: &ParseContext<'_>) -> Result<u32, CompatError> {
    u32::try_from(id)
        .ok()
        .filter(|&i| (i as usize) < ctx.codec.vocab_size())
        .ok_or_else(|| {
            CompatError::invalid(format!(
                "token id {id} is out of range (vocabulary size {})",
                ctx.codec.vocab_size()
            ))
        })
}

/// Turn a `prompt` value that is one prompt into segments.
fn segments_of(v: &Value) -> Result<Vec<Segment>, CompatError> {
    match v {
        Value::String(s) => Ok(vec![Segment::Text(s.clone())]),
        Value::Array(items) => items
            .iter()
            .map(|i| match i {
                Value::String(s) => Ok(Segment::Text(s.clone())),
                Value::Number(n) => n
                    .as_u64()
                    .and_then(|x| u32::try_from(x).ok())
                    .map(Segment::Id)
                    .ok_or_else(|| field_err("prompt", "token ids must be non-negative integers")),
                _ => Err(field_err(
                    "prompt",
                    "a prompt mixes strings and token ids only",
                )),
            })
            .collect(),
        _ => Err(field_err(
            "prompt",
            "must be a string or an array of tokens",
        )),
    }
}

/// `prompt`: a string, token ids, a mix of both, or `{"prompt_string": ...}`.
/// Several prompts in one request are refused (llama.cpp answers with an
/// array; Atlas answers one request per call).
pub(super) fn parse_prompt(v: &Value, ctx: &ParseContext<'_>) -> Result<Vec<u32>, CompatError> {
    let single: Vec<Segment> = match v {
        Value::String(_) => segments_of(v)?,
        Value::Array(items) => {
            if items.iter().any(Value::is_number) {
                segments_of(v)?
            } else if items.is_empty() {
                return Err(field_err("prompt", "must not be empty"));
            } else if items.len() == 1 {
                return parse_prompt(&items[0], ctx);
            } else {
                return Err(CompatError::unsupported(
                    "prompt",
                    "several prompts in one request are not supported; send one request per prompt",
                ));
            }
        }
        Value::Object(o) => {
            if o.contains_key("multimodal_data") {
                return Err(CompatError::unsupported(
                    "prompt.multimodal_data",
                    "multimodal input is not supported on /completion",
                ));
            }
            match o.get("prompt_string") {
                Some(p) => segments_of(p)?,
                None => {
                    return Err(field_err(
                        "prompt",
                        "an object prompt needs 'prompt_string'",
                    ));
                }
            }
        }
        _ => {
            return Err(field_err(
                "prompt",
                "must be a string or an array of tokens",
            ));
        }
    };
    // llama.cpp adds the model's start-of-text token when the tokenizer wants
    // one (`add_special`); the tokenizer's post-processor decides the same here.
    let ids = tokenize_segments(ctx.codec, &single, true)?;
    let vocab = ctx.codec.vocab_size();
    if let Some(&bad) = ids.iter().find(|&&i| (i as usize) >= vocab) {
        return Err(CompatError::invalid(format!(
            "token id {bad} is out of range (vocabulary size {vocab})"
        )));
    }
    if ids.is_empty() {
        return Err(field_err("prompt", "must not be empty"));
    }
    Ok(ids)
}
