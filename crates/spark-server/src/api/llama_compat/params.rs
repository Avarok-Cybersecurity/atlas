// SPDX-License-Identifier: AGPL-3.0-only

//! Reading a llama.cpp `/completion` request body.
//!
//! The field list, aliases, clamping and "null means absent" rule follow
//! `make_llama_cmpl_schema` in `tools/server/server-schema.cpp` (b11583).
//! Every field falls into one of three groups:
//!
//! 1. **Honored**: mapped onto what the scheduler already does (`prompt`,
//!    `n_predict`, `temperature`, `top_k`, `top_p`, `min_p`, `top_n_sigma`,
//!    `repeat_penalty`, `presence_penalty`, `frequency_penalty`, `dry_*`,
//!    `seed`, `stop`, `stream`, `logit_bias`, `n_probs`, `return_tokens`,
//!    `json_schema`, `grammar`).
//! 2. **No-op in Atlas, accepted**: `cache_prompt`, `n_keep`, `n_discard`,
//!    `n_cache_reuse`, `id_slot`, `echo`, `verbose`, and the others listed in
//!    the compatibility table. They change speed or llama.cpp bookkeeping, not
//!    the text.
//! 3. **Cannot be honored**: accepted only at the llama.cpp default (a request
//!    that does not use the feature); any other value is refused with HTTP 400
//!    naming the field. The request never runs with the field dropped.
//!
//! The readers and the refusal list are in `fields`; the two inputs that need
//! the tokenizer (`prompt`, `logit_bias`) are in `inputs`.

use serde_json::{Map, Value};

use super::codec::TextCodec;
use super::constraint::{Constraint, parse_constraint};
use super::error::CompatError;
use super::fields::{
    bounded_penalty, dry_params, field_err, first, flag, int, num, parse_stop,
    reject_unsupported_features,
};
use super::inputs::{parse_logit_bias, parse_prompt};
use super::props::SamplingDefaults;

/// Largest `n_probs`: the scheduler returns at most this many candidates.
pub(crate) const MAX_N_PROBS: u8 = 20;

/// What the parser needs to know about the running server.
pub(crate) struct ParseContext<'a> {
    pub codec: &'a dyn TextCodec,
    pub defaults: SamplingDefaults,
    pub max_seq_len: usize,
    pub grammar_available: bool,
}

/// A fully validated `/completion` request.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompletionSpec {
    pub prompt_tokens: Vec<u32>,
    /// The `n_predict` the client asked for (`-1` = until the context is full).
    pub n_predict: i64,
    /// The token budget handed to the scheduler.
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_n_sigma: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: u32,
    pub logit_bias: Vec<(u32, f32)>,
    pub stop: Vec<String>,
    pub seed: Option<u64>,
    pub stream: bool,
    /// 0 = no probabilities requested.
    pub n_probs: u8,
    pub return_tokens: bool,
    pub constraint: Option<Constraint>,
}

/// Parse and validate a `/completion` body.
pub(crate) fn parse_completion(
    body: &Map<String, Value>,
    ctx: &ParseContext<'_>,
) -> Result<CompletionSpec, CompatError> {
    let prompt = body
        .get("prompt")
        .filter(|v| !v.is_null())
        .ok_or_else(|| CompatError::invalid("\"prompt\" must be provided"))?;
    let prompt_tokens = parse_prompt(prompt, ctx)?;
    if prompt_tokens.len() >= ctx.max_seq_len {
        return Err(CompatError::exceed_context(
            prompt_tokens.len(),
            ctx.max_seq_len,
        ));
    }

    reject_unsupported_features(body)?;
    let constraint = parse_constraint(body, ctx.grammar_available)?;

    if let Some(n) = int(body, &["n_cmpl", "n"])?
        && n != 1
    {
        return Err(CompatError::unsupported(
            "n_cmpl",
            "one completion per request; send several requests",
        ));
    }

    // n_predict: -1 = until the context is full; 0 (evaluate the prompt only)
    // is not available.
    let n_predict = int(body, &["n_predict", "max_completion_tokens", "max_tokens"])?.unwrap_or(-1);
    if !(-1..=i64::from(i32::MAX)).contains(&n_predict) {
        return Err(field_err(
            "n_predict",
            "Value must be between -1 <= value <= 2147483647",
        ));
    }
    if n_predict == 0 {
        return Err(CompatError::unsupported(
            "n_predict",
            "n_predict=0 (evaluate the prompt and generate nothing) is not available",
        ));
    }
    let room = ctx.max_seq_len - prompt_tokens.len();
    let max_tokens = if n_predict < 0 {
        room
    } else {
        (n_predict as usize).min(room)
    };

    let d = ctx.defaults;
    let temperature = num(body, &["temperature"])?.map_or(d.temperature, |t| t.max(0.0) as f32);
    if temperature > 2.0 {
        return Err(CompatError::unsupported(
            "temperature",
            "Atlas accepts temperatures from 0 to 2",
        ));
    }
    let top_k = int(body, &["top_k"])?.map_or(d.top_k, |k| k.clamp(0, i64::from(i32::MAX)) as u32);
    let top_p = num(body, &["top_p"])?.map_or(d.top_p, |p| p.clamp(0.0, 1.0) as f32);
    let min_p = num(body, &["min_p"])?.map_or(d.min_p, |p| p.clamp(0.0, 1.0) as f32);
    let top_n_sigma = match num(body, &["top_n_sigma"])? {
        None => d.top_n_sigma,
        // llama.cpp: negative disables; Atlas: 0 disables.
        Some(s) if s < 0.0 => 0.0,
        Some(0.0) => {
            return Err(CompatError::unsupported(
                "top_n_sigma",
                "0 means 'keep only the top token' in llama.cpp, which Atlas's sampler \
                 cannot express; send a negative value to disable it",
            ));
        }
        Some(s) => s as f32,
    };
    let repetition_penalty =
        num(body, &["repeat_penalty"])?.map_or(d.repetition_penalty, |p| p as f32);
    let presence_penalty = bounded_penalty(body, "presence_penalty")?;
    let frequency_penalty = bounded_penalty(body, "frequency_penalty")?;
    let (dry_multiplier, dry_base, dry_allowed_length) = dry_params(body)?;

    let logit_bias = match first(body, &["logit_bias"]) {
        None => Vec::new(),
        Some((_, v)) => parse_logit_bias(v, ctx)?,
    };

    let seed = int(body, &["seed"])?.and_then(|s| u64::try_from(s).ok());

    let n_probs = match int(body, &["n_probs", "logprobs"])? {
        None => 0,
        Some(n) if n < 0 => return Err(field_err("n_probs", "must not be negative")),
        Some(n) if n > i64::from(MAX_N_PROBS) => {
            return Err(CompatError::unsupported(
                "n_probs",
                &format!("at most {MAX_N_PROBS} candidates per token are available"),
            ));
        }
        Some(n) => n as u8,
    };

    Ok(CompletionSpec {
        prompt_tokens,
        n_predict,
        max_tokens,
        temperature,
        top_k,
        top_p,
        min_p,
        top_n_sigma,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        logit_bias,
        stop: parse_stop(body)?,
        seed,
        stream: flag(body, "stream")?.unwrap_or(false),
        n_probs,
        return_tokens: flag(body, "return_tokens")?.unwrap_or(false),
        constraint,
    })
}

impl CompletionSpec {
    /// `generation_settings`: the settings this request ran with.
    pub(crate) fn generation_settings(&self, model: &str, n_ctx: usize) -> Value {
        let top_n_sigma = if self.top_n_sigma > 0.0 {
            serde_json::json!(self.top_n_sigma)
        } else {
            serde_json::json!(-1.0)
        };
        serde_json::json!({
            "n_ctx": n_ctx,
            "model": model,
            "seed": self.seed.map_or(-1, |s| s as i64),
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "top_n_sigma": top_n_sigma,
            "repeat_penalty": self.repetition_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "dry_multiplier": self.dry_multiplier,
            "dry_base": self.dry_base,
            "dry_allowed_length": self.dry_allowed_length,
            "stop": self.stop,
            "max_tokens": self.n_predict,
            "n_predict": self.n_predict,
            "stream": self.stream,
            "n_probs": self.n_probs,
            "grammar": self.constraint.as_ref().map_or("", Constraint::grammar_text),
        })
    }
}

#[cfg(test)]
#[path = "params_tests.rs"]
mod tests;
