// SPDX-License-Identifier: AGPL-3.0-only

//! llama.cpp `llama-server` compatibility layer.
//!
//! A client written against llama.cpp's own HTTP API (`/completion`,
//! `/tokenize`, `/detokenize`, `/apply-template`, `/props`, `/health`,
//! `/embedding`) can talk to Atlas unchanged. The contract followed here is
//! `tools/server/README.md` and the handlers in `tools/server/` at llama.cpp
//! tag b11583.
//!
//! There is no second inference path. `/completion` builds the same
//! `InferenceRequest` the OpenAI `/v1/completions` route builds and sends it to
//! the same scheduler; this module translates request and response shapes.
//!
//! Rule for fields Atlas cannot honor: a llama.cpp field that changes what the
//! model produces and that Atlas has no equivalent for is refused with HTTP 400
//! naming the field, never ignored. A field whose value is the llama.cpp
//! default (a no-op) is accepted. The table of every field and its treatment is
//! in `docs/llama-cpp-compat.md`.
//!
//! Sub-modules:
//! - `error`          llama.cpp's `{"error": {code, message, type}}` envelope
//! - `paths`          which compat routes the auth, rate-limit and fault layers guard
//! - `codec`          the tokenizer operations the routes need, behind a trait
//! - `tokenize`       `/tokenize` and `/detokenize` shapes
//! - `apply_template` `/apply-template` request and response
//! - `props`          `/props` document and the sampling defaults
//! - `handlers`       axum handlers for the three routes above and `/embedding`
//! - `params`         `/completion` request parsing; fields honored or refused
//! - `fields`         typed field readers and the list of refused fields
//! - `inputs`         `prompt` and `logit_bias`, which need the tokenizer
//! - `completion`     the `/completion` handler: request mapping, blocking answer
//! - `completion_stream` the streaming answer (bare SSE frames, no `[DONE]`)
//! - `completion_json` response and chunk bodies
//! - `generation`     tokens in, released text / kept tokens / probabilities out
//! - `finish`         the closing body, from the scheduler's `Done` numbers
//! - `constraint`     `json_schema` / `grammar`, validated before queueing
//! - `stop`           stop-string matching on the decoded text stream

pub(crate) mod apply_template;
pub(crate) mod codec;
pub(crate) mod completion;
pub(crate) mod completion_json;
pub(crate) mod completion_stream;
pub(crate) mod constraint;
pub(crate) mod error;
pub(crate) mod fields;
pub(crate) mod finish;
pub(crate) mod generation;
pub(crate) mod handlers;
pub(crate) mod inputs;
pub(crate) mod params;
pub(crate) mod paths;
pub(crate) mod props;
pub(crate) mod stop;
pub(crate) mod tokenize;
#[cfg(test)]
mod toy_codec;
