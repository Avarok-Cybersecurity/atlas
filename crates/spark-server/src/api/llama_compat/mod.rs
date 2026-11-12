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

pub(crate) mod codec;
pub(crate) mod error;
pub(crate) mod paths;
pub(crate) mod tokenize;
#[cfg(test)]
mod toy_codec;
