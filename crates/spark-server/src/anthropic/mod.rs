// SPDX-License-Identifier: AGPL-3.0-only

//! Anthropic Messages API translation layer.
//!
//! Converts Anthropic `/v1/messages` requests to internal `InferenceRequest`,
//! reuses the existing scheduler pipeline, and converts the response back to
//! Anthropic format. Supports both streaming (SSE) and non-streaming.

mod convert;
mod handlers;
mod handlers_stream;
mod helpers;
mod translate;
mod translator;
mod types;

#[cfg(test)]
mod tests;

pub use handlers::{count_tokens, messages};
