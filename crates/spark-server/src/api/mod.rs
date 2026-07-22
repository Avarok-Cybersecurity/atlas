// SPDX-License-Identifier: AGPL-3.0-only

//! HTTP API routes (axum handlers).
//!
//! This module was originally a single 8,979-line `api.rs`; wave 4b1
//! split it into cohesive sub-files under `api/`. Public items are
//! re-exported here so external callers (`crate::api::Foo`) keep
//! working unchanged.
//!
//! Sub-file layout:
//! - `compact`            — progressive context compaction + error helpers
//! - `chat` / `chat_stream` — OpenAI `/v1/chat/completions` (blocking + SSE)
//! - `completions`        — legacy `/v1/completions` + list/get models +
//!                          embeddings stub + cross-handler helpers
//! - `sanitizer`          — `<parameter=…>` leak suppression + bash bucketing
//! - `stubs`              — batches/files/audio/images/moderations stubs
//! - `responses`,
//!   `responses_stream`,
//!   `responses_translate` — `/v1/responses` (blocking + streaming)
//! - `stored`             — stored-completion / response retrieval
//! - `conversations`      — `/v1/conversations` CRUD
//! - `misc_handlers`      — cancel / metrics / health / tokenize / detokenize
//! - `inference_types`    — `GrammarSpec`, `InferenceRequest`,
//!                          `InferenceResponse`, `StreamEvent`,
//!                          `TokenLogprobs`
//! - `inference_impl`     — `impl InferenceRequest`
//! - `strip`              — legacy `strip_thinking_tags` for completions
//! - `tests/`             — extracted `sanitizer_tests` module split four ways

pub mod chat;
pub mod chat_blocking;
pub mod chat_phases;
pub mod chat_stream;
pub mod chat_stream_dispatch;
pub mod compact;
pub mod completions;
pub mod completions_exec;
pub mod completions_logprobs;
pub mod conversations;
pub mod inference_impl;
pub mod inference_types;
pub mod lora_control;
pub mod misc_handlers;
pub mod models;
pub mod responses;
pub mod responses_stream;
pub mod responses_stream_finalize;
pub mod responses_translate;
pub mod sanitizer;
pub mod stored;
pub mod stream_guards;
pub mod strip;
pub mod stubs;

#[cfg(test)]
mod tests;

// Re-exports to preserve the original `crate::api::*` import surface.
// `#[allow(unused_imports)]` is applied only where the re-export is
// part of the public surface but happens to be unreferenced this build
// (Request types kept for external clients / schema generation, plus
// `compact_messages` whose handler is wired directly in serve_router).
pub use chat::chat_completions;
pub(crate) use chat::chat_completions_inner;
pub(crate) use chat::{ChatOutcome, ResponseEcho};
#[allow(unused_imports)]
pub use compact::compact_messages;
pub use completions::completions;
#[allow(unused_imports)]
pub use conversations::{
    AddItemsRequest, CreateConversationRequest, UpdateConversationRequest, add_conversation_items,
    create_conversation, delete_conversation, delete_conversation_item, get_conversation,
    get_conversation_item, list_conversation_items, update_conversation,
};
pub use inference_types::{
    GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent, TokenLogprobs,
};
pub use lora_control::{load_lora_into_slot, set_active_lora};
#[allow(unused_imports)]
pub use misc_handlers::{
    DetokenizeRequest, cancel_response, detokenize, health, health_live, metrics_handler, tokenize,
};
pub use models::{embeddings_stub, get_model, list_models};
pub use responses::responses_endpoint;
pub use stored::{
    delete_stored_response, get_stored_completion, get_stored_response, list_response_input_items,
};
pub use stubs::{
    audio_stub, batch_get_stub, batch_list_stub, batches_stub, files_stub, images_stub,
    moderations_stub,
};
