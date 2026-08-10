// SPDX-License-Identifier: AGPL-3.0-only

//! OpenAI-compatible API types.

mod annotations;
mod chat_message;
mod chat_request;
mod chat_response;
mod completions;
mod encode;
mod encode_stream;
mod responses;
mod responses_lowering;
mod stream_chunk;
mod to_ir;

#[cfg(test)]
mod tests;

pub use annotations::*;
pub use chat_message::*;
pub use chat_request::*;
pub use chat_response::*;
pub use completions::*;
pub(crate) use encode::encode_chat_response;
pub(crate) use encode_stream::encode_sse_response;
pub use responses::*;
pub use responses_lowering::*;
pub use stream_chunk::*;

// ID/timestamp primitives live in the provider-neutral `crate::ids`;
// the private import keeps bare `uuid_v4()` / `unix_timestamp()` calls
// in this module's children (via `use super::*`) working.
use crate::ids::{unix_timestamp, uuid_v4};

/// Generate a new completion ID for SSE streaming.
pub fn new_completion_id() -> String {
    format!("cmpl-{}", uuid_v4())
}

/// Generate a new chunk ID for SSE streaming.
pub fn new_chunk_id() -> String {
    format!("chatcmpl-{}", uuid_v4())
}
