// SPDX-License-Identifier: AGPL-3.0-only
//
// Canonical, model-agnostic chat IR. One IR, N adapters: each API
// surface (OpenAI chat, OpenAI Responses, Anthropic) converts its wire
// format into these types, and the core pipeline (`build_msg_entries`,
// template rendering) reads only the IR — never an endpoint-specific
// request/response type. This is the narrow waist that keeps the
// OpenAI/Anthropic surfaces from drifting (AGENTS.md).

pub mod message;
pub mod request;
pub mod response;
pub mod stream;

pub use message::{ContentPart, ImageData, Message, Role};
pub use request::{ChatRequest, ResponseFormat, SamplingParams, ThinkingDirective};
pub use response::{ChatResponse, Choice, ChoiceLogprobs, FinishReason, TokenLogprob, Usage};
pub use stream::{DeltaStream, StreamDelta};

#[cfg(test)]
mod tests;
