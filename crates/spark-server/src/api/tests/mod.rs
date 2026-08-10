// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the streaming-side guards on the OpenAI-compatible API.
//!
//! - `harness`   — whole-stream driver for the content sanitizer
//! - `sanitizer` — orphan tool-call fragments must not reach the client
//! - `envelope`  — F73: inner tags inside a sanctioned envelope must
//! - `watchdog`  — repetition guard: fires on real loops, not on prose
//!
//! The F7/F28-F32/F39/F44/F49/F50 suites that used to live here were
//! deleted with the prompt-injection subtree they tested (#90); nothing
//! in the request path counts duplicate writes or appends stall
//! reminders any more.

mod envelope;
mod harness;
mod sanitizer;
mod watchdog;
