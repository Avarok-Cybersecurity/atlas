// SPDX-License-Identifier: AGPL-3.0-only

//! Anthropic test suites split by area.
//!
//! - `convert`             — wire deserialization + the stop-reason /
//!                           tool-choice / tool-definition conversions
//! - `to_ir_blocks`        — how one wire message splits into N IR messages
//! - `ir_carry`            — what each block carries into (and out of) the IR
//! - `claude_code_fixture` — the real 26 KB / 70-tool captured session
//! - `translator_stream`   — SSE event framing for `/v1/messages`

mod claude_code_fixture;
mod convert;
mod ir_carry;
mod to_ir_blocks;
mod translator_stream;
