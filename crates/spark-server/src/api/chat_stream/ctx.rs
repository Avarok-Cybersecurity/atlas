// SPDX-License-Identifier: AGPL-3.0-only
//
// Read-only context shared by every `StreamEvent` arm. Owned by the
// `flat_map` closure so the per-event handlers can borrow it
// immutably alongside `&mut StreamState`.

use std::sync::Arc;

use crate::AppState;
use crate::tool_parser;

use super::super::failures::F39FailureCache;

pub(super) struct StreamCtx {
    pub(super) state: Arc<AppState>,
    pub(super) model: String,
    pub(super) id: String,
    pub(super) prompt_len: usize,
    pub(super) enable_thinking: bool,
    pub(super) tool_defs_for_backfill: Vec<tool_parser::ToolDefinition>,
    pub(super) cwd_for_normalize: Option<String>,
    pub(super) stop_strings: Vec<String>,
    pub(super) leak_markers: tool_parser::LeakMarkers,
    pub(super) max_tool_calls_per_response: usize,
    pub(super) req_stream_include_usage: bool,
    pub(super) req_ctx: Option<crate::rate_limiter::RequestContext>,
    pub(super) dump_seq: Option<u64>,
    pub(super) f44_cache: F39FailureCache,
    pub(super) f44_cache_active: bool,
}
