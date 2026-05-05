// SPDX-License-Identifier: AGPL-3.0-only
//
// Resolve `(enable_thinking, thinking_budget)` for a single
// request. Precedence (highest wins):
//   1. `--disable-thinking` CLI flag (forces OFF for every request)
//   2. Request body (`reasoning_effort`, `thinking.budget_tokens`, …)
//   3. MODEL.toml `[behavior].thinking_default`
//
// Lifted out of `chat::chat_completions_inner` (wave 4g).

use std::sync::Arc;

use crate::AppState;
use crate::openai::ChatCompletionRequest;

use super::super::failures::recent_message_is_tool_error;

pub(super) fn resolve_thinking(
    state: &Arc<AppState>,
    req: &ChatCompletionRequest,
    tools_active: bool,
) -> (bool, Option<u32>) {
    if state.disable_thinking {
        return (false, None);
    }
    let (et, tb) = req.resolve_thinking(state.behavior.thinking_default);
    let mt = req.max_tokens as u32;
    let max_budget = state.behavior.max_thinking_budget;
    // `thinking_in_tools=false` is the MODEL.toml DEFAULT for tool-
    // active turns: it suppresses thinking when the client is silent.
    let et = if tools_active
        && !state.behavior.thinking_in_tools
        && !req.thinking_explicitly_requested()
    {
        false
    } else {
        et
    };
    // F28: auto-disable thinking on turns following a tool error.
    let et = if et && recent_message_is_tool_error(&req.messages) {
        tracing::info!("F28: disabling thinking on this turn (most recent message is tool error)");
        false
    } else {
        et
    };
    let budget = if et {
        let b = tb.unwrap_or(max_budget);
        let safety_cap_pct = if tools_active && state.behavior.thinking_in_tools {
            7
        } else {
            9
        };
        let max = ((mt * safety_cap_pct) / 10).max(1);
        Some(b.min(max))
    } else {
        None
    };
    (et, budget)
}
