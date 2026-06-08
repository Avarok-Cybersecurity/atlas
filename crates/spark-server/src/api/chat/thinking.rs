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
    let budget = if et {
        let b = tb.unwrap_or(max_budget);
        // 2026-05-23 sweep: dropped the 70% special case for
        // `tools_active && thinking_in_tools` (previously 7/10, now
        // 9/10 uniformly). With `thinking_in_tools=true` as the
        // project-wide default the 70% branch fired on every tool turn
        // and silently undermined the MODEL.toml `max_thinking_budget`
        // bump (opencode-style requests at max_tokens=2048 capped to
        // 1433 instead of 2048). 90% leaves headroom for content +
        // tool args without crippling reasoning chains that now run
        // naturally after the F1 reflection-penalty removal.
        let safety_cap_pct = 9;
        let max = ((mt * safety_cap_pct) / 10).max(1);
        Some(b.min(max))
    } else {
        None
    };
    (et, budget)
}
