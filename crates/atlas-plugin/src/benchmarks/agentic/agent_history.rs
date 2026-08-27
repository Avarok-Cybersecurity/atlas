// SPDX-License-Identifier: AGPL-3.0-only

//! Conversation construction: how one turn's outcome becomes the next turn's
//! history, and how that history is kept inside the window.
//!
//! Split out of `agent.rs` for the repo's 500-line ceiling, same as
//! `agent_tools.rs` before it. The containment and fidelity rules documented
//! there apply unchanged.

use serde_json::{Value, json};

use super::{HISTORY_BUDGET, LIVE_REASONING, LIVE_TOOL_RESULTS};

/// The `tool_call_id` this conversation carries — **ours, never the server's.**
///
/// Atlas mints ids from a per-process counter (`call_0000000000000004`), so the
/// same turn of the same work is labelled differently depending on how many
/// tool calls that server has answered since it started. Echoing it wrote a
/// value from outside the run into the model's context, where it changes the
/// next turn's tokens: measured here, five identical requests came back with
/// five distinct id sets and identical text. An id only has to pair one
/// assistant `tool_calls` entry with its `role: "tool"` reply inside this
/// request, so a positional one is both legal and reproducible.
/// Turn *and* index, because the two sites must agree — a `tool_call_id` that
/// pairs with nothing on the assistant message is a 400. They previously
/// numbered from different bases (`i` against `turn * 100 + i`) and only
/// matched because both echoed the server's id; a model that emits no ids hit
/// the mismatch.
pub(super) fn call_id(turn: usize, nth: usize) -> String {
    format!("call_{turn}_{nth}")
}

/// Elide the oldest tool results once the session outgrows the window — the
/// port of opencode's auto-compaction (`isOverflow` → `compaction`).
///
/// It rewrites tool *contents* and never removes a message: an assistant
/// `tool_calls` block whose matching `role: "tool"` reply went missing is a 400
/// from the server, which would end the run rather than shorten it.
pub(super) fn compact(messages: &mut [Value]) {
    let size = |m: &Value| {
        m["content"].as_str().map_or(64, str::len)
            + m["reasoning_content"].as_str().map_or(0, str::len)
    };
    let mut total: usize = messages.iter().map(size).sum();
    // Stale chain-of-thought goes FIRST when preserve-thinking is on: it is
    // the least useful thing left in the window and the fastest-growing (a
    // 2048-token budget over 40 turns outruns the window on its own — the
    // 2026-08-27 model-card run died at 16,395 prompt tokens on turn 32).
    // Elided to a MARKER, never removed: dropping the field would put that
    // turn back to an empty `<think></think>` wrapper, which is the exact
    // poisoning `preserve_thinking()` exists to avoid.
    let think: Vec<usize> = (0..messages.len())
        .filter(|i| messages[*i]["reasoning_content"].is_string())
        .collect();
    for &i in think
        .iter()
        .take(think.len().saturating_sub(LIVE_REASONING))
    {
        if total <= HISTORY_BUDGET {
            return;
        }
        let was = messages[i]["reasoning_content"]
            .as_str()
            .map_or(0, str::len);
        let marker = format!("[{was} characters of earlier reasoning elided]");
        total = total - was + marker.len();
        messages[i]["reasoning_content"] = Value::String(marker);
    }
    let tools: Vec<usize> = (0..messages.len())
        .filter(|i| messages[*i]["role"] == "tool")
        .collect();
    for &i in tools
        .iter()
        .take(tools.len().saturating_sub(LIVE_TOOL_RESULTS))
    {
        if total <= HISTORY_BUDGET {
            return;
        }
        let was = size(&messages[i]);
        let marker = format!("[{was} characters elided to stay inside the context window]");
        total = total - was + marker.len();
        messages[i]["content"] = Value::String(marker);
    }
}

/// `ATLAS_AGENTIC_PRESERVE_THINKING=1`: echo each turn's `reasoning_content`
/// back in its assistant message and ask the template to keep it.
///
/// Opt-in and never gated on — the recorded tiers stay thinking-off/greedy.
/// It exists because the default is NOT neutral for a thinking model. The
/// harness always dropped `outcome.reasoning`, while the Qwen3.8 templates
/// wrap every assistant turn after the last user query in `<think>` —
/// `preserve_thinking` cannot suppress that in an agentic shape, because the
/// single task prompt IS the last user query, so `loop.index0 >
/// last_query_index` holds for every later turn. With the field absent that
/// wrapper renders EMPTY, which is the "empty-think poisoning" the server's
/// own `msg_entry` docs name (premature `<|im_end|>`; vLLM/SGLang #131, MLC
/// d75d64e). So a 30-turn run fed the model 30 examples of "assistant
/// reasoned nothing, then acted".
///
/// The server side already accepts this: `reasoning_content` on an inbound
/// assistant message is forwarded into the Jinja render, and
/// `preserve_thinking` is resolved INDEPENDENTLY of `reasoning_effort`, so
/// sending it cannot disturb a serve-level effort pin.
pub(super) fn preserve_thinking() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("ATLAS_AGENTIC_PRESERVE_THINKING").as_deref() == Ok("1"))
}

pub(super) fn assistant_message(outcome: &crate::http::ChatOutcome, turn: usize) -> Value {
    let calls: Vec<Value> = outcome
        .tool_calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            json!({"id": call_id(turn, i), "type": "function", "function": {"name": c.name,
                // Some models emit no arguments at all for a zero-arg call; an
                // empty string is not valid JSON to a strict server.
                "arguments": if c.arguments.is_empty() { "{}" } else { &c.arguments }}})
        })
        .collect();
    let text = &outcome.text;
    let mut msg = json!({"role": "assistant", "tool_calls": calls,
        "content": if text.is_empty() { Value::Null } else { Value::String(text.clone()) }});
    if preserve_thinking() && !outcome.reasoning.trim().is_empty() {
        msg["reasoning_content"] = Value::String(outcome.reasoning.clone());
    }
    msg
}
