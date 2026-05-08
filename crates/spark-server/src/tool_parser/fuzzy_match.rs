// SPDX-License-Identifier: AGPL-3.0-only

//! Fuzzy tool-name repair for the validation pipeline.
//!
//! When a quantized / MTP-driven model emits a tool name that doesn't
//! match any registered tool exactly, the validator falls back to fuzzy
//! matching here. The strategies are deliberately ordered cheap → loose
//! and each strategy only returns a hit if exactly one tool matches —
//! ambiguous matches drop through so the final caller emits a clean
//! `Unknown tool` error rather than silently coercing.

use super::ToolDefinition;

/// Fuzzy match a hallucinated tool name to the closest available tool.
///
/// Strategies (in priority order):
/// 0. Separator-normalized exact match — handles MCP names where the
///    quantized model dropped or duplicated an underscore (e.g.
///    `mcp_discord__discord_send` → `mcp__discord__discord_send`) and
///    case / dash↔underscore drift.
/// 1. Substring containment — model name is substring of a tool name.
/// 2. Available tool name is substring of the model's name.
/// 3. Single-tool fallback — model clearly intended *something*.
///
/// Only returns a match if exactly one tool matches (ambiguous = reject).
pub(super) fn fuzzy_match_tool_name(model_name: &str, tools: &[ToolDefinition]) -> Option<String> {
    if tools.is_empty() || model_name.is_empty() {
        return None;
    }

    let lower = model_name.to_lowercase();

    // Strategy 0: separator-normalized exact match. MCP tool names commonly
    // contain double-underscores (e.g. `mcp__discord__discord_send`); MTP /
    // FP8 quantized models occasionally drop or duplicate one of those
    // underscores, producing `mcp_discord__discord_send` or
    // `mcp__discord_discord_send`. The substring strategies below miss
    // those because the missing/extra underscore makes neither side a
    // strict substring of the other. Collapsing runs of `_` and `-` and
    // lowercasing both sides recovers the intended match while still
    // requiring every other character to line up exactly. Discord report
    // `_trithemius` (2026-05-08): Qwen3.6-35B-A3B-FP8 + MCP — "Unknown
    // tool XXX" where XXX is in the registered list.
    let norm_model = normalize_tool_name(model_name);
    if !norm_model.is_empty() {
        let exact_norm: Vec<&str> = tools
            .iter()
            .filter(|t| normalize_tool_name(&t.function.name) == norm_model)
            .map(|t| t.function.name.as_str())
            .collect();
        if exact_norm.len() == 1 {
            return Some(exact_norm[0].to_string());
        }
    }

    // Strategy 1: exact substring — model name is substring of a tool name
    let matches: Vec<&str> = tools
        .iter()
        .filter(|t| t.function.name.to_lowercase().contains(&lower))
        .map(|t| t.function.name.as_str())
        .collect();
    if matches.len() == 1 {
        return Some(matches[0].to_string());
    }

    // Strategy 2: tool name is substring of model's name
    let matches: Vec<&str> = tools
        .iter()
        .filter(|t| lower.contains(&t.function.name.to_lowercase()))
        .map(|t| t.function.name.as_str())
        .collect();
    if matches.len() == 1 {
        return Some(matches[0].to_string());
    }

    // Strategy 3: if only one tool available, use it (model clearly intended to call a tool)
    if tools.len() == 1 {
        return Some(tools[0].function.name.clone());
    }

    None
}

/// Lowercase and collapse runs of `_` / `-` to a single `_`, trimming
/// leading and trailing underscores. Used by fuzzy tool-name repair to
/// match across separator-count drift introduced by MTP / quantization
/// errors (e.g. `mcp_discord__discord_send` → `mcp__discord__discord_send`).
fn normalize_tool_name(s: &str) -> String {
    let lower = s.trim().to_lowercase();
    let mut out = String::with_capacity(lower.len());
    let mut prev_sep = false;
    for c in lower.chars() {
        if c == '_' || c == '-' {
            if !prev_sep {
                out.push('_');
            }
            prev_sep = true;
        } else {
            out.push(c);
            prev_sep = false;
        }
    }
    out.trim_matches('_').to_string()
}
