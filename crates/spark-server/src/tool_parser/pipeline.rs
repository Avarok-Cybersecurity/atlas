// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

/// When in the pipeline a pass runs. Primary = known-good formats; Salvage
/// = last-ditch heuristic recovery for malformed output.
pub enum PassKind {
    Primary,
    Salvage,
}

/// Mutable state threaded through tool-call parsing passes.
///
/// A pass reads from `remaining` and, when it matches, consumes it (clears
/// or truncates) and pushes extracted calls / surrounding content into the
/// accumulators. Passes MUST leave `remaining` unchanged on a no-op so the
/// next pass can try its pattern against the same text.
pub struct PassState<'a> {
    pub remaining: &'a mut String,
    pub calls: &'a mut Vec<ToolCall>,
    pub content: &'a mut Vec<String>,
    pub call_counter: &'a mut u32,
}

/// One parsing step over the tool-call byte stream. Each pass owns one
/// format (e.g. bare `<function>` tag-style) and is responsible for scanning
/// the entire `remaining` buffer for matches of its pattern, pushing any
/// calls + surrounding content, then consuming what it parsed.
pub trait ToolCallPass: Send + Sync {
    /// Short name for logs (conventionally `lowercase_snake`).
    fn name(&self) -> &str;
    /// Scan `state.remaining` for this pass's pattern. No-op if absent.
    fn apply(&self, state: &mut PassState<'_>);
}

/// Ordered pipeline of tool-call parsing passes. Registered in order,
/// partitioned by `PassKind`. See module docs for the first-match
/// semantics.
pub struct ToolCallPipeline {
    primary: Vec<Box<dyn ToolCallPass>>,
    salvage: Vec<Box<dyn ToolCallPass>>,
}

impl Default for ToolCallPipeline {
    fn default() -> Self {
        Self::bare_function_default()
    }
}

impl ToolCallPipeline {
    pub fn new() -> Self {
        Self {
            primary: Vec::new(),
            salvage: Vec::new(),
        }
    }

    pub fn register(mut self, pass: Box<dyn ToolCallPass>, kind: PassKind) -> Self {
        match kind {
            PassKind::Primary => self.primary.push(pass),
            PassKind::Salvage => self.salvage.push(pass),
        }
        self
    }

    /// Default pipeline for bare-function recovery (outside `<tool_call>`
    /// wrappers). Primary: `<function>NAME</function>` tag-style,
    /// `<function=NAME>` attribute-style, `NAME{json}` Mistral-style.
    /// Salvage: `<parameter=NAME>` mis-typed opener.
    pub fn bare_function_default() -> Self {
        Self::new()
            .register(Box::new(BareFunctionTagPass), PassKind::Primary)
            .register(Box::new(BareFunctionAttrPass), PassKind::Primary)
            .register(Box::new(BareMistralNamePass), PassKind::Primary)
            .register(Box::new(ParamAsFunctionSalvagePass), PassKind::Salvage)
    }

    /// Run the pipeline. Returns (content_before_calls, calls).
    pub fn run(&self, text: &str) -> (Option<String>, Vec<ToolCall>) {
        let mut remaining = text.to_string();
        let mut calls = Vec::new();
        let mut content = Vec::new();
        let mut call_counter = 0u32;
        let mut state = PassState {
            remaining: &mut remaining,
            calls: &mut calls,
            content: &mut content,
            call_counter: &mut call_counter,
        };

        // Primary: first pass that extracts anything wins (existing
        // first-match semantics of parse_bare_function_calls).
        for pass in &self.primary {
            pass.apply(&mut state);
            if !state.calls.is_empty() {
                break;
            }
        }

        // Salvage: only if primary found nothing. Same first-match rule.
        if state.calls.is_empty() {
            for pass in &self.salvage {
                pass.apply(&mut state);
                if !state.calls.is_empty() {
                    tracing::warn!(
                        pass = pass.name(),
                        "tool-call salvage fired — model emitted malformed output"
                    );
                    break;
                }
            }
        }

        // Intentionally DO NOT emit `state.remaining` as trailing content —
        // legacy `parse_bare_function_calls` drops anything after the last
        // `</function>`, and callers (parse_tool_calls) expect that contract.
        // Content-before is already accumulated via pass.apply.
        let combined = if content.is_empty() {
            None
        } else {
            Some(content.join("\n"))
        };
        (combined, calls)
    }
}

// ── Pass impls ────────────────────────────────────────────────────────

/// `<function>NAME</function>` + optional `<parameters>{...}</parameters>` blocks.
pub struct BareFunctionTagPass;
impl ToolCallPass for BareFunctionTagPass {
    fn name(&self) -> &str {
        "bare_function_tag"
    }
    fn apply(&self, state: &mut PassState<'_>) {
        let text = state.remaining.clone();
        let mut cur = text.as_str();
        let mut first = true;
        while let Some(start) = cur.find("<function>") {
            if first {
                let before = cur[..start].trim();
                if !before.is_empty() {
                    state.content.push(before.to_string());
                }
                first = false;
            }
            if let Some(tc) = parse_tag_style_call(&cur[start..], *state.call_counter) {
                let call_end = cur[start..]
                    .find("</function>")
                    .map(|e| start + e + "</function>".len())
                    .unwrap_or(cur.len());
                cur = &cur[call_end..];
                state.calls.push(tc);
                *state.call_counter += 1;
            } else {
                cur = &cur[start + "<function>".len()..];
            }
        }
        // Match the legacy `parse_bare_function_calls` contract: on match,
        // return only content BEFORE the first call — drop trailing bytes
        // after the last `</function>`. `let _ = cur;` silences unused-var
        // after the loop; consumption of `state.remaining` signals "done".
        if !state.calls.is_empty() {
            let _ = cur;
            state.remaining.clear();
        }
    }
}

/// `<function=NAME>` / `<function NAME>` attribute-style calls.
pub struct BareFunctionAttrPass;
impl ToolCallPass for BareFunctionAttrPass {
    fn name(&self) -> &str {
        "bare_function_attr"
    }
    fn apply(&self, state: &mut PassState<'_>) {
        let text = state.remaining.clone();
        let mut cur = text.as_str();
        let mut first = true;
        while let Some(start) = cur.find("<function=").or_else(|| cur.find("<function ")) {
            if first {
                let before = cur[..start].trim();
                if !before.is_empty() {
                    state.content.push(before.to_string());
                }
                first = false;
            }
            if let Some(tc) = parse_qwen3_coder_call(&cur[start..], *state.call_counter) {
                let call_end = cur[start..]
                    .find("</function>")
                    .map(|e| start + e + "</function>".len())
                    .unwrap_or(cur.len());
                cur = &cur[call_end..];
                state.calls.push(tc);
                *state.call_counter += 1;
            } else {
                cur = &cur[start + "<function".len()..];
            }
        }
        if !state.calls.is_empty() {
            let _ = cur;
            state.remaining.clear();
        }
    }
}

/// Bare Mistral-style: `name{"arg": "value"}` — NVFP4 models that skip
/// the `<tool_call>` wrapper entirely.
pub struct BareMistralNamePass;
impl ToolCallPass for BareMistralNamePass {
    fn name(&self) -> &str {
        "bare_mistral_name"
    }
    fn apply(&self, state: &mut PassState<'_>) {
        let trimmed = state.remaining.trim();
        let Some(brace_pos) = trimmed.find('{') else {
            return;
        };
        let name_part = trimmed[..brace_pos].trim();
        let json_part = &trimmed[brace_pos..];
        if name_part.is_empty()
            || !name_part.chars().all(is_tool_name_or_namespace_char)
            || !json_part.ends_with('}')
        {
            return;
        }
        let func_name = normalize_tool_name(name_part);
        // Phantom `json:` / `tool_call:` prose keeps its colon through
        // normalization — no-op so later passes see the original text.
        if !is_normalized_tool_name(&func_name) {
            return;
        }
        let Ok(args_obj) = serde_json::from_str::<serde_json::Value>(json_part) else {
            return;
        };
        if !args_obj.is_object() {
            return;
        }
        state.calls.push(ToolCall {
            id: next_tool_call_id(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: func_name,
                arguments: json_part.to_string(),
            },
        });
        *state.call_counter += 1;
        state.remaining.clear();
    }
}

/// SALVAGE: `<parameter=NAME>...</parameter>` ... `</function>` — the model
/// typed `<parameter=NAME>` where `<function=NAME>` was expected (a common
/// slip for Qwen3.5-35B-FP8 in opencode sessions). Fires only when there's
/// a closing `</function>` AND no legitimate `<function>` / `<function=` /
/// `<function ` opener anywhere in the buffer. Rewrites the first
/// `<parameter=NAME>` to `<function=NAME>` and hands off to
/// `parse_qwen3_coder_call`, so the arguments block is parsed by the same
/// code path that handles well-formed calls.
pub struct ParamAsFunctionSalvagePass;
impl ToolCallPass for ParamAsFunctionSalvagePass {
    fn name(&self) -> &str {
        "param_as_function_salvage"
    }
    fn apply(&self, state: &mut PassState<'_>) {
        let text = state.remaining.as_str();
        if !text.contains("</function>") {
            return;
        }
        if text.contains("<function>")
            || text.contains("<function=")
            || text.contains("<function ")
            || text.contains("<|function=")
            || text.contains("<|function ")
        {
            return;
        }
        let Some(first_param) = text.find("<parameter=") else {
            return;
        };
        let after_eq = &text[first_param + "<parameter=".len()..];
        let Some(close_gt) = after_eq.find('>') else {
            return;
        };
        let name = after_eq[..close_gt].trim().to_string();
        if name.is_empty() || !name.chars().all(is_tool_name_or_namespace_char) {
            return;
        }
        // Rewrite only the FIRST `<parameter=NAME>`. Subsequent
        // `<parameter=K>V</parameter>` blocks remain as argument pairs.
        let before = &text[..first_param];
        let tail = &text[first_param..];
        let from = format!("<parameter={name}>");
        let func_name = normalize_tool_name(&name);
        if !is_normalized_tool_name(&func_name) {
            return;
        }
        let to = format!("<function={func_name}>");
        let fixed_tail = tail.replacen(&from, &to, 1);
        let reconstructed = format!("{before}{fixed_tail}");
        let func_start = match reconstructed.find("<function=") {
            Some(p) => p,
            None => return,
        };
        let Some(tc) = parse_qwen3_coder_call(&reconstructed[func_start..], *state.call_counter)
        else {
            return;
        };
        let before_trimmed = before.trim();
        if !before_trimmed.is_empty() {
            state.content.push(before_trimmed.to_string());
        }
        state.calls.push(tc);
        *state.call_counter += 1;
        state.remaining.clear();
    }
}
