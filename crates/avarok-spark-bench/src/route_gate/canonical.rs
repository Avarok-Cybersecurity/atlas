// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from the ARC-1.0 release files (config.json `arc.encoding_profile`, goldens.jsonl), Copyright 2026
// Atlas Futures, Inc., licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0). The rendering
// below is reconstructed from the release's own token counts, not copied from any closed code. By Atlas contributors.
// See RAYLINE-ROUTING-ATTRIBUTION.md.
//! Responsibility: the reconstructed request serializer of the release (`arc-role-blocks-v2`, projection
//! `canonical_v2`, `harness_shell: include`, `environment: scrub_v2`) and the choice between it and the old stand-in.
//!
//! What the 54 golden token counts and the signal goldens say (evidence in the report `88-serializer-tokens.md`):
//!
//! * A request is a list of role blocks, `<|im_start|>{role}\n{body}<|im_end|>\n`, one per message, and the readout
//!   is the last token of the last block (its trailing newline).
//! * The body is **not** the message text: it is the message content as a JSON list of typed parts, dumped as Python's
//!   `json.dumps(parts, sort_keys=True)` does (separators `", "` and `": "`, sorted keys), for example
//!   `[{"text": "What is two plus two?", "type": "text"}]`. A plain string content becomes one text part. This wrapper
//!   is 12 tokens for every user and assistant message, which is exactly the 12-per-message delta of the old stand-in,
//!   including the +1 on the three prompts that end in neither `.` nor `?` (the closing `"}]` merges with a final
//!   period or question mark in the tokenizer but not with a letter or a quote). The key order (`text` before `type`)
//!   is not visible in the counts; the signal goldens prefer it by a factor of 13 in mean error.
//! * Tool definitions are one system block holding one text part per tool whose text is the tool's JSON in the
//!   documented API key order (`name`, `description`, `input_schema` with `type`, `properties`, `required`). The text is
//!   JSON-escaped inside the part. The order inside the tool text is the one rule that is not `sort_keys`: with sorted
//!   keys the block is one token too long.
//! * A `tool_result` part carries the `tool_name` of the call its `tool_use_id` answers, which supplies the 8 tokens
//!   the tool conversation still needs.
//! * Nothing else of the request (attribution, epoch, prices) is serialized; `scrub_v2` changes nothing the goldens
//!   contain.
//!
//! Only two goldens exercise tools, so the tool rules fit two numbers (and one signal case); see the report.

use std::collections::HashMap;

use serde_json::{Map, Value};

use super::error::{GResult, value_err};
use super::serializer::{IM_END, IM_START, py_json_dumps, render_blocks};

/// Which request serializer renders the blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Serializer {
    /// `<|im_start|>{role}\n{flattened text}<|im_end|>\n` (the old stand-in; 12 tokens per message short).
    StandIn,
    /// The reconstruction of `arc-role-blocks-v2` / `canonical_v2` (default).
    CanonicalV2,
}

impl Serializer {
    /// `v2` (also `canonical_v2`) or `standin` (also `stand-in`).
    pub fn parse(s: &str) -> GResult<Self> {
        match s {
            "v2" | "canonical_v2" | "canonical-v2" => Ok(Self::CanonicalV2),
            "standin" | "stand-in" => Ok(Self::StandIn),
            other => value_err(format!("unknown serializer {other:?} (v2 or standin)")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::StandIn => "standin",
            Self::CanonicalV2 => "canonical_v2",
        }
    }

    /// The role blocks of one request object (`{"messages": [...], "tools": [...], "system": ...}`).
    pub fn render(self, request: &Value) -> Vec<String> {
        match self {
            Self::StandIn => render_blocks(messages(request)),
            Self::CanonicalV2 => render_v2(request),
        }
    }
}

fn messages(request: &Value) -> &[Value] {
    request
        .get("messages")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice)
}

fn frame(role: &str, parts: &[Value]) -> String {
    format!(
        "{IM_START}{role}\n{}{IM_END}\n",
        py_json_dumps(&Value::Array(parts.to_vec()))
    )
}

fn text_part(text: &str) -> Value {
    let mut m = Map::new();
    m.insert("type".into(), Value::String("text".into()));
    m.insert("text".into(), Value::String(text.to_string()));
    Value::Object(m)
}

fn content_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => py_json_dumps(other),
    }
}

/// One typed part of a message in its canonical form (keys are sorted when dumped).
fn norm_part(m: &Map<String, Value>, names: &mut HashMap<String, String>) -> Value {
    let ty = m.get("type").and_then(Value::as_str);
    match ty {
        Some("text") => text_part(&m.get("text").map_or_else(String::new, content_text)),
        None if m.contains_key("text") => text_part(&content_text(&m["text"])),
        Some("tool_use") => {
            if let (Some(id), Some(name)) = (
                m.get("id").and_then(Value::as_str),
                m.get("name").and_then(Value::as_str),
            ) {
                names.insert(id.to_string(), name.to_string());
            }
            Value::Object(m.clone())
        }
        Some("tool_result") => {
            let mut out = m.clone();
            if let Some(name) = m
                .get("tool_use_id")
                .and_then(Value::as_str)
                .and_then(|id| names.get(id))
            {
                out.insert("tool_name".into(), Value::String(name.clone()));
            }
            if let Some(Value::Array(a)) = m.get("content") {
                out.insert("content".into(), Value::Array(norm_parts(a, names)));
            }
            Value::Object(out)
        }
        _ => Value::Object(m.clone()),
    }
}

fn norm_parts(items: &[Value], names: &mut HashMap<String, String>) -> Vec<Value> {
    items
        .iter()
        .map(|p| match p {
            Value::String(s) => text_part(s),
            Value::Object(m) => norm_part(m, names),
            other => text_part(&content_text(other)),
        })
        .collect()
}

/// The parts of a message `content` (string, list of parts, one part object or null).
fn content_parts(content: &Value, names: &mut HashMap<String, String>) -> Vec<Value> {
    match content {
        Value::Array(a) => norm_parts(a, names),
        Value::Object(m) => vec![norm_part(m, names)],
        other => vec![text_part(&content_text(other))],
    }
}

/// Key order inside a tool definition's JSON text: the documented API order, the rest sorted.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Ctx {
    /// Sorted keys (arguments, free-form objects).
    Plain,
    /// `name`, `description`, `input_schema`, then the rest sorted.
    Tool,
    /// `type`, `properties`, `required`, then the rest sorted (a JSON schema).
    Schema,
    /// The map of property names: sorted, every value a schema.
    Props,
}

fn priority(ctx: Ctx) -> &'static [&'static str] {
    match ctx {
        Ctx::Tool => &["name", "description", "input_schema"],
        Ctx::Schema => &["type", "properties", "required"],
        Ctx::Plain | Ctx::Props => &[],
    }
}

fn child_ctx(ctx: Ctx, key: &str) -> Ctx {
    match (ctx, key) {
        (Ctx::Tool, "input_schema") => Ctx::Schema,
        (Ctx::Schema, "properties") => Ctx::Props,
        (Ctx::Schema | Ctx::Props, _) => Ctx::Schema,
        _ => Ctx::Plain,
    }
}

fn ordered_keys(m: &Map<String, Value>, ctx: Ctx) -> Vec<&String> {
    let prio = priority(ctx);
    let rank = |k: &str| prio.iter().position(|p| *p == k).unwrap_or(prio.len());
    let mut keys: Vec<&String> = m.keys().collect();
    keys.sort_by(|a, b| rank(a).cmp(&rank(b)).then_with(|| a.cmp(b)));
    keys
}

/// `json.dumps` formatting with the key order of `ctx`.
fn dump_ctx(v: &Value, ctx: Ctx, out: &mut String) {
    match v {
        Value::Object(m) => {
            out.push('{');
            for (i, k) in ordered_keys(m, ctx).into_iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                super::serializer::dump_string(k, out);
                out.push_str(": ");
                dump_ctx(&m[k], child_ctx(ctx, k), out);
            }
            out.push('}');
        }
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dump_ctx(x, if ctx == Ctx::Schema { ctx } else { Ctx::Plain }, out);
            }
            out.push(']');
        }
        other => out.push_str(&py_json_dumps(other)),
    }
}

/// A tool definition's JSON text; OpenAI `function` wrappers become `name`, `description`, `input_schema`.
fn tool_json(t: &Value) -> String {
    let t = match t.get("function").and_then(Value::as_object) {
        Some(f) => {
            let mut m = Map::new();
            for (from, to) in [
                ("name", "name"),
                ("description", "description"),
                ("parameters", "input_schema"),
            ] {
                if let Some(v) = f.get(from) {
                    m.insert(to.to_string(), v.clone());
                }
            }
            Value::Object(m)
        }
        None => t.clone(),
    };
    let mut s = String::new();
    dump_ctx(&t, Ctx::Tool, &mut s);
    s
}

/// The role blocks of a request, in the release's order: system prompt, tool definitions, messages.
pub fn render_v2(request: &Value) -> Vec<String> {
    let mut names = HashMap::new();
    let mut blocks = Vec::new();
    match request.get("system") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) if s.is_empty() => {}
        Some(sys) => {
            let parts = content_parts(sys, &mut names);
            if !parts.is_empty() {
                blocks.push(frame("system", &parts));
            }
        }
    }
    if let Some(tools) = request
        .get("tools")
        .and_then(Value::as_array)
        .filter(|t| !t.is_empty())
    {
        let parts: Vec<Value> = tools.iter().map(|t| text_part(&tool_json(t))).collect();
        blocks.push(frame("system", &parts));
    }
    for m in messages(request) {
        let role = match m.get("role") {
            Some(Value::String(r)) => r.clone(),
            Some(other) => py_json_dumps(other),
            None => String::new(),
        };
        let parts = content_parts(m.get("content").unwrap_or(&Value::Null), &mut names);
        blocks.push(frame(&role, &parts));
    }
    blocks
}
