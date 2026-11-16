// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from the ARC-1.0 release documentation and from rayline-ai/vllm-arc (examples/readouts.py), Copyright
// 2026 Atlas Futures, Inc., licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
// Changes: the stand-in serializer here is kept as an option; the reconstructed one is in `canonical.rs`. By Atlas
// contributors. See RAYLINE-ROUTING-ATTRIBUTION.md.
//! Responsibility: the stand-in request serializer, block tokenization with readout offsets, and the Python-style JSON
//! dump (no model needed).
//!
//! The request serializer of the release (`arc-role-blocks-v2`, `canonical_v2`) is closed; the reconstruction that
//! matches the release's token counts is `super::canonical`. [`render_blocks`] is the old stand-in,
//! `<|im_start|>{role}\n{content}<|im_end|>\n` per message, kept as `--serializer standin`: it undercounts the release
//! by 12 tokens per message (see `canonical.rs` for why).

use std::path::Path;

use serde_json::Value;

use super::error::{GResult, GateError, value_err};

pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
/// Readout at the block's final newline token, or at its `<|im_end|>` token.
pub const BLOCK_ENDS: [&str; 2] = ["nl", "im_end"];

/// Python's `str(x)` for a JSON value inside a flattened message.
fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_string(),
        Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        other => other.to_string(),
    }
}

pub(super) fn dump_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut units = [0u16; 2];
                for u in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `json.dumps(v, sort_keys=True)`: sorted keys, `", "` and `": "` separators, non-ASCII escaped.
pub fn py_json_dumps(v: &Value) -> String {
    match v {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            let parts: Vec<String> = keys
                .into_iter()
                .map(|k| {
                    let mut s = String::new();
                    dump_string(k, &mut s);
                    format!("{s}: {}", py_json_dumps(&m[k]))
                })
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(py_json_dumps).collect::<Vec<_>>().join(", ")
        ),
        Value::String(s) => {
            let mut out = String::new();
            dump_string(s, &mut out);
            out
        }
        other => other.to_string(),
    }
}

/// Message content as plain text. Lists of parts (text, tool_use, tool_result) are flattened in order.
pub fn flatten_content(content: &Value) -> String {
    match content {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Object(_) => flatten_parts(std::slice::from_ref(content)),
        Value::Array(a) => flatten_parts(a),
        other => other.to_string(),
    }
}

fn flatten_parts(parts: &[Value]) -> String {
    let rendered: Vec<String> = parts
        .iter()
        .map(|p| match p {
            Value::String(s) => s.clone(),
            Value::Object(m) => {
                let ty = m.get("type").and_then(Value::as_str);
                if ty == Some("text") || (m.contains_key("text") && !m.contains_key("type")) {
                    m.get("text").map_or_else(String::new, py_str)
                } else if ty == Some("tool_use") {
                    let name = m.get("name").map_or_else(String::new, py_str);
                    let input = m
                        .get("input")
                        .cloned()
                        .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
                    format!("[tool_use {name}] {}", py_json_dumps(&input))
                } else if ty == Some("tool_result") {
                    format!(
                        "[tool_result] {}",
                        flatten_content(m.get("content").unwrap_or(&Value::Null))
                    )
                } else {
                    py_json_dumps(p)
                }
            }
            other => py_json_dumps(other),
        })
        .collect();
    rendered.join("\n")
}

/// One role block per message: `<|im_start|>{role}\n{content}<|im_end|>\n` (the question-head manifest rendering).
pub fn render_blocks(messages: &[Value]) -> Vec<String> {
    messages
        .iter()
        .map(|m| {
            let role = m.get("role").map_or_else(String::new, py_str);
            format!(
                "{IM_START}{role}\n{}{IM_END}\n",
                flatten_content(m.get("content").unwrap_or(&Value::Null))
            )
        })
        .collect()
}

/// Anything that turns text into token ids (the real tokenizer, or a fake in tests).
pub trait TextTokenizer {
    type Id: Clone + PartialEq + std::fmt::Debug;
    fn encode(&self, text: &str) -> GResult<Vec<Self::Id>>;
}

/// The release's `tokenizer.json` through the `tokenizers` crate (special tokens are matched, none are added).
pub struct HfTokenizer(tokenizers::Tokenizer);

impl HfTokenizer {
    /// Loads `tokenizer.json` from a release directory.
    pub fn from_release(dir: impl AsRef<Path>) -> GResult<Self> {
        let path = dir.as_ref().join("tokenizer.json");
        tokenizers::Tokenizer::from_file(&path)
            .map(Self)
            .map_err(|e| GateError::Value(format!("{}: {e}", path.display())))
    }
}

impl HfTokenizer {
    /// The tokens of `text` as `(id, piece)` pairs (byte-level pieces, as the vocabulary spells them).
    pub fn pieces(&self, text: &str) -> GResult<Vec<(u32, String)>> {
        let e = self
            .0
            .encode(text, false)
            .map_err(|e| GateError::Value(format!("tokenizer: {e}")))?;
        Ok(e.get_ids()
            .iter()
            .copied()
            .zip(e.get_tokens().iter().cloned())
            .collect())
    }
}

impl TextTokenizer for HfTokenizer {
    type Id = u32;

    fn encode(&self, text: &str) -> GResult<Vec<u32>> {
        self.0
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| GateError::Value(format!("tokenizer: {e}")))
    }
}

/// Token ids of the concatenated blocks and, per block, the offsets of its two candidate readout tokens.
#[derive(Clone, Debug)]
pub struct EncodedBlocks<Id> {
    pub ids: Vec<Id>,
    pub nl: Vec<usize>,
    pub im_end: Vec<usize>,
}

/// Tokenizes every block separately (each is split at its trailing newline so the `<|im_end|>` offset is exact
/// whatever the tokenizer does with newlines). Special tokens delimit blocks, so this equals tokenizing the whole text.
pub fn encode_blocks<T: TextTokenizer>(
    tok: &T,
    blocks: &[String],
    shell_text: Option<&str>,
) -> GResult<EncodedBlocks<T::Id>> {
    let mut ids: Vec<T::Id> = Vec::new();
    if let Some(shell) = shell_text.filter(|s| !s.is_empty()) {
        ids.extend(tok.encode(shell)?);
    }
    let (mut nl, mut im_end) = (Vec::new(), Vec::new());
    let end = format!("{IM_END}\n");
    for b in blocks {
        if !(b.ends_with(&end) && b.starts_with(IM_START)) {
            return value_err("a block must be <|im_start|>...<|im_end|>\\n");
        }
        ids.extend(tok.encode(&b[..b.len() - 1])?);
        im_end.push(ids.len() - 1);
        ids.extend(tok.encode("\n")?);
        nl.push(ids.len() - 1);
    }
    Ok(EncodedBlocks { ids, nl, im_end })
}

impl<Id> EncodedBlocks<Id> {
    /// The readout offsets for `block_end` (`"nl"` or `"im_end"`).
    pub fn pick_offsets(&self, block_end: &str) -> GResult<Vec<usize>> {
        match block_end {
            "nl" => Ok(self.nl.clone()),
            "im_end" => Ok(self.im_end.clone()),
            _ => value_err(format!("block_end must be one of {BLOCK_ENDS:?}")),
        }
    }
}

/// Which per-block row feeds the heads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    Last,
    Mean,
    LastUser,
}

impl Selection {
    pub const ALL: [Selection; 3] = [Self::Last, Self::Mean, Self::LastUser];

    pub fn parse(s: &str) -> GResult<Self> {
        match s {
            "last" => Ok(Self::Last),
            "mean" => Ok(Self::Mean),
            "last_user" => Ok(Self::LastUser),
            other => value_err(format!("unknown row selection {other:?}")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Last => "last",
            Self::Mean => "mean",
            Self::LastUser => "last_user",
        }
    }
}

/// One row from the per-block rows: the last block, the mean of all blocks, or the last user block.
pub fn select_row(rows: &[Vec<f64>], roles: &[String], how: Selection) -> Vec<f64> {
    match how {
        Selection::Last => rows[rows.len() - 1].clone(),
        Selection::Mean => {
            let d = rows[0].len();
            (0..d)
                .map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / rows.len() as f64)
                .collect()
        }
        Selection::LastUser => {
            let idx = roles
                .iter()
                .rposition(|r| r == "user")
                .unwrap_or(rows.len() - 1);
            rows[idx].clone()
        }
    }
}
