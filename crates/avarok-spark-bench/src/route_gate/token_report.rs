// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from the ARC-1.0 release files (goldens.jsonl, goldens.signals.jsonl), Copyright 2026 Atlas Futures,
// Inc., licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0). By Atlas contributors.
// See RAYLINE-ROUTING-ATTRIBUTION.md.
//! Responsibility: the serializer length check (`tokens`): our token count of every golden request under a serializer
//! against the release's `expected.token_count`, per case and grouped by request shape, for the goldens file and the
//! signals file (no model needed, only the tokenizer).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::{Value, json};

use super::canonical::Serializer;
use super::goldens::load_jsonl;
use super::serializer::{HfTokenizer, TextTokenizer, encode_blocks};

/// One golden request: the release's token count against a serializer's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenCase {
    pub case_id: String,
    pub golden: i64,
    pub ours: i64,
    pub delta: i64,
    pub turns: usize,
    /// The request shape (roles, tools, tool use, tool results, system prompt, attachments, length bucket).
    pub shape: String,
}

impl TokenCase {
    /// Our count equals the release's.
    pub fn matches(&self) -> bool {
        self.delta == 0
    }
}

/// [`token_report`] result.
#[derive(Clone, Debug)]
pub struct TokenReport {
    pub cases: Vec<TokenCase>,
    pub distinct_deltas: Vec<i64>,
    pub distinct_deltas_single_message: Vec<i64>,
    pub constant_offset: bool,
}

impl TokenReport {
    /// How many cases match exactly.
    pub fn matched(&self) -> usize {
        self.cases.iter().filter(|c| c.matches()).count()
    }

    fn to_json(&self) -> Value {
        json!({
            "cases": self.cases.iter().map(|c| json!({
                "case_id": c.case_id, "golden": c.golden, "ours": c.ours, "delta": c.delta, "turns": c.turns,
                "shape": c.shape, "match": c.matches(),
            })).collect::<Vec<_>>(),
            "matched": self.matched(),
            "total": self.cases.len(),
            "distinct_deltas": self.distinct_deltas,
            "distinct_deltas_single_message": self.distinct_deltas_single_message,
            "constant_offset": self.constant_offset,
        })
    }
}

/// The request object of a golden record (`messages`, `tools`, ...); signals records have the same shape.
pub fn request_of(g: &Value) -> &Value {
    &g["request"]["request"]
}

fn messages_of(g: &Value) -> Vec<Value> {
    request_of(g)["messages"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

fn part_types(content: &Value) -> Vec<String> {
    match content {
        Value::Array(a) => a
            .iter()
            .map(|p| {
                p.get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("text")
                    .to_string()
            })
            .collect(),
        _ => vec!["text".to_string()],
    }
}

/// Request shape of a golden: roles present, tool definitions, tool use, tool results, system prompt, attachments and a
/// length bucket of the release's token count.
pub fn request_shape(g: &Value) -> String {
    let req = request_of(g);
    let msgs = messages_of(g);
    let mut roles: Vec<&str> = msgs.iter().filter_map(|m| m["role"].as_str()).collect();
    roles.sort_unstable();
    roles.dedup();
    let types: Vec<String> = msgs
        .iter()
        .flat_map(|m| part_types(&m["content"]))
        .collect();
    let has = |t: &str| types.iter().any(|x| x == t);
    let attach = types
        .iter()
        .any(|t| !matches!(t.as_str(), "text" | "tool_use" | "tool_result"))
        || req.get("attachments").is_some();
    let yn = |b: bool| if b { "yes" } else { "no" };
    let tokens = g["expected"]["token_count"].as_i64().unwrap_or(0);
    let bucket = match tokens {
        ..=32 => "le32",
        33..=64 => "le64",
        65..=128 => "le128",
        129..=256 => "le256",
        _ => "gt256",
    };
    format!(
        "roles={} turns={} tools_defined={} tool_use={} tool_result={} system={} attachments={} len={bucket}",
        roles.join("+"),
        msgs.len(),
        yn(req["tools"].as_array().is_some_and(|t| !t.is_empty())),
        yn(has("tool_use")),
        yn(has("tool_result")),
        yn(req.get("system").is_some_and(|s| !s.is_null())),
        yn(attach),
    )
}

/// Our token count (stand-in serializer) for every golden request against the release's `expected.token_count`.
pub fn token_report<T: TextTokenizer>(
    tok: &T,
    goldens: &[Value],
    shell: Option<&str>,
) -> super::error::GResult<TokenReport> {
    token_report_for(Serializer::StandIn, tok, goldens, shell)
}

/// Our token count for every golden request under `ser`, against the release's `expected.token_count`.
pub fn token_report_for<T: TextTokenizer>(
    ser: Serializer,
    tok: &T,
    goldens: &[Value],
    shell: Option<&str>,
) -> super::error::GResult<TokenReport> {
    let mut cases = Vec::new();
    for g in goldens {
        let ours = encode_blocks(tok, &ser.render(request_of(g)), shell)?
            .ids
            .len() as i64;
        let golden = g["expected"]["token_count"].as_i64().unwrap_or(0);
        cases.push(TokenCase {
            case_id: g["case_id"].as_str().unwrap_or_default().to_string(),
            golden,
            ours,
            delta: golden - ours,
            turns: messages_of(g).len(),
            shape: request_shape(g),
        });
    }
    let sorted = |it: Vec<i64>| {
        let mut v = it;
        v.sort_unstable();
        v.dedup();
        v
    };
    let distinct = sorted(cases.iter().map(|c| c.delta).collect());
    let single = sorted(
        cases
            .iter()
            .filter(|c| c.turns == 1)
            .map(|c| c.delta)
            .collect(),
    );
    let constant_offset = single.len() == 1 && distinct.len() == 1;
    Ok(TokenReport {
        cases,
        distinct_deltas: distinct,
        distinct_deltas_single_message: single,
        constant_offset,
    })
}

fn read_shell(path: &Option<PathBuf>) -> Result<Option<String>> {
    path.as_ref()
        .map(|p| std::fs::read_to_string(p).with_context(|| format!("read {}", p.display())))
        .transpose()
}

/// Prints every case (our count, the release's count, the delta) grouped by request shape.
fn print_token_groups(rep: &TokenReport) {
    let mut groups: BTreeMap<&str, Vec<&TokenCase>> = BTreeMap::new();
    for c in &rep.cases {
        groups.entry(c.shape.as_str()).or_default().push(c);
    }
    for (shape, cases) in &groups {
        let ok = cases.iter().filter(|c| c.matches()).count();
        println!("{shape}: {ok}/{} exact", cases.len());
        for c in cases {
            println!(
                "    {:<28} ours {:>4}  release {:>4}  delta {:>+4}{}",
                c.case_id,
                c.ours,
                c.golden,
                c.delta,
                if c.matches() { "" } else { "  <-- MISMATCH" }
            );
        }
    }
}

/// The signals file carries the same requests without prices: its token counts come from the golden of the same case.
fn signals_token_check<T: TextTokenizer>(
    ser: Serializer,
    tok: &T,
    goldens: &[Value],
    release: &Path,
) -> Result<Option<(usize, usize, usize)>> {
    let path = release.join("goldens.signals.jsonl");
    if !path.exists() {
        return Ok(None);
    }
    let (mut exact, mut total, mut other_request) = (0, 0, 0);
    for s in load_jsonl(&path)? {
        let id = s["case_id"].as_str().unwrap_or_default();
        let Some(g) = goldens.iter().find(|g| g["case_id"] == id) else {
            continue;
        };
        if request_of(&s) != request_of(g) {
            other_request += 1;
        }
        let ours = encode_blocks(tok, &ser.render(request_of(&s)), None)?
            .ids
            .len() as i64;
        total += 1;
        exact += usize::from(ours == g["expected"]["token_count"].as_i64().unwrap_or(-1));
    }
    Ok(Some((exact, total, other_request)))
}

/// `tokens`: per golden request our count, the release's count and the delta, grouped by request shape.
pub fn cmd_tokens(
    release: &Path,
    shell_file: &Option<PathBuf>,
    out: &Option<PathBuf>,
    ser: Serializer,
) -> Result<i32> {
    let tok = HfTokenizer::from_release(release)?;
    let goldens = load_jsonl(&release.join("goldens.jsonl"))?;
    let shell = read_shell(shell_file)?;
    let rep = token_report_for(ser, &tok, &goldens, shell.as_deref())?;
    print_token_groups(&rep);
    println!(
        "route_gate tokens: serializer {}; {}/{} golden requests exact; delta (release minus ours): all {:?}, single-message {:?}",
        ser.name(),
        rep.matched(),
        rep.cases.len(),
        rep.distinct_deltas,
        rep.distinct_deltas_single_message
    );
    let sig = signals_token_check(ser, &tok, &goldens, release)?;
    if let Some((exact, total, other)) = sig {
        println!(
            "route_gate tokens: signals file {exact}/{total} exact ({other} requests differ from the golden's)"
        );
    }
    if let Some(p) = out {
        std::fs::write(p, serde_json::to_string_pretty(&rep.to_json())?)?;
    }
    let all_exact = rep.matched() == rep.cases.len() && sig.is_none_or(|(e, t, _)| e == t);
    Ok(if all_exact { 0 } else { 3 })
}
