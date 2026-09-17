// SPDX-License-Identifier: AGPL-3.0-only

//! The MLPerf Agentic Inference trajectory file: flat JSONL, one row per
//! message, rows per conversation contiguous and turn-ordered from 1.
//!
//! This is a port of the upstream loader semantics
//! (`AgenticInferenceDataset._build_conversation_metadata` and its
//! validators, mlcommons/endpoints@7935df4) against the format the upstream
//! README documents. It is written to the DOCUMENTED format because the
//! official dataset itself does not exist yet — MLCommons' README says it
//! "can be downloaded from MLCommons storage (link TBD)". Until that file
//! ships, this loader is exercised only by its tests; nothing here has met
//! real data, and the first calibration run may surface format corners the
//! documentation does not.
//!
//! # Teacher forcing is the whole design
//!
//! The prompt for client turn N is the RECORDED history (system prompt +
//! recorded user/tool/assistant rows, including recorded `reasoning_content`
//! and `tool_calls`) plus the current client message, prebuilt here at load
//! time. The model's own output is scored but never fed into later turns —
//! so a run's per-turn context lengths are a pure function of the dataset,
//! which is what makes a pinned draw comparable across engines at all.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use super::scoring::{Domain, domain_of};

/// One client turn: what is sent, and the recorded assistant row that scores it.
#[derive(Clone, Debug)]
pub struct ClientTurn {
    /// The client row's turn number — the key the scorer joins on.
    pub turn: i64,
    /// Prebuilt prompt: system + recorded history + this client message.
    pub messages: Vec<Value>,
    /// The recorded assistant row that immediately follows this client row,
    /// raw. `None` for a trailing client row with no assistant after it.
    pub ground_truth: Option<Value>,
}

#[derive(Clone, Debug)]
pub struct Conversation {
    pub id: String,
    pub domain: Domain,
    /// Tool schemas from the conversation's first `user` row, sent with every
    /// request (upstream propagates them the same way).
    pub tools: Option<Value>,
    pub client_turns: Vec<ClientTurn>,
}

/// How many whole trajectories to take per domain, FIRST-K in file order —
/// deterministic and RNG-free, like the BFCL draw. `0` means all of that
/// domain. Partial conversations are never drawn: under teacher forcing a
/// half trajectory silently shifts every remaining turn's context length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrawSpec {
    pub coding: usize,
    pub workflow: usize,
}

impl DrawSpec {
    pub fn all() -> Self {
        Self {
            coding: 0,
            workflow: 0,
        }
    }
}

/// Load the trajectory file and apply the draw.
pub fn load(path: &Path, spec: &DrawSpec) -> Result<Vec<Conversation>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let conversations = parse(&text)?;
    let mut taken = Vec::new();
    let (mut coding_taken, mut workflow_taken) = (0usize, 0usize);
    for conv in conversations {
        let (count, limit) = match conv.domain {
            Domain::Coding => (&mut coding_taken, spec.coding),
            Domain::Workflow => (&mut workflow_taken, spec.workflow),
        };
        if limit == 0 || *count < limit {
            *count += 1;
            taken.push(conv);
        }
    }
    if taken.is_empty() {
        bail!("the draw selected no trajectories");
    }
    Ok(taken)
}

/// The draw fingerprint: SHA256 over the ordered conversations and every
/// prompt/scoring field they contain. Two runs with equal fingerprints
/// replayed the same teacher-forced messages, tools, and ground truth in the
/// same order; a differing fingerprint means the numbers are not comparable,
/// whatever the sample counts say. This is the content identity the BFCL legs
/// never had (their provisioning digest is computed and then dropped),
/// recorded here into the gate record itself.
pub fn draw_fingerprint(conversations: &[Conversation]) -> String {
    let mut digest = Sha256::new();
    let update = |digest: &mut Sha256, bytes: &[u8]| {
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    };
    for conv in conversations {
        digest.update(b"conversation\0");
        update(&mut digest, conv.id.as_bytes());
        update(
            &mut digest,
            &serde_json::to_vec(&conv.tools).expect("JSON values serialize"),
        );
        for turn in &conv.client_turns {
            digest.update(b"turn\0");
            digest.update(turn.turn.to_le_bytes());
            update(
                &mut digest,
                &serde_json::to_vec(&turn.messages).expect("JSON values serialize"),
            );
            update(
                &mut digest,
                &serde_json::to_vec(&turn.ground_truth).expect("JSON values serialize"),
            );
        }
    }
    format!("{:x}", digest.finalize())
}

fn parse(text: &str) -> Result<Vec<Conversation>> {
    // Group contiguous rows by conversation, enforcing the upstream
    // validators: non-empty printable-ASCII ids, contiguity, turns exactly
    // 1..=N, and the user/assistant/tool role grammar.
    let mut order: Vec<String> = Vec::new();
    let mut groups: std::collections::BTreeMap<String, Vec<Map<String, Value>>> =
        Default::default();
    let mut seen_closed: BTreeSet<String> = BTreeSet::new();
    let mut last_id: Option<String> = None;

    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: Map<String, Value> =
            serde_json::from_str(line).with_context(|| format!("line {}: malformed row", i + 1))?;
        let id = match row.get("conversation_id") {
            Some(Value::String(s)) if !s.is_empty() => s.clone(),
            other => bail!(
                "line {}: conversation_id must be a non-empty string, got {}",
                i + 1,
                other
                    .map(Value::to_string)
                    .unwrap_or_else(|| "nothing".into())
            ),
        };
        if id.bytes().any(|b| !(32..=126).contains(&b)) {
            bail!(
                "line {}: conversation_id {id:?} has non-printable or non-ASCII characters",
                i + 1
            );
        }
        if last_id.as_ref() != Some(&id) {
            if seen_closed.contains(&id) {
                bail!(
                    "rows for conversation {id:?} are not consecutive — the loader takes the \
                     first K per domain in file order, so interleaving would silently change \
                     which trajectories a draw selects"
                );
            }
            if let Some(prev) = last_id.replace(id.clone()) {
                seen_closed.insert(prev);
            }
            order.push(id.clone());
        }
        groups.entry(id).or_default().push(row);
    }

    let mut out = Vec::new();
    for id in order {
        let rows = groups.remove(&id).expect("grouped above");
        out.push(build_conversation(id, rows)?);
    }
    if out.is_empty() {
        bail!("the trajectory file holds no rows");
    }
    Ok(out)
}

fn turn_of(row: &Map<String, Value>) -> Result<i64> {
    row.get("turn")
        .and_then(Value::as_i64)
        .context("row is missing an integer 'turn'")
}

fn build_conversation(id: String, rows: Vec<Map<String, Value>>) -> Result<Conversation> {
    for row in &rows {
        turn_of(row).with_context(|| format!("conversation {id:?}"))?;
    }
    let turns: Vec<i64> = rows.iter().map(|r| turn_of(r).unwrap()).collect();
    if turns.iter().enumerate().any(|(i, &t)| t != i as i64 + 1) {
        bail!(
            "conversation {id:?}: turn numbers must be exactly 1..={} in file order, got \
             {turns:?}",
            rows.len()
        );
    }
    validate_roles(&id, &rows)?;

    let system = rows
        .iter()
        .find_map(|r| {
            r.get("system")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
        })
        .map(str::to_string);
    let tools = rows
        .iter()
        .find(|r| r.get("role").and_then(Value::as_str) == Some("user"))
        .and_then(|r| r.get("tools"))
        .filter(|t| !t.is_null())
        .cloned();

    // Single pass in turn order, carrying the running history — the port of
    // upstream `_build_conversation_metadata`. Client rows snapshot
    // (history + current) BEFORE extending; assistant rows only extend.
    let mut history: Vec<Value> = Vec::new();
    if let Some(system) = &system {
        history.push(json!({"role": "system", "content": system}));
    }
    let mut client_turns: Vec<ClientTurn> = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let role = row.get("role").and_then(Value::as_str).unwrap_or_default();
        let row_msgs = format_row(&id, row)?;
        if role == "user" || role == "tool" {
            let mut messages = history.clone();
            messages.extend(row_msgs.iter().cloned());
            // The scorer's ground-truth pairing: the assistant row that
            // immediately follows this client row in turn order.
            let ground_truth = rows.get(i + 1).and_then(|next| {
                (next.get("role").and_then(Value::as_str) == Some("assistant"))
                    .then(|| Value::Object(next.clone()))
            });
            client_turns.push(ClientTurn {
                turn: turn_of(row).unwrap(),
                messages,
                ground_truth,
            });
        }
        history.extend(row_msgs);
    }

    Ok(Conversation {
        domain: domain_of(&id),
        id,
        tools,
        client_turns,
    })
}

/// Format one row into OpenAI message(s): `tool_results` rows expand to one
/// tool message per result; everything else passes the recorded fields
/// through, with `content: null` filled in on tool-calling assistant rows.
fn format_row(id: &str, row: &Map<String, Value>) -> Result<Vec<Value>> {
    if let Some(results) = row.get("tool_results").and_then(Value::as_array)
        && !results.is_empty()
    {
        let mut msgs = Vec::new();
        for (i, result) in results.iter().enumerate() {
            let (Some(tool_call_id), Some(content)) =
                (result.get("tool_call_id"), result.get("content"))
            else {
                bail!(
                    "conversation {id:?} turn {}: tool_results[{i}] needs tool_call_id and content",
                    turn_of(row).unwrap_or_default()
                );
            };
            msgs.push(json!({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }));
        }
        return Ok(msgs);
    }
    let mut msg = Map::new();
    for key in [
        "role",
        "content",
        "name",
        "tool_calls",
        "tool_results",
        "reasoning_content",
    ] {
        if let Some(v) = row.get(key).filter(|v| !v.is_null()) {
            msg.insert(key.to_string(), v.clone());
        }
    }
    if msg.get("role").and_then(Value::as_str) == Some("assistant")
        && msg.contains_key("tool_calls")
        && !msg.contains_key("content")
    {
        msg.insert("content".into(), Value::Null);
    }
    Ok(if msg.contains_key("role") {
        vec![Value::Object(msg)]
    } else {
        Vec::new()
    })
}

/// The upstream role grammar: a conversation starts with `user`; `user` is
/// answered by `assistant`; `assistant` is followed by `tool` or `user`;
/// `tool` by `assistant` or `user`. A `tool` row must follow an assistant
/// that made tool calls, and must carry a non-empty `tool_results` list.
fn validate_roles(id: &str, rows: &[Map<String, Value>]) -> Result<()> {
    let mut state = "start";
    let mut prev_assistant_had_tool_calls = false;
    for row in rows {
        let role = row
            .get("role")
            .and_then(Value::as_str)
            .with_context(|| format!("conversation {id:?}: row without a role"))?;
        let valid: &[&str] = match state {
            "start" => &["user"],
            "user" => &["assistant"],
            "assistant" => &["tool", "user"],
            "tool" => &["assistant", "user"],
            _ => unreachable!(),
        };
        if !valid.contains(&role) {
            bail!(
                "conversation {id:?} turn {}: got role {role:?} after {state:?}",
                turn_of(row).unwrap_or_default()
            );
        }
        if role == "tool" {
            if state == "assistant" && !prev_assistant_had_tool_calls {
                bail!(
                    "conversation {id:?} turn {}: a tool row must follow an assistant row \
                     with non-empty tool_calls",
                    turn_of(row).unwrap_or_default()
                );
            }
            if row
                .get("tool_results")
                .and_then(Value::as_array)
                .is_none_or(|r| r.is_empty())
            {
                bail!(
                    "conversation {id:?} turn {}: tool rows need a non-empty tool_results list",
                    turn_of(row).unwrap_or_default()
                );
            }
        }
        if role == "assistant" {
            prev_assistant_had_tool_calls = row
                .get("tool_calls")
                .and_then(Value::as_array)
                .is_some_and(|c| !c.is_empty());
        }
        state = match role {
            "user" => "user",
            "assistant" => "assistant",
            _ => "tool",
        };
    }
    Ok(())
}

#[cfg(test)]
#[path = "dataset_tests.rs"]
mod tests;
