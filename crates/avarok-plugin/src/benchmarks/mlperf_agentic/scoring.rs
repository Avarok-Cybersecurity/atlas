// SPDX-License-Identifier: AGPL-3.0-only

//! Rust port of the MLPerf Agentic Inference inline scorer's comparison rules.
//!
//! Source: `AgenticInferenceInlineScorer` in mlcommons/endpoints@**7935df4**,
//! `src/inference_endpoint/evaluation/scoring.py`. Two rules and nothing else:
//!
//! * **Workflow** (`^sim_\d+$` conversation ids): 1.0 iff the intent code
//!   extracted from the model's text is in the ground truth's `intent_codes`.
//! * **Coding** (everything else): multiset IoU over normalized bash
//!   executables from `bash` tool calls.
//!
//! # Parity is proven, not assumed
//!
//! Every function here is pinned by `assets/mlperf-agentic/parity_fixtures.json`,
//! whose expected values were produced by EXECUTING the upstream class at the
//! commit above (see `gen_parity_fixtures.py` beside it) — including one case
//! per alias-table entry and per shell wrapper, so the whole table is pinned
//! rather than spot-checked. That matters because several upstream behaviours
//! are not what a reasonable porter would guess: `"reintent: I042"` still
//! scores (the bare-token fallback rescues it), unknown executables vanish
//! from BOTH sides of the IoU, and a lone `&` is not a command separator.
//! A port that guessed those wrong would emit numbers that look like MLPerf
//! inline accuracy and are not.
//!
//! # Known, deliberate divergences (all degenerate inputs)
//!
//! * Python's `\s` also matches `\x1c`-`\x1f`; [`char::is_whitespace`] does
//!   not. Affects only control characters between `intent:` and the code.
//! * Python's `\\.` in the double-quote pattern does not match
//!   backslash-newline (no DOTALL); this port treats any backslash-escaped
//!   character as escaped. Affects only a backslash at end-of-line inside an
//!   unterminated double quote.

use std::collections::BTreeMap;

use serde_json::Value;

/// Which scoring rule applies, decided by conversation id exactly as upstream:
/// `^sim_\d+$` is workflow, anything else is coding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    Workflow,
    Coding,
}

pub fn domain_of(conversation_id: &str) -> Domain {
    match conversation_id.strip_prefix("sim_") {
        Some(rest) if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) => {
            Domain::Workflow
        }
        _ => Domain::Coding,
    }
}

/// Upstream `_EXECUTABLE_ALIASES`, copied verbatim (58 entries, declaration
/// order preserved). Executables NOT in this table are dropped from the
/// multiset entirely — they are invisible to the IoU, on both sides.
const EXECUTABLE_ALIASES: [(&str, &str); 58] = [
    ("python", "python"),
    ("python2", "python"),
    ("python3", "python"),
    ("py", "python"),
    ("pip", "pip"),
    ("pip3", "pip"),
    ("pytest", "pytest"),
    ("pylint", "pylint"),
    ("sphinx-build", "sphinx"),
    ("sphinx-quickstart", "sphinx"),
    ("cython", "cython"),
    ("make", "make"),
    ("conda", "conda"),
    ("cat", "cat"),
    ("head", "head"),
    ("tail", "tail"),
    ("less", "cat"),
    ("more", "cat"),
    ("wc", "wc"),
    ("diff", "diff"),
    ("grep", "grep"),
    ("egrep", "grep"),
    ("fgrep", "grep"),
    ("rg", "grep"),
    ("ag", "grep"),
    ("sed", "sed"),
    ("awk", "awk"),
    ("gawk", "awk"),
    ("tr", "tr"),
    ("sort", "sort"),
    ("uniq", "uniq"),
    ("cut", "cut"),
    ("find", "find"),
    ("ls", "ls"),
    ("locate", "find"),
    ("xargs", "xargs"),
    ("cp", "cp"),
    ("mv", "mv"),
    ("rm", "rm"),
    ("mkdir", "mkdir"),
    ("touch", "touch"),
    ("tee", "tee"),
    ("source", "source"),
    (".", "source"),
    ("which", "which"),
    ("alias", "alias"),
    ("unset", "unset"),
    ("export", "export"),
    ("git", "git"),
    ("curl", "curl"),
    ("wget", "curl"),
    ("true", "true"),
    ("false", "false"),
    ("timeout", "timeout"),
    ("date", "date"),
    ("apt-get", "apt"),
    ("apt", "apt"),
    ("yum", "yum"),
];

/// Upstream `_SHELL_WRAPPERS`: tokens stripped from the front of a stage
/// before the executable is read.
const SHELL_WRAPPERS: [&str; 6] = ["env", "time", "nice", "sudo", "exec", "command"];

#[cfg(test)]
pub(super) fn alias_table() -> &'static [(&'static str, &'static str)] {
    &EXECUTABLE_ALIASES
}

/// Ground-truth intent codes from an expected assistant turn: strings from
/// `intent_codes`, upper-cased, empty and non-string entries dropped. An
/// empty set means the turn has no workflow ground truth and is excluded
/// from the denominator.
pub fn ground_truth_intents(turn: &Value) -> Vec<String> {
    let Some(codes) = turn.get("intent_codes").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out: Vec<String> = codes
        .iter()
        .filter_map(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_uppercase)
        .collect();
    out.sort();
    out.dedup();
    out
}

/// The model's intent code: the explicit `intent: I123` form searched over
/// `reasoning_content` then `content` FIRST — the explicit pass covers both
/// fields before the fallback runs over either — then the LAST bare `I123`
/// token as fallback. The explicit form is case-insensitive and upper-cased;
/// the bare fallback is case-sensitive (upstream compiles it without
/// IGNORECASE, deliberately or not — encoded in the fixtures either way).
pub fn model_intent(turn: &Value) -> Option<String> {
    let fields = ["reasoning_content", "content"];
    for field in fields {
        if let Some(text) = turn.get(field).and_then(Value::as_str)
            && let Some(code) = explicit_intent(text)
        {
            return Some(code);
        }
    }
    for field in fields {
        if let Some(text) = turn.get(field).and_then(Value::as_str)
            && let Some(code) = last_bare_intent(text)
        {
            return Some(code);
        }
    }
    None
}

/// Python's `\w` for the `\b` boundaries: alphanumerics plus underscore.
fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// `\bintent:\s*(I\d{3})\b`, IGNORECASE, first match.
fn explicit_intent(text: &str) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    for start in 0..n {
        if start > 0 && is_word(chars[start - 1]) {
            continue;
        }
        let keyword = "intent:";
        if start + keyword.len() > n
            || !chars[start..start + keyword.len()]
                .iter()
                .zip(keyword.chars())
                .all(|(&c, k)| c.to_ascii_lowercase() == k)
        {
            continue;
        }
        let mut i = start + keyword.len();
        while i < n && chars[i].is_whitespace() {
            i += 1;
        }
        if let Some(code) = intent_code_at(&chars, i) {
            return Some(code);
        }
    }
    None
}

/// `\bI(\d{3})\b` (case-sensitive), LAST match.
fn last_bare_intent(text: &str) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut last = None;
    for start in 0..chars.len() {
        if chars[start] != 'I' || (start > 0 && is_word(chars[start - 1])) {
            continue;
        }
        if let Some(code) = intent_code_at(&chars, start) {
            last = Some(code);
        }
    }
    last
}

/// `[Ii]\d{3}` at `i` with a word boundary after the third digit, upper-cased.
fn intent_code_at(chars: &[char], i: usize) -> Option<String> {
    if i + 4 > chars.len()
        || !matches!(chars[i], 'I' | 'i')
        || !chars[i + 1..i + 4].iter().all(|c| c.is_ascii_digit())
        || chars.get(i + 4).copied().is_some_and(is_word)
    {
        return None;
    }
    Some(format!("I{}{}{}", chars[i + 1], chars[i + 2], chars[i + 3]))
}

/// Normalized bash executables from a turn's tool calls, upstream
/// `_bash_actions`: only `function.name == "bash"` calls count; arguments may
/// be an object or a JSON string (malformed JSON skips that call);
/// `command` is preferred over `cmd` with Python truthiness (an empty
/// `command` falls through). Per command: quoted spans stripped, split on
/// `||` / `|` / `&&` / `;` / newline, leading env assignments and shell
/// wrappers dropped, executable basenamed, lowercased,
/// version-suffix-stripped and mapped through the alias table.
pub fn bash_actions(turn: &Value) -> Vec<String> {
    let Some(calls) = turn.get("tool_calls").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut actions = Vec::new();
    for call in calls {
        let Some(function) = call.get("function").filter(|f| f.is_object()) else {
            continue;
        };
        if function.get("name").and_then(Value::as_str) != Some("bash") {
            continue;
        }
        let arguments = match function.get("arguments") {
            Some(Value::String(raw)) => match serde_json::from_str::<Value>(raw) {
                Ok(parsed) => parsed,
                Err(_) => continue,
            },
            Some(v) => v.clone(),
            None => continue,
        };
        if !arguments.is_object() {
            continue;
        }
        // `args.get("command") or args.get("cmd")`: Python truthiness, so an
        // empty or null `command` falls through to `cmd`.
        let command = [arguments.get("command"), arguments.get("cmd")]
            .into_iter()
            .flatten()
            .find(|v| py_truthy(v));
        let Some(command) = command.and_then(Value::as_str) else {
            continue;
        };
        collect_stage_actions(command, &mut actions);
    }
    actions
}

/// Python truthiness for a JSON value, as `or` sees it.
fn py_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn collect_stage_actions(command: &str, actions: &mut Vec<String>) {
    let unquoted = strip_quoted_spans(command);
    for stage in split_separators(&unquoted) {
        let mut tokens = stage.split_whitespace();
        let executable = loop {
            match tokens.next() {
                None => break None,
                Some(t) if is_env_assignment(t) || SHELL_WRAPPERS.contains(&t) => continue,
                Some(t) => break Some(t),
            }
        };
        let Some(executable) = executable else {
            continue;
        };
        let base = executable
            .rsplit('/')
            .next()
            .unwrap_or(executable)
            .to_lowercase();
        let stripped = strip_version_suffix(&base);
        if let Some((_, alias)) = EXECUTABLE_ALIASES.iter().find(|(k, _)| *k == stripped) {
            actions.push((*alias).to_string());
        }
    }
}

/// Upstream `_QUOTED_RE.sub(" ", command)`: `'…'`, `"…"` (with backslash
/// escapes) and `` `…` `` spans each replaced by one space. An unterminated
/// quote matches nothing and the quote character stays in the text.
fn strip_quoted_spans(command: &str) -> String {
    let chars: Vec<char> = command.chars().collect();
    let mut out = String::with_capacity(command.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let close = match c {
            '\'' | '`' => chars[i + 1..]
                .iter()
                .position(|&x| x == c)
                .map(|p| i + 1 + p),
            '"' => {
                let mut j = i + 1;
                loop {
                    match chars.get(j) {
                        None => break None,
                        Some('\\') => j += 2,
                        Some('"') => break Some(j),
                        Some(_) => j += 1,
                    }
                }
            }
            _ => None,
        };
        match close {
            Some(j) => {
                out.push(' ');
                i = j + 1;
            }
            None => {
                out.push(c);
                i += 1;
            }
        }
    }
    out
}

/// Upstream `_COMMAND_SEPARATOR_RE.split`: `||`, `|`, `&&`, `;`, newline.
/// A lone `&` is deliberately NOT a separator — upstream's regex has none.
fn split_separators(text: &str) -> Vec<String> {
    let mut stages = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let two = (chars[i], chars.get(i + 1).copied());
        match two {
            ('|', Some('|')) | ('&', Some('&')) => {
                stages.push(std::mem::take(&mut current));
                i += 2;
            }
            ('|' | ';' | '\n', _) => {
                stages.push(std::mem::take(&mut current));
                i += 1;
            }
            (c, _) => {
                current.push(c);
                i += 1;
            }
        }
    }
    stages.push(current);
    stages
}

/// Upstream `_ENV_ASSIGNMENT_RE`: `^[A-Za-z_][A-Za-z0-9_]*=`.
fn is_env_assignment(token: &str) -> bool {
    let bytes = token.as_bytes();
    if bytes.is_empty() || !(bytes[0].is_ascii_alphabetic() || bytes[0] == b'_') {
        return false;
    }
    bytes[1..]
        .iter()
        .position(|&b| !(b.is_ascii_alphanumeric() || b == b'_'))
        .is_some_and(|p| bytes[1 + p] == b'=')
}

/// Upstream `_PY_VERSION_SUFFIX_RE.sub("", …)`: `\.\d+(\.\d+)?$`, leftmost
/// match — i.e. at most the last TWO trailing `.N` groups come off in one
/// match ("python3.11" → "python3", "python3.1.2.3" → "python3.1").
fn strip_version_suffix(name: &str) -> &str {
    let bytes = name.as_bytes();
    'outer: for start in 0..bytes.len() {
        if bytes[start] != b'.' {
            continue;
        }
        let mut i = start + 1;
        for groups_left in [true, false] {
            let digits_start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if i == digits_start {
                continue 'outer;
            }
            if i == bytes.len() {
                return &name[..start];
            }
            if !groups_left || bytes[i] != b'.' {
                continue 'outer;
            }
            i += 1;
        }
    }
    name
}

/// One turn's score. `gt_turn` must be scorable (see [`has_ground_truth`]);
/// a model turn that produced nothing scores 0 and STAYS in the denominator —
/// that is the upstream failure semantics, and dropping such turns instead is
/// exactly the denominator trick the 2026-08-02 BFCL re-score caught.
pub fn score_turn(domain: Domain, gt_turn: &Value, model_turn: &Value) -> f64 {
    match domain {
        Domain::Workflow => {
            let gt = ground_truth_intents(gt_turn);
            match model_intent(model_turn) {
                Some(code) if gt.contains(&code) => 1.0,
                _ => 0.0,
            }
        }
        Domain::Coding => multiset_iou(&bash_actions(gt_turn), &bash_actions(model_turn)),
    }
}

/// Whether an expected assistant turn carries scorable ground truth. Turns
/// without it are EXCLUDED from the denominator (upstream lists them as
/// `excluded_turns`), unlike issued-but-unanswered turns, which stay at 0.
pub fn has_ground_truth(domain: Domain, gt_turn: &Value) -> bool {
    match domain {
        Domain::Workflow => !ground_truth_intents(gt_turn).is_empty(),
        Domain::Coding => !bash_actions(gt_turn).is_empty(),
    }
}

/// `sum(gt ∩ model) / sum(gt ∪ model)` over action counts. The caller
/// guarantees `gt` is non-empty, so the union cannot be zero.
pub fn multiset_iou(gt: &[String], model: &[String]) -> f64 {
    let mut counts: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for a in gt {
        counts.entry(a).or_default().0 += 1;
    }
    for a in model {
        counts.entry(a).or_default().1 += 1;
    }
    let (mut intersection, mut union) = (0usize, 0usize);
    for (g, m) in counts.values() {
        intersection += g.min(m);
        union += g.max(m);
    }
    intersection as f64 / union as f64
}

#[cfg(test)]
#[path = "scoring_tests.rs"]
mod tests;
