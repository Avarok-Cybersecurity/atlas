// SPDX-License-Identifier: AGPL-3.0-only
// T-AT: Atlas-native. The prompt formats are the plain instruction styles these datasets are normally used with;
// the corpus covers GSM8K, MMLU, MMLU-Pro, MATH L4-5, HumanEval, MBPP and GPQA.
//! Responsibility: step 1 of the routing gate (B0): build about 600 labeled prompts from local dataset files, collect
//! the 4B's greedy replies from a running Atlas server (`collect`), grade them (`grade`) and summarize (`summary`).
//!
//! `build`, `grade` and `summary` are CPU only. `collect` talks to an already running Atlas server (our own code, so
//! no download or approval question arises) and belongs to the GPU queue. Datasets are read from local files only.
//! The seeded selection uses CPython's `random.Random` (see `mt.rs`), so a fixed `--seed` selects the same prompts the
//! Python harness selected.

mod collect;
mod label;
mod source;

pub use collect::{CollectArgs, cmd_collect};
pub use label::{GradeArgs, cmd_grade, cmd_hints, cmd_summary, strip_think};
pub use source::{load_source, parse_csv, read_records, sha256_file};

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Result, bail};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use super::grading::{LETTERS, gsm8k_gold, last_boxed};
use super::mt::PyRandom;

pub const VERSION: &str = "route-gate-prep-1";
/// The sources, in the order the default quota lists them.
pub const SOURCES: [&str; 7] = [
    "gsm8k",
    "mmlu",
    "mmlu_pro",
    "math",
    "humaneval",
    "mbpp",
    "gpqa",
];
pub(crate) const BUCKET_SOURCES: [&str; 7] = SOURCES;
pub const DEFAULT_QUOTAS: [(&str, usize); 7] = [
    ("gsm8k", 120),
    ("mmlu", 120),
    ("mmlu_pro", 90),
    ("math", 90),
    ("humaneval", 60),
    ("mbpp", 60),
    ("gpqa", 60),
];

/// `easy` or `hard` bucket of a source.
pub fn bucket(source: &str) -> &'static str {
    match source {
        "gsm8k" | "mmlu" => "easy",
        _ => "hard",
    }
}

/// Dataset license of a source.
pub fn license(source: &str) -> &'static str {
    match source {
        "mbpp" => "CC-BY-4.0",
        "gpqa" => "CC-BY-4.0 (gated; accept the terms; never commit or redistribute the questions)",
        _ => "MIT",
    }
}

/// Generation budget per grading kind (tokens). The 4B runs with thinking off, so replies are short chains of thought.
pub fn max_tokens(kind: &str) -> usize {
    match kind {
        "numeric" => 768,
        "mc" => 640,
        "boxed" => 1024,
        _ => 768,
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Prompt rendering. These are the exact texts the 4B receives AND the texts the router would see.
// ----------------------------------------------------------------------------------------------------------------

/// Python's `str(x)` for a JSON scalar.
pub(crate) fn as_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_string(),
        Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        other => other.to_string(),
    }
}

pub fn mc_prompt(question: &str, choices: &[String]) -> String {
    let mut lines = vec![question.trim().to_string(), String::new()];
    for (i, c) in choices.iter().enumerate() {
        lines.push(format!(
            "{}. {}",
            LETTERS.chars().nth(i).unwrap_or('?'),
            c.trim()
        ));
    }
    lines.push(String::new());
    lines.push(
        "Answer with the letter of the correct option. End your reply with a final line of the form: Answer: <letter>"
            .to_string(),
    );
    lines.join("\n")
}

pub fn numeric_prompt(question: &str) -> String {
    format!(
        "{}\n\nSolve the problem step by step. End your reply with a final line of the form: #### <number>",
        question.trim()
    )
}

pub fn boxed_prompt(problem: &str) -> String {
    format!("{}\n\nPut the final answer in \\boxed{{}}.", problem.trim())
}

pub fn humaneval_prompt(prompt: &str) -> String {
    format!(
        "Complete the following Python function. Reply with the full function in one ```python code block.\n\n{}\n",
        prompt.trim_end()
    )
}

pub fn mbpp_prompt(text: &str, tests: &[String]) -> String {
    format!(
        "{}\nYour code must pass these tests:\n{}\n\nReply with the full function in one ```python code block.",
        text.trim(),
        tests.join("\n")
    )
}

fn make_item(
    source: &str,
    native: &str,
    kind: &str,
    prompt: String,
    gold: Value,
    meta: Value,
) -> Value {
    json!({
        "id": format!("{source}-{native}"), "source": source, "kind": kind, "bucket": bucket(source), "prompt": prompt,
        "messages": [{"role": "user", "content": prompt}], "gold": gold, "meta": meta, "license": license(source),
    })
}

fn text_of<'a>(r: &'a Value, key: &str) -> Option<&'a str> {
    r.get(key)?.as_str()
}

fn string_list(v: &Value) -> Option<Vec<String>> {
    Some(v.as_array()?.iter().map(as_text).collect())
}

fn one_item(source: &str, i: usize, r: &Value) -> Option<Value> {
    match source {
        "gsm8k" => {
            let gold = gsm8k_gold(text_of(r, "answer")?).ok()?;
            Some(make_item(
                source,
                &i.to_string(),
                "numeric",
                numeric_prompt(text_of(r, "question")?),
                json!(gold),
                json!({}),
            ))
        }
        "mmlu" => {
            let choices: Vec<String> = match r.get("choices") {
                Some(c) => string_list(c)?,
                None => ["A", "B", "C", "D"]
                    .iter()
                    .map(|k| r.get(*k).map(as_text))
                    .collect::<Option<Vec<_>>>()?,
            };
            let ans = as_text(r.get("answer")?);
            let letter = if !ans.trim().is_empty() && ans.trim().chars().all(|c| c.is_ascii_digit())
            {
                ["A", "B", "C", "D"]
                    .get(ans.trim().parse::<usize>().ok()?)?
                    .to_string()
            } else {
                ans.trim().to_uppercase()
            };
            let meta = json!({"choices": choices, "subject": r.get("subject").cloned().unwrap_or(Value::Null)});
            Some(make_item(
                source,
                &i.to_string(),
                "mc",
                mc_prompt(text_of(r, "question")?, &choices),
                json!(letter),
                meta,
            ))
        }
        "mmlu_pro" => {
            let opts: Vec<String> = r
                .get("options")?
                .as_array()?
                .iter()
                .filter(|o| !o.is_null() && as_text(o) != "N/A")
                .map(as_text)
                .collect();
            let native = r.get("question_id").map_or_else(|| i.to_string(), as_text);
            let gold = as_text(r.get("answer")?).trim().to_uppercase();
            let meta = json!({"choices": opts, "category": r.get("category").cloned().unwrap_or(Value::Null)});
            Some(make_item(
                source,
                &native,
                "mc",
                mc_prompt(text_of(r, "question")?, &opts),
                json!(gold),
                meta,
            ))
        }
        "math" => {
            let level = as_text(r.get("level").unwrap_or(&Value::Null));
            if level != "Level 4" && level != "Level 5" {
                return None;
            }
            let gold = last_boxed(text_of(r, "solution")?)?;
            let meta = json!({"level": r.get("level")?.clone(), "type": r.get("type").cloned().unwrap_or(Value::Null)});
            Some(make_item(
                source,
                &i.to_string(),
                "boxed",
                boxed_prompt(text_of(r, "problem")?),
                json!(gold),
                meta,
            ))
        }
        "humaneval" => {
            let native = as_text(r.get("task_id")?)
                .rsplit('/')
                .next()
                .unwrap_or_default()
                .to_string();
            let meta = json!({"prompt": text_of(r, "prompt")?, "test": text_of(r, "test")?, "entry_point": text_of(r, "entry_point")?});
            Some(make_item(
                source,
                &native,
                "code",
                humaneval_prompt(text_of(r, "prompt")?),
                Value::Null,
                meta,
            ))
        }
        "mbpp" => {
            let tests = string_list(r.get("test_list")?)?;
            let setup = r
                .get("test_setup_code")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let meta = json!({"test_list": tests, "test_setup_code": setup});
            Some(make_item(
                source,
                &as_text(r.get("task_id")?),
                "code",
                mbpp_prompt(text_of(r, "text")?, &tests),
                Value::Null,
                meta,
            ))
        }
        "gpqa" => {
            let correct = text_of(r, "Correct Answer")?.to_string();
            let mut order = vec![correct.clone()];
            for k in 1..=3 {
                order.push(text_of(r, &format!("Incorrect Answer {k}"))?.to_string());
            }
            let rid = match r.get("Record ID") {
                Some(v) if !as_text(v).is_empty() && !v.is_null() => as_text(v),
                _ => i.to_string(),
            };
            PyRandom::from_str_seed(&format!("{:x}", Sha256::digest(rid.as_bytes())))
                .shuffle(&mut order);
            let letter = ["A", "B", "C", "D"][order.iter().position(|o| *o == correct)?];
            let meta = json!({"choices": order, "domain": r.get("High-level domain").cloned().unwrap_or(Value::Null)});
            Some(make_item(
                source,
                &rid,
                "mc",
                mc_prompt(text_of(r, "Question")?, &order),
                json!(letter),
                meta,
            ))
        }
        _ => None,
    }
}

/// Dataset records to prompt items (one schema). Records that cannot be graded are skipped, not guessed.
pub fn normalize(source: &str, records: &[Value]) -> Vec<Value> {
    let mut out: Vec<Value> = records
        .iter()
        .enumerate()
        .filter_map(|(i, r)| one_item(source, i, r))
        .collect();
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for it in &mut out {
        let id = it["id"].as_str().unwrap_or_default().to_string();
        let n = seen.entry(id.clone()).or_insert(0);
        let k = *n;
        *n += 1;
        if k > 0 {
            it["id"] = Value::String(format!("{id}~{k}"));
        }
    }
    out
}

/// Parses `gsm8k=120,mmlu=120,...` (the default mix when `None`).
pub fn parse_quota(text: Option<&str>) -> Result<BTreeMap<String, usize>> {
    let Some(text) = text.filter(|t| !t.is_empty()) else {
        return Ok(DEFAULT_QUOTAS
            .iter()
            .map(|(k, v)| ((*k).to_string(), *v))
            .collect());
    };
    let mut q = BTreeMap::new();
    for part in text.split(',') {
        let (k, v) = part.split_once('=').unwrap_or((part, ""));
        match v.trim().parse::<usize>() {
            Ok(n) if SOURCES.contains(&k.trim()) => {
                q.insert(k.trim().to_string(), n);
            }
            _ => bail!("bad --quota entry {part:?}; sources: {SOURCES:?}"),
        }
    }
    Ok(q)
}

fn id_of(v: &Value) -> &str {
    v["id"].as_str().unwrap_or_default()
}

/// Seeded, order-independent sample per source, then one seeded shuffle so a partial run covers every source.
pub fn select(
    by_source: &BTreeMap<String, Vec<Value>>,
    quotas: &BTreeMap<String, usize>,
    seed: u64,
) -> Vec<Value> {
    let mut chosen: Vec<Value> = Vec::new();
    for (source, items) in by_source {
        let mut pool = items.clone();
        pool.sort_by(|a, b| id_of(a).cmp(id_of(b)));
        let k = quotas.get(source).copied().unwrap_or(0).min(pool.len());
        chosen.extend(PyRandom::from_str_seed(&format!("{seed}:{source}")).sample(&pool, k));
    }
    let mut seen = std::collections::HashSet::new();
    let mut uniq: Vec<Value> = Vec::new();
    for it in chosen {
        let h = format!(
            "{:x}",
            Sha256::digest(it["prompt"].as_str().unwrap_or_default().as_bytes())
        );
        if seen.insert(h) {
            uniq.push(it);
        }
    }
    PyRandom::from_str_seed(&format!("{seed}:order")).shuffle(&mut uniq);
    for (n, it) in uniq.iter_mut().enumerate() {
        it["order"] = json!(n);
    }
    uniq
}

/// Hash of the ids and prompts, in order.
pub fn prompt_set_hash(items: &[Value]) -> String {
    let mut h = Sha256::new();
    for it in items {
        h.update(id_of(it).as_bytes());
        h.update([0u8]);
        h.update(it["prompt"].as_str().unwrap_or_default().as_bytes());
        h.update([0u8]);
    }
    format!("{:x}", h.finalize())
}

/// Arguments of `build`.
pub struct BuildArgs {
    pub data_dir: PathBuf,
    pub out: PathBuf,
    pub quota: Option<String>,
    pub seed: u64,
    pub require_all: bool,
}

/// `build`: writes `out` (one item per line) and `out.manifest.json`. Returns the exit code.
pub fn cmd_build(a: &BuildArgs) -> Result<i32> {
    let quotas = parse_quota(a.quota.as_deref())?;
    let mut by_source: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    let mut manifest_sources = Map::new();
    let mut short: Vec<String> = Vec::new();
    for (source, &quota) in &quotas {
        let Some(loaded) = load_source(&a.data_dir, source)? else {
            manifest_sources.insert(
                source.clone(),
                json!({"status": "missing", "requested": quota}),
            );
            short.push(source.clone());
            continue;
        };
        let items = normalize(source, &loaded.records);
        manifest_sources.insert(
            source.clone(),
            json!({
                "status": "ok", "files": loaded.files.iter().filter_map(|f| f.file_name().map(|n| n.to_string_lossy().into_owned())).collect::<Vec<_>>(),
                "sha256": loaded.sha256, "records": loaded.records.len(), "gradable": items.len(), "requested": quota,
                "license": license(source),
            }),
        );
        if items.len() < quota {
            short.push(source.clone());
        }
        by_source.insert(source.clone(), items);
    }
    if a.require_all && !short.is_empty() {
        eprintln!("prep build: missing or short sources: {short:?}");
        return Ok(2);
    }
    let items = select(&by_source, &quotas, a.seed);
    if items.is_empty() {
        eprintln!("prep build: no prompts selected (no source file found under --data-dir)");
        return Ok(2);
    }
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for it in &items {
        *counts
            .entry(it["source"].as_str().unwrap_or_default().to_string())
            .or_insert(0) += 1;
    }
    for (s, m) in &mut manifest_sources {
        m["selected"] = json!(counts.get(s).copied().unwrap_or(0));
    }
    let mut text = String::new();
    for it in &items {
        text.push_str(&serde_json::to_string(it)?);
        text.push('\n');
    }
    std::fs::write(&a.out, text)?;
    let manifest = json!({
        "tool": VERSION, "seed": a.seed, "n_prompts": items.len(), "prompt_set_sha256": prompt_set_hash(&items),
        "sources": manifest_sources, "short_or_missing": short,
        "note": "GPQA and MBPP carry attribution terms; question text stays out of the repository.",
    });
    let mut manifest_path = a.out.clone().into_os_string();
    manifest_path.push(".manifest.json");
    std::fs::write(manifest_path, serde_json::to_string_pretty(&manifest)?)?;
    println!(
        "prep build: {} prompts -> {}; per source {counts:?}; short or missing {short:?}",
        items.len(),
        a.out.display()
    );
    Ok(0)
}
