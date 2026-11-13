// SPDX-License-Identifier: AGPL-3.0-only
// T-AT: Atlas-native. Labeling and summary of the 4B's replies for the routing gate; no upstream code.
//! Responsibility: strip reasoning blocks, grade each reply into `labels.jsonl` (`grade`), print the per-source table
//! with its failure-rate warning (`summary`), and print where each dataset comes from (`hints`).

use std::path::PathBuf;
use std::sync::LazyLock;

use anyhow::Result;
use regex::Regex;
use serde_json::{Value, json};

use super::collect::{load_done, read_jsonl};
use super::{SOURCES, license};
use crate::route_gate::grading::{Grade, grade};

static THINK_PAIR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<think>.*?</think>").expect("static regex"));

/// Removes a reasoning block: `<think>...</think>` pairs, or everything up to a lone `</think>`; an unclosed
/// `<think>` (the budget ran out while thinking) leaves nothing to grade.
pub fn strip_think(text: &str) -> String {
    let mut text = THINK_PAIR.replace_all(text, "").into_owned();
    if let Some(i) = text.rfind("</think>") {
        text = text[i + "</think>".len()..].to_string();
    }
    if let Some(i) = text.find("<think>") {
        text.truncate(i);
    }
    text.trim().to_string()
}

/// Arguments of `grade`.
pub struct GradeArgs {
    pub prompts: PathBuf,
    pub generations: PathBuf,
    pub out: PathBuf,
    pub code_exec: String,
    pub timeout: f64,
}

/// `grade`: writes one label record per prompt. Returns the exit code.
pub fn cmd_grade(a: &GradeArgs) -> Result<i32> {
    let mut order: Vec<String> = Vec::new();
    let mut items = std::collections::HashMap::new();
    for it in read_jsonl(&a.prompts)? {
        let id = it["id"].as_str().unwrap_or_default().to_string();
        if !items.contains_key(&id) {
            order.push(id.clone());
        }
        items.insert(id, it);
    }
    let gens = load_done(&a.generations)?;
    let (mut n_true, mut n_false, mut n_none) = (0usize, 0usize, 0usize);
    let mut text = String::new();
    for id in &order {
        let it = &items[id];
        let g = gens.get(id);
        let empty = json!({});
        let (res, g) = match g {
            Some(g) if g["error"].is_null() => {
                let reply = strip_think(g["text"].as_str().unwrap_or_default());
                (grade(it, &reply, &a.code_exec, a.timeout)?, g)
            }
            other => {
                let reason = if other.is_none() {
                    "not_generated"
                } else {
                    "request_error"
                };
                (
                    Grade {
                        passed: None,
                        extracted: None,
                        reason: reason.to_string(),
                    },
                    other.unwrap_or(&empty),
                )
            }
        };
        match res.passed {
            Some(true) => n_true += 1,
            Some(false) => n_false += 1,
            None => n_none += 1,
        }
        let rec = json!({
            "id": it["id"], "source": it["source"], "kind": it["kind"], "bucket": it["bucket"], "prompt": it["prompt"],
            "passed": res.passed, "reason": res.reason, "extracted": res.extracted,
            "finish_reason": g.get("finish_reason").cloned().unwrap_or(Value::Null),
            "truncated": g.get("finish_reason").and_then(Value::as_str) == Some("length"),
            "completion_tokens": g.get("usage").and_then(|u| u.get("completion_tokens")).cloned().unwrap_or(Value::Null),
            "latency_s": g.get("latency_s").cloned().unwrap_or(Value::Null),
        });
        text.push_str(&serde_json::to_string(&rec)?);
        text.push('\n');
    }
    std::fs::write(&a.out, text)?;
    println!(
        "prep grade: passed {n_true}, failed {n_false}, not gradable {n_none} -> {}",
        a.out.display()
    );
    Ok(0)
}

/// `summary`: the per-source pass and fail table. Returns the exit code.
pub fn cmd_summary(labels: &std::path::Path) -> Result<i32> {
    let rows = read_jsonl(labels)?;
    let mut by: std::collections::BTreeMap<String, Vec<&Value>> = std::collections::BTreeMap::new();
    for r in &rows {
        by.entry(r["source"].as_str().unwrap_or_default().to_string())
            .or_default()
            .push(r);
    }
    println!(
        "{:<10} {:>4} {:>5} {:>5} {:>4} {:>6} {:>5} {:>6}",
        "source", "n", "pass", "fail", "n/a", "fail%", "trunc", "no_ans"
    );
    let mut tot = [0usize; 4];
    for (s, rs) in &by {
        let p = rs
            .iter()
            .filter(|r| r["passed"] == Value::Bool(true))
            .count();
        let fl = rs
            .iter()
            .filter(|r| r["passed"] == Value::Bool(false))
            .count();
        let na = rs.iter().filter(|r| r["passed"].is_null()).count();
        let tr = rs
            .iter()
            .filter(|r| r["truncated"].as_bool() == Some(true))
            .count();
        let no = rs
            .iter()
            .filter(|r| r["reason"].as_str() == Some("no_answer"))
            .count();
        println!(
            "{s:<10} {:>4} {p:>5} {fl:>5} {na:>4} {:>6.1} {tr:>5} {no:>6}",
            rs.len(),
            100.0 * fl as f64 / (p + fl).max(1) as f64
        );
        tot = [tot[0] + rs.len(), tot[1] + p, tot[2] + fl, tot[3] + na];
    }
    println!(
        "{:<10} {:>4} {:>5} {:>5} {:>4} {:>6.1}",
        "all",
        tot[0],
        tot[1],
        tot[2],
        tot[3],
        100.0 * tot[2] as f64 / (tot[1] + tot[2]).max(1) as f64
    );
    let fr = tot[2] as f64 / (tot[1] + tot[2]).max(1) as f64;
    if tot[1] + tot[2] > 0 && !(0.1..=0.9).contains(&fr) {
        println!(
            "WARNING: failure rate {fr:.2} is outside 0.10..0.90; the AUROC will be very noisy. Consider a harder or easier mix (--quota) before spending the CPU time."
        );
    }
    Ok(0)
}

/// `hints`: where each dataset comes from (prints only, never downloads).
pub fn cmd_hints() -> Result<i32> {
    println!(
        "Datasets are read from local files only; downloading needs owner approval (master plan section 6)."
    );
    println!(
        "Place each under --data-dir as <source>.jsonl|.jsonl.gz|.json|.csv, or a <source>/ directory."
    );
    let hints = [
        (
            "gsm8k",
            "openai/gsm8k, config main, split test (about 2.7 MB)",
            "gsm8k.jsonl with question, answer",
        ),
        (
            "mmlu",
            "cais/mmlu, config all, split test, or the original data.tar test CSVs (about 4 MB)",
            "mmlu.jsonl with question, choices[4], answer (index), subject; or a mmlu/ directory of *_test.csv",
        ),
        (
            "mmlu_pro",
            "TIGER-Lab/MMLU-Pro, split test (about 121 MB)",
            "mmlu_pro.jsonl with question_id, question, options, answer (letter), category",
        ),
        (
            "math",
            "EleutherAI/hendrycks_math, all subjects, split test",
            "math.jsonl with problem, level, type, solution",
        ),
        (
            "humaneval",
            "openai/openai_humaneval (HumanEval.jsonl.gz is also read)",
            "humaneval.jsonl with task_id, prompt, test, entry_point",
        ),
        (
            "mbpp",
            "google-research-datasets/mbpp, config full or sanitized, split test",
            "mbpp.jsonl with task_id, text, test_list, test_setup_code",
        ),
        (
            "gpqa",
            "Idavidrein/gpqa, config gpqa_diamond (gated: accept the terms first)",
            "gpqa.csv with Question, Correct Answer, Incorrect Answer 1-3, Record ID",
        ),
    ];
    debug_assert_eq!(hints.len(), SOURCES.len());
    for (s, src, shape) in hints {
        println!(
            "  {s:<10} {src}\n  {:<10} expected: {shape}\n  {:<10} license: {}",
            "",
            "",
            license(s)
        );
    }
    Ok(0)
}
