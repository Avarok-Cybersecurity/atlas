// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: unit tests of `prep` (corpus build, grade flow with code execution off, record readers), ported from
//! the Python `PrepTests`; datasets are tiny synthetic files and replies are fakes, so nothing touches a network.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use serde_json::{Value, json};

use super::prep::{BuildArgs, GradeArgs, cmd_build, cmd_grade, cmd_summary, read_records};

const QUOTA: &str = "gsm8k=5,mmlu=5,mmlu_pro=4,math=4,humaneval=3,mbpp=3,gpqa=3";

fn write_jsonl(path: &Path, rows: &[Value]) {
    let mut text = String::new();
    for r in rows {
        text.push_str(&serde_json::to_string(r).unwrap());
        text.push('\n');
    }
    std::fs::write(path, text).unwrap();
}

fn read_jsonl(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn synthetic_datasets(d: &Path) {
    let rows = |n: usize, f: &dyn Fn(usize) -> Value| (0..n).map(f).collect::<Vec<Value>>();
    write_jsonl(
        &d.join("gsm8k.jsonl"),
        &rows(
            30,
            &|i| json!({"question": format!("What is {i}+1?"), "answer": format!("Add one.\n#### {}", i + 1)}),
        ),
    );
    write_jsonl(
        &d.join("mmlu.jsonl"),
        &rows(
            30,
            &|i| json!({"question": format!("Q{i}"), "choices": ["a", "b", "c", "d"], "answer": i % 4, "subject": "x"}),
        ),
    );
    write_jsonl(
        &d.join("mmlu_pro.jsonl"),
        &rows(20, &|i| {
            let opts: Vec<String> = (0..10).map(|k| format!("o{k}")).collect();
            json!({"question_id": i, "question": format!("PQ{i}"), "options": opts,
                   "answer": &"ABCDEFGHIJ"[i % 10..i % 10 + 1], "category": "c"})
        }),
    );
    write_jsonl(
        &d.join("math.jsonl"),
        &rows(30, &|i| {
            json!({"problem": format!("P{i}"), "level": format!("Level {}", 1 + i % 5), "type": "t",
                   "solution": format!("so \\boxed{{{i}}}")})
        }),
    );
    write_jsonl(
        &d.join("humaneval.jsonl"),
        &rows(10, &|i| {
            json!({"task_id": format!("HumanEval/{i}"), "prompt": format!("def f{i}(x):\n    \"\"\"doc\"\"\"\n"),
                   "test": "def check(c):\n    assert c(1) == 1\n", "entry_point": format!("f{i}")})
        }),
    );
    write_jsonl(
        &d.join("mbpp.jsonl"),
        &rows(10, &|i| {
            json!({"task_id": i + 1, "text": format!("Write f{i}"), "test_list": [format!("assert f{i}(1) == 1")],
                   "test_setup_code": ""})
        }),
    );
    let mut csv = String::from(
        "Record ID,Question,Correct Answer,Incorrect Answer 1,Incorrect Answer 2,Incorrect Answer 3,High-level domain\n",
    );
    for i in 0..10 {
        csv.push_str(&format!("rec{i},GQ{i},right{i},w1,w2,w3,Physics\n"));
    }
    std::fs::write(d.join("gpqa.csv"), csv).unwrap();
}

fn build_args(d: &Path, out: &Path, seed: u64, require_all: bool) -> BuildArgs {
    BuildArgs {
        data_dir: d.to_path_buf(),
        out: out.to_path_buf(),
        quota: Some(QUOTA.to_string()),
        seed,
        require_all,
    }
}

fn manifest_of(out: &Path) -> Value {
    let mut p = out.as_os_str().to_owned();
    p.push(".manifest.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

#[test]
fn build_is_deterministic_and_respects_quotas() {
    let tmp = tempfile::tempdir().unwrap();
    let d = tmp.path();
    synthetic_datasets(d);
    let (out1, out2, out3) = (d.join("a.jsonl"), d.join("b.jsonl"), d.join("c.jsonl"));
    assert_eq!(cmd_build(&build_args(d, &out1, 7, false)).unwrap(), 0);
    assert_eq!(cmd_build(&build_args(d, &out2, 7, false)).unwrap(), 0);
    assert_eq!(cmd_build(&build_args(d, &out3, 8, false)).unwrap(), 0);
    assert_eq!(std::fs::read(&out1).unwrap(), std::fs::read(&out2).unwrap());
    assert_ne!(std::fs::read(&out1).unwrap(), std::fs::read(&out3).unwrap());
    let items = read_jsonl(&out1);
    assert_eq!(items.len(), 27);
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for it in &items {
        *counts
            .entry(it["source"].as_str().unwrap().to_string())
            .or_insert(0) += 1;
    }
    let want: BTreeMap<String, usize> = [
        ("gsm8k", 5),
        ("mmlu", 5),
        ("mmlu_pro", 4),
        ("math", 4),
        ("humaneval", 3),
        ("mbpp", 3),
        ("gpqa", 3),
    ]
    .iter()
    .map(|(k, v)| ((*k).to_string(), *v))
    .collect();
    assert_eq!(counts, want);
    let ids: std::collections::HashSet<&str> =
        items.iter().map(|it| it["id"].as_str().unwrap()).collect();
    assert_eq!(ids.len(), 27);
    let orders: Vec<i64> = items
        .iter()
        .map(|it| it["order"].as_i64().unwrap())
        .collect();
    assert_eq!(orders, (0..27).collect::<Vec<i64>>());
    for it in &items {
        assert_eq!(
            it["messages"],
            json!([{"role": "user", "content": it["prompt"]}])
        );
        if it["source"] == "math" {
            let level = it["meta"]["level"].as_str().unwrap();
            assert!(level == "Level 4" || level == "Level 5", "{level}");
        }
        if it["kind"] == "mc" {
            let choices = it["meta"]["choices"].as_array().unwrap();
            let letters = &"ABCDEFGHIJ"[..choices.len()];
            let gold = it["gold"].as_str().unwrap();
            let at = letters.find(gold).expect("gold letter is an option letter");
            let line = format!("{gold}. {}", choices[at].as_str().unwrap());
            assert!(it["prompt"].as_str().unwrap().contains(&line), "{line}");
        }
    }
    for it in items.iter().filter(|it| it["source"] == "gpqa") {
        let at = "ABCD".find(it["gold"].as_str().unwrap()).unwrap();
        assert!(
            it["meta"]["choices"][at]
                .as_str()
                .unwrap()
                .starts_with("right")
        );
    }
    let man = manifest_of(&out1);
    assert_eq!(man["n_prompts"], 27);
    assert_eq!(man["sources"]["gsm8k"]["selected"], 5);
    assert_eq!(man["prompt_set_sha256"].as_str().unwrap().len(), 64);
}

#[test]
fn missing_source_and_require_all() {
    let tmp = tempfile::tempdir().unwrap();
    let d = tmp.path();
    synthetic_datasets(d);
    std::fs::remove_file(d.join("gpqa.csv")).unwrap();
    let out = d.join("o.jsonl");
    assert_eq!(cmd_build(&build_args(d, &out, 0, false)).unwrap(), 0);
    assert_eq!(manifest_of(&out)["short_or_missing"], json!(["gpqa"]));
    assert_eq!(cmd_build(&build_args(d, &out, 0, true)).unwrap(), 2);
}

#[test]
fn grade_flow_with_code_off() {
    let tmp = tempfile::tempdir().unwrap();
    let d = tmp.path();
    synthetic_datasets(d);
    let pr = d.join("p.jsonl");
    assert_eq!(cmd_build(&build_args(d, &pr, 3, false)).unwrap(), 0);
    let items = read_jsonl(&pr);
    let gens = d.join("g.jsonl");
    let reply = |it: &Value| -> String {
        match it["kind"].as_str().unwrap() {
            "numeric" => format!("#### {}", it["gold"].as_f64().unwrap() as i64),
            "mc" => format!("Answer: {}", it["gold"].as_str().unwrap()),
            "boxed" => format!("\\boxed{{{}}}", it["gold"].as_str().unwrap()),
            _ => "```python\ndef x():\n    pass\n```".to_string(),
        }
    };
    // The last prompt was never generated.
    let rows: Vec<Value> = items[..items.len() - 1]
        .iter()
        .map(|it| {
            json!({"id": it["id"], "text": reply(it), "finish_reason": "stop", "usage": {"completion_tokens": 9}, "error": null})
        })
        .collect();
    write_jsonl(&gens, &rows);
    let lab = d.join("l.jsonl");
    let args = GradeArgs {
        prompts: pr.clone(),
        generations: gens,
        out: lab.clone(),
        code_exec: "off".to_string(),
        timeout: 10.0,
    };
    assert_eq!(cmd_grade(&args).unwrap(), 0);
    let labels: BTreeMap<String, Value> = read_jsonl(&lab)
        .into_iter()
        .map(|r| (r["id"].as_str().unwrap().to_string(), r))
        .collect();
    for it in &items[..items.len() - 1] {
        let l = &labels[it["id"].as_str().unwrap()];
        if it["kind"] == "code" {
            assert!(l["passed"].is_null());
            assert_eq!(l["reason"], "code_exec_off");
        } else {
            assert_eq!(l["passed"], json!(true), "{}", it["id"]);
        }
    }
    assert_eq!(
        labels[items.last().unwrap()["id"].as_str().unwrap()]["reason"],
        "not_generated"
    );
    assert_eq!(cmd_summary(&lab).unwrap(), 0);
}

#[test]
fn read_records_formats() {
    let tmp = tempfile::tempdir().unwrap();
    let d = tmp.path();
    let rows = vec![json!({"a": 1}), json!({"a": 2})];
    write_jsonl(&d.join("x.jsonl"), &rows);
    std::fs::write(
        d.join("y.json"),
        serde_json::to_string(&json!({"data": rows})).unwrap(),
    )
    .unwrap();
    let text = rows
        .iter()
        .map(|r| serde_json::to_string(r).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    let mut gz = flate2::write::GzEncoder::new(
        std::fs::File::create(d.join("z.jsonl.gz")).unwrap(),
        flate2::Compression::default(),
    );
    gz.write_all(text.as_bytes()).unwrap();
    gz.finish().unwrap();
    for n in ["x.jsonl", "y.json", "z.jsonl.gz"] {
        assert_eq!(read_records(&d.join(n), None).unwrap(), rows, "{n}");
    }
    std::fs::write(d.join("m.csv"), "Q1,a,b,c,d,B\n").unwrap();
    let rec = read_records(
        &d.join("m.csv"),
        Some(&["question", "A", "B", "C", "D", "answer"]),
    )
    .unwrap();
    assert_eq!(rec[0]["answer"], "B");
}
