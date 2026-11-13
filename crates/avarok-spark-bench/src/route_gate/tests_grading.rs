// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: unit tests of `grading` (letter, number and boxed extraction, think stripping, code programs and the
//! stdlib-only `python3` subprocess runner), ported from the Python `GradingTests`.

use serde_json::json;

use super::grading::{
    Grade, extract_code, extract_gsm8k, extract_mc_letter, grade, grade_boxed, grade_mc,
    grade_numeric, gsm8k_gold, humaneval_program, last_boxed, math_equal, mbpp_program,
    parse_number, resolve_sandbox, run_python,
};
use super::prep::strip_think;

fn no_answer() -> Grade {
    Grade {
        passed: Some(false),
        extracted: None,
        reason: "no_answer".to_string(),
    }
}

#[test]
fn multiple_choice() {
    assert_eq!(extract_mc_letter("blah blah\nAnswer: C", 10), Some('C'));
    assert_eq!(extract_mc_letter("So the answer is (B).", 10), Some('B'));
    assert_eq!(extract_mc_letter("**Answer: D**", 10), Some('D'));
    assert_eq!(
        extract_mc_letter("Answer: A\nWait, actually\nAnswer: B", 10),
        Some('B')
    );
    assert_eq!(extract_mc_letter("final: \\boxed{C}", 10), Some('C'));
    assert_eq!(extract_mc_letter("I think so\nB", 10), Some('B'));
    assert_eq!(extract_mc_letter("It depends on the context.", 10), None);
    assert_eq!(extract_mc_letter("Answer: F", 4), None);
    assert_eq!(grade_mc("Answer: C", "c", 4).passed, Some(true));
    assert_eq!(grade_mc("nothing", "A", 4), no_answer());
}

#[test]
fn numbers() {
    assert_eq!(gsm8k_gold("blah\n#### 1,234").unwrap(), 1234.0);
    assert_eq!(extract_gsm8k("so 3+4 = 7\n#### 7"), Some(7.0));
    assert_eq!(extract_gsm8k("The total is $12.50 in all."), Some(12.5));
    assert_eq!(extract_gsm8k("\\boxed{42}"), Some(42.0));
    assert_eq!(extract_gsm8k("no digits here"), None);
    assert_eq!(grade_numeric("#### 18", 18.0).passed, Some(true));
    assert_eq!(grade_numeric("#### 18.5", 18.0).passed, Some(false));
    assert_eq!(parse_number("1/4"), Some(0.25));
    assert_eq!(parse_number("\\frac{3}{4}"), Some(0.75));
    assert_eq!(parse_number("-\\frac{1}{2}"), Some(-0.5));
    assert!(gsm8k_gold("no marker").is_err());
}

#[test]
fn boxed_and_math_equality() {
    assert_eq!(
        last_boxed("x \\boxed{\\frac{1}{2}} y \\boxed{\\frac{3}{4}}").as_deref(),
        Some("\\frac{3}{4}")
    );
    assert_eq!(last_boxed("\\boxed{a{b}c}").as_deref(), Some("a{b}c"));
    assert_eq!(last_boxed("nothing"), None);
    assert!(math_equal("\\dfrac{1}{2}", "\\frac12"));
    assert!(math_equal("0.5", "\\frac{1}{2}"));
    assert!(math_equal("x = 5", "5"));
    assert!(math_equal("10", "10.0"));
    assert!(math_equal("90^\\circ", "90"));
    assert!(math_equal("1,000", "1000"));
    assert!(math_equal("\\text{(A)}", "(A)"));
    assert!(!math_equal("3", "4"));
    assert_eq!(grade_boxed("so \\boxed{3}", "3").passed, Some(true));
    assert_eq!(grade_boxed("no box", "3"), no_answer());
}

#[test]
fn think_stripping() {
    assert_eq!(strip_think("<think>hmm</think>Answer: A"), "Answer: A");
    assert_eq!(strip_think("long thought</think>\nAnswer: B"), "Answer: B");
    assert_eq!(strip_think("<think>never closed"), "");
    assert_eq!(strip_think("plain"), "plain");
}

#[test]
fn code_extraction_and_programs() {
    let reply = "Here:\n```python\ndef f(x):\n    return x\n```\nDone.";
    assert!(extract_code(reply).unwrap().contains("def f(x)"));
    assert_eq!(extract_code("no fences"), None);
    let he = json!({"source": "humaneval", "meta": {
        "prompt": "def f(x):\n    \"\"\"d\"\"\"\n", "entry_point": "f",
        "test": "def check(c):\n    assert c(1) == 1\n"}});
    let prog = humaneval_program(&he, reply).unwrap();
    assert!(prog.starts_with("def f(x):"));
    assert!(prog.contains("check(f)"));
    let body_only = "```python\nreturn x\n```";
    assert!(
        humaneval_program(&he, body_only)
            .unwrap()
            .contains("    return x")
    );
    let mb = json!({"source": "mbpp", "meta": {"test_list": ["assert f(2) == 2"], "test_setup_code": ""}});
    assert!(
        mbpp_program(&mb, reply)
            .unwrap()
            .contains("assert f(2) == 2")
    );
}

#[test]
fn run_python_modes() {
    assert_eq!(
        run_python("print(1)", 10.0, "off", 2048).unwrap(),
        (None, "code_exec_off".to_string())
    );
    let (ok, why) = run_python("assert 1 + 1 == 2", 10.0, "subprocess", 2048).unwrap();
    assert_eq!(ok, Some(true), "{why}");
    let (bad, why) = run_python("assert 1 + 1 == 3", 10.0, "subprocess", 2048).unwrap();
    assert_eq!(bad, Some(false));
    assert!(why.starts_with("error"), "{why}");
    let (slow, why) = run_python("while True: pass", 1.0, "subprocess", 2048).unwrap();
    assert_eq!(slow, Some(false));
    assert_eq!(why, "timeout");
    assert!(resolve_sandbox("bogus").is_err());
}

#[test]
fn grade_code_end_to_end_with_subprocess() {
    let he = json!({"source": "humaneval", "kind": "code", "meta": {
        "prompt": "def f(x):\n    \"\"\"d\"\"\"\n", "entry_point": "f",
        "test": "def check(c):\n    assert c(2) == 4\n"}});
    let good = grade(
        &he,
        "```python\ndef f(x):\n    return x * 2\n```",
        "subprocess",
        10.0,
    )
    .unwrap();
    assert_eq!(good.passed, Some(true), "{}", good.reason);
    let wrong = grade(
        &he,
        "```python\ndef f(x):\n    return x\n```",
        "subprocess",
        10.0,
    )
    .unwrap();
    assert_eq!(wrong.passed, Some(false));
    assert_eq!(
        grade(&he, "no code", "subprocess", 10.0).unwrap().reason,
        "no_answer"
    );
    let off = grade(
        &he,
        "```python\ndef f(x):\n    return x * 2\n```",
        "off",
        10.0,
    )
    .unwrap();
    assert_eq!(off.passed, None);
}
