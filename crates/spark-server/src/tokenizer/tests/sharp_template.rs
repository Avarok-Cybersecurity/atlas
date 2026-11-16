// SPDX-License-Identifier: AGPL-3.0-only

//! Sharp chat template: goldens, invariants, selection and the `terse` kwarg.
//!
//! The goldens in `test_data/chat_templates/sharp/goldens.json` are rendered by
//! Python Jinja2 (`tools/chat_template_goldens/gen_sharp_goldens.py`), not by
//! Atlas, so a divergence between minijinja plus Atlas's compatibility layer
//! and the Jinja2 the template was written for cannot hide. Every case here
//! renders through the production path, `chat_render::render_chat`, against the
//! template text `template_select::template_source` assembles for the choice.
//!
//! Nothing here touches process-global state except
//! `stored_choice_round_trips_and_is_applied`, which is the only test that
//! calls `set_choice` (so no other test can observe it).

use std::path::Path;

use serde_json::{Value, json};

use super::super::ChatTokenizer;
use super::super::chat_render::{RenderFlags, render_chat};
use super::super::jinja_helpers::{
    ToolJsonStyle, build_jinja_env_with, convert_python_jinja_to_minijinja,
};
use super::super::template_select::{self, ChatTemplateChoice};

const CASES: &str = include_str!("../../../test_data/chat_templates/sharp/cases.json");
const GOLDENS: &str = include_str!("../../../test_data/chat_templates/sharp/goldens.json");
const ASSET: &str = include_str!("../../../assets/chat_templates/qwen-sharp-v22.5.0.jinja");
pub(super) const STOCK_4B: &str =
    include_str!("../../../test_data/chat_templates/sharp/qwen3.5-4b-stock.jinja");
pub(super) const STOCK_08B: &str =
    include_str!("../../../test_data/chat_templates/sharp/qwen3.5-0.8b-stock.jinja");
pub(super) const TERSE_MARKER: &str = "Never: open with preamble";

fn env_for(variant: &str) -> &'static minijinja::Environment<'static> {
    use std::sync::OnceLock;
    static ENVS: OnceLock<[minijinja::Environment<'static>; 2]> = OnceLock::new();
    let envs = ENVS.get_or_init(|| {
        [ChatTemplateChoice::Sharp, ChatTemplateChoice::SharpNoTerse].map(|choice| {
            let src = template_select::template_source(&choice).unwrap().unwrap();
            let converted = convert_python_jinja_to_minijinja(&src);
            build_jinja_env_with(&converted, ToolJsonStyle::Compact).expect("sharp compiles")
        })
    });
    match variant {
        "sharp" => &envs[0],
        "sharp-no-terse" => &envs[1],
        other => panic!("unknown variant {other}"),
    }
}

fn cases() -> Vec<Value> {
    let doc: Value = serde_json::from_str(CASES).expect("cases.json parses");
    doc["cases"].as_array().expect("cases array").clone()
}

fn flags(case: &Value) -> RenderFlags<'_> {
    RenderFlags {
        enable_thinking: case["enable_thinking"].as_bool().unwrap(),
        disable_tool_steering: case["disable_tool_steering"].as_bool().unwrap_or(false),
        reasoning_effort: case["reasoning_effort"].as_str(),
        preserve_thinking: case["preserve_thinking"].as_bool(),
        terse: case["terse"].as_bool(),
        allow_continue_final: false,
    }
}

fn render_case(case: &Value) -> anyhow::Result<String> {
    let env = env_for(case["variant"].as_str().unwrap());
    let messages = case["messages"].as_array().unwrap();
    let tools = case.get("tools").map(|t| t.as_array().unwrap().clone());
    render_chat(env, messages, tools.as_deref(), flags(case))
}

fn first_diff(got: &str, want: &str) -> String {
    let at = got
        .bytes()
        .zip(want.bytes())
        .position(|(g, w)| g != w)
        .unwrap_or(got.len().min(want.len()));
    let window = |s: &str| {
        let lo = s.floor_char_boundary(at.saturating_sub(40));
        let hi = s.floor_char_boundary((at + 60).min(s.len()));
        s[lo..hi].to_string()
    };
    format!(
        "first difference at byte {at}\n got:  {:?}\n want: {:?}",
        window(got),
        window(want)
    )
}

#[test]
fn sharp_goldens_match_python_jinja2_byte_for_byte() {
    let goldens: Value = serde_json::from_str(GOLDENS).expect("goldens.json parses");
    let renders = goldens["renders"].as_object().expect("renders object");
    let mut failures = Vec::new();
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let Some(want) = renders.get(name) else {
            failures.push(format!("{name}: no golden"));
            continue;
        };
        match (want, render_case(&case)) {
            (Value::String(want), Ok(got)) if &got == want => {}
            (Value::String(want), Ok(got)) => {
                failures.push(format!("{name}: {}", first_diff(&got, want)))
            }
            (Value::Object(want), Err(err)) => {
                let text = format!("{err:#}");
                let expected = want["error"].as_str().unwrap();
                if !text.contains(expected) {
                    failures.push(format!("{name}: error {text:?} lacks {expected:?}"));
                }
            }
            (want, got) => failures.push(format!(
                "{name}: the golden is {} but the render {}",
                if want.is_object() { "an error" } else { "text" },
                if got.is_ok() { "succeeded" } else { "failed" }
            )),
        }
    }
    assert!(
        failures.is_empty(),
        "{} golden mismatch(es):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn golden_files_are_in_step_with_the_embedded_template() {
    let goldens: Value = serde_json::from_str(GOLDENS).unwrap();
    assert_eq!(
        goldens["template_bytes"].as_u64().unwrap() as usize,
        ASSET.len(),
        "the embedded template changed: rerun `python3 -I tools/chat_template_goldens/gen_sharp_goldens.py`"
    );
    let names: std::collections::BTreeSet<_> = cases()
        .iter()
        .map(|c| c["name"].as_str().unwrap().to_string())
        .collect();
    let golden_names: std::collections::BTreeSet<_> = goldens["renders"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    assert_eq!(
        names, golden_names,
        "cases.json and goldens.json list different cases"
    );
    assert!(
        names.len() >= 100,
        "the case list shrank to {}",
        names.len()
    );
}

#[test]
fn embedded_template_is_the_upstream_text_with_one_header() {
    let (header, body) = ASSET.split_once('\n').expect("header line");
    assert!(
        header.starts_with(
            "{#- Chat template derived from peculiar-ragdoll/Qwen-Sharp-Chat-Templates"
        )
    );
    assert!(header.ends_with("#}"));
    assert!(body.starts_with("{%- set template_version = \"qwen3.8-froggeric-v22.5.0\" %}\n"));
    assert!(ASSET.contains(&format!(
        "template_version = \"{}\"",
        template_select::SHARP_VERSION
    )));
    // No em dash may have been introduced into the header we wrote.
    assert!(!header.contains('\u{2014}'));
}

#[test]
fn every_sharp_render_is_well_formed() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let Ok(out) = render_case(&case) else {
            continue;
        };
        assert_eq!(
            out.matches("<|im_start|>").count(),
            out.matches("<|im_end|>").count() + 1,
            "{name}: every turn closes except the open generation prompt"
        );
        assert!(
            out.ends_with("<think>\n") || out.ends_with("<think>\n\n</think>\n\n"),
            "{name}: ends with a generation prompt, got {:?}",
            &out[out.floor_char_boundary(out.len().saturating_sub(40))..]
        );
        assert!(
            !out.contains("{%") && !out.contains("{#"),
            "{name}: template syntax leaked"
        );
        let terse_off = case["variant"] == "sharp-no-terse" || case["terse"] == Value::Bool(false);
        assert_eq!(
            out.matches(TERSE_MARKER).count(),
            usize::from(!terse_off),
            "{name}: terse block count"
        );
        for msg in case["messages"].as_array().unwrap() {
            if msg["role"] == "user"
                && let Some(text) = msg["content"].as_str()
                && !text.trim().is_empty()
            {
                assert!(
                    out.contains(text.trim()),
                    "{name}: user text {text:?} missing"
                );
            }
            for tc in msg["tool_calls"].as_array().into_iter().flatten() {
                for key in tc["function"]["arguments"]
                    .as_object()
                    .into_iter()
                    .flat_map(|m| m.keys())
                {
                    assert!(
                        out.contains(&format!("<parameter={key}>")),
                        "{name}: argument {key} missing"
                    );
                }
            }
        }
    }
}

/// A conversation, rendered with the Sharp template and thinking on.
fn sharp(messages: Value, preserve: Option<bool>) -> String {
    let case = json!({"variant": "sharp-no-terse", "messages": messages, "enable_thinking": true});
    let mut f = flags(&case);
    f.preserve_thinking = preserve;
    render_chat(
        env_for("sharp-no-terse"),
        case["messages"].as_array().unwrap(),
        None,
        f,
    )
    .unwrap()
}

#[test]
fn multi_turn_reasoning_content_renders_the_think_wrapper() {
    let out = sharp(
        json!([
            {"role": "user", "content": "Add 2 and 3."},
            {"role": "assistant", "content": "The sum is 5.", "reasoning_content": "2 plus 3 is 5."},
            {"role": "user", "content": "Now multiply that by 4."}
        ]),
        None,
    );
    assert_eq!(
        out,
        "<|im_start|>user\nAdd 2 and 3.<|im_end|>\n<|im_start|>assistant\n<think>\n2 plus 3 is 5.\n</think>\n\nThe sum is 5.<|im_end|>\n<|im_start|>user\nNow multiply that by 4.<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
}

#[test]
fn retained_history_makes_each_prompt_a_prefix_of_the_next() {
    let user = |t: &str| json!({"role": "user", "content": t});
    let asst =
        |c: &str, r: &str| json!({"role": "assistant", "content": c, "reasoning_content": r});
    let first = sharp(json!([user("One")]), None);
    let second = sharp(
        json!([user("One"), asst("A1", "think one"), user("Two")]),
        None,
    );
    let third = sharp(
        json!([
            user("One"),
            asst("A1", "think one"),
            user("Two"),
            asst("A2", "think two"),
            user("Three")
        ]),
        None,
    );
    assert!(
        second.starts_with(&first),
        "turn 2 extends turn 1 under the default (preserve_thinking on)"
    );
    assert!(
        third.starts_with(&second),
        "turn 3 extends turn 2 under the default (preserve_thinking on)"
    );
    let cut = sharp(
        json!([user("One"), asst("A1", "think one"), user("Two")]),
        Some(false),
    );
    assert!(
        !cut.starts_with(&first),
        "preserve_thinking=false strips the history think block"
    );
}

#[test]
fn terse_pins_decide_the_system_block() {
    let base = |terse: Value, variant: &str| {
        let mut case = json!({"variant": variant, "messages": [{"role": "system", "content": "Be formal."}, {"role": "user", "content": "Hi"}], "enable_thinking": true});
        if !terse.is_null() {
            case["terse"] = terse;
        }
        render_case(&case).unwrap()
    };
    let default = base(Value::Null, "sharp");
    assert!(default.contains("Be formal.\n\nAnswer directly, after thinking."));
    assert_eq!(base(json!(true), "sharp"), default);
    let off = base(json!(false), "sharp");
    assert!(off.starts_with("<|im_start|>system\nBe formal.<|im_end|>\n"));
    assert_eq!(
        base(Value::Null, "sharp-no-terse"),
        off,
        "sharp-no-terse equals terse=false"
    );
    assert_eq!(
        base(json!(true), "sharp-no-terse"),
        off,
        "the variant pins terse off for good"
    );
}

pub(super) fn model_dir(template: Option<&str>) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    tokenizers::Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default())
        .save(dir.path().join("tokenizer.json"), false)
        .unwrap();
    if let Some(t) = template {
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            json!({"chat_template": t}).to_string(),
        )
        .unwrap();
    }
    dir
}

pub(super) fn load(dir: &Path, model_type: &str, choice: &ChatTemplateChoice) -> ChatTokenizer {
    ChatTokenizer::from_model_dir_with_template(dir, 0, true, model_type, None, true, choice)
        .unwrap()
}

pub(super) fn msgs() -> Vec<Value> {
    vec![json!({"role": "user", "content": "hi"})]
}

pub(super) fn text(tok: &ChatTokenizer, thinking: bool, terse: Option<bool>) -> String {
    tok.render_text_with_terse(&msgs(), None, thinking, false, None, None, terse)
        .unwrap()
}

#[test]
fn choice_parsing_and_strict_resolution() {
    use ChatTemplateChoice::*;
    assert_eq!(ChatTemplateChoice::parse("sharp"), Some(Sharp));
    assert_eq!(
        ChatTemplateChoice::parse(" sharp-no-terse "),
        Some(SharpNoTerse)
    );
    assert_eq!(ChatTemplateChoice::parse("model"), Some(Model));
    assert_eq!(ChatTemplateChoice::parse(""), None);
    assert_eq!(ChatTemplateChoice::parse("   "), None);
    assert_eq!(
        ChatTemplateChoice::parse("a/b.jinja"),
        Some(File("a/b.jinja".into()))
    );

    assert_eq!(template_select::resolve(None, None).unwrap(), None);
    assert_eq!(
        template_select::resolve(Some(""), Some("  ")).unwrap(),
        None
    );
    assert_eq!(
        template_select::resolve(None, Some("sharp")).unwrap(),
        Some(Sharp)
    );
    assert_eq!(
        template_select::resolve(Some("model"), Some("sharp")).unwrap(),
        Some(Model),
        "the flag wins"
    );
    assert_eq!(
        template_select::resolve(Some(""), Some("sharp")).unwrap(),
        Some(Sharp),
        "an empty flag is unset"
    );

    let err = template_select::resolve(None, Some("shrap"))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("AVAROK_CHAT_TEMPLATE") && err.contains("shrap"),
        "{err}"
    );
    let err = template_select::resolve(Some("/nonexistent/dir/t.jinja"), None)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("--chat-template") && err.contains("readable"),
        "{err}"
    );

    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("t.jinja");
    std::fs::write(&file, "   \n").unwrap();
    let err = template_select::resolve(Some(file.to_str().unwrap()), None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("empty"), "{err}");
    std::fs::write(&file, "{{ messages | length }}").unwrap();
    assert_eq!(
        template_select::resolve(Some(file.to_str().unwrap()), None).unwrap(),
        Some(File(file.clone()))
    );
    assert_eq!(
        template_select::resolve(None, Some(file.to_str().unwrap())).unwrap(),
        Some(File(file))
    );

    template_select::validate(Some("sharp")).unwrap();
    template_select::validate(Some("/nonexistent/dir/t.jinja")).unwrap_err();
}
