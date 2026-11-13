// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: sharp template selection, stock-template parity and stored-choice tests.

use std::path::Path;

use serde_json::json;

use super::super::ChatTokenizer;
use super::super::chat_render::{RenderFlags, render_chat};
use super::super::jinja_helpers::{
    ToolJsonStyle, build_jinja_env_with, convert_python_jinja_to_minijinja,
};
use super::super::template_select::{self, ChatTemplateChoice};
use super::sharp_template::{STOCK_4B, STOCK_08B, TERSE_MARKER, load, model_dir, msgs, text};

#[test]
fn selection_replaces_the_template_and_owns_the_qwen_tool_prompt() {
    let dir = model_dir(Some(STOCK_4B));
    for model_type in ["qwen3_5", "qwen3_5_moe", "qwen3_6", "qwen3_6_moe"] {
        let stock = load(dir.path(), model_type, &ChatTemplateChoice::Model);
        // The loader also looks for `jinja-templates/openai/<type>.jinja` relative to the working directory,
        // which turns off native ownership for a type that has one when the tests run from the repo root.
        let cwd_variant = Path::new("jinja-templates/openai").join(format!("{model_type}.jinja"));
        if !cwd_variant.exists() {
            assert!(
                stock.uses_native_qwen_tool_template(),
                "{model_type}: checkpoint template owns tools"
            );
        }
        for choice in [ChatTemplateChoice::Sharp, ChatTemplateChoice::SharpNoTerse] {
            let tok = load(dir.path(), model_type, &choice);
            assert!(
                tok.uses_native_qwen_tool_template(),
                "{model_type} {choice:?}: parser prompt must not be injected twice"
            );
            assert!(tok.openai_jinja_env.is_none());
            assert!(
                tok.chat_template_source()
                    .contains("template_version = \"qwen3.8-froggeric-v22.5.0\"")
            );
        }
    }
    // A file: native only when it declares `tools` and renders the XML tool format.
    let files = tempfile::tempdir().unwrap();
    let put = |name: &str, body: &str| {
        let p = files.path().join(name);
        std::fs::write(&p, body).unwrap();
        ChatTemplateChoice::File(p)
    };
    let cases = [
        (
            "a.jinja",
            "{% for t in tools %}<function={{ t.function.name }}>{% endfor %}{{ messages | length }}",
            true,
        ),
        (
            "b.jinja",
            "{% for t in tools %}{{ t | tojson }}{% endfor %}",
            false,
        ),
        ("c.jinja", "{# <function= #}{{ messages | length }}", false),
    ];
    for (name, body, native) in cases {
        let tok = load(dir.path(), "qwen3_5", &put(name, body));
        assert_eq!(tok.uses_native_qwen_tool_template(), native, "{name}");
        assert_eq!(tok.chat_template_source(), body);
    }
}

#[test]
fn selection_beats_the_override_file_and_the_openai_variant() {
    let root = tempfile::tempdir().unwrap();
    let repo = root.path().join("repo");
    std::fs::create_dir_all(repo.join("jinja-templates/openai")).unwrap();
    std::fs::write(repo.join("jinja-templates/qwen3_5_moe.jinja"), "OVERRIDE").unwrap();
    std::fs::write(
        repo.join("jinja-templates/openai/qwen3_5_moe.jinja"),
        "OPENAI",
    )
    .unwrap();
    let dir = model_dir(Some(STOCK_4B));
    let build = |choice: &ChatTemplateChoice| {
        ChatTokenizer::from_model_dir_with_template(
            dir.path(),
            0,
            true,
            "qwen3_5_moe",
            Some(&repo),
            false,
            choice,
        )
        .unwrap()
    };
    let before = build(&ChatTemplateChoice::Model);
    assert!(before.openai_jinja_env.is_some() && !before.uses_native_qwen_tool_template());
    assert_eq!(before.chat_template_source(), "OVERRIDE");
    let after = build(&ChatTemplateChoice::Sharp);
    assert!(
        after.openai_jinja_env.is_none(),
        "the OpenAI variant must not shadow the choice"
    );
    assert!(after.uses_native_qwen_tool_template());
    assert!(text(&after, true, None).contains("Answer directly, after thinking."));
}

#[test]
fn model_choice_is_byte_identical_for_stock_qwen35() {
    for (stock, model_type) in [(STOCK_4B, "qwen3_5"), (STOCK_08B, "qwen3_5")] {
        let dir = model_dir(Some(stock));
        let plain =
            ChatTokenizer::from_model_dir(dir.path(), 0, true, model_type, None, true).unwrap();
        let chosen = load(dir.path(), model_type, &ChatTemplateChoice::Model);
        let control = build_jinja_env_with(
            &convert_python_jinja_to_minijinja(stock),
            ToolJsonStyle::Compact,
        )
        .unwrap();
        let convs = [
            msgs(),
            vec![
                json!({"role": "system", "content": "Be brief."}),
                json!({"role": "user", "content": "hi"}),
            ],
            vec![
                json!({"role": "user", "content": "a"}),
                json!({"role": "assistant", "content": "b", "reasoning_content": "c"}),
                json!({"role": "user", "content": "d"}),
            ],
        ];
        for conv in &convs {
            for thinking in [true, false] {
                let want = render_chat(
                    &control,
                    conv,
                    None,
                    RenderFlags {
                        enable_thinking: thinking,
                        ..Default::default()
                    },
                )
                .unwrap();
                let got = chosen
                    .render_chat_text_with_effort(conv, None, thinking, false, None, None)
                    .unwrap();
                assert_eq!(got, want);
                assert_eq!(
                    got,
                    plain
                        .render_chat_text_with_effort(conv, None, thinking, false, None, None)
                        .unwrap()
                );
                // The kwarg is inert for a template that never reads it.
                for terse in [None, Some(true), Some(false)] {
                    let via = chosen
                        .render_text_with_terse(conv, None, thinking, false, None, None, terse)
                        .unwrap();
                    assert_eq!(via, want, "terse={terse:?} must not change a stock prompt");
                }
            }
        }
        assert_eq!(chosen.chat_template_source(), plain.chat_template_source());
        assert!(!chosen.chat_template_source().contains("template_version"));
    }
}

#[test]
fn terse_reaches_the_render_context_only_when_set() {
    let dir = model_dir(Some(STOCK_4B));
    let file = dir.path().join("probe.jinja");
    std::fs::write(
        &file,
        "{% if terse is defined %}T={{ terse }}{% else %}U{% endif %}",
    )
    .unwrap();
    let tok = load(dir.path(), "qwen3_5", &ChatTemplateChoice::File(file));
    assert_eq!(text(&tok, true, None), "U");
    assert_eq!(text(&tok, true, Some(true)), "T=true");
    assert_eq!(text(&tok, true, Some(false)), "T=false");

    let sharp = load(dir.path(), "qwen3_5", &ChatTemplateChoice::Sharp);
    assert!(
        text(&sharp, true, None).contains(TERSE_MARKER),
        "template default is terse on"
    );
    assert!(text(&sharp, true, Some(true)).contains(TERSE_MARKER));
    assert!(!text(&sharp, true, Some(false)).contains(TERSE_MARKER));
    let no_terse = load(dir.path(), "qwen3_5", &ChatTemplateChoice::SharpNoTerse);
    assert!(!text(&no_terse, true, Some(true)).contains(TERSE_MARKER));
    assert!(
        text(&sharp, false, None).contains("Answer directly and concisely."),
        "thinking-off lead"
    );
    assert!(text(&sharp, false, None).ends_with("<think>\n\n</think>\n\n"));
}

#[test]
fn checkpoint_native_encodings_refuse_a_template() {
    let dir = model_dir(None);
    let tok =
        ChatTokenizer::from_model_dir(dir.path(), 0, true, "deepseek_v4", None, true).unwrap();
    let err = template_select::apply_choice(tok, &ChatTemplateChoice::Sharp)
        .err()
        .expect("must refuse");
    assert!(err.to_string().contains("checkpoint-native"), "{err}");
    // `Model` is always fine, whatever the encoding.
    let tok =
        ChatTokenizer::from_model_dir(dir.path(), 0, true, "deepseek_v4", None, true).unwrap();
    template_select::apply_choice(tok, &ChatTemplateChoice::Model).unwrap();
    // A template that does not compile fails startup, naming the choice.
    let bad = dir.path().join("bad.jinja");
    std::fs::write(&bad, "{% if %}").unwrap();
    let tok = ChatTokenizer::from_model_dir(dir.path(), 0, true, "qwen3_5", None, true).unwrap();
    let err = template_select::apply_choice(tok, &ChatTemplateChoice::File(bad))
        .err()
        .expect("must fail");
    assert!(format!("{err:#}").contains("does not compile"), "{err:#}");
}

#[test]
fn stored_choice_round_trips_and_is_applied() {
    let dir = model_dir(Some(STOCK_4B));
    let stock =
        || ChatTokenizer::from_model_dir(dir.path(), 0, true, "qwen3_5", None, true).unwrap();
    assert_eq!(
        template_select::selected_choice(),
        None,
        "no other test may store a choice"
    );
    assert!(
        !template_select::apply_selected(stock())
            .unwrap()
            .chat_template_source()
            .contains("template_version")
    );
    template_select::set_choice(ChatTemplateChoice::Sharp);
    assert_eq!(
        template_select::selected_choice(),
        Some(ChatTemplateChoice::Sharp)
    );
    assert!(
        template_select::apply_selected(stock())
            .unwrap()
            .chat_template_source()
            .contains("template_version")
    );
    template_select::set_choice(ChatTemplateChoice::Model);
    assert!(
        !template_select::apply_selected(stock())
            .unwrap()
            .chat_template_source()
            .contains("template_version")
    );
}
