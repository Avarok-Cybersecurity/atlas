// SPDX-License-Identifier: AGPL-3.0-only
//
// Jinja template helpers for ChatTokenizer. Free fns split out of
// chat_impl.rs to keep the parent under 500 LoC.

use anyhow::{Context, Result};
use std::path::Path;

pub(super) const TEMPLATE_OVERRIDE_DIR: &str = "jinja-templates";

/// Build a precompiled minijinja Environment from the chat template.
/// Leaks the template string to 'static — acceptable since one ChatTokenizer
/// lives for the entire server lifetime.
pub(super) fn build_jinja_env(chat_template: &str) -> Result<minijinja::Environment<'static>> {
    let template_static: &'static str = Box::leak(chat_template.to_string().into_boxed_str());
    let mut env = minijinja::Environment::new();
    env.set_lstrip_blocks(true);
    env.set_trim_blocks(true);

    env.add_function(
        "raise_exception",
        |msg: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    env.add_filter("rtrim", |s: String| -> String {
        s.trim_end_matches('\n').to_string()
    });
    env.add_filter("ltrim", |s: String| -> String {
        s.trim_start_matches('\n').to_string()
    });
    env.add_filter("split_first", |s: String, sep: String| -> String {
        s.split(&sep).next().unwrap_or("").to_string()
    });
    env.add_filter("split_last", |s: String, sep: String| -> String {
        s.rsplit(&sep).next().unwrap_or("").to_string()
    });

    // F76 (2026-04-29): bridge Python-style `.items()` / `.keys()` /
    // `.values()` methods on Maps to the corresponding minijinja
    // filters. MiniMax M2.7's chat_template.jinja line 112 calls
    // `_args.items()` on the tool_call arguments dict; without this
    // callback minijinja raises `UnknownMethod: map has no method
    // named items` and the second turn of every tool-use
    // conversation 500's. The callback only fires on
    // `UnknownMethod` errors, so it is a no-cost compat layer.
    env.set_unknown_method_callback(
        |state, value, method, args| -> Result<minijinja::Value, minijinja::Error> {
            use minijinja::value::{ValueKind, from_args};
            if value.kind() == ValueKind::Map {
                match method {
                    "items" => {
                        let _: () = from_args(args)?;
                        return state.apply_filter("items", std::slice::from_ref(value));
                    }
                    "keys" => {
                        let _: () = from_args(args)?;
                        return state
                            .apply_filter("dictsort", std::slice::from_ref(value))
                            .and_then(|sorted| {
                                state.apply_filter(
                                    "map",
                                    &[
                                        sorted,
                                        minijinja::Value::from("attribute"),
                                        minijinja::Value::from(0u32),
                                    ],
                                )
                            })
                            .or_else(|_| state.apply_filter("list", std::slice::from_ref(value)));
                    }
                    "values" => {
                        let _: () = from_args(args)?;
                        return state
                            .apply_filter("dictsort", std::slice::from_ref(value))
                            .and_then(|sorted| {
                                state.apply_filter(
                                    "map",
                                    &[
                                        sorted,
                                        minijinja::Value::from("attribute"),
                                        minijinja::Value::from(1u32),
                                    ],
                                )
                            });
                    }
                    "get" => {
                        let (key, default): (minijinja::Value, Option<minijinja::Value>) =
                            from_args(args)?;
                        return Ok(value
                            .get_item(&key)
                            .ok()
                            .filter(|v| !v.is_undefined())
                            .unwrap_or_else(|| default.unwrap_or(minijinja::Value::UNDEFINED)));
                    }
                    _ => {}
                }
            }
            Err(minijinja::Error::from(minijinja::ErrorKind::UnknownMethod))
        },
    );

    env.add_template("chat", template_static)
        .context("Failed to compile Jinja chat template")?;
    Ok(env)
}

/// Try loading an override template from jinja-templates/{model_type}.jinja.
pub(super) fn load_override_template(model_type: &str, repo_root: Option<&Path>) -> Option<String> {
    // Check relative to repo root (Docker: /build, dev: /workspace/atlas)
    let candidates = [
        repo_root.map(|r| {
            r.join(TEMPLATE_OVERRIDE_DIR)
                .join(format!("{model_type}.jinja"))
        }),
        Some(std::path::PathBuf::from(TEMPLATE_OVERRIDE_DIR).join(format!("{model_type}.jinja"))),
    ];
    for candidate in candidates.into_iter().flatten() {
        if candidate.exists() {
            match std::fs::read_to_string(&candidate) {
                Ok(raw) => {
                    let converted = convert_python_jinja_to_minijinja(&raw);
                    tracing::info!(
                        "Using override Jinja template from {} ({} chars)",
                        candidate.display(),
                        converted.len(),
                    );
                    return Some(converted);
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to read override template {}: {e}",
                        candidate.display()
                    );
                }
            }
        }
    }
    None
}

/// Try loading an OpenAI-variant template from jinja-templates/openai/{model_type}.jinja.
pub(super) fn load_openai_template(model_type: &str, repo_root: Option<&Path>) -> Option<String> {
    let candidates = [
        repo_root.map(|r| {
            r.join(TEMPLATE_OVERRIDE_DIR)
                .join("openai")
                .join(format!("{model_type}.jinja"))
        }),
        Some(
            std::path::PathBuf::from(TEMPLATE_OVERRIDE_DIR)
                .join("openai")
                .join(format!("{model_type}.jinja")),
        ),
    ];
    for candidate in candidates.into_iter().flatten() {
        if candidate.exists() {
            match std::fs::read_to_string(&candidate) {
                Ok(raw) => {
                    return Some(convert_python_jinja_to_minijinja(&raw));
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to read OpenAI template {}: {e}",
                        candidate.display()
                    );
                }
            }
        }
    }
    None
}

/// Load template from tokenizer_config.json in the model directory.
pub(super) fn load_config_template(model_dir: &Path) -> Result<Option<String>> {
    let config_path = model_dir.join("tokenizer_config.json");
    if !config_path.exists() {
        return Ok(None);
    }
    let config_json =
        std::fs::read_to_string(&config_path).context("Failed to read tokenizer_config.json")?;
    let config: serde_json::Value =
        serde_json::from_str(&config_json).context("Failed to parse tokenizer_config.json")?;
    match config.get("chat_template").and_then(|v| v.as_str()) {
        Some(t) => {
            let converted = convert_python_jinja_to_minijinja(t);
            tracing::info!(
                "Loaded Jinja chat template from tokenizer_config.json ({} chars)",
                converted.len()
            );
            Ok(Some(converted))
        }
        None => {
            // Some models ship chat_template.jinja as a standalone file
            // (e.g., Nemotron-H Super 120B) instead of embedding it in
            // tokenizer_config.json.
            let jinja_path = model_dir.join("chat_template.jinja");
            if jinja_path.exists() {
                let raw = std::fs::read_to_string(&jinja_path)
                    .context("Failed to read chat_template.jinja")?;
                let converted = convert_python_jinja_to_minijinja(&raw);
                tracing::info!(
                    "Loaded standalone chat_template.jinja ({} chars)",
                    converted.len()
                );
                return Ok(Some(converted));
            }
            Ok(None)
        }
    }
}

/// Default ChatML template for models without tokenizer_config.json.
/// Always includes an empty system block if no system message (required by Nemotron-H).
pub(super) fn default_chatml_template(supports_thinking: bool) -> String {
    let gen_prompt = if supports_thinking {
        "{{ '<|im_start|>assistant\\n<think>\\n' }}"
    } else {
        "{{ '<|im_start|>assistant\\n' }}"
    };
    format!(
        r#"{{% if messages[0].role != 'system' %}}{{{{ '<|im_start|>system\n<|im_end|>\n' }}}}{{% endif %}}{{% for message in messages %}}{{% if message.role == 'system' %}}{{% if loop.first %}}{{{{ '<|im_start|>system\n' + message.content + '<|im_end|>\n' }}}}{{% endif %}}{{% elif message.role == 'user' %}}{{{{ '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}}}{{% elif message.role == 'assistant' %}}{{{{ '<|im_start|>assistant\n' + message.content + '<|im_end|>\n' }}}}{{% endif %}}{{% endfor %}}{{% if add_generation_prompt %}}{gen_prompt}{{% endif %}}"#
    )
}

/// Convert Python Jinja2 syntax to minijinja-compatible syntax.
/// Handles: slice reversal, string methods, etc.
pub(super) fn convert_python_jinja_to_minijinja(template: &str) -> String {
    let mut t = template.to_string();

    // messages[::-1] → messages | reverse
    t = t.replace("messages[::-1]", "messages | reverse");

    // content.startswith('X') → content is startingwith('X')
    // Need regex for this — use simple string replacements for known patterns
    t = t.replace(
        ".startswith('<tool_response>')",
        " is startingwith '<tool_response>'",
    );
    t = t.replace(".startswith('\\n')", " is startingwith '\\n'");
    t = t.replace(
        ".endswith('</tool_response>')",
        " is endingwith '</tool_response>'",
    );
    t = t.replace(".endswith('\\n')", " is endingwith '\\n'");

    // content.split('X') → content | split('X') — minijinja has split filter
    // content.split('</think>')[0] → (content | split('</think>'))[0]
    // This is complex — use a custom filter instead
    // For now, register split as a method via custom function

    // content.rstrip('\n') → content | rtrim  (custom filter)
    // content.lstrip('\n') → content | ltrim  (custom filter)
    // content.strip('\n')  → content | rtrim | ltrim  (chain, custom filters)
    //
    // MiniMax M2's chat_template.jinja chains `.split('</think>')[0]
    // .strip('\n').split('<think>')[-1].strip('\n')`. After each
    // substring replace the chain remains a valid filter pipeline
    // because minijinja filters associate left-to-right.
    t = t.replace(".rstrip('\\n')", " | rtrim");
    t = t.replace(".lstrip('\\n')", " | ltrim");
    t = t.replace(".strip('\\n')", " | rtrim | ltrim");

    // .split('</think>')[0] → handled by custom split_first/split_last filters
    // Replace: content.split('</think>')[0] → content | split_first('</think>')
    // Replace: content.split('</think>')[-1] → content | split_last('</think>')
    t = t.replace(".split('</think>')[0]", " | split_first('</think>')");
    t = t.replace(".split('</think>')[-1]", " | split_last('</think>')");

    // .split('<think>')[-1] → | split_last('<think>')
    t = t.replace(".split('<think>')[-1]", " | split_last('<think>')");
    // .split('<think>')[0] → | split_first('<think>')
    t = t.replace(".split('<think>')[0]", " | split_first('<think>')");

    // `tojson(ensure_ascii=False)` → `tojson` (minijinja has no kwargs).
    // MiniMax M2's template renders tools with `tool.function |
    // tojson(ensure_ascii=False)`. minijinja rejects unknown kwargs
    // (and ensure_ascii=False is the Python default for non-ASCII
    // passthrough, which minijinja's tojson does by default anyway).
    // Strip the kwarg so the filter call type-checks.
    t = t.replace("tojson(ensure_ascii=False)", "tojson");
    t = t.replace("tojson(ensure_ascii=True)", "tojson");

    // messages[1:] — minijinja 2.x supports slice syntax natively

    t
}
