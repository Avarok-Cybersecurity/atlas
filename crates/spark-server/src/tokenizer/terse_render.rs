// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the per-request `terse` chat-template kwarg (the Sharp
//! template's terseness switch) on the request render path, and the
//! server-wide default for it.
//!
//! `terse` reaches the Jinja context only when it is `Some`; with `None` every
//! prompt is byte-identical to what `apply_chat_template_openai_with_effort`
//! produces, and a template that never reads the variable ignores it either
//! way. The value comes from the request's `chat_template_kwargs.terse`
//! (`ir::ChatRequest::template_terse`), else `--default-chat-template-kwargs`
//! (stored here by `serve_load`).

use std::sync::atomic::{AtomicU8, Ordering};

use anyhow::Result;

use super::chat_render::{RenderFlags, render_chat};
use super::{ChatEncoding, ChatTokenizer};

/// 0 = unset, 1 = false, 2 = true.
static DEFAULT_TERSE: AtomicU8 = AtomicU8::new(0);

/// Store the server-level `terse` default from `--default-chat-template-kwargs`.
pub fn set_default_terse(value: Option<bool>) {
    let code = match value {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    };
    DEFAULT_TERSE.store(code, Ordering::Relaxed);
}

/// The server-level `terse` default; `None` leaves the template's own default.
pub fn default_terse() -> Option<bool> {
    match DEFAULT_TERSE.load(Ordering::Relaxed) {
        1 => Some(false),
        2 => Some(true),
        _ => None,
    }
}

impl ChatTokenizer {
    /// Like `apply_chat_template_openai_with_effort`, plus the `terse` kwarg.
    /// With `terse == None` this IS that method, so the default path is
    /// byte-identical to before; so is any model whose chat is not Jinja-encoded.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_chat_template_openai_with_terse(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
        disable_tool_steering: bool,
        reasoning_effort: Option<&str>,
        preserve_thinking: Option<bool>,
        terse: Option<bool>,
    ) -> Result<Vec<u32>> {
        if terse.is_none() || self.chat_encoding != ChatEncoding::Jinja {
            return self.apply_chat_template_openai_with_effort(
                messages,
                tools,
                enable_thinking,
                disable_tool_steering,
                reasoning_effort,
                preserve_thinking,
            );
        }
        let rendered = self.render_text_with_terse(
            messages,
            tools,
            enable_thinking,
            disable_tool_steering,
            reasoning_effort,
            preserve_thinking,
            terse,
        )?;
        self.encode(&rendered)
    }

    /// The prompt text of the request path with `terse` in the render context:
    /// the OpenAI-variant environment when one is installed, else the main
    /// environment, exactly as `apply_chat_template_openai_with_effort` chooses.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn render_text_with_terse(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
        disable_tool_steering: bool,
        reasoning_effort: Option<&str>,
        preserve_thinking: Option<bool>,
        terse: Option<bool>,
    ) -> Result<String> {
        let (env, allow_continue_final) = match &self.openai_jinja_env {
            Some(env) => (env, false),
            None => (&self.jinja_env, true),
        };
        render_chat(
            env,
            messages,
            tools,
            RenderFlags {
                enable_thinking,
                disable_tool_steering,
                reasoning_effort,
                preserve_thinking,
                terse,
                allow_continue_final,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::jinja_helpers::{ToolJsonStyle, build_jinja_env_with};
    use super::*;

    fn render(template: &str, terse: Option<bool>) -> String {
        let env = build_jinja_env_with(template, ToolJsonStyle::Compact).expect("compiles");
        let messages = vec![json!({"role": "user", "content": "hi"})];
        render_chat(
            &env,
            &messages,
            None,
            RenderFlags {
                enable_thinking: true,
                terse,
                ..Default::default()
            },
        )
        .expect("renders")
    }

    #[test]
    fn terse_joins_the_context_only_when_pinned() {
        let probe = "{% if terse is defined %}T={{ terse }}{% else %}U{% endif %}";
        assert_eq!(render(probe, None), "U");
        assert_eq!(render(probe, Some(true)), "T=true");
        assert_eq!(render(probe, Some(false)), "T=false");
    }

    #[test]
    fn pinning_terse_leaves_every_other_variable_as_it_was() {
        let probe = "{{ enable_thinking }}|{{ reasoning_effort }}|{{ messages | length }}|\
                     {% if tools is defined %}tools{% else %}no tools{% endif %}|\
                     {% if preserve_thinking is defined %}pinned{% else %}unset{% endif %}|{{ add_generation_prompt }}";
        let plain = render(probe, None);
        assert_eq!(plain, "true|medium|1|no tools|unset|true");
        assert_eq!(render(probe, Some(true)), plain);
        assert_eq!(render(probe, Some(false)), plain);
    }

    #[test]
    fn default_terse_round_trips() {
        // The only test that touches the process-wide default.
        assert_eq!(default_terse(), None);
        set_default_terse(Some(false));
        assert_eq!(default_terse(), Some(false));
        set_default_terse(Some(true));
        assert_eq!(default_terse(), Some(true));
        set_default_terse(None);
        assert_eq!(default_terse(), None);
    }
}
