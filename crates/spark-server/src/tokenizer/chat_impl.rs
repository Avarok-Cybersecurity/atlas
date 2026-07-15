// SPDX-License-Identifier: AGPL-3.0-only

//! `impl ChatTokenizer` body.

use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::Tokenizer;

use super::{ChatTokenizer, StreamingDecoder, normalize_tool_call_arguments};

impl ChatTokenizer {
    /// Override directory for Jinja templates. Drop a `.jinja` file here
    /// named by model_type (e.g. `qwen3_5_moe.jinja`) to override the
    /// template from `tokenizer_config.json`. Useful for applying community
    /// fixes without re-downloading model weights.
    const TEMPLATE_OVERRIDE_DIR: &'static str = "jinja-templates";

    pub fn from_model_dir(
        model_dir: &Path,
        eos_token_id: u32,
        supports_thinking: bool,
        model_type: &str,
        repo_root: Option<&Path>,
    ) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("Failed to disable tokenizer truncation: {e}"))?;

        // Priority 1: Override template from jinja-templates/{model_type}.jinja
        // Priority 2: Template from tokenizer_config.json (shipped with model weights)
        // Priority 3: Default ChatML fallback
        let chat_template = if let Some(override_tmpl) =
            super::jinja_helpers::load_override_template(model_type, repo_root)
        {
            override_tmpl
        } else if let Some(config_tmpl) = super::jinja_helpers::load_config_template(model_dir)? {
            config_tmpl
        } else {
            tracing::warn!("No chat template found — using default ChatML");
            super::jinja_helpers::default_chatml_template(supports_thinking)
        };

        let jinja_env = super::jinja_helpers::build_jinja_env(&chat_template)?;

        // Load OpenAI-variant template if it exists (jinja-templates/openai/{model_type}.jinja).
        // This variant gates historical <think> wrappers on enable_thinking, preventing
        // spontaneous thinking during tool-use when thinking is disabled.
        let openai_jinja_env = super::jinja_helpers::load_openai_template(model_type, repo_root)
            .and_then(|tmpl| {
                tracing::info!("Loaded OpenAI-variant Jinja template for {model_type}");
                super::jinja_helpers::build_jinja_env(&tmpl).ok()
            });

        tracing::info!("Loaded tokenizer from {}", tokenizer_path.display());
        Ok(Self {
            tokenizer,
            eos_token_id,
            supports_thinking,
            chat_template,
            jinja_env,
            openai_jinja_env,
        })
    }

    /// Returns a borrowed reference to the underlying HF tokenizer (for
    /// callers that need to drive low-level encode/decode directly).
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenizer encode error: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer decode error: {e}"))
    }

    /// Decode without stripping special tokens. Use when tool calling is active —
    /// some tokenizers register `<tool_call>` as a special token, and skip_special
    /// would strip it, breaking tool call detection.
    pub fn decode_with_special(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("Tokenizer decode error: {e}"))
    }

    /// Incremental detokenizer (vLLM `detokenize_incrementally` scheme).
    /// Returns the newly-STABLE decoded bytes of `toks` since the last call and
    /// advances the offsets. Only the suffix window `toks[prefix_offset..]` is
    /// decoded each call (a handful of tokens since the last stable boundary),
    /// so streaming a full response is O(n) rather than re-decoding the whole
    /// history every token (O(n²)).
    ///
    /// Byte-identical to `decode(&all_toks)` + `trim_end_matches('\u{FFFD}')`
    /// for byte-level BPE and SentencePiece tokenizers: a token's decoded bytes
    /// do not depend on tokens before it, so `decode(toks[prefix_offset..])` is
    /// exactly the corresponding suffix of `decode(toks)`. A token whose window
    /// decode ends in U+FFFD (incomplete multibyte) is held back — the offsets
    /// stay put, so the window naturally extends until a later token completes
    /// the codepoint (same deferral the old `trim_end_matches` did). Uses the
    /// skip-special-tokens `decode`, matching the full-decode it replaces.
    pub fn incremental_decode(
        &self,
        toks: &[u32],
        prefix_offset: &mut usize,
        read_offset: &mut usize,
    ) -> String {
        // Guard against stale offsets after an `all_toks` reset.
        if *read_offset > toks.len() || *prefix_offset > *read_offset {
            *prefix_offset = 0;
            *read_offset = 0;
        }
        let prefix_text = self
            .decode(&toks[*prefix_offset..*read_offset])
            .unwrap_or_default();
        let new_text = self.decode(&toks[*prefix_offset..]).unwrap_or_default();
        if new_text.len() > prefix_text.len()
            && !new_text.ends_with('\u{FFFD}')
            && let Some(delta) = new_text.get(prefix_text.len()..)
        {
            let delta = delta.to_string();
            *prefix_offset = *read_offset;
            *read_offset = toks.len();
            return delta;
        }
        // Incomplete multibyte at the tail (or a non-boundary split): hold this
        // token; the offsets stay put so the next call retries with more context.
        String::new()
    }

    /// Create a stateful streaming decoder wrapper. Each `step(token_id)` returns
    /// `Ok(Some(chunk))` when enough bytes have accumulated for valid UTF-8,
    /// or `Ok(None)` for incomplete multi-byte sequences.
    pub fn streaming_decoder(&self, skip_special_tokens: bool) -> StreamingDecoder<'_> {
        StreamingDecoder {
            inner: self.tokenizer.decode_stream(skip_special_tokens),
        }
    }

    /// Apply the Jinja chat template and encode to token IDs.
    ///
    /// `messages`: Vec of serde_json::Value objects with `role`, `content`,
    ///             and optionally `tool_calls`, `reasoning_content`.
    /// `tools`: Optional tool definitions (passed to Jinja context).
    /// `enable_thinking`: Controls `<think>` generation prompt behavior.
    pub fn apply_chat_template_jinja(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
        disable_tool_steering: bool,
    ) -> Result<Vec<u32>> {
        let tmpl = self
            .jinja_env
            .get_template("chat")
            .context("Failed to get compiled template")?;

        // F76 (2026-04-29): MiniMax's chat template iterates
        // `tool_call.function.arguments` with `_args.items()`, expecting
        // a dict. The OpenAI wire format ships `arguments` as a
        // JSON-encoded *string*, so the template crashes with
        // "unknown method: map has no method named items" on the
        // second turn of any tool-use conversation. Pre-parse string
        // arguments to JSON values before handing to Jinja so the
        // template's iteration sees a dict. Other templates (Qwen,
        // Mistral, Hermes) typically wrap with `tojson` and don't
        // depend on `.items()`, so the parsed dict round-trips fine.
        let messages_for_render = normalize_tool_call_arguments(messages);
        let messages_val = minijinja::Value::from_serialize(&messages_for_render);
        let tools_val = tools.map(minijinja::Value::from_serialize);

        // Diagnostic "continue final message" mode: when the LAST message is an
        // assistant turn, render WITHOUT a generation prompt and strip the
        // trailing end-of-turn marker so the assistant content becomes the final
        // prefill token(s). This lets a prefill-vs-decode A/B place a generated
        // token at the exact position decode produced it. (Standard
        // continue_final_message convention.)
        let continue_final = messages
            .last()
            .and_then(|m| m.get("role"))
            .and_then(|r| r.as_str())
            == Some("assistant");

        // Pass enable_thinking as-is to the template. The Qwen3.5 template uses it
        // to emit <think>\n (thinking) or <think>\n\n</think>\n\n (no thinking).
        // Mistral template uses reasoning_effort instead.
        // The api.rs layer controls enable_thinking based on thinking_in_tools MODEL.toml.
        // Mistral's template defaults `reasoning_effort` to "high" when
        // undefined, so we must explicitly pass "none" to disable thinking.
        let reasoning_effort: minijinja::Value = if enable_thinking {
            "high".into()
        } else {
            "none".into()
        };
        let ctx = minijinja::context! {
            messages => messages_val,
            tools => tools_val.unwrap_or(minijinja::Value::UNDEFINED),
            add_generation_prompt => !continue_final,
            enable_thinking => enable_thinking,
            reasoning_effort => reasoning_effort,
            disable_tool_steering => disable_tool_steering,
            add_vision_id => false,
        };

        let mut rendered = tmpl.render(ctx).map_err(|e| {
            tracing::error!("Jinja template error: {e:#}");
            anyhow::anyhow!("Failed to render Jinja chat template: {e}")
        })?;

        if continue_final {
            // Strip the trailing end-of-turn so the assistant content is the
            // last prefill token (qwen-style templates close with
            // `<|im_end|>\n`). Trim trailing whitespace first, then the marker.
            let trimmed = rendered.trim_end();
            let stripped = trimmed.strip_suffix("<|im_end|>").unwrap_or(trimmed);
            rendered = stripped.to_string();
            tracing::info!("continue_final_message: stripped trailing EOT for prefill A/B");
        }

        // Debug: log the tail of the rendered template for the first few requests.
        // Use floor_char_boundary to avoid panicking on multi-byte UTF-8 (e.g. Swedish å ä ö).
        if rendered.len() < 2000 {
            let tail_start = rendered.floor_char_boundary(rendered.len().saturating_sub(200));
            tracing::info!(
                "Jinja rendered ({} chars): {:?}",
                rendered.len(),
                &rendered[tail_start..]
            );
        }

        self.encode(&rendered)
    }

    /// Apply the OpenAI-variant template (if available), falling back to the default.
    /// The OpenAI variant gates historical `<think>` wrappers on enable_thinking,
    /// preventing the model from learning a "always think" pattern during tool use.
    pub fn apply_chat_template_openai(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
        disable_tool_steering: bool,
    ) -> Result<Vec<u32>> {
        if let Some(ref env) = self.openai_jinja_env {
            let tmpl = env
                .get_template("chat")
                .context("Failed to get compiled OpenAI template")?;
            // F76: pre-parse tool_call argument strings into dicts. See
            // apply_chat_template_jinja above for the failure mode
            // (`map has no method named items` on the second turn).
            let messages_for_render = normalize_tool_call_arguments(messages);
            let messages_val = minijinja::Value::from_serialize(&messages_for_render);
            let tools_val = tools.map(minijinja::Value::from_serialize);
            let reasoning_effort: minijinja::Value = if enable_thinking {
                "high".into()
            } else {
                "none".into()
            };
            let ctx = minijinja::context! {
                messages => messages_val,
                tools => tools_val.unwrap_or(minijinja::Value::UNDEFINED),
                add_generation_prompt => true,
                enable_thinking => enable_thinking,
                reasoning_effort => reasoning_effort,
                disable_tool_steering => disable_tool_steering,
                add_vision_id => false,
            };
            let rendered = tmpl
                .render(ctx)
                .map_err(|e| anyhow::anyhow!("Failed to render OpenAI Jinja template: {e}"))?;
            self.encode(&rendered)
        } else {
            self.apply_chat_template_jinja(messages, tools, enable_thinking, disable_tool_steering)
        }
    }

    /// Legacy apply_chat_template for callers that pass (role, content) tuples.
    /// Converts to JSON messages and delegates to apply_chat_template_jinja.
    pub fn apply_chat_template(
        &self,
        messages: &[(String, String)],
        enable_thinking: bool,
        _image_pad_counts: &[usize],
    ) -> Result<Vec<u32>> {
        let json_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|(role, content)| {
                serde_json::json!({
                    "role": role,
                    "content": content,
                })
            })
            .collect();

        self.apply_chat_template_jinja(&json_messages, None, enable_thinking, false)
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn think_end_token_id(&self) -> Option<u32> {
        if !self.supports_thinking {
            return None;
        }
        match self.encode("</think>") {
            Ok(ids) if ids.len() == 1 => Some(ids[0]),
            _ => None,
        }
    }

    pub fn supports_thinking(&self) -> bool {
        self.supports_thinking
    }

    /// Encode the `<|image_pad|>` placeholder token and return its ID.
    /// Returns `None` when the tokenizer doesn't have this token (text-only
    /// models). Cheap to call repeatedly — the underlying tokenizer caches
    /// single-token encodes.
    pub fn image_pad_token_id(&self) -> Option<u32> {
        self.encode("<|image_pad|>")
            .ok()
            .and_then(|ids| if ids.len() == 1 { Some(ids[0]) } else { None })
    }

    /// Post-process a rendered token sequence to expand `<|image_pad|>`
    /// placeholders. The Qwen3-VL / Qwen3.6 chat template emits exactly one
    /// `<|image_pad|>` per image, but the vision encoder produces
    /// `grid_h * grid_w` patches per image. At embed-injection time the
    /// server expects one pad token per patch so each patch's embedding
    /// lands at the right hidden-state position — this helper does the
    /// fan-out.
    ///
    /// `pad_counts[i]` is the number of patches the i-th image produces.
    /// Extra or missing `<|image_pad|>` occurrences (vs `pad_counts.len()`)
    /// pass through unchanged, matching counts are replicated in place.
    pub fn expand_image_pads(&self, tokens: Vec<u32>, pad_counts: &[usize]) -> Vec<u32> {
        if pad_counts.is_empty() || pad_counts.iter().all(|&c| c <= 1) {
            return tokens;
        }
        let Some(pad_id) = self.image_pad_token_id() else {
            return tokens;
        };
        let extra: usize = pad_counts.iter().map(|c| c.saturating_sub(1)).sum();
        let mut out = Vec::with_capacity(tokens.len() + extra);
        let mut img_idx = 0usize;
        for t in tokens {
            if t == pad_id {
                let count = pad_counts.get(img_idx).copied().unwrap_or(1).max(1);
                for _ in 0..count {
                    out.push(pad_id);
                }
                img_idx += 1;
            } else {
                out.push(t);
            }
        }
        out
    }
}
