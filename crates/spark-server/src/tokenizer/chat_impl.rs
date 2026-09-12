// SPDX-License-Identifier: AGPL-3.0-only

//! `impl ChatTokenizer` body.

use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer;

use super::{
    ChatEncoding, ChatTokenizer, StreamingDecoder, autoclose_assistant_think,
    normalize_tool_call_arguments, remap_developer_role, resolve_think_control,
};

/// Run Atlas's cross-cutting message preprocessing (formerly encoded in
/// per-model jinja overrides) so it applies to EVERY model's own template:
///   1. parse stringified `tool_calls[*].function.arguments` (F76),
///   2. auto-close an unclosed `<think>` before a `<tool_call>` in
///      assistant history,
///   3. strip inline `<|think_on|>`/`<|think_off|>` control tokens and
///      resolve the effective `enable_thinking`.
///
/// Returns the rewritten messages plus the thinking flag to render with
/// (the inline control tokens override the caller's value when present).
pub(crate) fn preprocess_for_render(
    messages: &[serde_json::Value],
    enable_thinking: bool,
) -> (Vec<serde_json::Value>, bool) {
    // F76: stringified tool-call args → dicts (see normalize_tool_call_arguments).
    let prepared = normalize_tool_call_arguments(messages);
    // Behavior 0: developer→system role remap (model templates reject `developer`;
    // folds developer+system into one leading system message).
    let mut prepared = remap_developer_role(prepared);
    // Behavior 1: auto-close dangling <think> before <tool_call> in history.
    autoclose_assistant_think(&mut prepared);
    // Behavior 2: resolve + strip inline think-control tokens.
    let (prepared, control_override) = resolve_think_control(&prepared);
    let effective_thinking = control_override.unwrap_or(enable_thinking);
    (prepared, effective_thinking)
}

impl ChatTokenizer {
    pub fn from_model_dir(
        model_dir: &Path,
        eos_token_id: u32,
        supports_thinking: bool,
        model_type: &str,
        repo_root: Option<&Path>,
        disable_template_overrides: bool,
    ) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("Failed to disable tokenizer truncation: {e}"))?;

        // Template-source priority.
        //
        // Conceptually the default is now MODEL-FIRST: render off the
        // model's OWN `chat_template.jinja` / `tokenizer_config.json`.
        // Atlas's cross-cutting behaviors (autoclose-think,
        // think-control, F76 arg-parse) are applied in Rust
        // message-preprocessing (see `preprocess_for_render`), so a model
        // no longer needs a bespoke `jinja-templates/{model_type}.jinja`
        // override that is otherwise a byte-copy of its own template.
        // This is what makes `holo3_1_moe.jinja` REDUNDANT: Holo renders
        // correctly off its own template + Rust behaviors. (The override
        // file itself is still present for now only because
        // `tokenizer/tests.rs::render_holo_template_*` reads it directly;
        // it goes away together with those tests.)
        //
        // A `jinja-templates/{model_type}.jinja` override is OPT-IN by
        // FILE PRESENCE: dropping the file in is the explicit signal that
        // this model genuinely needs a template fix the Rust preprocessing
        // can't express (MiniMax's `_args.items()`, Gemma-4's
        // `strip_thinking`, etc.). We deliberately do NOT prefer the
        // model's own template when such a file exists — that would
        // silently undo those fixes. Instead, the operator opts OUT of all
        // overrides with `--disable-template-overrides`, which forces
        // every model onto its own template (relying purely on the Rust
        // behaviors).
        //
        // Priority (high → low):
        //   1. jinja-templates/{model_type}.jinja override
        //      (opt-in: file present AND overrides not disabled)
        //   2. tokenizer_config.json / chat_template.jinja (the MODEL's own)
        //   3. Default ChatML fallback
        let override_tmpl = if disable_template_overrides {
            None
        } else {
            super::jinja_helpers::load_override_template(model_type, repo_root)
        };
        let (chat_template, checkpoint_template) = if let Some(override_tmpl) = override_tmpl {
            (override_tmpl, false)
        } else if let Some(config_tmpl) = super::jinja_helpers::load_config_template(model_dir)? {
            (config_tmpl, true)
        } else {
            tracing::warn!("No chat template found — using default ChatML");
            (
                super::jinja_helpers::default_chatml_template(supports_thinking),
                false,
            )
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
        let chat_encoding = if model_type == "deepseek_v4" {
            tracing::info!("Using checkpoint-native DeepSeek-V4 message encoding");
            ChatEncoding::DeepseekV4
        } else {
            ChatEncoding::Jinja
        };

        let native_qwen_tool_template = checkpoint_owns_qwen_tool_prompt(
            model_type,
            checkpoint_template,
            openai_jinja_env.is_some(),
        ) && jinja_env
            .get_template("chat")?
            .undeclared_variables(false)
            .contains("tools");
        tracing::info!("Loaded tokenizer from {}", tokenizer_path.display());
        Ok(Self {
            tokenizer,
            eos_token_id,
            supports_thinking,
            chat_encoding,
            native_qwen_tool_template,
            chat_template,
            jinja_env,
            openai_jinja_env,
        })
    }

    pub(crate) fn uses_native_qwen_tool_template(&self) -> bool {
        self.native_qwen_tool_template
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
        self.apply_chat_template_jinja_with_effort(
            messages,
            tools,
            enable_thinking,
            disable_tool_steering,
            None,
            None,
        )
    }

    pub fn apply_chat_template_jinja_with_effort(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
        disable_tool_steering: bool,
        reasoning_effort: Option<&str>,
        preserve_thinking: Option<bool>,
    ) -> Result<Vec<u32>> {
        if self.chat_encoding == ChatEncoding::DeepseekV4 {
            let rendered = super::deepseek_v4::encode_messages(
                messages,
                tools,
                enable_thinking,
                reasoning_effort,
            )?;
            return self.encode(&rendered);
        }

        let rendered = super::chat_render::render_chat(
            &self.jinja_env,
            messages,
            tools,
            super::chat_render::RenderFlags {
                enable_thinking,
                disable_tool_steering,
                reasoning_effort,
                preserve_thinking,
                allow_continue_final: true,
            },
        )?;

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
        self.apply_chat_template_openai_with_effort(
            messages,
            tools,
            enable_thinking,
            disable_tool_steering,
            None,
            None,
        )
    }

    pub fn apply_chat_template_openai_with_effort(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
        disable_tool_steering: bool,
        reasoning_effort: Option<&str>,
        preserve_thinking: Option<bool>,
    ) -> Result<Vec<u32>> {
        if self.chat_encoding == ChatEncoding::DeepseekV4 {
            return self.apply_chat_template_jinja_with_effort(
                messages,
                tools,
                enable_thinking,
                disable_tool_steering,
                reasoning_effort,
                preserve_thinking,
            );
        }
        if let Some(ref env) = self.openai_jinja_env {
            // Same render core as apply_chat_template_jinja, minus the
            // continue-final diagnostic (this path always adds the
            // generation prompt, preserving historical behavior).
            let rendered = super::chat_render::render_chat(
                env,
                messages,
                tools,
                super::chat_render::RenderFlags {
                    enable_thinking,
                    disable_tool_steering,
                    reasoning_effort,
                    preserve_thinking,
                    allow_continue_final: false,
                },
            )
            .map_err(|e| anyhow::anyhow!("Failed to render OpenAI Jinja template: {e}"))?;
            self.encode(&rendered)
        } else {
            self.apply_chat_template_jinja_with_effort(
                messages,
                tools,
                enable_thinking,
                disable_tool_steering,
                reasoning_effort,
                preserve_thinking,
            )
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

    pub fn uses_deepseek_v4_encoding(&self) -> bool {
        self.chat_encoding == ChatEncoding::DeepseekV4
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

    /// `<|video_pad|>`, the temporal sibling. `None` on a tokenizer without
    /// it — every text-only model, and any VL model that predates video.
    pub fn video_pad_token_id(&self) -> Option<u32> {
        self.encode("<|video_pad|>")
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
    pub fn expand_vision_pads(&self, tokens: Vec<u32>, pad_counts: &[usize]) -> Vec<u32> {
        if pad_counts.is_empty() || pad_counts.iter().all(|&c| c <= 1) {
            return tokens;
        }
        let image_pad = self.image_pad_token_id();
        let video_pad = self.video_pad_token_id();
        if image_pad.is_none() && video_pad.is_none() {
            return tokens;
        }
        let extra: usize = pad_counts.iter().map(|c| c.saturating_sub(1)).sum();
        let mut out = Vec::with_capacity(tokens.len() + extra);
        let mut img_idx = 0usize;
        for t in tokens {
            let pad_id = t;
            if Some(t) == image_pad || Some(t) == video_pad {
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

/// Only the known Qwen checkpoint templates own XML tool instructions.
/// Custom overrides and ChatML fallback retain parser-provided instructions.
fn checkpoint_owns_qwen_tool_prompt(
    model_type: &str,
    checkpoint: bool,
    openai_override: bool,
) -> bool {
    checkpoint
        && !openai_override
        && matches!(
            model_type,
            "qwen3_5" | "qwen3_5_moe" | "qwen3_6" | "qwen3_6_moe"
        )
}
