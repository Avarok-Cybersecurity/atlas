// SPDX-License-Identifier: AGPL-3.0-only

use serde::Deserialize;

use super::*;
use crate::api::inference_types::RepetitionDetectionParams;
use crate::ir::ThinkingDirective;

/// Chat completion request (subset of OpenAI spec).
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ChatCompletionRequest {
    pub model: String,
    /// M2 per-request LoRA routing: optional resident adapter NAME for THIS
    /// request (independent of `model`). Unset = installed active adapter;
    /// unknown = 400. Resolved to a pool slot at request time.
    #[serde(default)]
    pub adapter: Option<String>,
    /// NLLB per-request source/target language names (tokenizer-resolved).
    #[serde(default)]
    pub src_lang: Option<String>,
    #[serde(default)]
    pub tgt_lang: Option<String>,
    /// NLLB beam search: beams per request (None/1 = greedy).
    #[serde(default)]
    pub num_beams: Option<u32>,
    /// NLLB beam search: length penalty (None = 1.0).
    #[serde(default)]
    pub length_penalty: Option<f32>,
    /// NLLB beam search: early-stop when enough hypotheses finish (None = false).
    #[serde(default)]
    pub early_stopping: Option<bool>,
    pub messages: Vec<IncomingMessage>,
    #[serde(default = "default_max_tokens", alias = "max_completion_tokens")]
    pub max_tokens: usize,
    pub temperature: Option<f32>,
    /// Top-k: keep only the k highest-probability tokens before sampling.
    /// None = use server default from generation_config.json.
    pub top_k: Option<u32>,
    /// Top-p (nucleus): keep smallest set of tokens whose cumulative probability >= p.
    /// None = use server default from generation_config.json.
    pub top_p: Option<f32>,
    /// Top-n-sigma: keep tokens with logit >= mean - n*sigma (temperature-
    /// invariant). None = server default. 0.0 = disabled.
    pub top_n_sigma: Option<f32>,
    /// Min-p: keep tokens with prob >= min_p * max_prob. None = server default.
    pub min_p: Option<f32>,
    /// Repetition penalty. None = server default. 1.0 = disabled.
    pub repetition_penalty: Option<f32>,
    /// Presence penalty (OpenAI-style): flat additive penalty per token seen at
    /// least once. Range [-2.0, 2.0], default 0.0 (disabled).
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// Frequency penalty (OpenAI-style): additive penalty proportional to
    /// occurrence count. Range [-2.0, 2.0], default 0.0 (disabled).
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Per-token logit bias: {"token_id": bias_value}. Positive boosts, negative suppresses.
    /// Applied additively to logits before sampling. OpenAI-compatible.
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    #[serde(default)]
    pub stream: bool,
    /// Emit exact sampled token IDs on each streamed chunk's
    /// `choices[0].token_ids` (vLLM-compatible extension) for precise
    /// `usage.completion_tokens` counting. Defaults false (opt-in).
    #[serde(default)]
    pub return_token_ids: bool,
    /// Enable chain-of-thought reasoning (Qwen3.5 thinking models).
    /// `Some(true)`: model generates its reasoning first. `Some(false)`: model
    /// answers directly. `None` (field omitted): defer to the model's design
    /// intent (MODEL.toml `thinking_default`). Must be `Option` so an EXPLICIT
    /// `false` disables thinking — a bare `bool` cannot tell "false" from
    /// "absent", which silently ignored `enable_thinking: false`. Mirrors
    /// `chat_template_kwargs.enable_thinking`.
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    /// Anthropic-style thinking budget: `{"thinking": {"budget_tokens": N}}`
    /// Hard limit on thinking tokens before forcing `</think>`.
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
    /// vLLM PR-style thinking budget (top-level integer).
    /// `max_thinking_tokens` is accepted as an alias — it's the intuitive
    /// name several clients send, and silently dropping it left the budget
    /// unenforced (reasoning ran unbounded). See community report 2026-06.
    /// `thinking_budget` is the DashScope/Qwen spelling that OpenAI-
    /// compatible gateways inject top-level; dropping it left gateway-
    /// injected reasoning caps unenforced on this surface.
    #[serde(default, alias = "max_thinking_tokens", alias = "thinking_budget")]
    pub thinking_token_budget: Option<u32>,
    /// Per-request override for the vLLM-anchored token-loop detector
    /// (content-loop + thinking-loop). Mirrors vLLM's
    /// `RepetitionDetectionParams` shape (`sampling_params.py:111-144`):
    /// `{min_pattern_size, max_pattern_size, min_count}`. When `Some`,
    /// the scheduler uses these values for THIS sequence's anchored
    /// loop detection instead of the boot-global watchdog defaults
    /// derived from MODEL.toml. None = use server default.
    #[serde(default)]
    pub repetition_detection: Option<RepetitionDetectionParams>,
    /// OpenAI-style reasoning effort: `{"reasoning": {"effort": "low"}}`
    #[serde(default)]
    pub reasoning: Option<ReasoningConfig>,
    /// vLLM-style chat template kwargs: `{"chat_template_kwargs": {"enable_thinking": true}}`
    #[serde(default)]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
    /// Tool definitions for function calling (OpenAI-compatible).
    #[serde(default)]
    pub tools: Option<Vec<crate::tool_parser::ToolDefinition>>,
    /// Tool choice: "auto" (default), "none", "required", or specific function.
    #[serde(default)]
    pub tool_choice: Option<crate::tool_parser::ToolChoice>,
    /// Stop sequences: generation stops when any of these strings is produced.
    /// Accepts a single string or array of strings (OpenAI spec).
    #[serde(default, deserialize_with = "deserialize_stop")]
    pub stop: Vec<String>,
    /// Response format constraint (OpenAI-compatible).
    /// `{"type":"text"}` = unconstrained (default),
    /// `{"type":"json_object"}` = any valid JSON,
    /// `{"type":"json_schema","json_schema":{...}}` = JSON matching a schema.
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    /// Minimum number of tokens to generate before allowing EOS/stop.
    /// 0 = no minimum (default). Useful for preventing empty responses.
    #[serde(default)]
    pub min_tokens: usize,
    /// Seed for deterministic sampling. When set, stochastic sampling uses this
    /// seed for the RNG, producing reproducible output for the same inputs.
    /// None = non-deterministic (default).
    pub seed: Option<u64>,
    /// Whether to return log-probabilities. OpenAI SDK sends this as a boolean;
    /// Atlas uses `top_logprobs` for the count. Accepted for compatibility but
    /// the actual count is controlled by `top_logprobs`.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Number of top log-probabilities to return per token (0-20). None = disabled.
    #[serde(default)]
    pub top_logprobs: Option<u8>,
    /// Request timeout in seconds. None = server default.
    #[serde(default)]
    pub timeout: Option<f32>,
    /// Number of chat completion choices to generate (default 1).
    /// Only supported in blocking (non-streaming) mode.
    #[serde(default = "default_n")]
    pub n: usize,
    /// Stream options (OpenAI-compatible). When `include_usage=true`,
    /// a final `choices:[]` chunk with populated `usage` is emitted before
    /// `[DONE]`. `include_obfuscation` defaults true on OpenAI; here we
    /// accept the field but do not emit padding (no side-channel risk on
    /// self-hosted deployments).
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Whether the model may call multiple tools in one turn (OpenAI default
    /// `true`). Atlas currently emits one tool call per turn regardless — the
    /// field is accepted for compatibility but does not change behavior.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// Controls response length beyond `max_tokens` on gpt-5.x class models
    /// (`low | medium | high`). Atlas accepts the field for compatibility
    /// but does not currently steer output length on top of `max_tokens`.
    #[serde(default)]
    pub verbosity: Option<String>,
    /// Service tier (`auto | default | flex | scale | priority`). Atlas runs
    /// one tier only — accepted for compatibility, echoed back in response.
    #[serde(default)]
    pub service_tier: Option<String>,
    /// Persist the completion for later retrieval via GET `/v1/chat/completions/{id}`.
    /// Atlas does not currently have a completion store — field accepted, ignored.
    #[serde(default)]
    pub store: Option<bool>,
    /// User-supplied metadata (≤16 key/value pairs, value ≤512 chars).
    /// Echoed back in the response. OpenAI uses these for completion store
    /// filtering; Atlas just round-trips them.
    #[serde(default)]
    pub metadata: Option<std::collections::HashMap<String, String>>,
    /// Stable identifier for end-users (abuse detection). Atlas accepts,
    /// ignores; kept for back-compat with the deprecated `user` field.
    #[serde(default)]
    pub safety_identifier: Option<String>,
    /// Key used by OpenAI to cache prompt prefixes across requests. Atlas's
    /// prefix cache is content-addressed (hash of prompt tokens), so this
    /// field is accepted and ignored.
    #[serde(default)]
    pub prompt_cache_key: Option<String>,
    /// Deprecated (per OpenAI spec) — replaced by `safety_identifier` and
    /// `prompt_cache_key`. Accepted for back-compat with older SDK versions.
    #[serde(default)]
    pub user: Option<String>,
    /// Output modalities requested by the client (`["text"]`, `["text",
    /// "audio"]`, …). Atlas only emits text — when audio is requested
    /// we log a warning and return text only. Accepted for compat with
    /// the gpt-4o-audio / gpt-5-audio family SDKs.
    #[serde(default)]
    pub modalities: Option<Vec<String>>,
    /// Audio-output configuration (voice + format). Atlas does not
    /// serve audio; the field is accepted and ignored so clients that
    /// unconditionally attach it don't 4xx.
    #[serde(default)]
    pub audio: Option<serde_json::Value>,
    /// Predicted Outputs — a hint that large parts of the response are
    /// known ahead of time (e.g. regenerating a file with one edit).
    /// Atlas does not currently run speculative decoding against the
    /// prediction; accepted and ignored. Dropping vs rejecting matches
    /// OpenAI's forward-compat behavior on models that don't support it.
    #[serde(default)]
    pub prediction: Option<serde_json::Value>,
    /// Web-search tool configuration (`web_search_options: {...}`).
    /// Atlas has no web-search backend — accepted and ignored.
    #[serde(default)]
    pub web_search_options: Option<serde_json::Value>,
    /// Reasoning-effort shorthand (`minimal | low | medium | high`).
    /// 2026 SDKs send this as a top-level field on gpt-5.x chat models;
    /// `client_thinking_directive` maps it through the same effort→budget
    /// ladder as the nested `reasoning.effort` object.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

/// Stream options (OpenAI-compatible).
#[derive(Debug, Clone, Copy, Deserialize, Default)]
#[serde(default)]
pub struct StreamOptions {
    /// Emit a final chunk with `choices:[]` and populated `usage` before `[DONE]`.
    pub include_usage: bool,
    /// Include a random-padding `obfuscation` field on each chunk. Accepted
    /// but not emitted on Atlas — no multi-tenant side-channel risk to defend.
    pub include_obfuscation: bool,
}

/// Response format constraint (OpenAI-compatible).
///
/// Discriminated by `"type"` field:
/// - `"text"`: no constraint (default behavior)
/// - `"json_object"`: output must be valid JSON
/// - `"json_schema"`: output must match the provided JSON schema
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema { json_schema: JsonSchemaSpec },
}

/// JSON schema specification for `response_format.type = "json_schema"`.
#[derive(Debug, Deserialize)]
pub struct JsonSchemaSpec {
    /// Schema name (required by OpenAI spec, used for logging).
    pub name: String,
    /// Optional human-readable description.
    #[serde(default)]
    pub description: Option<String>,
    /// The JSON Schema object.
    pub schema: serde_json::Value,
    /// Whether to enforce strict schema adherence (default: true).
    #[serde(default = "default_true")]
    pub strict: bool,
}

fn default_true() -> bool {
    true
}

/// Anthropic-style thinking configuration.
#[derive(Debug, Deserialize)]
pub struct ThinkingConfig {
    /// Hard token budget for thinking. Min 0 (disabled).
    pub budget_tokens: Option<u32>,
    /// "enabled", "disabled", or "adaptive"
    #[serde(rename = "type")]
    pub thinking_type: Option<String>,
}

/// OpenAI-style reasoning configuration.
#[derive(Debug, Deserialize)]
pub struct ReasoningConfig {
    /// Qualitative effort level.
    pub effort: Option<String>,
}

/// vLLM-style chat template kwargs (request-body wire field). The
/// server-level `--default-chat-template-kwargs` CLI flag is parsed at
/// the CLI edge (`main_modules/serve.rs`), not through this type.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatTemplateKwargs {
    pub enable_thinking: Option<bool>,
    pub thinking_budget: Option<u32>,
    /// Qwen3.6+ dense-family template flag: keep historical `<think>`
    /// blocks in re-rendered assistant turns. Absent = defer to the
    /// MODEL.toml `[behavior].preserve_thinking` override, then to the
    /// model template's own default.
    pub preserve_thinking: Option<bool>,
    /// Qwen3.8 template kwarg (vLLM passes it straight into Jinja).
    /// Was silently DROPPED by serde before 2026-08-15 — a client
    /// sending `chat_template_kwargs.reasoning_effort` got the server
    /// default instead, with no error. Ranks below `reasoning.effort`
    /// and the top-level `reasoning_effort` shorthand, and below the
    /// other `chat_template_kwargs` keys in the directive ladder
    /// (vLLM's template gates the effort block on `enable_thinking`).
    pub reasoning_effort: Option<String>,
}

impl ChatCompletionRequest {
    /// The dedicated effort channels: nested `reasoning.effort` wins
    /// over the top-level `reasoning_effort` shorthand.
    fn body_reasoning_effort(&self) -> Option<&str> {
        self.reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.as_deref())
            .or(self.reasoning_effort.as_deref())
    }

    /// Every channel an effort string can arrive on, in priority order
    /// (`chat_template_kwargs.reasoning_effort` last). This is the
    /// string the TEMPLATE side consumes and the one `validate` checks.
    fn requested_reasoning_effort(&self) -> Option<&str> {
        self.body_reasoning_effort().or_else(|| {
            self.chat_template_kwargs
                .as_ref()
                .and_then(|kw| kw.reasoning_effort.as_deref())
        })
    }

    /// Fail-fast check for the wire-only effort vocabulary, which is
    /// lowered LOSSILY into the IR (the raw string does not survive
    /// `From<ChatCompletionRequest>`), so the handler must reject it
    /// BEFORE lowering. Without this, a typo'd effort (`"hgih"`)
    /// silently resolved to the template's unset default — on Qwen3.8
    /// historically the most expensive `xhigh` directive — while the
    /// budget path resolved a DIFFERENT rung. Covers every channel:
    /// `reasoning.effort`, the top-level shorthand, and
    /// `chat_template_kwargs.reasoning_effort`.
    /// Every present channel is checked — even one shadowed by a
    /// higher-priority valid value — so a bad string never parses clean.
    pub fn validate_reasoning_effort(&self) -> Result<(), String> {
        let channels = [
            self.reasoning.as_ref().and_then(|r| r.effort.as_deref()),
            self.reasoning_effort.as_deref(),
            self.chat_template_kwargs
                .as_ref()
                .and_then(|kw| kw.reasoning_effort.as_deref()),
        ];
        match channels
            .into_iter()
            .flatten()
            .find(|s| crate::ir::parse_wire_effort(s).is_none())
        {
            None => Ok(()),
            Some(bad) => Err(format!(
                "invalid reasoning_effort {bad:?}: expected one of \
                 none, minimal, low, medium, high, xhigh, max"
            )),
        }
    }

    /// The template-facing effort. Unknown spellings resolve to `None`
    /// (= the unset default) — never to a maximal tier — and are already
    /// rejected with a 400 by `invalid_reasoning_effort` on the HTTP
    /// path, so this branch is reachable only from internal callers.
    pub fn client_reasoning_effort(&self) -> Option<crate::ir::ReasoningEffort> {
        crate::ir::parse_wire_effort(self.requested_reasoning_effort()?)
            .and_then(|(template_effort, _)| template_effort)
    }

    /// Resolve the client's thinking intent from all supported
    /// request-body formats into the neutral [`ThinkingDirective`].
    /// This is the OpenAI-edge half of thinking resolution; the model
    /// default (MODEL.toml `[behavior].thinking_default`), the server
    /// default directive, and the `--disable-thinking` kill switch are
    /// folded in later by `api/chat/thinking.rs`, which never sees the
    /// wire fields.
    ///
    /// Request-body priority (highest to lowest):
    /// 1. `thinking.budget_tokens` (Anthropic) — explicit budget
    /// 2. `thinking_token_budget` (vLLM PR; aliases `max_thinking_tokens`,
    ///    `thinking_budget`) — explicit budget
    /// 3. `reasoning.effort` object / top-level `reasoning_effort`
    ///    shorthand (OpenAI) — mapped to budget
    /// 4. `chat_template_kwargs` (vLLM stable) — enable/disable + optional budget
    /// 5. `enable_thinking` (Atlas legacy) — boolean
    ///
    /// No channel present → [`ThinkingDirective::Unspecified`] (the old
    /// `thinking_explicitly_requested() == false`).
    pub fn client_thinking_directive(&self) -> ThinkingDirective {
        // 1. Anthropic: thinking.budget_tokens / thinking.type
        if let Some(ref tc) = self.thinking {
            if let Some(ref t) = tc.thinking_type
                && t == "disabled"
            {
                return ThinkingDirective::Off;
            }
            if let Some(budget) = tc.budget_tokens {
                return ThinkingDirective::On {
                    budget: Some(budget),
                };
            }
            // Anthropic "adaptive" / thinking object with no explicit budget
            // means "think as long as needed" — defer to the per-model
            // `max_thinking_budget` (budget: None), NOT a conservative
            // hardcoded default. A hard 256-class cut force-injects
            // </think> mid-reasoning on agentic turns and wrecks tool
            // selection. Mirrors the step-5 enable_thinking path below.
            return ThinkingDirective::On { budget: None };
        }

        // 2. vLLM PR: thinking_token_budget
        if let Some(budget) = self.thinking_token_budget {
            return if budget > 0 {
                ThinkingDirective::On {
                    budget: Some(budget),
                }
            } else {
                ThinkingDirective::Off
            };
        }

        // 3. OpenAI: reasoning.effort object, or the top-level
        // `reasoning_effort` shorthand the Chat Completions wire (and the
        // 2026 SDKs) send. Nested object wins when both are present.
        // Dropping the shorthand silently demoted every effort-level
        // request to the server/model default — including `"none"`,
        // which must force thinking OFF.
        if let Some(effort) = self.body_reasoning_effort() {
            // DEDICATED channels only (nested reasoning.effort / top-level
            // shorthand) — the kwargs effort string is deliberately handled in
            // step 4 AFTER kwargs.enable_thinking, because vLLM's template
            // gates the effort block on enable_thinking: an explicit
            // `enable_thinking: false` must beat an effort string in the SAME
            // kwargs object. (#514-merge note: its parallel `_ => Medium` arm
            // — which silently forced thinking ON at the Medium budget while
            // the template rendered the xhigh directive, Trap C — is
            // superseded by the 400 on the HTTP path.)
            // Kept SYMBOLIC: the token budget for an effort level is
            // server policy, resolved in `api/chat/thinking.rs` against
            // the model's effective `max_thinking_budget` so MODEL.toml
            // and `--max-thinking-budget` reach it. The old absolute
            // ladder here (64/128/256/512/1024) silently capped every
            // effort-sending client at 256-class budgets no matter what
            // the operator configured.
            //
            // SSOT with the template path (`client_reasoning_effort`):
            // one `parse_wire_effort` match feeds both, so directive
            // and budget can never disagree. Unknown spellings fall
            // through as if ABSENT (they 400 earlier on the HTTP path);
            // the old `_ => Medium` arm silently forced thinking ON at
            // the Medium budget while the template rendered the xhigh
            // directive — the two halves of Trap C.
            if let Some((_, directive)) = crate::ir::parse_wire_effort(effort) {
                return directive;
            }
        }

        // 4. vLLM stable: chat_template_kwargs. Rung order inside the
        // object: thinking_budget > enable_thinking > reasoning_effort —
        // vLLM's own template gates the effort block on enable_thinking,
        // so an explicit `enable_thinking: false` wins over an effort
        // string in the same object.
        if let Some(ref kwargs) = self.chat_template_kwargs {
            if let Some(budget) = kwargs.thinking_budget {
                return if budget > 0 {
                    ThinkingDirective::On {
                        budget: Some(budget),
                    }
                } else {
                    ThinkingDirective::Off
                };
            }
            if let Some(enabled) = kwargs.enable_thinking {
                if !enabled {
                    // An explicit `enable_thinking: false` beats an effort
                    // string in the same object (vLLM's template gates the
                    // effort block on enable_thinking).
                    return ThinkingDirective::Off;
                }
                // `enable_thinking: true` is only load-bearing when it is the
                // ONLY signal. If an effort string rides in the same object,
                // fall through to the effort rung: previously the redundant
                // `true` returned `On{budget: None}` here, silently cutting an
                // `"xhigh"` request's budget 4E -> E while the template still
                // rendered the xhigh sentence — the tier divergence the
                // parse_wire_effort SSOT exists to prevent (review finding F1).
                if kwargs.reasoning_effort.is_none() {
                    // Defer the budget to the per-model max_thinking_budget
                    // (None) rather than a conservative hardcoded 256. Without
                    // this, a server default of '{"enable_thinking":true}'
                    // silently capped EVERY request's thinking at 256.
                    return ThinkingDirective::On { budget: None };
                }
            }
            if let Some(effort) = kwargs.reasoning_effort.as_deref()
                && let Some((_, directive)) = crate::ir::parse_wire_effort(effort)
            {
                // Previously this key was silently DROPPED by serde
                // (Trap B): the request parsed fine and rendered with
                // the server default instead of the asked-for tier.
                return directive;
            }
            if let Some(effort) = kwargs.reasoning_effort.as_deref()
                && let Some((_, directive)) = crate::ir::parse_wire_effort(effort)
            {
                // Previously this key was silently DROPPED by serde
                // (Trap B): the request parsed fine and rendered with
                // the server default instead of the asked-for tier.
                return directive;
            }
        }

        // 5. Atlas legacy: enable_thinking in the request body. Now Option:
        // Some(true) -> On, Some(false) -> Off (an explicit opt-out is now
        // honored, previously it was silently ignored), None (field absent) ->
        // fall through to Unspecified so clients that don't know this flag
        // inherit the model's design intent. `budget: None` so
        // `api/chat/thinking.rs` uses the per-model MODEL.toml cap rather than
        // a conservative hardcoded default (opencode-style clients
        // otherwise hit a 256-token mid-sentence cut on thinking-tier models).
        if let Some(enabled) = self.enable_thinking {
            return if enabled {
                ThinkingDirective::On { budget: None }
            } else {
                ThinkingDirective::Off
            };
        }

        ThinkingDirective::Unspecified
    }
}

pub(super) fn default_max_tokens() -> usize {
    4096
}
pub(super) fn default_n() -> usize {
    1
}

/// Deserialize `stop` as null, a single string, or array of strings (OpenAI spec).
pub(super) fn deserialize_stop<'de, D>(d: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum RawStop {
        Str(String),
        Arr(Vec<String>),
        Null(()),
    }
    match RawStop::deserialize(d)? {
        RawStop::Str(s) => Ok(vec![s]),
        RawStop::Arr(v) => Ok(v),
        RawStop::Null(()) => Ok(Vec::new()),
    }
}

#[cfg(test)]
mod alias_tests {
    use super::ChatCompletionRequest;

    fn base(extra: &str) -> String {
        format!(
            r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}],"max_tokens":16{extra}}}"#
        )
    }

    #[test]
    fn max_thinking_tokens_aliases_thinking_token_budget() {
        // Several clients send `max_thinking_tokens`; it must map to the
        // budget instead of being silently dropped (community report 2026-06).
        let req: ChatCompletionRequest =
            serde_json::from_str(&base(r#","max_thinking_tokens":128"#)).unwrap();
        assert_eq!(req.thinking_token_budget, Some(128));
    }

    #[test]
    fn canonical_thinking_token_budget_still_works() {
        let req: ChatCompletionRequest =
            serde_json::from_str(&base(r#","thinking_token_budget":256"#)).unwrap();
        assert_eq!(req.thinking_token_budget, Some(256));
    }
}
