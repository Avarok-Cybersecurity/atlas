// SPDX-License-Identifier: AGPL-3.0-only

use serde::Deserialize;

use super::*;

/// Chat completion request (subset of OpenAI spec).
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ChatCompletionRequest {
    pub model: String,
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
    /// Top-n-sigma: filter tokens in logit space before temperature scaling.
    /// Keep only tokens with logit >= mean - n*sigma. Temperature-invariant.
    /// None = use server default. 0.0 = disabled.
    pub top_n_sigma: Option<f32>,
    /// Min-p: keep tokens with prob >= min_p * max_prob (post-softmax).
    /// None = use server default. 0.0 = disabled.
    pub min_p: Option<f32>,
    /// Repetition penalty: penalize tokens that have already been generated.
    /// None = use server default. 1.0 = disabled.
    pub repetition_penalty: Option<f32>,
    /// Presence penalty (OpenAI-style): flat additive penalty for each token that
    /// appeared at least once. Range [-2.0, 2.0], default 0.0 (disabled).
    /// Positive values encourage topic diversity.
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// Frequency penalty (OpenAI-style): additive penalty proportional to occurrence
    /// count. Range [-2.0, 2.0], default 0.0 (disabled).
    /// Positive values discourage token repetition.
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Per-token logit bias: {"token_id": bias_value}. Positive boosts, negative suppresses.
    /// Applied additively to logits before sampling. OpenAI-compatible.
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    #[serde(default)]
    pub stream: bool,
    /// Enable chain-of-thought reasoning (Qwen3.5 thinking models).
    /// false (default): appends `<think></think>` — model answers directly.
    /// true: appends `<think>\n` — model generates its reasoning first.
    #[serde(default)]
    pub enable_thinking: bool,
    /// Anthropic-style thinking budget: `{"thinking": {"budget_tokens": N}}`
    /// Hard limit on thinking tokens before forcing `</think>`.
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
    /// vLLM PR-style thinking budget (top-level integer).
    #[serde(default)]
    pub thinking_token_budget: Option<u32>,
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
    /// Atlas maps it to the existing `reasoning.effort` knob when the
    /// model's reasoning parser supports it.
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

/// vLLM-style chat template kwargs.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatTemplateKwargs {
    pub enable_thinking: Option<bool>,
    pub thinking_budget: Option<u32>,
}

impl ChatTemplateKwargs {
    /// Parse from a JSON string. Returns `None` if parsing fails or string is empty.
    pub fn from_json(s: &str) -> Option<Self> {
        if s.trim().is_empty() {
            return None;
        }
        serde_json::from_str(s).ok()
    }
}

/// Default thinking budget when thinking is enabled but no explicit budget set.
/// 256 tokens is enough for the model to plan without overthinking — longer
/// budgets waste decode throughput on reasoning that rarely improves output.
const DEFAULT_THINKING_BUDGET: u32 = 256;

impl ChatCompletionRequest {
    /// Resolve thinking parameters from all supported request-body formats
    /// into a single `(enable_thinking: bool, thinking_budget: Option<u32>)`
    /// pair. The client's per-request choice always wins over the model
    /// default; the model default (from MODEL.toml `[behavior].thinking_default`)
    /// is used only when the client sends NO thinking parameter at all.
    ///
    /// The `--disable-thinking` CLI flag is a higher-priority kill switch
    /// applied by the caller (api.rs / anthropic.rs) — this function does
    /// not know about it.
    ///
    /// Request-body priority (highest to lowest):
    /// 1. `thinking.budget_tokens` (Anthropic) — explicit budget
    /// 2. `thinking_token_budget` (vLLM PR) — explicit budget
    /// 3. `reasoning.effort` (OpenAI) — mapped to budget
    /// 4. `chat_template_kwargs` (vLLM stable) — enable/disable + optional budget
    /// 5. `enable_thinking` (Atlas legacy) — boolean with default budget
    /// 6. `model_default` argument (from MODEL.toml) — model-specific fallback
    ///
    /// Returns `true` if any of channels 1-5 carried an explicit thinking
    /// intent from the client (i.e. the resolved value did NOT fall through
    /// to `model_default`). Callers use this to decide whether a
    /// server-side policy (e.g. `thinking_in_tools=false`) is allowed to
    /// override the model default OR must respect the explicit request.
    pub fn thinking_explicitly_requested(&self) -> bool {
        if self.thinking.is_some() {
            return true;
        }
        if self.thinking_token_budget.is_some() {
            return true;
        }
        if let Some(ref rc) = self.reasoning
            && rc.effort.is_some()
        {
            return true;
        }
        if let Some(ref kw) = self.chat_template_kwargs
            && (kw.thinking_budget.is_some() || kw.enable_thinking.is_some())
        {
            return true;
        }
        if self.enable_thinking {
            return true;
        }
        false
    }

    pub fn resolve_thinking(&self, model_default: bool) -> (bool, Option<u32>) {
        // 1. Anthropic: thinking.budget_tokens
        if let Some(ref tc) = self.thinking {
            if let Some(ref t) = tc.thinking_type
                && t == "disabled"
            {
                return (false, Some(0));
            }
            if let Some(budget) = tc.budget_tokens {
                return (true, Some(budget));
            }
            // thinking object present with no budget → enable with default
            return (true, Some(DEFAULT_THINKING_BUDGET));
        }

        // 2. vLLM PR: thinking_token_budget
        if let Some(budget) = self.thinking_token_budget {
            return (budget > 0, Some(budget));
        }

        // 3. OpenAI: reasoning.effort
        if let Some(ref rc) = self.reasoning
            && let Some(ref effort) = rc.effort
        {
            let budget = match effort.as_str() {
                "none" => 0,
                "minimal" => 64,
                "low" => 128,
                "medium" => 256,
                "high" => 512,
                "xhigh" | "max" => 1024,
                _ => DEFAULT_THINKING_BUDGET,
            };
            return (budget > 0, Some(budget));
        }

        // 4. vLLM stable: chat_template_kwargs
        if let Some(ref kwargs) = self.chat_template_kwargs {
            if let Some(budget) = kwargs.thinking_budget {
                return (budget > 0, Some(budget));
            }
            if let Some(enabled) = kwargs.enable_thinking {
                let budget = if enabled { DEFAULT_THINKING_BUDGET } else { 0 };
                return (enabled, Some(budget));
            }
        }

        // 5. Atlas legacy: enable_thinking boolean in the request body.
        // Only honored when explicitly true — a false value (including the
        // serde default when the field is absent) falls through to the
        // MODEL.toml default so clients that don't know about this flag
        // inherit the model's design intent instead of silently opting out.
        if self.enable_thinking {
            return (true, Some(DEFAULT_THINKING_BUDGET));
        }

        // 6. Model default from MODEL.toml [behavior].thinking_default.
        if model_default {
            (true, Some(DEFAULT_THINKING_BUDGET))
        } else {
            (false, None)
        }
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
