// SPDX-License-Identifier: AGPL-3.0-only

//! Shared application state passed to all HTTP handlers.

use std::sync::Arc;

use tokio::sync::mpsc;

use crate::api::InferenceRequest;
use crate::tokenizer::ChatTokenizer;
use crate::{
    auth, conversation_store, rate_limiter, reasoning_parser, request_dumper, response_store,
    tool_parser,
};

/// Shared application state accessible from all HTTP handlers.
pub struct AppState {
    pub tokenizer: ChatTokenizer,
    pub model_name: String,
    pub max_seq_len: usize,
    pub request_tx: mpsc::Sender<InferenceRequest>,
    /// Vision config for VL models — None for text-only models.
    pub vision_config: Option<atlas_core::config::VisionConfig>,
    /// Default sampling temperature from generation_config.json.
    pub default_temperature: f32,
    /// Default top-k from generation_config.json.
    pub default_top_k: u32,
    /// Default top-p from generation_config.json.
    pub default_top_p: f32,
    /// Default top-n-sigma from generation_config.json or CLI.
    pub default_top_n_sigma: f32,
    /// Default min-p from generation_config.json or CLI.
    pub default_min_p: f32,
    /// Tool call parser. None = tool calling disabled.
    /// F69 (2026-04-29): Arc instead of Box so the same instance can
    /// be cloned into per-request `GrammarSpec::ToolCall { parser, … }`
    /// for symmetric grammar dispatch via the trait.
    pub tool_call_parser: Option<std::sync::Arc<dyn tool_parser::ToolCallParser>>,
    /// Reasoning parser for thinking block detection. None = no thinking support.
    pub reasoning_parser: Option<Box<dyn reasoning_parser::ReasoningParser>>,
    /// Token ID for end-of-thinking — used to split thinking from content in blocking path.
    /// Derived from reasoning_parser.end_token_id() at startup.
    pub think_end_token_id: Option<u32>,
    /// Token ID for `<think>` — used to detect template-injected
    /// thinking-mode start so we can flip `enable_thinking=true` even
    /// when the request didn't ask for it. MiniMax M2's chat template
    /// always appends `<think>\n` at `add_generation_prompt`, so the
    /// model is implicitly inside thinking from token 1; without this
    /// detection Atlas would never enforce `max_thinking_budget` and
    /// the model can ramble for the full `max_tokens`.
    pub think_start_token_id: Option<u32>,
    /// Max output tokens for tool-calling requests (CLI --tool-max-tokens).
    pub tool_max_tokens: usize,
    /// Model-specific sampling presets from MODEL.toml (per-category defaults).
    pub sampling_presets: atlas_kernels::SamplingPresets,
    /// Token ID for `<tool_call>` — used for logit bias boost when tools are active.
    pub tool_call_start_token_id: Option<u32>,
    /// Auto-compact threshold (fraction of max_seq_len). None = disabled.
    pub auto_compact_threshold: Option<f32>,
    /// Readiness flag: true after model is loaded and scheduler is running.
    pub model_ready: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Default request timeout in seconds. 0 = no timeout.
    pub request_timeout: u32,
    /// Effective context length for agentic tasks (from MODEL.toml).
    /// Compaction triggers when prompt exceeds 50% of this value.
    /// 0 = use max_seq_len instead.
    pub effective_context: usize,
    /// Model-specific behavior overrides from MODEL.toml `[behavior]`.
    /// Embedded at build time via atlas-kernels.
    pub behavior: atlas_kernels::ModelBehavior,
    /// Global kill switch for thinking / reasoning output. When true,
    /// thinking is forced OFF regardless of the request body or the
    /// model's MODEL.toml default. Wired from `--disable-thinking`.
    pub disable_thinking: bool,
    /// Server-level default chat template kwargs applied when the client
    /// sends no thinking parameters. Overridden per-request by the request
    /// body. Wired from `--default-chat-template-kwargs`.
    pub default_chat_template_kwargs: Option<crate::openai::ChatTemplateKwargs>,
    /// Shared in-memory store for stateful Responses API resume
    /// (`previous_response_id`) and opt-in Chat-Completions storage
    /// (`store: true`). Bounded LRU + TTL; env-configured at startup.
    pub response_store: Arc<response_store::ResponseStore>,
    /// Per-identity rate limiter. Pure passthrough when both
    /// ATLAS_RATE_LIMIT_RPM and ATLAS_RATE_LIMIT_TPM are 0 (default).
    pub rate_limiter: Arc<rate_limiter::RateLimiter>,
    /// Conversations API store (items indexed by conv_id).
    pub conversation_store: Arc<conversation_store::ConversationStore>,
    /// Request/response dumper for `--dump`. None = disabled (zero
    /// overhead; handler call sites short-circuit on Option::None).
    pub dump_writer: Option<request_dumper::DumpHandle>,
    /// Bearer-token auth configuration. `Some` ⇒ `--require-auth` was set
    /// and the middleware enforces `Authorization: Bearer <token>` against
    /// the loaded set. `None` ⇒ auth is disabled (every request passes).
    pub auth: Option<Arc<auth::AuthConfig>>,
}

/// Re-export for convenience in api.rs / anthropic.rs.
pub type ModelBehavior = atlas_kernels::ModelBehavior;
