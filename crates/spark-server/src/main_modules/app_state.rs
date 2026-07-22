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

/// Resolve a per-request `adapter` name to a LoRA pool slot index for M2
/// routing. Rules (pure — unit-tested):
///   `None`               → `Some(-1)` : defer to the installed active adapter
///                          (byte-identical to today for unset requests).
///   known name           → `Some(slot)` : its index in `adapter_names`
///                          (slot order, matching the pool pack order).
///   unknown name         → `None` : the caller returns HTTP 400.
/// A request that names the base `model` is treated as "unknown adapter" here
/// (callers pass only the explicit `adapter` field, never `model`).
pub fn resolve_adapter_slot(adapter_names: &[String], adapter: Option<&str>) -> Option<i32> {
    match adapter {
        None => Some(-1),
        Some(name) => adapter_names
            .iter()
            .position(|n| n == name)
            .map(|i| i as i32),
    }
}

/// Shared application state accessible from all HTTP handlers.
pub struct AppState {
    pub tokenizer: ChatTokenizer,
    pub model_name: String,
    /// Startup LoRA adapter name (`--lora-adapter NAME=…`). `None` = no
    /// adapter (every existing deployment byte-identical). This is the DEFAULT
    /// route (slot 0) advertised first and matched by /v1/models/{id}.
    pub adapter_name: Option<String>,
    /// All resident LoRA adapter names (one per `--lora-adapter`, slot order).
    /// Advertised by /v1/models; a name here is a valid `POST /v1/lora/active`
    /// target. Empty when no adapter is loaded.
    pub adapter_names: Vec<String>,
    /// The currently-active adapter (updated by `POST /v1/lora/active`). Starts
    /// at slot 0. Purely for status/advertise; the scheduler's model owns the
    /// authoritative active slot.
    pub active_adapter: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    pub max_seq_len: usize,
    pub request_tx: mpsc::Sender<InferenceRequest>,
    /// LoRA adapter-rotation control channel (`POST /v1/lora/active`). `None`
    /// when no adapter is loaded. Carries `(adapter_name, ack)` to the
    /// scheduler, which applies the rotation at a quiescent point.
    pub rotation_tx: Option<mpsc::Sender<crate::scheduler::LoraRotation>>,
    /// Vision config for VL models — None for text-only models.
    pub vision_config: Option<atlas_core::config::VisionConfig>,
    /// Optional vLLM-style image area cap applied before vision patching.
    pub vision_max_pixels: Option<usize>,
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
    /// Server-level default thinking directive applied when the client
    /// sends no thinking parameters. Overridden per-request by the request
    /// body. Wired from `--default-chat-template-kwargs` (parsed at the
    /// CLI edge into the neutral directive).
    pub default_thinking: crate::ir::ThinkingDirective,
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
    /// Task #27: STAGEABLE registry — adapters promotable-but-not-resident,
    /// `name -> {peer_stage_id, peft}`, from `--lora-stageable`. Empty ⇒ no
    /// promotion (resident-only serve byte-identical).
    pub lora_stageable:
        std::collections::HashMap<String, crate::main_modules::promotion::StageableAdapter>,
    /// Task #27: the `$ATLAS_LORA_PEER` weight-peer address a promote reads from.
    /// `None` ⇒ promotion disabled.
    pub lora_peer_addr: Option<String>,
    /// Task #27: load-coalescing single-flight coordinator. `Some` only when
    /// promotion is fully armed (registry non-empty + peer set + rotation_tx).
    pub promotion: Option<Arc<crate::main_modules::promotion::PromotionManager>>,
    /// Task #27: promoted-name -> cache slot overlay. A successful promote inserts
    /// `name -> slot` (and drops any evicted name) so subsequent requests for the
    /// same adapter fast-path to the resident cache slot without another promote.
    pub promoted_slots: Arc<std::sync::RwLock<std::collections::HashMap<String, i32>>>,
    /// DISK-stageable registry: `name -> (resolved adapter dir, peft)`. Parsed at
    /// startup (fail-fast) for `/v1/models` advertising + rank validation; the
    /// disk swap re-reads the config at promote time. Empty ⇒ no disk fault-in
    /// (byte-identical to today). The no-RDMA sibling of `lora_stageable`.
    pub lora_disk_stageable: std::collections::HashMap<
        String,
        (std::path::PathBuf, atlas_core::config::PeftAdapterConfig),
    >,
}

use crate::main_modules::promotion::PromoteReject;

impl AppState {
    /// Task #27: on a resolver MISS (`resolve_adapter_slot == None`), try to make
    /// the named adapter HOT via demand-driven RDMA promotion. Returns:
    ///   * `Ok(None)`      — not stageable / promotion disabled → the caller emits
    ///                       the byte-identical resident-only 400 (constraint a).
    ///   * `Ok(Some(slot))`— resident-in-cache (fast path) or freshly promoted; the
    ///                       caller uses `slot` as the request's `adapter_slot`.
    ///   * `Err(reject)`   — the promote was attempted but failed (pool full /
    ///                       peer error) → caller maps to 503 / 502.
    ///
    /// Coalesced single-flight: N concurrent misses for the SAME name collapse to
    /// ONE promote (they all resolve to the same slot). The coalescing lock is
    /// never held across the scheduler round-trip.
    pub async fn ensure_adapter_hot_opt(&self, name: &str) -> Result<Option<i32>, PromoteReject> {
        // Gate on (promotion armed + rotation channel) — NOT on peer_addr, so a
        // no-peer DISK-stageable serve still faults in. The per-backing branch
        // below decides how to source the adapter (RDMA peer or local disk).
        let (Some(promotion), Some(tx)) = (&self.promotion, &self.rotation_tx) else {
            return Ok(None);
        };
        // Classify the miss: peer-stageable (needs a peer_addr) or disk-stageable.
        // A peer-stageable name with no peer configured is inert (Ok(None), the
        // byte-identical resident-only 400 at the caller).
        enum Backing {
            Peer(crate::main_modules::promotion::StageableAdapter, String),
            Disk(std::path::PathBuf),
        }
        let backing = if let Some(st) = self.lora_stageable.get(name).cloned() {
            match &self.lora_peer_addr {
                Some(addr) => Backing::Peer(st, addr.clone()),
                None => return Ok(None),
            }
        } else if let Some((dir, _peft)) = self.lora_disk_stageable.get(name) {
            Backing::Disk(dir.clone())
        } else {
            return Ok(None);
        };

        // Fast path: already promoted and still mapped to a cache slot.
        if let Some(&slot) = self
            .promoted_slots
            .read()
            .expect("promoted_slots poisoned")
            .get(name)
        {
            return Ok(Some(slot));
        }

        let promoted_slots = Arc::clone(&self.promoted_slots);
        let tx = tx.clone();
        let name_owned = name.to_string();

        let slot = promotion
            .coalesce(name, move || async move {
                // Re-check the overlay under the leadership window — a prior
                // leader for this same name may have just landed it.
                if let Some(&slot) = promoted_slots
                    .read()
                    .expect("promoted_slots poisoned")
                    .get(&name_owned)
                {
                    return Ok(slot);
                }
                let (slot, evicted) = match backing {
                    Backing::Peer(st, addr) => {
                        Self::dispatch_promote(&tx, &addr, &name_owned, &st).await?
                    }
                    Backing::Disk(dir) => {
                        Self::dispatch_promote_disk(&tx, &name_owned, &dir).await?
                    }
                };
                // Update the overlay: drop the evicted name, map the new one.
                let mut ov = promoted_slots.write().expect("promoted_slots poisoned");
                if let Some(ev) = evicted {
                    ov.remove(&ev);
                }
                ov.insert(name_owned.clone(), slot);
                Ok(slot)
            })
            .await?;
        Ok(Some(slot))
    }

    /// True when `name` is a stageable (promotable-but-not-resident) adapter —
    /// either peer-backed (`--lora-stageable`) or disk-backed
    /// (`--lora-stageable-disk`). Used by routing/advertising to treat cold
    /// names as selectable without adding them to the resident `adapter_names`
    /// routing index.
    pub fn is_stageable_name(&self, name: &str) -> bool {
        self.lora_stageable.contains_key(name) || self.lora_disk_stageable.contains_key(name)
    }

    /// Send one `Promote` command to the scheduler and await its ack. The RDMA
    /// stage + victim selection run on the scheduler thread at quiescence; a
    /// bounded timeout surfaces a retryable `PoolFull` if the scheduler stays
    /// busy under sustained load (rather than hanging the request).
    async fn dispatch_promote(
        tx: &mpsc::Sender<crate::scheduler::LoraRotation>,
        peer_addr: &str,
        name: &str,
        stageable: &crate::main_modules::promotion::StageableAdapter,
    ) -> Result<(i32, Option<String>), PromoteReject> {
        use crate::scheduler::{LoraAck, LoraCommand};

        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        let cmd = LoraCommand::Promote {
            peer_addr: peer_addr.to_string(),
            adapter_id: stageable.peer_stage_id.clone(),
            name: name.to_string(),
            peft: stageable.peft.clone(),
        };
        if tx.send((cmd, ack_tx)).await.is_err() {
            return Err(PromoteReject::Peer(
                "scheduler promote channel closed".to_string(),
            ));
        }
        // Generous bound: a cold peer faults pages on first stage, and the drain
        // only runs at quiescence. On timeout the request retries.
        let acked = tokio::time::timeout(std::time::Duration::from_secs(30), ack_rx).await;
        match acked {
            Err(_timeout) => Err(PromoteReject::PoolFull(
                "promotion timed out waiting for scheduler quiescence; retry".to_string(),
            )),
            Ok(Err(_recv)) => Err(PromoteReject::Peer(
                "scheduler dropped the promote ack (shutting down?)".to_string(),
            )),
            Ok(Ok(Err(reason))) => {
                // POOL_FULL / busy-slot refusals are retryable; everything else
                // (peer/RDMA/config) is an upstream error.
                if reason.contains("POOL_FULL") || reason.contains("ref_count>0") {
                    Err(PromoteReject::PoolFull(reason))
                } else {
                    Err(PromoteReject::Peer(reason))
                }
            }
            Ok(Ok(Ok(LoraAck::Promoted { slot, evicted }))) => Ok((slot as i32, evicted)),
            Ok(Ok(Ok(LoraAck::Done))) => Err(PromoteReject::Peer(
                "scheduler returned a non-promote ack for a promote".to_string(),
            )),
        }
    }

    /// No-RDMA sibling of [`Self::dispatch_promote`]: send one `PromoteDisk`
    /// command (disk fault-in into an LRU victim slot) to the scheduler and
    /// await its ack. Same 30s quiescence bound and same retryable
    /// POOL_FULL / `ref_count>0` classification as the peer path; the disk swap
    /// re-parses the dir's `adapter_config.json`, so no peft arg is needed.
    async fn dispatch_promote_disk(
        tx: &mpsc::Sender<crate::scheduler::LoraRotation>,
        name: &str,
        dir: &std::path::Path,
    ) -> Result<(i32, Option<String>), PromoteReject> {
        use crate::scheduler::{LoraAck, LoraCommand};

        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        let cmd = LoraCommand::PromoteDisk {
            name: name.to_string(),
            dir: dir.to_path_buf(),
        };
        if tx.send((cmd, ack_tx)).await.is_err() {
            return Err(PromoteReject::Peer(
                "scheduler promote channel closed".to_string(),
            ));
        }
        let acked = tokio::time::timeout(std::time::Duration::from_secs(30), ack_rx).await;
        match acked {
            Err(_timeout) => Err(PromoteReject::PoolFull(
                "disk promotion timed out waiting for scheduler quiescence; retry".to_string(),
            )),
            Ok(Err(_recv)) => Err(PromoteReject::Peer(
                "scheduler dropped the promote ack (shutting down?)".to_string(),
            )),
            Ok(Ok(Err(reason))) => {
                if reason.contains("POOL_FULL") || reason.contains("ref_count>0") {
                    Err(PromoteReject::PoolFull(reason))
                } else {
                    Err(PromoteReject::Peer(reason))
                }
            }
            Ok(Ok(Ok(LoraAck::Promoted { slot, evicted }))) => Ok((slot as i32, evicted)),
            Ok(Ok(Ok(LoraAck::Done))) => Err(PromoteReject::Peer(
                "scheduler returned a non-promote ack for a disk promote".to_string(),
            )),
        }
    }
}

/// Re-export for convenience in api.rs / anthropic.rs.
pub type ModelBehavior = atlas_kernels::ModelBehavior;

#[cfg(test)]
mod tests {
    use super::resolve_adapter_slot;

    #[test]
    fn adapter_slot_resolution_rules() {
        let names = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
        // Unset defers to installed active (-1) — byte-identical to today.
        assert_eq!(resolve_adapter_slot(&names, None), Some(-1));
        // Known names resolve to their slot index (pool pack order).
        assert_eq!(resolve_adapter_slot(&names, Some("alpha")), Some(0));
        assert_eq!(resolve_adapter_slot(&names, Some("beta")), Some(1));
        assert_eq!(resolve_adapter_slot(&names, Some("gamma")), Some(2));
        // Unknown name → None → caller returns 400.
        assert_eq!(resolve_adapter_slot(&names, Some("delta")), None);
        // No adapters resident: any explicit name is unknown; None still defers.
        assert_eq!(resolve_adapter_slot(&[], Some("alpha")), None);
        assert_eq!(resolve_adapter_slot(&[], None), Some(-1));
    }
}
