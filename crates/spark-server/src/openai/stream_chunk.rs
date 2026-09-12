// SPDX-License-Identifier: AGPL-3.0-only

//! OpenAI-compatible SSE streaming chunk types.
//!
//! Split out of `chat_response.rs` (2026-05-19) so each file
//! stays under the 500-LoC cap. Re-exported via `openai::*`;
//! no call site changes (all refs go through the glob).

use serde::Serialize;

use super::*;

// ── SSE streaming types (OpenAI format) ──

/// SSE streaming chunk.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
    pub logprobs: Option<ChoiceLogprobs>,
    /// Exact sampled token IDs for this chunk (vLLM-compatible
    /// `return_token_ids`). Skipped when empty so the default wire
    /// format is byte-identical for clients that did not opt in.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub token_ids: Vec<u32>,
    /// WHICH server-side degeneration guard cut this response, when one
    /// did — `"content_loop_watchdog"`, `"fuzzy_repetition"`,
    /// `"simhash_semantic_loop"`, `"token_loop_watchdog"`, … Set only on
    /// the terminal chunk (the one carrying `finish_reason`), and only
    /// when a guard actually fired; skipped otherwise, so a normal
    /// stop/length chunk is byte-identical to what shipped before.
    ///
    /// WHY A FIELD AND NOT A FINISH REASON (#927 / #1000 / #1002).
    /// Round-13 cell V (`--prefill-varlen-batch`) produced 6 of 16
    /// responses truncated at 49 tokens by the content-loop / fuzzy /
    /// SimHash watchdogs, and every one of them reported
    /// `finish_reason: "length"` on the wire — a client could not tell
    /// a quality cut from a budget stop. The wire `finish_reason` still
    /// must not move: relabelling guard cuts to `"stop"` measurably
    /// cost 2/10 then 6/10 episodes of the agentic gate
    /// (`scheduler::lifecycle::guard_stop_wire_reason`), and minting a
    /// fifth enum value hard-fails strictly typed clients (Rust
    /// `async-openai` fails deserialization outright; pydantic-ai
    /// raised on OpenRouter's non-standard `"error"`). Unknown FIELDS,
    /// by contrast, are ignored by every SDK — the same seam vLLM uses
    /// for its own `stop_reason`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Reasoning trace chunk (from <think>...</think>). Streamed during
    /// the thinking phase when enable_thinking=true. Cline and Roo Code
    /// both check for this field via `"reasoning_content" in delta`.
    /// Single canonical field — no `reasoning` mirror is emitted (a delta
    /// carrying both is rejected by strict OpenAI-compatible clients).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Omitted when absent — Cline, Roo Code, and most OpenAI-compatible clients
    /// expect content to be missing (not `null`) in role and done chunks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<crate::tool_parser::ChunkToolCall>>,
    /// Refusal signal. Atlas emits this on a single terminal delta chunk
    /// (just before the `done` chunk) when the accumulated streamed
    /// content matches a known refusal pattern. OpenAI's streaming
    /// refusal model is fragment-by-fragment; Atlas only classifies
    /// post-hoc so the signal lands as one chunk. Safety-aware clients
    /// that branch on `delta.refusal` will still see a non-null value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

impl ChatCompletionChunk {
    /// First chunk: sends the assistant role.
    pub fn role_chunk(model: &str, id: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some("assistant".to_string()),
                    reasoning_content: None,
                    content: None,
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: None,
                logprobs: None,
                token_ids: Vec::new(),
                stop_reason: None,
            }],
            usage: None,
        }
    }

    /// Reasoning content delta chunk (from <think>...</think> when enable_thinking=true).
    /// Cline and Roo Code check for `delta.reasoning_content` in streaming.
    pub fn reasoning_chunk(model: &str, id: &str, text: String) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: Some(text),
                    content: None,
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: None,
                logprobs: None,
                token_ids: Vec::new(),
                stop_reason: None,
            }],
            usage: None,
        }
    }

    /// Content delta chunk.
    pub fn content_chunk(model: &str, id: &str, text: String) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: None,
                    content: Some(text),
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: None,
                logprobs: None,
                token_ids: Vec::new(),
                stop_reason: None,
            }],
            usage: None,
        }
    }

    /// Tool call start chunk — emits role, id, type, name with empty arguments.
    /// Per OpenAI streaming spec, the first tool_call delta carries
    /// `role: "assistant"` and `content: null` alongside the metadata.
    pub fn tool_call_start_chunk(
        model: &str,
        id: &str,
        tc: &crate::tool_parser::ToolCall,
        tc_index: usize,
    ) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some("assistant".to_string()),
                    reasoning_content: None,
                    content: None,
                    tool_calls: Some(vec![crate::tool_parser::ChunkToolCall {
                        index: tc_index,
                        id: Some(tc.id.clone()),
                        call_type: Some(tc.call_type.clone()),
                        function: crate::tool_parser::ChunkFunction {
                            name: Some(tc.function.name.clone()),
                            arguments: String::new(),
                        },
                    }]),
                    refusal: None,
                },
                finish_reason: None,
                logprobs: None,
                token_ids: Vec::new(),
                stop_reason: None,
            }],
            usage: None,
        }
    }

    /// Tool call argument fragment chunk — emits a partial arguments string.
    /// Per OpenAI streaming spec, subsequent deltas carry incremental argument
    /// fragments. Callers should split the full arguments into small pieces
    /// (~20 chars) and call this for each fragment.
    pub fn tool_call_args_fragment(model: &str, id: &str, tc_index: usize, fragment: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: None,
                    content: None,
                    tool_calls: Some(vec![crate::tool_parser::ChunkToolCall {
                        index: tc_index,
                        id: None,
                        call_type: None,
                        function: crate::tool_parser::ChunkFunction {
                            name: None,
                            arguments: fragment.to_string(),
                        },
                    }]),
                    refusal: None,
                },
                finish_reason: None,
                logprobs: None,
                token_ids: Vec::new(),
                stop_reason: None,
            }],
            usage: None,
        }
    }

    /// Final chunk with finish_reason, empty delta, and usage.
    pub fn done_chunk(model: &str, id: &str, finish_reason: &str, usage: Usage) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: None,
                    content: None,
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: Some(finish_reason.to_string()),
                logprobs: None,
                token_ids: Vec::new(),
                stop_reason: None,
            }],
            usage: Some(usage),
        }
    }

    /// When `stream_options.include_usage=true`, OpenAI emits a separate
    /// chunk with `choices:[]` and populated `usage` BEFORE the final
    /// `finish_reason`-carrying chunk. This helper emits that usage-only
    /// chunk.
    pub fn usage_only_chunk(model: &str, id: &str, usage: Usage) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: Vec::new(),
            usage: Some(usage),
        }
    }

    /// Delta chunk carrying only `refusal: "<sentence>"`. Atlas emits
    /// this once, just before the terminal `done`/`usage_only` chunk,
    /// when `refusal::detect` classifies the accumulated streamed
    /// content as a refusal. OpenAI's streaming refusal model sends
    /// multiple `delta.refusal` fragments; we send a single post-hoc
    /// signal because Atlas classifies after the stream is complete.
    pub fn refusal_chunk(model: &str, id: &str, refusal: String) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: None,
                    content: None,
                    tool_calls: None,
                    refusal: Some(refusal),
                },
                finish_reason: None,
                logprobs: None,
                token_ids: Vec::new(),
                stop_reason: None,
            }],
            usage: None,
        }
    }

    /// Final chunk carrying only `finish_reason`, with `usage:null`. Used
    /// when `include_usage=true` (the usage sits in `usage_only_chunk`
    /// emitted just before this one).
    pub fn final_chunk_no_usage(model: &str, id: &str, finish_reason: &str) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_atlas".to_string()),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    reasoning_content: None,
                    content: None,
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: Some(finish_reason.to_string()),
                logprobs: None,
                token_ids: Vec::new(),
                stop_reason: None,
            }],
            usage: None,
        }
    }

    /// `return_token_ids`: stamp the exact sampled token IDs onto this
    /// chunk's first choice. No-op when `ids` is empty, so requests that
    /// did not opt in keep a byte-identical wire format. Summed across a
    /// stream, the stamped IDs equal `usage.completion_tokens` — a
    /// benchmark can count them instead of re-tokenizing decoded text.
    pub(crate) fn with_token_ids(mut self, ids: Vec<u32>) -> Self {
        if !ids.is_empty()
            && let Some(choice) = self.choices.first_mut()
        {
            choice.token_ids = ids;
        }
        self
    }

    /// Stamp the guard name onto this chunk's first choice. Same
    /// no-op-when-absent shape as [`Self::with_token_ids`]: `None`
    /// leaves the chunk byte-identical to what a client received
    /// before the field existed, so no snapshot of an ordinary
    /// stop/length response moves.
    ///
    /// Only the terminal chunk is stamped (`encode_stream` calls this
    /// on the `Finish` arm alone) — a guard cut is one event, not a
    /// property of every delta that preceded it. See
    /// [`ChunkChoice::stop_reason`] for the #927 / #1000 / #1002
    /// receipt behind the field.
    pub(crate) fn with_stop_reason(mut self, guard: Option<&'static str>) -> Self {
        if guard.is_some()
            && let Some(choice) = self.choices.first_mut()
        {
            choice.stop_reason = guard;
        }
        self
    }
}
