// SPDX-License-Identifier: AGPL-3.0-only
//
// Adapter: Anthropic Messages API wire request → canonical chat IR.
// Replaces the historical hand-built OpenAI-wire-JSON hop (the request
// used to be rewritten into chat-completions JSON, re-parsed as an
// OpenAI request, and only then lowered) — the Anthropic surface is
// now a peer adapter of the OpenAI edge, and structural information
// (tool_result `is_error`, image source kinds) survives as typed IR
// instead of text conventions.

use crate::ir;
use crate::ir::message::{ImageSource, Reasoning, ToolCall};
use crate::ir::{ContentPart, ImageData, Message, Role, ThinkingDirective};

use super::types::{
    AnthropicContent, ContentBlock, MessagesRequest, SystemContent, ToolResultContent,
};

impl From<MessagesRequest> for ir::ChatRequest {
    /// Lower the parsed Anthropic wire request into the
    /// provider-agnostic [`ir::ChatRequest`] envelope.
    ///
    /// Message mapping (mirrors the retired JSON-hop translation
    /// byte-for-byte at the rendered-prompt level):
    /// * `system` (string or `text` blocks, `x-anthropic-*` billing
    ///   blocks filtered) → one leading System message when non-empty.
    /// * assistant blocks → ONE assistant message: `[image*, text]`
    ///   parts, `tool_use` → structured tool_calls (arguments stay
    ///   parsed JSON), `thinking` blocks → first-class reasoning
    ///   (joined with `\n`).
    /// * user blocks → an optional user message (`[image*, text]`,
    ///   only when text/images exist) followed by one Tool message per
    ///   `tool_result` in block order. `is_error` travels as
    ///   `Message::tool_error` — the `[tool error]\n` marker is
    ///   rendered by the shared pipeline (`msg_entry`), not baked into
    ///   the text here.
    /// * unknown roles collapse to user (Anthropic wire only defines
    ///   user/assistant).
    fn from(req: MessagesRequest) -> Self {
        let mut messages: Vec<Message> = Vec::with_capacity(req.messages.len() + 1);

        // System message (filter x-anthropic- billing/config blocks).
        if let Some(sys) = &req.system {
            let sys_text = match sys {
                SystemContent::Text(s) => s.clone(),
                SystemContent::Blocks(blocks) => blocks
                    .iter()
                    .filter(|b| {
                        b.block_type == "text"
                            && !b.text.as_deref().unwrap_or("").starts_with("x-anthropic-")
                    })
                    .filter_map(|b| b.text.clone())
                    .collect::<Vec<_>>()
                    .join("\n"),
            };
            if !sys_text.is_empty() {
                messages.push(Message::synthetic_system(sys_text));
            }
        }

        // Conversation history.
        for m in req.messages {
            let role = match m.role.as_str() {
                "assistant" => Role::Assistant,
                _ => Role::User,
            };
            match m.content {
                AnthropicContent::Text(s) => {
                    let content = if s.is_empty() {
                        Vec::new()
                    } else {
                        vec![ContentPart::Text(s)]
                    };
                    messages.push(Message {
                        role,
                        content,
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                        name: None,
                        reasoning: None,
                        tool_error: false,
                    });
                }
                AnthropicContent::Blocks(blocks) => {
                    let mut text_parts: Vec<String> = Vec::new();
                    let mut images: Vec<String> = Vec::new();
                    let mut reasoning_parts: Vec<String> = Vec::new();
                    let mut tool_calls: Vec<ToolCall> = Vec::new();
                    // (tool_use_id, text, images, is_error)
                    let mut tool_results: Vec<(String, String, Vec<String>, bool)> = Vec::new();
                    for b in blocks {
                        match b {
                            ContentBlock::Text { text } => text_parts.push(text),
                            ContentBlock::Image { source } => {
                                if let Some(uri) = source.maybe_get_image_uri() {
                                    images.push(uri);
                                }
                            }
                            ContentBlock::ToolUse { id, name, input } => {
                                // `input` is already structured JSON — no
                                // stringify/re-parse roundtrip.
                                tool_calls.push(ToolCall {
                                    id,
                                    name,
                                    arguments: input,
                                });
                            }
                            ContentBlock::ToolResult {
                                tool_use_id,
                                content,
                                is_error,
                            } => {
                                let text =
                                    content.as_ref().map(|c| c.to_text()).unwrap_or_default();
                                // Carry images embedded in a tool result
                                // (e.g. a screenshot the tool returned) so
                                // they reach the vision encoder (issue #165).
                                let tr_images: Vec<String> = match content {
                                    Some(ToolResultContent::Blocks(inner)) => inner
                                        .iter()
                                        .filter_map(|ib| match ib {
                                            ContentBlock::Image { source } => {
                                                source.maybe_get_image_uri()
                                            }
                                            _ => None,
                                        })
                                        .collect(),
                                    _ => Vec::new(),
                                };
                                tool_results.push((
                                    tool_use_id,
                                    text,
                                    tr_images,
                                    is_error.unwrap_or(false),
                                ));
                            }
                            ContentBlock::Thinking { thinking } => {
                                if let Some(t) = thinking
                                    && !t.is_empty()
                                {
                                    reasoning_parts.push(t);
                                }
                            }
                            ContentBlock::Unknown => {}
                        }
                    }
                    let text_content = text_parts.join("");
                    if role == Role::Assistant {
                        messages.push(Message {
                            role: Role::Assistant,
                            content: parts_from(&images, &text_content),
                            tool_calls,
                            tool_call_id: None,
                            name: None,
                            reasoning: if reasoning_parts.is_empty() {
                                None
                            } else {
                                Some(Reasoning {
                                    text: reasoning_parts.join("\n"),
                                })
                            },
                            tool_error: false,
                        });
                    } else {
                        if !text_content.is_empty() || !images.is_empty() {
                            messages.push(Message {
                                role: Role::User,
                                content: parts_from(&images, &text_content),
                                tool_calls: Vec::new(),
                                tool_call_id: None,
                                name: None,
                                reasoning: None,
                                tool_error: false,
                            });
                        }
                        for (tool_use_id, text, tr_images, is_error) in tool_results {
                            messages.push(Message {
                                role: Role::Tool,
                                content: parts_from(&tr_images, &text),
                                tool_calls: Vec::new(),
                                tool_call_id: Some(tool_use_id),
                                name: None,
                                reasoning: None,
                                tool_error: is_error,
                            });
                        }
                    }
                }
            }
        }

        // Anthropic thinking config → neutral directive. Same rungs the
        // OpenAI ladder applies to its `thinking` channel: disabled wins,
        // then an explicit budget, then budget-less enabled/adaptive
        // (think as long as needed — defer to the per-model cap).
        let thinking = match &req.thinking {
            None => ThinkingDirective::Unspecified,
            Some(tc) if tc.thinking_type == "disabled" => ThinkingDirective::Off,
            Some(tc) => match tc.budget_tokens {
                Some(b) => ThinkingDirective::On {
                    budget: Some(u32::try_from(b).unwrap_or(u32::MAX)),
                },
                None => ThinkingDirective::On { budget: None },
            },
        };

        ir::ChatRequest {
            model: req.model,
            messages,
            tools: req
                .tools
                .as_deref()
                .map(|ts| ts.iter().map(Into::into).collect())
                .unwrap_or_default(),
            tool_choice: req.tool_choice.as_ref().map(Into::into),
            sampling: ir::SamplingParams {
                temperature: req.temperature,
                top_k: req.top_k,
                top_p: req.top_p,
                ..Default::default()
            },
            max_tokens: req.max_tokens,
            min_tokens: 0,
            stop: req.stop_sequences,
            stream: req.stream,
            n: 1,
            response_format: None,
            thinking,
            repetition_detection: None,
            adapter: None,
            src_lang: None,
            tgt_lang: None,
            num_beams: None,
            length_penalty: None,
            early_stopping: None,
            logit_bias: Vec::new(),
            top_logprobs: None,
            seed: None,
            timeout_secs: None,
            return_token_ids: false,
        }
    }
}

/// `[image*, text]` content parts — image order preserved, single
/// joined text part last, omitted when empty. Matches the OpenAI
/// adapter's shape so the template sees one canonical layout.
fn parts_from(images: &[String], text: &str) -> Vec<ContentPart> {
    let mut content: Vec<ContentPart> = Vec::with_capacity(images.len() + 1);
    for uri in images {
        content.push(ContentPart::Image(ImageSource {
            data: ImageData::from_uri(uri.clone()),
        }));
    }
    if !text.is_empty() {
        content.push(ContentPart::Text(text.to_string()));
    }
    content
}
