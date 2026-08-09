// SPDX-License-Identifier: AGPL-3.0-only
//
// Adapter: OpenAI Chat Completions wire request → canonical chat IR.
// This is the OpenAI request-direction edge of the narrow waist: the
// wire message becomes `ir::Message`, and the whole wire request
// becomes the `ir::ChatRequest` envelope. Wire-only knobs (logit_bias
// string keys, the logprobs/top_logprobs pair, response_format "text",
// the five thinking channels) are resolved HERE so no internal module
// ever reads them.

use super::{ChatCompletionRequest, IncomingMessage, ResponseFormat};
use crate::ir;
use crate::ir::message::{ContentPart, ImageData, ImageSource, Message, Reasoning, Role, ToolCall};

impl From<&IncomingMessage> for Message {
    fn from(m: &IncomingMessage) -> Self {
        // Images first (matching the template's historical `[image*N,
        // text]` content-array shape), then a single text part. The
        // original wire interleaving was already flattened away by
        // `ParsedContent`, and `build_msg_entries`/the template read
        // `text()` and `image_count()` independently, so part order here
        // does not affect rendering.
        let mut content: Vec<ContentPart> = Vec::new();
        for img in &m.content.images {
            content.push(ContentPart::Image(ImageSource {
                data: ImageData::from_uri(img.clone()),
            }));
        }
        if !m.content.text.is_empty() {
            content.push(ContentPart::Text(m.content.text.clone()));
        }

        // Tool calls: parse the wire `arguments` string into structured
        // JSON exactly as the historical path did
        // (`from_str(..).unwrap_or(Object::default())`); a missing id
        // becomes the empty string, also matching today.
        let tool_calls: Vec<ToolCall> = m
            .tool_calls
            .as_ref()
            .map(|tcs| {
                tcs.iter()
                    .map(|tc| ToolCall {
                        id: tc.id.clone().unwrap_or_default(),
                        name: tc.function.name.clone(),
                        arguments: serde_json::from_str(&tc.function.arguments)
                            .unwrap_or_else(|_| serde_json::Value::Object(Default::default())),
                    })
                    .collect()
            })
            .unwrap_or_default();

        Message {
            role: Role::from(m.role.as_str()),
            content,
            tool_calls,
            tool_call_id: m.tool_call_id.clone(),
            name: m.name.clone(),
            reasoning: m.reasoning_content.clone().map(|text| Reasoning { text }),
            tool_error: false,
        }
    }
}

impl From<ChatCompletionRequest> for ir::ChatRequest {
    /// Lower the parsed OpenAI wire request into the provider-agnostic
    /// [`ir::ChatRequest`] envelope. Infallible: range validation
    /// happens on the envelope (`chat_phases::validate_input`), and the
    /// historical lenient parses (non-numeric logit_bias keys dropped,
    /// malformed tool args → `{}`) are preserved.
    ///
    /// Echo-only fields (service_tier, store, metadata,
    /// stream_options) are NOT lowered — the handler keeps the wire
    /// request for encode-time echoes.
    fn from(req: ChatCompletionRequest) -> Self {
        let thinking = req.client_thinking_directive();
        let top_logprobs = resolve_top_logprobs(req.logprobs, req.top_logprobs);
        // Logit bias: OpenAI's string-keyed map → typed pairs. Keys that
        // don't parse as token ids are dropped (historical behavior).
        let logit_bias: Vec<(u32, f32)> = req.logit_bias.as_ref().map_or(Vec::new(), |map| {
            map.iter()
                .filter_map(|(k, &v)| k.parse::<u32>().ok().map(|id| (id, v)))
                .collect()
        });
        let response_format = match req.response_format {
            // "text" is the wire spelling of "no constraint".
            None | Some(ResponseFormat::Text) => None,
            Some(ResponseFormat::JsonObject) => Some(ir::ResponseFormat::JsonObject),
            Some(ResponseFormat::JsonSchema { json_schema }) => {
                Some(ir::ResponseFormat::JsonSchema {
                    name: json_schema.name,
                    schema: json_schema.schema,
                    strict: json_schema.strict,
                })
            }
        };
        ir::ChatRequest {
            model: req.model,
            messages: req.messages.iter().map(Into::into).collect(),
            tools: req.tools.unwrap_or_default(),
            tool_choice: req.tool_choice,
            sampling: ir::SamplingParams {
                temperature: req.temperature,
                top_k: req.top_k,
                top_p: req.top_p,
                top_n_sigma: req.top_n_sigma,
                min_p: req.min_p,
                repetition_penalty: req.repetition_penalty,
                presence_penalty: req.presence_penalty,
                frequency_penalty: req.frequency_penalty,
            },
            max_tokens: req.max_tokens,
            min_tokens: req.min_tokens,
            stop: req.stop,
            stream: req.stream,
            n: req.n,
            response_format,
            thinking,
            repetition_detection: req.repetition_detection,
            adapter: req.adapter,
            src_lang: req.src_lang,
            tgt_lang: req.tgt_lang,
            num_beams: req.num_beams,
            length_penalty: req.length_penalty,
            early_stopping: req.early_stopping,
            logit_bias,
            top_logprobs,
            seed: req.seed,
            timeout_secs: req.timeout,
            return_token_ids: req.return_token_ids,
        }
    }
}

/// Resolve chat logprobs params (OpenAI spec): an explicit
/// `top_logprobs` count wins (clamped 0-20); `logprobs: true` alone
/// enables sampled-token logprobs with no alternatives (count 0);
/// otherwise disabled.
pub(crate) fn resolve_top_logprobs(logprobs: Option<bool>, top_logprobs: Option<u8>) -> Option<u8> {
    match (logprobs, top_logprobs) {
        (_, Some(n)) => Some(n.min(20)),
        (Some(true), None) => Some(0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::message::{ContentPart, ImageData, ImageSource, Reasoning};
    use crate::openai::{IncomingMessage, ParsedContent};
    use crate::tool_parser::{IncomingFunction, IncomingToolCall};

    fn msg(role: &str) -> IncomingMessage {
        IncomingMessage {
            role: role.to_string(),
            content: ParsedContent::default(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
        }
    }

    #[test]
    fn text_only_user_message() {
        let mut m = msg("user");
        m.content.text = "hi".into();
        let ir: Message = (&m).into();
        assert_eq!(ir.role, Role::User);
        assert_eq!(ir.content, vec![ContentPart::Text("hi".into())]);
        assert_eq!(ir.image_count(), 0);
        assert!(ir.tool_calls.is_empty());
        assert!(ir.reasoning.is_none());
        assert!(!ir.tool_error);
    }

    #[test]
    fn empty_content_yields_no_parts() {
        let m = msg("user");
        let ir: Message = (&m).into();
        assert!(ir.content.is_empty());
        assert_eq!(ir.text(), "");
    }

    #[test]
    fn images_precede_text_and_are_preserved_verbatim() {
        let mut m = msg("user");
        m.content.text = "see".into();
        m.content.images = vec!["data:image/png;base64,AAA".into()];
        let ir: Message = (&m).into();
        // Images first (matches the template's [image*N, text] order), then text.
        assert_eq!(
            ir.content,
            vec![
                ContentPart::Image(ImageSource {
                    data: ImageData::Base64("data:image/png;base64,AAA".into()),
                }),
                ContentPart::Text("see".into()),
            ]
        );
        assert_eq!(ir.image_count(), 1);
    }

    #[test]
    fn remote_url_image_classified_as_url_variant() {
        // http(s) strings must become ImageData::Url so the pipeline can
        // reject them explicitly — mislabeling them Base64 produced a
        // confusing "base64 decode failed" from the vision preprocessor.
        let mut m = msg("user");
        m.content.images = vec![
            "https://example.com/cat.png".into(),
            "data:image/png;base64,AAA".into(),
        ];
        let ir: Message = (&m).into();
        assert_eq!(
            ir.content,
            vec![
                ContentPart::Image(ImageSource {
                    data: ImageData::Url("https://example.com/cat.png".into()),
                }),
                ContentPart::Image(ImageSource {
                    data: ImageData::Base64("data:image/png;base64,AAA".into()),
                }),
            ]
        );
    }

    #[test]
    fn assistant_tool_calls_parse_arguments_to_json() {
        let mut m = msg("assistant");
        m.tool_calls = Some(vec![IncomingToolCall {
            id: Some("call_1".into()),
            function: IncomingFunction {
                name: "get_weather".into(),
                arguments: r#"{"city":"SF"}"#.into(),
            },
        }]);
        let ir: Message = (&m).into();
        assert_eq!(ir.role, Role::Assistant);
        assert_eq!(ir.tool_calls.len(), 1);
        assert_eq!(ir.tool_calls[0].id, "call_1");
        assert_eq!(ir.tool_calls[0].name, "get_weather");
        assert_eq!(
            ir.tool_calls[0].arguments,
            serde_json::json!({"city": "SF"})
        );
    }

    #[test]
    fn malformed_tool_args_default_to_empty_object() {
        // Mirrors the historical `from_str(...).unwrap_or(Object::default())`.
        let mut m = msg("assistant");
        m.tool_calls = Some(vec![IncomingToolCall {
            id: None,
            function: IncomingFunction {
                name: "f".into(),
                arguments: "not json".into(),
            },
        }]);
        let ir: Message = (&m).into();
        assert_eq!(ir.tool_calls[0].id, ""); // missing id → empty string (preserved)
        assert_eq!(ir.tool_calls[0].arguments, serde_json::json!({}));
    }

    #[test]
    fn reasoning_content_maps_to_first_class_reasoning() {
        let mut m = msg("assistant");
        m.reasoning_content = Some("ponder".into());
        let ir: Message = (&m).into();
        assert_eq!(
            ir.reasoning,
            Some(Reasoning {
                text: "ponder".into()
            })
        );

        let m2 = msg("assistant");
        let ir2: Message = (&m2).into();
        assert!(ir2.reasoning.is_none());
    }

    #[test]
    fn tool_message_preserves_call_id_and_name() {
        let mut m = msg("tool");
        m.content.text = "exit 0".into();
        m.tool_call_id = Some("call_1".into());
        m.name = Some("bash".into());
        let ir: Message = (&m).into();
        assert_eq!(ir.role, Role::Tool);
        assert_eq!(ir.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(ir.name.as_deref(), Some("bash"));
        assert_eq!(ir.text(), "exit 0");
    }

    #[test]
    fn unknown_role_is_preserved_losslessly() {
        let m = msg("developer");
        let ir: Message = (&m).into();
        assert_eq!(ir.role, Role::Other("developer".into()));
        assert_eq!(ir.role.as_wire(), "developer");
    }

    // ── whole-request envelope lowering ──

    fn wire(body: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(body).expect("valid chat request")
    }

    #[test]
    fn envelope_lowers_scalars_and_parses_logit_bias() {
        let req = wire(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
            "temperature": 0.5,
            "logit_bias": {"42": 1.5, "not-a-token": -1.0},
            "logprobs": true,
            "stop": ["END"],
            "seed": 7,
            "n": 2
        }));
        let ir = ir::ChatRequest::from(req);
        assert_eq!(ir.model, "m");
        assert_eq!(ir.messages.len(), 1);
        assert_eq!(ir.max_tokens, 64);
        assert_eq!(ir.sampling.temperature, Some(0.5));
        // Non-numeric keys dropped (historical behavior).
        assert_eq!(ir.logit_bias, vec![(42, 1.5)]);
        // logprobs:true alone → count 0 (sampled token only).
        assert_eq!(ir.top_logprobs, Some(0));
        assert_eq!(ir.stop, vec!["END".to_string()]);
        assert_eq!(ir.seed, Some(7));
        assert_eq!(ir.n, 2);
        assert!(ir.tools.is_empty());
        assert!(ir.response_format.is_none());
    }

    #[test]
    fn envelope_maps_text_response_format_to_none() {
        let req = wire(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "text"}
        }));
        assert!(ir::ChatRequest::from(req).response_format.is_none());

        let req = wire(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "s", "schema": {"type": "object"}}}
        }));
        match ir::ChatRequest::from(req).response_format {
            Some(ir::ResponseFormat::JsonSchema {
                name,
                schema,
                strict,
            }) => {
                assert_eq!(name, "s");
                assert_eq!(schema, serde_json::json!({"type": "object"}));
                assert!(strict); // default_true
            }
            other => panic!("expected JsonSchema, got {other:?}"),
        }
    }

    #[test]
    fn resolve_top_logprobs_matrix() {
        assert_eq!(resolve_top_logprobs(Some(true), None), Some(0));
        assert_eq!(resolve_top_logprobs(None, Some(5)), Some(5));
        assert_eq!(resolve_top_logprobs(Some(false), Some(3)), Some(3));
        assert_eq!(resolve_top_logprobs(Some(true), Some(99)), Some(20));
        assert_eq!(resolve_top_logprobs(None, None), None);
        assert_eq!(resolve_top_logprobs(Some(false), None), None);
    }
}
