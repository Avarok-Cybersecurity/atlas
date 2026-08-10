// SPDX-License-Identifier: AGPL-3.0-only

//! Responses-API lowering tests: tool shapes, tool_choice, input items,
//! and stream event names.

use crate::openai::*;

fn lower_with_tools(
    tools: serde_json::Value,
) -> Result<ChatCompletionRequest, LowerResponsesError> {
    let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "input": "ping",
        "tools": tools,
    }))
    .expect("ResponsesRequest deserialize");
    lower_responses_to_chat(req, |_| None)
}

#[test]
fn responses_flat_function_tool_accepted() {
    // OpenAI's official Python SDK sends function tools in the flat
    // shape `{type, name, description, parameters}` — no nested
    // `function` object. Atlas must accept both shapes.
    let chat = lower_with_tools(serde_json::json!([
        {
            "type": "function",
            "name": "get_weather",
            "description": "look up weather",
            "parameters": {"type": "object", "properties": {"loc": {"type": "string"}}, "required": ["loc"]}
        }
    ])).expect("flat-form function tool should parse");
    let tools = chat.tools.expect("tools present");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].tool_type, "function");
    assert_eq!(tools[0].function.name, "get_weather");
}

#[test]
fn responses_nested_function_tool_still_accepted() {
    // Backwards-compat: chat-completions-style `{type, function:{...}}`
    // must keep working since older clients send it.
    let chat = lower_with_tools(serde_json::json!([
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object"}
            }
        }
    ]))
    .expect("nested-form function tool should parse");
    let tools = chat.tools.expect("tools present");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");
}

#[test]
fn responses_flat_tool_choice_accepted() {
    let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "input": "ping",
        "tool_choice": {"type": "function", "name": "get_weather"},
    }))
    .unwrap();
    let chat = lower_responses_to_chat(req, |_| None).expect("flat tool_choice");
    match chat.tool_choice {
        Some(crate::tool_parser::ToolChoice::Specific { function }) => {
            assert_eq!(function.name, "get_weather");
        }
        other => panic!("expected Specific tool_choice, got {other:?}"),
    }
}

#[test]
fn responses_string_tool_choice_accepted() {
    let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "input": "ping",
        "tool_choice": "required",
    }))
    .unwrap();
    let chat = lower_responses_to_chat(req, |_| None).expect("string tool_choice");
    match chat.tool_choice {
        Some(crate::tool_parser::ToolChoice::Mode(s)) => {
            assert_eq!(s, "required");
        }
        other => panic!("expected Mode tool_choice, got {other:?}"),
    }
}

#[test]
fn responses_input_image_string_url_is_carried() {
    // Regression: the Responses adapter used to drop images entirely
    // (`images: Vec::new()`), silently losing multimodal input.
    let item = serde_json::json!({
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_text", "text": "what is this?"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAA"}
        ]
    });
    let m = IncomingMessage::from_responses_input_item(&item).expect("message");
    assert_eq!(m.content.text, "what is this?");
    assert_eq!(
        m.content.images,
        vec!["data:image/png;base64,AAA".to_string()]
    );
}

#[test]
fn responses_input_image_object_url_is_carried() {
    // Some SDKs nest the url: `{image_url: {url: "..."}}`.
    let item = serde_json::json!({
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": {"url": "https://example.com/a.png"}}
        ]
    });
    let m = IncomingMessage::from_responses_input_item(&item).expect("message");
    assert_eq!(
        m.content.images,
        vec!["https://example.com/a.png".to_string()]
    );
    assert_eq!(m.content.text, "");
}

#[test]
fn responses_function_call_output_image_is_carried() {
    // #165 parity for the Responses surface: a screenshot returned by a
    // tool (`function_call_output` with structured output parts) must
    // reach the vision encoder, exactly like Anthropic tool_result
    // images and chat-completions role:"tool" array content.
    let item = serde_json::json!({
        "type": "function_call_output",
        "call_id": "call_1",
        "output": [
            {"type": "output_text", "text": "screenshot follows"},
            {"type": "input_image", "image_url": "data:image/png;base64,BBB"}
        ]
    });
    let m = IncomingMessage::from_responses_input_item(&item).expect("tool message");
    assert_eq!(m.role, "tool");
    assert_eq!(m.tool_call_id.as_deref(), Some("call_1"));
    assert_eq!(m.content.text, "screenshot follows");
    assert_eq!(
        m.content.images,
        vec!["data:image/png;base64,BBB".to_string()]
    );
}

#[test]
fn responses_function_call_output_string_unchanged() {
    let item = serde_json::json!({
        "type": "function_call_output",
        "call_id": "call_2",
        "output": "plain result"
    });
    let m = IncomingMessage::from_responses_input_item(&item).expect("tool message");
    assert_eq!(m.content.text, "plain result");
    assert!(m.content.images.is_empty());
}

#[test]
fn responses_function_call_output_opaque_array_stringified() {
    // Out-of-spec array (no recognizable parts) keeps the historical
    // stringified-JSON behavior instead of silently emptying the result.
    let opaque = serde_json::json!([{"weather": "sunny", "temp_c": 21}]);
    let item = serde_json::json!({
        "type": "function_call_output",
        "call_id": "call_3",
        "output": opaque.clone()
    });
    let m = IncomingMessage::from_responses_input_item(&item).expect("tool message");
    assert_eq!(m.content.text, opaque.to_string());
    assert!(m.content.images.is_empty());
}

#[test]
fn responses_in_progress_event_name() {
    let ev = ResponsesStreamEvent::InProgress {
        sequence_number: 1,
        response: ResponsesStreamEnvelope {
            id: "resp_test".into(),
            object: "response",
            created_at: 0,
            model: "m".into(),
            status: "in_progress",
            metadata: None,
        },
    };
    assert_eq!(responses_event_name(&ev), "response.in_progress");
}
