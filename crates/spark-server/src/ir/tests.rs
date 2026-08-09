// SPDX-License-Identifier: AGPL-3.0-only

use super::message::*;

#[test]
fn role_wire_roundtrip() {
    for (role, wire) in [
        (Role::System, "system"),
        (Role::User, "user"),
        (Role::Assistant, "assistant"),
        (Role::Tool, "tool"),
    ] {
        assert_eq!(role.as_wire(), wire);
        assert_eq!(Role::from_wire(wire), Some(role));
    }
    // Unknown roles do not silently map to a default.
    assert_eq!(Role::from_wire("function"), None);
}

#[test]
fn text_concatenates_text_parts_in_order_ignoring_images() {
    let m = Message {
        role: Role::User,
        content: vec![
            ContentPart::Text("a".into()),
            ContentPart::Image(ImageSource {
                data: ImageData::Base64("img".into()),
            }),
            ContentPart::Text("b".into()),
        ],
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
        reasoning: None,
        tool_error: false,
    };
    // Mirrors the historical `text_parts.join("")` flattening.
    assert_eq!(m.text(), "ab");
    assert_eq!(m.image_count(), 1);
}

#[test]
fn assistant_tool_call_and_reasoning_are_first_class() {
    let m = Message {
        role: Role::Assistant,
        content: vec![ContentPart::Text("hi".into())],
        tool_calls: vec![ToolCall {
            id: "call_1".into(),
            name: "get_weather".into(),
            arguments: serde_json::json!({"city": "SF"}),
        }],
        tool_call_id: None,
        name: None,
        reasoning: Some(Reasoning {
            text: "let me think".into(),
        }),
        tool_error: false,
    };
    assert_eq!(m.role.as_wire(), "assistant");
    assert_eq!(m.tool_calls[0].name, "get_weather");
    assert_eq!(m.tool_calls[0].arguments["city"], "SF");
    assert_eq!(m.reasoning.as_ref().unwrap().text, "let me think");
    assert_eq!(m.image_count(), 0);
}

#[test]
fn tool_message_carries_error_flag_and_call_id() {
    let m = Message {
        role: Role::Tool,
        content: vec![ContentPart::Text("exit 127".into())],
        tool_calls: Vec::new(),
        tool_call_id: Some("call_1".into()),
        name: Some("bash".into()),
        reasoning: None,
        tool_error: true,
    };
    assert!(m.tool_error);
    assert_eq!(m.tool_call_id.as_deref(), Some("call_1"));
    assert_eq!(m.text(), "exit 127");
}
