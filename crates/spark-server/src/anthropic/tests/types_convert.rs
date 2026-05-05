// SPDX-License-Identifier: AGPL-3.0-only

use super::super::*;

#[test]
fn test_convert_stop_reason() {
    assert_eq!(convert_stop_reason("stop"), "end_turn");
    assert_eq!(convert_stop_reason("tool_calls"), "tool_use");
    assert_eq!(convert_stop_reason("length"), "max_tokens");
    assert_eq!(convert_stop_reason("unknown"), "end_turn");
}

#[test]
fn test_convert_tool_choice() {
    let any = AnthropicToolChoice {
        choice_type: "any".to_string(),
        name: None,
    };
    assert!(
        matches!(convert_tool_choice(&any), tool_parser::ToolChoice::Mode(s) if s == "required")
    );

    let auto = AnthropicToolChoice {
        choice_type: "auto".to_string(),
        name: None,
    };
    assert!(
        matches!(convert_tool_choice(&auto), tool_parser::ToolChoice::Mode(s) if s == "auto")
    );

    let specific = AnthropicToolChoice {
        choice_type: "tool".to_string(),
        name: Some("get_weather".to_string()),
    };
    assert!(
        matches!(convert_tool_choice(&specific), tool_parser::ToolChoice::Specific { function } if function.name == "get_weather")
    );
}

#[test]
fn test_flatten_content_text() {
    let content = AnthropicContent::Text("hello".to_string());
    let (text, calls) = flatten_content(&content);
    assert_eq!(text, "hello");
    assert!(calls.is_empty());
}

#[test]
fn test_flatten_content_blocks() {
    let content = AnthropicContent::Blocks(vec![
        ContentBlock::Text {
            text: "part1".to_string(),
        },
        ContentBlock::Text {
            text: "part2".to_string(),
        },
    ]);
    let (text, calls) = flatten_content(&content);
    assert_eq!(text, "part1part2");
    assert!(calls.is_empty());
}

#[test]
fn test_flatten_content_with_tool_use() {
    let content = AnthropicContent::Blocks(vec![
        ContentBlock::Text {
            text: "Let me check.".to_string(),
        },
        ContentBlock::ToolUse {
            id: "toolu_123".to_string(),
            name: "get_weather".to_string(),
            input: serde_json::json!({"location": "Paris"}),
        },
    ]);
    let (text, calls) = flatten_content(&content);
    assert_eq!(text, "Let me check.");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
}

#[test]
fn test_flatten_content_tool_result() {
    let content = AnthropicContent::Blocks(vec![ContentBlock::ToolResult {
        tool_use_id: "toolu_123".to_string(),
        content: Some(ToolResultContent::Text("Sunny, 22C".to_string())),
        is_error: None,
    }]);
    let (text, calls) = flatten_content(&content);
    assert_eq!(text, "Sunny, 22C");
    assert!(calls.is_empty());
}

#[test]
fn test_system_content_text() {
    let sys = SystemContent::Text("You are helpful.".to_string());
    assert_eq!(sys.to_text(), "You are helpful.");
}

#[test]
fn test_system_content_blocks() {
    let sys = SystemContent::Blocks(vec![
        SystemBlock {
            block_type: "text".to_string(),
            text: Some("You are helpful.".to_string()),
        },
        SystemBlock {
            block_type: "text".to_string(),
            text: Some("Be concise.".to_string()),
        },
    ]);
    assert_eq!(sys.to_text(), "You are helpful.\nBe concise.");
}

#[test]
fn test_convert_tools() {
    let tools = vec![AnthropicTool {
        name: "get_weather".to_string(),
        description: Some("Get weather".to_string()),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }),
    }];
    let oai = convert_tools(&tools);
    assert_eq!(oai.len(), 1);
    assert_eq!(oai[0].tool_type, "function");
    assert_eq!(oai[0].function.name, "get_weather");
    assert_eq!(oai[0].function.description.as_deref(), Some("Get weather"));
    assert!(oai[0].function.parameters.is_some());
}

#[test]
fn test_deserialize_messages_request() {
    let json = serde_json::json!({
        "model": "qwen3-80b",
        "max_tokens": 1024,
        "system": "You are helpful.",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    });
    let req: MessagesRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.model, "qwen3-80b");
    assert_eq!(req.max_tokens, 1024);
    assert!(matches!(req.system, Some(SystemContent::Text(ref s)) if s == "You are helpful."));
    assert_eq!(req.messages.len(), 1);
    assert_eq!(req.messages[0].role, "user");
}

#[test]
fn test_deserialize_content_blocks() {
    let json = serde_json::json!({
        "model": "qwen3-80b",
        "max_tokens": 1024,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "text", "text": " More text."}
            ]
        }]
    });
    let req: MessagesRequest = serde_json::from_value(json).unwrap();
    let (text, _) = flatten_content(&req.messages[0].content);
    assert_eq!(text, "What is this? More text.");
}

#[test]
fn test_deserialize_thinking_config() {
    let json = serde_json::json!({
        "model": "qwen3-80b",
        "max_tokens": 1024,
        "thinking": {"type": "enabled", "budget_tokens": 4096},
        "messages": [{"role": "user", "content": "Think hard."}]
    });
    let req: MessagesRequest = serde_json::from_value(json).unwrap();
    assert!(req.thinking.is_some());
    let t = req.thinking.unwrap();
    assert_eq!(t.thinking_type, "enabled");
    assert_eq!(t.budget_tokens, Some(4096));
}

#[test]
fn test_response_serialization() {
    let resp = MessagesResponse {
        id: "msg_123".to_string(),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content: vec![
            ResponseBlock::Thinking {
                thinking: "Let me think...".to_string(),
            },
            ResponseBlock::Text {
                text: "The answer is 42.".to_string(),
            },
        ],
        model: "qwen3-80b".to_string(),
        stop_reason: Some("end_turn".to_string()),
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens: 10,
            output_tokens: 20,
        },
    };
    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["type"], "message");
    assert_eq!(json["role"], "assistant");
    assert_eq!(json["content"].as_array().unwrap().len(), 2);
    assert_eq!(json["content"][0]["type"], "thinking");
    assert_eq!(json["content"][1]["type"], "text");
    assert_eq!(json["stop_reason"], "end_turn");
    assert_eq!(json["usage"]["input_tokens"], 10);
    assert_eq!(json["usage"]["output_tokens"], 20);
}

#[test]
fn test_tool_use_response_serialization() {
    let resp = MessagesResponse {
        id: "msg_456".to_string(),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content: vec![ResponseBlock::ToolUse {
            id: "toolu_abc".to_string(),
            name: "get_weather".to_string(),
            input: serde_json::json!({"location": "Paris"}),
        }],
        model: "qwen3-80b".to_string(),
        stop_reason: Some("tool_use".to_string()),
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens: 50,
            output_tokens: 30,
        },
    };
    let json = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["content"][0]["type"], "tool_use");
    assert_eq!(json["content"][0]["name"], "get_weather");
    assert_eq!(json["content"][0]["input"]["location"], "Paris");
    assert_eq!(json["stop_reason"], "tool_use");
}

// ── Translator tests (anthropic_to_chat_request_json) ──
