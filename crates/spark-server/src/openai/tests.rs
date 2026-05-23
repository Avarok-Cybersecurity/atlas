// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn url_of(a: &Annotation) -> (usize, usize, &str, &str) {
    match a {
        Annotation::UrlCitation {
            url_citation:
                UrlCitation {
                    start_index,
                    end_index,
                    url,
                    title,
                },
        } => (*start_index, *end_index, url.as_str(), title.as_str()),
    }
}

#[test]
fn bare_url_extracted() {
    let got = extract_url_annotations("see https://example.com/foo for more").unwrap();
    assert_eq!(got.len(), 1);
    let (s, e, u, t) = url_of(&got[0]);
    assert_eq!(u, "https://example.com/foo");
    assert_eq!(t, "https://example.com/foo");
    assert_eq!(s, 4);
    assert_eq!(e, 4 + "https://example.com/foo".len());
}

#[test]
fn trailing_sentence_punct_stripped() {
    let got = extract_url_annotations("go to https://example.com.").unwrap();
    let (_, _, u, _) = url_of(&got[0]);
    assert_eq!(u, "https://example.com");
}

#[test]
fn wikipedia_parens_preserved() {
    let got = extract_url_annotations("see https://en.wikipedia.org/wiki/Foo_(bar) now").unwrap();
    let (_, _, u, _) = url_of(&got[0]);
    assert_eq!(u, "https://en.wikipedia.org/wiki/Foo_(bar)");
}

#[test]
fn markdown_link_uses_title() {
    let got = extract_url_annotations("read [the docs](https://example.com/api) today").unwrap();
    assert_eq!(got.len(), 1);
    let (_, _, u, t) = url_of(&got[0]);
    assert_eq!(u, "https://example.com/api");
    assert_eq!(t, "the docs");
}

#[test]
fn url_in_fenced_code_skipped() {
    let input = "run this:\n```bash\ncurl https://example.com\n```\ndone";
    assert!(extract_url_annotations(input).is_none());
}

#[test]
fn url_in_inline_code_skipped() {
    let input = "use `curl https://example.com` to fetch";
    assert!(extract_url_annotations(input).is_none());
}

#[test]
fn multiple_urls_sorted_by_position() {
    let input = "first https://a.example.com and [second](https://b.example.com)";
    let got = extract_url_annotations(input).unwrap();
    assert_eq!(got.len(), 2);
    let (s0, _, u0, _) = url_of(&got[0]);
    let (s1, _, u1, _) = url_of(&got[1]);
    assert!(s0 < s1);
    assert_eq!(u0, "https://a.example.com");
    assert_eq!(u1, "https://b.example.com");
}

#[test]
fn non_http_ignored() {
    assert!(extract_url_annotations("ftp://example.com not a citation").is_none());
}

#[test]
fn empty_input_returns_none() {
    assert!(extract_url_annotations("").is_none());
    assert!(extract_url_annotations("no URLs here").is_none());
}

#[test]
fn query_and_fragment_preserved() {
    let got = extract_url_annotations("see https://example.com/p?q=1&r=2#frag here").unwrap();
    let (_, _, u, _) = url_of(&got[0]);
    assert_eq!(u, "https://example.com/p?q=1&r=2#frag");
}

// TODO: `mask_code_spans` was an internal helper that no longer exists
// after the URL-annotations refactor. The remaining call to
// `extract_url_annotations` is exercised by the other tests in this file;
// re-add a UTF-8 boundary test once the new internal mask helper has a
// stable name.

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
fn markdown_link_with_parens_in_url_preserved() {
    // Wikipedia URLs contain `(...)` which the bare `find(')')` would
    // truncate. Verify the balanced-paren scan keeps the full URL.
    let got =
        extract_url_annotations("see [Foo (bar)](https://en.wikipedia.org/wiki/Foo_(bar)) here")
            .unwrap();
    assert_eq!(got.len(), 1);
    let (_, _, u, t) = url_of(&got[0]);
    assert_eq!(u, "https://en.wikipedia.org/wiki/Foo_(bar)");
    assert_eq!(t, "Foo (bar)");
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

// ── ChatTemplateKwargs ────────────────────────────────────────────

#[test]
fn chat_template_kwargs_parse() {
    let kw = ChatTemplateKwargs::from_json(r#"{"enable_thinking":true,"thinking_budget":1024}"#)
        .expect("should parse");
    assert_eq!(kw.enable_thinking, Some(true));
    assert_eq!(kw.thinking_budget, Some(1024));

    assert!(ChatTemplateKwargs::from_json("").is_none());
}

fn empty_chat_request() -> ChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
    }))
    .expect("valid chat request")
}

#[test]
fn server_default_merged_when_request_silent() {
    let mut req = empty_chat_request();
    assert!(req.chat_template_kwargs.is_none());

    let server_kw = ChatTemplateKwargs {
        enable_thinking: Some(true),
        thinking_budget: None,
    };
    if !req.thinking_explicitly_requested() {
        req.chat_template_kwargs = Some(server_kw);
    }
    assert!(req.chat_template_kwargs.is_some());

    let (enabled, budget) = req.resolve_thinking(false);
    assert!(enabled);
    assert!(budget.is_some());
}

#[test]
fn server_default_not_merged_when_request_explicit() {
    let mut req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "enable_thinking": true,
    }))
    .expect("valid chat request");
    assert!(req.thinking_explicitly_requested());

    let server_kw = ChatTemplateKwargs {
        enable_thinking: Some(false),
        thinking_budget: None,
    };
    if !req.thinking_explicitly_requested() {
        req.chat_template_kwargs = Some(server_kw);
    }
    assert!(req.chat_template_kwargs.is_none());
    assert!(req.resolve_thinking(false).0);
}
