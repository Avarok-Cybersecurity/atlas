// SPDX-License-Identifier: AGPL-3.0-only

//! Shared helpers used by sibling test modules. Hoisted from the
//! original `api.rs::sanitizer_tests` module so the F28-F32 and F49
//! tests can both reference `mk_msg`/`mk_tool_msg`/`mk_assistant_with_tool_call`
//! without duplicating bodies.

#![allow(dead_code)]

pub(super) fn mk_msg(role: &str, text: &str) -> crate::openai::IncomingMessage {
    crate::openai::IncomingMessage {
        role: role.to_string(),
        content: crate::openai::ParsedContent {
            text: text.to_string(),
            images: Vec::new(),
        },
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}

pub(super) fn mk_tool_msg(call_id: &str, text: &str) -> crate::openai::IncomingMessage {
    crate::openai::IncomingMessage {
        role: "tool".to_string(),
        content: crate::openai::ParsedContent {
            text: text.to_string(),
            images: Vec::new(),
        },
        tool_calls: None,
        tool_call_id: Some(call_id.to_string()),
        name: None,
    }
}

pub(super) fn mk_assistant_with_tool_call(
    id: &str,
    name: &str,
    args: &str,
) -> crate::openai::IncomingMessage {
    crate::openai::IncomingMessage {
        role: "assistant".to_string(),
        content: crate::openai::ParsedContent::default(),
        tool_calls: Some(vec![crate::tool_parser::IncomingToolCall {
            id: Some(id.to_string()),
            function: crate::tool_parser::IncomingFunction {
                name: name.to_string(),
                arguments: args.to_string(),
            },
        }]),
        tool_call_id: None,
        name: None,
    }
}
