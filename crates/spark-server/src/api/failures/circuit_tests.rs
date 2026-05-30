// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the pure helpers in `circuit.rs`. Kept in a sibling
//! file (mounted under `#[cfg(test)]` from `failures/mod.rs`) so
//! `circuit.rs` stays under the 500-LoC file-size-cap.

use super::circuit::{
    F39PermanentFailureMatch, f32_reposition_failed_tool_result, f39_build_circuit_breaker_banner,
    f39_class_label, f39_extract_binary_name,
};
use super::classification::F37FailureClass;

fn msg(role: &str, text: &str) -> crate::openai::IncomingMessage {
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

fn assistant_tool_call(id: &str) -> crate::openai::IncomingMessage {
    crate::openai::IncomingMessage {
        role: "assistant".to_string(),
        content: crate::openai::ParsedContent::default(),
        tool_calls: Some(vec![crate::tool_parser::IncomingToolCall {
            id: Some(id.to_string()),
            function: crate::tool_parser::IncomingFunction {
                name: "Bash".to_string(),
                arguments: r#"{"command":"cargo"}"#.to_string(),
            },
        }]),
        tool_call_id: None,
        name: None,
    }
}

fn tool_result(id: &str, text: &str) -> crate::openai::IncomingMessage {
    crate::openai::IncomingMessage {
        role: "tool".to_string(),
        content: crate::openai::ParsedContent {
            text: text.to_string(),
            images: Vec::new(),
        },
        tool_calls: None,
        tool_call_id: Some(id.to_string()),
        name: None,
    }
}

// ── f32_reposition_failed_tool_result ────────────────────────────

#[test]
fn f32_surfaces_error_without_orphan_tool_message() {
    let mut messages = vec![
        msg("user", "do it"),
        assistant_tool_call("toolu_1"),
        tool_result("toolu_1", "[tool error]\ncargo: command not found"),
        msg("user", "intervening 1"),
        msg("user", "intervening 2"),
    ];
    let before = messages.len();

    assert!(f32_reposition_failed_tool_result(&mut messages));

    assert_eq!(messages.len(), before);
    let last = messages.last().unwrap();
    assert_eq!(last.role, "user");
    assert!(last.content.text.contains("<failed_tool_result>"));
    assert!(
        last.content
            .text
            .contains("[tool error]\ncargo: command not found")
    );
}

#[test]
fn f32_no_op_when_failed_tool_is_already_fresh() {
    let mut messages = vec![
        msg("user", "do it"),
        assistant_tool_call("toolu_1"),
        tool_result("toolu_1", "[tool error]\ncargo: command not found"),
    ];
    let before = messages.len();

    assert!(!f32_reposition_failed_tool_result(&mut messages));
    assert_eq!(messages.len(), before);
}

// ── f39_extract_binary_name ───────────────────────────────────────

#[test]
fn binary_name_bash_first_word() {
    assert_eq!(
        f39_extract_binary_name("Bash", "cargo init --name x"),
        Some("cargo".to_string())
    );
}

#[test]
fn binary_name_case_insensitive_tool() {
    // F51: lowercase 'bash' must also be recognised.
    assert_eq!(
        f39_extract_binary_name("bash", "npm install"),
        Some("npm".to_string())
    );
}

#[test]
fn binary_name_non_bash_tool_returns_none() {
    assert_eq!(f39_extract_binary_name("Write", "cargo init"), None);
    assert_eq!(f39_extract_binary_name("Read", "ls"), None);
}

#[test]
fn binary_name_empty_arg_returns_none() {
    // No first whitespace-split word.
    assert_eq!(f39_extract_binary_name("Bash", ""), None);
    assert_eq!(f39_extract_binary_name("Bash", "   "), None);
}

// ── f39_class_label ───────────────────────────────────────────────

#[test]
fn class_label_each_variant() {
    assert_eq!(
        f39_class_label(F37FailureClass::BinaryMissing),
        "binary not installed (command not found / exit 127)"
    );
    assert_eq!(
        f39_class_label(F37FailureClass::AlreadyExists),
        "destination already exists"
    );
    assert_eq!(
        f39_class_label(F37FailureClass::PermissionDenied),
        "permission denied"
    );
    assert_eq!(
        f39_class_label(F37FailureClass::NotFound),
        "path/file not found"
    );
    assert_eq!(
        f39_class_label(F37FailureClass::InvalidArgument),
        "invalid argument or environment-state error"
    );
    assert_eq!(
        f39_class_label(F37FailureClass::StallGuard),
        "Atlas stall-guard refused this call"
    );
}

// ── f39_build_circuit_breaker_banner ──────────────────────────────

#[test]
fn banner_singular_pluralization() {
    let m1 = F39PermanentFailureMatch {
        tool: "Bash".into(),
        primary_arg: "cargo init".into(),
        class: F37FailureClass::BinaryMissing,
        prior_failure_count: 1,
    };
    let body = f39_build_circuit_breaker_banner(&[m1]);
    assert!(
        body.contains("failed 1 time "),
        "singular form expected: {body}"
    );
    assert!(body.contains("<atlas_circuit_breaker>"));
    assert!(body.contains("Bash(cargo init)"));
}

#[test]
fn banner_plural_pluralization() {
    let m = F39PermanentFailureMatch {
        tool: "Bash".into(),
        primary_arg: "npm install".into(),
        class: F37FailureClass::BinaryMissing,
        prior_failure_count: 3,
    };
    let body = f39_build_circuit_breaker_banner(&[m]);
    assert!(
        body.contains("failed 3 times "),
        "plural form expected: {body}"
    );
}

#[test]
fn banner_multiple_matches_listed() {
    let m1 = F39PermanentFailureMatch {
        tool: "Bash".into(),
        primary_arg: "cargo init".into(),
        class: F37FailureClass::BinaryMissing,
        prior_failure_count: 2,
    };
    let m2 = F39PermanentFailureMatch {
        tool: "Write".into(),
        primary_arg: "/etc/passwd".into(),
        class: F37FailureClass::PermissionDenied,
        prior_failure_count: 1,
    };
    let body = f39_build_circuit_breaker_banner(&[m1, m2]);
    assert!(body.contains("Bash(cargo init)"));
    assert!(body.contains("Write(/etc/passwd)"));
    assert!(body.contains("permission denied"));
    assert!(body.contains("binary not installed"));
}
