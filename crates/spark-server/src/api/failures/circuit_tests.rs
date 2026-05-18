// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the pure helpers in `circuit.rs`. Kept in a sibling
//! file (mounted under `#[cfg(test)]` from `failures/mod.rs`) so
//! `circuit.rs` stays under the 500-LoC file-size-cap.

use super::circuit::{
    F39PermanentFailureMatch, f39_build_circuit_breaker_banner, f39_class_label,
    f39_extract_binary_name,
};
use super::classification::F37FailureClass;

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
