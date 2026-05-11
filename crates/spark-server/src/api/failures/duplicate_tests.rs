// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the pure helpers in `duplicate.rs`. Sibling file
//! mounted under `#[cfg(test)]` from `failures/mod.rs`.

use super::duplicate::{
    F49DuplicateWrite, f49_build_banner, f49_extract_write_path_and_content,
    strip_xml_leaks_from_assistant_content,
};
use crate::tool_parser::{FunctionDefinition, ToolDefinition};

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: name.to_string(),
            description: None,
            parameters: None,
        },
    }
}

// ── f49_extract_write_path_and_content ────────────────────────────

#[test]
fn write_extract_path_and_content_hash() {
    // Standard Write: file_path + content → Some((path, hash)).
    let out = f49_extract_write_path_and_content(
        "Write",
        r#"{"file_path":"/tmp/x.rs","content":"fn main(){}"}"#,
    );
    let (path, _hash) = out.expect("Write must extract");
    assert_eq!(path, "/tmp/x.rs");
}

#[test]
fn write_extract_camelcase_path() {
    // opencode uses filePath instead of file_path.
    let out =
        f49_extract_write_path_and_content("write", r#"{"filePath":"/tmp/y.rs","content":"x"}"#);
    let (path, _hash) = out.expect("camelCase must work");
    assert_eq!(path, "/tmp/y.rs");
}

#[test]
fn write_same_content_same_hash() {
    // Hash is deterministic — same content twice → same hash.
    let a =
        f49_extract_write_path_and_content("Write", r#"{"file_path":"/tmp/x.rs","content":"abc"}"#);
    let b =
        f49_extract_write_path_and_content("Write", r#"{"file_path":"/tmp/x.rs","content":"abc"}"#);
    assert_eq!(a, b);
}

#[test]
fn write_different_content_different_hash() {
    let a =
        f49_extract_write_path_and_content("Write", r#"{"file_path":"/tmp/x.rs","content":"abc"}"#);
    let b =
        f49_extract_write_path_and_content("Write", r#"{"file_path":"/tmp/x.rs","content":"xyz"}"#);
    assert_ne!(a.unwrap().1, b.unwrap().1);
}

#[test]
fn edit_extracts_old_new_pair() {
    let out = f49_extract_write_path_and_content(
        "Edit",
        r#"{"file_path":"/tmp/x.rs","oldString":"a","newString":"b"}"#,
    );
    assert!(out.is_some());
    let (path, _hash) = out.unwrap();
    assert_eq!(path, "/tmp/x.rs");
}

#[test]
fn edit_missing_old_new_returns_none() {
    let out = f49_extract_write_path_and_content(
        "Edit",
        r#"{"file_path":"/tmp/x.rs","content":"only-write-key"}"#,
    );
    assert!(out.is_none());
}

#[test]
fn extract_non_write_tool_returns_none() {
    assert!(f49_extract_write_path_and_content("Bash", r#"{"command":"cargo build"}"#,).is_none());
    assert!(f49_extract_write_path_and_content("Read", r#"{"file_path":"/tmp/x.rs"}"#,).is_none());
}

#[test]
fn extract_malformed_json_returns_none() {
    assert!(f49_extract_write_path_and_content("Write", "not json").is_none());
}

// ── f49_build_banner ──────────────────────────────────────────────

#[test]
fn banner_singular_plural() {
    let one = F49DuplicateWrite {
        file_path: "/a".into(),
        prior_count: 1,
    };
    let body = f49_build_banner(&[one]);
    assert!(body.contains("written 1 time "), "singular: {body}");

    let two = F49DuplicateWrite {
        file_path: "/b".into(),
        prior_count: 3,
    };
    let body = f49_build_banner(&[two]);
    assert!(body.contains("written 3 times "), "plural: {body}");
}

#[test]
fn banner_multiple_files_listed() {
    let hits = vec![
        F49DuplicateWrite {
            file_path: "/a".into(),
            prior_count: 1,
        },
        F49DuplicateWrite {
            file_path: "/b".into(),
            prior_count: 2,
        },
    ];
    let body = f49_build_banner(&hits);
    assert!(body.contains("/a"));
    assert!(body.contains("/b"));
    assert!(body.contains("<atlas_duplicate_write>"));
}

// ── strip_xml_leaks_from_assistant_content ────────────────────────

#[test]
fn strip_empty_content_returns_empty() {
    let out = strip_xml_leaks_from_assistant_content("", &[]);
    assert_eq!(out, "");
}

#[test]
fn strip_no_leak_preserves_prose() {
    let tools = vec![tool("write")];
    let out = strip_xml_leaks_from_assistant_content("just prose here", &tools);
    assert_eq!(out, "just prose here");
}

#[test]
fn strip_removes_declared_tool_xml() {
    let tools = vec![tool("read")];
    let out = strip_xml_leaks_from_assistant_content(
        "before <read><filePath>/x</filePath></read> after",
        &tools,
    );
    assert!(out.contains("before"));
    assert!(out.contains("after"));
    assert!(!out.contains("<read>"));
    assert!(!out.contains("filePath"));
}

#[test]
fn strip_removes_harness_tag_even_without_tool_def() {
    // `<task>` is a hardcoded harness tag.
    let out = strip_xml_leaks_from_assistant_content("a <task>b</task> c", &[]);
    assert!(!out.contains("<task>"));
    assert!(out.contains("a"));
    assert!(out.contains("c"));
}

#[test]
fn strip_short_tool_name_does_not_match() {
    // Tool names shorter than 3 chars are skipped to avoid prose
    // false positives.
    let tools = vec![tool("rm")];
    let out = strip_xml_leaks_from_assistant_content("a <rm>x</rm> b", &tools);
    // `<rm>` is 2 chars → not matched, content preserved.
    assert!(out.contains("<rm>"));
}

#[test]
fn strip_case_insensitive() {
    let tools = vec![tool("Write")];
    let out = strip_xml_leaks_from_assistant_content("<WRITE>body</WRITE>", &tools);
    assert!(out.trim().is_empty());
}
