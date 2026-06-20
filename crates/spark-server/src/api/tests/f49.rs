// SPDX-License-Identifier: AGPL-3.0-only

//! F49 / F50 / F44 (fix35) tests (hoisted from
//! `api.rs::sanitizer_tests`, lines 8800-8978 of the pre-split file).
//! Shared `mk_msg`/`mk_tool_msg`/`mk_assistant_with_tool_call` helpers
//! live in `super::common`.

#![cfg(test)]

mod f49_tests {
    use super::super::super::*;
    use super::super::common::{mk_assistant_with_tool_call, mk_msg, mk_tool_msg};

    // ── F49 / F50 / F44 (fix35) tests ──

    #[test]
    fn f49_single_write_no_trip() {
        let msgs = vec![
            mk_msg("user", "do x"),
            mk_assistant_with_tool_call(
                "c1",
                "Write",
                "{\"file_path\":\"/tmp/Cargo.toml\",\"content\":\"a\"}",
            ),
        ];
        assert!(f49_detect_duplicate_writes(&msgs).is_empty());
    }

    #[test]
    fn f49_two_identical_writes_trip() {
        let args = "{\"file_path\":\"/tmp/Cargo.toml\",\"content\":\"[package]\\nname=\\\"x\\\"\"}";
        let msgs = vec![
            mk_msg("user", "do x"),
            mk_assistant_with_tool_call("c1", "Write", args),
            mk_tool_msg("c1", "Wrote file successfully."),
            mk_assistant_with_tool_call("c2", "Write", args),
        ];
        let hits = f49_detect_duplicate_writes(&msgs);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].file_path, "/tmp/Cargo.toml");
        assert_eq!(hits[0].prior_count, 1);
    }

    #[test]
    fn f51_f49_matches_lowercase_write_for_opencode() {
        // F51 (2026-04-27): regression pin. opencode (OpenAI direct)
        // emits tool name "write" (lowercase); Atlas's F49 must
        // catch this just like the Anthropic-style "Write".
        // opencode uses `filePath` (camelCase) in args.
        let args = "{\"filePath\":\"/tmp/Cargo.toml\",\"content\":\"[package]\\nname=\\\"x\\\"\"}";
        let msgs = vec![
            mk_msg("user", "do x"),
            mk_assistant_with_tool_call("c1", "write", args),
            mk_tool_msg("c1", "Wrote file successfully."),
            mk_assistant_with_tool_call("c2", "write", args),
        ];
        let hits = f49_detect_duplicate_writes(&msgs);
        assert_eq!(hits.len(), 1, "F51: lowercase `write` must trip F49");
        assert_eq!(hits[0].file_path, "/tmp/Cargo.toml");
    }

    #[test]
    fn f51_primary_arg_handles_lowercase_bash() {
        // opencode bash tool name with `command` arg — must extract
        // the same primary_arg as Claude Code's "Bash".
        let p_lower = primary_arg_for_tool("bash", "{\"command\":\"cd /tmp && cargo init\"}");
        let p_upper = primary_arg_for_tool("Bash", "{\"command\":\"cd /tmp && cargo init\"}");
        assert_eq!(p_lower, p_upper);
        assert_eq!(p_lower.as_deref(), Some("cargo init"));
    }

    #[test]
    fn f51_f39_binary_name_handles_lowercase() {
        assert_eq!(
            f39_extract_binary_name("bash", "cargo init"),
            Some("cargo".to_string())
        );
        assert_eq!(
            f39_extract_binary_name("Bash", "cargo init"),
            Some("cargo".to_string())
        );
        assert_eq!(f39_extract_binary_name("Write", "cargo init"), None);
    }

    #[test]
    fn f49_different_content_same_path_no_trip() {
        let a = "{\"file_path\":\"/tmp/Cargo.toml\",\"content\":\"v1\"}";
        let b = "{\"file_path\":\"/tmp/Cargo.toml\",\"content\":\"v2\"}";
        let msgs = vec![
            mk_msg("user", "do x"),
            mk_assistant_with_tool_call("c1", "Write", a),
            mk_tool_msg("c1", "ok"),
            mk_assistant_with_tool_call("c2", "Write", b),
        ];
        assert!(f49_detect_duplicate_writes(&msgs).is_empty());
    }

    #[test]
    fn f49_same_content_different_path_no_trip() {
        let a = "{\"file_path\":\"/tmp/A.toml\",\"content\":\"x\"}";
        let b = "{\"file_path\":\"/tmp/B.toml\",\"content\":\"x\"}";
        let msgs = vec![
            mk_msg("user", "do x"),
            mk_assistant_with_tool_call("c1", "Write", a),
            mk_tool_msg("c1", "ok"),
            mk_assistant_with_tool_call("c2", "Write", b),
        ];
        assert!(f49_detect_duplicate_writes(&msgs).is_empty());
    }

    #[test]
    fn f50_appends_when_original_error_present() {
        let args = "{\"file_path\":\"/tmp/Cargo.toml\",\"content\":\"x\"}";
        let mut msgs = vec![
            mk_msg("user", "run tests"),
            mk_assistant_with_tool_call("c0", "Bash", "{\"command\":\"cargo test\"}"),
            mk_tool_msg(
                "c0",
                "[tool error]\nerror: couldn't read src/main.rs: No such file or directory",
            ),
            mk_assistant_with_tool_call("c1", "Write", args),
            mk_tool_msg("c1", "Wrote file successfully."),
            mk_assistant_with_tool_call("c2", "Write", args),
        ];
        let before = msgs.len();
        assert!(f50_append_original_error(&mut msgs));
        assert_eq!(msgs.len(), before + 1);
        let last = msgs.last().unwrap();
        assert_eq!(last.role, "tool");
        assert!(last.content.text.starts_with("[atlas-original-error]"));
        assert!(last.content.text.contains("src/main.rs"));
    }

    #[test]
    fn f50_idempotent_on_second_call() {
        let args = "{\"file_path\":\"/tmp/Cargo.toml\",\"content\":\"x\"}";
        let mut msgs = vec![
            mk_msg("user", "run tests"),
            mk_assistant_with_tool_call("c0", "Bash", "{\"command\":\"cargo test\"}"),
            mk_tool_msg("c0", "[tool error]\nerror: couldn't read src/main.rs"),
            mk_assistant_with_tool_call("c1", "Write", args),
            mk_tool_msg("c1", "ok"),
            mk_assistant_with_tool_call("c2", "Write", args),
        ];
        assert!(f50_append_original_error(&mut msgs));
        assert!(!f50_append_original_error(&mut msgs));
    }

    #[test]
    fn f44_check_direct_match() {
        let mut cache = F39FailureCache::default();
        cache.direct.insert(
            ("Bash".into(), "cargo init".into()),
            (1, F37FailureClass::BinaryMissing),
        );
        assert!(f44_check_permanent_failure(
            &cache,
            "Bash",
            "{\"command\":\"cargo init\"}",
        ));
    }

    #[test]
    fn f44_check_first_word_binary_fallback() {
        let mut cache = F39FailureCache::default();
        let mut bins = std::collections::HashMap::new();
        bins.insert("cargo".to_string(), 1u32);
        cache.missing_bins_by_tool.insert("Bash".to_string(), bins);
        // Different primary_arg from the original failure, but
        // same binary first-word — must trip via fallback.
        assert!(f44_check_permanent_failure(
            &cache,
            "Bash",
            "{\"command\":\"cargo init --offline --name x\"}",
        ));
    }

    #[test]
    fn f44_no_match_for_different_binary() {
        let mut cache = F39FailureCache::default();
        let mut bins = std::collections::HashMap::new();
        bins.insert("cargo".to_string(), 1u32);
        cache.missing_bins_by_tool.insert("Bash".to_string(), bins);
        assert!(!f44_check_permanent_failure(
            &cache,
            "Bash",
            "{\"command\":\"npm install\"}",
        ));
    }
}
