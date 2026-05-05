// SPDX-License-Identifier: AGPL-3.0-only

//! F28-F32 / F37 / F39 / F53 / F40 tests (hoisted from
//! `api.rs::sanitizer_tests`, lines 8361-8798 of the pre-split file).
//! Shared `mk_msg`/`mk_tool_msg`/`mk_assistant_with_tool_call` helpers
//! live in `super::common`.

#![cfg(test)]

mod f28_f32_tests {
    use super::super::super::*;
    use super::super::common::{
    mk_assistant_with_tool_call,
    mk_msg,
    mk_tool_msg,
};

    // ── F28-F32 tests (tool-error-forgetfulness fixes) ──
    // (mk_msg / mk_tool_msg / mk_assistant_with_tool_call hoisted to
    //  super::common to share with the f49 test file.)

    #[test]
    fn f28_recent_message_is_tool_error_detects_f6_prefix() {
        let msgs = vec![
            mk_msg("user", "do x"),
            mk_msg("assistant", "calling tool"),
            mk_tool_msg("c1", "[tool error]\ncargo: command not found"),
        ];
        assert!(recent_message_is_tool_error(&msgs));
    }

    #[test]
    fn f28_recent_message_is_tool_error_negates_on_success() {
        let msgs = vec![
            mk_msg("user", "do x"),
            mk_msg("assistant", "calling tool"),
            mk_tool_msg("c1", "ok\nfile written"),
        ];
        assert!(!recent_message_is_tool_error(&msgs));
    }

    #[test]
    fn f28_recent_message_is_tool_error_finds_last_tool_skipping_user() {
        // Reminder injection may leave a user/system message after the tool
        // result; F28 should still find the most-recent role:tool.
        let msgs = vec![
            mk_msg("user", "do x"),
            mk_msg("assistant", "calling tool"),
            mk_tool_msg("c1", "[tool error]\nExit code 127"),
        ];
        assert!(recent_message_is_tool_error(&msgs));
    }

    #[test]
    fn f29_extract_binary_basic_command_not_found() {
        assert_eq!(
            f29_extract_binary_from_error_line("cargo: command not found"),
            Some("cargo".to_string())
        );
    }

    #[test]
    fn f29_extract_binary_bash_path_prefix() {
        assert_eq!(
            f29_extract_binary_from_error_line("/bin/bash: line 1: cargo: command not found"),
            Some("cargo".to_string())
        );
    }

    #[test]
    fn f29_extract_binary_rejects_non_alphanum() {
        // A "binary" with slashes is a path, not a binary name.
        assert_eq!(
            f29_extract_binary_from_error_line("/usr/local/cargo: command not found"),
            None
        );
    }

    #[test]
    fn f29_extract_facts_filters_below_threshold() {
        // F36 (2026-04-26): threshold is now 2; one occurrence should
        // NOT yield a fact (transient, could be a typo or path issue).
        let msgs = vec![
            mk_tool_msg("c1", "[tool error]\ncargo: command not found"),
        ];
        let facts = f29_extract_environment_facts(&msgs);
        assert!(facts.is_empty(), "got {facts:?}");
    }

    #[test]
    fn f29_extract_facts_two_occurrences_yields_fact() {
        // F36: two same-binary failures = permanent. Yields fact.
        let msgs = vec![
            mk_tool_msg("c1", "[tool error]\ncargo: command not found"),
            mk_tool_msg("c2", "[tool error]\ncargo: command not found"),
        ];
        let facts = f29_extract_environment_facts(&msgs);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].binary, "cargo");
        assert_eq!(facts[0].observed_count, 2);
    }

    #[test]
    fn f29_inject_creates_system_message_when_absent() {
        let mut msgs = vec![mk_msg("user", "do x")];
        let facts = vec![F29EnvironmentFact {
            binary: "cargo".to_string(),
            observed_count: 4,
        }];
        f29_inject_environment_facts(&mut msgs, &facts);
        assert_eq!(msgs[0].role, "system");
        assert!(msgs[0].content.text.contains("<environment_facts>"));
        assert!(msgs[0].content.text.contains("cargo"));
    }

    #[test]
    fn f29_inject_replaces_existing_block_idempotently() {
        let mut msgs = vec![mk_msg("system", "stay helpful"), mk_msg("user", "do x")];
        let facts1 = vec![F29EnvironmentFact {
            binary: "cargo".to_string(),
            observed_count: 3,
        }];
        f29_inject_environment_facts(&mut msgs, &facts1);
        let facts2 = vec![F29EnvironmentFact {
            binary: "cargo".to_string(),
            observed_count: 5,
        }];
        f29_inject_environment_facts(&mut msgs, &facts2);
        // Should NOT have two blocks — it's replaced.
        let count = msgs[0].content.text.matches("<environment_facts>").count();
        assert_eq!(count, 1, "expected exactly one block, got {count}");
        assert!(msgs[0].content.text.contains("5 times"));
    }

    #[test]
    fn f30_prepend_creates_system_when_absent() {
        let mut msgs = vec![mk_msg("user", "hi")];
        prepend_reminder_to_system(&mut msgs, "stop retrying");
        assert_eq!(msgs[0].role, "system");
        assert!(msgs[0].content.text.contains("<atlas_runtime_notice>"));
        assert!(msgs[0].content.text.contains("stop retrying"));
    }

    #[test]
    fn f30_prepend_to_existing_system_replaces_prior_notice() {
        let mut msgs = vec![mk_msg("system", "you are helpful"), mk_msg("user", "hi")];
        prepend_reminder_to_system(&mut msgs, "first reminder");
        prepend_reminder_to_system(&mut msgs, "second reminder");
        let count = msgs[0].content.text.matches("<atlas_runtime_notice>").count();
        assert_eq!(count, 1);
        assert!(msgs[0].content.text.contains("second reminder"));
        assert!(!msgs[0].content.text.contains("first reminder"));
        assert!(msgs[0].content.text.contains("you are helpful"));
    }

    #[test]
    fn f31_no_anchor_no_inject() {
        let mut msgs = vec![mk_msg("user", "hi")];
        let metrics = F23ProgressMetrics { score: -3, attempts: 10 };
        assert!(!f31_inject_hard_refusal(&mut msgs, metrics));
    }

    #[test]
    fn f31_below_refuse_no_inject() {
        let mut msgs = vec![
            mk_msg("user", "hi"),
            mk_assistant_with_tool_call("toolu_1", "Bash", "{\"command\":\"x\"}"),
            mk_tool_msg("toolu_1", "[tool error]\ncargo: command not found"),
        ];
        let metrics = F23ProgressMetrics { score: -1, attempts: 3 };
        assert!(!f31_inject_hard_refusal(&mut msgs, metrics));
    }

    #[test]
    fn f31_at_refuse_injects_synth_tool_result() {
        let mut msgs = vec![
            mk_msg("user", "hi"),
            mk_assistant_with_tool_call("toolu_1", "Bash", "{\"command\":\"x\"}"),
            mk_tool_msg("toolu_1", "[tool error]\ncargo: command not found"),
        ];
        let metrics = F23ProgressMetrics { score: -3, attempts: 10 };
        assert!(f31_inject_hard_refusal(&mut msgs, metrics));
        let last = msgs.last().unwrap();
        assert_eq!(last.role, "tool");
        assert_eq!(last.tool_call_id.as_deref(), Some("toolu_1"));
        assert!(last.content.text.contains("[atlas-stall-guard]"));
    }

    #[test]
    fn f31_idempotent_no_double_inject() {
        let mut msgs = vec![
            mk_msg("user", "hi"),
            mk_assistant_with_tool_call("toolu_1", "Bash", "{\"command\":\"x\"}"),
            mk_tool_msg("toolu_1", "[tool error]\ncargo: command not found"),
        ];
        let metrics = F23ProgressMetrics { score: -3, attempts: 10 };
        assert!(f31_inject_hard_refusal(&mut msgs, metrics));
        assert!(!f31_inject_hard_refusal(&mut msgs, metrics));
        let count = msgs
            .iter()
            .filter(|m| {
                m.role == "tool"
                    && m.content.text.starts_with("[tool error]\n[atlas-stall-guard]")
            })
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn f32_no_op_when_failed_tool_at_tail() {
        // Failed tool result IS the last message — gap < 2, no
        // duplication.
        let mut msgs = vec![
            mk_msg("user", "hi"),
            mk_assistant_with_tool_call("toolu_1", "Bash", "{\"command\":\"x\"}"),
            mk_tool_msg("toolu_1", "[tool error]\ncargo: command not found"),
        ];
        let before = msgs.len();
        assert!(!f32_reposition_failed_tool_result(&mut msgs));
        assert_eq!(msgs.len(), before);
    }

    #[test]
    fn f32_duplicates_when_messages_after_failed_tool() {
        let mut msgs = vec![
            mk_msg("user", "hi"),
            mk_assistant_with_tool_call("toolu_1", "Bash", "{\"command\":\"x\"}"),
            mk_tool_msg("toolu_1", "[tool error]\ncargo: command not found"),
            mk_msg("user", "intervening 1"),
            mk_msg("user", "intervening 2"),
        ];
        assert!(f32_reposition_failed_tool_result(&mut msgs));
        let last = msgs.last().unwrap();
        assert_eq!(last.role, "tool");
        assert!(last.content.text.starts_with("[tool error]"));
    }

    #[test]
    fn f37_classify_binary_missing() {
        assert_eq!(
            f37_classify_failure("[tool error]\ncargo: command not found"),
            Some(F37FailureClass::BinaryMissing)
        );
        assert_eq!(
            f37_classify_failure("Exit code 127"),
            Some(F37FailureClass::BinaryMissing)
        );
    }

    #[test]
    fn f37_classify_already_exists() {
        assert_eq!(
            f37_classify_failure("error: destination /tmp/x already exists"),
            Some(F37FailureClass::AlreadyExists)
        );
        assert_eq!(
            f37_classify_failure("error: cargo init cannot be run on existing Cargo packages"),
            Some(F37FailureClass::AlreadyExists)
        );
    }

    #[test]
    fn f37_classify_invalid_arg() {
        assert_eq!(
            f37_classify_failure("Error: You must read file /tmp/f.toml before overwriting it. Use the Read tool first"),
            Some(F37FailureClass::InvalidArgument)
        );
        assert_eq!(
            f37_classify_failure("TypeError [ERR_INVALID_ARG_VALUE]: ..."),
            Some(F37FailureClass::InvalidArgument)
        );
    }

    #[test]
    fn f37_classify_returns_none_for_success() {
        assert!(f37_classify_failure("File written successfully").is_none());
    }

    #[test]
    fn f39_no_history_no_match() {
        let msgs = vec![mk_msg("user", "do x")];
        assert!(f39_detect_recent_retries(&msgs).is_empty());
    }

    #[test]
    fn f39_first_call_no_match() {
        // Single tool_call, no prior failure → no match.
        let msgs = vec![
            mk_msg("user", "do x"),
            mk_assistant_with_tool_call("c1", "Bash", "{\"command\":\"cargo init\"}"),
        ];
        assert!(f39_detect_recent_retries(&msgs).is_empty());
    }

    #[test]
    fn f39_retry_after_command_not_found_matches() {
        // 1st cargo failed → 2nd cargo emitted in last assistant turn.
        let msgs = vec![
            mk_msg("user", "do x"),
            mk_assistant_with_tool_call("c1", "Bash", "{\"command\":\"cargo init\"}"),
            mk_tool_msg("c1", "[tool error]\ncargo: command not found"),
            mk_assistant_with_tool_call("c2", "Bash", "{\"command\":\"cargo init --offline\"}"),
        ];
        let m = f39_detect_recent_retries(&msgs);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].tool, "Bash");
        assert_eq!(m[0].class, F37FailureClass::BinaryMissing);
        assert_eq!(m[0].prior_failure_count, 1);
    }

    #[test]
    fn f39_extract_bash_final_action_buckets_cargo_variants() {
        // F21's bucketing should make `cargo init`, `cd /tmp && cargo init`,
        // and `mkdir -p /tmp/x && cd /tmp/x && cargo init` all collapse
        // to the same primary_arg, so F39 catches all three as the same
        // failing pattern.
        let msgs = vec![
            mk_msg("user", "do x"),
            mk_assistant_with_tool_call("c1", "Bash", "{\"command\":\"cargo init\"}"),
            mk_tool_msg("c1", "[tool error]\ncargo: command not found"),
            mk_assistant_with_tool_call(
                "c2",
                "Bash",
                "{\"command\":\"cd /tmp && rm -rf x && mkdir -p x && cd x && cargo init\"}",
            ),
        ];
        let m = f39_detect_recent_retries(&msgs);
        assert_eq!(m.len(), 1, "F21 buckets must collapse cargo variants");
    }

    #[test]
    fn f39_banner_contains_class_and_count() {
        let matches = vec![F39PermanentFailureMatch {
            tool: "Bash".into(),
            primary_arg: "cargo init".into(),
            class: F37FailureClass::BinaryMissing,
            prior_failure_count: 2,
        }];
        let b = f39_build_circuit_breaker_banner(&matches);
        assert!(b.contains("<atlas_circuit_breaker>"));
        assert!(b.contains("Bash(cargo init)"));
        assert!(b.contains("binary not installed"));
        assert!(b.contains("2 times"));
    }

    #[test]
    fn f53_mkdir_loop_scores_negative() {
        // F53 (2026-04-27): same Bash mkdir 4 times with empty
        // result must score negative. F40 collision penalty alone
        // gives -3 (3 collisions); F53 adds -2 from the 3rd
        // occurrence onward → total -5 (vs prior 0).
        let cmd = "{\"command\":\"mkdir -p /tmp/x/src\"}";
        let msgs = vec![
            mk_msg("user", "do x"),
            mk_assistant_with_tool_call("c1", "Bash", cmd),
            mk_tool_msg("c1", ""),
            mk_assistant_with_tool_call("c2", "Bash", cmd),
            mk_tool_msg("c2", ""),
            mk_assistant_with_tool_call("c3", "Bash", cmd),
            mk_tool_msg("c3", ""),
            mk_assistant_with_tool_call("c4", "Bash", cmd),
            mk_tool_msg("c4", ""),
        ];
        let m = f23_score_progress(&msgs);
        assert_eq!(m.attempts, 4);
        assert!(
            m.score <= -5,
            "F53: 4 identical mkdir with empty results must score <= -5; got {}",
            m.score
        );
    }

    #[test]
    fn f40_three_writes_to_same_path_score_negative() {
        // 3 Cargo.toml writes (each succeeds with novel content) →
        // F23 should now score:
        //   call 1: novel result +1, call collision 0 → +1
        //   call 2: novel result +1, call collision -1 → 0
        //   call 3: novel result +1, call collision -1 → 0
        // total = +1 (vs prior +3 without F40)
        let cargo_args = "{\"file_path\":\"/tmp/Cargo.toml\",\"content\":\"a\"}";
        let cargo_args2 = "{\"file_path\":\"/tmp/Cargo.toml\",\"content\":\"b\"}";
        let cargo_args3 = "{\"file_path\":\"/tmp/Cargo.toml\",\"content\":\"c\"}";
        let msgs = vec![
            mk_msg("user", "do x"),
            mk_assistant_with_tool_call("c1", "Write", cargo_args),
            mk_tool_msg("c1", "wrote file ok at /tmp/Cargo.toml v1"),
            mk_assistant_with_tool_call("c2", "Write", cargo_args2),
            mk_tool_msg("c2", "wrote file ok at /tmp/Cargo.toml v2"),
            mk_assistant_with_tool_call("c3", "Write", cargo_args3),
            mk_tool_msg("c3", "wrote file ok at /tmp/Cargo.toml v3"),
        ];
        let m = f23_score_progress(&msgs);
        assert_eq!(m.attempts, 3);
        assert_eq!(m.score, 1, "F40 must offset +3 novel by -2 collision");
    }

    #[test]
    fn f32_skip_when_tail_is_stall_guard() {
        // F31 already put a stall-guard at the tail; F32 should
        // not also duplicate (avoids piling on).
        let mut msgs = vec![
            mk_msg("user", "hi"),
            mk_assistant_with_tool_call("toolu_1", "Bash", "{\"command\":\"x\"}"),
            mk_tool_msg("toolu_1", "[tool error]\ncargo: command not found"),
            mk_msg("user", "intervening"),
            mk_tool_msg(
                "toolu_1",
                "[tool error]\n[atlas-stall-guard] retried too many times",
            ),
        ];
        let before = msgs.len();
        assert!(!f32_reposition_failed_tool_result(&mut msgs));
        assert_eq!(msgs.len(), before);
    }
}
