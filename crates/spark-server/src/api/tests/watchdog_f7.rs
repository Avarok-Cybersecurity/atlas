// SPDX-License-Identifier: AGPL-3.0-only

//! Loop-watchdog and F7 (cross-turn tool-arg-path stall guard) tests
//! (hoisted from `api.rs::sanitizer_tests`, lines 7969-8359 of the
//! pre-split file). The body is wrapped in an inner module so existing
//! `super::` paths (referring to the original `api` parent) keep
//! resolving via `super::super::super::*`.

#![cfg(test)]

mod watchdog_f7_tests {
    use super::super::super::*;

    // ── check_loop_watchdog substring-fuzzy fix tests ──
    //
    // The 4× repeated phrase in claude-export.txt's tail rolled out
    // of the prior 3 KB window because each instance was preceded by
    // ~3 KB of source-dump prose. The bumped 8 KB window + substring
    // fallback must catch it.

    #[test]
    fn watchdog_catches_phrase_repeats_across_large_interstitials() {
        let phrase = "I'll create the project files and verify everything works:";
        // Five copies of phrase + ~1500 chars of unique filler each =
        // ~7.5 KB total — fits in the 8 KB window but would have
        // rolled out of the previous 3 KB.
        let filler: String = (0..1500)
            .map(|i| (b'a' + ((i % 26) as u8)) as char)
            .collect();
        let mut feed = String::new();
        for _ in 0..5 {
            feed.push_str(phrase);
            feed.push('\n');
            feed.push_str(&filler);
            feed.push('\n');
        }
        let mut scan_buf = String::new();
        // Feed in 64-byte chunks to mimic streaming.
        let mut triggered = false;
        for chunk in feed.as_bytes().chunks(64) {
            let s = std::str::from_utf8(chunk).unwrap();
            if check_loop_watchdog(s, &mut scan_buf, false) {
                triggered = true;
                break;
            }
        }
        assert!(
            triggered,
            "watchdog must fire on 5× repeated phrase across large interstitials"
        );
    }

    #[test]
    fn watchdog_substring_catches_mid_line_repeat() {
        // Mirrors export.txt line 919 — the 4th instance was attached
        // to other text ("…everything works:        let body =").
        let phrase = "I'll create the project files and verify everything works:";
        let mut buf = String::new();
        // 3 clean repeats then a mid-line continuation.
        for _ in 0..3 {
            buf.push_str(phrase);
            buf.push('\n');
            buf.push_str("intermediate prose\n");
        }
        buf.push_str(phrase);
        buf.push_str("        let body = vec![];");
        let mut scan = String::new();
        let mut triggered = false;
        for chunk in buf.as_bytes().chunks(48) {
            let s = std::str::from_utf8(chunk).unwrap();
            if check_loop_watchdog(s, &mut scan, false) {
                triggered = true;
                break;
            }
        }
        assert!(
            triggered,
            "substring scan must catch repeated phrase even when one \
             instance is mid-line"
        );
    }

    // ── F7 (2026-04-26): cross-turn tool-arg-path stall guard tests ──

    fn make_assistant_tool_call(name: &str, args_json: &str) -> crate::openai::IncomingMessage {
        use crate::openai::{IncomingMessage, ParsedContent};
        use crate::tool_parser::{IncomingFunction, IncomingToolCall};
        IncomingMessage {
            role: "assistant".to_string(),
            content: ParsedContent::default(),
            tool_calls: Some(vec![IncomingToolCall {
                id: Some(format!("toolu_{}", name)),
                function: IncomingFunction {
                    name: name.to_string(),
                    arguments: args_json.to_string(),
                },
            }]),
            tool_call_id: None,
            name: None,
        }
    }

    fn make_user(text: &str) -> crate::openai::IncomingMessage {
        use crate::openai::{IncomingMessage, ParsedContent};
        IncomingMessage {
            role: "user".to_string(),
            content: ParsedContent {
                text: text.to_string(),
                images: Vec::new(),
            },
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    #[test]
    fn f7_four_writes_to_same_path_warns() {
        // F14 raised warn threshold from 3 → 4. Test updated to
        // match. Four same-path Writes now trips the warn-level
        // reminder (without escalating to refuse).
        let mut msgs = vec![
            make_user("please write Cargo.toml"),
            make_assistant_tool_call(
                "Write",
                r#"{"file_path":"/tmp/x/Cargo.toml","content":"v1"}"#,
            ),
            make_user("retry"),
            make_assistant_tool_call(
                "Write",
                r#"{"file_path":"/tmp/x/Cargo.toml","content":"v2"}"#,
            ),
            make_user("again"),
            make_assistant_tool_call(
                "Write",
                r#"{"file_path":"/tmp/x/Cargo.toml","content":"v3"}"#,
            ),
            make_user("once more"),
            make_assistant_tool_call(
                "Write",
                r#"{"file_path":"/tmp/x/Cargo.toml","content":"v4"}"#,
            ),
            make_user("now what?"),
        ];
        let buckets = collect_f7_stall_buckets(&msgs);
        assert_eq!(
            buckets
                .get(&("Write".to_string(), "/tmp/x/Cargo.toml".to_string()))
                .copied(),
            Some(4)
        );
        let reminder = build_f7_stall_reminder(&buckets).unwrap();
        assert!(reminder.contains("Write(/tmp/x/Cargo.toml) × 4"));
        // Threshold 4 warning, not yet refuse-level (5).
        assert!(reminder.contains("not making progress"));
        assert!(!reminder.contains("Do NOT call any tool"));
        append_f7_reminder_to_last_user(&mut msgs, &reminder);
        let last_user = msgs.iter().rev().find(|m| m.role == "user").unwrap();
        assert!(
            last_user
                .content
                .text
                .contains("Write(/tmp/x/Cargo.toml) × 4")
        );
        assert!(last_user.content.text.starts_with("now what?"));
    }

    #[test]
    fn f7_three_writes_to_different_paths_does_not_warn() {
        let msgs = vec![
            make_assistant_tool_call("Write", r#"{"file_path":"/a","content":"x"}"#),
            make_assistant_tool_call("Write", r#"{"file_path":"/b","content":"x"}"#),
            make_assistant_tool_call("Write", r#"{"file_path":"/c","content":"x"}"#),
        ];
        let buckets = collect_f7_stall_buckets(&msgs);
        assert_eq!(buckets.values().max().copied(), Some(1));
        assert!(build_f7_stall_reminder(&buckets).is_none());
    }

    #[test]
    fn f7_five_writes_escalates_to_refuse_directive() {
        let msgs: Vec<_> = (0..5)
            .map(|i| {
                make_assistant_tool_call(
                    "Write",
                    &format!(r#"{{"file_path":"/tmp/x/Cargo.toml","content":"v{}"}}"#, i),
                )
            })
            .collect();
        let buckets = collect_f7_stall_buckets(&msgs);
        assert_eq!(
            buckets
                .get(&("Write".to_string(), "/tmp/x/Cargo.toml".to_string()))
                .copied(),
            Some(5)
        );
        let reminder = build_f7_stall_reminder(&buckets).unwrap();
        assert!(
            reminder.contains("Do NOT call any tool"),
            "5+ hits must produce the strong directive, got: {reminder}"
        );
        assert!(reminder.contains("× 5"));
    }

    #[test]
    fn f7_bash_command_prefix_buckets_correctly() {
        // Same `cargo init` prefix with different trailing flags
        // should collapse to one bucket because we truncate to
        // BASH_COMMAND_PREFIX_LEN. Four identical (post-F14
        // threshold of 4) trips the warn level.
        let msgs = vec![
            make_assistant_tool_call("Bash", r#"{"command":"cd /tmp/x && cargo init --name a"}"#),
            make_assistant_tool_call("Bash", r#"{"command":"cd /tmp/x && cargo init --name a"}"#),
            make_assistant_tool_call("Bash", r#"{"command":"cd /tmp/x && cargo init --name a"}"#),
            make_assistant_tool_call("Bash", r#"{"command":"cd /tmp/x && cargo init --name a"}"#),
        ];
        let buckets = collect_f7_stall_buckets(&msgs);
        assert_eq!(
            buckets
                .iter()
                .filter(|(_, c)| **c >= F7_STALL_WARN_THRESHOLD)
                .count(),
            1
        );
    }

    #[test]
    fn f7_no_assistant_tool_calls_no_op() {
        let msgs = vec![
            make_user("hello"),
            crate::openai::IncomingMessage::synthetic_system("you are helpful".into()),
        ];
        let buckets = collect_f7_stall_buckets(&msgs);
        assert!(buckets.is_empty());
        assert!(build_f7_stall_reminder(&buckets).is_none());
    }

    #[test]
    fn f7_reminder_falls_back_to_synthetic_system_when_no_user() {
        // Only assistant messages — no user/tool to append to.
        // Four identical writes (post-F14 threshold=4) trips warn.
        let mut msgs = vec![
            make_assistant_tool_call("Write", r#"{"file_path":"/a","content":"x"}"#),
            make_assistant_tool_call("Write", r#"{"file_path":"/a","content":"x"}"#),
            make_assistant_tool_call("Write", r#"{"file_path":"/a","content":"x"}"#),
            make_assistant_tool_call("Write", r#"{"file_path":"/a","content":"x"}"#),
        ];
        let buckets = collect_f7_stall_buckets(&msgs);
        let reminder = build_f7_stall_reminder(&buckets).unwrap();
        let initial_len = msgs.len();
        append_f7_reminder_to_last_user(&mut msgs, &reminder);
        assert_eq!(msgs.len(), initial_len + 1, "synthetic system appended");
        let last = msgs.last().unwrap();
        assert_eq!(last.role, "system");
        assert!(last.content.text.contains("Write(/a) × 4"));
    }

    // ── F8 (2026-04-26): orphan partial-tool-call XML sanitiser ──

    #[test]
    fn f8_strips_orphan_function_open() {
        // `<function=Bash>` appearing in content (not as part of a real
        // `<tool_call>` envelope) is a model leak — the sanitizer
        // suppresses until a close tag arrives.
        let markers = Qwen3CoderParser.leak_markers();
        let mut buf = String::new();
        let mut suppress = false;
        let out = sanitize_content_chunk(
            "before<function=Bash>garbage</function>after",
            &mut buf,
            &mut suppress,
            &markers,
        );
        assert_eq!(out, "before");
        // The `</function>` close consumed the suppression; "after"
        // is still in the tag_scan_buf retained tail.
        assert!(!suppress, "close tag must clear suppression");
        let out2 = sanitize_content_chunk("", &mut buf, &mut suppress, &markers);
        assert_eq!(out + &out2, "before");
        assert_eq!(buf, "after");
    }

    #[test]
    fn f8_strips_orphan_tool_call_envelope() {
        let markers = Qwen3CoderParser.leak_markers();
        let mut buf = String::new();
        let mut suppress = false;
        let out = sanitize_content_chunk(
            "p<tool_call><function=X></function></tool_call>q",
            &mut buf,
            &mut suppress,
            &markers,
        );
        // Open `<tool_call>` triggers suppression. Inside, `<function=`
        // is *also* an open but we're already suppressing; the first
        // close (`</function>` here) ends suppression. Then
        // `</tool_call>q` flows through normally — the close tag is
        // dropped, and `q` is retained.
        assert_eq!(out, "p");
        // Drain the retained tail.
        let out2 = sanitize_content_chunk("", &mut buf, &mut suppress, &markers);
        let total = out + &out2;
        // `q` is in the retained buf since the chunk-boundary tail keeps
        // up to (tag_max - 1) bytes; flushed at stream end.
        assert!(
            total == "p" || total == "pq",
            "expected `p` (with `q` in tail) or `pq` (full flush), got: {total:?}"
        );
    }

    #[test]
    fn f8_legitimate_rust_source_with_fn_keyword_unchanged() {
        // Real Rust source uses `fn name()`, NOT `<function=name>`.
        // The latter is the qwen3_coder structural marker — only
        // emitted by the model as a tool-call leak. Verify our
        // sanitizer doesn't false-positive on legitimate code.
        let markers = Qwen3CoderParser.leak_markers();
        let mut buf = String::new();
        let mut suppress = false;
        let prose = "Use `fn add(a: i32, b: i32) -> i32 { a + b }` to define addition.";
        let out = sanitize_content_chunk(prose, &mut buf, &mut suppress, &markers);
        let final_out = out + &sanitize_content_chunk("", &mut buf, &mut suppress, &markers);
        // Because tag_scan_buf retains a tail to handle chunk
        // boundaries, the suffix may sit in `buf` — tolerate either
        // ordering.
        assert!(
            final_out.contains("fn add") || buf.contains("fn add"),
            "legitimate `fn add` text must be preserved; got out={final_out:?}, buf={buf:?}"
        );
        assert!(!suppress, "no leak markers in legitimate prose");
    }

    #[test]
    fn f8_truncated_open_at_chunk_boundary_buffers_correctly() {
        let markers = Qwen3CoderParser.leak_markers();
        let mut buf = String::new();
        let mut suppress = false;
        // Chunk 1: ends mid-tag. The sanitizer retains the tail in
        // `buf` to handle the boundary; out1 may be empty if the
        // whole chunk fits in the retained tail (depends on
        // max-tag length).
        let out1 = sanitize_content_chunk("prose <fun", &mut buf, &mut suppress, &markers);
        assert!(!suppress);
        // Chunk 2: completes the orphan `<function=` open, contains
        // the leak body and a close.
        let out2 = sanitize_content_chunk(
            "ction=Bash>cmd</function>tail",
            &mut buf,
            &mut suppress,
            &markers,
        );
        // After fusing `<fun` + `ction=Bash>`, suppression engages,
        // closes on `</function>`. The leak content (`<function=Bash>`,
        // `cmd`) must not appear ANYWHERE in the combined output —
        // either flushed as out2 or retained in buf, the leak is
        // gone for good.
        let combined = out1 + &out2 + &buf;
        assert!(
            !combined.contains("<function=Bash>"),
            "open tag must be stripped; got combined={combined:?}"
        );
        assert!(
            !combined.contains("cmd"),
            "leak body must be stripped; got combined={combined:?}"
        );
        assert!(
            combined.contains("prose"),
            "legitimate prose before the leak must survive; got combined={combined:?}"
        );
        assert!(!suppress, "close tag must end suppression");
    }

    #[test]
    fn f8_orphan_tool_use_pair_stripped() {
        let markers = Qwen3CoderParser.leak_markers();
        let mut buf = String::new();
        let mut suppress = false;
        let out = sanitize_content_chunk(
            "ok <tool_use>{\"name\":\"x\"}</tool_use> done",
            &mut buf,
            &mut suppress,
            &markers,
        );
        let final_out = out + &sanitize_content_chunk("", &mut buf, &mut suppress, &markers);
        // Should strip the leak envelope; both "ok " and "done" are
        // legitimate content.
        assert!(
            !final_out.contains("<tool_use>")
                && !final_out.contains("</tool_use>")
                && !final_out.contains(r#"{"name":"x"}"#),
            "orphan <tool_use> envelope must be fully suppressed; got {final_out:?}"
        );
    }
}
