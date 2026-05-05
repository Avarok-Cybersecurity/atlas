// SPDX-License-Identifier: AGPL-3.0-only

//! Sanitizer-related tests (hoisted from `api.rs::sanitizer_tests`,
//! lines 7584-7967 of the pre-split file). The original
//! `super::sanitize_content_chunk` etc. paths are preserved by keeping
//! the body inside an inner `sanitizer_tests` module — `super::` here
//! resolves to `api::tests::sanitizer`, then `use super::super::*;`
//! pulls in the original parent (`api`) namespace.

#![cfg(test)]

mod sanitizer_tests {
    use super::super::super::*;
    use crate::tool_parser::{
    LeakMarkers,
    Qwen3CoderParser,
    ToolCallParser,
};

    /// F73 (2026-04-29): test wrapper that defaults the new
    /// `inside_envelope: &mut bool` parameter. Tests in this module
    /// that pre-date F73 don't exercise envelope semantics — they
    /// either use `LeakMarkers::EMPTY` (no envelope markers) or the
    /// Qwen3-coder marker set (no envelope_open/close). Either way,
    /// inside_envelope stays false throughout. Keeping the wrapper
    /// avoids per-test mechanical churn.
    fn sanitize_content_chunk(
        text: &str,
        tag_scan_buf: &mut String,
        suppressing_param_leak: &mut bool,
        markers: &LeakMarkers,
    ) -> String {
        let mut inside_envelope = false;
        super::sanitize_content_chunk(
            text,
            tag_scan_buf,
            suppressing_param_leak,
            &mut inside_envelope,
            markers,
        )
    }

    /// F73 (2026-04-29): inner `<invoke ...></invoke>` block passes
    /// through unsuppressed when wrapped in any of the three
    /// recognised MiniMax envelope forms (canonical, BPE-broken,
    /// rewritten). Verifies the live failure mode where opencode
    /// 9-tool sessions emitted `<minimax:_call>...<invoke ...>
    /// </invoke>...</minimax:_call>` and the prior sanitizer
    /// dropped the inner block.
    #[test]
    fn sanitizer_envelope_open_disables_orphan_suppression() {
        // Use MinimaxXmlParser's markers via the trait so the test
        // tracks what the parser actually exports.
        let markers = crate::tool_parser::MinimaxXmlParser.leak_markers();

        for envelope_open in
            &["<minimax:tool_call>", "<minimax:_call>", "<tool_call>"]
        {
            let envelope_close = match *envelope_open {
                "<minimax:tool_call>" => "</minimax:tool_call>",
                "<minimax:_call>" => "</minimax:_call>",
                _ => "</tool_call>",
            };
            let body = format!(
                "{envelope_open}\n<invoke name=\"bash\">\n<parameter name=\"command\">uname -r</parameter>\n</invoke>\n{envelope_close}"
            );
            let mut buf = String::new();
            let mut suppress = false;
            let mut env = false;
            let out = super::sanitize_content_chunk(
                &body, &mut buf, &mut suppress, &mut env, &markers,
            );
            // Inner content + envelope tags survive — the parser
            // downstream extracts the tool call from this stream.
            assert!(
                out.contains("<invoke name=\"bash\">"),
                "envelope {envelope_open}: <invoke> must survive: out={out:?}"
            );
            assert!(
                out.contains("uname -r"),
                "envelope {envelope_open}: command must survive: out={out:?}"
            );
            assert!(
                out.contains("</invoke>"),
                "envelope {envelope_open}: </invoke> must survive: out={out:?}"
            );
            // Envelope markers themselves are content too — the
            // parser normalises `<minimax:_call>` → `<tool_call>`
            // downstream and pulls out the inner block.
            assert!(
                out.contains(envelope_open),
                "envelope_open bytes must pass through: out={out:?}"
            );
            assert!(
                out.contains(envelope_close),
                "envelope_close bytes must pass through: out={out:?}"
            );
            assert!(!suppress, "envelope path must not enter orphan suppression");
            // After envelope_close the flag is back to false.
            assert!(!env, "envelope state cleared after close");
        }
    }

    /// F73 (2026-04-29): orphan-suppression behaviour preserved when
    /// `<invoke ...>` appears OUTSIDE any envelope. Unchanged from the
    /// pre-F73 sanitizer for a stray-fragment hallucination case.
    #[test]
    fn sanitizer_orphan_invoke_outside_envelope_still_suppressed() {
        let markers = crate::tool_parser::MinimaxXmlParser.leak_markers();
        let body = "prefix<invoke name=\"bash\">cmd</invoke>tail";
        let mut buf = String::new();
        let mut suppress = false;
        let mut env = false;
        let out = super::sanitize_content_chunk(
            body, &mut buf, &mut suppress, &mut env, &markers,
        );
        assert!(out.starts_with("prefix"), "non-orphan prefix emits: {out:?}");
        assert!(
            !out.contains("<invoke"),
            "stray <invoke> must still be suppressed: {out:?}"
        );
        assert!(
            !out.contains("cmd"),
            "suppressed body bytes must not leak: {out:?}"
        );
    }

    #[test]
    fn sanitizer_noop_for_empty_markers() {
        // A parser that opts out (Hermes, Gemma4, Mistral, BareJson)
        // passes text through verbatim. No buffering, no latency tail.
        let mut buf = String::new();
        let mut suppress = false;
        let out = sanitize_content_chunk(
            "<parameter=foo>value</parameter>",
            &mut buf,
            &mut suppress,
            &LeakMarkers::EMPTY,
        );
        assert_eq!(out, "<parameter=foo>value</parameter>");
        assert!(buf.is_empty(), "no markers → no tail buffering");
        assert!(!suppress);
    }

    #[test]
    fn sanitizer_suppresses_for_qwen3_markers() {
        // Existing Qwen3-coder behaviour via trait-delivered markers.
        // The orphan `<parameter=...>VALUE</parameter>` block is dropped
        // entirely; only the bytes outside the leak survive.
        let markers = Qwen3CoderParser.leak_markers();
        let mut buf = String::new();
        let mut suppress = false;
        let out = sanitize_content_chunk(
            "prefix<parameter=filePath>/tmp/x.txt</parameter>suffix</function>tail",
            &mut buf,
            &mut suppress,
            &markers,
        );
        // "prefix" emits; the `<parameter=filePath>...</parameter>` body
        // is suppressed; the stray `</function>` is dropped; "tail" is
        // short enough to stay buffered (no trailing tag-chars).
        assert!(out.starts_with("prefix"), "got: {out:?}");
        assert!(
            !out.contains("<parameter="),
            "orphan open must not leak: {out:?}"
        );
        assert!(
            !out.contains("/tmp/x.txt"),
            "suppressed body must not leak: {out:?}"
        );
        assert!(
            !out.contains("</function>"),
            "stray close must be stripped: {out:?}"
        );
    }

    #[test]
    fn sanitizer_fuses_tag_across_chunks() {
        // The whole point of the tail buffer: a tag arriving split
        // across two calls still matches. The first chunk is shorter
        // than (tag_max - 1), so nothing is emitted yet — we cannot
        // prove the `<param` suffix is not a tag prefix.
        let markers = Qwen3CoderParser.leak_markers();
        let mut buf = String::new();
        let mut suppress = false;
        let out1 = sanitize_content_chunk("abc<param", &mut buf, &mut suppress, &markers);
        assert!(!suppress, "partial tag must not trigger suppression");
        assert_eq!(out1, "", "short chunk stays in tail buffer awaiting fusion");
        let out2 = sanitize_content_chunk(
            "eter=x>body</parameter>tail",
            &mut buf,
            &mut suppress,
            &markers,
        );
        // Fusion: `<parameter=x>` found in the combined buffer.
        // "abc" prefix emits; body suppressed; `</parameter>` ends
        // suppression; "tail" stays buffered (too short to flush).
        assert!(out2.starts_with("abc"), "prefix emits after fusion: {out2:?}");
        assert!(!out2.contains("body"), "suppressed body must not leak: {out2:?}");
        assert!(!out2.contains("<parameter="), "orphan open must not leak: {out2:?}");
        assert!(!suppress, "close tag exits suppression state");
    }

    #[test]
    fn flush_empty_markers_emits_tail_verbatim() {
        // With EMPTY markers the fast path never buffers, but the flush
        // must still handle any residual correctly (it should always be
        // empty in practice).
        let mut buf = String::from("anything");
        let mut suppress = false;
        let out = flush_content_sanitizer(&mut buf, &mut suppress, &LeakMarkers::EMPTY);
        assert_eq!(out, "anything");
        assert!(buf.is_empty());
    }

    #[test]
    fn flush_drops_partial_tag_prefix() {
        // A bare `<par` tail could fuse into `<parameter=` on a next
        // chunk, but stream ended — drop it to avoid emitting mid-tag.
        let markers = Qwen3CoderParser.leak_markers();
        let mut buf = String::from("<par");
        let mut suppress = false;
        let out = flush_content_sanitizer(&mut buf, &mut suppress, &markers);
        assert_eq!(out, "");
    }

    // Note: bash-fence salvage tests have moved to
    // `crate::tool_salvage::tests` and now exercise the generic
    // schema-driven extractor (which handles bash fences as one
    // of several shapes). See `tool_salvage.rs` for coverage.

    #[test]
    fn strip_leaks_removes_mirror_block_from_real_dump_msg4() {
        // Verbatim content from dump seq=3 message[4] assistant turn
        // (opencode session that collapsed at turn 3, 2026-04-24).
        let tool_defs = vec![
            tool_parser::ToolDefinition {
                tool_type: "function".to_string(),
                function: tool_parser::FunctionDefinition {
                    name: "read".to_string(),
                    description: None,
                    parameters: Some(serde_json::json!({"type": "object"})),
                },
            },
            tool_parser::ToolDefinition {
                tool_type: "function".to_string(),
                function: tool_parser::FunctionDefinition {
                    name: "write".to_string(),
                    description: None,
                    parameters: Some(serde_json::json!({"type": "object"})),
                },
            },
        ];
        let content = "\n\nLet me create the Rust calculator module with the source files first.\n\n<read>\n<filePath>\n/tmp/calc-test40/src/lib.rs\n</filePath>\n<offset>\n1\n</offset>\n<limit>\n100\n</limit>\n</read>";
        let out = strip_xml_leaks_from_assistant_content(content, &tool_defs);
        assert!(out.contains("Let me create the Rust calculator module"),
            "prose must survive: {out:?}");
        assert!(!out.contains("<read>"), "read leak must be stripped: {out:?}");
        assert!(!out.contains("<filePath>"), "inner tags must be stripped: {out:?}");
        assert!(!out.contains("/tmp/calc-test40/src/lib.rs"),
            "leaked path must be removed: {out:?}");
    }

    #[test]
    fn strip_leaks_preserves_prose_without_blocks() {
        let tool_defs = vec![tool_parser::ToolDefinition {
            tool_type: "function".to_string(),
            function: tool_parser::FunctionDefinition {
                name: "read".to_string(),
                description: None,
                parameters: Some(serde_json::json!({"type": "object"})),
            },
        }];
        let content = "I'll read the config then write it back.";
        let out = strip_xml_leaks_from_assistant_content(content, &tool_defs);
        assert_eq!(out, content, "no XML block → no change");
    }

    #[test]
    fn strip_leaks_skips_unclosed_block() {
        // An unclosed `<read>` with no matching `</read>` must NOT be
        // stripped — might be a mid-prose angle-bracket artefact.
        let tool_defs = vec![tool_parser::ToolDefinition {
            tool_type: "function".to_string(),
            function: tool_parser::FunctionDefinition {
                name: "read".to_string(),
                description: None,
                parameters: Some(serde_json::json!({"type": "object"})),
            },
        }];
        let content = "prose <read> more prose";
        let out = strip_xml_leaks_from_assistant_content(content, &tool_defs);
        assert_eq!(out, content);
    }

    #[test]
    fn strip_leaks_handles_multiple_blocks() {
        // Two separate leaked blocks, both matching declared tools.
        let tool_defs = vec![
            tool_parser::ToolDefinition {
                tool_type: "function".to_string(),
                function: tool_parser::FunctionDefinition {
                    name: "read".to_string(),
                    description: None,
                    parameters: Some(serde_json::json!({"type": "object"})),
                },
            },
            tool_parser::ToolDefinition {
                tool_type: "function".to_string(),
                function: tool_parser::FunctionDefinition {
                    name: "write".to_string(),
                    description: None,
                    parameters: Some(serde_json::json!({"type": "object"})),
                },
            },
        ];
        let content = "A <read><filePath>/a</filePath></read> B <write><filePath>/b</filePath></write> C";
        let out = strip_xml_leaks_from_assistant_content(content, &tool_defs);
        assert!(!out.contains("<read>"));
        assert!(!out.contains("<write>"));
        assert!(out.contains('A') && out.contains('B') && out.contains('C'),
            "prose between blocks must survive: {out:?}");
    }

    #[test]
    fn strip_leaks_ignores_blocks_for_undeclared_tools() {
        // `<edit>` appears but no `edit` tool declared — leave it
        // alone (could be a legit HTML fragment in prose).
        let tool_defs = vec![tool_parser::ToolDefinition {
            tool_type: "function".to_string(),
            function: tool_parser::FunctionDefinition {
                name: "write".to_string(),
                description: None,
                parameters: Some(serde_json::json!({"type": "object"})),
            },
        }];
        let content = "prose <edit>something</edit> more";
        let out = strip_xml_leaks_from_assistant_content(content, &tool_defs);
        assert_eq!(out, content);
    }

    // Note: bare-XML tool-call salvage tests have moved to
    // `crate::tool_salvage::tests` and now exercise the generic
    // schema-driven extractor. See `tool_salvage.rs` for coverage.

    #[test]
    fn flush_before_tool_boundary_recovers_from_stuck_suppression() {
        // Simulates the production bug: model emits `<parameter=` in
        // prose (sanitizer enters suppression), then a real structured
        // tool call arrives and its `</parameter>` is consumed by the
        // detector — never reaching the sanitizer. Without the pre-tool
        // flush introduced alongside this test, `suppressing_param_leak`
        // would stay `true` forever and eat all post-tool content.
        let markers = Qwen3CoderParser.leak_markers();
        let mut buf = String::new();
        let mut suppress = false;

        // Step 1: prose orphan triggers suppression.
        let prose = sanitize_content_chunk(
            "Let me write it: <parameter=content>foo",
            &mut buf,
            &mut suppress,
            &markers,
        );
        assert_eq!(prose, "Let me write it: ", "prefix emits: {prose:?}");
        assert!(suppress, "orphan `<parameter=` enters suppression");

        // Step 2: simulate Content → Tool boundary (detector emits Tool
        // event). Our fix calls flush here.
        let pre_tool = flush_content_sanitizer(&mut buf, &mut suppress, &markers);
        assert_eq!(pre_tool, "", "suppressed tail is correctly dropped");
        assert!(!suppress, "flush clears the suppression flag");
        assert!(buf.is_empty(), "flush clears the tail buffer");

        // Step 3: post-tool content must flow through — this is the
        // regression we're pinning.
        let post_tool = sanitize_content_chunk(
            "Done — here is the result.",
            &mut buf,
            &mut suppress,
            &markers,
        );
        assert!(
            post_tool.starts_with("Done"),
            "post-tool content must reach the client: {post_tool:?}"
        );
        assert!(!suppress, "no new orphan, must stay out of suppression");
    }

    // Note: prose→Write salvage tests have moved to
    // `crate::tool_salvage::tests` and now exercise the generic
    // schema-driven extractor (header+body shape). See
    // `tool_salvage.rs` for coverage.
    //
    // Note: cross-turn prose-prefix Layer 4 was deleted along with
    // its `normalise_text_prefix` helper; the unified loop detector
    // in `crate::loop_detector` covers the same ground via shingle
    // similarity over assistant text. See `loop_detector.rs` tests
    // (`three_identical_intros_fire_loop`,
    // `slightly_varied_intros_still_fire`).
}
