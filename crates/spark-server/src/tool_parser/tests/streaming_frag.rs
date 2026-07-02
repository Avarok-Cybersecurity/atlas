// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

//! MTP / speculative-decode fragmentation robustness for the
//! Qwen3-Coder streaming detector.
//!
//! vLLM PR #35615 ("Qwen3Coder streaming tool parser silently drops
//! parameters with speculative decoding") identified three bugs that
//! caused parameter loss when multi-token bursts arrived from spec
//! decode. Atlas's StreamingToolDetector is structurally immune
//! because it buffers everything between `<tool_call>` and
//! `</tool_call>` then parses the complete inner block — there is no
//! per-parameter early-return path that could drop fragments. These
//! tests lock that property in: deltas can be split at arbitrary byte
//! boundaries (mid-tag, mid-value, mid-XML opener) and the final
//! emitted arguments JSON must remain byte-exact.

use super::super::*;

/// Concatenate all `ToolCallArgsFragment` payloads emitted for any idx, in
/// emission order. Live-streaming mode (`!buffer_args`) emits these instead of
/// a single `ToolCallDelta`; their concatenation is the complete JSON args.
fn collect_fragments(outputs: &[DetectorOutput]) -> String {
    let mut s = String::new();
    for o in outputs {
        if let DetectorOutput::ToolCallArgsFragment { fragment, .. } = o {
            s.push_str(fragment);
        }
    }
    s
}

/// Concatenated-fragment args OR (fallback) the single `ToolCallDelta` args.
/// Lets a test accept either the live-stream shape or the buffered shape.
fn args_from_outputs(outputs: &[DetectorOutput]) -> String {
    let frags = collect_fragments(outputs);
    if !frags.is_empty() {
        return frags;
    }
    for o in outputs {
        if let DetectorOutput::ToolCallDelta { args, .. } = o {
            return args.clone();
        }
    }
    panic!("no args emitted (neither fragments nor delta)");
}

#[test]
fn qwen3_coder_streaming_fragmented_at_path_boundary() {
    // Simulate MTP K=2 boundary splitting `/home/nologik` mid-string —
    // the failure shape from opencode-session.md where `/home/nologik`
    // arrived as `/home/nologin` (k → n char drop). Splitting the value
    // mid-character must not corrupt the final args. Live-streaming mode
    // emits the value once the `</parameter>` close lands, so the
    // concatenated fragments still carry the complete path.
    let mut det = StreamingToolDetector::new();
    let chunks = [
        "<tool_call>",
        "<function=Read>",
        "<parameter=file_path>",
        "/home/nolo",  // first fragment ends mid-word
        "gik/test.rs", // second fragment completes path
        "</parameter>",
        "</function>",
        "</tool_call>",
    ];
    let mut outputs = Vec::new();
    for c in chunks {
        outputs.extend(det.process(c));
    }
    let args: serde_json::Value = serde_json::from_str(&args_from_outputs(&outputs)).unwrap();
    assert_eq!(args["file_path"], "/home/nologik/test.rs");
}

#[test]
fn qwen3_coder_streaming_fragmented_at_xml_opener() {
    // Simulate spec-decode delivering a `<parameter=` opener split
    // across two deltas (`<param` then `eter=key>`). safe_emit_len
    // should hold back the partial tag instead of leaking it as
    // content; once complete it routes to the in-tag path.
    let mut det = StreamingToolDetector::new();
    let chunks = [
        "<tool_call><function=Read>",
        "<param",          // partial tag
        "eter=file_path>", // tag completes
        "/etc/hosts</parameter></function></tool_call>",
    ];
    let mut outputs = Vec::new();
    for c in chunks {
        outputs.extend(det.process(c));
    }
    let args: serde_json::Value = serde_json::from_str(&args_from_outputs(&outputs)).unwrap();
    assert_eq!(args["file_path"], "/etc/hosts");
}

#[test]
fn qwen3_coder_streaming_same_name_tool_calls_no_collision() {
    // vLLM bug 3 (name-based dedup in prev_tool_call_arr) would
    // collide two consecutive `Read` calls into one. Atlas keys by
    // call_counter, so two same-name calls must produce two distinct
    // outputs (whether ToolCall in bulk-fed mode or
    // ToolCallStart/Delta/End in incremental mode) with distinct
    // indices 0 and 1.
    //
    // This test uses bulk feed (close arrives in same chunk as
    // openers), which exercises the parse_one_call fast path and
    // emits two `ToolCall(tc, idx)` events.
    let mut det = StreamingToolDetector::new();
    let input = "<tool_call>\
                <function=Read>\
                <parameter=file_path>/a.rs</parameter>\
                </function>\
                </tool_call>\
                <tool_call>\
                <function=Read>\
                <parameter=file_path>/b.rs</parameter>\
                </function>\
                </tool_call>";
    let outputs = det.process(input);
    let calls: Vec<_> = outputs
        .iter()
        .filter_map(|o| match o {
            DetectorOutput::ToolCall(tc, idx) => Some((
                *idx,
                tc.function.name.clone(),
                tc.function.arguments.clone(),
            )),
            _ => None,
        })
        .collect();
    assert_eq!(
        calls.len(),
        2,
        "two ToolCall events for two same-name calls"
    );
    assert_eq!(calls[0].0, 0);
    assert_eq!(calls[1].0, 1);
    assert_eq!(calls[0].1, "Read");
    assert_eq!(calls[1].1, "Read");
    let args0: serde_json::Value = serde_json::from_str(&calls[0].2).unwrap();
    let args1: serde_json::Value = serde_json::from_str(&calls[1].2).unwrap();
    assert_eq!(args0["file_path"], "/a.rs");
    assert_eq!(args1["file_path"], "/b.rs");
}

#[test]
fn qwen3_coder_streaming_close_with_final_value_in_same_chunk() {
    // vLLM bug 1 (close-before-params ordering): a single burst
    // delivered `value</function>` together; their close check fired
    // first and dropped the value. Atlas's buffer-until-close design
    // means the value lands in the buffer BEFORE `</tool_call>` is
    // found; the close trigger then parses the whole inner block.
    // This test pins the property.
    let mut det = StreamingToolDetector::new();
    let chunks = [
        "<tool_call><function=Write>",
        "<parameter=path>/tmp/x</parameter>",
        // Final param value and ALL closing tags arrive in one burst.
        "<parameter=content>hello world</parameter></function></tool_call>",
    ];
    let mut outputs = Vec::new();
    for c in chunks {
        outputs.extend(det.process(c));
    }
    let args: serde_json::Value = serde_json::from_str(&args_from_outputs(&outputs)).unwrap();
    assert_eq!(args["path"], "/tmp/x");
    assert_eq!(
        args["content"], "hello world",
        "final-param-with-close burst must preserve the value"
    );
}

/// Build a tool def with the given name + parameters JSON schema.
pub(super) fn tool_def(name: &str, params: serde_json::Value) -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: name.into(),
            description: None,
            parameters: Some(params),
        },
    }
}

pub(super) fn write_and_bash_tools() -> Vec<ToolDefinition> {
    vec![
        tool_def(
            "Write",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["file_path", "content"]
            }),
        ),
        tool_def(
            "Bash",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"}
                },
                "required": ["command"]
            }),
        ),
    ]
}

#[test]
fn qwen3_coder_live_streams_multiple_fragments_before_end() {
    // With tool schemas + default (live) mode, a Write call fed in small
    // 5-char chunks must emit ToolCallStart, then MULTIPLE
    // ToolCallArgsFragment events BEFORE ToolCallEnd, and the concatenated
    // fragments must parse to the expected JSON (semantic equality —
    // key order follows model emission order, not byte order).
    let mut det = StreamingToolDetector::new_with_tools(write_and_bash_tools());
    let full = "<tool_call>\n<function=Write>\n\
                <parameter=file_path>\n/tmp/x.rs\n</parameter>\n\
                <parameter=content>\nhello\n</parameter>\n\
                </function>\n</tool_call>";
    let bytes = full.as_bytes();
    let mut outputs = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        // Advance by 5 bytes but stop on a UTF-8 char boundary (ASCII here).
        let end = (i + 5).min(bytes.len());
        outputs.extend(det.process(&full[i..end]));
        i = end;
    }
    outputs.extend(det.flush());

    let start_count = outputs
        .iter()
        .filter(|o| matches!(o, DetectorOutput::ToolCallStart { .. }))
        .count();
    assert!(start_count >= 1, "expected at least one ToolCallStart");

    // Index of the LAST fragment and the ToolCallEnd: every fragment must
    // precede the end.
    let end_pos = outputs
        .iter()
        .position(|o| matches!(o, DetectorOutput::ToolCallEnd { .. }))
        .expect("ToolCallEnd emitted");
    let frag_positions: Vec<usize> = outputs
        .iter()
        .enumerate()
        .filter(|(_, o)| matches!(o, DetectorOutput::ToolCallArgsFragment { .. }))
        .map(|(i, _)| i)
        .collect();
    assert!(
        frag_positions.len() >= 2,
        "expected MULTIPLE ToolCallArgsFragment events, got {}",
        frag_positions.len()
    );
    assert!(
        frag_positions.iter().all(|&p| p < end_pos),
        "all fragments must be emitted BEFORE ToolCallEnd"
    );

    let args: serde_json::Value = serde_json::from_str(&collect_fragments(&outputs)).unwrap();
    let expected = serde_json::json!({"file_path": "/tmp/x.rs", "content": "hello"});
    assert_eq!(args, expected);
}

#[test]
fn qwen3_coder_live_flush_path_emits_closing_brace() {
    // Live-server reality: `</tool_call>` is a stop/grammar-terminal token and
    // is NOT fed to process() — the residual (closing `}`) is emitted by
    // flush() at end-of-turn. This is a DIFFERENT path than the in-stream
    // close branch (which the test above exercises by feeding `</tool_call>`).
    // Regression: flush() bumped call_counter BEFORE stream_ready_fragments,
    // so the closing `}` fragment was emitted under idx+1 and the handler
    // dropped it — the streamed args were missing their trailing `}` and
    // failed JSON parsing client-side. Assert the concatenation is valid JSON.
    let mut det = StreamingToolDetector::new_with_tools(write_and_bash_tools());
    // NOTE: no `</tool_call>` — mirrors the stop-token-consumed live path.
    let full = "<tool_call>\n<function=Bash>\n\
                <parameter=command>\nls -lR /etc\n</parameter>\n\
                <parameter=timeout>\n30\n</parameter>\n\
                </function>\n";
    let bytes = full.as_bytes();
    let mut outputs = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let end = (i + 5).min(bytes.len());
        outputs.extend(det.process(&full[i..end]));
        i = end;
    }
    outputs.extend(det.flush());

    // The concatenated fragments MUST be complete, parseable JSON (with the
    // closing brace from flush) — this is exactly what failed live.
    let joined = collect_fragments(&outputs);
    let args: serde_json::Value = serde_json::from_str(&joined)
        .unwrap_or_else(|e| panic!("streamed args not valid JSON: {e}; joined={joined:?}"));
    let expected = serde_json::json!({"command": "ls -lR /etc", "timeout": 30});
    assert_eq!(
        args, expected,
        "flush-path streamed args must match + coerce"
    );
}

#[test]
fn qwen3_coder_live_coerces_integer_param() {
    // Bash.timeout is declared integer in the schema. The live fragment
    // must coerce the raw "30" text to a JSON number, matching the
    // buffered close-time coercion (coerce_all SSOT).
    let mut det = StreamingToolDetector::new_with_tools(write_and_bash_tools());
    let chunks = [
        "<tool_call>",
        "<function=Bash>",
        "<parameter=command>",
        "ls /tmp",
        "</parameter>",
        "<parameter=timeout>",
        "30",
        "</parameter>",
        "</function>",
        "</tool_call>",
    ];
    let mut outputs = Vec::new();
    for c in chunks {
        outputs.extend(det.process(c));
    }
    let args: serde_json::Value = serde_json::from_str(&collect_fragments(&outputs)).unwrap();
    assert_eq!(args["command"], "ls /tmp");
    assert_eq!(
        args["timeout"],
        serde_json::json!(30),
        "integer schema must coerce \"30\" → 30 (number, not string)"
    );
    assert!(args["timeout"].is_number(), "timeout must be a JSON number");
}

// `#[ignore]`: this test mutates the process-global env var
// `ATLAS_BUFFER_TOOL_ARGS`, which `StreamingToolDetector::new_with_tools`
// reads at construction. Under the default parallel test runner that read
// races other tests in this binary that build detectors expecting the live
// (default) path, so the var must not be set while they run. Run it
// explicitly and serially:
//   cargo test -p spark-server --bin spark -- --ignored --test-threads=1 \
//       tool_parser::tests::streaming_frag::kill_switch
#[test]
#[ignore = "mutates process-global ATLAS_BUFFER_TOOL_ARGS; run serially with --ignored --test-threads=1"]
fn kill_switch_buffers_full_args_no_fragments() {
    // ATLAS_BUFFER_TOOL_ARGS=1 restores legacy buffer-until-close: a
    // single ToolCallDelta with the full args, and NO
    // ToolCallArgsFragment events.
    let _guard = env_guard::set("ATLAS_BUFFER_TOOL_ARGS", "1");
    let mut det = StreamingToolDetector::new_with_tools(write_and_bash_tools());
    let chunks = [
        "<tool_call>",
        "<function=Write>",
        "<parameter=file_path>",
        "/tmp/x.rs",
        "</parameter>",
        "<parameter=content>",
        "hello",
        "</parameter>",
        "</function>",
        "</tool_call>",
    ];
    let mut outputs = Vec::new();
    for c in chunks {
        outputs.extend(det.process(c));
    }
    let frag_count = outputs
        .iter()
        .filter(|o| matches!(o, DetectorOutput::ToolCallArgsFragment { .. }))
        .count();
    let delta_count = outputs
        .iter()
        .filter(|o| matches!(o, DetectorOutput::ToolCallDelta { .. }))
        .count();
    assert_eq!(frag_count, 0, "kill-switch must emit NO live fragments");
    assert_eq!(
        delta_count, 1,
        "kill-switch must emit exactly one buffered ToolCallDelta"
    );
    let args: serde_json::Value = serde_json::from_str(&args_from_outputs(&outputs)).unwrap();
    assert_eq!(args["file_path"], "/tmp/x.rs");
    assert_eq!(args["content"], "hello");
}

/// Minimal serial env-var guard for the kill-switch test. Sets a var for the
/// duration of a guard, restoring the prior value on drop. A process-wide
/// mutex serialises env mutation so the env test cannot race a parallel test
/// reading the same var.
mod env_guard {
    use std::sync::{Mutex, MutexGuard, OnceLock};

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    pub struct Guard {
        key: &'static str,
        prev: Option<String>,
        _lock: MutexGuard<'static, ()>,
    }

    pub fn set(key: &'static str, val: &str) -> Guard {
        let lock = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let prev = std::env::var(key).ok();
        // SAFETY: env mutation is serialised by ENV_LOCK; no other thread in
        // this test binary touches this var without the same lock.
        unsafe {
            std::env::set_var(key, val);
        }
        Guard {
            key,
            prev,
            _lock: lock,
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            // SAFETY: still holding ENV_LOCK via `_lock`.
            unsafe {
                match &self.prev {
                    Some(v) => std::env::set_var(self.key, v),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }
}
