// SPDX-License-Identifier: AGPL-3.0-only

//! The env-gated kill-switch test and its serial guard — split from
//! `streaming_frag` for the file-size cap. Kept together because the guard
//! exists solely for the test below it. The fixtures come from the sibling
//! module, which is where the rest of the fragment tests live.

use super::super::*;
use super::streaming_frag::{args_from_outputs, collect_fragments, write_and_bash_tools};

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

    // STATIC, DELIBERATELY — process lifecycle, test harness. It guards the
    // PROCESS environment, which is exactly the thing that cannot be scoped:
    // cargo runs unit tests in parallel threads of one binary, so a test that
    // mutates an env var must exclude every other test reading it. A
    // per-anything lock would not serialise the threads that need serialising.
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

#[test]
fn bare_function_streams_incrementally_and_emits_call_once() {
    // BARE `<function=...>` — no `<tool_call>` wrapper. This is the shape the
    // shipped GB10 `qwen3_xml` config actually receives, and it used to buffer
    // the ENTIRE block: a client saw nothing until `</function>` landed, while
    // the wrapped shape had been streaming header+params all along.
    //
    // Two things are asserted, and the second is the dangerous one: the
    // incremental path and the complete-block path must be MUTUALLY EXCLUSIVE,
    // or the call is delivered twice (once as fragments, once as a whole
    // `ToolCall`).
    let mut det = StreamingToolDetector::new_with_tools(write_and_bash_tools());
    let full = "<function=Write>\n\
                <parameter=file_path>\n/tmp/x.rs\n</parameter>\n\
                <parameter=content>\nhello\n</parameter>\n\
                </function>";
    let bytes = full.as_bytes();
    let mut outputs = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let end = (i + 5).min(bytes.len());
        outputs.extend(det.process(&full[i..end]));
        i = end;
    }
    outputs.extend(det.flush());

    // 1. The header must arrive BEFORE the block closes (the whole point).
    let start_pos = outputs
        .iter()
        .position(|o| matches!(o, DetectorOutput::ToolCallStart { .. }))
        .expect("bare <function=> must emit ToolCallStart before the block closes");
    let frag_positions: Vec<usize> = outputs
        .iter()
        .enumerate()
        .filter(|(_, o)| matches!(o, DetectorOutput::ToolCallArgsFragment { .. }))
        .map(|(i, _)| i)
        .collect();
    assert!(
        frag_positions.len() >= 2,
        "expected MULTIPLE incremental fragments for a bare function, got {}",
        frag_positions.len()
    );
    assert!(
        frag_positions.iter().all(|&p| p > start_pos),
        "fragments must follow ToolCallStart"
    );

    // 2. NO duplicate delivery: incrementally-streamed calls must NOT also be
    //    emitted as a complete `ToolCall`.
    let whole_calls = outputs
        .iter()
        .filter(|o| matches!(o, DetectorOutput::ToolCall(..)))
        .count();
    assert_eq!(
        whole_calls, 0,
        "a call streamed incrementally must not ALSO be emitted whole (duplicate delivery)"
    );

    // 3. The reassembled arguments must still be exactly right.
    let args: serde_json::Value = serde_json::from_str(&collect_fragments(&outputs)).unwrap();
    assert_eq!(
        args,
        serde_json::json!({"file_path": "/tmp/x.rs", "content": "hello"})
    );
}
