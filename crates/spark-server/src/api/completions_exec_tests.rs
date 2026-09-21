// SPDX-License-Identifier: AGPL-3.0-only

use super::finish_completion_text;

#[test]
fn raw_xtml_completion_preserves_literal_qwen_markers() {
    for text in [
        "<|open|>response<|sep|>Use <think>literal</think> tags.<|close|>response<|sep|>",
        "<|open|>argument key=\"text\" type=\"string\"<|sep|></think>\n  café 世界  ",
        "  <think>unfinished literal",
    ] {
        assert_eq!(finish_completion_text(text.into(), &[], true), text);
    }
}

#[test]
fn raw_xtml_preserves_stop_filter_and_legacy_reasoning_behavior() {
    let text = "<think>literal</think> answer STOP";
    let stops = ["STOP".to_owned()];
    assert_eq!(
        finish_completion_text(text.into(), &stops, true),
        "<think>literal</think> answer "
    );
    assert_eq!(finish_completion_text(text.into(), &stops, false), "answer");
}
