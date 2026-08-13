// SPDX-License-Identifier: AGPL-3.0-only

use serde_json::Value;

use crate::tokenizer::deepseek_v4::encode_messages;

const INPUT_1: &str = include_str!("../../../test_data/deepseek_v4_encoding/test_input_1.json");
const INPUT_2: &str = include_str!("../../../test_data/deepseek_v4_encoding/test_input_2.json");
const INPUT_3: &str = include_str!("../../../test_data/deepseek_v4_encoding/test_input_3.json");
const INPUT_4: &str = include_str!("../../../test_data/deepseek_v4_encoding/test_input_4.json");
const OUTPUT_1: &str = include_str!("../../../test_data/deepseek_v4_encoding/test_output_1.txt");
const OUTPUT_2: &str = include_str!("../../../test_data/deepseek_v4_encoding/test_output_2.txt");
const OUTPUT_3: &str = include_str!("../../../test_data/deepseek_v4_encoding/test_output_3.txt");
const OUTPUT_4: &str = include_str!("../../../test_data/deepseek_v4_encoding/test_output_4.txt");

fn render(input: &str, thinking: bool) -> String {
    let fixture: Value = serde_json::from_str(input).expect("official fixture JSON");
    let (messages, tools) = if let Some(object) = fixture.as_object() {
        (
            object["messages"].as_array().expect("messages"),
            object
                .get("tools")
                .and_then(Value::as_array)
                .map(Vec::as_slice),
        )
    } else {
        (fixture.as_array().expect("message array"), None)
    };
    encode_messages(messages, tools, thinking, Some("low")).expect("official encoding")
}

fn assert_exact(actual: &str, expected: &str) {
    if actual == expected {
        return;
    }
    let mismatch = actual
        .bytes()
        .zip(expected.bytes())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| actual.len().min(expected.len()));
    let actual_start = actual.floor_char_boundary(mismatch.saturating_sub(80));
    let expected_start = expected.floor_char_boundary(mismatch.saturating_sub(80));
    let actual_end = actual.floor_char_boundary(actual.len().min(mismatch + 120));
    let expected_end = expected.floor_char_boundary(expected.len().min(mismatch + 120));
    panic!(
        "DeepSeek-V4 fixture mismatch at byte {mismatch}; lengths actual={} expected={}\nactual: {:?}\nexpected: {:?}",
        actual.len(),
        expected.len(),
        &actual[actual_start..actual_end],
        &expected[expected_start..expected_end],
    );
}

#[test]
fn official_encoding_fixtures_are_byte_exact() {
    for (input, output, thinking) in [
        (INPUT_1, OUTPUT_1, true),
        (INPUT_2, OUTPUT_2, true),
        (INPUT_3, OUTPUT_3, true),
        (INPUT_4, OUTPUT_4, false),
    ] {
        assert_exact(&render(input, thinking), output);
    }
}

#[test]
fn reasoning_effort_prefixes_are_distinct_and_generation_suffix_is_native() {
    let messages = serde_json::json!([{"role": "user", "content": "test"}]);
    let messages = messages.as_array().unwrap();
    let low = encode_messages(messages, None, true, Some("low")).unwrap();
    let high = encode_messages(messages, None, true, Some("high")).unwrap();
    let max = encode_messages(messages, None, true, Some("max")).unwrap();
    assert!(low.ends_with("<｜Assistant｜><think>"));
    assert!(high.starts_with("<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum"));
    assert!(max.starts_with("<｜begin▁of▁sentence｜>Reasoning Effort: Beyond maximum"));
    assert_ne!(high, max);
}

#[test]
fn non_thinking_generation_is_preclosed() {
    let messages = serde_json::json!([{"role": "user", "content": "test"}]);
    let rendered =
        encode_messages(messages.as_array().unwrap(), None, false, None).expect("direct mode");
    assert!(rendered.ends_with("<｜Assistant｜></think>"));
}
