// SPDX-License-Identifier: AGPL-3.0-only

//! Server-side repair for `response_format=json_object` output that the
//! BPE-merged JSON-opener token corrupts at grammar step 0.

use crate::openai::{ChatCompletionRequest, ResponseFormat};

/// Repair the malformed-prefix output that `response_format=json_object`
/// produces on Qwen3.x tokenizers. Issue #43 (philoo99999, 2026-05-08):
/// the model's first emitted token at JSON-grammar step 0 is the BPE-
/// merged `{"` (Qwen vocab id 4754) because object-with-key is the
/// densest JSON-object opening in training. xgrammar's bitmask
/// correctly accepts `{"` as a valid 2-char JSON prefix (open object,
/// open key string) — but the model continues as if writing a fresh
/// top-level object inside the now-open key string, producing
/// `{"{"score":50,"reason":"test"}` (an unterminated outer object
/// whose key is itself a JSON object literal as raw text).
///
/// Server-side band-aid: when the emitted content fails to parse as
/// JSON but parses cleanly with the leading `{"` stripped, drop the
/// spurious prefix. The streaming path has the same bug; a grammar-
/// level fix that pre-advances the matcher past `{` is the proper
/// long-term solution and is tracked separately.
pub(crate) fn repair_json_object_prefix(req: &ChatCompletionRequest, content: String) -> String {
    if !matches!(req.response_format, Some(ResponseFormat::JsonObject)) {
        return content;
    }
    repair_inner(content)
}

fn repair_inner(content: String) -> String {
    if !content.starts_with("{\"") {
        return content;
    }
    if serde_json::from_str::<serde_json::Value>(content.trim()).is_ok() {
        return content;
    }
    let stripped = &content[2..];
    if serde_json::from_str::<serde_json::Value>(stripped.trim()).is_ok() {
        tracing::info!(
            "json_object: stripped spurious leading `{{\\\"` from response \
             (issue #43; BPE-merged token at grammar step 0). \
             Original len={}, repaired len={}",
            content.len(),
            stripped.len(),
        );
        return stripped.to_string();
    }
    content
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_when_already_valid_json() {
        let s = "{\"score\":50}".to_string();
        assert_eq!(repair_inner(s.clone()), s);
    }

    #[test]
    fn strips_spurious_prefix_when_strip_yields_valid_json() {
        let bug = "{\"{\"score\":50,\"reason\":\"test\"}".to_string();
        let repaired = repair_inner(bug);
        assert_eq!(repaired, "{\"score\":50,\"reason\":\"test\"}");
        let _: serde_json::Value = serde_json::from_str(&repaired).unwrap();
    }

    #[test]
    fn passthrough_when_strip_does_not_yield_valid_json() {
        let s = "{\"unterminated".to_string();
        assert_eq!(repair_inner(s.clone()), s);
    }

    #[test]
    fn handles_trailing_whitespace() {
        let bug = "{\"{\"x\":1}\n".to_string();
        let repaired = repair_inner(bug);
        assert_eq!(repaired, "{\"x\":1}\n");
    }

    #[test]
    fn passthrough_when_does_not_start_with_quoted_brace() {
        let s = "Sorry, I cannot do that.".to_string();
        assert_eq!(repair_inner(s.clone()), s);
    }
}
