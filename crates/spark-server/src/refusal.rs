// SPDX-License-Identifier: AGPL-3.0-only

//! Heuristic refusal classifier.
//!
//! Populates `message.refusal` on the blocking chat-completion path so
//! safety-aware clients (OpenAI Python SDK, Vercel AI SDK) that branch on
//! `message.refusal != null` see the expected shape. Atlas does **not**
//! train safety behavior into its models — this detector only recognizes
//! the text the underlying model emits when it declines to answer.
//!
//! Honest scope: a prefix-matcher, not a safety classifier. It catches the
//! common refusal openings ("I cannot help with…", "I'm sorry, but I
//! can't…", "As an AI…") that instruction-tuned models produce. It will
//! miss subtle refusals and will false-positive on content that quotes a
//! refusal. Clients that need real safety-classification should run their
//! own moderation pass — `/v1/moderations` is a 501 stub on this server.
//!
//! Set `ATLAS_DISABLE_REFUSAL_DETECTION=1` to force `refusal: None` on all
//! responses, matching pre-PR-4 behavior byte-for-byte.

/// Prefix patterns matched case-insensitively against the stripped leading
/// text of the assistant message. Order matters only for determinism — the
/// first match wins.
const REFUSAL_PREFIXES: &[&str] = &[
    "i cannot ",
    "i can't help with ",
    "i can't assist with ",
    "i'm not able to ",
    "i am not able to ",
    "i'm unable to ",
    "i am unable to ",
    "i must decline",
    "i won't assist",
    "i will not assist",
    "i won't help",
    "i will not help",
    "sorry, i cannot",
    "sorry, but i can't",
    "sorry, but i cannot",
    "i'm sorry, but i can't",
    "i'm sorry, but i cannot",
    "i apologize, but i can't",
    "i apologize, but i cannot",
    "as an ai, i cannot",
    "as an ai, i can't",
    "as an ai language model, i cannot",
    "as an ai language model, i can't",
];

/// Returns the refusal sentence when `content` opens with one of the known
/// patterns, else `None`. The returned sentence is the first sentence
/// (truncated at `.`, `?`, or `!`) with trailing whitespace trimmed. When
/// the kill-switch env var is set, always returns `None`.
pub fn detect(content: &str) -> Option<String> {
    if std::env::var("ATLAS_DISABLE_REFUSAL_DETECTION").as_deref() == Ok("1") {
        return None;
    }
    let trimmed = content.trim_start();
    if trimmed.is_empty() {
        return None;
    }
    // Compare against prefixes using a lowercase view but return the
    // original-cased sentence so the client sees the model's exact words.
    let head: String = trimmed
        .chars()
        .take(48)
        .flat_map(|c| c.to_lowercase())
        .collect();
    let matched = REFUSAL_PREFIXES.iter().any(|p| head.starts_with(p));
    if !matched {
        return None;
    }
    // First sentence ends at the first terminal punctuation. If none is
    // present within a reasonable bound, fall back to the first line.
    let end_idx = trimmed
        .char_indices()
        .take(512)
        .find(|(_, c)| matches!(c, '.' | '?' | '!'))
        .map(|(i, c)| i + c.len_utf8());
    let sentence = match end_idx {
        Some(i) => &trimmed[..i],
        None => trimmed
            .split_once('\n')
            .map(|(line, _)| line)
            .unwrap_or(trimmed),
    };
    Some(sentence.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_canonical_refusal() {
        let r = detect("I cannot help with that request. Here is why…").unwrap();
        assert_eq!(r, "I cannot help with that request.");
    }

    #[test]
    fn matches_with_leading_whitespace() {
        let r = detect("   I'm sorry, but I can't assist with weapons design.").unwrap();
        assert_eq!(r, "I'm sorry, but I can't assist with weapons design.");
    }

    #[test]
    fn mixed_case_matches() {
        assert!(detect("As AN ai, I cannot provide that.").is_some());
    }

    #[test]
    fn non_refusal_returns_none() {
        assert!(detect("Sure, here's how to do that.").is_none());
        assert!(detect("").is_none());
        assert!(detect("I can do that for you.").is_none());
    }

    #[test]
    fn kill_switch_returns_none() {
        // SAFETY: single-threaded test; this env var isn't set elsewhere.
        unsafe {
            std::env::set_var("ATLAS_DISABLE_REFUSAL_DETECTION", "1");
        }
        let got = detect("I cannot help with that.");
        unsafe {
            std::env::remove_var("ATLAS_DISABLE_REFUSAL_DETECTION");
        }
        assert!(got.is_none());
    }

    #[test]
    fn no_terminator_falls_back_to_line() {
        let r = detect("I cannot answer that\nnext paragraph").unwrap();
        assert_eq!(r, "I cannot answer that");
    }
}
