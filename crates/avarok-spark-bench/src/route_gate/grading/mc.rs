// SPDX-License-Identifier: AGPL-3.0-only
// T-AT: Atlas-native. Multiple-choice answer extraction, written from scratch for this lane.
//! Responsibility: the letter a multiple-choice reply commits to, and the comparison with the gold letter.

use std::sync::LazyLock;

use regex::Regex;

use super::{Grade, LETTERS};

// The keywords are case-insensitive; the option letter itself is not ("the answer is a polynomial" is not option A).
// Python's first pattern ends in a lookahead `(?![A-Za-z])`; the regex crate has none, so the letter is matched and the
// character after it is checked by `next_is_letter` (equivalent: the optional parts after the letter cannot consume
// a letter, so a letter right after the option letter is the only way the lookahead can fail).
static MC_ANSWER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i:answer\s*(?:is|:|=))\s*[:\-]?\s*\**\(?\s*([A-J])").expect("static regex")
});
static MC_BOXED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\\boxed\{\s*\\?(?:text\{)?\(?([A-J])\)?\}?\s*\}").expect("static regex")
});
static MC_CORRECT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i:(?:option|choice))\s*\(?([A-J])\)?\s+(?i:is)\s+(?i:(?:the\s+)?correct)")
        .expect("static regex")
});
static MC_LAST_LINE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\A\**\(?([A-J])[\).]?\)?\**\z").expect("static regex"));

fn next_is_letter(text: &str, at: usize) -> bool {
    text[at..]
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic())
}

/// Every option letter the pattern finds, in order (`finditer` semantics, optionally with the lookahead).
fn letters(re: &Regex, text: &str, check_next: bool) -> Vec<char> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos <= text.len() {
        let Some(caps) = re.captures_at(text, pos) else {
            break;
        };
        let (whole, group) = (caps.get(0).expect("group 0"), caps.get(1).expect("group 1"));
        if check_next && next_is_letter(text, group.end()) {
            pos = whole.start()
                + text[whole.start()..]
                    .chars()
                    .next()
                    .map_or(1, char::len_utf8);
            continue;
        }
        if let Some(c) = group.as_str().chars().next() {
            out.push(c.to_ascii_uppercase());
        }
        pos = whole.end().max(group.end());
    }
    out
}

/// The letter the reply commits to: the last "Answer: X" style match, else the last `\boxed{X}`, else the last
/// "option X is correct", else a final line that is a bare letter. Letters beyond the option count are ignored.
pub fn extract_mc_letter(text: &str, n_options: usize) -> Option<char> {
    let allowed: Vec<char> = LETTERS.chars().take(n_options).collect();
    for (re, check) in [
        (&*MC_ANSWER, true),
        (&*MC_BOXED, false),
        (&*MC_CORRECT, false),
    ] {
        let hits: Vec<char> = letters(re, text, check)
            .into_iter()
            .filter(|h| allowed.contains(h))
            .collect();
        if let Some(last) = hits.last() {
            return Some(*last);
        }
    }
    let last_line = text
        .trim()
        .lines()
        .map(str::trim)
        .rfind(|l| !l.is_empty())?;
    let caps = MC_LAST_LINE.captures(last_line)?;
    let c = caps.get(1)?.as_str().chars().next()?;
    allowed.contains(&c).then_some(c)
}

/// Grades a multiple-choice reply against the gold letter (case-insensitive).
pub fn grade_mc(text: &str, gold_letter: &str, n_options: usize) -> Grade {
    match extract_mc_letter(text, n_options) {
        None => Grade::no_answer(),
        Some(got) => {
            let gold = gold_letter.trim().to_uppercase();
            Grade::new(got.to_string() == gold, Some(got.to_string()), "ok")
        }
    }
}
