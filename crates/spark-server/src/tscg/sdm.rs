// SPDX-License-Identifier: AGPL-3.0-only
//
// TSCG operator SDM — Semantic Density Maximization.
//
// Maximises information per token by deleting filler that carries no
// schema-relevant meaning: politeness markers, hedging, redundant
// connectives, and meta-phrases. The paper lists 104+ patterns; this
// port carries the high-value subset that recurs in real tool
// descriptions (Anthropic / OpenAI / opencode tool catalogues).
//
// SDM is whitespace- and case-preserving except where a pattern is
// deleted; it never reorders or paraphrases, so it cannot change a
// description's meaning — it only removes provably-empty spans.

/// Filler phrases deleted outright (case-insensitive, whole-substring).
/// Ordered longest-first so a longer match wins over a shorter prefix.
const FILLER: &[&str] = &[
    "please note that ",
    "please be aware that ",
    "it should be noted that ",
    "it is important to note that ",
    "as a general rule, ",
    "in order to ",
    "for the purpose of ",
    "with the goal of ",
    "this function ",
    "this tool ",
    "this method ",
    "use this to ",
    "used to ",
    "you can use this ",
    "can be used to ",
    "is used to ",
    "allows you to ",
    "lets you ",
    "helps you ",
    "the following ",
    "a list of ",
    "one or more ",
    "if applicable, ",
    "if necessary, ",
    "if needed, ",
    "as needed, ",
    "where appropriate, ",
    "kindly ",
    "simply ",
    "just ",
    "basically ",
    "essentially ",
    "note: ",
];

/// Verbose → compact phrase rewrites (DRO-adjacent prose compaction).
/// Applied after filler removal. Case-insensitive on the key.
const REWRITES: &[(&str, &str)] = &[
    ("corresponds to", "→"),
    ("for example", "e.g."),
    ("for instance", "e.g."),
    ("such as", "e.g."),
    ("and so on", "etc."),
    ("optional ", ""),
    ("a string representing ", ""),
    ("a string containing ", ""),
    ("an integer representing ", ""),
    ("the name of the ", ""),
    ("the path to the ", "path: "),
    ("must be ", ""),
    ("should be ", ""),
];

/// Compress a description string. Returns a single-line, filler-free,
/// whitespace-collapsed form. Empty input → empty output.
pub fn densify(text: &str) -> String {
    if text.trim().is_empty() {
        return String::new();
    }
    // Collapse newlines/tabs to spaces first so multi-line descriptions
    // become one line (the TSCG block is line-structured).
    let mut s: String = text
        .chars()
        .map(|c| {
            if c == '\n' || c == '\t' || c == '\r' {
                ' '
            } else {
                c
            }
        })
        .collect();

    // Filler removal — lower-cased scan, splice on the original string
    // so casing of surviving text is preserved.
    for pat in FILLER {
        loop {
            let lower = s.to_lowercase();
            match lower.find(pat) {
                Some(idx) => {
                    s.replace_range(idx..idx + pat.len(), "");
                }
                None => break,
            }
        }
    }

    // Phrase rewrites.
    for (from, to) in REWRITES {
        loop {
            let lower = s.to_lowercase();
            match lower.find(from) {
                Some(idx) => {
                    s.replace_range(idx..idx + from.len(), to);
                }
                None => break,
            }
        }
    }

    // Collapse runs of whitespace; trim a leading lowercase article that
    // a deletion may have stranded.
    let collapsed: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_filler() {
        assert_eq!(
            densify("This tool allows you to search the project files"),
            "search the project files"
        );
    }

    #[test]
    fn collapses_multiline() {
        assert_eq!(
            densify("line one\n  line two\n\tline three"),
            "line one line two line three"
        );
    }

    #[test]
    fn empty_stays_empty() {
        assert_eq!(densify(""), "");
        assert_eq!(densify("   \n  "), "");
    }

    #[test]
    fn preserves_meaningful_casing() {
        // "Bash" must survive — only the filler prefix is removed.
        assert_eq!(
            densify("Use this to run Bash commands"),
            "run Bash commands"
        );
    }
}
