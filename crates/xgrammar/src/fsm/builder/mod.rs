// SPDX-License-Identifier: AGPL-3.0-only
//
// FSM builders — port of `cpp/fsm_builder.{h,cc}`.
//
//   regex_ir  — the regex intermediate representation + NFA construction
//   regex     — the regex string parser (`RegexFsmBuilder`)
//   trie      — the trie / Aho-Corasick builder (`TrieFsmBuilder`)
//
// `GrammarFSMBuilder` (TagDispatch / ByteString / RuleRef / Choices) is
// NOT ported here — it lives in `cpp/grammar_functor.h` and belongs to a
// later port wave (W3 functors), since it depends on the grammar AST.

pub mod regex;
pub mod regex_ir;
pub mod regex_leaf;
pub mod regex_parse;
pub mod trie;

use crate::fsm::with_start_end::FsmWithStartEnd;

/// Builder converting a regex string to an FSM.
///
/// Mirrors the C++ `RegexFSMBuilder` (a stateless namespace-like class).
pub struct RegexFsmBuilder;

impl RegexFsmBuilder {
    /// Parse `regex` and build its NFA. See [`regex::build_regex`].
    pub fn build(regex: &str) -> Result<FsmWithStartEnd, String> {
        regex::build_regex(regex)
    }
}

/// Builder converting a pattern list to a trie / Aho-Corasick FSM.
///
/// Mirrors the C++ `TrieFSMBuilder`.
pub struct TrieFsmBuilder;

impl TrieFsmBuilder {
    /// Build a trie-based FSM. See [`trie::build_trie`] for the full
    /// contract (overlap rules, back edges, excluded patterns).
    pub fn build(
        patterns: &[&[u8]],
        excluded_patterns: &[&[u8]],
        allow_overlap: bool,
        add_back_edges: bool,
    ) -> Option<trie::TrieBuildResult> {
        trie::build_trie(patterns, excluded_patterns, allow_overlap, add_back_edges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regex_builder_facade() {
        let f = RegexFsmBuilder::build("ab*").unwrap();
        assert!(f.accept_string(b"a"));
        assert!(f.accept_string(b"abbb"));
        assert!(!f.accept_string(b"b"));
    }

    #[test]
    fn trie_builder_facade() {
        let pats: Vec<&[u8]> = vec![b"hi", b"hello"];
        let res = TrieFsmBuilder::build(&pats, &[], true, false).unwrap();
        assert_eq!(res.end_states.len(), 2);
    }
}
