// SPDX-License-Identifier: AGPL-3.0-only
//
// ParserState — the Earley item.
// Port of `struct ParserState` from xgrammar `cpp/earley_parser.h`.

/// A `sequence_id` of this value means the rule has not been expanded
/// yet (used for the initial state pushed into the parser).
pub const UNEXPANDED_RULE_START_SEQUENCE_ID: i32 = 128_000;

/// A `rule_start_pos` of this value marks an item as a root of the
/// parsing stack (no preceding input position).
pub const NO_PREV_INPUT_POS: i32 = -1;

/// An Earley item: a position inside a rule's body, paired with the
/// input position the rule was predicted at.
///
/// In the grammar a rule is always either a `Choices` of `Sequence`s
/// (or `EmptyStr`), or a `TagDispatch`. After FSM acceleration the body
/// is walked as an FSM, so `element_id` doubles as the FSM node id.
///
/// Faithful port of the C++ fields:
/// - `rule_id`        — which rule, or `-1` for the non-FSM body view.
/// - `sequence_id`    — which choice/sequence (a `GrammarExpr` id).
/// - `element_id`     — element index in the sequence, or FSM node id.
/// - `rule_start_pos` — input position the rule was predicted at.
/// - `sub_element_id` — sub-position inside the current element
///   (byte index in a `ByteString`, or remaining UTF-8 bytes).
/// - `repeat_count`   — times a `Repeat` element has matched.
/// - `partial_codepoint` — UTF-8 codepoint accumulated mid-character.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParserState {
    pub rule_id: i32,
    pub sequence_id: i32,
    pub element_id: i32,
    pub rule_start_pos: i32,
    pub sub_element_id: i32,
    pub repeat_count: i32,
    pub partial_codepoint: i32,
}

impl ParserState {
    /// Construct an item. `repeat_count` and `partial_codepoint`
    /// default to 0 — matching the C++ constructor defaults.
    pub fn new(
        rule_id: i32,
        sequence_id: i32,
        element_id: i32,
        rule_start_pos: i32,
        sub_element_id: i32,
    ) -> Self {
        Self {
            rule_id,
            sequence_id,
            element_id,
            rule_start_pos,
            sub_element_id,
            repeat_count: 0,
            partial_codepoint: 0,
        }
    }

    /// Construct an item with an explicit `repeat_count`.
    pub fn with_repeat(
        rule_id: i32,
        sequence_id: i32,
        element_id: i32,
        rule_start_pos: i32,
        sub_element_id: i32,
        repeat_count: i32,
    ) -> Self {
        Self {
            rule_id,
            sequence_id,
            element_id,
            rule_start_pos,
            sub_element_id,
            repeat_count,
            partial_codepoint: 0,
        }
    }

    /// The invalid sentinel item (`sequence_id == -1`).
    pub fn invalid() -> Self {
        Self::new(-1, -1, -1, -1, -1)
    }

    /// True if this is the invalid sentinel.
    pub fn is_invalid(&self) -> bool {
        self.sequence_id == -1
    }

    /// True if this item is the root of its parsing stack.
    pub fn is_root(&self) -> bool {
        self.rule_start_pos == NO_PREV_INPUT_POS
    }

    /// True if the rule still needs expansion before parsing.
    pub fn is_unexpanded(&self) -> bool {
        self.sequence_id == UNEXPANDED_RULE_START_SEQUENCE_ID
    }
}

/// The C++ uses two equality notions; we model the parsing one as
/// `PartialEq` (all fields). The cache notion ignores `rule_start_pos`
/// and `repeat_count` — see [`cache_key`].
///
/// `cache_key` returns the tuple the C++ `StateHashForCache` /
/// `StateEqualForParsing`-minus-position uses to deduplicate states
/// when computing the acceptable-token mask. Consumed by the W6
/// `GrammarMatcher` token-mask cache.
#[allow(dead_code)]
pub fn cache_key(s: &ParserState) -> (i32, i32, i32, i32) {
    (s.rule_id, s.sequence_id, s.element_id, s.sub_element_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_sentinel() {
        let s = ParserState::invalid();
        assert!(s.is_invalid());
        assert!(!ParserState::new(0, 0, 0, 0, 0).is_invalid());
    }

    #[test]
    fn root_detection() {
        assert!(ParserState::new(0, 0, 0, NO_PREV_INPUT_POS, 0).is_root());
        assert!(!ParserState::new(0, 0, 0, 3, 0).is_root());
    }

    #[test]
    fn unexpanded_detection() {
        let s = ParserState::new(
            0,
            UNEXPANDED_RULE_START_SEQUENCE_ID,
            0,
            NO_PREV_INPUT_POS,
            0,
        );
        assert!(s.is_unexpanded());
        assert!(!ParserState::new(0, 5, 0, 0, 0).is_unexpanded());
    }

    #[test]
    fn cache_key_ignores_position_and_repeat() {
        let a = ParserState::with_repeat(1, 2, 3, 10, 4, 5);
        let b = ParserState::with_repeat(1, 2, 3, 99, 4, 7);
        assert_eq!(cache_key(&a), cache_key(&b));
        assert_ne!(a, b);
    }
}
