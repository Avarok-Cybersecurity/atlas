// SPDX-License-Identifier: AGPL-3.0-only
//
// GrammarExpr — the node type of the BNF grammar AST.
// Port of the `GrammarExprType` enum + `GrammarExpr` view from
// xgrammar `cpp/grammar_impl.h`.

/// The type of a grammar expression node.
///
/// Each `GrammarExpr` is a `(type, data[])` pair; the `data` layout
/// depends on the type — see each variant's doc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GrammarExprType {
    /// `[byte0, byte1, …]` — a string of bytes (0-255), UTF-8 capable.
    ByteString = 0,
    /// `[is_negative, lower0, upper0, lower1, upper1, …]` — a set of
    /// unicode codepoint ranges, e.g. `[a-z]`, negatable `[^a-z]`.
    CharacterClass = 1,
    /// A star quantifier over a character class, e.g. `[a-z]*`.
    CharacterClassStar = 2,
    /// `[]` — the empty string `""`.
    EmptyStr = 3,
    /// `[rule_id]` — a reference to another rule.
    RuleRef = 4,
    /// `[expr_id0, expr_id1, …]` — concatenation of sub-expressions.
    Sequence = 5,
    /// `[expr_id0, expr_id1, …]` — alternation; any branch may match.
    Choices = 6,
    /// `[tag_expr0, rule_id0, …, stop_eos, stop_str_expr_id,
    /// loop_after_dispatch, excluded_str_expr_id]` — tag dispatch.
    TagDispatch = 7,
    /// `[rule_id, min_repeat_count, max_repeat_count]`.
    Repeat = 8,
}

impl GrammarExprType {
    /// Reconstruct from the raw `i32` stored in the CSR data vector.
    pub fn from_i32(v: i32) -> Option<Self> {
        Some(match v {
            0 => Self::ByteString,
            1 => Self::CharacterClass,
            2 => Self::CharacterClassStar,
            3 => Self::EmptyStr,
            4 => Self::RuleRef,
            5 => Self::Sequence,
            6 => Self::Choices,
            7 => Self::TagDispatch,
            8 => Self::Repeat,
            _ => return None,
        })
    }
}

/// A borrowed view of one grammar expression: its type and a slice of
/// its `i32` payload. Equivalent to xgrammar's `GrammarExpr` struct,
/// but lifetime-bound to the owning [`super::data::GrammarData`]
/// instead of holding a raw pointer.
#[derive(Debug, Clone, Copy)]
pub struct GrammarExpr<'a> {
    pub kind: GrammarExprType,
    pub data: &'a [i32],
}

impl<'a> GrammarExpr<'a> {
    /// Number of elements in the payload.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the payload is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl std::ops::Index<usize> for GrammarExpr<'_> {
    type Output = i32;
    fn index(&self, i: usize) -> &i32 {
        &self.data[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_roundtrips() {
        for v in 0..=8 {
            let t = GrammarExprType::from_i32(v).unwrap();
            assert_eq!(t as i32, v);
        }
        assert!(GrammarExprType::from_i32(9).is_none());
        assert!(GrammarExprType::from_i32(-1).is_none());
    }
}
