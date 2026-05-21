// SPDX-License-Identifier: AGPL-3.0-only
//
// GrammarData — the BNF grammar AST.
// Port of `Grammar::Impl` from xgrammar `cpp/grammar_impl.h`.
//
// The AST is a set of rules; each rule's body is a `GrammarExpr`.
// Expressions are stored CSR-style: all payloads packed end-to-end in
// one `expr_data` vector, with `expr_indptr` recording each
// expression's start offset. At offset `s` the layout is
// `[type, data_len, data0, data1, …]`.

use super::expr::{GrammarExpr, GrammarExprType};
use crate::fsm::{CompactFsm, CompactFsmWithStartEnd};

/// A named production rule. `rule_id` is this rule's index in
/// [`GrammarData::rules`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rule {
    /// The rule name.
    pub name: String,
    /// The `GrammarExpr` id of the rule body.
    pub body_expr_id: i32,
    /// Id of the associated lookahead-assertion expr (a Sequence),
    /// or `-1` when the rule has none.
    pub lookahead_assertion_id: i32,
    /// Whether the lookahead assertion is exact.
    pub is_exact_lookahead: bool,
}

impl Rule {
    /// A rule with no lookahead assertion.
    pub fn new(name: impl Into<String>, body_expr_id: i32) -> Self {
        Self {
            name: name.into(),
            body_expr_id,
            lookahead_assertion_id: -1,
            is_exact_lookahead: false,
        }
    }
}

/// Decoded form of a `TagDispatch` expression.
/// Port of `Grammar::Impl::TagDispatch`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagDispatch {
    /// `(tag, rule_id)` pairs — matching a tag dispatches to its rule.
    pub tag_rule_pairs: Vec<(String, i32)>,
    /// If true, EOS may be generated and stops the dispatch.
    pub stop_eos: bool,
    /// Strings that stop the dispatch (only when `stop_eos` is false).
    pub stop_str: Vec<String>,
    /// If true, the dispatch loops after dispatching.
    pub loop_after_dispatch: bool,
    /// Strings excluded by the tag dispatch.
    pub excluded_str: Vec<String>,
}

/// Number of trailing scalar parameters in a `TagDispatch` payload:
/// `stop_eos`, `stop_str_expr_id`, `loop_after_dispatch`,
/// `excluded_str_expr_id`.
const TAG_DISPATCH_EXTRA: usize = 4;

/// The BNF grammar AST. Equivalent to xgrammar's `Grammar::Impl`.
///
/// The FSM acceleration fields (`complete_fsm`, `per_rule_fsms`,
/// `per_rule_fsm_hashes`, `per_rule_fsm_new_state_ids`) are populated
/// by the W3 grammar-functor passes (`GrammarFSMBuilder` /
/// `GrammarFSMHasher`); they are `Default::default()` until then.
#[derive(Debug, Clone, Default)]
pub struct GrammarData {
    /// Rules, indexed by `rule_id`.
    rules: Vec<Rule>,
    /// CSR payload store for all expressions.
    expr_data: Vec<i32>,
    /// CSR offsets: `expr_indptr[id]` is expr `id`'s start in `expr_data`.
    expr_indptr: Vec<i32>,
    /// Root rule id, or `-1` if unset.
    root_rule_id: i32,
    /// Rule ids permitted to match the empty string.
    pub allow_empty_rule_ids: Vec<i32>,
    /// Whether grammar optimization passes have run.
    pub optimized: bool,
    /// The single FSM holding every rule's states/edges, built by
    /// `GrammarFSMBuilder`. Empty until the FSM builder runs.
    pub complete_fsm: CompactFsm,
    /// Per-rule FSM views into `complete_fsm`. `None` for a rule whose
    /// body could not be expressed as an FSM. Empty until built.
    pub per_rule_fsms: Vec<Option<CompactFsmWithStartEnd>>,
    /// Per-rule structural hash of the rule's FSM, or `None` when the
    /// rule could not be hashed. Set by `GrammarFSMHasher`.
    pub per_rule_fsm_hashes: Vec<Option<u64>>,
    /// Per-rule `(original_state_id, new_state_id)` mapping produced
    /// alongside the hash by `GrammarFSMHasher`.
    pub per_rule_fsm_new_state_ids: Vec<Vec<(i32, i32)>>,
}

impl GrammarData {
    /// An empty grammar with no root.
    pub fn new() -> Self {
        Self {
            root_rule_id: -1,
            ..Default::default()
        }
    }

    /// Construct directly from CSR parts — used by the grammar builder
    /// (W2) and the deserializer.
    pub fn from_parts(
        rules: Vec<Rule>,
        expr_data: Vec<i32>,
        expr_indptr: Vec<i32>,
        root_rule_id: i32,
    ) -> Self {
        Self {
            rules,
            expr_data,
            expr_indptr,
            root_rule_id,
            ..Default::default()
        }
    }

    /// Mutable access to a rule's name — used by `RootRuleRenamer`.
    pub fn set_rule_name(&mut self, rule_id: i32, name: impl Into<String>) {
        self.rules[rule_id as usize].name = name.into();
    }

    /// Borrowed view of expr `expr_id` as `(type, &mut data[..])`. Used
    /// by `RepetitionNormalizer` to rewrite a `Repeat` payload in place.
    pub fn set_expr_data(&mut self, expr_id: i32, index: usize, value: i32) {
        let start = self.expr_indptr[expr_id as usize] as usize;
        self.expr_data[start + 2 + index] = value;
    }

    /// Number of rules.
    pub fn num_rules(&self) -> i32 {
        self.rules.len() as i32
    }

    /// The rule with the given id. Panics if out of bounds (matches
    /// the C++ `XGRAMMAR_DCHECK` contract).
    pub fn rule(&self, rule_id: i32) -> &Rule {
        &self.rules[rule_id as usize]
    }

    /// Mutable access to a rule.
    pub fn rule_mut(&mut self, rule_id: i32) -> &mut Rule {
        &mut self.rules[rule_id as usize]
    }

    /// All rules.
    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }

    /// The root rule id (`-1` if unset).
    pub fn root_rule_id(&self) -> i32 {
        self.root_rule_id
    }

    /// Set the root rule id.
    pub fn set_root_rule_id(&mut self, id: i32) {
        self.root_rule_id = id;
    }

    /// The root rule. Panics if the root is unset/out of bounds.
    pub fn root_rule(&self) -> &Rule {
        &self.rules[self.root_rule_id as usize]
    }

    /// Number of expressions.
    pub fn num_exprs(&self) -> i32 {
        self.expr_indptr.len() as i32
    }

    /// Append a raw `(type, data[])` expression to the CSR store and
    /// return its new expr id. Used by `LookaheadAssertionAnalyzer`,
    /// which adds Sequence exprs to an already-built grammar.
    pub fn append_expr(&mut self, kind: GrammarExprType, data: &[i32]) -> i32 {
        self.expr_indptr.push(self.expr_data.len() as i32);
        self.expr_data.push(kind as i32);
        self.expr_data.push(data.len() as i32);
        self.expr_data.extend_from_slice(data);
        self.expr_indptr.len() as i32 - 1
    }

    /// Borrowed view of expression `expr_id`. The CSR slot is
    /// `[type, data_len, data0, …]`; this returns `(type, &data[..])`.
    pub fn expr(&self, expr_id: i32) -> GrammarExpr<'_> {
        let start = self.expr_indptr[expr_id as usize] as usize;
        let kind = GrammarExprType::from_i32(self.expr_data[start])
            .expect("corrupt grammar: unknown GrammarExprType tag");
        let data_len = self.expr_data[start + 1] as usize;
        let data = &self.expr_data[start + 2..start + 2 + data_len];
        GrammarExpr { kind, data }
    }

    /// Decode a `ByteString` expression to its `String`.
    pub fn byte_string(&self, expr_id: i32) -> String {
        let e = self.expr(expr_id);
        e.data.iter().map(|&b| (b as u8) as char).collect()
    }

    /// Decode a `TagDispatch` expression. Panics if `expr_id` is not a
    /// tag dispatch (matches the C++ `XGRAMMAR_DCHECK`).
    pub fn tag_dispatch(&self, expr_id: i32) -> TagDispatch {
        let e = self.expr(expr_id);
        assert_eq!(
            e.kind,
            GrammarExprType::TagDispatch,
            "expr {expr_id} is not a TagDispatch"
        );
        let n = e.len();
        assert!(n >= TAG_DISPATCH_EXTRA, "TagDispatch payload too short");
        let pair_end = n - TAG_DISPATCH_EXTRA;

        let mut tag_rule_pairs = Vec::with_capacity(pair_end / 2);
        let mut i = 0;
        while i < pair_end {
            let tag = self.byte_string(e[i]);
            let rule_id = e[i + 1];
            tag_rule_pairs.push((tag, rule_id));
            i += 2;
        }

        let stop_eos = e[pair_end] != 0;
        let stop_str = self.choices_of_byte_strings(e[pair_end + 1]);
        let loop_after_dispatch = e[pair_end + 2] != 0;
        let excluded_str = self.choices_of_byte_strings(e[pair_end + 3]);

        TagDispatch {
            tag_rule_pairs,
            stop_eos,
            stop_str,
            loop_after_dispatch,
            excluded_str,
        }
    }

    /// Decode a `Choices` expression whose branches are all
    /// `ByteString`s into a `Vec<String>`.
    fn choices_of_byte_strings(&self, expr_id: i32) -> Vec<String> {
        let e = self.expr(expr_id);
        assert_eq!(e.kind, GrammarExprType::Choices, "expected a Choices expr");
        e.data.iter().map(|&id| self.byte_string(id)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny grammar by hand: one rule whose body is the byte
    /// string "ab". CSR slot for a ByteString: `[type=0, len=2, 'a','b']`.
    fn byte_string_grammar() -> GrammarData {
        let expr_data = vec![
            GrammarExprType::ByteString as i32,
            2,
            b'a' as i32,
            b'b' as i32,
        ];
        let expr_indptr = vec![0];
        let rules = vec![Rule::new("root", 0)];
        GrammarData::from_parts(rules, expr_data, expr_indptr, 0)
    }

    #[test]
    fn expr_view_decodes_csr() {
        let g = byte_string_grammar();
        assert_eq!(g.num_rules(), 1);
        assert_eq!(g.num_exprs(), 1);
        let e = g.expr(0);
        assert_eq!(e.kind, GrammarExprType::ByteString);
        assert_eq!(e.data, &[b'a' as i32, b'b' as i32]);
    }

    #[test]
    fn byte_string_decodes() {
        assert_eq!(byte_string_grammar().byte_string(0), "ab");
    }

    #[test]
    fn root_rule_resolves() {
        let g = byte_string_grammar();
        assert_eq!(g.root_rule_id(), 0);
        assert_eq!(g.root_rule().name, "root");
        assert_eq!(g.root_rule().body_expr_id, 0);
    }
}
