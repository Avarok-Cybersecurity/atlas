// SPDX-License-Identifier: AGPL-3.0-only
//
// GrammarMutator — the base visitor/transformer pattern.
// Port of the `GrammarFunctor<int32_t, Grammar>` (a.k.a. `GrammarMutator`)
// CRTP template from `cpp/grammar_functor.h`.
//
// The C++ uses CRTP so a subclass overrides `VisitXxx`. Rust has no CRTP;
// instead this is a trait with default methods. A pass implements
// `GrammarMutator` and overrides only the visit methods it cares about.
// `MutatorState` holds the `base_grammar` + `builder` the C++ keeps as
// member fields.

use crate::grammar::builder::GrammarBuilder;
use crate::grammar::data::{GrammarData, TagDispatch};
use crate::grammar::expr::GrammarExprType;

/// The mutable state every mutator pass carries: the input grammar and
/// the builder for the output grammar.
pub struct MutatorState {
    /// The grammar being transformed (read-only input).
    pub base: GrammarData,
    /// The builder accumulating the output grammar.
    pub builder: GrammarBuilder,
    /// Name of the rule currently being visited.
    pub cur_rule_name: String,
}

impl MutatorState {
    /// New state wrapping `grammar`, with a fresh empty builder.
    pub fn new(grammar: GrammarData) -> Self {
        Self {
            base: grammar,
            builder: GrammarBuilder::new(),
            cur_rule_name: String::new(),
        }
    }
}

/// Decode a `TagDispatch` and rebuild it into `builder` unchanged.
/// Shared by the default `visit_tag_dispatch` and several passes.
pub fn rebuild_tag_dispatch(builder: &mut GrammarBuilder, td: &TagDispatch) -> i32 {
    use crate::grammar::builder::TagDispatchSpec;
    let spec = TagDispatchSpec {
        tag_rule_pairs: td.tag_rule_pairs.clone(),
        stop_eos: td.stop_eos,
        stop_str: td.stop_str.clone(),
        loop_after_dispatch: td.loop_after_dispatch,
        excluded_str: td.excluded_str.clone(),
    };
    builder.add_tag_dispatch(&spec)
}

/// A grammar-transforming pass. Mirrors C++ `GrammarMutator`: the default
/// `apply` adds empty rules to keep ids stable, then visits each rule
/// body and lookahead assertion.
pub trait GrammarMutator {
    /// The pass's mutable state. Implementors store a `MutatorState`.
    fn state(&mut self) -> &mut MutatorState;

    /// Run the pass, returning the transformed grammar.
    fn apply(&mut self, grammar: GrammarData) -> GrammarData {
        *self.state() = MutatorState::new(grammar);
        let num_rules = self.state().base.num_rules();
        for i in 0..num_rules {
            let name = self.state().base.rule(i).name.clone();
            self.state()
                .builder
                .add_empty_rule(name)
                .expect("duplicate rule name");
        }
        for i in 0..num_rules {
            let rule = self.state().base.rule(i).clone();
            self.state().cur_rule_name = rule.name.clone();
            let new_body = self.visit_expr_id(rule.body_expr_id);
            self.state()
                .builder
                .update_rule_body(i, new_body)
                .expect("rule id out of range");
            let new_la = self.visit_lookahead(rule.lookahead_assertion_id);
            self.state()
                .builder
                .update_lookahead_assertion(i, new_la)
                .expect("rule id out of range");
        }
        let root_name = self.state().base.root_rule().name.clone();
        let builder = std::mem::take(&mut self.state().builder);
        builder.get(&root_name).expect("root rule not found")
    }

    /// Visit a lookahead assertion by id (`-1` => `-1`).
    fn visit_lookahead(&mut self, id: i32) -> i32 {
        if id == -1 {
            return -1;
        }
        self.visit_expr_id(id)
    }

    /// Visit an expression by id.
    fn visit_expr_id(&mut self, id: i32) -> i32 {
        let expr = self.state().base.owned_expr(id);
        self.visit_expr(&expr)
    }

    /// Visit an expression, dispatching on its type.
    fn visit_expr(&mut self, e: &OwnedExpr) -> i32 {
        match e.kind {
            GrammarExprType::Sequence => self.visit_sequence(e),
            GrammarExprType::Choices => self.visit_choices(e),
            GrammarExprType::EmptyStr => self.visit_empty_str(e),
            GrammarExprType::ByteString => self.visit_byte_string(e),
            GrammarExprType::CharacterClass => self.visit_character_class(e),
            GrammarExprType::CharacterClassStar => self.visit_character_class_star(e),
            GrammarExprType::RuleRef => self.visit_rule_ref(e),
            GrammarExprType::TagDispatch => self.visit_tag_dispatch(e),
            GrammarExprType::Repeat => self.visit_repeat(e),
        }
    }

    /// Default: visit each child, rebuild a `Choices`.
    fn visit_choices(&mut self, e: &OwnedExpr) -> i32 {
        let ids: Vec<i32> = e.data.iter().map(|&c| self.visit_expr_id(c)).collect();
        self.state().builder.add_choices(&ids)
    }

    /// Default: visit each child, rebuild a `Sequence`.
    fn visit_sequence(&mut self, e: &OwnedExpr) -> i32 {
        let ids: Vec<i32> = e.data.iter().map(|&c| self.visit_expr_id(c)).collect();
        self.state().builder.add_sequence(&ids)
    }

    /// Default: decode the tag dispatch and rebuild it unchanged.
    fn visit_tag_dispatch(&mut self, e: &OwnedExpr) -> i32 {
        let td = e.tag_dispatch.clone().expect("tag dispatch payload");
        rebuild_tag_dispatch(&mut self.state().builder, &td)
    }

    /// Default: copy a leaf element verbatim into the builder.
    fn visit_element(&mut self, e: &OwnedExpr) -> i32 {
        self.state().builder.add_grammar_expr(e.kind, &e.data)
    }

    /// Default leaf visitors all defer to `visit_element`.
    fn visit_empty_str(&mut self, e: &OwnedExpr) -> i32 {
        self.visit_element(e)
    }
    /// Default: copy verbatim.
    fn visit_byte_string(&mut self, e: &OwnedExpr) -> i32 {
        self.visit_element(e)
    }
    /// Default: copy verbatim.
    fn visit_character_class(&mut self, e: &OwnedExpr) -> i32 {
        self.visit_element(e)
    }
    /// Default: copy verbatim.
    fn visit_character_class_star(&mut self, e: &OwnedExpr) -> i32 {
        self.visit_element(e)
    }
    /// Default: copy verbatim.
    fn visit_rule_ref(&mut self, e: &OwnedExpr) -> i32 {
        self.visit_element(e)
    }
    /// Default: copy verbatim.
    fn visit_repeat(&mut self, e: &OwnedExpr) -> i32 {
        self.visit_element(e)
    }
}

/// An owned copy of a `GrammarExpr` — needed because visiting borrows
/// `state` mutably while the expr lives in `state.base`.
#[derive(Debug, Clone)]
pub struct OwnedExpr {
    /// Expression kind.
    pub kind: GrammarExprType,
    /// CSR payload.
    pub data: Vec<i32>,
    /// Pre-decoded tag dispatch (only set when `kind == TagDispatch`).
    pub tag_dispatch: Option<TagDispatch>,
}

impl GrammarData {
    /// Decode `expr_id` into an [`OwnedExpr`], pre-decoding tag dispatch.
    pub(crate) fn owned_expr(&self, expr_id: i32) -> OwnedExpr {
        let e = self.expr(expr_id);
        let tag_dispatch = if e.kind == GrammarExprType::TagDispatch {
            Some(self.tag_dispatch(expr_id))
        } else {
            None
        };
        OwnedExpr {
            kind: e.kind,
            data: e.data.to_vec(),
            tag_dispatch,
        }
    }
}
