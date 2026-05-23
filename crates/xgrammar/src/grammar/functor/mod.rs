// SPDX-License-Identifier: AGPL-3.0-only
//
// Grammar functor passes — port wave W3.
//
// Pure-Rust port of `cpp/grammar_functor.{h,cc}`: the normalization,
// optimization, analysis and FSM-construction passes that turn a parsed
// BNF AST into an optimized, FSM-accelerated grammar.
//
// The C++ uses a CRTP visitor (`GrammarFunctor` / `GrammarMutator`);
// here that is a `GrammarMutator` trait with default visit methods, and
// each concrete pass is its own struct.
//
// Module map:
//   mutator              — GrammarMutator trait + shared state
//   normalizer           — SingleElementExprEliminator, RootRuleRenamer,
//                          GrammarNormalizer
//   structure_normalizer — StructureNormalizer
//   analyzer             — UsedRulesAnalyzer, RuleRefGraphFinder,
//                          AllowEmptyRuleAnalyzer
//   lookahead            — LookaheadAssertionAnalyzer
//   optimizer            — ByteStringFuser, RuleInliner,
//                          DeadCodeEliminator, RepetitionNormalizer,
//                          GrammarOptimizer
//   char_range           — UTF-8 character-range FSM helpers
//   fsm_builder          — GrammarFsmBuilder (per-rule FSM construction)
//   tag_dispatch_fsm     — TagDispatch FSM construction
//   hash_walk            — per-FSM BFS structural hashing
//   fsm_hasher           — GrammarFsmHasher
//   constructor          — SubGrammarAdder, GrammarUnion, GrammarConcat

pub mod analyzer;
pub mod char_range;
pub mod constructor;
pub mod fsm_builder;
pub mod fsm_hasher;
pub mod hash_walk;
pub mod lookahead;
pub mod mutator;
pub mod normalizer;
pub mod optimizer;
pub mod structure_normalizer;
pub mod tag_dispatch_fsm;

pub use analyzer::{AllowEmptyRuleAnalyzer, RuleRefGraphFinder, UsedRulesAnalyzer};
pub use constructor::{GrammarConcat, GrammarUnion, add_sub_grammar};
pub use fsm_builder::GrammarFsmBuilder;
pub use fsm_hasher::{GrammarFsmHasher, hash_sequence};
pub use lookahead::LookaheadAssertionAnalyzer;
pub use mutator::{GrammarMutator, MutatorState};
pub use normalizer::{GrammarNormalizer, RootRuleRenamer, SingleElementExprEliminator};
pub use optimizer::{
    ByteStringFuser, DeadCodeEliminator, GrammarOptimizer, RepetitionNormalizer, RuleInliner,
};
pub use structure_normalizer::StructureNormalizer;

#[cfg(test)]
mod integration_tests {
    //! End-to-end pipeline tests: parse -> normalize -> optimize.
    use crate::grammar::functor::{GrammarNormalizer, GrammarOptimizer};
    use crate::grammar::parse_ebnf_default;
    use crate::grammar::printer::print_grammar;

    #[test]
    fn full_pipeline_byte_string() {
        let g = parse_ebnf_default("root ::= \"hello world\"\n").unwrap();
        let norm = GrammarNormalizer::apply(g);
        let opt = GrammarOptimizer::apply(norm);
        assert!(opt.optimized);
        let fsm = opt.per_rule_fsms[opt.root_rule_id() as usize]
            .as_ref()
            .unwrap();
        assert!(fsm.accept_string(b"hello world"));
    }

    #[test]
    fn full_pipeline_recursive_grammar() {
        let ebnf = "root ::= \"(\" root \")\" | \"\"\n";
        let g = parse_ebnf_default(ebnf).unwrap();
        let norm = GrammarNormalizer::apply(g);
        let opt = GrammarOptimizer::apply(norm);
        assert!(opt.optimized);
        assert!(!opt.allow_empty_rule_ids.is_empty());
    }

    #[test]
    fn full_pipeline_alternation() {
        let g = parse_ebnf_default("root ::= \"yes\" | \"no\"\n").unwrap();
        let norm = GrammarNormalizer::apply(g);
        let opt = GrammarOptimizer::apply(norm);
        let fsm = opt.per_rule_fsms[opt.root_rule_id() as usize]
            .as_ref()
            .unwrap();
        assert!(fsm.accept_string(b"yes"));
        assert!(fsm.accept_string(b"no"));
        assert!(!fsm.accept_string(b"maybe"));
    }

    #[test]
    fn normalize_then_print_is_stable() {
        let g = parse_ebnf_default("root ::= \"a\" sub\nsub ::= \"b\" | \"c\"\n").unwrap();
        let norm = GrammarNormalizer::apply(g);
        let printed = print_grammar(&norm);
        assert!(printed.contains("root ::="));
        assert!(printed.contains("sub ::="));
    }

    #[test]
    fn optimizer_idempotent_on_optimized_grammar() {
        let g = parse_ebnf_default("root ::= \"x\" \"y\"\n").unwrap();
        let norm = GrammarNormalizer::apply(g);
        let opt1 = GrammarOptimizer::apply(norm);
        let opt2 = GrammarOptimizer::apply(opt1.clone());
        assert_eq!(print_grammar(&opt1), print_grammar(&opt2));
    }
}
