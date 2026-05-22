// SPDX-License-Identifier: AGPL-3.0-only
//
// Unit tests for the WGRAMMAR static/dynamic decomposition pass
// (Tier 3c). These exercise the classification + byte-precompute in
// isolation on hand-written EBNF; the end-to-end drive through the
// real `GrammarCompiler` lives in `tests/decompose_tests.rs`.

use super::*;
use crate::grammar::functor::{GrammarNormalizer, GrammarOptimizer};
use crate::grammar::parse_ebnf;

/// Parse + normalize + optimize an EBNF grammar — the same pipeline
/// `GrammarCompiler::compile_normalized` runs before compilation.
fn optimized(ebnf: &str) -> GrammarData {
    let g = parse_ebnf(ebnf, "root").expect("parse");
    GrammarOptimizer::apply(GrammarNormalizer::apply(g))
}

#[test]
fn pure_literal_rule_is_one_static_segment() {
    let g = optimized("root ::= \"hello\"\n");
    let d = decompose_static_regions(&g);
    let root = d.rule_region(g.root_rule_id());
    assert!(
        root.is_fully_static(),
        "a pure literal must be fully static"
    );
    assert_eq!(
        root.segments(),
        &[Segment::Static {
            bytes: b"hello".to_vec()
        }]
    );
}

#[test]
fn character_class_rule_is_one_dynamic_segment() {
    let g = optimized("root ::= [a-z]\n");
    let d = decompose_static_regions(&g);
    let root = d.rule_region(g.root_rule_id());
    assert!(!root.is_fully_static());
    assert_eq!(root.segments(), &[Segment::Dynamic]);
}

#[test]
fn interleaved_body_splits_into_static_and_dynamic_segments() {
    // A literal, a value slot, a literal — the WGRAMMAR pattern.
    let g = optimized("root ::= \"{\" [0-9] \"}\"\n");
    let d = decompose_static_regions(&g);
    let segs = d.rule_region(g.root_rule_id()).segments();
    assert_eq!(
        segs,
        &[
            Segment::Static {
                bytes: b"{".to_vec()
            },
            Segment::Dynamic,
            Segment::Static {
                bytes: b"}".to_vec()
            },
        ]
    );
}

#[test]
fn consecutive_literals_coalesce_into_one_static_segment() {
    // Two adjacent ByteStrings must merge into a single Static span.
    let g = optimized("root ::= \"ab\" \"cd\" [0-9]\n");
    let d = decompose_static_regions(&g);
    let segs = d.rule_region(g.root_rule_id()).segments();
    assert_eq!(
        segs[0],
        Segment::Static {
            bytes: b"abcd".to_vec()
        }
    );
    assert_eq!(segs[1], Segment::Dynamic);
    assert_eq!(segs.len(), 2);
}

#[test]
fn multi_branch_choice_decomposes_each_branch() {
    // A multi-branch choice becomes one `Choice` segment whose
    // branches are each decomposed — the per-branch literals stay
    // visible as static spans inside the branch.
    let g = optimized("root ::= \"yes\" | \"no\"\n");
    let d = decompose_static_regions(&g);
    let segs = d.rule_region(g.root_rule_id()).segments();
    let Segment::Choice { branches } = &segs[0] else {
        panic!("expected a Choice segment, got {segs:?}");
    };
    assert_eq!(branches.len(), 2);
    // Each branch is itself a fixed literal — static scaffolding
    // exposed behind the choice.
    for b in branches {
        assert!(b.is_fully_static(), "each literal branch is fully static");
    }
    // Both branch literals are counted as static segments.
    assert_eq!(d.static_segment_count(), 2);
    assert_eq!(d.dynamic_segment_count(), 0);
    assert_eq!(d.total_static_bytes(), 5); // "yes" + "no"
}

#[test]
fn choice_branch_scaffolding_is_recovered() {
    // The WGRAMMAR pattern: a choice whose non-empty branch carries
    // a fixed `{"k":` prefix before its value slot. That prefix
    // must surface as a static segment inside the branch.
    let g = optimized("root ::= (\"{\" [0-9] \"}\") | \"{}\"\n");
    let d = decompose_static_regions(&g);
    let segs = d.rule_region(g.root_rule_id()).segments();
    let Segment::Choice { branches } = &segs[0] else {
        panic!("expected a Choice segment, got {segs:?}");
    };
    // The non-empty branch decomposes to: static "{", dynamic, static "}".
    let nonempty = branches
        .iter()
        .find(|b| b.segments().len() == 3)
        .expect("the non-empty branch has three segments");
    assert!(nonempty.segments()[0].is_static());
    assert!(!nonempty.segments()[1].is_static());
    assert!(nonempty.segments()[2].is_static());
}
