// SPDX-License-Identifier: AGPL-3.0-only
//
// Tier-3c integration tests: WGRAMMAR static/dynamic decomposition.
//
// The unit-level segment mechanics (static literal vs dynamic value
// slot, coalescing, byte precompute) live in `compiler/decompose.rs`;
// here we drive the pass through the real `GrammarCompiler` end to end
// and assert the load-bearing invariants:
//
//   (1) the decomposition is reachable on every `CompiledGrammar` and
//       splits a tool-call schema body into static scaffolding spans
//       and dynamic value-slot spans;
//   (2) the compile-time-precomputed static bytes are BYTE-IDENTICAL to
//       what the matcher's byte-level forced path (`find_jump_forward_
//       string`, the Tier-3b byte chain) discovers at decode — they ARE
//       the same scaffolding, computed earlier;
//   (3) a genuine value slot (a character class / integer) is dynamic.

use super::{compiler, small_tokenizer};
use crate::compiler::{GrammarCompiler, RuleDecomposition, Segment};
use crate::matcher::GrammarMatcher;

/// Collect every static byte run in a decomposition, recursing into
/// `Choice` branches — the test counterpart of the recursive accessors.
fn all_static_runs(rule: &RuleDecomposition) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for seg in rule.segments() {
        match seg {
            Segment::Static { bytes } => out.push(bytes.clone()),
            Segment::Dynamic => {}
            Segment::Choice { branches } => {
                for b in branches {
                    out.extend(all_static_runs(b));
                }
            }
        }
    }
    out
}

// ----- decomposition is present and classifies scaffolding ---------

#[test]
fn pure_literal_grammar_is_one_static_segment() {
    let c = compiler(1);
    let cg = c
        .compile_grammar_from_ebnf("root ::= \"yes\"\n", "root")
        .expect("compile");
    let decomp = cg.decomposition();
    let root = decomp.rule_region(cg.grammar().root_rule_id());
    assert!(
        root.is_fully_static(),
        "a pure literal root must be fully static"
    );
    assert_eq!(
        root.segments(),
        &[Segment::Static {
            bytes: b"yes".to_vec()
        }]
    );
    assert_eq!(decomp.total_static_bytes(), 3);
    assert_eq!(decomp.dynamic_segment_count(), 0);
}

#[test]
fn character_class_grammar_is_one_dynamic_segment() {
    let c = compiler(1);
    let cg = c
        .compile_grammar_from_ebnf("root ::= [a-z]\n", "root")
        .expect("compile");
    let root = cg.decomposition().rule_region(cg.grammar().root_rule_id());
    assert!(!root.is_fully_static());
    assert!(root.segments().iter().all(|s| !s.is_static()));
}

// ----- tool-call schema: scaffolding static, value slots dynamic ---

#[test]
fn tool_schema_separates_scaffolding_from_value_slots() {
    // A minimal tool-call argument schema, compiled with FIXED
    // whitespace (`any_whitespace=false`, `indent=None`) so the JSON
    // separators are literal — the realistic tool-call setup. The
    // `root` body then interleaves fixed scaffolding (`{`, the property
    // key `"name":`, `}`) with the one value slot (the `basic_integer`
    // sub-rule). WGRAMMAR's intra-rule decomposition recovers both.
    let c = compiler(1);
    let schema = r#"{"type":"object","properties":{"name":{"type":"integer"}}}"#;
    let cg = c
        .compile_json_schema(schema, false, None, None, true, None)
        .expect("schema");
    let decomp = cg.decomposition();

    // Every rule is decomposed — the index spans the whole grammar.
    assert_eq!(decomp.regions().len(), cg.grammar().num_rules() as usize);
    // The schema body yields BOTH static scaffolding spans and dynamic
    // value-slot spans — that is the static/dynamic decomposition.
    assert!(
        decomp.static_segment_count() > 0,
        "a tool schema must have static scaffolding segments"
    );
    assert!(
        decomp.dynamic_segment_count() > 0,
        "a tool schema's value slots must be dynamic segments"
    );
    // The literal property key `"name"` is fixed scaffolding (it sits
    // inside the non-empty-object branch of a `Choice`): it must appear
    // verbatim in some static segment — exactly what WGRAMMAR
    // precomputes once. `all_static_runs` recurses into Choice branches.
    let key_is_static = decomp
        .regions()
        .iter()
        .flat_map(all_static_runs)
        .any(|b| contains(&b, br#""name""#));
    assert!(
        key_is_static,
        "the property key \"name\" must be precomputed static bytes"
    );
}

/// True if `needle` occurs anywhere in `haystack`.
fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

// ----- the compile-time precompute equals the byte-level forced path

#[test]
fn precomputed_static_bytes_match_byte_forced_path() {
    // THE correctness invariant of Tier 3c. A grammar whose body is one
    // long fixed literal: WGRAMMAR classifies it static and precomputes
    // its bytes AT COMPILE TIME. The matcher's byte-level forced walk
    // (`find_jump_forward_string`) instead DISCOVERS the same forced
    // byte run LAZILY at decode. The two must agree byte-for-byte — the
    // precompute is the same scaffolding, computed earlier, not a new
    // mechanism.
    let c = compiler(1);
    let cg = c
        .compile_grammar_from_ebnf("root ::= \"abcyes\"\n", "root")
        .expect("compile");

    // (a) the compile-time precompute.
    let root = cg.decomposition().rule_region(cg.grammar().root_rule_id());
    assert!(root.is_fully_static());
    let precomputed: &[u8] = match root.segments() {
        [Segment::Static { bytes }] => bytes,
        other => panic!("expected one static segment, got {other:?}"),
    };

    // (b) the lazy byte-level discovery from the start state.
    let mut matcher = GrammarMatcher::new(cg.clone(), None, false, -1);
    let lazy = matcher.find_jump_forward_string();

    assert_eq!(
        precomputed, lazy,
        "compile-time static precompute must equal the byte forced path"
    );
}

#[test]
fn precomputed_static_prefix_matches_forced_path_on_schema() {
    // On a real tool-call schema the `root` body is a `Choice` (empty
    // object vs non-empty object). Every branch opens with the same
    // fixed `{` scaffolding, so the byte-level forced path emits `{`
    // before reaching the genuine choice point. The precomputed
    // per-branch leading static segment must match that forced byte
    // run — same scaffolding, computed at compile time.
    let c = compiler(1);
    let schema = r#"{"type":"object","properties":{"name":{"type":"integer"}}}"#;
    let cg = c
        .compile_json_schema(schema, true, None, None, true, None)
        .expect("schema");
    let root = cg.decomposition().rule_region(cg.grammar().root_rule_id());
    // The body's first (and only) segment is the object Choice.
    let Segment::Choice { branches } = &root.segments()[0] else {
        panic!(
            "schema root body must be a Choice, got {:?}",
            root.segments()
        );
    };
    // Every branch's leading static segment starts with the opening `{`.
    for b in branches {
        let lead = b
            .segments()
            .first()
            .and_then(Segment::static_bytes)
            .expect("each object branch opens with a static `{`");
        assert!(
            lead.starts_with(b"{"),
            "branch must open with `{{`, got {lead:?}"
        );
    }

    // The byte-level forced path emits exactly that shared `{`.
    let mut matcher = GrammarMatcher::new(cg.clone(), None, false, -1);
    let forced = matcher.find_jump_forward_string();
    assert!(
        forced.starts_with(b"{"),
        "byte forced path {forced:?} must start with the precomputed `{{` scaffolding"
    );
}

// ----- decomposition is deterministic ------------------------------

#[test]
fn decomposition_is_deterministic_across_compiles() {
    let info = small_tokenizer();
    let c1 = GrammarCompiler::new(info.clone(), 1, true, -1);
    let c2 = GrammarCompiler::new(info, 1, true, -1);
    let g = "root ::= \"hello\" body\nbody ::= [0-9]\n";
    let d1 = c1
        .compile_grammar_from_ebnf(g, "root")
        .expect("compile")
        .decomposition()
        .clone();
    let d2 = c2
        .compile_grammar_from_ebnf(g, "root")
        .expect("compile")
        .decomposition()
        .clone();
    assert_eq!(d1, d2, "decomposition must be deterministic");
}
