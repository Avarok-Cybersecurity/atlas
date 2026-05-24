// SPDX-License-Identifier: AGPL-3.0-only
//
// Tests for `GrammarBuilder`. Split out of `builder.rs` to keep each
// file under the 250-line cap.

use super::{BuilderError, CharacterClassElement, GrammarBuilder, TagDispatchSpec};
use crate::grammar::expr::GrammarExprType;

#[test]
fn byte_string_roundtrips() {
    let mut b = GrammarBuilder::new();
    let e = b.add_byte_string("ab");
    b.add_rule_named("root", e).unwrap();
    let g = b.get("root").unwrap();
    assert_eq!(g.byte_string(0), "ab");
    assert_eq!(g.num_rules(), 1);
}

#[test]
fn byte_string_uses_utf8_bytes() {
    let mut b = GrammarBuilder::new();
    // "é" is 2 UTF-8 bytes.
    let e = b.add_byte_string("é");
    b.add_rule_named("root", e).unwrap();
    let g = b.get("root").unwrap();
    assert_eq!(g.expr(0).len(), 2);
}

#[test]
fn byte_string_bytes_overload() {
    let mut b = GrammarBuilder::new();
    let e = b.add_byte_string_bytes(&[0x41, 0x42]);
    b.add_rule_named("root", e).unwrap();
    let g = b.get("root").unwrap();
    assert_eq!(g.byte_string(0), "AB");
}

#[test]
fn character_class_layout() {
    let mut b = GrammarBuilder::new();
    let e = b.add_character_class(
        &[
            CharacterClassElement::new('a' as i32, 'z' as i32),
            CharacterClassElement::new('0' as i32, '9' as i32),
        ],
        true,
    );
    b.add_rule_named("root", e).unwrap();
    let g = b.get("root").unwrap();
    let ex = g.expr(0);
    assert_eq!(ex.kind, GrammarExprType::CharacterClass);
    assert_eq!(
        ex.data,
        &[1, 'a' as i32, 'z' as i32, '0' as i32, '9' as i32]
    );
}

#[test]
fn character_class_star_layout() {
    let mut b = GrammarBuilder::new();
    let e =
        b.add_character_class_star(&[CharacterClassElement::new('a' as i32, 'z' as i32)], false);
    b.add_rule_named("root", e).unwrap();
    let g = b.get("root").unwrap();
    assert_eq!(g.expr(0).kind, GrammarExprType::CharacterClassStar);
}

#[test]
fn empty_str_and_rule_ref() {
    let mut b = GrammarBuilder::new();
    let empty = b.add_empty_str();
    let r0 = b.add_rule_named("a", empty).unwrap();
    let rref = b.add_rule_ref(r0);
    b.add_rule_named("root", rref).unwrap();
    let g = b.get("root").unwrap();
    assert_eq!(g.expr(empty).kind, GrammarExprType::EmptyStr);
    assert_eq!(g.expr(rref).kind, GrammarExprType::RuleRef);
    assert_eq!(g.expr(rref).data, &[r0]);
}

#[test]
fn sequence_and_choices() {
    let mut b = GrammarBuilder::new();
    let a = b.add_byte_string("a");
    let c = b.add_byte_string("c");
    let seq = b.add_sequence(&[a, c]);
    let ch = b.add_choices(&[a, seq]);
    b.add_rule_named("root", ch).unwrap();
    let g = b.get("root").unwrap();
    assert_eq!(g.expr(seq).kind, GrammarExprType::Sequence);
    assert_eq!(g.expr(seq).data, &[a, c]);
    assert_eq!(g.expr(ch).kind, GrammarExprType::Choices);
}

#[test]
fn num_grammar_exprs_tracks_additions() {
    let mut b = GrammarBuilder::new();
    assert_eq!(b.num_grammar_exprs(), 0);
    b.add_empty_str();
    b.add_empty_str();
    assert_eq!(b.num_grammar_exprs(), 2);
}

#[test]
fn duplicate_rule_errors() {
    let mut b = GrammarBuilder::new();
    let e = b.add_empty_str();
    b.add_rule_named("root", e).unwrap();
    let err = b.add_rule_named("root", e).unwrap_err();
    assert_eq!(err, BuilderError::DuplicateRule("root".to_string()));
}

#[test]
fn missing_root_errors() {
    let mut b = GrammarBuilder::new();
    let e = b.add_empty_str();
    b.add_rule_named("notroot", e).unwrap();
    let err = b.get("root").unwrap_err();
    assert_eq!(err, BuilderError::RootRuleNotFound("root".to_string()));
}

#[test]
fn empty_rule_and_update_body() {
    let mut b = GrammarBuilder::new();
    let rid = b.add_empty_rule("root").unwrap();
    assert_eq!(b.get_rule(rid).body_expr_id, -1);
    let body = b.add_empty_str();
    b.update_rule_body(rid, body).unwrap();
    assert_eq!(b.get_rule(rid).body_expr_id, body);
    b.update_rule_body_named("root", body).unwrap();
    assert!(b.update_rule_body(99, body).is_err());
    assert!(b.update_rule_body_named("missing", body).is_err());
}

#[test]
fn new_rule_name_dedups() {
    let mut b = GrammarBuilder::new();
    let e = b.add_empty_str();
    assert_eq!(b.get_new_rule_name("root"), "root");
    b.add_rule_named("root", e).unwrap();
    assert_eq!(b.get_new_rule_name("root"), "root_1");
    b.add_rule_named("root_1", e).unwrap();
    assert_eq!(b.get_new_rule_name("root"), "root_2");
}

#[test]
fn new_rule_name_cache_amortized() {
    // The next_cnt_per_hint cache must produce the same names as a
    // fresh-probe implementation even across many same-hint calls.
    let mut b = GrammarBuilder::new();
    let e = b.add_empty_str();
    b.add_rule_named("t", e).unwrap();
    for i in 1..=20 {
        let name = b.get_new_rule_name("t");
        assert_eq!(name, format!("t_{i}"));
        b.add_rule_named(&name, e).unwrap();
    }
    // A still-free lower suffix introduced out of band is still found:
    // a different hint shares no cache entry.
    assert_eq!(b.get_new_rule_name("u"), "u");
}

#[test]
fn rule_with_hint_unique() {
    let mut b = GrammarBuilder::new();
    let e = b.add_empty_str();
    let r1 = b.add_rule_with_hint("tmp", e).unwrap();
    let r2 = b.add_rule_with_hint("tmp", e).unwrap();
    assert_ne!(r1, r2);
    assert_eq!(b.get_rule(r1).name, "tmp");
    assert_eq!(b.get_rule(r2).name, "tmp_1");
}

#[test]
fn empty_rule_with_hint_unique() {
    let mut b = GrammarBuilder::new();
    let e = b.add_empty_str();
    b.add_rule_named("h", e).unwrap();
    let r = b.add_empty_rule_with_hint("h").unwrap();
    assert_eq!(b.get_rule(r).name, "h_1");
}

#[test]
fn lookahead_assertion_set() {
    let mut b = GrammarBuilder::new();
    let body = b.add_empty_str();
    let rid = b.add_rule_named("root", body).unwrap();
    let la = b.add_sequence(&[body]);
    b.update_lookahead_assertion(rid, la).unwrap();
    b.update_lookahead_exact(rid, true).unwrap();
    assert_eq!(b.get_rule(rid).lookahead_assertion_id, la);
    assert!(b.get_rule(rid).is_exact_lookahead);
    b.update_lookahead_assertion_named("root", -1).unwrap();
    assert_eq!(b.get_rule(rid).lookahead_assertion_id, -1);
}

#[test]
fn lookahead_out_of_range_errors() {
    let mut b = GrammarBuilder::new();
    assert!(b.update_lookahead_assertion(0, 0).is_err());
    assert!(b.update_lookahead_exact(0, true).is_err());
    assert!(b.update_lookahead_assertion_named("x", 0).is_err());
}

#[test]
fn repeat_expr() {
    let mut b = GrammarBuilder::new();
    let body = b.add_empty_str();
    let rid = b.add_rule_named("a", body).unwrap();
    let rep = b.add_repeat(rid, 2, 5);
    b.add_rule_named("root", rep).unwrap();
    let g = b.get("root").unwrap();
    assert_eq!(g.expr(rep).kind, GrammarExprType::Repeat);
    assert_eq!(g.expr(rep).data, &[rid, 2, 5]);
}

#[test]
fn tag_dispatch_roundtrips() {
    let mut b = GrammarBuilder::new();
    let body = b.add_empty_str();
    let r0 = b.add_rule_named("handler", body).unwrap();
    let spec = TagDispatchSpec {
        tag_rule_pairs: vec![("<call>".to_string(), r0)],
        stop_eos: true,
        stop_str: vec![],
        loop_after_dispatch: false,
        excluded_str: vec![],
    };
    let td = b.add_tag_dispatch(&spec);
    b.add_rule_named("root", td).unwrap();
    let g = b.get("root").unwrap();
    let decoded = g.tag_dispatch(td);
    assert_eq!(decoded.tag_rule_pairs, vec![("<call>".to_string(), r0)]);
    assert!(decoded.stop_eos);
    assert!(!decoded.loop_after_dispatch);
}

#[test]
fn get_by_id_out_of_bounds() {
    let mut b = GrammarBuilder::new();
    let e = b.add_empty_str();
    b.add_rule_named("root", e).unwrap();
    assert!(b.get_by_id(5).is_err());
}

#[test]
fn get_by_id_negative() {
    let mut b = GrammarBuilder::new();
    let e = b.add_empty_str();
    b.add_rule_named("root", e).unwrap();
    assert!(b.get_by_id(-1).is_err());
}

#[test]
fn get_by_id_ok() {
    let mut b = GrammarBuilder::new();
    let e = b.add_empty_str();
    let rid = b.add_rule_named("root", e).unwrap();
    let g = b.get_by_id(rid).unwrap();
    assert_eq!(g.root_rule_id(), rid);
}
