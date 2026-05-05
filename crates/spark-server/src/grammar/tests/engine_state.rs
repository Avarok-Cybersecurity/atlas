// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn test_grammar_engine_creation() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32]; // <eos>
    let engine = GrammarEngine::new(&vocab, &stop_ids);
    assert!(engine.is_ok());
    let engine = engine.unwrap();
    assert_eq!(engine.vocab_size(), vocab.len());
}

#[test]
fn test_json_schema_compilation() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let schema = r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }"#;

    let result = engine.compile_json_schema(schema);
    assert!(
        result.is_ok(),
        "JSON schema compilation failed: {}",
        result.as_ref().err().unwrap()
    );
}

#[test]
fn test_builtin_json_compilation() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let result = engine.compile_json_grammar();
    assert!(
        result.is_ok(),
        "Builtin JSON compilation failed: {}",
        result.as_ref().err().unwrap()
    );
}

#[test]
fn test_grammar_state_basic_json() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let compiled = engine.compile_json_grammar().unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();

    // Grammar is not terminated at start.
    assert!(!state.is_terminated());

    // Fill bitmask — should constrain tokens.
    let has_constraint = state.fill_bitmask();
    // JSON must start with { or [ or " or digit etc.
    // Many tokens should be masked, so has_constraint should be true.
    assert!(has_constraint);

    // The '{' character (ASCII 123) should be allowed for JSON start.
    assert!(state.is_token_allowed(b'{' as u32));
}

#[test]
fn test_grammar_state_accept_and_terminate() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let compiled = engine.compile_json_grammar().unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();

    // Feed a minimal valid JSON: `{}`
    assert!(state.accept_token(b'{' as u32));
    assert!(state.accept_token(b'}' as u32));

    // After a complete JSON value, grammar should allow EOS.
    // Fill bitmask to check.
    state.fill_bitmask();
    // The stop token (130) should now be allowed.
    assert!(state.is_token_allowed(130));
}

#[test]
fn test_grammar_state_rollback() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let compiled = engine.compile_json_grammar().unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();

    // Accept `{` then `"` — start of a JSON object with a key.
    assert!(state.accept_token(b'{' as u32));
    assert!(state.accept_token(b'"' as u32));

    // Rollback 1 token (the `"`).
    state.rollback(1);

    // After rollback, we should be back to the state after `{`.
    // `}` should be allowed (empty object).
    state.fill_bitmask();
    assert!(state.is_token_allowed(b'}' as u32));
}

#[test]
fn test_apply_bitmask_to_logits() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let compiled = engine.compile_json_grammar().unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();

    state.fill_bitmask();

    // Create logits with uniform values.
    let mut logits = vec![1.0f32; engine.vocab_size()];
    state.apply_bitmask_to_logits(&mut logits);

    // '{' should not be masked (it starts valid JSON).
    assert!(logits[b'{' as usize].is_finite());
    // A control character like 0x01 should likely be masked.
    assert!(logits[1].is_infinite() && logits[1].is_sign_negative());
}
