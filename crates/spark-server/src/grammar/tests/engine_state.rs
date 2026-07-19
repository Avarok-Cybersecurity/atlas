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

/// #131 regression: a `json_schema` grammar must mask the VERY FIRST token,
/// not just tokens 2..N. The bug was that `prefill_*_step` sampled the first
/// decode token with plain `sample_token` (no grammar bitmask), so a leading
/// prose token (e.g. `H` of "Here", or the schema `name` "suggest") escaped
/// before the grammar's opening `{` and broke strict parsing. The fix
/// (`sample_first_token`) applies the bitmask from the initial state — this
/// asserts the invariant that fix relies on: at generation-start `{` is
/// grammar-legal while arbitrary leading letters are masked.
#[test]
fn test_json_schema_masks_leading_prose_token_at_start() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let schema = r#"{
        "type": "object",
        "properties": {
            "answer": {"type": "string"}
        },
        "required": ["answer"]
    }"#;
    let compiled = engine.compile_json_schema(schema).unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();

    // Initial-state bitmask must constrain the first token.
    assert!(
        state.fill_bitmask(),
        "json_schema grammar must constrain the first token"
    );

    // `{` is the only legal structural opener for an object — must be allowed.
    assert!(
        state.is_token_allowed(b'{' as u32),
        "'{{' must be grammar-legal as the first token"
    );

    // Leading prose tokens that previously LEAKED before `{` must be masked.
    // `H` = "Here" leak; `s` = the schema-name "suggest" leak from the issue.
    assert!(
        !state.is_token_allowed(b'H' as u32),
        "leading prose 'H' must be masked at generation-start (#131)"
    );
    assert!(
        !state.is_token_allowed(b's' as u32),
        "schema-name leak 's' (\"suggest\") must be masked at start (#131)"
    );
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

// ── Forced-token fast-path (xgrammar Tier 3b, Coalescence) ──────────────────

/// Count the allowed tokens in `state`'s current bitmask and return the
/// single allowed token id when exactly one is allowed. This is the
/// authoritative "what would the masked-sample path do" reference: with
/// all-but-one token masked to `-inf`, any sampler (greedy or stochastic)
/// returns that one surviving token.
fn single_allowed_token(state: &GrammarState, vocab_size: usize) -> Option<u32> {
    let mut found: Option<u32> = None;
    for id in 0..vocab_size as u32 {
        if state.is_token_allowed(id) {
            if found.is_some() {
                return None; // two or more allowed → genuine choice
            }
            found = Some(id);
        }
    }
    found
}

/// CORRECTNESS: `forced_token()` agrees with the masked-sample path at
/// every step. Whenever the grammar admits exactly one legal token, the
/// fast-path token must equal the token the sampler would have produced
/// from the all-but-one-masked logit vector; whenever two or more tokens
/// are legal, `forced_token()` must decline (`None`) so the caller
/// falls through to a real sample.
#[test]
fn test_forced_token_matches_masked_sample_path() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let vocab_size = engine.vocab_size();

    // JSON-schema grammar with a required literal key forces a run of
    // scaffolding bytes (`{`, then the `"name"` key spelling, `:` …).
    let schema = r#"{
        "type": "object",
        "properties": { "name": { "type": "string" } },
        "required": ["name"]
    }"#;
    let compiled = engine.compile_json_schema(schema).unwrap();
    let mut state = GrammarState::new(&compiled, vocab_size).unwrap();

    // Walk the grammar greedily for a bounded number of steps. At each
    // step, the fast-path verdict must agree with the bitmask cardinality.
    let mut saw_forced = false;
    for _ in 0..64 {
        if state.is_terminated() {
            break;
        }
        let has_constraint = state.fill_bitmask();
        assert!(has_constraint, "tool grammar always constrains");
        let reference = single_allowed_token(&state, vocab_size);
        let forced = state.forced_token().map(|t| t as u32);
        assert_eq!(
            forced, reference,
            "forced_token() must equal the single-allowed-token reference",
        );
        // Advance: take the forced token when forced, else the first
        // allowed token (any legal choice keeps the walk going).
        let next = match reference {
            Some(t) => {
                saw_forced = true;
                t
            }
            None => (0..vocab_size as u32)
                .find(|&id| state.is_token_allowed(id))
                .expect("non-terminated state has at least one allowed token"),
        };
        assert!(state.accept_token(next), "allowed token must be accepted");
    }
    // The required-key schema guarantees at least one forced step (the
    // literal key spelling) — otherwise the test isn't exercising the
    // fast-path at all.
    assert!(saw_forced, "expected at least one grammar-forced token");
}

/// A forced token, fed back through `accept_token`, lands the matcher in
/// exactly the state the normal sample-then-accept path would — so a
/// fast-path emit is interchangeable with a sampled emit.
#[test]
fn test_forced_token_accept_advances_identically() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let vocab_size = engine.vocab_size();

    let schema = r#"{
        "type": "object",
        "properties": { "name": { "type": "string" } },
        "required": ["name"]
    }"#;
    let compiled = engine.compile_json_schema(schema).unwrap();

    // Path A: the fast-path — emit `forced_token()`, then `accept_token`.
    let mut fast = GrammarState::new(&compiled, vocab_size).unwrap();
    // Path B: the normal path — `fill_bitmask`, then accept the (single)
    // surviving token. With every other token masked the sampler is
    // forced to this token, so this reproduces the masked-sample emit.
    let mut slow = GrammarState::new(&compiled, vocab_size).unwrap();

    for step in 0..32 {
        if fast.is_terminated() || slow.is_terminated() {
            break;
        }
        let Some(forced) = fast.forced_token() else {
            break; // first genuine choice — fast-path no longer applies
        };
        let forced = forced as u32;

        slow.fill_bitmask();
        let slow_tok = single_allowed_token(&slow, vocab_size)
            .expect("forced step on `fast` implies forced step on `slow`");
        assert_eq!(forced, slow_tok, "step {step}: same forced token");

        assert!(fast.accept_token(forced));
        assert!(slow.accept_token(slow_tok));
        // After accepting, both matchers must agree on the next mask.
        let f_constraint = fast.fill_bitmask();
        let s_constraint = slow.fill_bitmask();
        assert_eq!(f_constraint, s_constraint, "step {step}: mask parity");
        for id in 0..vocab_size as u32 {
            assert_eq!(
                fast.is_token_allowed(id),
                slow.is_token_allowed(id),
                "step {step}: token {id} allowed-bit parity",
            );
        }
        assert_eq!(fast.is_terminated(), slow.is_terminated());
    }
}

/// `forced_token()` declines once the matcher has terminated — symmetric
/// with `fill_bitmask()`'s terminated guard, so the fast-path never
/// emits past a completed grammar.
#[test]
fn test_forced_token_none_after_termination() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let compiled = engine.compile_json_grammar().unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();

    // Drive a complete value `{}` then the stop token to terminate.
    assert!(state.accept_token(b'{' as u32));
    assert!(state.accept_token(b'}' as u32));
    assert!(state.accept_token(130)); // <eos>
    assert!(state.is_terminated());

    assert_eq!(
        state.forced_token(),
        None,
        "forced_token must decline on a terminated matcher",
    );
}

/// A genuine multi-way choice is never reported as forced. After `{` a
/// JSON object may continue with `"` (a key) or `}` (empty object) — two
/// legal tokens — so `forced_token()` must return `None`.
#[test]
fn test_forced_token_none_on_genuine_choice() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let compiled = engine.compile_json_grammar().unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();

    assert!(state.accept_token(b'{' as u32));
    // Sanity: both continuations are genuinely legal.
    state.fill_bitmask();
    assert!(state.is_token_allowed(b'"' as u32));
    assert!(state.is_token_allowed(b'}' as u32));

    assert_eq!(
        state.forced_token(),
        None,
        "a two-way choice must not be reported as forced",
    );
}

/// #144 budget-aware graceful close: inside an OPEN JSON string value the
/// stop token is grammar-illegal — a length-stop here yields unparseable
/// output. `completion_token_ids` must return a grammar-legal close that,
/// once applied, makes the stop token legal (the output is parseable).
#[test]
fn test_budget_close_completes_open_json_string() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let schema = r#"{
        "type": "object",
        "properties": { "name": {"type": "string"} },
        "required": ["name"]
    }"#;
    let compiled = engine.compile_json_schema(schema).unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();

    // Drive the grammar to inside an open string value: `{"name":"ab`
    for ch in br#"{"name":"ab"# {
        assert!(
            state.accept_token(u32::from(*ch)),
            "byte {ch} must be grammar-legal while building the object",
        );
    }
    // Mid-string the response may NOT stop — EOS would truncate to invalid JSON.
    assert!(
        !state.stop_legal(&[130]),
        "EOS must be illegal inside an open JSON string",
    );

    // A bounded grammar-legal close exists.
    let close = state
        .completion_token_ids(32)
        .expect("a bounded close exists for an open string");
    assert!(!close.is_empty(), "a non-empty close is required here");

    // Applying the close makes stopping legal — the truncated output parses.
    for tok in &close {
        assert!(
            state.accept_token(*tok as u32),
            "every close token must be grammar-legal",
        );
    }
    assert!(
        state.stop_legal(&[130]),
        "after the close, EOS is legal (output is parseable)",
    );
}

/// When the grammar can already stop (a complete value), `stop_legal` is
/// true and the budget-close is empty — no spurious tokens are appended.
#[test]
fn test_budget_close_noop_when_already_stop_legal() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let compiled = engine.compile_json_grammar().unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();

    // `{}` is a complete JSON value.
    assert!(state.accept_token(b'{' as u32));
    assert!(state.accept_token(b'}' as u32));

    assert!(state.stop_legal(&[130]), "a complete value may stop");
    assert_eq!(
        state.completion_token_ids(32),
        Some(Vec::new()),
        "already stop-legal: the close is empty",
    );
}
