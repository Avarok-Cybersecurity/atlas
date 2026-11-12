// SPDX-License-Identifier: AGPL-3.0-only

//! `json_schema` and `grammar` on `/completion`: constrained output.
//!
//! The scheduler owns the XGrammar engine and, when a constraint fails to
//! compile, logs a warning and decodes unconstrained. That is a fine fallback
//! for a chat `response_format` hint, and a wrong one for a client that asked
//! llama.cpp for a grammar: it would get free text with HTTP 200. So the
//! constraint is parsed with XGrammar's own parser HERE, before the request is
//! queued, and a constraint that cannot be enforced is a 400.
//!
//! Two checks cannot be made at the HTTP layer and are the residual risk,
//! stated in `docs/llama-cpp-compat.md`: the engine itself failing to
//! initialize (covered by `AppState::grammar_available`, checked here), and a
//! tokenizer-dependent compile failure after the text parsed.

use serde_json::{Map, Value};

use super::error::CompatError;

/// A constraint on the generated text.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Constraint {
    /// A JSON Schema, as the JSON text XGrammar compiles.
    JsonSchema(String),
    /// A GBNF/EBNF grammar whose start rule is `root`.
    Ebnf(String),
}

impl Constraint {
    /// The text `generation_settings.grammar` reports.
    pub(crate) fn grammar_text(&self) -> &str {
        match self {
            Constraint::JsonSchema(s) | Constraint::Ebnf(s) => s,
        }
    }
}

fn present<'a>(body: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    body.get(key).filter(|v| !v.is_null())
}

/// Read `json_schema` / `grammar` from the request body.
///
/// Returns `Ok(None)` when the request is unconstrained. Fails, with HTTP 400,
/// when both are given, when the engine is unavailable, or when the text does
/// not parse.
pub(crate) fn parse_constraint(
    body: &Map<String, Value>,
    grammar_available: bool,
) -> Result<Option<Constraint>, CompatError> {
    // Lazy grammars (applied only after a trigger) change what is constrained
    // and when; Atlas applies a grammar from the first token.
    if matches!(present(body, "grammar_lazy"), Some(Value::Bool(true))) {
        return Err(CompatError::unsupported(
            "grammar_lazy",
            "grammars are enforced from the first token; lazy grammars have no equivalent",
        ));
    }
    if let Some(Value::Array(t)) = present(body, "grammar_triggers")
        && !t.is_empty()
    {
        return Err(CompatError::unsupported(
            "grammar_triggers",
            "triggers belong to lazy grammars, which Atlas does not implement",
        ));
    }

    let schema = present(body, "json_schema");
    let grammar = match present(body, "grammar") {
        None => None,
        Some(Value::String(s)) if s.is_empty() => None,
        Some(Value::String(s)) => Some(s.as_str()),
        Some(_) => return Err(CompatError::invalid("Field 'grammar': must be a string")),
    };
    let constraint = match (schema, grammar) {
        (None, None) => return Ok(None),
        (Some(_), Some(_)) => {
            return Err(CompatError::invalid(
                "Cannot use both json_schema and grammar; send one of them",
            ));
        }
        (Some(schema), None) => Constraint::JsonSchema(schema_text(schema)?),
        (None, Some(g)) => Constraint::Ebnf(g.to_string()),
    };

    if !grammar_available {
        return Err(CompatError::unsupported(
            match constraint {
                Constraint::JsonSchema(_) => "json_schema",
                Constraint::Ebnf(_) => "grammar",
            },
            "constrained decoding is unavailable on this server (the grammar engine \
             did not initialize at startup), so the output could not be enforced",
        ));
    }
    match &constraint {
        Constraint::JsonSchema(s) => check_schema(s)
            .map_err(|e| CompatError::invalid(format!("Field 'json_schema': {e}")))?,
        Constraint::Ebnf(g) => {
            check_ebnf(g).map_err(|e| CompatError::invalid(format!("Field 'grammar': {e}")))?
        }
    }
    Ok(Some(constraint))
}

/// The schema as JSON text. llama.cpp reads an empty object as "any object".
fn schema_text(schema: &Value) -> Result<String, CompatError> {
    match schema {
        Value::Object(m) if m.is_empty() => Ok(r#"{"type":"object"}"#.to_string()),
        Value::Object(_) | Value::Bool(_) => Ok(schema.to_string()),
        _ => Err(CompatError::invalid(
            "Field 'json_schema': must be a JSON Schema object",
        )),
    }
}

/// Parse a JSON Schema with the converter the scheduler will compile with
/// (same options as `GrammarEngine::compile_json_schema`).
fn check_schema(schema: &str) -> Result<(), String> {
    xgrammar::Grammar::from_json_schema(
        schema,
        true,
        None,
        None::<(&str, &str)>,
        true,
        Some(8),
        false,
    )
    .map(|_| ())
}

/// Parse a grammar with the EBNF parser the scheduler will compile with.
fn check_ebnf(grammar: &str) -> Result<(), String> {
    xgrammar::Grammar::from_ebnf(grammar, "root").map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(v: Value, available: bool) -> Result<Option<Constraint>, CompatError> {
        parse_constraint(v.as_object().expect("object"), available)
    }

    #[test]
    fn unconstrained_requests_pass_through() {
        assert_eq!(parse(json!({"prompt": "x"}), true).unwrap(), None);
        // null means "absent" in llama.cpp, and so does an empty grammar.
        assert_eq!(
            parse(json!({"json_schema": null, "grammar": ""}), true).unwrap(),
            None
        );
    }

    #[test]
    fn a_json_schema_becomes_a_schema_constraint() {
        let c = parse(
            json!({"json_schema": {"type": "object", "properties": {"a": {"type": "integer"}}}}),
            true,
        )
        .unwrap()
        .expect("constraint");
        match c {
            Constraint::JsonSchema(s) => assert!(s.contains("\"integer\"")),
            other => panic!("expected a schema, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_schema_means_any_object() {
        let c = parse(json!({"json_schema": {}}), true).unwrap().unwrap();
        assert_eq!(c, Constraint::JsonSchema(r#"{"type":"object"}"#.into()));
    }

    #[test]
    fn a_gbnf_grammar_becomes_an_ebnf_constraint() {
        let g = r#"root ::= "yes" | "no""#;
        let c = parse(json!({"grammar": g}), true).unwrap().unwrap();
        assert_eq!(c, Constraint::Ebnf(g.to_string()));
        assert_eq!(c.grammar_text(), g);
    }

    #[test]
    fn a_grammar_that_does_not_parse_is_a_400_not_unconstrained_output() {
        let err = parse(json!({"grammar": "root ::= ("}), true).expect_err("must reject");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("grammar"), "{}", err.message);
        // A grammar with no `root` rule has no start symbol.
        assert!(parse(json!({"grammar": r#"start ::= "a""#}), true).is_err());
    }

    #[test]
    fn a_schema_that_does_not_convert_is_a_400() {
        let err = parse(
            json!({"json_schema": {"type": "object", "properties": {"a": {"$ref": "#/nope"}}}}),
            true,
        )
        .expect_err("dangling $ref");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("json_schema"));
        assert!(parse(json!({"json_schema": "not a schema"}), true).is_err());
        assert!(parse(json!({"json_schema": 7}), true).is_err());
    }

    #[test]
    fn both_json_schema_and_grammar_is_a_400() {
        let err = parse(
            json!({"json_schema": {}, "grammar": r#"root ::= "a""#}),
            true,
        )
        .expect_err("ambiguous");
        assert!(err.message.contains("both"));
    }

    #[test]
    fn no_grammar_engine_means_no_constrained_output_at_all() {
        // The case this module exists for: engine down, client asked for a
        // constraint. Unconstrained 200 would be silently wrong.
        let err =
            parse(json!({"json_schema": {"type": "object"}}), false).expect_err("cannot enforce");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("json_schema"));
        assert!(err.message.contains("unavailable"));
        let err = parse(json!({"grammar": r#"root ::= "a""#}), false).expect_err("cannot enforce");
        assert!(err.message.contains("grammar"));
        // An unconstrained request is unaffected by the engine being down.
        assert_eq!(parse(json!({"prompt": "x"}), false).unwrap(), None);
    }

    #[test]
    fn lazy_grammars_are_refused() {
        let g = r#"root ::= "a""#;
        assert!(parse(json!({"grammar": g, "grammar_lazy": true}), true).is_err());
        assert!(parse(json!({"grammar": g, "grammar_lazy": false}), true).is_ok());
        assert!(
            parse(
                json!({"grammar": g, "grammar_triggers": [{"type": 0, "value": "x"}]}),
                true
            )
            .is_err()
        );
        assert!(parse(json!({"grammar": g, "grammar_triggers": []}), true).is_ok());
    }
}
