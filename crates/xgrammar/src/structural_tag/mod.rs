// SPDX-License-Identifier: AGPL-3.0-only
//
// Structural tag — pure-Rust port of `cpp/structural_tag.{cc,h}`.
//
// WHAT THIS IS
// ------------
// The structural tag is the tool-calling envelope mechanism. A model
// is constrained to emit a stream where, between *trigger* points,
// free text is allowed, and once a trigger prefix appears the model
// must produce a tagged body — `begin <schema-conforming content> end`
// — i.e. `<tool_call>{json}</tool_call>` style enforcement. Atlas's
// `grammar/compile_tools.rs` calls into this for every tool request.
//
// PIPELINE (mirrors `StructuralTagToGrammar`)
// -------------------------------------------
//   1. `parser`      — structural-tag JSON document -> `StructuralTag`
//                      AST (`Format` tree).
//   2. `analyzer`    — fills analyzer-only fields: detected end strings
//                      and the `is_unlimited` flags.
//   3. `converter`   — `Format` tree -> BNF rules via `GrammarBuilder`,
//                      reusing `src/schema` for embedded JSON schemas
//                      and `src/regex` for regex content.
//   4. normalize     — `GrammarNormalizer` runs as the final step.
//
// FAITHFULNESS / SIMPLIFICATIONS vs C++
// -------------------------------------
//  * C++ returns `Result<Grammar, StructuralTagError>` where the error
//    is a `std::variant`; we return `Result<_, StructuralTagError>`
//    with a two-variant enum carrying the same JSON-vs-tag distinction.
//  * C++ `RecursionGuard` aborts on overflow; we return an error.
//  * The C++ `StructuralTagAnalyzer` keeps a raw-pointer stack to find
//    the enclosing tag's end strings; we thread that list down the
//    recursion explicitly — equivalent, zero `unsafe`.
//  * Embedded `json_schema` sub-documents are re-serialized with an
//    order-preserving serializer (`json_serialize`) so JSON-Schema
//    property order survives, matching picojson `serialize(false)`.

mod analyzer;
mod converter;
mod converter_sep;
mod converter_tags;
mod error;
mod format;
mod json_serialize;
mod parser;
mod parser_tags;

#[cfg(test)]
mod tests;

pub use error::{StructuralTagError, StructuralTagResult};
pub use format::{Format, SchemaStyle, StructuralTag, StructuralTagItem, TagFormat};

use crate::grammar::GrammarData;
use crate::grammar::builder::TagDispatchSpec;

/// Build a [`TagDispatchSpec`] from the structural-tag fields. Shared
/// by the `any_text` and `triggered_tags` converters.
pub(crate) fn tag_dispatch_spec(
    tag_rule_pairs: Vec<(String, i32)>,
    stop_eos: bool,
    stop_str: Vec<String>,
    loop_after_dispatch: bool,
    excluded_str: Vec<String>,
) -> TagDispatchSpec {
    TagDispatchSpec {
        tag_rule_pairs,
        stop_eos,
        stop_str,
        loop_after_dispatch,
        excluded_str,
    }
}

/// Convert a structural-tag JSON document into a grammar.
///
/// Port of `StructuralTagToGrammar` / `Grammar::FromStructuralTag(json)`.
/// The document must be a `{"type": "structural_tag", "format": {...}}`
/// object (the `type` field is optional but checked when present).
pub fn structural_tag_to_grammar(structural_tag_json: &str) -> StructuralTagResult<GrammarData> {
    let mut structural_tag = parser::StructuralTagParser::from_json(structural_tag_json)?;
    analyzer::analyze(&mut structural_tag)?;
    converter::StructuralTagConverter::convert(&structural_tag)
}

/// Build a grammar from a list of [`StructuralTagItem`]s and trigger
/// prefixes — the legacy `Grammar::FromStructuralTag(tags, triggers)`
/// entry point.
///
/// Each item becomes a `tag` (`begin`, a `json_schema` body, `end`)
/// inside a single `triggered_tags` format whose triggers are
/// `triggers`. This is the exact shape Atlas's tool-call compiler
/// uses: one tag per tool, one trigger per opening marker.
///
/// Port of `StructuralTag::from_legacy_structural_tag` followed by
/// `StructuralTagToGrammar`.
pub fn structural_tag_from_items(
    tags: &[StructuralTagItem],
    triggers: &[String],
) -> StructuralTagResult<GrammarData> {
    let json = build_legacy_json(tags, triggers)?;
    structural_tag_to_grammar(&json)
}

/// Assemble the structural-tag JSON document for the legacy
/// `(tags, triggers)` form. Each tag's `schema` string is embedded
/// verbatim as the `json_schema` value (it must itself be valid JSON).
fn build_legacy_json(
    tags: &[StructuralTagItem],
    triggers: &[String],
) -> StructuralTagResult<String> {
    let mut tag_objs = Vec::with_capacity(tags.len());
    for item in tags {
        // Validate the schema parses as JSON before embedding it.
        crate::schema::JsonValue::parse(&item.schema).map_err(|e| {
            StructuralTagError::json(format!("structural tag item schema is not valid JSON: {e}"))
        })?;
        tag_objs.push(format!(
            r#"{{"type":"tag","begin":{},"content":{{"type":"json_schema","json_schema":{}}},"end":{}}}"#,
            json_string(&item.begin),
            item.schema,
            json_string(&item.end),
        ));
    }
    let trigger_strs: Vec<String> = triggers.iter().map(|t| json_string(t)).collect();
    Ok(format!(
        r#"{{"type":"structural_tag","format":{{"type":"triggered_tags","triggers":[{}],"tags":[{}]}}}}"#,
        trigger_strs.join(","),
        tag_objs.join(","),
    ))
}

/// Render `s` as a JSON string literal.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
