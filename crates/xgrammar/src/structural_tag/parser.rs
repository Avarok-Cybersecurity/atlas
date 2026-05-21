// SPDX-License-Identifier: AGPL-3.0-only
//
// Structural-tag JSON parser — port of `StructuralTagParser` from
// `cpp/structural_tag.cc`.
//
// Parses a structural-tag JSON document into the [`StructuralTag`] AST.
// The C++ uses picojson; here we parse with the crate's order-preserving
// [`JsonValue`] so embedded JSON schemas keep their key order.
//
// The basic-format parsers (`const_string`, `json_schema`, `any_text`,
// `grammar`, `regex`) live here; combinatorial formats (`sequence`,
// `or`, `tag`, `triggered_tags`, `tags_with_separator`) live in
// `parser_tags.rs` — split to keep each file under the 250-line cap.

use super::error::{StructuralTagError, StructuralTagResult};
use super::format::{Format, SchemaStyle, StructuralTag};
use super::json_serialize::serialize_compact;
use crate::schema::JsonValue;

/// Maximum format-nesting depth. Port of `RecursionGuard`'s
/// `kDefaultMaxRecursionDepth` (10000).
pub(super) const MAX_FORMAT_DEPTH: u32 = 10_000;

/// Recursive-descent parser for structural-tag documents.
pub(super) struct StructuralTagParser {
    pub(super) depth: u32,
}

impl StructuralTagParser {
    /// Parse a structural-tag JSON document.
    /// Port of `StructuralTagParser::FromJSON`.
    pub(super) fn from_json(json: &str) -> StructuralTagResult<StructuralTag> {
        let value = JsonValue::parse(json)
            .map_err(|e| StructuralTagError::json(format!("Failed to parse JSON: {e}")))?;
        StructuralTagParser { depth: 0 }.parse_structural_tag(&value)
    }

    /// Port of `StructuralTagParser::ParseStructuralTag`.
    fn parse_structural_tag(&mut self, value: &JsonValue) -> StructuralTagResult<StructuralTag> {
        let obj = value
            .as_object()
            .ok_or_else(|| StructuralTagError::invalid("Structural tag must be an object"))?;
        // The type field is optional but must be "structural_tag" if present.
        if let Some(t) = find(obj, "type")
            && t.as_str() != Some("structural_tag")
        {
            return Err(StructuralTagError::invalid(
                "Structural tag's type must be a string \"structural_tag\"",
            ));
        }
        // The format field is required.
        let format_val = find(obj, "format").ok_or_else(|| {
            StructuralTagError::invalid("Structural tag must have a format field")
        })?;
        let format = self.parse_format(format_val)?;
        Ok(StructuralTag { format })
    }

    /// Parse a Format object. The `type` field is checked here.
    /// Port of `StructuralTagParser::ParseFormat`.
    pub(super) fn parse_format(&mut self, value: &JsonValue) -> StructuralTagResult<Format> {
        self.depth += 1;
        if self.depth > MAX_FORMAT_DEPTH {
            return Err(StructuralTagError::invalid("Format nesting too deep"));
        }
        let result = self.parse_format_inner(value);
        self.depth -= 1;
        result
    }

    fn parse_format_inner(&mut self, value: &JsonValue) -> StructuralTagResult<Format> {
        let obj = value
            .as_object()
            .ok_or_else(|| StructuralTagError::invalid("Format must be an object"))?;

        // If type is present, use it to determine the format.
        if let Some(type_val) = find(obj, "type") {
            let type_str = type_val
                .as_str()
                .ok_or_else(|| StructuralTagError::invalid("Format's type must be a string"))?;
            return match type_str {
                "const_string" => parse_const_string(obj),
                "json_schema" => parse_json_schema(obj, None),
                "any_text" => parse_any_text(obj),
                "sequence" => self.parse_sequence(obj),
                "or" => self.parse_or(obj),
                "tag" => Ok(Format::Tag(self.parse_tag_obj(obj)?)),
                "triggered_tags" => self.parse_triggered_tags(obj),
                "tags_with_separator" => self.parse_tags_with_separator(obj),
                "qwen_xml_parameter" => parse_json_schema(obj, Some(SchemaStyle::QwenXml)),
                "grammar" => parse_grammar(obj),
                "regex" => parse_regex(obj),
                other => Err(StructuralTagError::invalid(format!(
                    "Format type not recognized: {other}"
                ))),
            };
        }

        // If type is not present, try every format type. Tag is prioritized.
        if let Ok(tag) = self.parse_tag_obj(obj) {
            return Ok(Format::Tag(tag));
        }
        if let Ok(f) = parse_const_string(obj) {
            return Ok(f);
        }
        if let Ok(f) = parse_json_schema(obj, None) {
            return Ok(f);
        }
        if let Ok(f) = parse_any_text(obj) {
            return Ok(f);
        }
        if let Ok(f) = self.parse_sequence(obj) {
            return Ok(f);
        }
        if let Ok(f) = self.parse_or(obj) {
            return Ok(f);
        }
        if let Ok(f) = self.parse_triggered_tags(obj) {
            return Ok(f);
        }
        if let Ok(f) = self.parse_tags_with_separator(obj) {
            return Ok(f);
        }
        Err(StructuralTagError::invalid("Invalid format"))
    }
}

/// Look up `key` in an ordered object slice.
pub(super) fn find<'a>(obj: &'a [(String, JsonValue)], key: &str) -> Option<&'a JsonValue> {
    obj.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

/// Port of `ParseConstStringFormat`.
fn parse_const_string(obj: &[(String, JsonValue)]) -> StructuralTagResult<Format> {
    let value = find(obj, "value")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| {
            StructuralTagError::invalid("ConstString format must have a value field with a string")
        })?;
    Ok(Format::ConstString(value.to_string()))
}

/// Port of `ParseJSONSchemaFormat`.
fn parse_json_schema(
    obj: &[(String, JsonValue)],
    style_override: Option<SchemaStyle>,
) -> StructuralTagResult<Format> {
    let schema_val = find(obj, "json_schema").ok_or_else(|| {
        StructuralTagError::invalid(
            "JSON schema format must have a json_schema field with a object or boolean value",
        )
    })?;
    if !(schema_val.is_object() || schema_val.as_bool().is_some()) {
        return Err(StructuralTagError::invalid(
            "JSON schema format must have a json_schema field with a object or boolean value",
        ));
    }
    let style = match style_override {
        Some(s) => s,
        None => match find(obj, "style").and_then(JsonValue::as_str) {
            None => SchemaStyle::Json,
            Some("json") => SchemaStyle::Json,
            Some("qwen_xml") => SchemaStyle::QwenXml,
            Some("minimax_xml") => SchemaStyle::MiniMaxXml,
            Some("deepseek_xml") => SchemaStyle::DeepSeekXml,
            Some(_) => {
                return Err(StructuralTagError::invalid(
                    "style must be \"json\", \"qwen_xml\", \"minimax_xml\", or \"deepseek_xml\"",
                ));
            }
        },
    };
    Ok(Format::JsonSchema {
        json_schema: serialize_compact(schema_val),
        style,
    })
}

/// Port of `ParseAnyTextFormat`.
fn parse_any_text(obj: &[(String, JsonValue)]) -> StructuralTagResult<Format> {
    let excludes_val = match find(obj, "excludes") {
        None => {
            if find(obj, "type").is_none() {
                return Err(StructuralTagError::invalid(
                    "Any text format should not have any fields other than type",
                ));
            }
            return Ok(Format::AnyText {
                excludes: Vec::new(),
                detected_end_strs: Vec::new(),
            });
        }
        Some(v) => v,
    };
    let array = excludes_val.as_array().ok_or_else(|| {
        StructuralTagError::invalid("AnyText format's excluded_strs field must be an array")
    })?;
    let mut excludes = Vec::with_capacity(array.len());
    for item in array {
        let s = item.as_str().ok_or_else(|| {
            StructuralTagError::invalid("AnyText format's excluded_strs array must contain strings")
        })?;
        excludes.push(s.to_string());
    }
    Ok(Format::AnyText {
        excludes,
        detected_end_strs: Vec::new(),
    })
}

/// Port of `ParseGrammarFormat`.
fn parse_grammar(obj: &[(String, JsonValue)]) -> StructuralTagResult<Format> {
    let grammar = find(obj, "grammar")
        .and_then(JsonValue::as_str)
        .filter(|s| !s.is_empty());
    match grammar {
        Some(g) => Ok(Format::Grammar(g.to_string())),
        None => Err(StructuralTagError::invalid(
            "Grammar format must have a grammar field with a non-empty string",
        )),
    }
}

/// Port of `ParseRegexFormat`.
fn parse_regex(obj: &[(String, JsonValue)]) -> StructuralTagResult<Format> {
    let pattern = find(obj, "pattern")
        .and_then(JsonValue::as_str)
        .filter(|s| !s.is_empty());
    match pattern {
        Some(p) => Ok(Format::Regex(p.to_string())),
        None => Err(StructuralTagError::invalid(
            "Regex format must have a pattern field with a non-empty string",
        )),
    }
}
