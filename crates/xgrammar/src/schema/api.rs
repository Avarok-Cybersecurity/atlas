// SPDX-License-Identifier: AGPL-3.0-only
//
// Public API for the JSON-schema -> EBNF converter — port of the
// `JSONSchemaToEBNF` / `*XMLToolCallingToEBNF` free functions from
// `cpp/json_schema_converter.{h,cc}`.

use super::converter::JsonSchemaConverter;
use super::error::{SchemaError, SchemaResult};
use super::json_value::JsonValue;
use super::options::{JsonFormat, SchemaConverterOptions};
use super::parser::SchemaParser;
use crate::grammar::{GrammarData, parse_ebnf_default};

/// Convert a JSON Schema document (already parsed into a [`JsonValue`])
/// into an EBNF grammar string. Port of the `picojson::value`
/// overload of `JSONSchemaToEBNF`.
pub fn json_value_to_ebnf(
    schema: &JsonValue,
    options: &SchemaConverterOptions,
) -> SchemaResult<String> {
    let mut parser = SchemaParser::new(schema.clone(), options.strict_mode);
    let spec = parser.parse(schema, "root", None)?;
    let mut converter = JsonSchemaConverter::new(options, &mut parser);
    converter.convert(&spec)
}

/// Convert a JSON Schema *string* into an EBNF grammar string. Port
/// of the `std::string` overload of `JSONSchemaToEBNF`.
pub fn json_schema_to_ebnf(schema: &str, options: &SchemaConverterOptions) -> SchemaResult<String> {
    let value = JsonValue::parse(schema)
        .map_err(|e| SchemaError::invalid(format!("Failed to parse JSON schema: {e}")))?;
    json_value_to_ebnf(&value, options)
}

/// Convert a JSON Schema string into a parsed [`GrammarData`] AST by
/// generating the EBNF and feeding it through the EBNF parser.
pub fn json_schema_to_grammar(
    schema: &str,
    options: &SchemaConverterOptions,
) -> SchemaResult<GrammarData> {
    let ebnf = json_schema_to_ebnf(schema, options)?;
    parse_ebnf_default(&ebnf)
        .map_err(|e| SchemaError::invalid(format!("generated EBNF failed to parse: {e}")))
}

/// Validate that `schema` is a tool-calling object schema and convert
/// it with the given XML `format`. Shared by the three XML helpers.
fn xml_tool_calling_to_ebnf(schema: &str, format: JsonFormat) -> SchemaResult<String> {
    let value = JsonValue::parse(schema)
        .map_err(|e| SchemaError::invalid(format!("Failed to parse JSON schema: {e}")))?;
    if value.as_bool().is_some() {
        return Err(SchemaError::invalid(
            "Expected JSON schema object, got boolean",
        ));
    }
    match value.get("type").and_then(JsonValue::as_str) {
        Some("object") => {}
        _ => {
            return Err(SchemaError::invalid(
                "Function calling must have a 'type' field of 'object'",
            ));
        }
    }
    let options = SchemaConverterOptions {
        json_format: format,
        ..SchemaConverterOptions::default()
    };
    json_value_to_ebnf(&value, &options)
}

/// Convert a function-call parameter schema into a Qwen-XML-style
/// EBNF grammar. Port of `QwenXMLToolCallingToEBNF`.
pub fn qwen_xml_tool_calling_to_ebnf(schema: &str) -> SchemaResult<String> {
    xml_tool_calling_to_ebnf(schema, JsonFormat::QwenXml)
}

/// Convert a function-call parameter schema into a MiniMax-XML-style
/// EBNF grammar. Port of `MiniMaxXMLToolCallingToEBNF`.
pub fn minimax_xml_tool_calling_to_ebnf(schema: &str) -> SchemaResult<String> {
    xml_tool_calling_to_ebnf(schema, JsonFormat::MiniMaxXml)
}

/// Convert a function-call parameter schema into a DeepSeek-XML-style
/// EBNF grammar. Port of `DeepSeekXMLToolCallingToEBNF`.
pub fn deepseek_xml_tool_calling_to_ebnf(schema: &str) -> SchemaResult<String> {
    xml_tool_calling_to_ebnf(schema, JsonFormat::DeepSeekXml)
}

/// Build the EBNF grammar for the "any JSON value" schema (`{}`).
///
/// This is the schema-converter analogue of `compile_builtin_json
/// grammar`: an empty schema accepts any JSON document, so its
/// generated grammar is the builtin-JSON grammar surface for this
/// subsystem.
pub fn builtin_json_grammar_ebnf() -> String {
    // `{}` is always a valid schema and never fails to convert.
    json_schema_to_ebnf("{}", &SchemaConverterOptions::default()).unwrap_or_default()
}

/// Build the parsed [`GrammarData`] for the builtin-JSON grammar.
pub fn builtin_json_grammar() -> SchemaResult<GrammarData> {
    json_schema_to_grammar("{}", &SchemaConverterOptions::default())
}
