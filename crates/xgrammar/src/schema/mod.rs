// SPDX-License-Identifier: AGPL-3.0-only
//
// JSON-schema -> EBNF converter — port of
// `cpp/json_schema_converter.{h,cc}` and
// `cpp/json_schema_converter_ext.{h,cc}`.
//
// Converts a JSON Schema document into an EBNF grammar string (and,
// via the existing EBNF parser, into a `GrammarData` AST). It is the
// most behaviour-critical subsystem for tool calling, since tool
// parameters are JSON schemas.
//
// PIPELINE
// --------
//   1. `json_value`   — order-preserving JSON parse of the document.
//   2. `parser*`      — JSON Schema -> `SchemaSpec` intermediate
//                       representation, with sub-schema dedup and
//                       `$ref` resolution.
//   3. `converter` + `gen_*` — `SchemaSpec` -> EBNF rule script.
//   4. `api`          — public entry points.
//
// FAITHFULNESS / SIMPLIFICATIONS vs C++
// -------------------------------------
//  * C++ aborts the process (`XGRAMMAR_LOG(FATAL)`) on a malformed
//    schema or unresolvable `$ref`; we return `Err(SchemaError)` —
//    no panics, no `unsafe`.
//  * C++ emits stderr warnings for unsupported keywords (`not`,
//    `if`/`then`/`else`, `multipleOf`, `uniqueItems`, ...). We drop
//    those warnings silently — keeping the converter pure — but the
//    keywords are still ignored exactly as upstream ignores them.
//  * `allOf` with multiple sub-schemas degrades to "any", matching
//    the upstream "support is still ongoing" warning path.
//  * Object key ordering is preserved with a dedicated ordered JSON
//    value type (`json_value`), because `serde_json::Map` only keeps
//    insertion order under the `preserve_order` feature, which is not
//    enabled in this crate's `Cargo.toml`.

mod api;
mod cache_key;
mod converter;
mod error;
mod float_regex;
mod formats;
mod gen_array;
mod gen_composite;
mod gen_object;
mod gen_object_props;
mod gen_object_props_constrained;
mod gen_scalar;
mod indent;
mod json_value;
mod options;
mod parser;
mod parser_collections;
mod parser_composite;
mod range_regex;
mod script;
mod spec;

#[cfg(test)]
mod tests;

pub use api::{
    builtin_json_grammar, builtin_json_grammar_ebnf, deepseek_xml_tool_calling_to_ebnf,
    json_schema_to_ebnf, json_schema_to_grammar, json_value_to_ebnf,
    minimax_xml_tool_calling_to_ebnf, qwen_xml_tool_calling_to_ebnf,
};
pub use error::{SchemaError, SchemaErrorKind, SchemaResult};
pub use float_regex::generate_float_range_regex;
pub use json_value::JsonValue;
pub use options::{JsonFormat, SchemaConverterOptions};
pub use range_regex::generate_range_regex;
