// SPDX-License-Identifier: AGPL-3.0-only
//
// Conversion options and the root output format — ports of the
// `JSONFormat` enum and the `JSONSchemaToEBNF` keyword arguments from
// `cpp/json_schema_converter.h`.

/// Root output format. `Json` produces a fully JSON-style grammar;
/// the XML variants produce an XML-style root object with JSON-style
/// inner values (used for tool calling). Mirrors C++ `JSONFormat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JsonFormat {
    /// Standard JSON grammar.
    Json,
    /// Qwen XML tool-calling style: `<parameter=name>value</parameter>`.
    QwenXml,
    /// MiniMax XML tool-calling style.
    MiniMaxXml,
    /// DeepSeek XML tool-calling style.
    DeepSeekXml,
}

impl JsonFormat {
    /// Whether this is one of the XML tool-calling formats.
    pub fn is_xml(self) -> bool {
        !matches!(self, JsonFormat::Json)
    }
}

/// Options controlling JSON-schema -> EBNF conversion. Field meanings
/// match the `JSONSchemaToEBNF` parameters in the C++ header.
#[derive(Debug, Clone)]
pub struct SchemaConverterOptions {
    /// Allow any whitespace between JSON tokens instead of enforcing
    /// the indentation/separator restrictions. Default `true`.
    pub any_whitespace: bool,
    /// Number of spaces per indent level. `None` => single-line.
    pub indent: Option<i32>,
    /// `(comma, colon)` separators. `None` => derived defaults.
    pub separators: Option<(String, String)>,
    /// Strict mode disallows properties/items not named in the schema
    /// (equivalent to `unevaluatedProperties:false` /
    /// `unevaluatedItems:false`). Default `true`.
    pub strict_mode: bool,
    /// Cap on whitespace characters when `any_whitespace` is set.
    /// `None` => unlimited.
    pub max_whitespace_cnt: Option<i32>,
    /// Root output format.
    pub json_format: JsonFormat,
}

impl Default for SchemaConverterOptions {
    fn default() -> Self {
        SchemaConverterOptions {
            any_whitespace: true,
            indent: None,
            separators: None,
            strict_mode: true,
            max_whitespace_cnt: None,
            json_format: JsonFormat::Json,
        }
    }
}

impl SchemaConverterOptions {
    /// Resolve the `(comma, colon)` separator pair, applying the same
    /// defaulting logic as the C++ `JSONSchemaConverter` constructor.
    pub fn resolved_separators(&self) -> (String, String) {
        if let Some((c, col)) = &self.separators {
            return (c.clone(), col.clone());
        }
        // No space after the comma when whitespace is flexible or an
        // explicit indent is set (the indent supplies the separation).
        let comma = if self.any_whitespace || self.indent.is_some() {
            ","
        } else {
            ", "
        };
        let colon = if self.any_whitespace { ":" } else { ": " };
        (comma.to_string(), colon.to_string())
    }
}
