// SPDX-License-Identifier: AGPL-3.0-only
//
// Structural-tag format AST — port of the `Format` variant family and
// `StructuralTag` from `cpp/structural_tag.h`.
//
// The C++ uses a `std::variant` of ten `*Format` structs plus several
// `private:` analyzer-only fields (`detected_end_strs_`, `is_unlimited_`).
// Here that is a single `Format` enum; the analyzer-only fields are
// public-in-crate so the analyzer pass (`analyzer.rs`) can fill them in,
// exactly as the C++ `friend class StructuralTagAnalyzer` does.

/// The schema style used to convert a [`Format::JsonSchema`] into EBNF.
/// Port of `JSONSchemaFormat::style` (`"json"`, `"qwen_xml"`,
/// `"minimax_xml"`, `"deepseek_xml"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaStyle {
    /// Plain JSON schema (`JSONSchemaToEBNF`).
    Json,
    /// Qwen XML tool-calling style.
    QwenXml,
    /// MiniMax XML tool-calling style.
    MiniMaxXml,
    /// DeepSeek XML tool-calling style.
    DeepSeekXml,
}

/// One `begin content end` tag. Port of `TagFormat`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagFormat {
    /// The begin marker, e.g. `<tool_call>`.
    pub begin: String,
    /// The tag body format.
    pub content: Box<Format>,
    /// Accepted end markers. Multiple markers => any of them ends the
    /// tag. May be emptied by the analyzer for unlimited content.
    pub end: Vec<String>,
}

/// A structural-tag format node. Port of the `Format` `std::variant`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Format {
    /// A fixed literal string. Port of `ConstStringFormat`.
    ConstString(String),
    /// Content conforming to a JSON schema (in the given style).
    /// Port of `JSONSchemaFormat`.
    JsonSchema {
        /// The JSON-schema document, serialized as a JSON string.
        json_schema: String,
        /// Conversion style.
        style: SchemaStyle,
    },
    /// Arbitrary text, optionally excluding some substrings. Port of
    /// `AnyTextFormat`.
    AnyText {
        /// Substrings that must not appear in the matched text.
        excludes: Vec<String>,
        /// End strings detected by the analyzer from an enclosing tag.
        detected_end_strs: Vec<String>,
    },
    /// Content conforming to a raw EBNF grammar. Port of `GrammarFormat`.
    Grammar(String),
    /// Content conforming to a regex pattern. Port of `RegexFormat`.
    Regex(String),
    /// An ordered concatenation of sub-formats. Port of `SequenceFormat`.
    Sequence {
        /// The element formats.
        elements: Vec<Format>,
        /// Set by the analyzer: true if the last element is unlimited.
        is_unlimited: bool,
    },
    /// An alternation of sub-formats. Port of `OrFormat`.
    Or {
        /// The alternative formats.
        elements: Vec<Format>,
        /// Set by the analyzer: true if (all) elements are unlimited.
        is_unlimited: bool,
    },
    /// A single tag. Port of `TagFormat`.
    Tag(TagFormat),
    /// Triggered tag dispatch — the tool-call envelope mechanism.
    /// Port of `TriggeredTagsFormat`.
    TriggeredTags {
        /// Trigger prefixes; each tag's `begin` starts with one trigger.
        triggers: Vec<String>,
        /// The tags dispatched to.
        tags: Vec<TagFormat>,
        /// Substrings excluded from inter-tag free text.
        excludes: Vec<String>,
        /// If true, at least one tag must be produced.
        at_least_one: bool,
        /// If true, generation stops after the first tag.
        stop_after_first: bool,
        /// End strings detected by the analyzer from an enclosing tag.
        detected_end_strs: Vec<String>,
    },
    /// Tags joined by a separator. Port of `TagsWithSeparatorFormat`.
    TagsWithSeparator {
        /// The tags.
        tags: Vec<TagFormat>,
        /// The separator string between tags.
        separator: String,
        /// If true, at least one tag must be produced.
        at_least_one: bool,
        /// If true, generation stops after the first tag.
        stop_after_first: bool,
        /// End strings detected by the analyzer from an enclosing tag.
        detected_end_strs: Vec<String>,
    },
}

/// The top-level structural tag. Port of `StructuralTag`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuralTag {
    /// The root format.
    pub format: Format,
}

/// A single legacy structural-tag item: `begin`, a JSON `schema`
/// string, and `end`. Port of the public `StructuralTagItem` struct
/// from `include/xgrammar/grammar.h`.
///
/// This is the tool-calling envelope descriptor: each item says
/// "a tool call starts with `begin`, its arguments conform to
/// `schema`, and it ends with `end`".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuralTagItem {
    /// The begin marker of the tag, e.g. `<function=get_weather>`.
    pub begin: String,
    /// The JSON-schema document (as a string) for the tag body.
    pub schema: String,
    /// The end marker of the tag, e.g. `</function>`.
    pub end: String,
}

impl StructuralTagItem {
    /// Construct a [`StructuralTagItem`].
    pub fn new(
        begin: impl Into<String>,
        schema: impl Into<String>,
        end: impl Into<String>,
    ) -> Self {
        Self {
            begin: begin.into(),
            schema: schema.into(),
            end: end.into(),
        }
    }
}
