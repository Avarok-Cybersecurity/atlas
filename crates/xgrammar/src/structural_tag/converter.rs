// SPDX-License-Identifier: AGPL-3.0-only
//
// Structural-tag → grammar converter — port of
// `StructuralTagGrammarConverter` from `cpp/structural_tag.cc`.
//
// Walks the analyzed [`Format`] tree and emits BNF rules into a
// [`GrammarBuilder`], returning the rule id of each format. The
// top-level `convert` adds a `root` rule referencing the result and
// runs the grammar normalizer, exactly as the C++ does.
//
// Basic + sequence/or conversion lives here; the tag-family conversion
// (`Tag`, `TriggeredTags`, `TagsWithSeparator`) lives in
// `converter_tags.rs` — split to keep each file under the 250-line cap.

use super::error::{StructuralTagError, StructuralTagResult};
use super::format::{Format, SchemaStyle, StructuralTag};
use crate::grammar::functor::{GrammarNormalizer, add_sub_grammar};
use crate::grammar::{GrammarBuilder, GrammarData, parse_ebnf_default};
use crate::regex::regex_to_ebnf;
use crate::schema::{
    SchemaConverterOptions, deepseek_xml_tool_calling_to_ebnf, json_schema_to_ebnf,
    minimax_xml_tool_calling_to_ebnf, qwen_xml_tool_calling_to_ebnf,
};

/// Converts an analyzed structural tag into a [`GrammarData`].
pub(super) struct StructuralTagConverter {
    pub(super) builder: GrammarBuilder,
}

impl StructuralTagConverter {
    /// Convert a fully-analyzed structural tag into a normalized
    /// grammar. Port of `StructuralTagGrammarConverter::Convert` plus
    /// the `GrammarNormalizer::Apply` at the end of `StructuralTagToGrammar`.
    pub(super) fn convert(structural_tag: &StructuralTag) -> StructuralTagResult<GrammarData> {
        let mut converter = StructuralTagConverter {
            builder: GrammarBuilder::new(),
        };
        let root_ref = converter.visit(&structural_tag.format)?;
        let grammar = converter.add_root_rule(root_ref)?;
        Ok(GrammarNormalizer::apply(grammar))
    }

    /// Add the top-level `root` rule referencing `ref_rule_id`.
    /// Port of `AddRootRuleAndGetGrammar`.
    fn add_root_rule(mut self, ref_rule_id: i32) -> StructuralTagResult<GrammarData> {
        let expr = self.builder.add_rule_ref(ref_rule_id);
        let seq = self.builder.add_sequence(&[expr]);
        let choices = self.builder.add_choices(&[seq]);
        let root_id = self
            .builder
            .add_rule_with_hint("root", choices)
            .map_err(|e| StructuralTagError::invalid(e.to_string()))?;
        self.builder
            .get_by_id(root_id)
            .map_err(|e| StructuralTagError::invalid(e.to_string()))
    }

    /// Visit a format, returning the rule id of the rule added for it.
    /// Port of `StructuralTagGrammarConverter::Visit`.
    pub(super) fn visit(&mut self, format: &Format) -> StructuralTagResult<i32> {
        match format {
            Format::ConstString(value) => self.visit_const_string(value),
            Format::JsonSchema { json_schema, style } => {
                self.visit_json_schema(json_schema, *style)
            }
            Format::AnyText {
                excludes,
                detected_end_strs,
            } => self.visit_any_text(excludes, detected_end_strs),
            Format::Grammar(grammar) => self.visit_grammar(grammar),
            Format::Regex(pattern) => self.visit_regex(pattern),
            Format::Sequence { elements, .. } => self.visit_sequence(elements),
            Format::Or { elements, .. } => self.visit_or(elements),
            Format::Tag(tag) => self.visit_tag(tag),
            Format::TriggeredTags {
                triggers,
                tags,
                excludes,
                at_least_one,
                stop_after_first,
                detected_end_strs,
            } => self.visit_triggered_tags(
                triggers,
                tags,
                excludes,
                *at_least_one,
                *stop_after_first,
                detected_end_strs,
            ),
            Format::TagsWithSeparator {
                tags,
                separator,
                at_least_one,
                stop_after_first,
                detected_end_strs,
            } => self.visit_tags_with_separator(
                tags,
                separator,
                *at_least_one,
                *stop_after_first,
                detected_end_strs,
            ),
        }
    }

    /// Add a rule with a unique name derived from `hint`.
    pub(super) fn add_rule(&mut self, hint: &str, body: i32) -> StructuralTagResult<i32> {
        self.builder
            .add_rule_with_hint(hint, body)
            .map_err(|e| StructuralTagError::invalid(e.to_string()))
    }

    /// Port of `VisitSub(ConstStringFormat)`.
    fn visit_const_string(&mut self, value: &str) -> StructuralTagResult<i32> {
        let expr = if value.is_empty() {
            self.builder.add_empty_str()
        } else {
            self.builder.add_byte_string(value)
        };
        let seq = self.builder.add_sequence(&[expr]);
        let choices = self.builder.add_choices(&[seq]);
        self.add_rule("const_string", choices)
    }

    /// Splice a sub-grammar built from EBNF into the builder.
    /// Port of the `SubGrammarAdder().Apply(...)` calls.
    fn add_ebnf_sub_grammar(&mut self, ebnf: &str) -> StructuralTagResult<i32> {
        let sub = parse_ebnf_default(ebnf)
            .map_err(|e| StructuralTagError::invalid(format!("EBNF parse failed: {e}")))?;
        Ok(add_sub_grammar(&mut self.builder, &sub))
    }

    /// Port of `VisitSub(JSONSchemaFormat)`.
    fn visit_json_schema(
        &mut self,
        json_schema: &str,
        style: SchemaStyle,
    ) -> StructuralTagResult<i32> {
        let ebnf = match style {
            SchemaStyle::Json => {
                json_schema_to_ebnf(json_schema, &SchemaConverterOptions::default())
            }
            SchemaStyle::QwenXml => qwen_xml_tool_calling_to_ebnf(json_schema),
            SchemaStyle::MiniMaxXml => minimax_xml_tool_calling_to_ebnf(json_schema),
            SchemaStyle::DeepSeekXml => deepseek_xml_tool_calling_to_ebnf(json_schema),
        }
        .map_err(|e| StructuralTagError::invalid(e.to_string()))?;
        self.add_ebnf_sub_grammar(&ebnf)
    }

    /// Port of `VisitSub(GrammarFormat)`.
    fn visit_grammar(&mut self, grammar: &str) -> StructuralTagResult<i32> {
        self.add_ebnf_sub_grammar(grammar)
    }

    /// Port of `VisitSub(RegexFormat)`.
    fn visit_regex(&mut self, pattern: &str) -> StructuralTagResult<i32> {
        let ebnf =
            regex_to_ebnf(pattern, true).map_err(|e| StructuralTagError::invalid(e.to_string()))?;
        self.add_ebnf_sub_grammar(&ebnf)
    }

    /// Port of `VisitSub(AnyTextFormat)`.
    fn visit_any_text(
        &mut self,
        excludes: &[String],
        detected_end_strs: &[String],
    ) -> StructuralTagResult<i32> {
        let non_empty_ends: Vec<String> = detected_end_strs
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect();
        if !non_empty_ends.is_empty() {
            let expr = self.builder.add_tag_dispatch(&super::tag_dispatch_spec(
                Vec::new(),
                false,
                non_empty_ends,
                false,
                excludes.to_vec(),
            ));
            return self.add_rule("any_text", expr);
        }
        if !excludes.is_empty() {
            let expr = self.builder.add_tag_dispatch(&super::tag_dispatch_spec(
                Vec::new(),
                true,
                Vec::new(),
                false,
                excludes.to_vec(),
            ));
            return self.add_rule("any_text", expr);
        }
        let any = self.builder.add_character_class_star(
            &[crate::grammar::builder::CharacterClassElement::new(
                0, 0x10_FFFF,
            )],
            false,
        );
        let seq = self.builder.add_sequence(&[any]);
        let choices = self.builder.add_choices(&[seq]);
        self.add_rule("any_text", choices)
    }

    /// Port of `VisitSub(SequenceFormat)`.
    fn visit_sequence(&mut self, elements: &[Format]) -> StructuralTagResult<i32> {
        let mut rule_refs = Vec::with_capacity(elements.len());
        for element in elements {
            let sub = self.visit(element)?;
            rule_refs.push(self.builder.add_rule_ref(sub));
        }
        let seq = self.builder.add_sequence(&rule_refs);
        let expr = self.builder.add_choices(&[seq]);
        self.add_rule("sequence", expr)
    }

    /// Port of `VisitSub(OrFormat)`.
    fn visit_or(&mut self, elements: &[Format]) -> StructuralTagResult<i32> {
        let mut sequence_ids = Vec::with_capacity(elements.len());
        for element in elements {
            let sub = self.visit(element)?;
            let rule_ref = self.builder.add_rule_ref(sub);
            sequence_ids.push(self.builder.add_sequence(&[rule_ref]));
        }
        let expr = self.builder.add_choices(&sequence_ids);
        self.add_rule("or", expr)
    }
}
