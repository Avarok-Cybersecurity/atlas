// SPDX-License-Identifier: AGPL-3.0-only
//
// Object EBNF generator — port of `JSONSchemaConverter::GenerateObject`
// and the property-formatting hooks, plus the XML overrides.

use super::converter::{JsonSchemaConverter, xml_wrapper};
use super::error::SchemaResult;
use super::spec::{ObjectSpec, SchemaSpec, SchemaSpecPtr, SpecKind};
use crate::regex::regex_to_ebnf;

impl<'p> JsonSchemaConverter<'p> {
    /// Format a property key literal. Port of `FormatPropertyKey`
    /// plus the XML override.
    pub(super) fn format_property_key(&self, key: &str) -> String {
        if self.at_xml_layer() {
            let (prefix, suffix, _) = xml_wrapper(self.format);
            return format!("\"{prefix}{key}{suffix}\"");
        }
        format!("\"\\\"{key}\\\"\"")
    }

    /// Format a `key: value` property. Port of `FormatProperty` plus
    /// the XML override.
    pub(super) fn format_property(&self, key: &str, value_rule: &str) -> String {
        if self.at_xml_layer() {
            let (prefix, suffix, param_suffix) = xml_wrapper(self.format);
            let ws = self.whitespace_pattern();
            return format!("\"{prefix}{key}{suffix}\" {ws} {value_rule} {ws} \"{param_suffix}\"");
        }
        format!(
            "{} {} {value_rule}",
            self.format_property_key(key),
            self.colon_pattern
        )
    }

    /// Format an additional/unevaluated property. Port of
    /// `FormatOtherProperty` plus the XML override.
    pub(super) fn format_other_property(&self, key_pattern: &str, value_rule: &str) -> String {
        if self.at_xml_layer() {
            let (prefix, suffix, param_suffix) = xml_wrapper(self.format);
            let ws = self.whitespace_pattern();
            return format!(
                "\"{prefix}\" {key_pattern} \"{suffix}\" {ws} {value_rule} {ws} \
                 \"{param_suffix}\""
            );
        }
        format!("{key_pattern} {} {value_rule}", self.colon_pattern)
    }

    /// Generate the EBNF body for an object spec. Port of
    /// `GenerateObject` plus the XML `nested_object_level` wrapper.
    pub(super) fn generate_object(
        &mut self,
        spec: &ObjectSpec,
        rule_name: &str,
        need_braces: bool,
    ) -> SchemaResult<String> {
        if self.is_xml() {
            self.nested_object_level += 1;
            let need_brace = self.nested_object_level > 1;
            let result = self.generate_object_inner(spec, rule_name, need_brace);
            self.nested_object_level -= 1;
            return result;
        }
        self.generate_object_inner(spec, rule_name, need_braces)
    }

    fn generate_object_inner(
        &mut self,
        spec: &ObjectSpec,
        rule_name: &str,
        need_braces: bool,
    ) -> SchemaResult<String> {
        let mut result = String::new();
        if need_braces {
            result.push_str("\"{\"");
        }
        let mut could_be_empty = false;

        // Choose the additional-property schema, if any.
        let (additional_suffix, additional_property): (&str, Option<SchemaSpecPtr>) =
            if spec.allow_additional_properties && spec.additional_properties_schema.is_some() {
                ("addl", spec.additional_properties_schema.clone())
            } else if spec.allow_unevaluated_properties
                && spec.unevaluated_properties_schema.is_some()
            {
                ("uneval", spec.unevaluated_properties_schema.clone())
            } else if spec.allow_additional_properties || spec.allow_unevaluated_properties {
                ("addl", Some(SchemaSpec::make(SpecKind::Any, "", "any")))
            } else {
                ("", None)
            };

        self.indent.start_indent();

        let has_pattern = !spec.pattern_properties.is_empty() || spec.property_names.is_some();
        if !spec.properties.is_empty() && has_pattern {
            // Case 1a: named `properties` coexist with `patternProperties`
            // and/or `propertyNames`. Feed the pattern alternatives into
            // the named-property machinery as an additional-property
            // override (upstream commit a6aeabb, #594).
            could_be_empty = self.gen_object_props_and_patterns(
                spec,
                rule_name,
                additional_suffix,
                additional_property.as_ref(),
                &mut result,
            )?;
        } else if has_pattern {
            // Case 1b: patternProperties / propertyNames without named properties.
            could_be_empty = self.gen_object_pattern_case(spec, rule_name, &mut result)?;
        } else if !spec.properties.is_empty() {
            let partial = self.partial_rule_for_properties(
                &spec.properties,
                &spec.required,
                additional_property.as_ref(),
                rule_name,
                additional_suffix,
                spec.min_properties,
                spec.max_properties,
                None,
            )?;
            result.push(' ');
            result.push_str(&partial);
            could_be_empty = spec.required.is_empty() && spec.min_properties == 0;
        } else if let Some(add) = &additional_property {
            if spec.max_properties != 0 {
                let add_value_rule =
                    self.create_rule(add, &format!("{rule_name}_{additional_suffix}"))?;
                let key = self.key_pattern().to_string();
                let other = self.format_other_property(&key, &add_value_rule);
                let sep1 = self.next_separator(false);
                let sep2 = self.next_separator(false);
                let constrained = self.property_with_number_constraints(
                    &format!("{sep2} {other}"),
                    spec.min_properties,
                    spec.max_properties,
                    1,
                );
                let sep3 = self.next_separator(true);
                result.push_str(&format!(" {sep1} {other} {constrained} {sep3}"));
            }
            could_be_empty = spec.min_properties == 0;
        }

        self.indent.end_indent();

        if need_braces {
            result.push_str(" \"}\"");
        }
        if could_be_empty {
            result = self.apply_could_be_empty(result, need_braces);
        }
        Ok(result)
    }

    /// Case 1a: named `properties` coexisting with `patternProperties`
    /// and/or `propertyNames`. Builds the pattern alternatives as an
    /// additional-property override fed into the named-property rule
    /// machinery. Port of `GenerateObject` Case 1a (upstream a6aeabb, #594).
    fn gen_object_props_and_patterns(
        &mut self,
        spec: &ObjectSpec,
        rule_name: &str,
        additional_suffix: &str,
        additional_property: Option<&SchemaSpecPtr>,
        result: &mut String,
    ) -> SchemaResult<bool> {
        let mut effective_additional: Option<SchemaSpecPtr> = additional_property.cloned();
        let mut effective_suffix = additional_suffix.to_string();
        let mut pp_override = String::new();

        if !spec.pattern_properties.is_empty() {
            // Build patternProperties as additional-property alternatives.
            let mut pp_body = String::new();
            for (i, pp) in spec.pattern_properties.iter().enumerate() {
                let value = self.create_rule(&pp.schema, &format!("{rule_name}_pp_{i}"))?;
                let key_regex = regex_to_ebnf(&pp.pattern, false).map_err(|e| {
                    super::error::SchemaError::invalid(format!(
                        "patternProperties regex failed: {e}"
                    ))
                })?;
                let pp_single =
                    format!("\"\\\"\"{key_regex}\"\\\"\" {} {value}", self.colon_pattern);
                if i != 0 {
                    pp_body.push_str(" | ");
                }
                pp_body.push_str(&pp_single);
            }
            // Merge with any explicit additionalProperties schema.
            if let Some(add) = &effective_additional {
                let add_value_rule =
                    self.create_rule(add, &format!("{rule_name}_{effective_suffix}"))?;
                let key = self.key_pattern().to_string();
                let add_prop = self.format_other_property(&key, &add_value_rule);
                pp_body.push_str(&format!(" | {add_prop}"));
            }
            // Parenthesize to keep EBNF precedence correct with `|` present.
            pp_override = format!("({pp_body})");
            if effective_additional.is_none() {
                effective_additional = Some(SchemaSpec::make(SpecKind::Any, "", "any"));
            }
            effective_suffix = "pp".to_string();
        } else if let Some(add) = &effective_additional {
            // propertyNames constrains the keys of additional properties.
            // Only applied when additional properties are allowed.
            if let Some(prop_names) = &spec.property_names {
                let key_pattern = self.create_rule(prop_names, &format!("{rule_name}_name"))?;
                let val_rule = self.create_rule(add, &format!("{rule_name}_{effective_suffix}"))?;
                pp_override = format!("{key_pattern} {} {val_rule}", self.colon_pattern);
                effective_suffix = "propnames".to_string();
            }
        }

        let partial = self.partial_rule_for_properties(
            &spec.properties,
            &spec.required,
            effective_additional.as_ref(),
            rule_name,
            &effective_suffix,
            spec.min_properties,
            spec.max_properties,
            if pp_override.is_empty() {
                None
            } else {
                Some(pp_override.as_str())
            },
        )?;
        result.push(' ');
        result.push_str(&partial);
        Ok(spec.required.is_empty() && spec.min_properties == 0)
    }

    /// Handle the `patternProperties` / `propertyNames` object case.
    /// Returns whether the object can be empty.
    fn gen_object_pattern_case(
        &mut self,
        spec: &ObjectSpec,
        rule_name: &str,
        result: &mut String,
    ) -> SchemaResult<bool> {
        let beg_seq = self.next_separator(false);
        let mut property_rule_body = String::from("(");

        if spec.max_properties != 0 {
            if !spec.pattern_properties.is_empty() {
                for (i, pp) in spec.pattern_properties.iter().enumerate() {
                    let value = self.create_rule(&pp.schema, &format!("{rule_name}_prop_{i}"))?;
                    let key_regex = regex_to_ebnf(&pp.pattern, false).map_err(|e| {
                        super::error::SchemaError::invalid(format!(
                            "patternProperties regex failed: {e}"
                        ))
                    })?;
                    let property_pattern =
                        format!("\"\\\"\"{key_regex}\"\\\"\" {} {value}", self.colon_pattern);
                    if i != 0 {
                        property_rule_body.push_str(" | ");
                    }
                    property_rule_body.push_str(&format!("({beg_seq} {property_pattern})"));
                }
                property_rule_body.push(')');
            } else if let Some(prop_names) = &spec.property_names {
                let key_pattern = self.create_rule(prop_names, &format!("{rule_name}_name"))?;
                let any = self.basic_any_rule();
                property_rule_body.push_str(&format!(
                    "{beg_seq} {key_pattern} {} {any})",
                    self.colon_pattern
                ));
            }

            let prop_rule_name = self.script.allocate_rule_name(&format!("{rule_name}_prop"));
            self.script
                .add_rule_with_allocated_name(&prop_rule_name, &property_rule_body);

            let next1 = self.next_separator(false);
            let constrained = self.property_with_number_constraints(
                &format!("{next1} {prop_rule_name}"),
                spec.min_properties,
                spec.max_properties,
                1,
            );
            let next_end = self.next_separator(true);
            result.push_str(&format!(" {prop_rule_name} {constrained}{next_end}"));
            return Ok(spec.min_properties == 0);
        }
        Ok(false)
    }

    /// Apply the "object could be empty" wrapping. Port of the tail
    /// of `GenerateObject`.
    fn apply_could_be_empty(&self, result: String, need_braces: bool) -> String {
        let ws = self.whitespace_pattern();
        let rest = if need_braces {
            if self.any_whitespace {
                format!("\"{{\" {ws} \"}}\"")
            } else {
                "\"{\" \"}\"".to_string()
            }
        } else if self.any_whitespace {
            ws
        } else {
            String::new()
        };
        if result == "\"{\"  \"}\"" || result.is_empty() {
            rest
        } else {
            format!("({result}) | {rest}")
        }
    }
}
