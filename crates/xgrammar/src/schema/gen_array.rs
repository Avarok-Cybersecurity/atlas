// SPDX-License-Identifier: AGPL-3.0-only
//
// Array EBNF generator — port of `JSONSchemaConverter::GenerateArray`
// (and the XML `GenerateArray` wrapper) from
// `cpp/json_schema_converter*.cc`.

use super::converter::JsonSchemaConverter;
use super::error::SchemaResult;
use super::script::EbnfScriptCreator;
use super::spec::ArraySpec;

impl<'p> JsonSchemaConverter<'p> {
    /// Generate the EBNF body for an array spec. Port of
    /// `GenerateArray`; the XML wrapper just bumps `nested_object_level`.
    pub(super) fn generate_array(
        &mut self,
        spec: &ArraySpec,
        rule_name: &str,
    ) -> SchemaResult<String> {
        if self.is_xml() {
            self.nested_object_level += 1;
        }
        let result = self.generate_array_inner(spec, rule_name);
        if self.is_xml() {
            self.nested_object_level -= 1;
        }
        result
    }

    fn generate_array_inner(&mut self, spec: &ArraySpec, rule_name: &str) -> SchemaResult<String> {
        self.indent.start_indent();
        let start_sep = self.indent.start_separator();
        let mid_sep = self.indent.middle_separator();
        let end_sep = self.indent.end_separator();
        let empty_sep = self.indent.empty_separator();

        // Create item rules — prefix items first, then additional.
        let mut item_rule_names = Vec::new();
        for (i, item) in spec.prefix_items.iter().enumerate() {
            let name = self.create_rule(item, &format!("{rule_name}_item_{i}"))?;
            item_rule_names.push(name);
        }
        let mut additional_rule_name = String::new();
        if spec.allow_additional_items
            && let Some(add) = &spec.additional_items
        {
            additional_rule_name = self.create_rule(add, &format!("{rule_name}_additional"))?;
        }

        self.indent.end_indent();

        let left = EbnfScriptCreator::str_lit("[");
        let right = EbnfScriptCreator::str_lit("]");

        if spec.prefix_items.is_empty() {
            let empty_part = EbnfScriptCreator::concat(&[left.clone(), empty_sep, right.clone()]);
            if !spec.allow_additional_items {
                return Ok(empty_part);
            }
            if spec.min_items == 0 && spec.max_items == 0 {
                return Ok(empty_part);
            }
            if spec.min_items == 0 && spec.max_items != 0 {
                let repeat = EbnfScriptCreator::repeat(
                    &EbnfScriptCreator::concat(&[mid_sep.clone(), additional_rule_name.clone()]),
                    0,
                    if spec.max_items == -1 {
                        -1
                    } else {
                        (spec.max_items - 1) as i32
                    },
                );
                let body = EbnfScriptCreator::concat(&[
                    left,
                    start_sep,
                    additional_rule_name,
                    repeat,
                    end_sep,
                    right,
                ]);
                return Ok(EbnfScriptCreator::or(&[body, empty_part]));
            }
            let repeat = EbnfScriptCreator::repeat(
                &EbnfScriptCreator::concat(&[mid_sep.clone(), additional_rule_name.clone()]),
                (spec.min_items - 1) as i32,
                if spec.max_items == -1 {
                    -1
                } else {
                    (spec.max_items - 1) as i32
                },
            );
            return Ok(EbnfScriptCreator::concat(&[
                left,
                start_sep,
                additional_rule_name,
                repeat,
                end_sep,
                right,
            ]));
        }

        // Has prefix items.
        let mut prefix_part: Vec<String> = Vec::new();
        for (i, name) in item_rule_names.iter().enumerate() {
            if i > 0 {
                prefix_part.push(mid_sep.clone());
            }
            prefix_part.push(name.clone());
        }
        let prefix_str = EbnfScriptCreator::concat(&prefix_part);
        if !spec.allow_additional_items {
            return Ok(EbnfScriptCreator::concat(&[
                left, start_sep, prefix_str, end_sep, right,
            ]));
        }
        let min_items = 0i64.max(spec.min_items - item_rule_names.len() as i64);
        let repeat = EbnfScriptCreator::repeat(
            &EbnfScriptCreator::concat(&[mid_sep, additional_rule_name]),
            min_items as i32,
            if spec.max_items == -1 {
                -1
            } else {
                (spec.max_items - item_rule_names.len() as i64) as i32
            },
        );
        Ok(EbnfScriptCreator::concat(&[
            left, start_sep, prefix_str, repeat, end_sep, right,
        ]))
    }
}
