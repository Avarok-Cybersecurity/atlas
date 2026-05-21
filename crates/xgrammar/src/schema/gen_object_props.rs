// SPDX-License-Identifier: AGPL-3.0-only
//
// Object property-rule generation, part 1 — port of
// `JSONSchemaConverter::{GetPropertyWithNumberConstraints,
// GetPartialRuleForProperties}` (the dispatcher + unconstrained
// case 1) from `cpp/json_schema_converter.cc`.

use super::converter::JsonSchemaConverter;
use super::error::SchemaResult;
use super::spec::{Property, SchemaSpecPtr};

impl<'p> JsonSchemaConverter<'p> {
    /// Build the repetition wrapper enforcing min/max property counts
    /// on a repeated property `pattern`. Port of
    /// `GetPropertyWithNumberConstraints`.
    pub(super) fn property_with_number_constraints(
        &self,
        pattern: &str,
        min_properties: i64,
        max_properties: i64,
        already_repeated: i64,
    ) -> String {
        if max_properties != -1 && max_properties == already_repeated {
            return "\"\"".to_string();
        }
        let lower = 0i64.max(min_properties - already_repeated);
        let upper = if max_properties == -1 {
            -1
        } else {
            (-1i64).max(max_properties - already_repeated)
        };
        if lower == 0 && upper == -1 {
            format!("({pattern})*")
        } else if lower == 0 && upper == 1 {
            format!("({pattern})?")
        } else if lower == 1 && upper == 1 {
            pattern.to_string()
        } else {
            let upper_str = if upper == -1 {
                String::new()
            } else {
                upper.to_string()
            };
            format!("({pattern}){{{lower},{upper_str}}} ")
        }
    }

    /// Generate the partial rules for an object's named properties.
    /// Dispatches to the three cases of `GetPartialRuleForProperties`.
    ///
    /// `additional_prop_pattern_override`, when set, replaces the
    /// auto-derived additional-property pattern — used to feed
    /// `patternProperties`/`propertyNames` alternatives into the named
    /// property machinery (upstream commit a6aeabb, #594).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn partial_rule_for_properties(
        &mut self,
        properties: &[Property],
        required: &[String],
        additional: Option<&SchemaSpecPtr>,
        rule_name: &str,
        additional_suffix: &str,
        min_properties: i64,
        max_properties: i64,
        additional_prop_pattern_override: Option<&str>,
    ) -> SchemaResult<String> {
        if max_properties == 0 {
            return Ok(String::new());
        }

        let first_sep = self.next_separator(false);
        let mid_sep = self.next_separator(false);
        let last_sep = self.next_separator(true);

        // Pre-create the value rules and property patterns.
        let mut prop_patterns: Vec<String> = Vec::with_capacity(properties.len());
        for (idx, prop) in properties.iter().enumerate() {
            let value_rule = self.create_rule(&prop.schema, &format!("{rule_name}_prop_{idx}"))?;
            prop_patterns.push(self.format_property(&prop.name, &value_rule));
        }

        if min_properties == 0 && max_properties == -1 {
            self.partial_rule_case1(
                properties,
                required,
                additional,
                rule_name,
                additional_suffix,
                &prop_patterns,
                &first_sep,
                &mid_sep,
                &last_sep,
                additional_prop_pattern_override,
            )
        } else {
            self.partial_rule_constrained(
                properties,
                required,
                additional,
                rule_name,
                additional_suffix,
                &prop_patterns,
                &first_sep,
                &mid_sep,
                &last_sep,
                min_properties,
                max_properties,
                additional_prop_pattern_override,
            )
        }
    }

    /// Case 1: no property-count constraints. Port of the first
    /// branch of `GetPartialRuleForProperties`.
    #[allow(clippy::too_many_arguments)]
    fn partial_rule_case1(
        &mut self,
        properties: &[Property],
        required: &[String],
        additional: Option<&SchemaSpecPtr>,
        rule_name: &str,
        additional_suffix: &str,
        prop_patterns: &[String],
        first_sep: &str,
        mid_sep: &str,
        last_sep: &str,
        additional_prop_pattern_override: Option<&str>,
    ) -> SchemaResult<String> {
        let n = properties.len();
        let mut rule_names: Vec<String> = vec![String::new(); n];
        let mut is_required: Vec<bool> = vec![false; n];
        let allow_additional = additional.is_some();

        // Last rule: either trailing additionals or empty.
        let mut additional_prop_pattern = String::new();
        if let Some(add) = additional {
            // Override site 1 (upstream commit a6aeabb, #594).
            if let Some(ovr) = additional_prop_pattern_override {
                additional_prop_pattern = ovr.to_string();
            } else {
                let add_value_rule =
                    self.create_rule(add, &format!("{rule_name}_{additional_suffix}"))?;
                let key = self.key_pattern().to_string();
                additional_prop_pattern = self.format_other_property(&key, &add_value_rule);
            }
            let last_body = format!("({mid_sep} {additional_prop_pattern})*");
            let last_name = self
                .script
                .add_rule(&format!("{rule_name}_part_{}", n - 1), &last_body);
            rule_names[n - 1] = last_name;
        } else {
            rule_names[n - 1] = "\"\"".to_string();
        }

        // Build rules 0..=n-2 backwards.
        for i in (0..n.saturating_sub(1)).rev() {
            let prop_pattern = &prop_patterns[i + 1];
            let last_rule_name = &rule_names[i + 1];
            let mut body = format!("{mid_sep} {prop_pattern} {last_rule_name}");
            if !required.iter().any(|r| r == &properties[i + 1].name) {
                body = format!("{last_rule_name} | {body}");
            } else {
                is_required[i + 1] = true;
            }
            let cur_name = self
                .script
                .add_rule(&format!("{rule_name}_part_{i}"), &body);
            rule_names[i] = cur_name;
        }
        if required.iter().any(|r| r == &properties[0].name) {
            is_required[0] = true;
        }

        // Root rule.
        let mut res = String::new();
        for i in 0..n {
            if i != 0 {
                res.push_str(" | ");
            }
            res.push_str(&format!("({} {})", prop_patterns[i], rule_names[i]));
            if is_required[i] {
                break;
            }
        }
        if allow_additional && required.is_empty() {
            res.push_str(&format!(
                " | {additional_prop_pattern} {}",
                rule_names[n - 1]
            ));
        }
        Ok(format!("{first_sep} ({res}) {last_sep}"))
    }
}
