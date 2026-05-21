// SPDX-License-Identifier: AGPL-3.0-only
//
// Object property-rule generation, part 2 — port of the two
// property-count-constrained branches (cases 2 and 3) of
// `JSONSchemaConverter::GetPartialRuleForProperties` from
// `cpp/json_schema_converter.cc`.

use super::converter::JsonSchemaConverter;
use super::error::SchemaResult;
use super::spec::{Property, SchemaSpecPtr};

impl<'p> JsonSchemaConverter<'p> {
    /// Cases 2 and 3: object with a lower bound (case 2: `max == -1`)
    /// or both bounds (case 3) on the property count.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn partial_rule_constrained(
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
        min_properties: i64,
        max_properties: i64,
        additional_prop_pattern_override: Option<&str>,
    ) -> SchemaResult<String> {
        let n = properties.len() as i32;
        let is_req = |name: &str| required.iter().any(|r| r == name);
        let allow_additional = additional.is_some();
        let has_max = max_properties != -1;

        // Resolve the additional-property pattern up front.
        let mut additional_prop_pattern = String::new();
        if let Some(add) = additional {
            // Override site 2/3 (upstream commit a6aeabb, #594).
            if let Some(ovr) = additional_prop_pattern_override {
                additional_prop_pattern = ovr.to_string();
            } else {
                let add_value_rule =
                    self.create_rule(add, &format!("{rule_name}_{additional_suffix}"))?;
                let key = self.key_pattern().to_string();
                additional_prop_pattern = self.format_other_property(&key, &add_value_rule);
            }
        }

        // ---- Compute key_matched_min / key_matched_max ranges. ----
        let mut key_matched_min = vec![0i32; n as usize];
        let mut key_matched_max = vec![n; n as usize];
        let mut is_required = vec![false; n as usize];

        let mut get_first_required = is_req(&properties[0].name);
        key_matched_min[0] = 1;
        key_matched_max[0] = 1;
        for i in 1..n as usize {
            if is_req(&properties[i].name) {
                is_required[i] = true;
                key_matched_min[i] = key_matched_min[i - 1] + 1;
            } else {
                key_matched_min[i] = key_matched_min[i - 1];
            }
            if !get_first_required {
                key_matched_min[i] = 1;
            }
            key_matched_max[i] = key_matched_max[i - 1] + 1;
            if is_required[i] {
                get_first_required = true;
            }
        }
        if is_req(&properties[0].name) {
            is_required[0] = true;
        }

        let last = (n - 1) as usize;
        if allow_additional {
            key_matched_min[last] = 1.max(key_matched_min[last]);
            if has_max {
                key_matched_max[last] = (max_properties as i32).min(key_matched_max[last]);
            }
        } else {
            key_matched_min[last] = (min_properties as i32).max(key_matched_min[last]);
            if has_max {
                key_matched_max[last] = (max_properties as i32).min(key_matched_max[last]);
            }
        }
        for i in (0..last).rev() {
            key_matched_min[i] = key_matched_min[i].max(key_matched_min[i + 1] - 1);
            if has_max {
                if is_required[i + 1] {
                    key_matched_max[i] = key_matched_max[i].min(key_matched_max[i + 1] - 1);
                } else {
                    key_matched_max[i] = key_matched_max[i].min(key_matched_max[i + 1]);
                }
            }
        }

        // For case 2 (`max == -1`) the upper iteration bound is `i+1`.
        let upper_for = |i: usize| -> i32 {
            if has_max {
                key_matched_max[i]
            } else {
                i as i32 + 1
            }
        };

        // ---- Build rule_names. ----
        let mut rule_names: Vec<Vec<String>> = vec![Vec::new(); n as usize];

        // Last rule(s).
        let last_upper = if has_max { key_matched_max[last] } else { n };
        for matched in key_matched_min[last]..=last_upper {
            if allow_additional {
                let body = self.property_with_number_constraints(
                    &format!("{mid_sep} {additional_prop_pattern}"),
                    min_properties,
                    max_properties,
                    matched as i64,
                );
                let name = self
                    .script
                    .add_rule(&format!("{rule_name}_part_{}_{matched}", n - 1), &body);
                rule_names[last].push(name);
            } else {
                rule_names[last].push("\"\"".to_string());
            }
        }

        // Rules 0..=n-2 backwards.
        for i in (0..last).rev() {
            let prop_pattern = &prop_patterns[i + 1];
            let lo = key_matched_min[i];
            let hi = upper_for(i);
            for matched in lo..=hi {
                let next_min = key_matched_min[i + 1];
                let body = if has_max && matched == key_matched_max[i + 1] {
                    rule_names[i + 1][(matched - next_min) as usize].clone()
                } else if is_required[i + 1] || matched == next_min - 1 {
                    let idx = (matched + 1 - next_min) as usize;
                    format!("{mid_sep} {prop_pattern} {}", rule_names[i + 1][idx])
                } else {
                    let idx0 = (matched - next_min) as usize;
                    let idx1 = (matched - next_min + 1) as usize;
                    format!(
                        "{} | {mid_sep} {prop_pattern} {}",
                        rule_names[i + 1][idx0],
                        rule_names[i + 1][idx1]
                    )
                };
                let name = self
                    .script
                    .add_rule(&format!("{rule_name}_part_{i}_{matched}"), &body);
                rule_names[i].push(name);
            }
        }

        // ---- Root rule. ----
        let mut res = String::new();
        let mut is_first = true;
        for i in 0..n as usize {
            if has_max && key_matched_max[i] < key_matched_min[i] {
                continue;
            }
            if key_matched_min[i] > 1 {
                break;
            }
            if !is_first {
                res.push_str(" | ");
            } else {
                is_first = false;
            }
            let idx = (1 - key_matched_min[i]) as usize;
            res.push_str(&format!("({} {})", prop_patterns[i], rule_names[i][idx]));
            if is_required[i] {
                break;
            }
        }

        if allow_additional && required.is_empty() {
            if !is_first {
                res.push_str(" | ");
            }
            let inner = self.property_with_number_constraints(
                &format!("{mid_sep} {additional_prop_pattern}"),
                min_properties,
                max_properties,
                1,
            );
            res.push_str(&format!("({additional_prop_pattern} {inner})"));
        }

        Ok(format!("{first_sep} ({res}) {last_sep}"))
    }
}
