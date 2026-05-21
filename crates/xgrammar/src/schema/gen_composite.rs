// SPDX-License-Identifier: AGPL-3.0-only
//
// Composite EBNF generators — port of
// `JSONSchemaConverter::Generate{Ref,AnyOf,AllOf,TypeArray}` from
// `cpp/json_schema_converter.cc`.

use super::converter::JsonSchemaConverter;
use super::error::SchemaResult;
use super::spec::{SchemaSpecPtr, SpecKind};

impl<'p> JsonSchemaConverter<'p> {
    /// Generate the EBNF for a `$ref`. Resolves the URI via the
    /// parser, allocating the rule name first to break recursion.
    /// Port of `GenerateRef`.
    pub(super) fn generate_ref(&mut self, uri: &str) -> SchemaResult<String> {
        if let Some(name) = self.uri_to_rule.get(uri) {
            return Ok(name.clone());
        }

        // Derive a rule-name hint from the URI path.
        let mut hint = String::from("ref");
        if uri.len() >= 2 && uri.starts_with("#/") {
            let mut prefix = String::new();
            for part in uri[2..].split('/') {
                if part.is_empty() {
                    continue;
                }
                if !prefix.is_empty() {
                    prefix.push('_');
                }
                for c in part.chars() {
                    if c.is_alphabetic() || c == '_' || c == '-' || c == '.' {
                        prefix.push(c);
                    }
                }
            }
            if !prefix.is_empty() {
                hint = prefix;
            }
        }

        let allocated = self.script.allocate_rule_name(&hint);
        self.uri_to_rule.insert(uri.to_string(), allocated.clone());

        let resolved = self.parser.resolve_ref(uri, &allocated)?;
        let body = self.generate_from_spec(&resolved, &allocated)?;
        self.script.add_rule_with_allocated_name(&allocated, &body);

        if !resolved.cache_key.is_empty() {
            self.add_cache(&resolved.cache_key, &allocated);
        }
        Ok(allocated)
    }

    /// Generate the EBNF for `anyOf` / `oneOf`. Port of `GenerateAnyOf`.
    pub(super) fn generate_any_of(
        &mut self,
        options: &[SchemaSpecPtr],
        rule_name: &str,
    ) -> SchemaResult<String> {
        let mut out = String::new();
        for (i, opt) in options.iter().enumerate() {
            if i != 0 {
                out.push_str(" | ");
            }
            let name = self.create_rule(opt, &format!("{rule_name}_case_{i}"))?;
            out.push_str(&name);
        }
        Ok(out)
    }

    /// Generate the EBNF for `allOf`. Only the single-schema case is
    /// fully supported; multiple schemas degrade to "any" (matching
    /// the C++ warning path). Port of `GenerateAllOf`.
    pub(super) fn generate_all_of(
        &mut self,
        schemas: &[SchemaSpecPtr],
        rule_name: &str,
    ) -> SchemaResult<String> {
        if schemas.len() == 1 {
            return self.generate_from_spec(&schemas[0], &format!("{rule_name}_case_0"));
        }
        // Multi-schema allOf is not fully supported upstream either —
        // fall back to "any".
        let any = super::spec::SchemaSpec::make(SpecKind::Any, "", "any");
        self.generate_from_spec(&any, rule_name)
    }

    /// Generate the EBNF for a `"type": [...]` array. Port of
    /// `GenerateTypeArray`.
    pub(super) fn generate_type_array(
        &mut self,
        schemas: &[SchemaSpecPtr],
        rule_name: &str,
    ) -> SchemaResult<String> {
        let mut out = String::new();
        for (i, schema) in schemas.iter().enumerate() {
            if i != 0 {
                out.push_str(" | ");
            }
            let name = self.create_rule(schema, &format!("{rule_name}_type_{i}"))?;
            out.push_str(&name);
        }
        Ok(out)
    }
}
