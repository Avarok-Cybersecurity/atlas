// SPDX-License-Identifier: AGPL-3.0-only
//
// SchemaParser — port of the internal `SchemaParser` class from
// `cpp/json_schema_converter.cc`.
//
// Converts a JSON Schema document (an order-preserving [`JsonValue`])
// into the [`SchemaSpec`] intermediate representation, with caching
// for sub-schema deduplication and `$ref` resolution.
//
// This file holds the top-level `parse` dispatch plus the scalar
// type parsers (integer/number/string/boolean/null). Collections and
// composites live in `parser_collections.rs` / `parser_composite.rs`.

use super::cache_key::compute_cache_key;
use super::error::{SchemaError, SchemaResult};
use super::json_value::JsonValue;
use super::spec::{IntegerSpec, NumberSpec, SchemaSpec, SchemaSpecPtr, SpecKind, StringSpec};
use std::collections::HashMap;

/// Parses a JSON Schema document into [`SchemaSpec`]s.
pub struct SchemaParser {
    /// Whether strict mode is on (disallows unspecified members).
    pub(super) strict_mode: bool,
    /// The full root document, for `$ref` resolution.
    pub(super) root_schema: JsonValue,
    /// `$ref` URI -> resolved spec.
    pub(super) ref_cache: HashMap<String, SchemaSpecPtr>,
    /// cache_key -> spec, deduplicating identical sub-schemas.
    pub(super) schema_cache: HashMap<String, SchemaSpecPtr>,
}

impl SchemaParser {
    /// Build a parser over `root_schema`.
    pub fn new(root_schema: JsonValue, strict_mode: bool) -> Self {
        SchemaParser {
            strict_mode,
            root_schema,
            ref_cache: HashMap::new(),
            schema_cache: HashMap::new(),
        }
    }

    /// Parse `schema` into a spec. `rule_name_hint` seeds the EBNF
    /// rule name; `default_type` is applied when no `type` keyword is
    /// present (used by `propertyNames`). Port of `SchemaParser::Parse`.
    pub fn parse(
        &mut self,
        schema: &JsonValue,
        rule_name_hint: &str,
        default_type: Option<&str>,
    ) -> SchemaResult<SchemaSpecPtr> {
        let cache_key = compute_cache_key(schema);
        if let Some(cached) = self.schema_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Boolean schemas: `true` => any, `false` => unsatisfiable.
        if let Some(b) = schema.as_bool() {
            if !b {
                return Err(SchemaError::unsatisfiable(
                    "Schema 'false' cannot accept any value",
                ));
            }
            let spec = SchemaSpec::make(SpecKind::Any, cache_key.clone(), rule_name_hint);
            self.schema_cache.insert(cache_key, spec.clone());
            return Ok(spec);
        }

        let obj = schema
            .as_object()
            .ok_or_else(|| SchemaError::invalid("Schema should be an object or bool"))?;

        let result = self.dispatch(schema, obj, &cache_key, rule_name_hint, default_type)?;
        self.schema_cache.insert(cache_key, result.clone());
        Ok(result)
    }

    /// Decide which keyword group drives parsing and build the spec.
    fn dispatch(
        &mut self,
        schema: &JsonValue,
        obj: &[(String, JsonValue)],
        cache_key: &str,
        hint: &str,
        default_type: Option<&str>,
    ) -> SchemaResult<SchemaSpecPtr> {
        let has = |k: &str| obj.iter().any(|(key, _)| key == k);
        let get = |k: &str| obj.iter().find(|(key, _)| key == k).map(|(_, v)| v);

        if has("$ref") {
            let uri = self.parse_ref(schema)?;
            return Ok(SchemaSpec::make(SpecKind::Ref(uri), cache_key, hint));
        }
        if has("const") {
            let kind = self.parse_const(schema)?;
            return Ok(SchemaSpec::make(kind, cache_key, hint));
        }
        if has("enum") {
            let kind = self.parse_enum(schema)?;
            return Ok(SchemaSpec::make(kind, cache_key, hint));
        }
        if has("anyOf") || has("oneOf") {
            let kind = self.parse_any_of(schema)?;
            return Ok(SchemaSpec::make(kind, cache_key, hint));
        }
        if has("allOf") {
            let kind = self.parse_all_of(schema)?;
            return Ok(SchemaSpec::make(kind, cache_key, hint));
        }
        if has("type") || default_type.is_some() {
            let type_val = get("type");
            if let Some(JsonValue::Array(_)) = type_val {
                let kind = self.parse_type_array(schema, hint)?;
                return Ok(SchemaSpec::make(kind, cache_key, hint));
            }
            let type_name: String = match type_val {
                Some(JsonValue::String(s)) => s.clone(),
                Some(_) => return Err(SchemaError::invalid("Type should be a string")),
                None => default_type.unwrap().to_string(),
            };
            let kind = self.parse_typed(schema, &type_name)?;
            return Ok(SchemaSpec::make(kind, cache_key, hint));
        }
        if has("properties") || has("additionalProperties") || has("unevaluatedProperties") {
            let kind = self.parse_object(schema)?;
            return Ok(SchemaSpec::make(kind, cache_key, hint));
        }
        if has("items") || has("prefixItems") || has("unevaluatedItems") {
            let kind = self.parse_array(schema)?;
            return Ok(SchemaSpec::make(kind, cache_key, hint));
        }
        Ok(SchemaSpec::make(SpecKind::Any, cache_key, hint))
    }

    /// Dispatch a concrete `type` string to its parser.
    fn parse_typed(&mut self, schema: &JsonValue, type_name: &str) -> SchemaResult<SpecKind> {
        match type_name {
            "integer" => Ok(SpecKind::Integer(self.parse_integer(schema)?)),
            "number" => Ok(SpecKind::Number(self.parse_number(schema)?)),
            "string" => Ok(SpecKind::String(self.parse_string(schema)?)),
            "boolean" => Ok(SpecKind::Boolean),
            "null" => Ok(SpecKind::Null),
            "array" => Ok(self.parse_array(schema)?),
            "object" => Ok(self.parse_object(schema)?),
            other => Err(SchemaError::invalid(format!(
                "Unsupported type \"{other}\""
            ))),
        }
    }

    /// Parse an integer bound, accepting integers or whole doubles.
    fn integer_bound(value: &JsonValue) -> SchemaResult<i64> {
        match value {
            JsonValue::Number {
                as_i64: Some(i), ..
            } => Ok(*i),
            JsonValue::Number { as_f64, .. } => {
                if *as_f64 != as_f64.floor() {
                    return Err(SchemaError::invalid(
                        "Integer constraint must be a whole number",
                    ));
                }
                if *as_f64 > i64::MAX as f64 || *as_f64 < i64::MIN as f64 {
                    return Err(SchemaError::invalid("Integer exceeds limit"));
                }
                Ok(*as_f64 as i64)
            }
            _ => Err(SchemaError::invalid("Value must be a number")),
        }
    }

    /// Parse the `integer` type keywords. Port of `ParseInteger`.
    fn parse_integer(&self, schema: &JsonValue) -> SchemaResult<IntegerSpec> {
        let mut spec = IntegerSpec::default();
        if let Some(v) = schema.get("minimum") {
            spec.minimum = Some(Self::integer_bound(v)?);
        }
        if let Some(v) = schema.get("maximum") {
            spec.maximum = Some(Self::integer_bound(v)?);
        }
        if let Some(v) = schema.get("exclusiveMinimum") {
            let val = Self::integer_bound(v)?;
            if val == i64::MAX {
                return Err(SchemaError::unsatisfiable(
                    "exclusiveMinimum would cause integer overflow",
                ));
            }
            spec.exclusive_minimum = Some(val);
        }
        if let Some(v) = schema.get("exclusiveMaximum") {
            let val = Self::integer_bound(v)?;
            if val == i64::MIN {
                return Err(SchemaError::unsatisfiable(
                    "exclusiveMaximum would cause integer underflow",
                ));
            }
            spec.exclusive_maximum = Some(val);
        }
        let mut eff_min = spec.minimum.unwrap_or(i64::MIN);
        let mut eff_max = spec.maximum.unwrap_or(i64::MAX);
        if let Some(em) = spec.exclusive_minimum {
            eff_min = eff_min.max(em.saturating_add(1));
        }
        if let Some(em) = spec.exclusive_maximum {
            eff_max = eff_max.min(em.saturating_sub(1));
        }
        if eff_min > eff_max {
            return Err(SchemaError::unsatisfiable(
                "Invalid range: minimum greater than maximum",
            ));
        }
        Ok(spec)
    }

    /// Parse the `number` type keywords. Port of `ParseNumber`.
    fn parse_number(&self, schema: &JsonValue) -> SchemaResult<NumberSpec> {
        let get_double = |v: &JsonValue| -> SchemaResult<f64> {
            match v {
                JsonValue::Number { as_f64, .. } => Ok(*as_f64),
                _ => Err(SchemaError::invalid("Value must be a number")),
            }
        };
        let mut spec = NumberSpec::default();
        if let Some(v) = schema.get("minimum") {
            spec.minimum = Some(get_double(v)?);
        }
        if let Some(v) = schema.get("maximum") {
            spec.maximum = Some(get_double(v)?);
        }
        if let Some(v) = schema.get("exclusiveMinimum") {
            spec.exclusive_minimum = Some(get_double(v)?);
        }
        if let Some(v) = schema.get("exclusiveMaximum") {
            spec.exclusive_maximum = Some(get_double(v)?);
        }
        let mut eff_min = spec.minimum.unwrap_or(f64::NEG_INFINITY);
        let mut eff_max = spec.maximum.unwrap_or(f64::INFINITY);
        if let Some(em) = spec.exclusive_minimum {
            eff_min = eff_min.max(em);
        }
        if let Some(em) = spec.exclusive_maximum {
            eff_max = eff_max.min(em);
        }
        if eff_min > eff_max {
            return Err(SchemaError::unsatisfiable(
                "Invalid range: minimum greater than maximum",
            ));
        }
        Ok(spec)
    }

    /// Parse the `string` type keywords. Port of `ParseString`.
    fn parse_string(&self, schema: &JsonValue) -> SchemaResult<StringSpec> {
        let mut spec = StringSpec::default();
        if let Some(JsonValue::String(s)) = schema.get("format") {
            spec.format = Some(s.clone());
        }
        if let Some(JsonValue::String(s)) = schema.get("pattern") {
            spec.pattern = Some(s.clone());
        }
        if let Some(v) = schema.get("minLength") {
            match v {
                JsonValue::Number {
                    as_i64: Some(i), ..
                } => spec.min_length = *i,
                _ => return Err(SchemaError::invalid("minLength must be an integer")),
            }
        }
        if let Some(v) = schema.get("maxLength") {
            match v {
                JsonValue::Number {
                    as_i64: Some(i), ..
                } => spec.max_length = *i,
                _ => return Err(SchemaError::invalid("maxLength must be an integer")),
            }
        }
        if spec.max_length != -1 && spec.min_length > spec.max_length {
            return Err(SchemaError::unsatisfiable(format!(
                "minLength {} is greater than maxLength {}",
                spec.min_length, spec.max_length
            )));
        }
        Ok(spec)
    }
}
