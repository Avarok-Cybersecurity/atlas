// SPDX-License-Identifier: AGPL-3.0-only
//
// SchemaSpec intermediate representation — port of the `SchemaSpec`
// variant family from `cpp/json_schema_converter.h`.
//
// The C++ uses `std::variant` wrapped in `std::shared_ptr<SchemaSpec>`
// for deduplication and circular-reference handling. We mirror that
// with an `enum SpecKind` inside an `Rc<SchemaSpec>`.

use std::rc::Rc;

/// Shared, ref-counted handle to a parsed schema spec — the Rust
/// analogue of C++ `SchemaSpecPtr`.
pub type SchemaSpecPtr = Rc<SchemaSpec>;

/// Integer-type constraints (`cpp` `IntegerSpec`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IntegerSpec {
    pub minimum: Option<i64>,
    pub maximum: Option<i64>,
    pub exclusive_minimum: Option<i64>,
    pub exclusive_maximum: Option<i64>,
}

/// Number-type constraints (`cpp` `NumberSpec`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NumberSpec {
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub exclusive_minimum: Option<f64>,
    pub exclusive_maximum: Option<f64>,
}

/// String-type constraints (`cpp` `StringSpec`).
#[derive(Debug, Clone, PartialEq)]
pub struct StringSpec {
    pub pattern: Option<String>,
    pub format: Option<String>,
    /// `0` means no lower bound.
    pub min_length: i64,
    /// `-1` means no upper bound.
    pub max_length: i64,
}

impl Default for StringSpec {
    fn default() -> Self {
        StringSpec {
            pattern: None,
            format: None,
            min_length: 0,
            max_length: -1,
        }
    }
}

/// A named object property.
#[derive(Debug, Clone)]
pub struct Property {
    pub name: String,
    pub schema: SchemaSpecPtr,
}

/// A `patternProperties` entry — a regex key pattern + value schema.
#[derive(Debug, Clone)]
pub struct PatternProperty {
    pub pattern: String,
    pub schema: SchemaSpecPtr,
}

/// Array-type constraints (`cpp` `ArraySpec`).
#[derive(Debug, Clone)]
pub struct ArraySpec {
    pub prefix_items: Vec<SchemaSpecPtr>,
    pub allow_additional_items: bool,
    /// `None` means additional items are not allowed.
    pub additional_items: Option<SchemaSpecPtr>,
    pub min_items: i64,
    /// `-1` means no upper bound.
    pub max_items: i64,
}

impl Default for ArraySpec {
    fn default() -> Self {
        ArraySpec {
            prefix_items: Vec::new(),
            allow_additional_items: true,
            additional_items: None,
            min_items: 0,
            max_items: -1,
        }
    }
}

/// Object-type constraints (`cpp` `ObjectSpec`).
#[derive(Debug, Clone)]
pub struct ObjectSpec {
    pub properties: Vec<Property>,
    pub pattern_properties: Vec<PatternProperty>,
    pub required: Vec<String>,
    pub allow_additional_properties: bool,
    pub additional_properties_schema: Option<SchemaSpecPtr>,
    pub allow_unevaluated_properties: bool,
    pub unevaluated_properties_schema: Option<SchemaSpecPtr>,
    pub property_names: Option<SchemaSpecPtr>,
    pub min_properties: i64,
    /// `-1` means no upper bound.
    pub max_properties: i64,
}

impl Default for ObjectSpec {
    fn default() -> Self {
        ObjectSpec {
            properties: Vec::new(),
            pattern_properties: Vec::new(),
            required: Vec::new(),
            allow_additional_properties: false,
            additional_properties_schema: None,
            allow_unevaluated_properties: true,
            unevaluated_properties_schema: None,
            property_names: None,
            min_properties: 0,
            max_properties: -1,
        }
    }
}

impl ObjectSpec {
    /// Whether `name` is in the `required` set. Part of the faithful
    /// `ObjectSpec` surface; the generators inline the equivalent
    /// check for clarity.
    #[allow(dead_code)]
    pub fn is_required(&self, name: &str) -> bool {
        self.required.iter().any(|r| r == name)
    }
}

/// The variant payload of a [`SchemaSpec`] — mirrors the C++
/// `SchemaSpecVariant`.
#[derive(Debug, Clone)]
pub enum SpecKind {
    Integer(IntegerSpec),
    Number(NumberSpec),
    String(StringSpec),
    Boolean,
    Null,
    Array(ArraySpec),
    Object(ObjectSpec),
    /// Matches any JSON value.
    Any,
    /// A single constant value, stored as serialized JSON.
    Const(String),
    /// An enumeration of constant values, each serialized JSON.
    Enum(Vec<String>),
    /// A `$ref` URI, resolved lazily at generation time.
    Ref(String),
    /// `anyOf` / `oneOf`.
    AnyOf(Vec<SchemaSpecPtr>),
    /// `allOf`.
    AllOf(Vec<SchemaSpecPtr>),
    /// `"type": ["string", "integer", ...]`.
    TypeArray(Vec<SchemaSpecPtr>),
}

/// A parsed schema node: the variant payload plus a deduplication
/// cache key and a suggested rule name.
#[derive(Debug, Clone)]
pub struct SchemaSpec {
    pub kind: SpecKind,
    /// Canonical cache key used to deduplicate identical sub-schemas.
    pub cache_key: String,
    /// Suggested rule name for the generated EBNF rule. Retained from
    /// the C++ IR; the generators pass explicit hints, so this is
    /// informational only.
    #[allow(dead_code)]
    pub rule_name_hint: String,
}

impl SchemaSpec {
    /// Build a shared spec — the Rust analogue of `SchemaSpec::Make`.
    pub fn make(
        kind: SpecKind,
        cache_key: impl Into<String>,
        hint: impl Into<String>,
    ) -> SchemaSpecPtr {
        Rc::new(SchemaSpec {
            kind,
            cache_key: cache_key.into(),
            rule_name_hint: hint.into(),
        })
    }
}
