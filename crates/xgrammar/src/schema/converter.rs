// SPDX-License-Identifier: AGPL-3.0-only
//
// JSONSchemaConverter — port of `class JSONSchemaConverter` and
// `class XMLToolCallingConverter` from `cpp/json_schema_converter.cc`
// and `cpp/json_schema_converter_ext.cc`.
//
// C++ uses virtual-method overriding for the XML tool-calling format.
// We collapse that single inheritance into one struct that branches
// on `format` / `nested_object_level`, which is the same dispatch the
// C++ vtable performs at runtime.
//
// This file holds the converter state, basic-rule registration, the
// scalar generators and the dispatch entry points; arrays, objects
// and composites live in `gen_array.rs` / `gen_object.rs` /
// `gen_composite.rs`.

use super::error::SchemaResult;
use super::indent::IndentManager;
use super::options::{JsonFormat, SchemaConverterOptions};
use super::parser::SchemaParser;
use super::script::EbnfScriptCreator;
use super::spec::{SchemaSpec, SchemaSpecPtr, SpecKind};
use std::collections::HashMap;

/// Basic helper rule names — match the C++ `kBasic*` constants.
pub(super) const BASIC_ANY: &str = "basic_any";
pub(super) const BASIC_INTEGER: &str = "basic_integer";
pub(super) const BASIC_NUMBER: &str = "basic_number";
pub(super) const BASIC_STRING: &str = "basic_string";
pub(super) const BASIC_BOOLEAN: &str = "basic_boolean";
pub(super) const BASIC_NULL: &str = "basic_null";
pub(super) const BASIC_ARRAY: &str = "basic_array";
pub(super) const BASIC_OBJECT: &str = "basic_object";
pub(super) const BASIC_ESCAPE: &str = "basic_escape";
pub(super) const BASIC_STRING_SUB: &str = "basic_string_sub";

/// XML-format helper rule names — match the C++ `kXML*` constants.
pub(super) const XML_STRING: &str = "xml_string";
pub(super) const XML_ANY: &str = "xml_any";
pub(super) const XML_OBJECT: &str = "xml_object";
pub(super) const XML_VARIABLE_NAME: &str = "xml_variable_name";

/// XML tag wrapper strings: `(key_prefix, key_suffix, param_suffix)`.
pub(super) fn xml_wrapper(format: JsonFormat) -> (&'static str, &'static str, &'static str) {
    match format {
        JsonFormat::QwenXml => ("<parameter=", ">", "</parameter>"),
        JsonFormat::MiniMaxXml => ("<parameter name=\\\"", "\\\">", "</parameter>"),
        JsonFormat::DeepSeekXml => (
            "<｜DSML｜parameter name=\\\"",
            "\\\" string=\\\"\" (\"true\" | \"false\") \"\\\">",
            "</｜DSML｜parameter>",
        ),
        JsonFormat::Json => ("", "", ""),
    }
}

/// Converts a parsed [`SchemaSpec`] tree into an EBNF grammar string.
pub struct JsonSchemaConverter<'p> {
    pub(super) script: EbnfScriptCreator,
    pub(super) indent: IndentManager,
    pub(super) colon_pattern: String,
    pub(super) any_whitespace: bool,
    pub(super) max_whitespace_cnt: Option<i32>,
    pub(super) format: JsonFormat,
    /// Saved snapshot of `indent`, restored after basic-rule setup.
    pub(super) saved_indent: Option<IndentManager>,
    /// `cache_key` -> rule name. The C++ keys this on
    /// `(key, is_inner_layer)`; for XML the inner layer corresponds to
    /// `nested_object_level > 1`.
    pub(super) cache_outer: HashMap<String, String>,
    pub(super) cache_inner: HashMap<String, String>,
    /// `$ref` URI -> rule name (circular-reference handling).
    pub(super) uri_to_rule: HashMap<String, String>,
    /// XML nesting depth (0 for the pure-JSON converter).
    pub(super) nested_object_level: i32,
    /// The parser, used as the `$ref` resolver at generate time.
    pub(super) parser: &'p mut SchemaParser,
}

impl<'p> JsonSchemaConverter<'p> {
    /// Build a converter from the resolved options.
    pub fn new(options: &SchemaConverterOptions, parser: &'p mut SchemaParser) -> Self {
        let (comma, colon) = options.resolved_separators();
        let indent = IndentManager::new(
            options.indent,
            &comma,
            options.any_whitespace,
            options.max_whitespace_cnt,
        );
        let colon_pattern = if options.any_whitespace {
            let ws = match options.max_whitespace_cnt {
                None => "[ \\n\\t]*".to_string(),
                Some(n) => format!("[ \\n\\t]{{0,{n}}}"),
            };
            format!("{ws} \"{colon}\" {ws}")
        } else {
            format!("\"{colon}\"")
        };
        JsonSchemaConverter {
            script: EbnfScriptCreator::new(),
            indent,
            colon_pattern,
            any_whitespace: options.any_whitespace,
            max_whitespace_cnt: options.max_whitespace_cnt,
            format: options.json_format,
            saved_indent: None,
            cache_outer: HashMap::new(),
            cache_inner: HashMap::new(),
            uri_to_rule: HashMap::new(),
            nested_object_level: 0,
            parser,
        }
    }

    /// Whether this converter emits the XML tool-calling format.
    pub(super) fn is_xml(&self) -> bool {
        self.format.is_xml()
    }

    /// Whether XML-style emission applies at the current nesting
    /// depth (root or first level).
    pub(super) fn at_xml_layer(&self) -> bool {
        self.is_xml() && self.nested_object_level <= 1
    }

    /// The whitespace fragment used between JSON tokens.
    pub(super) fn whitespace_pattern(&self) -> String {
        match self.max_whitespace_cnt {
            None => "[ \\n\\t]*".to_string(),
            Some(n) => format!("[ \\n\\t]{{0,{n}}}"),
        }
    }

    /// Add a `(cache_key -> rule_name)` mapping in the layer-correct
    /// cache. Port of `AddCache`.
    pub(super) fn add_cache(&mut self, key: &str, value: &str) {
        if key.is_empty() {
            return;
        }
        if self.nested_object_level > 1 {
            self.cache_inner.insert(key.to_string(), value.to_string());
        } else {
            self.cache_outer.insert(key.to_string(), value.to_string());
        }
    }

    /// Look up a cached rule name. Port of `GetCache`.
    pub(super) fn get_cache(&self, key: &str) -> Option<String> {
        if key.is_empty() {
            return None;
        }
        if self.nested_object_level > 1 {
            self.cache_inner.get(key).cloned()
        } else {
            self.cache_outer.get(key).cloned()
        }
    }

    /// The basic string rule name for the current layer. Port of
    /// `GetKeyPattern`.
    pub(super) fn key_pattern(&self) -> &'static str {
        if self.at_xml_layer() {
            XML_VARIABLE_NAME
        } else {
            BASIC_STRING
        }
    }

    /// The "any" rule name for the current layer. Port of
    /// `GetBasicAnyRuleName`.
    pub(super) fn basic_any_rule(&self) -> &'static str {
        if self.at_xml_layer() {
            XML_ANY
        } else {
            BASIC_ANY
        }
    }

    /// The next contextual separator. For XML layers it is empty.
    /// Port of the (overridden) `NextSeparator`.
    pub(super) fn next_separator(&mut self, is_end: bool) -> String {
        if self.at_xml_layer() {
            return String::new();
        }
        self.indent.next_separator(is_end)
    }

    /// Top-level conversion entry. Port of `Convert` (both the JSON
    /// and XML variants).
    pub fn convert(&mut self, spec: &SchemaSpecPtr) -> SchemaResult<String> {
        self.add_basic_rules()?;

        if self.is_xml() {
            self.nested_object_level = 0;
            let root_rule_name = self.script.allocate_rule_name("root");
            let body = self.generate_from_spec(spec, &root_rule_name)?;
            self.script
                .add_rule_with_allocated_name(&root_rule_name, &body);
            return Ok(self.script.get_script());
        }

        // JSON format: register `root` for `$ref: "#"`.
        let root_rule_name = self.script.allocate_rule_name("root");
        self.uri_to_rule
            .insert("#".to_string(), root_rule_name.clone());

        if let Some(cached) = self.get_cache(&spec.cache_key) {
            self.script
                .add_rule_with_allocated_name(&root_rule_name, &cached);
        } else {
            if !spec.cache_key.is_empty() {
                self.add_cache(&spec.cache_key, &root_rule_name);
            }
            let body = self.generate_from_spec(spec, &root_rule_name)?;
            self.script
                .add_rule_with_allocated_name(&root_rule_name, &body);
        }
        Ok(self.script.get_script())
    }

    /// Register the basic helper rules. Port of `AddBasicRules` /
    /// `AddHelperRules` for both formats.
    fn add_basic_rules(&mut self) -> SchemaResult<()> {
        // Helper rules (escape + string-sub) — always present.
        self.script.add_rule(
            BASIC_ESCAPE,
            "[\"\\\\/bfnrt] | \"u\" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]",
        );
        let ws = self.whitespace_pattern();
        self.script.add_rule(
            BASIC_STRING_SUB,
            &format!(
                "(\"\\\"\" | [^\\0-\\x1f\\\"\\\\\\r\\n] {BASIC_STRING_SUB} | \
                 \"\\\\\" {BASIC_ESCAPE} {BASIC_STRING_SUB}) (= {ws} [,}}\\]:])"
            ),
        );

        // Basic typed rules are built with a compact indent manager.
        let outer_xml = self.is_xml();
        if outer_xml {
            // C++ sets nested_object_level=2 so basic rules become the
            // JSON inner layer.
            self.nested_object_level = 2;
        }
        self.saved_indent = Some(self.indent.clone());
        self.indent = if self.any_whitespace {
            IndentManager::new(None, ",", true, None)
        } else {
            IndentManager::new(None, ", ", false, None)
        };

        self.register_basic_typed_rules()?;

        self.indent = self.saved_indent.take().unwrap();
        if outer_xml {
            // The XML string/any rules belong to the param layer (level 1);
            // the XML object rule belongs to the root layer (level 0).
            // (upstream commit 41dbbb1, #634).
            self.nested_object_level = 1;
            self.register_xml_basic_rules()?;
        }
        Ok(())
    }

    /// Register `basic_any/integer/number/string/boolean/null/array/
    /// object` and seed their cache keys.
    fn register_basic_typed_rules(&mut self) -> SchemaResult<()> {
        let any_spec = SchemaSpec::make(SpecKind::Any, "{}", BASIC_ANY);
        let any_body = self.generate_any(BASIC_ANY);
        self.script.add_rule(BASIC_ANY, &any_body);
        self.add_cache("{}", BASIC_ANY);

        let int_body = self.generate_integer(&super::spec::IntegerSpec::default());
        self.script.add_rule(BASIC_INTEGER, &int_body);
        self.add_cache("{\"type\":\"integer\"}", BASIC_INTEGER);

        let num_body = self.generate_number(&super::spec::NumberSpec::default());
        self.script.add_rule(BASIC_NUMBER, &num_body);
        self.add_cache("{\"type\":\"number\"}", BASIC_NUMBER);

        self.script
            .add_rule(BASIC_STRING, &format!("[\"] {BASIC_STRING_SUB}"));
        self.add_cache("{\"type\":\"string\"}", BASIC_STRING);

        self.script.add_rule(BASIC_BOOLEAN, "\"true\" | \"false\"");
        self.add_cache("{\"type\":\"boolean\"}", BASIC_BOOLEAN);

        self.script.add_rule(BASIC_NULL, "\"null\"");
        self.add_cache("{\"type\":\"null\"}", BASIC_NULL);

        let array_spec = super::spec::ArraySpec {
            allow_additional_items: true,
            additional_items: Some(any_spec.clone()),
            ..Default::default()
        };
        let array_body = self.generate_array(&array_spec, BASIC_ARRAY)?;
        self.script.add_rule(BASIC_ARRAY, &array_body);
        self.add_cache("{\"type\":\"array\"}", BASIC_ARRAY);

        let obj_spec = super::spec::ObjectSpec {
            allow_additional_properties: true,
            additional_properties_schema: Some(any_spec),
            ..Default::default()
        };
        let obj_body = self.generate_object(&obj_spec, BASIC_OBJECT, true)?;
        self.script.add_rule(BASIC_OBJECT, &obj_body);
        self.add_cache("{\"type\":\"object\"}", BASIC_OBJECT);
        Ok(())
    }

    /// Register the XML-format helper rules (`xml_string` etc.). Port
    /// of `XMLToolCallingConverter::AddBasicRules`.
    fn register_xml_basic_rules(&mut self) -> SchemaResult<()> {
        let (_, _, param_suffix) = xml_wrapper(self.format);
        self.script.add_rule(
            XML_STRING,
            &format!(
                "TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,\
                 excludes=(\"{param_suffix}\"))"
            ),
        );
        self.add_cache("{\"type\":\"string\"}", XML_STRING);

        let any_body = self.generate_any(XML_ANY);
        self.script.add_rule(XML_ANY, &any_body);
        self.add_cache("{}", XML_ANY);

        // Reset to the root layer (level 0) for the XML object rule
        // (upstream commit 41dbbb1, #634).
        self.nested_object_level = 0;

        let obj_spec = super::spec::ObjectSpec {
            allow_additional_properties: true,
            additional_properties_schema: Some(SchemaSpec::make(SpecKind::Any, "{}", XML_ANY)),
            ..Default::default()
        };
        let obj_body = self.generate_object(&obj_spec, XML_OBJECT, true)?;
        self.script.add_rule(XML_OBJECT, &obj_body);
        self.add_cache("{\"type\":\"object\"}", XML_OBJECT);

        self.script
            .add_rule(XML_VARIABLE_NAME, "[a-zA-Z_][a-zA-Z0-9_]*");
        Ok(())
    }

    /// Create a rule for `spec` (or return a cached rule name) and
    /// return its name. Port of `CreateRule`.
    pub(super) fn create_rule(&mut self, spec: &SchemaSpecPtr, hint: &str) -> SchemaResult<String> {
        if let Some(cached) = self.get_cache(&spec.cache_key) {
            return Ok(cached);
        }
        let rule_name = self.script.allocate_rule_name(hint);
        let body = self.generate_from_spec(spec, &rule_name)?;
        self.script.add_rule_with_allocated_name(&rule_name, &body);
        Ok(rule_name)
    }

    /// Dispatch to the per-kind generator. Port of `GenerateFromSpec`.
    pub(super) fn generate_from_spec(
        &mut self,
        spec: &SchemaSpecPtr,
        rule_name: &str,
    ) -> SchemaResult<String> {
        match &spec.kind {
            SpecKind::Integer(s) => Ok(self.generate_integer(s)),
            SpecKind::Number(s) => Ok(self.generate_number(s)),
            SpecKind::String(s) => self.generate_string(s),
            SpecKind::Boolean => Ok("\"true\" | \"false\"".to_string()),
            SpecKind::Null => Ok("\"null\"".to_string()),
            SpecKind::Array(s) => self.generate_array(s, rule_name),
            SpecKind::Object(s) => self.generate_object(s, rule_name, true),
            SpecKind::Any => Ok(self.generate_any(rule_name)),
            SpecKind::Const(v) => Ok(self.generate_const(v)),
            SpecKind::Enum(vs) => Ok(self.generate_enum(vs)),
            SpecKind::Ref(uri) => self.generate_ref(uri),
            SpecKind::AnyOf(opts) => self.generate_any_of(opts, rule_name),
            SpecKind::AllOf(schemas) => self.generate_all_of(schemas, rule_name),
            SpecKind::TypeArray(schemas) => self.generate_type_array(schemas, rule_name),
        }
    }
}
