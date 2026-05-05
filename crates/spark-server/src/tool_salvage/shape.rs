// SPDX-License-Identifier: AGPL-3.0-only

// ── Internal: per-tool schema introspection ────────────────────

use crate::tool_parser::ToolDefinition;

/// Cached projection of a `ToolDefinition` for matching purposes.
pub(super) struct ToolShape<'a> {
    def: &'a ToolDefinition,
    /// Lowercased property names declared in the schema.
    properties: Vec<String>,
    /// Lowercased required property names.
    required: Vec<String>,
}

impl<'a> ToolShape<'a> {
    pub(super) fn new(def: &'a ToolDefinition) -> Self {
        let mut properties = Vec::new();
        let mut required = Vec::new();
        if let Some(params) = &def.function.parameters {
            if let Some(props) = params.get("properties").and_then(|v| v.as_object()) {
                for k in props.keys() {
                    properties.push(k.to_ascii_lowercase());
                }
            }
            if let Some(reqs) = params.get("required").and_then(|v| v.as_array()) {
                for v in reqs {
                    if let Some(s) = v.as_str() {
                        required.push(s.to_ascii_lowercase());
                    }
                }
            }
        }
        Self {
            def,
            properties,
            required,
        }
    }

    pub(super) fn name(&self) -> &str {
        &self.def.function.name
    }

    pub(super) fn name_lower(&self) -> String {
        self.name().to_ascii_lowercase()
    }

    /// Original property name (preserved case) matching `lower_name`,
    /// if any. Used to emit arguments in the schema's expected case.
    pub(super) fn original_property(&self, lower_name: &str) -> Option<String> {
        if let Some(params) = &self.def.function.parameters
            && let Some(props) = params.get("properties").and_then(|v| v.as_object())
        {
            for k in props.keys() {
                if k.to_ascii_lowercase() == lower_name {
                    return Some(k.clone());
                }
            }
        }
        None
    }

    /// Identify a single required string parameter (the canonical
    /// shape for command-runners like bash). Returns `None` when
    /// the schema has zero, two, or more required parameters.
    pub(super) fn single_required_string(&self) -> Option<String> {
        if self.required.len() != 1 {
            return None;
        }
        let name = &self.required[0];
        if self.is_string_property(name) {
            self.original_property(name)
        } else {
            None
        }
    }

    fn is_string_property(&self, lower_name: &str) -> bool {
        let Some(params) = &self.def.function.parameters else {
            return false;
        };
        let Some(props) = params.get("properties").and_then(|v| v.as_object()) else {
            return false;
        };
        for (k, v) in props {
            if k.to_ascii_lowercase() != lower_name {
                continue;
            }
            // Accept either explicit "type": "string" or untyped
            // schemas (assume string-ish — can't refute).
            if let Some(t) = v.get("type") {
                if t.as_str() == Some("string") {
                    return true;
                }
                if let Some(arr) = t.as_array() {
                    return arr.iter().any(|x| x.as_str() == Some("string"));
                }
                return false;
            }
            return true;
        }
        false
    }

    /// Identify path-like + content-like property names — the
    /// "write a file" shape. Returns `(path_prop, content_prop)` in
    /// schema casing, or `None`.
    pub(super) fn path_and_content(&self) -> Option<(String, String)> {
        const PATH_NAMES: &[&str] = &["path", "file_path", "filepath", "file"];
        const CONTENT_NAMES: &[&str] = &[
            "content",
            "text",
            "body",
            "data",
            "file_content",
            "filecontent",
        ];
        let path = PATH_NAMES.iter().find_map(|n| {
            if self.properties.iter().any(|p| p == n) {
                self.original_property(n)
            } else {
                None
            }
        })?;
        let content = CONTENT_NAMES.iter().find_map(|n| {
            if self.properties.iter().any(|p| p == n) {
                self.original_property(n)
            } else {
                None
            }
        })?;
        Some((path, content))
    }
}
