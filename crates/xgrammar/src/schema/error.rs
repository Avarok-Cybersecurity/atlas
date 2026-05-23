// SPDX-License-Identifier: AGPL-3.0-only
//
// Error type for the JSON-schema -> EBNF converter — port of the
// internal `SchemaError` / `SchemaErrorType` from
// `cpp/json_schema_converter.cc`.
//
// The C++ converter aborts the process (`XGRAMMAR_LOG(FATAL)`) on a
// malformed schema; here we return `Err(SchemaError)` — no panics.

use std::fmt;

/// Classification of a schema-conversion failure, mirroring the C++
/// `SchemaErrorType` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaErrorKind {
    /// The schema document is structurally invalid (wrong JSON type
    /// for a keyword, non-string `type`, unparsable JSON, ...).
    InvalidSchema,
    /// The schema is well-formed but cannot accept any value (e.g.
    /// `false`, `minItems > maxItems`).
    UnsatisfiableSchema,
    /// A `$ref` could not be resolved or a regex/sub-conversion
    /// failed. (`InvalidSchema` is reused in C++; we keep a dedicated
    /// variant to make ref failures easy to spot.)
    RefResolution,
}

/// An error raised while converting a JSON Schema document into an
/// EBNF grammar string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaError {
    /// The category of the failure.
    pub kind: SchemaErrorKind,
    /// Human-readable description.
    pub message: String,
}

impl SchemaError {
    /// Construct an `InvalidSchema` error.
    pub fn invalid(message: impl Into<String>) -> Self {
        SchemaError {
            kind: SchemaErrorKind::InvalidSchema,
            message: message.into(),
        }
    }

    /// Construct an `UnsatisfiableSchema` error.
    pub fn unsatisfiable(message: impl Into<String>) -> Self {
        SchemaError {
            kind: SchemaErrorKind::UnsatisfiableSchema,
            message: message.into(),
        }
    }

    /// Construct a `RefResolution` error.
    pub fn ref_error(message: impl Into<String>) -> Self {
        SchemaError {
            kind: SchemaErrorKind::RefResolution,
            message: message.into(),
        }
    }
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tag = match self.kind {
            SchemaErrorKind::InvalidSchema => "invalid schema",
            SchemaErrorKind::UnsatisfiableSchema => "unsatisfiable schema",
            SchemaErrorKind::RefResolution => "ref resolution",
        };
        write!(f, "{tag}: {}", self.message)
    }
}

impl std::error::Error for SchemaError {}

/// Convenience alias for converter results.
pub type SchemaResult<T> = Result<T, SchemaError>;
