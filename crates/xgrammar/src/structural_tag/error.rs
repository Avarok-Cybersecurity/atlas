// SPDX-License-Identifier: AGPL-3.0-only
//
// Error type for the structural-tag subsystem — port of the
// `StructuralTagError` family (`InvalidJSONError`,
// `InvalidStructuralTagError`) from `cpp/exception.h`.
//
// The C++ uses a `Result<T, StructuralTagError>` where the error is a
// `std::variant` of the two error classes. Here a single `enum` carries
// the same distinction plus a human-readable message, returned as a
// plain `Result` — no panics, no `unsafe`.

/// Error produced while parsing or converting a structural tag.
///
/// Port of `StructuralTagError`. The two variants mirror the C++
/// `InvalidJSONError` (the structural-tag JSON document is not valid
/// JSON) and `InvalidStructuralTagError` (the JSON is well-formed but
/// is not a valid structural tag).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StructuralTagError {
    /// The structural-tag document failed JSON parsing.
    #[error("invalid structural tag JSON: {0}")]
    InvalidJson(String),
    /// The document parsed as JSON but is not a valid structural tag,
    /// or grammar construction failed.
    #[error("invalid structural tag: {0}")]
    InvalidStructuralTag(String),
}

impl StructuralTagError {
    /// Construct an [`StructuralTagError::InvalidJson`].
    pub(crate) fn json(msg: impl Into<String>) -> Self {
        StructuralTagError::InvalidJson(msg.into())
    }

    /// Construct an [`StructuralTagError::InvalidStructuralTag`].
    pub(crate) fn invalid(msg: impl Into<String>) -> Self {
        StructuralTagError::InvalidStructuralTag(msg.into())
    }
}

/// Convenience alias for results in this module.
pub type StructuralTagResult<T> = Result<T, StructuralTagError>;
