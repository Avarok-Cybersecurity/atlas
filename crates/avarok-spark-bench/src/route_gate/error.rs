// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the error type shared by the route_gate modules (the Python harness raised `ValueError`,
//! `KeyError` and four custom subclasses; the variants below keep those distinctions for the tests).

/// Every failure a route_gate library function reports.
#[derive(Debug, thiserror::Error)]
pub enum GateError {
    /// The Stage-1 tensors select a branch of the reference head that this port does not implement (or have the
    /// wrong names or shapes).
    #[error("{0}")]
    Stage1Unsupported(String),
    /// The release directory is not one consistent release, or its pinned hashes do not match.
    #[error("{0}")]
    ReleaseRefused(String),
    /// The file is not a well-formed safetensors file, or a requested tensor is not in it.
    #[error("{0}")]
    Safetensors(String),
    /// A bad argument or a bad shape (Python `ValueError`).
    #[error("{0}")]
    Value(String),
    /// A name that matches nothing, or too many things (Python `KeyError`).
    #[error("{0}")]
    Key(String),
    /// An I/O failure.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// A JSON failure.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

/// Result alias used by the library code.
pub type GResult<T> = Result<T, GateError>;

/// `Err(GateError::Value(msg))`.
pub fn value_err<T>(msg: impl Into<String>) -> GResult<T> {
    Err(GateError::Value(msg.into()))
}

/// `Err(GateError::Key(msg))`.
pub fn key_err<T>(msg: impl Into<String>) -> GResult<T> {
    Err(GateError::Key(msg.into()))
}
