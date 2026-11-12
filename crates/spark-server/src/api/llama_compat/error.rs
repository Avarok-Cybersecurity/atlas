// SPDX-License-Identifier: AGPL-3.0-only

//! llama.cpp's error envelope.
//!
//! `llama-server` answers every failure with
//! `{"error": {"code": <http status>, "message": "...", "type": "..."}}`
//! (`format_error_response` in `tools/server/server-common.cpp`). This is not
//! OpenAI's envelope: `code` is the integer status, and there is no `param`.
//! Clients written for llama.cpp read `error.message`, so the routes in this
//! module family answer in llama.cpp's shape.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

/// llama.cpp's `error_type` values that Atlas can produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ErrorKind {
    InvalidRequest,
    NotSupported,
    Unavailable,
    Server,
}

impl ErrorKind {
    /// The `error.type` string llama.cpp uses for this kind.
    pub(crate) fn type_str(self) -> &'static str {
        match self {
            ErrorKind::InvalidRequest => "invalid_request_error",
            ErrorKind::NotSupported => "not_supported_error",
            ErrorKind::Unavailable => "unavailable_error",
            ErrorKind::Server => "server_error",
        }
    }
}

/// A failure that renders as llama.cpp's error body.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompatError {
    pub status: StatusCode,
    pub kind: ErrorKind,
    pub message: String,
}

impl CompatError {
    /// HTTP 400, `invalid_request_error`: the request itself is wrong.
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: ErrorKind::InvalidRequest,
            message: message.into(),
        }
    }

    /// HTTP 400 for a llama.cpp field that Atlas cannot honor. The message
    /// names the field so a client can drop it; the request is never run with
    /// the field silently ignored.
    pub(crate) fn unsupported(field: &str, why: &str) -> Self {
        Self::invalid(format!(
            "'{field}' is not supported by Atlas: {why}. \
             The request was rejected rather than run without it."
        ))
    }

    /// HTTP 501 for a whole endpoint that Atlas does not serve.
    pub(crate) fn not_implemented(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            kind: ErrorKind::NotSupported,
            message: message.into(),
        }
    }

    /// HTTP 500 for a failure inside the server.
    pub(crate) fn server(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: ErrorKind::Server,
            message: message.into(),
        }
    }

    /// HTTP 503 for a request that cannot be taken right now.
    pub(crate) fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            kind: ErrorKind::Unavailable,
            message: message.into(),
        }
    }

    /// The body: `{"error": {"code", "message", "type"}}`.
    pub(crate) fn body(&self) -> serde_json::Value {
        error_object(self.status, self.kind, &self.message)
    }
}

impl IntoResponse for CompatError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body())).into_response()
    }
}

/// The bare `error` object (also nested into `/health`'s 503 body).
pub(crate) fn error_object(
    status: StatusCode,
    kind: ErrorKind,
    message: &str,
) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "code": status.as_u16(),
            "message": message,
            "type": kind.type_str(),
        }
    })
}

/// Parse a request body as a JSON object, tolerating any `Content-Type`.
///
/// `llama-server` parses the body whatever the header says, so `curl -d` (which
/// sends `application/x-www-form-urlencoded`) works against it. An axum `Json`
/// extractor would answer 415 there; the compat routes take raw bytes instead.
pub(crate) fn parse_json_object(
    body: &[u8],
) -> Result<serde_json::Map<String, serde_json::Value>, CompatError> {
    match serde_json::from_slice::<serde_json::Value>(body) {
        Ok(serde_json::Value::Object(map)) => Ok(map),
        Ok(_) => Err(CompatError::invalid("request body must be a JSON object")),
        Err(e) => Err(CompatError::invalid(format!("invalid JSON body: {e}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_has_llama_cpp_shape() {
        let e = CompatError::invalid("bad");
        assert_eq!(
            e.body(),
            serde_json::json!({
                "error": {"code": 400, "message": "bad", "type": "invalid_request_error"}
            })
        );
    }

    #[test]
    fn code_tracks_status_and_kind() {
        let v = CompatError::not_implemented("no").body();
        assert_eq!(v["error"]["code"], 501);
        assert_eq!(v["error"]["type"], "not_supported_error");
        let v = CompatError::unavailable("loading").body();
        assert_eq!(v["error"]["code"], 503);
        assert_eq!(v["error"]["type"], "unavailable_error");
    }

    #[test]
    fn unsupported_names_the_field_and_is_a_400() {
        let e = CompatError::unsupported("mirostat", "Atlas has no Mirostat sampler");
        assert_eq!(e.status, StatusCode::BAD_REQUEST);
        assert!(e.message.contains("'mirostat'"));
        assert!(e.message.contains("rejected"));
    }

    #[test]
    fn non_object_and_garbage_bodies_are_400s() {
        assert!(parse_json_object(b"[1,2]").is_err());
        assert!(parse_json_object(b"not json").is_err());
        assert!(parse_json_object(b"").is_err());
        assert!(parse_json_object(b"{\"a\":1}").is_ok());
    }
}
