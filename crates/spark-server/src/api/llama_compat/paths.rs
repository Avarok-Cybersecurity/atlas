// SPDX-License-Identifier: AGPL-3.0-only

//! Which llama.cpp routes the server middleware must treat as protected.
//!
//! The auth, rate-limit and GPU-fault layers decide by request path, and they
//! were written when every protected route lived under `/v1/` (plus
//! `/tokenize` and `/detokenize`). The llama.cpp routes sit at the root, so
//! without an entry here `--require-auth` would leave `/completion` open: a
//! text-generation endpoint with no token check. Keep this list next to the
//! router registration in `main_modules/serve_router.rs`.

/// Routes that run the model: they take a rate-limit slot and are refused with
/// 503 once the GPU fault latch is set.
pub(crate) fn is_inference_path(path: &str) -> bool {
    matches!(path, "/completion" | "/completions")
}

/// Routes that need the bearer token when `--require-auth` is on. This is the
/// inference routes plus the ones that read the model's templates and
/// configuration. `/health` stays public, as in llama.cpp.
pub(crate) fn needs_auth(path: &str) -> bool {
    is_inference_path(path)
        || matches!(
            path,
            "/apply-template" | "/props" | "/embedding" | "/embeddings"
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_routes_are_inference_and_gated() {
        for p in ["/completion", "/completions"] {
            assert!(is_inference_path(p), "{p}");
            assert!(needs_auth(p), "{p}");
        }
    }

    #[test]
    fn template_and_props_need_auth_but_do_not_run_the_model() {
        for p in ["/apply-template", "/props", "/embedding", "/embeddings"] {
            assert!(needs_auth(p), "{p}");
            assert!(!is_inference_path(p), "{p}");
        }
    }

    #[test]
    fn health_stays_public_and_unrelated_paths_are_untouched() {
        for p in ["/health", "/health/live", "/metrics", "/completion/x", "/"] {
            assert!(!needs_auth(p), "{p}");
            assert!(!is_inference_path(p), "{p}");
        }
    }
}
