#!/usr/bin/env bash
# Pre-warm a SHARED cargo build cache for the webserver_ok scorer.
#
# WHY
#   The scorer (score_run.py:webserver_test) runs `cargo build --release` on
#   each generated Axum project. On a COLD cache that cold-compiles the full
#   dependency tree (libc, proc-macro2, hyper, tokio, axum, …) — ~150-300s
#   under CPU contention, which blows the scorer build timeout and mislabels
#   a VALID generation as build_ok=false. That is an ENVIRONMENTAL artifact,
#   not a model failure.
#
# WHAT
#   Builds a template Axum project that imports the union of dependencies the
#   model's generations actually use (axum, tokio, serde, serde_json, tower,
#   hyper, reqwest, tracing*). The build populates two shared artifacts:
#     1. ${CARGO_HOME}/registry  — downloaded + extracted crate sources.
#     2. ${ATLAS_WARM_TARGET_DIR} — COMPILED dependency rlibs (the slow part).
#   The scorer exports CARGO_TARGET_DIR=${ATLAS_WARM_TARGET_DIR} so every
#   per-project build reuses the already-compiled deps and only recompiles
#   the project's own tiny crate — seconds, not minutes.
#
# SSOT
#   ATLAS_WARM_TARGET_DIR is the single source of truth for the warm target
#   path; both this script and score_run.py read the same env var (with the
#   same explicit default), so the two never drift.
#
# Idempotent: re-running is a fast no-op once the cache is warm.
set -euo pipefail

WARM_TARGET_DIR="${ATLAS_WARM_TARGET_DIR:-${HOME}/.cargo/atlas-warm-target}"
TEMPLATE_DIR="${ATLAS_WARM_TEMPLATE_DIR:-${HOME}/.cargo/atlas-warm-template}"

echo "[warm] warm target dir : ${WARM_TARGET_DIR}" >&2
echo "[warm] template project: ${TEMPLATE_DIR}" >&2

mkdir -p "${TEMPLATE_DIR}/src"

# Dependency UNION across observed generations. Versions are left to cargo's
# resolver (caret ranges) so a warm rlib matches whatever a generation pins
# within the same minor — the registry + compiled std deps are shared even if
# the leaf crate version differs slightly.
cat > "${TEMPLATE_DIR}/Cargo.toml" <<'TOML'
[package]
name = "atlas-warm-template"
version = "0.1.0"
edition = "2021"

[dependencies]
axum = "0.8"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tower = "0.5"
hyper = { version = "1", features = ["full"] }
reqwest = { version = "0.12", features = ["json"] }
tracing = "0.1"
tracing-subscriber = "0.3"

[dev-dependencies]
reqwest = { version = "0.12", features = ["json"] }
TOML

cat > "${TEMPLATE_DIR}/src/main.rs" <<'RUST'
// Touches each dependency so its rlib is compiled into the warm target dir.
use axum::{routing::get, Router};

async fn ping() -> &'static str {
    "pong"
}

#[tokio::main]
async fn main() {
    let _ = serde_json::json!({"ok": true});
    let _v: tower::ServiceBuilder<tower::layer::util::Identity> = tower::ServiceBuilder::new();
    let app = Router::new().route("/ping", get(ping));
    let port: u16 = std::env::var("ATLAS_HARNESS_PORT")
        .unwrap_or_else(|_| "3001".to_string())
        .parse()
        .unwrap();
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
RUST

mkdir -p "${WARM_TARGET_DIR}"

echo "[warm] compiling dependency tree into the shared target dir (cold = slow, once)..." >&2
CARGO_TARGET_DIR="${WARM_TARGET_DIR}" cargo build --release \
    --manifest-path "${TEMPLATE_DIR}/Cargo.toml" >&2

echo "[warm] warm cache ready." >&2
du -sh "${WARM_TARGET_DIR}" 2>/dev/null | sed 's/^/[warm] target dir size: /' >&2 || true
