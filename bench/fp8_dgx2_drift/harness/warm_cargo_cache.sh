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
# Feature sets are a SUPERSET of what generations request. Cargo keys a
# cached rlib by (crate, version, feature-set, profile); if the agent's
# project enables a feature the warm rlib lacks (e.g. tower "util",
# axum "json"), cargo recompiles that crate from scratch and the warm hit
# is lost. Enabling the union here keeps the agent's build a pure
# incremental link of its own crate.
axum = { version = "0.8", features = ["json", "macros", "ws", "multipart"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tower = { version = "0.5", features = ["full"] }
tower-http = { version = "0.6", features = ["full"] }
hyper = { version = "1", features = ["full"] }
reqwest = { version = "0.12", features = ["json"] }
anyhow = "1"
thiserror = "2"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

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

# Pre-warm EVERY cargo profile the agent actually drives. Measured agent tool
# calls over a N=10 webserver_ok tier: `cargo test` (debug) 141×, `cargo run`
# (debug) 97×, `cargo build --release` 25×, `cargo run --release` 25×. The
# debug profile compiles a SEPARATE set of dep rlibs from release; warming
# only release (the old behaviour) left every debug `cargo test`/`cargo run`
# cold-compiling axum/tokio/hyper — ~5s idle but 150-250s under the model's
# memory-pressure swap thrash, which was the entire 92s↔305s wall variance.
#
#   cargo test   → warms the DEBUG profile rlibs + the test-harness link
#   cargo build  → warms the RELEASE profile rlibs
# Both share ${WARM_TARGET_DIR}; the agent's CARGO_TARGET_DIR points here so
# its builds become ~1.4s incremental links of just the project's own crate.
echo "[warm] compiling DEBUG profile (cargo test) into shared target dir..." >&2
CARGO_TARGET_DIR="${WARM_TARGET_DIR}" cargo test --no-run \
    --manifest-path "${TEMPLATE_DIR}/Cargo.toml" >&2

echo "[warm] compiling RELEASE profile (cargo build --release) into shared target dir..." >&2
CARGO_TARGET_DIR="${WARM_TARGET_DIR}" cargo build --release \
    --manifest-path "${TEMPLATE_DIR}/Cargo.toml" >&2

echo "[warm] warm cache ready (debug + release profiles)." >&2
du -sh "${WARM_TARGET_DIR}" 2>/dev/null | sed 's/^/[warm] target dir size: /' >&2 || true
