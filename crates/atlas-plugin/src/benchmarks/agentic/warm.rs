// SPDX-License-Identifier: AGPL-3.0-only

//! The shared warm cargo target dir — a port of the harness's
//! `warm_cargo_cache.sh`, plus the wiring `run_tier.sh` puts around it.
//!
//! `warm_cargo_cache.sh`'s own WHY: on a COLD cache each generated Axum project
//! cold-compiles the full dependency tree (libc, proc-macro2, hyper, tokio,
//! axum, …) — "~150-300s under CPU contention, which blows the scorer build
//! timeout and mislabels a VALID generation as build_ok=false. That is an
//! ENVIRONMENTAL artifact, not a model failure."
//!
//! `run_tier.sh:75-96` adds the half that bites hardest here: the AGENT's own
//! builds must hit the same dir. It measured a tier driving `cargo test` 141×
//! and `cargo run` 97× — both DEBUG profile, which compiles a *separate* set of
//! rlibs from release — and records that warming release only "left every debug
//! `cargo test`/`cargo run` cold-compiling axum/tokio/hyper … which was the
//! entire 92s↔305s wall variance."
//!
//! So this module warms BOTH profiles, from a template whose feature sets are a
//! SUPERSET of what generations request, and hands back the one path the agent
//! and the scorer both point `CARGO_TARGET_DIR` at.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, bail};

use crate::plugin::PluginHandle;

/// A cold warm-up is minutes, not seconds — `warm_cargo_cache.sh` budgets
/// "~150-300s under CPU contention" for one profile, and we run two. The ceiling
/// exists only so a wedged `cargo` cannot hang the benchmark forever.
const PREWARM_TIMEOUT: Duration = Duration::from_secs(1800);

/// Verbatim from `warm_cargo_cache.sh`. Its comment is the specification: cargo
/// keys a cached rlib by (crate, version, feature-set, profile), so if the
/// agent's project enables a feature the warm rlib lacks, cargo recompiles that
/// crate from scratch and the warm hit is lost. The union is what keeps the
/// agent's build a pure incremental link of its own crate.
pub const TEMPLATE_MANIFEST: &str = r#"[package]
name = "atlas-warm-template"
version = "0.1.0"
edition = "2021"

[dependencies]
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
"#;

/// Touches each dependency so its rlib is actually compiled into the warm dir —
/// a manifest entry alone does not build a crate.
pub const TEMPLATE_MAIN: &str = r#"use axum::{routing::get, Router};

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
"#;

/// Resolve one of the harness's two `${VAR:-${HOME}/.cargo/<leaf>}` paths.
///
/// Kept pure (env read by the callers) so the SSOT default can be asserted in a
/// test without mutating process environment.
fn dir_from(explicit: Option<OsString>, home: Option<OsString>, leaf: &str) -> Result<PathBuf> {
    if let Some(p) = explicit.filter(|p| !p.is_empty()) {
        return Ok(PathBuf::from(p));
    }
    let home = home
        .filter(|h| !h.is_empty())
        .context("HOME is not set, so the shared warm cargo dir has no home")?;
    Ok(PathBuf::from(home).join(".cargo").join(leaf))
}

/// SSOT with `warm_cargo_cache.sh`, `score_run.py::_warm_target_dir` and
/// `run_tier.sh:100` — same env var, same explicit default, so the four never
/// drift. Pointing somewhere else is what made this benchmark cold on a box
/// where the harness's cache was already warm.
pub fn warm_target_dir() -> Result<PathBuf> {
    dir_from(
        std::env::var_os("ATLAS_WARM_TARGET_DIR"),
        std::env::var_os("HOME"),
        "atlas-warm-target",
    )
}

/// SSOT with `warm_cargo_cache.sh`'s `ATLAS_WARM_TEMPLATE_DIR`.
pub fn template_dir() -> Result<PathBuf> {
    dir_from(
        std::env::var_os("ATLAS_WARM_TEMPLATE_DIR"),
        std::env::var_os("HOME"),
        "atlas-warm-template",
    )
}

/// Materialise the template project. Idempotent, like the shell script — the
/// second call is a no-op so a warm box pays nothing.
pub fn write_template(dir: &Path) -> Result<()> {
    let src = dir.join("src");
    std::fs::create_dir_all(&src)
        .with_context(|| format!("creating warm template dir {}", src.display()))?;
    write_if_changed(&dir.join("Cargo.toml"), TEMPLATE_MANIFEST)?;
    write_if_changed(&src.join("main.rs"), TEMPLATE_MAIN)?;
    Ok(())
}

/// Rewriting an identical file would bump its mtime and make cargo rebuild the
/// template crate on every single run.
fn write_if_changed(path: &Path, content: &str) -> Result<()> {
    if std::fs::read_to_string(path).is_ok_and(|old| old == content) {
        return Ok(());
    }
    std::fs::write(path, content).with_context(|| format!("writing {}", path.display()))
}

/// One `cargo` invocation against the template, output into the warm dir.
async fn cargo(args: &[&str], warm: &Path, manifest: &Path) -> Result<()> {
    let mut cmd = tokio::process::Command::new("cargo");
    cmd.args(args)
        .arg("--manifest-path")
        .arg(manifest)
        // Network stays ON (no `CARGO_NET_OFFLINE`) for the reason
        // `run_tier.sh:103-106` gives: a generation that pins a dep version
        // outside the pre-warmed set must still resolve, or it would be a false
        // build failure.
        .env("CARGO_TARGET_DIR", warm)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let out = tokio::time::timeout(PREWARM_TIMEOUT, cmd.output())
        .await
        .with_context(|| {
            format!(
                "`cargo {}` exceeded {}s while warming {}",
                args.join(" "),
                PREWARM_TIMEOUT.as_secs(),
                warm.display()
            )
        })?
        .with_context(|| format!("`cargo {}` could not start", args.join(" ")))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let tail = err.lines().rev().take(8).collect::<Vec<_>>().join(" ");
        bail!(
            "`cargo {}` failed while warming {}: {}",
            args.join(" "),
            warm.display(),
            super::super::one_line(tail)
        );
    }
    Ok(())
}

/// Prepare the shared warm target dir and return it.
///
/// Fails loudly rather than falling back to a cold dir: a silently cold cache is
/// exactly the environmental artifact this exists to remove, and it does not
/// announce itself — it just makes the model look like it gave up.
pub async fn prepare(handle: &PluginHandle) -> Result<PathBuf> {
    let warm = warm_target_dir()?;
    let template = template_dir()?;
    std::fs::create_dir_all(&warm)
        .with_context(|| format!("creating warm target dir {}", warm.display()))?;
    write_template(&template)?;
    let manifest = template.join("Cargo.toml");

    // `cargo test --no-run` warms the DEBUG rlibs + the test-harness link, which
    // is what the agent's own `cargo test`/`cargo run` need; `cargo build
    // --release` warms the RELEASE rlibs the scorer needs. Both profiles, or the
    // half that is cold is the half that times out.
    handle.check_cancelled()?;
    handle.status(format!("warming cargo debug profile in {}", warm.display()));
    cargo(&["test", "--no-run"], &warm, &manifest).await?;
    handle.check_cancelled()?;
    handle.status(format!(
        "warming cargo release profile in {}",
        warm.display()
    ));
    cargo(&["build", "--release"], &warm, &manifest).await?;
    Ok(warm)
}

#[cfg(test)]
#[path = "warm_tests.rs"]
mod tests;
