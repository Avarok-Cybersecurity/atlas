// SPDX-License-Identifier: AGPL-3.0-only

//! The `AVAROK_*` environment a gate's server is measured under.
//!
//! A serve LEVER is any `AVAROK_*` variable the server reads: `AVAROK_FP8_ROWWISE`
//! builds FP8 twin weights (+23.4 GB on a dense 27B), the MTP ladder variables
//! move decode, the co-dispatch controls move admission. None of them is a
//! recipe key, so none renders into the argv a leased server is fingerprinted
//! by — and until #1242 none was disclosed either. A node whose agent exported
//! them for one gate handed them to every gate on the box: `bfcl-subset` at
//! util 0.70 died at boot with "No memory left for KV cache", and a record that
//! survived could not be told from one measured on the pinned recipe.
//!
//! So the recipe DECLARES the levers it is measured under (its `env:` block;
//! the gate's `BENCH.toml` entry adds a pin the shared recipe must not carry),
//! and the harness applies exactly that set. A lever in the harness's own
//! environment that nothing declared is refused by name — not inherited, not
//! stripped. A declared lever the harness carries at another value is a
//! contradiction and refused too. The set is fingerprinted into the serve
//! identity (`serve_identity`), so a leased server started under one set is
//! never reused for a run that needs another, and it is disclosed on the gate
//! record (`gate::GateRecord::serve_env`).
//!
//! Lives in this crate because both ends need ONE spelling of "the lever set":
//! the server fingerprints its own environment for `GET /serve-config`, and
//! the harness computes what that fingerprint must be.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use sha2::{Digest, Sha256};

/// The prefix every lever carries.
pub const PREFIX: &str = "AVAROK_";
/// The pre-rebrand prefix. `avarok_core::env_compat` mirrors it onto [`PREFIX`]
/// at process start, so by the time anything here runs a legacy export is
/// already present under its current name — which is why a mirrored variable
/// counts exactly as a native one.
pub const LEGACY_PREFIX: &str = "ATLAS_";

/// `AVAROK_*` names that are NOT levers: they place, log or build, and the
/// server measures the same whatever they hold. Every other name under the
/// prefix is a lever. Each entry names the reader that makes it harmless.
///
/// Closed on purpose. A name that is not here is a lever, so a new `AVAROK_*`
/// read added to the server is refused-until-declared by default rather than
/// silently inherited. Adding a name is a claim that the server's OUTPUT does
/// not depend on it.
pub const HARNESS_VARS: &[&str] = &[
    // The artifact store root (`artifacts::home`): where records and the lease
    // live, not what is measured. The node agent exports it as `ATLAS_HOME`.
    "AVAROK_HOME",
    // Where the TUI tees its log (`tui::init`).
    "AVAROK_TUI_LOG_FILE",
    // Build-script inputs (`avarok-kernels/build.rs`, `spark-model/build.rs`,
    // `spark-runtime/build.rs`): they chose what was COMPILED, and the compiled
    // binary is already fingerprinted by its bytes.
    "AVAROK_SKIP_BUILD",
    "AVAROK_TARGET_HW",
    "AVAROK_TARGET_MODEL",
    "AVAROK_TARGET_QUANT",
    "AVAROK_HIPCC",
    // The BENCHMARK's side of the agentic harness (`benchmarks::agentic::warm`):
    // read by the process driving requests, never by the server.
    "AVAROK_HARNESS_PORT",
    "AVAROK_WARM_TEMPLATE_DIR",
];

/// Whether `name` is a serve lever.
pub fn is_lever(name: &str) -> bool {
    name.starts_with(PREFIX) && !HARNESS_VARS.contains(&name)
}

/// The levers in an environment, from any `(name, value)` iterator.
pub fn levers<I, K, V>(env: I) -> BTreeMap<String, String>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: AsRef<str>,
{
    env.into_iter()
        .filter(|(k, _)| is_lever(k.as_ref()))
        .map(|(k, v)| (k.as_ref().to_string(), v.as_ref().to_string()))
        .collect()
}

/// The levers of THIS process — what the server fingerprints, and what the
/// harness must reconcile against the recipe before it serves.
///
/// Lossy on a non-UTF-8 name or value rather than skipping it: a skipped
/// variable would be a lever the server reads and the fingerprint omits.
pub fn process_levers() -> BTreeMap<String, String> {
    levers(std::env::vars_os().map(|(k, v)| {
        (
            k.to_string_lossy().into_owned(),
            v.to_string_lossy().into_owned(),
        )
    }))
}

/// The digest of a lever set: `name=value` in key order, NUL-terminated, so
/// `A=1 B=2` and `A=1B=2` differ and the empty set has one fixed digest.
#[must_use]
pub fn fingerprint(levers: &BTreeMap<String, String>) -> String {
    let mut h = Sha256::new();
    for (k, v) in levers {
        h.update(k.as_bytes());
        h.update(b"=");
        h.update(v.as_bytes());
        h.update([0u8]);
    }
    format!("{:x}", h.finalize())
}

/// Validate one declaration — a recipe's `env:` block or a `BENCH.toml`
/// `[benchmarks.serve_env]` table — into the lever set it means.
///
/// `owner` names the declaring file in every refusal. A legacy `ATLAS_*`
/// spelling is accepted and renamed, because the server mirrors it anyway
/// and a declaration that meant one thing under `sparkrun` and another under
/// the gate would be worse than either; the one case refused is both
/// spellings of one lever at different values.
///
/// # Errors
/// A key that is not a lever (no prefix, or a [`HARNESS_VARS`] entry, which
/// is the box's to set) or an empty value (how a shell spells "unset").
pub fn declared(owner: &str, raw: &BTreeMap<String, String>) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for (name, value) in raw {
        let current = match name.strip_prefix(LEGACY_PREFIX) {
            Some(rest) => format!("{PREFIX}{rest}"),
            None => name.clone(),
        };
        if !current.starts_with(PREFIX) {
            bail!(
                "{owner} declares env {name}, which is not an AVAROK_* serve lever. The gate \
                 applies exactly the levers a recipe declares and nothing else, so a variable \
                 the server does not read cannot be declared here."
            );
        }
        if HARNESS_VARS.contains(&current.as_str()) {
            bail!(
                "{owner} declares env {name}, which places, logs or builds rather than \
                 measuring ({current} is a harness variable, not a serve lever) — it is the \
                 box's to set, never the recipe's."
            );
        }
        if value.trim().is_empty() {
            bail!(
                "{owner} declares env {name} with an empty value. An exported-but-empty variable \
                 is how a shell spells \"unset\": declare a value or drop the key."
            );
        }
        if let Some(other) = out.insert(current.clone(), value.clone())
            && other != *value
        {
            bail!(
                "{owner} declares {current} twice, under its current and legacy spellings, at \
                 different values ({other:?} and {value:?})"
            );
        }
    }
    Ok(out)
}

/// The recipe's declaration under the gate entry's pin; the pin wins on a
/// clash, exactly as a baseline `serve_overrides` pin does over the recipe.
#[must_use]
pub fn merge_declared(
    recipe: BTreeMap<String, String>,
    baseline: BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut out = recipe;
    out.extend(baseline);
    out
}

/// What the harness must do to serve `declared` from an environment that
/// already holds `present`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reconciled {
    /// The lever set the server runs under — the declaration, whole.
    pub env: BTreeMap<String, String>,
    /// The declared levers the harness does not already carry: what a child
    /// serve is given on top of the inherited environment, and what an
    /// in-process serve cannot be given at all.
    pub missing: BTreeMap<String, String>,
}

/// Reconcile the recipe's declaration with the harness's own levers
/// ([`process_levers`]).
///
/// # Errors
/// A present lever that `declared` does not name — refused, not inherited
/// and not stripped, because either would change what the gate measures
/// without a trace. A declared lever present at another value — refused,
/// because neither side may win silently.
pub fn reconcile(
    owner: &str,
    declared: &BTreeMap<String, String>,
    present: &BTreeMap<String, String>,
) -> Result<Reconciled> {
    let undeclared: Vec<String> = present
        .iter()
        .filter(|(k, _)| !declared.contains_key(*k))
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    if !undeclared.is_empty() {
        bail!(
            "the harness environment carries {} serve lever(s) that {owner} does not declare: \
             {}. A lever the recipe did not ask for changes what the gate measures \
             (AVAROK_FP8_ROWWISE=1 builds FP8 twin weights that a 0.70-util serve cannot \
             boot with — #1242), so it is refused rather than inherited. Unset it — on a node \
             agent, remove it from bench.yaml `env:`, which reaches EVERY gate on the box — or \
             declare it in the recipe's `env:` (or the gate's [benchmarks.serve_env]) if the \
             gate is measured under it. A legacy ATLAS_* export counts: this process mirrored \
             it onto the AVAROK_* name at start.",
            undeclared.len(),
            undeclared.join(", ")
        );
    }
    let mut missing = BTreeMap::new();
    let mut contradicted = Vec::new();
    for (k, want) in declared {
        match present.get(k) {
            Some(got) if got == want => {}
            Some(got) => {
                contradicted.push(format!("{k}: harness {got:?}, {owner} declares {want:?}"))
            }
            None => {
                missing.insert(k.clone(), want.clone());
            }
        }
    }
    if !contradicted.is_empty() {
        bail!(
            "the harness environment contradicts {owner} on {} serve lever(s): {}. Neither value \
             wins silently: unset the variable so the declaration applies, or change the \
             declaration.",
            contradicted.len(),
            contradicted.join("; ")
        );
    }
    Ok(Reconciled {
        env: declared.clone(),
        missing,
    })
}

#[cfg(test)]
#[path = "serve_env_tests.rs"]
mod tests;
