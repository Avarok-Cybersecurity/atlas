// SPDX-License-Identifier: AGPL-3.0-only

//! What a gate run would serve: the recipe its baseline names, the checkpoint
//! that recipe must agree on, and the override set the record will state.
//!
//! Split out of `bench_selfstart` so the two ways of getting a server — start
//! one in this process, or reuse the one a campaign left running
//! (`bench_lease`) — plan from ONE resolution and cannot disagree about what
//! "the same server" means.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use avarok_plugin::{gate, serve_env};

use super::bench_resolve::Resolved;

/// A resolved serve: everything but the port.
pub struct ServePlan {
    pub model: String,
    pub recipe_id: String,
    pub recipe: crate::recipe::Recipe,
    /// The served variant's baseline entry — see `SelfServed::baseline_entry`.
    pub entry: gate::ModelBaseline,
    /// The merged `[benchmarks.serve_overrides]` pin + `--serve-override` set,
    /// after `--hermetic` expansion: what the record states.
    pub requested: BTreeMap<String, String>,
    /// The `AVAROK_*` serve levers the server runs under: the recipe's `env:`
    /// block under the entry's `[benchmarks.serve_env]` pin, both validated
    /// by `serve_env::declared`. What a leased server's `env_sha256` must
    /// fingerprint to, and what the record discloses (#1242).
    pub serve_env: BTreeMap<String, String>,
    /// The box class the run is for, and its declared limits
    /// (`kernels/<hw>/HARDWARE.toml` `[benchmarks.limits]`): the memory floor
    /// a self-start applies and how long a server may take to come up.
    pub hardware: String,
    pub limits: avarok_plugin::hardware::limits::Limits,
}

impl ServePlan {
    /// The argv `spark serve` is started with on `port` — the recipe's own
    /// rendering, which is also what a reused server must fingerprint to.
    /// `["spark", "serve", <model>, …]`; callers skip the program name.
    pub fn argv(&self, port: u16) -> Result<Vec<String>> {
        let mut overrides = self.requested.clone();
        overrides.insert("port".to_string(), port.to_string());
        self.recipe.argv(&overrides).with_context(|| {
            format!(
                "rendering serve args from recipe {:?} (port override {port})",
                self.recipe_id
            )
        })
    }

    /// The parsed, validated form of [`Self::argv`].
    pub fn serve_args(&self, port: u16) -> Result<crate::cli::ServeArgs> {
        let mut overrides = self.requested.clone();
        overrides.insert("port".to_string(), port.to_string());
        self.recipe.serve_args(&overrides).with_context(|| {
            format!(
                "rendering serve args from recipe {:?} (port override {port})",
                self.recipe_id
            )
        })
    }

    /// What the gate record DISCLOSES about this serve — the knobs resolved
    /// the way the server resolves them (`ServeArgs::mtp_gate_force`, so
    /// `--hermetic` counts), keyed as `gate::record_serve` names them. Read
    /// off the same rendering `argv`/`serve_args` produce, so a leased server
    /// — which `bench_lease` fingerprints against that rendering — discloses
    /// the same values as one started here.
    pub fn disclosed(&self, port: u16) -> Result<BTreeMap<String, String>> {
        Ok(disclosed_from(&self.serve_args(port)?))
    }

    /// Reconcile the declared lever set with the levers THIS process carries
    /// — refusing, by name, any the recipe did not declare and any it declares
    /// at another value (`serve_env::reconcile`). Called before a server is
    /// probed or started, so a contaminated harness gets no server at all.
    pub fn reconcile_env(&self) -> Result<serve_env::Reconciled> {
        serve_env::reconcile(
            &format!("recipe {}", self.recipe_id),
            &self.serve_env,
            &serve_env::process_levers(),
        )
    }
}

/// The disclosure for one rendered, validated serve — the single expression
/// [`ServePlan::disclosed`] evaluates, kept free so a test can feed it a
/// recipe without standing up a whole plan.
pub(crate) fn disclosed_from(args: &crate::cli::ServeArgs) -> BTreeMap<String, String> {
    gate::record_serve::disclosure(
        args.mtp_gate_force(),
        args.speculative,
        args.prefill_codispatch,
    )
}

/// Resolve what `benchmark_id`'s gate run serves.
pub fn plan_serve(
    benchmark_id: &str,
    hardware: Option<&str>,
    checkpoint: Option<&str>,
    overrides: BTreeMap<String, String>,
) -> Result<ServePlan> {
    let root = super::bench_run::repo_root()?;
    // A shard serves what its GROUP serves — same recipe, same checkpoint —
    // and differs only in which rows it measures.
    let serve_id = gate::group::serve_baseline_id(benchmark_id);
    let baseline = gate::read_baseline(&root, serve_id)?;
    let Resolved {
        model,
        recipe_id,
        entry,
        hardware,
    } = super::bench_resolve::resolve(&baseline, serve_id, hardware, checkpoint)?;
    let Some(limits) = avarok_plugin::hardware::limits::limits(&root, &hardware)? else {
        bail!(
            "kernels/{hardware}/HARDWARE.toml declares no [benchmarks.limits]: a gate run on this \
             class has no memory floor to check the box against and no boot timeout for its \
             server. Measure them and declare the tables (see kernels/gb10/HARDWARE.toml)."
        );
    };

    let store = avarok_plugin::ArtifactStore::discover()?;
    let index = crate::recipe::fetch::cached(store.root());
    let recipe = index
        .recipes
        .iter()
        .find(|r| r.id == recipe_id)
        .with_context(|| {
            format!(
                "recipe {recipe_id:?} is not in the local index ({} cached). The index is read \
                 from {}.{} Populate it with:\n    spark sync-recipes\n\
                 (this used to say \"open the TUI Library once\", which a CI runner, a \
                 container, or a machine reached over ssh cannot do.)",
                index.recipes.len(),
                // Named through `cache_dir` so the path printed is the path
                // read, including on a box that predates the rename.
                crate::recipe::fetch::cache_dir(store.root())
                    .join("index.json")
                    .display(),
                // Why the index is empty, when the index layer knows. Without
                // it an index that exists and cannot be READ -- a `$HOME`
                // owned by another uid is the measured case -- reads as one
                // that was never written, and `sync-recipes` is the wrong
                // remedy: it fetches from GitHub and then fails on the same
                // unwritable path, having spent the round trip to say so.
                index
                    .offline
                    .as_deref()
                    .map(|why| format!(" That index could not be used: {why}."))
                    .unwrap_or_default()
            )
        })?
        .clone();

    // The baseline and the recipe must agree on the checkpoint, or the run
    // would be scored against thresholds measured on a different one — the
    // exact substitution `check_record` refuses after the fact. Catch it before
    // spending a model load on it.
    if recipe.model != model {
        bail!(
            "recipe {recipe_id:?} serves {:?} but {benchmark_id}'s baseline is defined on \
             {model:?}. Scoring one checkpoint against another's thresholds is not a lenient \
             comparison, it is a meaningless one.",
            recipe.model
        );
    }

    // The levers the serve runs under: the recipe's own `env:` block with the
    // entry's `[benchmarks.serve_env]` pin on top (the pin wins, as a serve
    // override does). Validated NOW — a legacy spelling or a harness variable
    // in either is refused before a model load, not by the serve after one.
    let serve_env = serve_env::merge_declared(
        serve_env::declared(&format!("recipe {recipe_id}"), &recipe.env)?,
        serve_env::declared(
            &format!("{benchmark_id}'s baseline entry for {model} ([benchmarks.serve_env])"),
            &entry.serve_env,
        )?,
    );
    if !serve_env.is_empty() {
        let shown = serve_env
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("gate: recipe {recipe_id} is measured under serve env: {shown}");
    }

    // `--hermetic` expands into the keys it closes BEFORE the recipe renders,
    // so a recipe default that turns one of them on does not produce a command
    // line contradicting itself. See `cli::hermetic::CLOSED_KEYS`.
    let requested = crate::cli::hermetic::expand(gate::merge_serve_overrides(
        entry.serve_overrides.clone(),
        overrides,
    ));
    if !requested.is_empty() {
        let shown = requested
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ");
        tracing::warn!(
            "serving recipe {recipe_id} with OVERRIDES: {shown} — this run does not measure the \
             recipe as pinned; the gate record will say so"
        );
    }
    Ok(ServePlan {
        model,
        recipe_id,
        recipe,
        entry,
        requested,
        serve_env,
        hardware,
        limits,
    })
}

#[cfg(test)]
#[path = "bench_serve_plan_tests.rs"]
mod tests;
