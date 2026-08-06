// SPDX-License-Identifier: AGPL-3.0-only

//! Serve a benchmark's own recipe for the duration of a gate run.
//!
//! A gate record is only worth what its serve config is worth. Driving a
//! hand-started endpoint means one mistyped flag silently moves every number in
//! the record, and nothing downstream can tell — which is the failure this gate
//! exists to catch, reproduced one level up. So `--pull-request-gate` does not
//! trust the caller: it reads the recipe the benchmark's baseline names, serves
//! that, and measures what it started.
//!
//! Without the flag nothing here runs and `--url`/`--model` drive an existing
//! server exactly as before.
//!
//! ## Start-once-per-process
//!
//! Teardown goes through `shutdown::request`, and that latch is ONE-WAY — there
//! is no reset, so once it is tripped `run_blocking`'s cancel check and
//! `model_swap`'s load guard both stay tripped for the life of the process. One
//! invocation therefore serves exactly one model and then exits. A second
//! self-start in the same process is refused with a real message rather than
//! left to hang on a listener that will never come up.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use atlas_plugin::{TargetEndpoint, gate};

/// How long to wait for the endpoint to answer with the model we asked for.
/// A cold NVFP4 load on GB10 is minutes, not seconds.
const BOOT_TIMEOUT: Duration = Duration::from_secs(900);
const POLL: Duration = Duration::from_millis(500);

static STARTED: AtomicBool = AtomicBool::new(false);

/// A server this process started, and the endpoint that reaches it.
pub struct SelfServed {
    pub target: TargetEndpoint,
    /// The recipe that produced it, for the record's provenance.
    pub recipe_id: String,
    server: tokio::task::JoinHandle<Result<()>>,
}

impl SelfServed {
    /// Stop the server.
    ///
    /// ★ Call this AFTER the gate record is written, never before. The record's
    /// hardware fingerprint is fetched from the endpoint, and that fetch
    /// degrades to `Hardware::unknown()` on every failure path WITHOUT
    /// surfacing an error — so tearing down first yields a committed record
    /// that claims an unknown box and still exits successfully.
    pub async fn shutdown(self) {
        crate::tui::shutdown::request("benchmark gate run finished");
        self.server.abort();
        let _ = self.server.await;
    }
}

/// Resolve the recipe for `benchmark_id` and serve it on a free port.
///
/// `hardware` picks the baseline entry; `None` uses the sole entry when the
/// baseline has exactly one, and otherwise refuses rather than guessing which
/// box's config to serve.
pub async fn serve_for(benchmark_id: &str, hardware: Option<&str>) -> Result<SelfServed> {
    if STARTED.swap(true, Ordering::SeqCst) {
        bail!(
            "a benchmark gate already started a server in this process; the shutdown latch is \
             one-way, so a second one cannot come up. Run one benchmark per invocation."
        );
    }

    let root = super::bench_run::repo_root()?;
    let baseline = gate::read_baseline(&root, benchmark_id)?;

    let hw_key = match hardware {
        Some(h) => h.to_string(),
        None => {
            let mut keys = baseline.hardware.keys();
            match (keys.next(), keys.next()) {
                (Some(only), None) => only.clone(),
                (Some(_), Some(_)) => bail!(
                    "{benchmark_id} has baselines for several box classes ([{}]); pass \
                     --hardware to say which one this run is for rather than guessing",
                    baseline
                        .hardware
                        .keys()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                (None, _) => bail!("{benchmark_id} has no hardware entries in its baseline"),
            }
        }
    };

    let (model, entry) = baseline.resolve(&hw_key, None)?;
    let recipe_id = entry.recipe.clone().ok_or_else(|| {
        anyhow::anyhow!(
            "no recipe is bound to {model:?} on {hw_key:?} for {benchmark_id}. Self-start needs \
             one; either add `recipe` to the baseline entry or drive an existing server with \
             --url/--model and no --pull-request-gate."
        )
    })?;

    let store = atlas_plugin::ArtifactStore::discover()?;
    let index = crate::recipe::fetch::cached(store.root());
    let recipe = index
        .recipes
        .iter()
        .find(|r| r.id == recipe_id)
        .with_context(|| {
            format!(
                "recipe {recipe_id:?} is not in the local index ({} cached). The index is read \
                 from {}/atlas-recipes/index.json; open the TUI Library once to populate it.",
                index.recipes.len(),
                store.root().display()
            )
        })?;

    // The baseline and the recipe must agree on the checkpoint, or the run
    // would be scored against thresholds measured on a different one — the
    // exact substitution `check_record` refuses after the fact. Catch it before
    // spending a model load on it.
    if recipe.model != model {
        bail!(
            "recipe {recipe_id:?} serves {:?} but {benchmark_id}'s baseline for {hw_key:?} is \
             defined on {model:?}. Scoring one checkpoint against another's thresholds is not a \
             lenient comparison, it is a meaningless one.",
            recipe.model
        );
    }

    let port = atlas_plugin::benchmarks::agentic::score::free_port()?;
    let mut overrides = BTreeMap::new();
    overrides.insert("port".to_string(), port.to_string());
    let serve_args = recipe.serve_args(&overrides).with_context(|| {
        format!("rendering serve args from recipe {recipe_id:?} (port override {port})")
    })?;

    eprintln!("gate: serving {model} from recipe {recipe_id} on port {port}");
    let server =
        tokio::spawn(async move { crate::main_modules::serve::serve(serve_args, None).await });

    let target = TargetEndpoint::local(port, &model);
    await_serving(&target, &model, &server).await?;
    eprintln!("gate: endpoint is serving {model}");

    Ok(SelfServed {
        target,
        recipe_id,
        server,
    })
}

/// Block until `/v1/models` names `model`.
///
/// Naming the model is the point: a load that fails and leaves some earlier
/// checkpoint answering would otherwise be measured and recorded as this one.
/// The server task is polled too, so a serve that dies during startup reports
/// its own error instead of timing out fifteen minutes later.
async fn await_serving(
    target: &TargetEndpoint,
    model: &str,
    server: &tokio::task::JoinHandle<Result<()>>,
) -> Result<()> {
    let deadline = Instant::now() + BOOT_TIMEOUT;
    loop {
        if server.is_finished() {
            bail!("the server exited before it began serving {model:?}");
        }
        if let Ok(models) = atlas_plugin::http::list_models(target, Duration::from_secs(5)).await
            && models.iter().any(|m| m == model)
        {
            return Ok(());
        }
        if Instant::now() >= deadline {
            bail!(
                "{model:?} did not come up within {}s",
                BOOT_TIMEOUT.as_secs()
            );
        }
        tokio::time::sleep(POLL).await;
    }
}
