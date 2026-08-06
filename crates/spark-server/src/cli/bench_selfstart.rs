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

/// What a baseline says to serve.
#[derive(Debug)]
pub(super) struct Resolved {
    pub model: String,
    pub recipe_id: String,
}

/// Pick the (model, recipe) a gate run should serve.
///
/// Split out from the serving itself so the branching — which box class, which
/// model, and whether a recipe is bound at all — is testable without a GPU.
/// Every refusal names both what was asked for and what exists; an unresolvable
/// baseline must never read as "nothing to serve".
pub(super) fn resolve(
    baseline: &gate::GateBaseline,
    benchmark_id: &str,
    hardware: Option<&str>,
) -> Result<Resolved> {
    let hw_key = match hardware {
        Some(h) => h.to_string(),
        None => {
            let mut keys = baseline.hardware.keys();
            match (keys.next(), keys.next()) {
                (Some(only), None) => only.clone(),
                // Two box classes and no instruction is not a coin flip: TTFT
                // ceilings differ per box, so guessing would score the run
                // against another machine's numbers.
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
    Ok(Resolved { model, recipe_id })
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
    let Resolved { model, recipe_id } = resolve(&baseline, benchmark_id, hardware)?;

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
            "recipe {recipe_id:?} serves {:?} but {benchmark_id}'s baseline is defined on \
             {model:?}. Scoring one checkpoint against another's thresholds is not a lenient \
             comparison, it is a meaningless one.",
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
    let mut server =
        tokio::spawn(async move { crate::main_modules::serve::serve(serve_args, None).await });

    let target = TargetEndpoint::local(port, &model);
    await_serving(&target, &model, &mut server).await?;
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
    server: &mut tokio::task::JoinHandle<Result<()>>,
) -> Result<()> {
    let deadline = Instant::now() + BOOT_TIMEOUT;
    loop {
        if server.is_finished() {
            // Await the finished task for the REASON it stopped. Reporting only
            // "the server exited" would send the reader to a fifteen-minute
            // timeout hunt for an error the task is already holding — the whole
            // point of watching the handle is to surface it.
            return match server.await {
                Ok(Err(e)) => Err(e).with_context(|| {
                    format!("the server failed before it began serving {model:?}")
                }),
                Ok(Ok(())) => bail!(
                    "the server returned before it began serving {model:?} — it stopped without \
                     an error, which should not happen while the accept loop is running"
                ),
                Err(join) => Err(anyhow::Error::new(join))
                    .with_context(|| format!("the server task died serving {model:?}")),
            };
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

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_plugin::gate::{GateBaseline, HardwareBaseline, ModelBaseline};

    fn baseline(entries: &[(&str, &str, Option<&str>)]) -> GateBaseline {
        let mut hardware = BTreeMap::new();
        for (hw, model, recipe) in entries {
            let e = hardware
                .entry(hw.to_string())
                .or_insert_with(|| HardwareBaseline {
                    default: model.to_string(),
                    models: BTreeMap::new(),
                });
            e.models.insert(
                model.to_string(),
                ModelBaseline {
                    recipe: recipe.map(str::to_string),
                    note: String::new(),
                    metrics: BTreeMap::new(),
                },
            );
        }
        GateBaseline {
            schema: 2,
            hardware,
        }
    }

    #[test]
    fn a_single_box_class_is_inferred() {
        let b = baseline(&[("gb10", "unsloth/Qwen3.6-27B-NVFP4", Some("qwen3.6/x"))]);
        let r = resolve(&b, "bfcl-subset", None).expect("inferred");
        assert_eq!(r.model, "unsloth/Qwen3.6-27B-NVFP4");
        assert_eq!(r.recipe_id, "qwen3.6/x");
    }

    #[test]
    fn several_box_classes_refuse_to_guess() {
        // Guessing here would serve one box's config and score it against the
        // other's thresholds — TTFT ceilings are box-local.
        let b = baseline(&[("gb10", "m", Some("r")), ("mi300x", "m", Some("r2"))]);
        let err = resolve(&b, "ttft-warm-gate", None).expect_err("refused");
        let msg = format!("{err:#}");
        assert!(msg.contains("gb10"), "{msg}");
        assert!(msg.contains("mi300x"), "{msg}");
        assert!(msg.contains("--hardware"), "names the fix: {msg}");
    }

    #[test]
    fn an_explicit_box_class_picks_its_entry() {
        let b = baseline(&[
            ("gb10", "a", Some("recipe-a")),
            ("mi300x", "b", Some("recipe-b")),
        ]);
        let r = resolve(&b, "ttft-warm-gate", Some("mi300x")).expect("picked");
        assert_eq!(r.recipe_id, "recipe-b");
    }

    #[test]
    fn an_unknown_box_class_names_what_exists() {
        let b = baseline(&[("gb10", "m", Some("r"))]);
        let err = resolve(&b, "bfcl-subset", Some("h100")).expect_err("refused");
        let msg = format!("{err:#}");
        assert!(msg.contains("h100"), "{msg}");
        assert!(msg.contains("gb10"), "lists what it has: {msg}");
    }

    #[test]
    fn a_baseline_without_a_recipe_cannot_self_start() {
        // The honest failure: this gate has thresholds but nothing says how to
        // serve them, so it must refuse rather than invent a config.
        let b = baseline(&[("gb10", "m", None)]);
        let err = resolve(&b, "bfcl-subset", None).expect_err("refused");
        let msg = format!("{err:#}");
        assert!(msg.contains("no recipe is bound"), "{msg}");
        assert!(
            msg.contains("--url/--model"),
            "offers the alternative: {msg}"
        );
    }

    #[test]
    fn an_empty_baseline_is_an_error_not_a_default() {
        let b = baseline(&[]);
        assert!(resolve(&b, "bfcl-subset", None).is_err());
    }
}
