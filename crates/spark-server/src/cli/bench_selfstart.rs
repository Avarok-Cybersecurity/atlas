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
//!
//! ## Teardown on EVERY path
//!
//! This mode manages a model on a unified-memory box, where a leaked serve does
//! not waste a GPU, it wedges the machine. So the handle lives inside
//! [`SelfServed`], which tears it down in `Drop` — not only in the explicit
//! `shutdown()`. Dropping a `JoinHandle` DETACHES the task, so every `?` between
//! the spawn and the caller's teardown used to leak a loaded model: a boot
//! timeout inside [`serve_for`] and an artifact-store failure in `bench_run`
//! both did. The `Drop` makes those paths — and a panic — teardowns rather than
//! leaks.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use atlas_plugin::{TargetEndpoint, gate};

use super::bench_resolve::Resolved;

/// How long to wait for the endpoint to answer with the model we asked for.
/// A cold NVFP4 load on GB10 is minutes, not seconds.
const BOOT_TIMEOUT: Duration = Duration::from_secs(900);
const POLL: Duration = Duration::from_millis(500);

/// The fraction of this box's memory that must still be available before a
/// self-start will serve.
///
/// The recipe's `gpu_memory_utilization` is honoured VERBATIM — a gate that
/// quietly serves a different config than the one its thresholds were measured
/// under is the substitution this whole mode exists to prevent. So this does
/// NOT judge that number. `gpu_memory_utilization` is an allocator budget, not
/// resident host RAM, and values up to 0.90 have served fine for a long time on
/// a clean box.
///
/// What actually turns a working utilisation into an OOM freeze is CO-TENANCY:
/// the recorded incident was two serves plus a heavy Python client on one
/// unified 121 GB pool, which took SSH with it. Co-tenancy also corrupts the
/// measurement long before it freezes anything — 16.3 GB of co-tenants was
/// measured to cost 32 % at C=16 while costing vLLM ~0 %.
///
/// 0.85 therefore means "nothing else is holding more than ~15 % of this box".
/// A clean GB10 sits at ~0.94 available and passes; a single 16 GB container
/// drops it to ~0.81 and is refused — which is the case worth catching.
const MIN_FREE_FRACTION: f64 = 0.85;

static STARTED: AtomicBool = AtomicBool::new(false);

/// A server this process started, and the endpoint that reaches it.
pub struct SelfServed {
    pub target: TargetEndpoint,
    /// The recipe that produced it, for the record's provenance.
    pub recipe_id: String,
    /// The recipe keys the operator overrode, for the record's provenance.
    ///
    /// Empty means the recipe ran exactly as pinned. Non-empty means the
    /// numbers describe a config that exists nowhere in the repo, which the
    /// record must state — otherwise it reads as a measurement of the recipe.
    pub overrides: BTreeMap<String, String>,
    /// The served variant's baseline entry — its committed thresholds, note
    /// and label. Carried so the run can DERIVE anything its own verdict
    /// shares with the gate (`BenchmarkDescriptor::threshold_params`) from the
    /// variant that is actually serving, instead of from a schema default that
    /// is only right for one variant.
    pub baseline_entry: gate::ModelBaseline,
    /// `None` once teardown has taken it. An `Option` rather than a bare handle
    /// so that `Drop` — which cannot move out of `self` — can still tear down
    /// whatever `shutdown()` did not.
    server: Option<tokio::task::JoinHandle<Result<()>>>,
}

impl SelfServed {
    /// Stop the server and WAIT for it to be gone.
    ///
    /// ★ Call this AFTER the gate record is written, never before. The record's
    /// hardware fingerprint is fetched from the endpoint, and that fetch
    /// degrades to `Hardware::unknown()` on every failure path WITHOUT
    /// surfacing an error — so tearing down first yields a committed record
    /// that claims an unknown box and still exits successfully.
    pub async fn shutdown(mut self) {
        crate::tui::shutdown::request("benchmark gate run finished");
        if let Some(server) = self.server.take() {
            server.abort();
            let _ = server.await;
        }
    }
}

impl Drop for SelfServed {
    /// Last-resort teardown for the paths that never reach [`Self::shutdown`].
    ///
    /// A dropped `JoinHandle` DETACHES its task — the server would keep the
    /// model resident with nothing left holding a reference to it. On a unified
    /// 121 GB pool that is not a leaked GPU, it is a wedged box, and "the
    /// process exits soon anyway" is not a teardown: it is an assumption about
    /// the caller that `--no-fail-on-verdict` and any in-process caller break.
    ///
    /// Best-effort by construction: `Drop` cannot await, so this aborts and
    /// returns. The awaiting version is [`Self::shutdown`], which is what the
    /// normal paths call; by the time this runs on those, `server` is already
    /// `None` and there is nothing to do.
    ///
    /// It deliberately does NOT call `shutdown::request` the way
    /// [`Self::shutdown`] does. That latch is process-wide and has no reset, so
    /// tripping it from a destructor would make one gate's failed boot refuse
    /// every later `model_swap` and cancel every later `run_blocking` in the
    /// process — a scope far wider than the thing being cleaned up. The abort is
    /// enough on its own: it drops the serve future, which drops the host, which
    /// closes the scheduler's request channel, and THAT is what releases the
    /// weights. Refusing a second self-start is `claim_start_slot`'s job, not a
    /// side effect of this one.
    fn drop(&mut self) {
        let Some(server) = self.server.take() else {
            return;
        };
        eprintln!("gate: tearing down the self-started server (no explicit shutdown ran)");
        server.abort();
    }
}

/// Resolve the recipe for `benchmark_id` and serve it on a free port.
///
/// `hardware` picks the baseline entry; `None` uses the sole entry when the
/// baseline has exactly one, and otherwise refuses rather than guessing which
/// box's config to serve.
///
/// `overrides` are recipe keys the operator changed on the command line. They
/// are merged on top of any `[benchmarks.serve_overrides]` pin the resolved
/// baseline entry declares (the operator wins a clash) and the merged set is
/// returned in [`SelfServed::overrides`], so the gate record names the config
/// that actually ran rather than the recipe it started from — baseline pins
/// included, since `check_record` demands them on the record.
pub async fn serve_for(
    benchmark_id: &str,
    hardware: Option<&str>,
    checkpoint: Option<&str>,
    overrides: BTreeMap<String, String>,
) -> Result<SelfServed> {
    let root = super::bench_run::repo_root()?;
    // A shard serves what its GROUP serves — same recipe, same checkpoint —
    // and differs only in which rows it measures. Without this a member cannot
    // be run at all.
    let serve_id = gate::group::serve_baseline_id(benchmark_id);
    let baseline = gate::read_baseline(&root, serve_id)?;
    let Resolved {
        model,
        recipe_id,
        entry,
    } = super::bench_resolve::resolve(&baseline, serve_id, hardware, checkpoint)?;

    let store = atlas_plugin::ArtifactStore::discover()?;
    let index = crate::recipe::fetch::cached(store.root());
    let recipe = index
        .recipes
        .iter()
        .find(|r| r.id == recipe_id)
        .with_context(|| {
            format!(
                "recipe {recipe_id:?} is not in the local index ({} cached). The index is read \
                 from {}/atlas-recipes/index.json.{} Populate it with:\n    spark sync-recipes\n\
                 (this used to say \"open the TUI Library once\", which a CI runner, a \
                 container, or a machine reached over ssh cannot do.)",
                index.recipes.len(),
                store.root().display(),
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
    let requested = gate::merge_serve_overrides(entry.serve_overrides.clone(), overrides);
    let mut overrides = requested.clone();
    overrides.insert("port".to_string(), port.to_string());
    let serve_args = recipe.serve_args(&overrides).with_context(|| {
        format!("rendering serve args from recipe {recipe_id:?} (port override {port})")
    })?;
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

    check_box_is_free_enough(serve_args.gpu_memory_utilization, &recipe_id)?;

    // Claimed HERE — immediately before the spawn — and not on entry. Everything
    // above this line can fail without a server ever existing and therefore
    // without tripping the shutdown latch, so poisoning the process on a
    // mistyped `--hardware` would refuse a second attempt that would have
    // worked.
    claim_start_slot(&STARTED, crate::tui::shutdown::requested())?;

    eprintln!("gate: serving {model} from recipe {recipe_id} on port {port}");
    let server =
        tokio::spawn(async move { crate::main_modules::serve::serve(serve_args, None).await });

    // The handle is handed to `SelfServed` BEFORE the wait, so the `?` below is
    // a teardown and not a leak: a boot timeout used to drop the handle, which
    // detaches the task and leaves the model resident.
    let mut served = SelfServed {
        target: TargetEndpoint::local(port, &model),
        recipe_id,
        overrides: requested,
        baseline_entry: entry,
        server: Some(server),
    };
    await_serving(
        &served.target,
        &model,
        served.server.as_mut().expect("just constructed as Some"),
    )
    .await?;
    eprintln!("gate: endpoint is serving {model}");

    Ok(served)
}

/// Claim this process's ONE self-start slot.
///
/// Pure so both refusals are testable without a GPU: the decision is the whole
/// of the invariant, and the state it reads is process-global.
///
/// Two ways to be refused, and they are different failures:
///
/// * A gate already started a server here. Teardown tripped the one-way
///   shutdown latch, so a second serve would come up into a process that is
///   already draining and would never begin serving.
/// * A shutdown was requested for some other reason (Ctrl+C during the recipe
///   resolution, say). Same outcome, different cause — and worth saying so,
///   because "run one benchmark per invocation" would be wrong advice here.
fn claim_start_slot(started: &AtomicBool, shutdown_requested: bool) -> Result<()> {
    if shutdown_requested {
        bail!(
            "a shutdown has already been requested in this process, and that latch has no reset. \
             A server started now would return without ever serving, so the run is refused here \
             rather than after a fifteen-minute wait for a listener that is not coming."
        );
    }
    if started.swap(true, Ordering::SeqCst) {
        bail!(
            "a benchmark gate already started a server in this process; the shutdown latch is \
             one-way, so a second one cannot come up. Run one benchmark per invocation."
        );
    }
    Ok(())
}

/// Refuse to self-start onto a box that is already holding memory.
///
/// The recipe's utilisation is honoured verbatim — the question is only whether
/// this box can honour it right now. A value that serves fine on a clean box is
/// what OOM-freezes one that is already running a container or another serve,
/// and on unified memory that freeze takes SSH with it. So this reads the live
/// figure rather than judging the recipe's number.
///
/// Named remedies, because "not enough memory" without saying what is holding
/// it sends the reader looking in the wrong place.
fn check_box_is_free_enough(util: f64, recipe_id: &str) -> Result<()> {
    let Some((total_gib, avail_gib)) = host_memory_gib() else {
        // No /proc/meminfo (non-Linux, or a container without it). Say so and
        // continue: refusing on a box we cannot measure would block every
        // platform that is not this one.
        eprintln!("gate: cannot read host memory; skipping the free-memory preflight");
        return Ok(());
    };
    eprintln!(
        "{}",
        headroom_verdict(total_gib, avail_gib, util, recipe_id)?
    );
    Ok(())
}

/// The preflight decision for one reading: the line to print, or the refusal.
///
/// Pure, because the threshold is the whole of the check and a threshold that
/// is only exercised on a box that happens to be busy is a threshold nobody has
/// tested. `total_gib` is guaranteed positive by [`host_memory_gib`].
///
/// Named remedies in the refusal: "not enough memory" without saying what is
/// holding it sends the reader looking in the wrong place.
fn headroom_verdict(total_gib: f64, avail_gib: f64, util: f64, recipe_id: &str) -> Result<String> {
    let free_fraction = avail_gib / total_gib;
    if free_fraction < MIN_FREE_FRACTION {
        bail!(
            "this box is not free enough to serve recipe {recipe_id:?}: only {avail_gib:.0} GiB \
             of {total_gib:.0} GiB is available ({:.0} %, below the {:.0} % a self-start \
             requires). Something else is holding memory — check `sudo docker ps` and \
             `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`, free it, and \
             re-run. This is not a judgement on the recipe's --gpu-memory-utilization {util:.2}, \
             which is served exactly as written: co-tenancy is what turns a working \
             utilisation into an OOM freeze on unified memory, and it corrupts the measurement \
             well before that.",
            free_fraction * 100.0,
            MIN_FREE_FRACTION * 100.0,
        );
    }
    Ok(format!(
        "gate: {avail_gib:.0} GiB of {total_gib:.0} GiB free ({:.0} %); serving at {util:.2} as the recipe states",
        free_fraction * 100.0
    ))
}

/// `(MemTotal, MemAvailable)` in GiB from `/proc/meminfo`.
///
/// `MemAvailable`, not `MemFree`: page cache is reclaimable, and `MemFree`
/// reads near-zero on a healthy box that has been running a while, which would
/// make this refuse everything.
///
/// A non-positive `MemTotal` reads as UNREADABLE rather than as a reading: the
/// fraction would be NaN or infinite, and `NaN < MIN_FREE_FRACTION` is false —
/// i.e. an unparsable `/proc/meminfo` would silently PASS the preflight it
/// exists to fail.
fn host_memory_gib() -> Option<(f64, f64)> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let field = |name: &str| -> Option<f64> {
        text.lines()
            .find(|l| l.starts_with(name))?
            .split_whitespace()
            .nth(1)?
            .parse::<f64>()
            .ok()
            .map(|kb| kb / 1024.0 / 1024.0)
    };
    let (total, avail) = (field("MemTotal:")?, field("MemAvailable:")?);
    (total > 0.0 && avail.is_finite()).then_some((total, avail))
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
        // What the endpoint just said, so a timeout can distinguish "nothing
        // was listening" from "something answered, with a DIFFERENT
        // checkpoint". The second is the case this function exists to refuse,
        // and reporting it as a bare timeout would send the reader hunting a
        // slow load instead.
        let last = match atlas_plugin::http::list_models(target, Duration::from_secs(5)).await {
            Ok(models) if models.iter().any(|m| m == model) => return Ok(()),
            Ok(models) => format!("the endpoint is serving {models:?}"),
            Err(e) => format!("{e:#}"),
        };
        if Instant::now() >= deadline {
            bail!(
                "{model:?} did not come up within {}s — {last}",
                BOOT_TIMEOUT.as_secs()
            );
        }
        tokio::time::sleep(POLL).await;
    }
}

#[cfg(test)]
#[path = "bench_selfstart_tests.rs"]
mod tests;
