// SPDX-License-Identifier: AGPL-3.0-only

//! The Serve Matrix's [`ServeHost`], backed by this process's model host.
//!
//! `atlas-plugin` must stay GPU-free and server-free, so the benchmark declares
//! the seam and this supplies it. Two things it owes the benchmark, and both
//! are the fix for a defect in `tests/run_all_models.py`:
//!
//! * **The roster is derived.** `library::scan` reports what is in the HF cache
//!   and `atlas_kernels::ptx_for_config` decides whether this build compiled
//!   kernels for it. Nothing here lists a model. The Python's `ROUNDS` is a
//!   hand-maintained Qwen3.5-era list of twelve checkpoints, not one of which
//!   is in this box's cache — a second roster that went stale silently.
//!
//! * **Readiness is the endpoint answering.** The Python watched the container
//!   log for `Listening on`, a substring that also matches
//!   `Listening on 127.0.0.1:8888` — a server bound to loopback inside a
//!   bridged namespace, which every probe then failed to reach with
//!   "connection reset by peer" that read exactly like a model regression.
//!   Here the round is served by an in-process swap onto the port this server
//!   is ALREADY bound to, so there is no second bind to get wrong, and `serve`
//!   still does not return until `/v1/models` names the new checkpoint.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use atlas_plugin::TargetEndpoint;
use atlas_plugin::benchmarks::serve_matrix::host::{
    Absence, ServeCandidate, ServeHost, ServeOptions,
};
use futures::future::BoxFuture;

use crate::main_modules::model_host::ModelHost;

/// How long a checkpoint gets to load before the round is REPORTED as a boot
/// failure. Weight loading on the largest cached checkpoints is minutes; this
/// is the same 10-minute bound the Python orchestrator used.
///
/// It bounds the REPORT, not the load. `model_swap::swap` is blocking and
/// un-cancellable — dropping a `spawn_blocking` handle does not stop the task
/// — so a load that overruns keeps going, and the next round's swap queues
/// behind it on `ModelHost::swap_guard`. The failure that follows says so.
const BOOT_TIMEOUT: Duration = Duration::from_secs(600);

pub struct TuiServeHost {
    host: Arc<ModelHost>,
    /// The argv that was serving when the dashboard started, so the matrix can
    /// put the box back. `None` when nothing was loaded — restoring to nothing
    /// is a real state and is not the same as having nothing to restore.
    original: parking_lot::Mutex<Option<crate::cli::ServeArgs>>,
    cache_dir: Option<std::path::PathBuf>,
}

impl TuiServeHost {
    pub fn new(host: Arc<ModelHost>, cache_dir: Option<std::path::PathBuf>) -> Self {
        Self {
            host,
            original: parking_lot::Mutex::new(None),
            cache_dir,
        }
    }

    /// Where this server is listening. The matrix never opens a port of its
    /// own — it drives the one the dashboard is already attached to.
    fn endpoint(&self, model: &str) -> Result<TargetEndpoint> {
        let (_, port) = self
            .host
            .bound()
            .context("this server has not finished binding its port yet")?;
        Ok(TargetEndpoint::local(port, model))
    }

    /// Build the argv for one round. Everything not stated here comes from the
    /// checkpoint's own `MODEL.toml` defaults — which is what the matrix is
    /// supposed to be measuring, so overriding more would measure this file.
    fn argv_for(&self, model: &str, opts: ServeOptions) -> Result<crate::cli::ServeArgs> {
        use clap::Parser as _;
        let (_, port) = self
            .host
            .bound()
            .context("this server has not finished binding its port yet")?;
        let mut argv = vec![
            "spark".to_string(),
            "serve".to_string(),
            model.to_string(),
            "--port".to_string(),
            port.to_string(),
            "--max-seq-len".to_string(),
            opts.max_seq_len.to_string(),
        ];
        if opts.speculative {
            argv.push("--speculative".to_string());
        }
        if let Some(dir) = &self.cache_dir {
            argv.push("--cache-dir".to_string());
            argv.push(dir.display().to_string());
        }
        let cli = crate::cli::Cli::try_parse_from(&argv).with_context(|| {
            format!("serve matrix produced an invalid command line for {model}")
        })?;
        let crate::cli::Command::Serve(args) = cli.command else {
            bail!("serve matrix did not produce a serve command");
        };
        crate::cli::validate_serve_args(&args).map_err(|e| anyhow!("{model}: {e}"))?;
        Ok(args)
    }

    /// Run a swap off the async runtime. `model_swap::swap` blocks for minutes
    /// — holding a runtime worker for that would stall every other task on it,
    /// the dashboard's redraw included.
    async fn swap_blocking(&self, args: crate::cli::ServeArgs) -> Result<()> {
        let host = self.host.clone();
        let handle = tokio::task::spawn_blocking(move || {
            crate::main_modules::model_swap::swap(&host, args).map(|_| ())
        });
        match tokio::time::timeout(BOOT_TIMEOUT, handle).await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => bail!("the loader thread failed: {e}"),
            Err(_) => bail!(
                "did not load within {}s — the load is still running and the next round will \
                 queue behind it",
                BOOT_TIMEOUT.as_secs()
            ),
        }
    }

    /// Wait until the endpoint reports it is serving `model`.
    ///
    /// This is the readiness check, and it deliberately asks the SERVER what it
    /// is serving rather than trusting the swap's return: a swap that failed
    /// and auto-restored the previous checkpoint would otherwise be scored as
    /// this round's model.
    async fn await_serving(&self, target: &TargetEndpoint) -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut last = String::new();
        while Instant::now() < deadline {
            match atlas_plugin::http::list_models(target, Duration::from_secs(10)).await {
                Ok(served) if served.iter().any(|m| m == &target.model) => return Ok(()),
                Ok(served) => last = format!("serving {:?}", served),
                Err(e) => last = format!("{e:#}"),
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        bail!(
            "{} never reported {} as loaded ({last})",
            target.base_url,
            target.model
        )
    }
}

/// Why a scanned checkpoint cannot take part, or `None` when it can.
///
/// The only place this decision is made — `roster` calls it and so does its
/// test, so the two cannot disagree about what a skip is.
///
/// Order matters: weights that are not all on disk cannot be inspected for
/// architecture support, so "no weights" is the specific answer and "no
/// kernels" would be a guess.
pub fn classify(e: &crate::tui::data::library::LibraryEntry) -> Option<Absence> {
    if !e.has_weights {
        Some(Absence::NoWeights)
    } else if e.model_type.is_empty() {
        // `library::scan` fills `model_type` only when it could read AND parse
        // `config.json`; an empty one also leaves `optimized` false. Reporting
        // that as "no kernels for this architecture" would be a claim about an
        // architecture nothing managed to read.
        Some(Absence::NoConfig)
    } else if !e.optimized {
        Some(Absence::NoKernels)
    } else {
        None
    }
}

impl ServeHost for TuiServeHost {
    fn roster(&self) -> Result<Vec<ServeCandidate>> {
        Ok(crate::tui::data::library::scan(self.cache_dir.as_deref())
            .into_iter()
            .map(|e| match classify(&e) {
                Some(why) => ServeCandidate::absent(e.id, e.quant, why),
                None => ServeCandidate::ready(e.id, e.quant),
            })
            .collect())
    }

    fn serve(&self, model: &str, opts: ServeOptions) -> BoxFuture<'_, Result<TargetEndpoint>> {
        let model = model.to_string();
        Box::pin(async move {
            // Captured on the FIRST boot, not at construction: the dashboard
            // may have swapped models before the matrix started, and the state
            // to put back is the one the operator left it in.
            {
                let mut original = self.original.lock();
                if original.is_none() {
                    *original = self.host.args();
                }
            }
            let args = self.argv_for(&model, opts)?;
            let target = self.endpoint(&model)?;
            self.swap_blocking(args).await?;
            self.await_serving(&target).await?;
            Ok(target)
        })
    }

    fn restore(&self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            let Some(args) = self.original.lock().clone() else {
                // Nothing was serving when the matrix started. Leaving the last
                // round loaded is closer to where the box was found than
                // tearing it down to nothing would be, and either way there is
                // no argv to restore.
                return Ok(());
            };
            // No short-circuit on the model id. `model_swap::swap` compares
            // the WHOLE argv because switching between two recipes for the same
            // checkpoint is a real swap — and the matrix serves every round at
            // its own --max-seq-len (and possibly --speculative), so a box left
            // on the same model id can still be on a different configuration
            // than the operator left it in.
            self.swap_blocking(args)
                .await
                .context("could not put the previous model back")
        })
    }
}

#[cfg(test)]
#[path = "bench_host_tests.rs"]
mod tests;
