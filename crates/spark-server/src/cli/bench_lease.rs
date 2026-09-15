// SPDX-License-Identifier: AGPL-3.0-only

//! A server that outlives one gate run, so the next run on this box can
//! measure against it instead of loading the checkpoint again.
//!
//! `spark benchmark run --pull-request-gate --serve-reuse` does not serve in
//! its own process. It looks for the LEASED server — one `spark serve`
//! started by an earlier run of this mode, described in
//! `<ATLAS_HOME>/serve-lease.json` — and takes it if, and only if, it is the
//! server this run would have started itself: the same binary bytes, the
//! same recipe rendering with the same overrides (`GET /serve-config`, two
//! digests), and `/v1/models` naming the checkpoint. Anything else is
//! stopped and replaced. When the run ends the server is LEFT RUNNING for
//! the next one; `spark benchmark serve-release` (or the campaign driver at
//! its end) stops it.
//!
//! Why the digests and not trust: a gate record is only worth what its serve
//! config is worth (`bench_selfstart`). A server the caller merely CLAIMS is
//! right is the one-mistyped-flag failure this mode exists to prevent, one
//! level up. So the server states what it is, the run states what it needs,
//! and the record is written only when the two are the same thing.
//!
//! Why a lease and not a discovery: a stray `spark serve` on a shared box is
//! never taken, whoever started it and whatever it serves — only the one this
//! file names, which this mode started. `owner_pid` names the driver that
//! asked for reuse; a lease whose owner is gone is a campaign that died, and
//! its server is stopped rather than kept warm for nobody.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use atlas_plugin::serve_identity::{ServeIdentity, argv_fingerprint, file_sha256};
use atlas_plugin::{ArtifactStore, TargetEndpoint};

use super::bench_selfstart::SelfServed;
use super::bench_serve_plan::ServePlan;

const POLL: Duration = Duration::from_millis(500);
/// SIGTERM, then this long, then SIGKILL.
const STOP_GRACE: Duration = Duration::from_secs(60);

/// The leased server, as written beside the runs it serves.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Lease {
    pub pid: u32,
    pub port: u16,
    pub model: String,
    pub recipe_id: String,
    pub argv_sha256: String,
    pub binary_sha256: String,
    /// The process that asked for the lease (a campaign driver), or the run
    /// itself when nobody did.
    pub owner_pid: u32,
    pub started_at: u64,
}

pub fn lease_path(store: &ArtifactStore) -> PathBuf {
    store.root().join("serve-lease.json")
}

pub fn log_path(store: &ArtifactStore) -> PathBuf {
    store.root().join("serve-lease.log")
}

/// The lease on file, if any. A malformed file is an error, not "no lease":
/// a server it named may be running.
pub fn read(store: &ArtifactStore) -> Result<Option<Lease>> {
    let path = lease_path(store);
    match std::fs::read_to_string(&path) {
        Ok(text) => Ok(Some(
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?,
        )),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

fn write(store: &ArtifactStore, lease: &Lease) -> Result<()> {
    let path = lease_path(store);
    std::fs::write(&path, serde_json::to_string_pretty(lease)?)
        .with_context(|| format!("writing {}", path.display()))
}

/// Read from procfs; where there is none (this binary builds on Windows and
/// macOS) no pid is ever alive, so a lease is never taken — and never
/// signalled — there: the feature is inert rather than wrong.
fn pid_alive(pid: u32) -> bool {
    cfg!(target_os = "linux") && Path::new(&format!("/proc/{pid}")).exists()
}

/// What this process would want a reused server to be: the plan's own
/// rendering on the leased port, from this binary.
fn expected(plan: &ServePlan, port: u16) -> Result<(String, String)> {
    let argv = plan.argv(port)?;
    let mine = std::env::current_exe().context("current_exe")?;
    Ok((argv_fingerprint(&argv[1..]), file_sha256(&mine)?))
}

/// Why a leased server is not the one this run needs, or `None` when it is.
///
/// Pure: the decision the reuse hinges on, testable without a server.
pub fn mismatch(
    lease: &Lease,
    reported: &ServeIdentity,
    expected: &(String, String),
    model: &str,
) -> Option<String> {
    if reported.pid != lease.pid {
        return Some(format!(
            "pid {} answered on port {}, the lease names pid {}",
            reported.pid, lease.port, lease.pid
        ));
    }
    if reported.binary_sha256 != expected.1 {
        return Some("it was built from another binary".into());
    }
    if reported.argv_sha256 != expected.0 {
        return Some(format!(
            "it serves {} under another rendering (recipe {}, overrides or hermetic set differ)",
            lease.model, lease.recipe_id
        ));
    }
    if lease.model != model {
        return Some(format!("it serves {}, this run needs {model}", lease.model));
    }
    None
}

/// Take the leased server if it is the one `plan` would start, else replace
/// it. Either way the returned server is left running when dropped.
pub async fn acquire(plan: ServePlan, owner_pid: Option<u32>) -> Result<SelfServed> {
    let store = ArtifactStore::discover()?;
    if let Some(lease) = read(&store)? {
        if pid_alive(lease.pid) {
            let target = TargetEndpoint::local(lease.port, &plan.model);
            let verdict = match probe(&target, &lease, &plan).await {
                Ok(None) => None,
                Ok(Some(why)) => Some(why),
                Err(e) => Some(format!("{e:#}")),
            };
            match verdict {
                None => {
                    eprintln!(
                        "gate: reusing the leased server (pid {}, port {}, recipe {}) — same binary, \
                         same rendering",
                        lease.pid, lease.port, lease.recipe_id
                    );
                    return Ok(SelfServed::external(
                        target,
                        plan.recipe_id,
                        plan.requested,
                        plan.entry,
                    ));
                }
                Some(why) => {
                    eprintln!(
                        "gate: the leased server (pid {}, port {}) is not this run's: {why}; replacing it",
                        lease.pid, lease.port
                    );
                    stop(&lease);
                }
            }
        } else {
            eprintln!(
                "gate: the lease names pid {}, which is gone; starting afresh",
                lease.pid
            );
        }
        let _ = std::fs::remove_file(lease_path(&store));
    }
    start(&store, plan, owner_pid.unwrap_or_else(std::process::id)).await
}

/// Ask the leased server what it is and compare.
async fn probe(target: &TargetEndpoint, lease: &Lease, plan: &ServePlan) -> Result<Option<String>> {
    let doc =
        atlas_plugin::http::get_json(target, "/serve-config", Duration::from_secs(10)).await?;
    let reported: ServeIdentity = serde_json::from_value(doc).context("parsing /serve-config")?;
    let want = expected(plan, lease.port)?;
    if let Some(why) = mismatch(lease, &reported, &want, &plan.model) {
        return Ok(Some(why));
    }
    let models = atlas_plugin::http::list_models(target, Duration::from_secs(10)).await?;
    if !models.contains(&plan.model) {
        return Ok(Some(format!(
            "it is serving {models:?}, not {}",
            plan.model
        )));
    }
    Ok(None)
}

/// Start `spark serve` as a child in its own process group, record the
/// lease, and wait for the model.
async fn start(store: &ArtifactStore, plan: ServePlan, owner_pid: u32) -> Result<SelfServed> {
    let port = atlas_plugin::benchmarks::agentic::score::free_port()?;
    let serve_args = plan.serve_args(port)?;
    super::bench_selfstart::check_box_is_free_enough(
        serve_args.gpu_memory_utilization,
        &plan.recipe_id,
        plan.limits.memory.min_free_fraction,
    )?;
    let argv = plan.argv(port)?;
    let exe = std::env::current_exe().context("current_exe")?;
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path(store))
        .with_context(|| format!("opening {}", log_path(store).display()))?;
    let mut cmd = std::process::Command::new(&exe);
    cmd.args(&argv[1..])
        .stdin(std::process::Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawning {} serve", exe.display()))?;
    let lease = Lease {
        pid: child.id(),
        port,
        model: plan.model.clone(),
        recipe_id: plan.recipe_id.clone(),
        argv_sha256: argv_fingerprint(&argv[1..]),
        binary_sha256: file_sha256(&exe)?,
        owner_pid,
        started_at: super::bench_certify::lockfile::now_unix(),
    };
    write(store, &lease)?;
    eprintln!(
        "gate: serving {} from recipe {} on port {port} as a LEASED server (pid {}); it stays up \
         after this run — `spark benchmark serve-release` stops it",
        plan.model, plan.recipe_id, lease.pid
    );
    let target = TargetEndpoint::local(port, &plan.model);
    let boot_timeout = Duration::from_secs(plan.limits.timing.boot_timeout_s);
    if let Err(e) = await_serving(&target, &plan.model, &mut child, boot_timeout).await {
        stop(&lease);
        let _ = std::fs::remove_file(lease_path(store));
        return Err(e);
    }
    eprintln!("gate: endpoint is serving {}", plan.model);
    Ok(SelfServed::external(
        target,
        plan.recipe_id,
        plan.requested,
        plan.entry,
    ))
}

/// Block until `/v1/models` names `model`, watching the child so a serve that
/// dies during startup reports so instead of timing out.
async fn await_serving(
    target: &TargetEndpoint,
    model: &str,
    child: &mut std::process::Child,
    boot_timeout: Duration,
) -> Result<()> {
    let deadline = Instant::now() + boot_timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            bail!(
                "the leased server exited ({status}) before it began serving {model:?} — see serve-lease.log"
            );
        }
        let last = match atlas_plugin::http::list_models(target, Duration::from_secs(5)).await {
            Ok(models) if models.iter().any(|m| m == model) => return Ok(()),
            Ok(models) => format!("the endpoint is serving {models:?}"),
            Err(e) => format!("{e:#}"),
        };
        if Instant::now() >= deadline {
            bail!(
                "{model:?} did not come up within {}s — {last}",
                boot_timeout.as_secs()
            );
        }
        tokio::time::sleep(POLL).await;
    }
}

/// SIGTERM the leased server's process group, wait, SIGKILL what is left.
pub fn stop(lease: &Lease) {
    let pid = lease.pid;
    let _ = std::process::Command::new("kill")
        .args(["-TERM", "--", &format!("-{pid}")])
        .status();
    let _ = std::process::Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status();
    let until = Instant::now() + STOP_GRACE;
    while pid_alive(pid) && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(250));
    }
    if pid_alive(pid) {
        let _ = std::process::Command::new("kill")
            .args(["-KILL", "--", &format!("-{pid}")])
            .status();
    }
}

/// Stop the leased server, if any, and forget the lease. Returns what was
/// released.
pub fn release(store: &ArtifactStore) -> Result<Option<Lease>> {
    let Some(lease) = read(store)? else {
        return Ok(None);
    };
    if pid_alive(lease.pid) {
        stop(&lease);
    }
    std::fs::remove_file(lease_path(store))
        .with_context(|| format!("removing {}", lease_path(store).display()))?;
    Ok(Some(lease))
}

/// Stop a leased server whose owner is gone — a campaign that died left it
/// resident. Returns what was released.
pub fn release_if_orphaned(store: &ArtifactStore) -> Result<Option<Lease>> {
    match read(store)? {
        Some(l) if !pid_alive(l.owner_pid) => release(store),
        _ => Ok(None),
    }
}

/// `spark benchmark serve-release`.
pub fn release_cmd() -> Result<i32> {
    let store = ArtifactStore::discover()?;
    match release(&store)? {
        Some(l) => {
            eprintln!(
                "released the leased server (pid {}, port {}, {})",
                l.pid, l.port, l.model
            );
            Ok(0)
        }
        None => {
            eprintln!("no leased server ({})", lease_path(&store).display());
            Ok(0)
        }
    }
}

#[cfg(test)]
#[path = "bench_lease_tests.rs"]
mod tests;
