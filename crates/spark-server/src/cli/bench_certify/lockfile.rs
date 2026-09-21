// SPDX-License-Identifier: AGPL-3.0-only

//! The campaign lockfile: `.oracle_should_begin_cert` at the repo root,
//! schema `oracle_should_begin_cert/v1`.
//!
//! One box, one campaign. The file is created atomically (`link(2)`), names
//! its owner and driver, and is heartbeated between gates so a second session
//! — a human, an agent, another certify — can tell a live campaign from a
//! dead one without guessing. Liveness beats age: a lock whose driver is
//! still running is never stale, however old; a lock is reclaimed only when
//! its driver is gone AND its heartbeat is older than [`STALE_AFTER_SECS`].
//!
//! The same file the O.R.A.C.L.E skill and the shell driver wrote, so the
//! three can read each other.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

pub const LOCK_NAME: &str = ".oracle_should_begin_cert";
pub const SCHEMA: &str = "oracle_should_begin_cert/v1";
/// A heartbeat older than this, with no live driver, marks the lock stale.
pub const STALE_AFTER_SECS: u64 = 30 * 60;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct LockOwner {
    pub session_id: String,
    pub hostname: String,
    pub user: String,
    pub cwd: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Campaign {
    pub pr: Option<u64>,
    pub branch: String,
    pub anchor_sha: String,
    pub avarok_home: String,
    pub driver_pid: Option<u32>,
    pub driver_cmdline: Option<String>,
    pub started_at: Option<String>,
    pub current_gate: Option<String>,
    pub gates_done: Vec<String>,
    pub heartbeat_at: Option<String>,
    pub guard_last_rc: Option<i32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LockFile {
    pub schema: String,
    pub status: String,
    pub created_at: String,
    pub updated_at: String,
    pub expires_at: Option<String>,
    pub owner: LockOwner,
    pub campaign: Campaign,
    #[serde(default)]
    pub oracle: serde_json::Value,
    #[serde(default)]
    pub overrides: Vec<serde_json::Value>,
    #[serde(default)]
    pub history: Vec<serde_json::Value>,
    #[serde(default)]
    pub superseded: Option<serde_json::Value>,
}

/// The statuses `release_as` writes when a campaign is over. With no live
/// driver, a lock in one of these is reclaimable at once.
pub const TERMINAL_STATUSES: [&str; 2] = ["campaign_done", "aborted"];

/// The decision about an existing lock, made from its contents and two
/// liveness facts supplied by the caller.
#[derive(Debug, PartialEq, Eq)]
pub enum Existing {
    /// Someone's campaign is running; refuse, naming them.
    Live(String),
    /// Dead driver, old heartbeat: reclaim after archiving.
    Stale(String),
}

/// Is the lock at `existing` live or stale? `driver_alive` answers `kill -0`
/// for the recorded pid; `now` is unix seconds.
pub fn classify(existing: &LockFile, driver_alive: bool, now: u64) -> Existing {
    let who = format!(
        "{}@{} ({}), session {}, status {}, pr {}, driver pid {}",
        existing.owner.user,
        existing.owner.hostname,
        existing.owner.cwd,
        existing.owner.session_id,
        existing.status,
        existing
            .campaign
            .pr
            .map_or("-".to_string(), |p| p.to_string()),
        existing
            .campaign
            .driver_pid
            .map_or("-".to_string(), |p| p.to_string()),
    );
    if driver_alive {
        return Existing::Live(format!("{who}: its driver is still running"));
    }
    // A campaign that has SAID it is over holds nothing. `release_as` leaves
    // the file for the post phase to read, and the heartbeat it carries is
    // the last one the campaign wrote — fresh for an hour after a two-hour
    // fleet run. Waiting that hour out for a lock whose owner is dead and
    // whose status is terminal guards nobody; stack #1073's third campaign
    // was refused by its second's `campaign_done` file.
    if TERMINAL_STATUSES.contains(&existing.status.as_str()) {
        return Existing::Stale(format!(
            "{who}: driver gone and the campaign reported itself {}",
            existing.status
        ));
    }
    let beat = existing
        .campaign
        .heartbeat_at
        .as_deref()
        .or(Some(existing.updated_at.as_str()))
        .and_then(parse_rfc3339);
    match beat {
        Some(t) if now.saturating_sub(t) < STALE_AFTER_SECS => Existing::Live(format!(
            "{who}: heartbeat {} s ago (stale only after {STALE_AFTER_SECS} s with no driver)",
            now.saturating_sub(t)
        )),
        Some(t) => Existing::Stale(format!(
            "{who}: driver gone and heartbeat {} s old",
            now.saturating_sub(t)
        )),
        None => Existing::Stale(format!("{who}: driver gone and no readable heartbeat")),
    }
}

/// A held lock. Dropping it removes the file.
#[derive(Debug)]
pub struct LockGuard {
    path: PathBuf,
    file: LockFile,
}

impl LockGuard {
    /// Create the lock atomically. An existing live lock is refused; a stale
    /// one is archived beside itself as `<name>.stale.<unix>` and replaced.
    pub fn claim(
        root: &Path,
        owner: LockOwner,
        campaign: Campaign,
        now: u64,
        driver_alive: &dyn Fn(u32) -> bool,
    ) -> Result<Self> {
        let path = root.join(LOCK_NAME);
        if path.exists() {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            match serde_json::from_str::<LockFile>(&text) {
                Ok(existing) => {
                    let alive = existing.campaign.driver_pid.is_some_and(driver_alive);
                    match classify(&existing, alive, now) {
                        Existing::Live(why) => bail!(
                            "another certification campaign holds {}: {why}. Wait for it, \
                             or if it is truly dead remove the file.",
                            path.display()
                        ),
                        Existing::Stale(why) => {
                            let archive = root.join(format!("{LOCK_NAME}.stale.{now}"));
                            std::fs::rename(&path, &archive)
                                .with_context(|| format!("archiving {}", path.display()))?;
                            eprintln!(
                                "certify: reclaimed a stale lock ({why}); archived as {}",
                                archive.display()
                            );
                        }
                    }
                }
                Err(e) => bail!(
                    "{} exists but is not a v1 lockfile ({e}); refusing to guess whether a \
                     campaign is running — inspect and remove it",
                    path.display()
                ),
            }
        }
        let stamp = rfc3339(now);
        let file = LockFile {
            schema: SCHEMA.into(),
            status: "running_certification".into(),
            created_at: stamp.clone(),
            updated_at: stamp,
            expires_at: None,
            owner,
            campaign,
            oracle: serde_json::Value::Null,
            overrides: vec![],
            history: vec![],
            superseded: None,
        };
        let tmp = root.join(format!("{LOCK_NAME}.{}.tmp", std::process::id()));
        std::fs::write(&tmp, serde_json::to_vec_pretty(&file)?)?;
        let linked = std::fs::hard_link(&tmp, &path);
        let _ = std::fs::remove_file(&tmp);
        linked.with_context(|| {
            format!(
                "{} appeared while claiming it — another campaign started first",
                path.display()
            )
        })?;
        Ok(Self { path, file })
    }

    pub fn file(&self) -> &LockFile {
        &self.file
    }

    fn write(&self) -> Result<()> {
        let tmp = self
            .path
            .with_extension(format!("{}.tmp", std::process::id()));
        std::fs::write(&tmp, serde_json::to_vec_pretty(&self.file)?)?;
        std::fs::rename(&tmp, &self.path).context("rewriting the lockfile")
    }

    /// Record which gate is running and what the guard last said.
    pub fn beat(&mut self, current_gate: &str, guard_rc: i32, now: u64) -> Result<()> {
        let stamp = rfc3339(now);
        self.file.updated_at = stamp.clone();
        self.file.campaign.heartbeat_at = Some(stamp);
        self.file.campaign.current_gate = Some(current_gate.to_string());
        self.file.campaign.guard_last_rc = Some(guard_rc);
        self.write()
    }

    pub fn done(&mut self, gate: &str) -> Result<()> {
        self.file.campaign.gates_done.push(gate.to_string());
        self.write()
    }

    /// Leave the file in place with a terminal status (for the post phase),
    /// instead of removing it on drop.
    pub fn release_as(mut self, status: &str, now: u64) -> Result<()> {
        self.file.status = status.to_string();
        self.file.updated_at = rfc3339(now);
        self.write()?;
        self.path = PathBuf::new();
        Ok(())
    }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        if self.path.as_os_str().is_empty() {
            return;
        }
        let _ = std::fs::remove_file(&self.path);
    }
}

pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// `YYYY-MM-DDTHH:MM:SSZ`.
pub fn rfc3339(unix: u64) -> String {
    let days = unix / 86_400;
    let rem = unix % 86_400;
    let (y, m, d) = civil_from_days(days as i64);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

fn parse_rfc3339(s: &str) -> Option<u64> {
    let (date, time) = s.split_once('T')?;
    let time = time.strip_suffix('Z')?;
    let mut d = date.split('-').map(|p| p.parse::<i64>().ok());
    let (y, m, day) = (d.next()??, d.next()??, d.next()??);
    let mut t = time.split(':').map(|p| p.parse::<u64>().ok());
    let (h, mi, sec) = (t.next()??, t.next()??, t.next()??);
    let days = days_from_civil(y, m, day);
    Some(days as u64 * 86_400 + h * 3600 + mi * 60 + sec)
}

// Howard Hinnant's civil-from-days, proleptic Gregorian.
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
#[path = "lockfile_tests.rs"]
mod lockfile_tests;
