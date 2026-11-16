// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache csrc/storage_backends/fs/connector.cpp and lmcache/v1/distributed/key_codec.py (commit 3aefae1, Apache-2.0); the small translated pieces are in `disk/record.rs`, which carries their attribution.

//! Responsibility: the disk tier and warm restart of the host KV tier (lane L27, row
//! `kv_tier_disk`, contract 3.7.5): its configuration, the [`DiskTier`] type, opening a directory
//! (lock, restart scan) and the [`ColdStore`] implementation.
//!
//! * **Records**: one file per block, `<key hex>.kvb`, written to `<key hex>.<pid>.<n>.tmp` and
//!   renamed into place, so a reader never sees a partial record (`disk/record.rs`). O_DIRECT is
//!   used only with an aligned buffer, size and offset, and falls back to buffered I/O with one
//!   log line when the filesystem refuses it.
//! * **Restart**: no manifest. [`DiskTier::open`] scans the directory, removes stray `.tmp`
//!   files and records whose header (magic, version, header checksum, key, size) is invalid, and
//!   indexes the rest by file mtime. [`ColdStore::adopt_on_restart`] then returns the keys whose
//!   header carries the served model's fingerprint; records of another fingerprint are never
//!   adopted or promoted and age out first. Payload checksums (and stored tokens, when the host
//!   tier passes them) are checked on every promote (`disk/ops.rs`).
//! * **Budget**: `--kv-tier-disk-gb`, least recently used first (by last use, seeded from mtime,
//!   and a hit touches the mtime so the order survives a restart; `disk/index.rs`).
//! * **One process per directory**: an advisory `LOCK` file (`flock`) refuses a second tier.
//!
//! Off unless the row is on, `--kv-tier-dir` and `--kv-tier-disk-gb` are given and the host tier
//! exists; `--hermetic` refuses it (it carries state across restarts). This crate never reads the
//! row: `port_apply` passes the resolved value to [`configure`], and the host-tier glue (lane
//! L26b) calls [`attach_configured`] after it named the tier's namespace.

mod index;
mod ops;
mod record;
pub mod stats;

#[cfg(test)]
#[path = "disk_tests.rs"]
mod tests;

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result, anyhow, bail};
use parking_lot::Mutex;

use self::index::{State, unix_nanos};
use self::record::{AlignedBuf, IoMode, SUFFIX, TMP_SUFFIX};
pub use self::stats::{DiskSnapshot, DiskStats, render_prometheus};
use super::{BlockKey, ColdStore, KvHostTier};

/// One GiB, the unit of `--kv-tier-disk-gb`.
pub const GIB: u64 = 1 << 30;

/// What `--kv-tier-dir` and `--kv-tier-disk-gb` resolve to when the disk tier is on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiskConfig {
    pub dir: PathBuf,
    pub budget_bytes: u64,
    /// Try O_DIRECT (Linux only; falls back to buffered I/O on the first refusal).
    pub direct_io: bool,
}

/// Validate the disk tier's inputs: `row_on` is the RESOLVED `kv_tier_disk` row, `hermetic` the
/// resolved `--hermetic`. `Ok(None)` = the disk tier stays off.
pub fn validate(
    row_on: bool,
    dir: Option<&Path>,
    disk_gb: Option<u64>,
    hermetic: bool,
) -> Result<Option<DiskConfig>> {
    if !row_on {
        return Ok(None);
    }
    let Some(dir) = dir else {
        bail!(
            "kv_tier_disk is on (AVAROK_KV_TIER_DISK or the target) but --kv-tier-dir is not set"
        );
    };
    let gb = disk_gb.unwrap_or(0);
    if gb == 0 {
        bail!(
            "kv_tier_disk is on but --kv-tier-disk-gb is missing or 0; give the disk budget in GiB"
        );
    }
    if hermetic {
        bail!(
            "--hermetic refuses the disk KV tier: it carries KV state across requests and restarts \
             (unset AVAROK_KV_TIER_DISK or drop --hermetic)"
        );
    }
    if dir.exists() && !dir.is_dir() {
        bail!(
            "--kv-tier-dir {} exists and is not a directory",
            dir.display()
        );
    }
    let budget_bytes = gb
        .checked_mul(GIB)
        .ok_or_else(|| anyhow!("--kv-tier-disk-gb {gb} overflows"))?;
    Ok(Some(DiskConfig {
        dir: dir.to_path_buf(),
        budget_bytes,
        direct_io: true,
    }))
}

static CONFIG: OnceLock<Option<DiskConfig>> = OnceLock::new();

/// Publish the disk tier's configuration once at startup (`port_apply`). Flags given while the row
/// is off are reported and ignored.
pub fn configure(
    row_on: bool,
    dir: Option<&Path>,
    disk_gb: Option<u64>,
    hermetic: bool,
) -> Result<()> {
    let cfg = validate(row_on, dir, disk_gb, hermetic)?;
    if cfg.is_none() && (dir.is_some() || disk_gb.is_some()) {
        tracing::info!(
            "kv tier disk: --kv-tier-dir/--kv-tier-disk-gb given but the kv_tier_disk row is off"
        );
    }
    if let Some(c) = &cfg {
        tracing::info!(
            "kv tier disk: on, dir {}, budget {} GiB (attached when the host KV tier starts)",
            c.dir.display(),
            c.budget_bytes / GIB
        );
    }
    if CONFIG.set(cfg).is_err() {
        tracing::debug!("kv tier disk: configuration already published; keeping the first");
    }
    Ok(())
}

/// The published configuration, `None` while the disk tier is off.
pub fn configured() -> Option<&'static DiskConfig> {
    CONFIG.get().and_then(|c| c.as_ref())
}

/// Open the configured disk tier for `tier` and attach it as the tier's cold store. Call it after
/// `KvHostTier::set_namespace` (attaching adopts the records of that namespace). `Ok(None)` when
/// the disk tier is off. An error leaves `tier` without a cold store and serving normally.
pub fn attach_configured(tier: &KvHostTier) -> Result<Option<Arc<DiskTier>>> {
    let Some(cfg) = configured() else {
        return Ok(None);
    };
    let disk = Arc::new(DiskTier::open(cfg, tier.block_bytes())?);
    tier.set_cold_store(disk.clone());
    Ok(Some(disk))
}

/// The disk tier: one directory of block records under a byte budget. `Send + Sync`.
pub struct DiskTier {
    dir: PathBuf,
    budget: u64,
    payload_bytes: usize,
    align: usize,
    io: IoMode,
    state: Mutex<State>,
    scratch: Mutex<Vec<AlignedBuf>>,
    tmp_seq: AtomicU64,
    stats: Arc<DiskStats>,
    _lock: Option<File>,
}

#[cfg(unix)]
fn lock_dir(dir: &Path) -> Result<Option<File>> {
    use std::os::unix::io::AsRawFd;
    let path = dir.join("LOCK");
    let f = File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .with_context(|| format!("open {}", path.display()))?;
    // SAFETY: `f` is an open file for the whole call; flock only takes its descriptor and flags
    // and keeps no pointer.
    let rc = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if rc != 0 {
        bail!(
            "--kv-tier-dir {} is in use by another disk tier ({} is locked)",
            dir.display(),
            path.display()
        );
    }
    Ok(Some(f))
}

#[cfg(not(unix))]
fn lock_dir(_dir: &Path) -> Result<Option<File>> {
    Ok(None)
}

#[cfg(unix)]
fn dir_align(dir: &Path) -> usize {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(dir)
        .map(|m| record::clamp_align(m.blksize()))
        .unwrap_or(record::MIN_ALIGN)
}

#[cfg(not(unix))]
fn dir_align(_dir: &Path) -> usize {
    record::clamp_align(record::MIN_ALIGN as u64)
}

impl DiskTier {
    /// Open (create) the directory, take its lock, remove stray temp files and invalid records,
    /// index the valid ones and trim to the budget. `payload_bytes` is the host tier's block size.
    pub fn open(cfg: &DiskConfig, payload_bytes: usize) -> Result<Self> {
        if payload_bytes == 0 {
            bail!("kv tier disk: the host tier's block size is 0");
        }
        std::fs::create_dir_all(&cfg.dir)
            .with_context(|| format!("create {}", cfg.dir.display()))?;
        let lock = lock_dir(&cfg.dir)?;
        let stats = Arc::new(DiskStats::default());
        stats::register(&stats);
        stats.set_capacity(cfg.budget_bytes);
        let io = IoMode::new(cfg.direct_io);
        stats.set_direct_io(io.direct());
        let tier = Self {
            dir: cfg.dir.clone(),
            budget: cfg.budget_bytes,
            payload_bytes,
            align: dir_align(&cfg.dir),
            io,
            state: Mutex::new(State::default()),
            scratch: Mutex::new(Vec::new()),
            tmp_seq: AtomicU64::new(0),
            stats,
            _lock: lock,
        };
        tier.scan()?;
        let (n, bytes) = tier.usage();
        tracing::info!(
            "kv tier disk: {} holds {n} record(s), {:.2} GiB of {} GiB; alignment {}, O_DIRECT {}",
            tier.dir.display(),
            bytes as f64 / GIB as f64,
            tier.budget / GIB,
            tier.align,
            if tier.io.direct() { "on" } else { "off" }
        );
        Ok(tier)
    }

    fn scan(&self) -> Result<()> {
        let rd =
            std::fs::read_dir(&self.dir).with_context(|| format!("scan {}", self.dir.display()))?;
        let mut st = self.state.lock();
        for entry in rd.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if name.ends_with(TMP_SUFFIX) {
                // A record a previous process never renamed into place (it died mid-write).
                let _ = std::fs::remove_file(&path);
                self.stats.stray(1);
                continue;
            }
            let Some(key) = name.strip_suffix(SUFFIX).and_then(record::key_from_hex) else {
                continue; // the LOCK file, or a file that is not ours
            };
            match Self::check_file(&path, &key) {
                Ok((h, nanos)) => st.insert(key, h.namespace, h.payload_len, h.file_len(), nanos),
                Err(e) => {
                    tracing::debug!("kv tier disk: removing invalid {}: {e:#}", path.display());
                    let _ = std::fs::remove_file(&path);
                    self.stats.corrupt();
                }
            }
        }
        let victims = st.victims(0, self.budget);
        self.unlink(&victims);
        self.stats.evicted(victims.len() as u64);
        self.publish_usage(&st);
        Ok(())
    }

    fn check_file(path: &Path, key: &[u8; 16]) -> Result<(record::Header, u64)> {
        let meta = std::fs::metadata(path)?;
        let h = record::read_header(path)?;
        if &h.key != key {
            bail!("header key does not match the file name");
        }
        if meta.len() != h.file_len() {
            bail!("file is {} bytes, header says {}", meta.len(), h.file_len());
        }
        let nanos = meta.modified().map(unix_nanos).unwrap_or(0);
        Ok((h, nanos))
    }

    fn path_of(&self, key: &[u8; 16]) -> PathBuf {
        self.dir.join(format!("{}{SUFFIX}", record::key_hex(key)))
    }

    fn unlink(&self, keys: &[[u8; 16]]) {
        for k in keys {
            let _ = std::fs::remove_file(self.path_of(k));
        }
    }

    /// Records and bytes currently held.
    pub fn usage(&self) -> (usize, u64) {
        let st = self.state.lock();
        (st.entries.len(), st.used)
    }

    pub fn stats(&self) -> DiskSnapshot {
        self.stats.snapshot()
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn budget_bytes(&self) -> u64 {
        self.budget
    }

    /// The alignment records are written with (the directory's block size, at least 4096).
    pub fn align(&self) -> usize {
        self.align
    }
}

impl ColdStore for DiskTier {
    fn demote(&self, k: &BlockKey, b: &[u8]) -> Result<()> {
        self.put(&k.0, self.namespace()?, &[], b)
    }

    fn promote(&self, k: &BlockKey, out: &mut [u8]) -> Result<bool> {
        self.get(&k.0, None, out)
    }

    fn adopt_on_restart(&self, ns_fp: u64) -> Result<Vec<BlockKey>> {
        Ok(self.adopt(ns_fp))
    }

    /// The record keeps `tokens`, so a later promote can compare them.
    fn demote_with_tokens(&self, k: &BlockKey, tokens: &[u32], b: &[u8]) -> Result<()> {
        self.put(&k.0, self.namespace()?, tokens, b)
    }

    /// A record whose stored tokens differ from `tokens` is a miss (and is removed).
    fn promote_with_tokens(&self, k: &BlockKey, tokens: &[u32], out: &mut [u8]) -> Result<bool> {
        self.get(&k.0, Some(tokens), out)
    }
}

impl Drop for DiskTier {
    fn drop(&mut self) {
        stats::retire(&self.stats);
    }
}
