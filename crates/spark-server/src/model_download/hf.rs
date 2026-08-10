// SPDX-License-Identifier: AGPL-3.0-only

//! The only file that talks to huggingface.co.
//!
//! Written against `ureq`, which is already in the lock, rather than adding a
//! Hub client. That was measured, not assumed: `hf-hub` 1.0 with
//! `default-features = false` pulls **147 crates** — `reqwest`, `quinn`,
//! `aws-lc-sys` (cmake + a C toolchain at build time), `redb`, `sysinfo`, the
//! `xet-*` storage stack and the `windows-*` family. For "GET some files over
//! HTTPS" that is not a trade worth making on a repo that ships a
//! multiplatform release build.
//!
//! Writing it here also buys the three things a Hub client would have made
//! awkward, all of which the UI needs: byte-level progress, resume from a
//! partial file, and cancellation that takes effect *inside* a 5 GB shard
//! rather than only between files.

use anyhow::{Context, Result, bail};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use super::plan::RemoteFile;
use super::{DownloadError, HOST};

/// Read/write chunk. Also the cancellation granularity: the flag is checked
/// once per chunk, so a cancel takes effect within a megabyte rather than at
/// the end of a multi-gigabyte file.
const CHUNK: usize = 1024 * 1024;

/// The agent, shared for connection reuse across every file in a download.
fn agent() -> &'static ureq::Agent {
    static POOL: std::sync::OnceLock<ureq::Agent> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        ureq::Agent::config_builder()
            // No global timeout: a shard is gigabytes and legitimately takes
            // minutes. Progress stalling is visible in the UI, which is the
            // honest place to notice it.
            .timeout_connect(Some(std::time::Duration::from_secs(30)))
            .build()
            .into()
    })
}

/// The token, if the user has one. Public models need none.
///
/// Resolved ONCE per job by the caller and threaded down, not called from each
/// request: a nine-shard download otherwise re-read the same file eleven
/// times, and — more importantly — a function that reaches into the
/// environment on every call cannot be tested against an unauthenticated Hub
/// without mutating process-global state.
///
/// Looked up explicitly rather than left to a client library so the behaviour
/// does not change under us: the env vars first, then the file
/// `hf auth login` writes (`huggingface-cli login` before huggingface_hub 1.0),
/// which is what a user who has "logged in" will expect to work.
pub fn token() -> Option<String> {
    let env = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"].map(|v| std::env::var(v).ok());
    let file = std::env::var_os("HOME")
        .and_then(|h| std::fs::read_to_string(Path::new(&h).join(".cache/huggingface/token")).ok());
    pick_token(&env, file.as_deref())
}

/// The precedence rule, with the lookups hoisted out.
///
/// Separated from [`token`] because the rule is the part worth testing and the
/// I/O is the part that makes it untestable: asserting on real environment
/// variables means mutating process-global state from a test, which is how you
/// get a suite that fails one run in four depending on ordering.
///
/// Blank is treated as absent throughout — an exported-but-empty `HF_TOKEN` is
/// a very common shell accident, and sending `Authorization: Bearer ` turns a
/// public model into a 401.
pub(super) fn pick_token(env: &[Option<String>], file: Option<&str>) -> Option<String> {
    for v in env.iter().flatten() {
        if !v.trim().is_empty() {
            return Some(v.trim().to_string());
        }
    }
    file.map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
}

pub(super) fn classify(repo: &str, status: u16, had_token: bool) -> DownloadError {
    match status {
        401 | 403 => DownloadError::Gated {
            repo: repo.into(),
            had_token,
        },
        404 => DownloadError::NotFound { repo: repo.into() },
        429 => DownloadError::RateLimited,
        s => DownloadError::Http {
            repo: repo.into(),
            status: s,
        },
    }
}

fn get(
    url: &str,
    token: Option<&str>,
    range_from: Option<u64>,
) -> Result<ureq::http::Response<ureq::Body>, (u16, anyhow::Error)> {
    let mut req = agent().get(url).header("User-Agent", super::AGENT);
    if let Some(t) = token {
        req = req.header("Authorization", &format!("Bearer {t}"));
    }
    if let Some(from) = range_from {
        req = req.header("Range", &format!("bytes={from}-"));
    }
    match req.call() {
        Ok(r) => Ok(r),
        Err(ureq::Error::StatusCode(code)) => Err((code, anyhow::anyhow!("HTTP {code}"))),
        Err(e) => Err((0, anyhow::Error::new(e))),
    }
}

/// The repo's current `main` revision sha, and every file in it.
pub fn repo_info(
    repo: &str,
    tok: Option<&str>,
) -> Result<(String, Vec<RemoteFile>), DownloadError> {
    let had = tok.is_some();
    let url = format!("{HOST}/api/models/{repo}");
    let body = match get(&url, tok, None) {
        Ok(r) => r
            .into_body()
            .read_to_string()
            .map_err(|e| DownloadError::Io(e.to_string()))?,
        Err((0, e)) => return Err(DownloadError::Offline(e.to_string())),
        Err((s, _)) => return Err(classify(repo, s, had)),
    };
    let doc: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| DownloadError::Io(e.to_string()))?;
    let sha = doc
        .get("sha")
        .and_then(|s| s.as_str())
        .ok_or_else(|| DownloadError::Io("model info has no sha".into()))?
        .to_string();
    let files = doc
        .get("siblings")
        .and_then(|s| s.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|e| e.get("rfilename").and_then(|n| n.as_str()))
                .map(|name| RemoteFile {
                    name: name.to_string(),
                    size: None,
                })
                .collect()
        })
        .unwrap_or_default();
    Ok((sha, files))
}

/// File sizes for a revision, best-effort.
///
/// A separate endpoint because `/api/models/{repo}` does not report sizes. If
/// this fails the download still runs — the UI shows bytes transferred without
/// a percentage, which is degraded but never blocking.
pub fn sizes(repo: &str, revision: &str, tok: Option<&str>) -> Vec<(String, u64)> {
    let url = format!("{HOST}/api/models/{repo}/tree/{revision}?recursive=1");
    let Ok(r) = get(&url, tok, None) else {
        return Vec::new();
    };
    let Ok(body) = r.into_body().read_to_string() else {
        return Vec::new();
    };
    let Ok(doc) = serde_json::from_str::<serde_json::Value>(&body) else {
        return Vec::new();
    };
    doc.as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    let path = e.get("path")?.as_str()?.to_string();
                    // LFS files report their real size in a nested object; the
                    // outer `size` for a pointer file is the pointer's size,
                    // which for a 5 GB shard reads as about 130 bytes.
                    let size = e
                        .get("lfs")
                        .and_then(|l| l.get("size"))
                        .and_then(|s| s.as_u64())
                        .or_else(|| e.get("size").and_then(|s| s.as_u64()))?;
                    Some((path, size))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Download one file into `dest`, resuming a partial one, reporting bytes.
///
/// `on_bytes` is called with the running total for THIS file. `cancel` is
/// checked every [`CHUNK`], so it takes effect inside a large shard.
///
/// Returns `Ok(false)` when cancelled, `Ok(true)` when the file is complete.
pub fn fetch_file(
    repo: &str,
    revision: &str,
    name: &str,
    dest: &Path,
    tok: Option<&str>,
    cancel: &AtomicBool,
    on_bytes: &mut dyn FnMut(u64),
) -> Result<bool, DownloadError> {
    let had = tok.is_some();
    // A `.part` sibling is the resume record; no state file of our own.
    //
    // APPENDED, not `with_extension`: that REPLACES the extension, so
    // `foo.safetensors` and `foo.json` would both map to `foo.part` — and the
    // resume path reads a byte count from that file, so a collision would
    // append one file's body onto another's and produce a corrupt weight file
    // that still passes every existence check.
    let part = part_path(dest);
    let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);

    let url = format!("{HOST}/{repo}/resolve/{revision}/{name}");
    let resp = match get(&url, tok, (have > 0).then_some(have)) {
        Ok(r) => r,
        Err((0, e)) => return Err(DownloadError::Offline(e.to_string())),
        Err((s, _)) => return Err(classify(repo, s, had)),
    };
    // 206 means the server honoured the range; 200 means it ignored it and is
    // sending the whole file, so anything already on disk must be discarded
    // rather than prepended to a second copy.
    let resuming = resp.status().as_u16() == 206;
    let mut written = if resuming { have } else { 0 };
    on_bytes(written);

    if let Some(parent) = part.parent() {
        std::fs::create_dir_all(parent).map_err(|e| DownloadError::Io(e.to_string()))?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(resuming)
        .truncate(!resuming)
        .open(&part)
        .map_err(|e| DownloadError::Io(e.to_string()))?;

    let mut reader = resp.into_body().into_reader();
    let mut buf = vec![0u8; CHUNK];
    loop {
        if cancel.load(Ordering::Relaxed) {
            // Leave the .part in place: it is resume credit, not litter.
            file.flush().ok();
            return Ok(false);
        }
        let n = reader
            .read(&mut buf)
            .map_err(|e| DownloadError::Io(e.to_string()))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(write_error)?;
        written += n as u64;
        on_bytes(written);
    }
    file.flush().map_err(|e| DownloadError::Io(e.to_string()))?;
    drop(file);
    // Rename last: a `.part` is never mistaken for a complete file, so a kill
    // at any point leaves either resume credit or a finished file.
    std::fs::rename(&part, dest).map_err(|e| DownloadError::Io(e.to_string()))?;
    Ok(true)
}

/// The resume record for `dest`.
///
/// APPENDED, not `with_extension`: that REPLACES the extension, so
/// `foo.safetensors` and `foo.json` would both map to `foo.part` — and since
/// resume reads a byte count from that file, a collision would append one
/// file's body onto another's and produce a corrupt weight file that still
/// passes every existence check.
pub fn part_path(dest: &Path) -> PathBuf {
    dest.with_file_name(format!(
        "{}.part",
        dest.file_name().unwrap_or_default().to_string_lossy()
    ))
}

/// Classify a write failure.
///
/// ENOSPC is the one worth naming: it is actionable and common on a multi-
/// gigabyte download, and "failed to write" sends the reader looking for a
/// permissions or corruption problem they do not have.
///
/// Separated out because the alternative way to test this is to fill a real
/// filesystem — and the only one small enough to fill on this box is a 61 GB
/// tmpfs shared with a machine that is serving a 27B model.
pub(super) fn write_error(e: std::io::Error) -> DownloadError {
    if e.kind() == std::io::ErrorKind::StorageFull {
        DownloadError::DiskFull
    } else {
        DownloadError::Io(e.to_string())
    }
}

/// Free bytes on the filesystem that will hold `path`.
///
/// Walks up to the nearest EXISTING ancestor, because on a first download the
/// cache directory does not exist yet — and `statvfs` on a missing path fails,
/// which silently skipped the space check at exactly the moment it was most
/// useful.
///
/// It must measure the cache root itself (or its nearest real ancestor) and
/// never a fixed `parent()`: a cache holding terabytes of checkpoints is very
/// plausibly its own mount, and its parent is then a different filesystem — so
/// the check would have guarded the wrong disk.
/// `None` on platforms with no `statvfs` — see the non-Unix arm below. Callers
/// already treat `None` as "cannot measure", so the space pre-flight is simply
/// skipped there rather than guessing.
pub fn free_bytes(path: &Path) -> Option<u64> {
    let mut cur = Some(path);
    while let Some(p) = cur {
        if let Some(n) = statvfs_avail(p) {
            return Some(n);
        }
        cur = p.parent();
    }
    None
}

/// Windows has no `statvfs`, and the answer there needs `GetDiskFreeSpaceExW`
/// — i.e. the `windows` crate, a dependency this module exists to avoid.
///
/// The honest fallback is to admit we cannot measure: the pre-flight is skipped
/// and a genuinely full disk surfaces mid-write as `DownloadError::DiskFull`,
/// which is a worse experience but an accurate one. `spark-server` is built for
/// Windows by the release matrix, so this arm is not hypothetical — an
/// unconditional `libc::statvfs` failed that build with "cannot find function
/// `statvfs` in crate `libc`".
#[cfg(not(unix))]
fn statvfs_avail(_path: &Path) -> Option<u64> {
    None
}

/// Free and total bytes of the filesystem holding `path`.
///
/// Shares one `statvfs` with [`free_bytes`] rather than adding a second
/// syscall wrapper: the download pre-flight wants "how much room is left" and
/// the startup check wants "how full is it", and those are two readings of the
/// same number.
pub fn disk_usage(path: &Path) -> Option<(u64, u64)> {
    let mut cur = Some(path);
    while let Some(p) = cur {
        if let Some(u) = statvfs_usage(p) {
            return Some(u);
        }
        cur = p.parent();
    }
    None
}

#[cfg(not(unix))]
fn statvfs_usage(_path: &Path) -> Option<(u64, u64)> {
    None
}

#[cfg(unix)]
fn statvfs_usage(path: &Path) -> Option<(u64, u64)> {
    let c = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).ok()?;
    // SAFETY: as in `statvfs_avail` — valid NUL-terminated path, `stat` read
    // only after a successful call.
    unsafe {
        let mut stat: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c.as_ptr(), &mut stat) != 0 {
            return None;
        }
        // `f_bavail` is what a NON-ROOT process may use; `f_blocks` is the
        // whole filesystem. Using `f_bfree` here would under-report fullness,
        // because it counts the reserved blocks an unprivileged writer cannot
        // touch — the ones a download would fail on anyway.
        #[allow(clippy::unnecessary_cast)]
        let free = stat.f_bavail as u64 * stat.f_frsize as u64;
        #[allow(clippy::unnecessary_cast)]
        let total = stat.f_blocks as u64 * stat.f_frsize as u64;
        (total > 0).then_some((free, total))
    }
}

#[cfg(unix)]
fn statvfs_avail(path: &Path) -> Option<u64> {
    let c = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).ok()?;
    // SAFETY: `c` is a valid NUL-terminated path; `stat` is written only by
    // the call and read only on success.
    unsafe {
        let mut stat: libc::statvfs = std::mem::zeroed();
        // Call FIRST, read after — `stat` is zeroed until `statvfs` fills it.
        if libc::statvfs(c.as_ptr(), &mut stat) != 0 {
            return None;
        }
        // The casts are NOT redundant, whatever clippy says on this host:
        // `statvfs`'s field widths differ by platform. On Linux both are u64,
        // so clippy flags them as `unnecessary_cast` — on macOS `f_frsize` is
        // u32 and `f_bavail` is u64, and without the casts the multiply does
        // not compile at all. Removing them on clippy's advice is exactly how
        // the macOS/metal CI job broke.
        #[allow(clippy::unnecessary_cast)]
        Some(stat.f_bavail as u64 * stat.f_frsize as u64)
    }
}

/// `models--org--name` under the cache root.
pub fn repo_dir(cache_root: &Path, repo: &str) -> PathBuf {
    cache_root.join(format!("models--{}", repo.replace('/', "--")))
}

/// The revision this cache currently considers current, from `refs/main`.
pub fn local_revision(cache_root: &Path, repo: &str) -> Option<String> {
    let p = repo_dir(cache_root, repo).join("refs/main");
    let s = std::fs::read_to_string(p).ok()?;
    let s = s.trim().to_string();
    (!s.is_empty()).then_some(s)
}

/// Publish a completed snapshot by writing `refs/main`, atomically.
///
/// **Called last, and only after the files verify.** `snapshot_has_weights`
/// returns true as soon as ONE shard lands, so a crash at shard 2 of 9 with
/// `refs/main` already written would give the Library a green tick and hand a
/// half-model to the loader.
pub fn publish(cache_root: &Path, repo: &str, revision: &str) -> Result<()> {
    let dir = repo_dir(cache_root, repo);
    let snapshot = dir.join("snapshots").join(revision);
    if !snapshot.join("config.json").exists() && !snapshot.join("params.json").exists() {
        bail!("refusing to publish {repo}@{revision}: no config.json or params.json");
    }
    // A REAL shard, not the resolver's `snapshot_has_weights` — that counts
    // `model.safetensors.index.json` as weights, and the index is small so it
    // lands FIRST. Using it here would publish a model consisting of a
    // tokenizer, a config and an index naming shards that are not there: the
    // resolver would then accept it and the loader would fail deep inside,
    // which is precisely the outcome writing refs/main last exists to prevent.
    let has_shard = std::fs::read_dir(&snapshot)
        .map(|rd| {
            rd.filter_map(|e| e.ok()).any(|e| {
                e.file_name()
                    .to_str()
                    .is_some_and(|n| n.ends_with(".safetensors"))
            })
        })
        .unwrap_or(false);
    if !has_shard {
        bail!("refusing to publish {repo}@{revision}: no .safetensors shard on disk");
    }
    let refs = dir.join("refs");
    std::fs::create_dir_all(&refs).context("creating refs/")?;
    let tmp = refs.join("main.tmp");
    std::fs::write(&tmp, revision)?;
    std::fs::rename(&tmp, refs.join("main"))?;
    Ok(())
}
