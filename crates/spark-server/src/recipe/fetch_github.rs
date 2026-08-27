// SPDX-License-Identifier: AGPL-3.0-only

//! The GitHub half of the recipe fetch: two endpoints and the cache write.
//!
//! Split from `fetch.rs` so each file stays inside the 250-line limit. The
//! threading contract is stated there and holds here too: every call in this
//! file blocks, so none of it may run anywhere a future is being polled.

use anyhow::{Context, Result, bail};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::Recipe;
use super::fetch::{AGENT, INDEX, Index, REPO, TIMEOUT, cache_dir, unix_now};

/// How many recipe bodies to fetch at once.
///
/// The files are a few KB each, so this is bounded by round trips rather than
/// bandwidth, and the win is almost entirely in overlapping them. Eight is
/// enough to hide the latency without opening a socket per recipe against a
/// host that would be within its rights to object.
const FETCH_WIDTH: usize = 8;

pub(super) fn try_refresh(root: &Path, cancel: &AtomicBool) -> Result<Index> {
    let (tree_sha, paths) = list_recipe_paths()?;
    if paths.is_empty() {
        bail!("{REPO}@{tree_sha} lists no recipes/**/*.yaml");
    }
    // Fetched concurrently. Serially, each of the ~25 files paid its own TCP
    // and TLS handshake — the agent below now reuses the connection, and this
    // overlaps what is left. That sequential loop was the whole of the
    // "fetching recipes…" wait.
    let next = AtomicUsize::new(0);
    let out: Mutex<Vec<(String, String)>> = Mutex::new(Vec::with_capacity(paths.len()));
    let width = FETCH_WIDTH.min(paths.len());
    std::thread::scope(|scope| {
        for _ in 0..width {
            scope.spawn(|| {
                loop {
                    if cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(path) = paths.get(i) else { return };
                    let url = format!("https://raw.githubusercontent.com/{REPO}/{tree_sha}/{path}");
                    match get(&url) {
                        Ok(body) => out.lock().push((recipe_id(path), body)),
                        // One unreachable file must not cost the other 24; it
                        // simply will not appear in this refresh.
                        Err(e) => tracing::warn!("skipping recipe {path}: {e:#}"),
                    }
                }
            });
        }
    });
    if cancel.load(Ordering::Relaxed) {
        bail!("refresh cancelled");
    }

    if out.lock().is_empty() {
        // Every file failed. Bailing (rather than returning an empty index)
        // sends this down the offline path, which serves the existing cache
        // untouched. Returning `Ok` with no recipes would have replaced a good
        // cache with an empty one BEFORE the caller could refuse it.
        bail!("{REPO}@{tree_sha} listed {} recipe file(s) but none could be fetched", paths.len());
    }

    let mut files: BTreeMap<String, String> = BTreeMap::new();
    let mut recipes = Vec::new();
    for (id, body) in out.into_inner() {
        match Recipe::parse(id.clone(), &body) {
            Ok(r) => recipes.push(r),
            // One malformed recipe upstream must not cost the other 24.
            Err(e) => tracing::warn!("skipping recipe {id}: {e:#}"),
        }
        files.insert(id, body);
    }
    // Sorted explicitly: the fetch order is now nondeterministic, and the
    // Library's row order must not be.
    recipes.sort_by(|a, b| a.id.cmp(&b.id));

    let fetched_at = unix_now();

    // The cache is replaced only by a COMPLETE fetch.
    //
    // Skipping an unreachable file is right for the index we return — one bad
    // file must not cost the other 24 this session. It is wrong for the cache:
    // writing the survivors DELETES the ones that did not come back, so a
    // transient failure of `raw.githubusercontent.com` (a proxy that resolves
    // the API host but not the raw host is a real topology) turns a complete
    // 30-recipe cache into a 3-recipe one, permanently, while reporting
    // success. Keeping the old cache is strictly better: it is complete, and it
    // is what the caller already had.
    let missing = paths.len().saturating_sub(files.len());
    let incomplete = if missing > 0 {
        Some(format!(
            "{missing} of {} recipe file(s) could not be fetched, so the cache was left alone",
            paths.len()
        ))
    } else {
        // A write failure is NOT a warning here. `sync-recipes` dispatches
        // before the tracing subscriber exists, so a `warn!` on this path goes
        // nowhere at all and the command prints "recipe index written to …"
        // over a file it never wrote — the exact success-with-stale-data this
        // command exists to prevent.
        write_cache(root, &tree_sha, fetched_at, &files)
            .err()
            .map(|e| format!("the index could not be cached: {e:#}"))
    };

    Ok(Index {
        recipes,
        tree_sha,
        fetched_at,
        offline: None,
        incomplete,
    })
}

/// `recipes/qwen3.6/foo.yaml` → `qwen3.6/foo`.
pub(super) fn recipe_id(path: &str) -> String {
    path.trim_start_matches("recipes/")
        .trim_end_matches(".yaml")
        .to_string()
}

/// One API call: the tree sha and every `recipes/**/*.yaml` under it.
fn list_recipe_paths() -> Result<(String, Vec<String>)> {
    let body = get(&format!(
        "https://api.github.com/repos/{REPO}/git/trees/main?recursive=1"
    ))
    .context("listing the recipe tree")?;
    let doc: serde_json::Value =
        serde_json::from_str(&body).context("tree response is not JSON")?;
    let sha = doc
        .get("sha")
        .and_then(|s| s.as_str())
        .context("tree response has no sha")?
        .to_string();
    // `truncated` means GitHub cut the listing short; a partial Library that
    // looks complete is worse than an error.
    if doc.get("truncated").and_then(|t| t.as_bool()) == Some(true) {
        bail!("GitHub truncated the tree listing for {REPO}");
    }
    let entries = doc
        .get("tree")
        .and_then(|t| t.as_array())
        .context("tree response has no tree")?;
    let mut paths: Vec<String> = entries
        .iter()
        .filter(|e| e.get("type").and_then(|t| t.as_str()) == Some("blob"))
        .filter_map(|e| e.get("path").and_then(|p| p.as_str()))
        .filter(|p| p.starts_with("recipes/") && p.ends_with(".yaml"))
        .map(str::to_string)
        .collect();
    paths.sort();
    Ok((sha, paths))
}

/// One agent for the whole process, so TLS sessions and TCP connections are
/// reused across requests.
///
/// `ureq::get(url)` builds a fresh agent per call, which meant every one of the
/// ~25 recipe bodies paid a full handshake to the same host. Sharing one agent
/// is the larger half of the refresh speed-up; the concurrency above is the
/// other half.
fn agent() -> &'static ureq::Agent {
    static AGENT_POOL: std::sync::OnceLock<ureq::Agent> = std::sync::OnceLock::new();
    AGENT_POOL.get_or_init(|| {
        ureq::Agent::config_builder()
            .timeout_global(Some(TIMEOUT))
            .build()
            .into()
    })
}

fn get(url: &str) -> Result<String> {
    let response = agent()
        .get(url)
        .header("User-Agent", AGENT)
        .call()
        .with_context(|| format!("GET {url}"))?;
    Ok(response.into_body().read_to_string()?)
}

/// Write the cache atomically: a half-written index read by the next start-up
/// would be indistinguishable from a corrupt one.
pub(super) fn write_cache(
    root: &Path,
    tree_sha: &str,
    fetched_at: u64,
    files: &BTreeMap<String, String>,
) -> Result<()> {
    let dir = cache_dir(root);
    std::fs::create_dir_all(&dir)?;
    let doc = serde_json::json!({
        "tree_sha": tree_sha,
        "fetched_at": fetched_at,
        "files": files,
    });
    let tmp = dir.join("index.json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&doc)?)?;
    std::fs::rename(&tmp, dir.join(INDEX))?;
    Ok(())
}

/// The date of the last commit touching one recipe file, as `YYYY-MM-DD`.
///
/// The fallback for a recipe that carries no `metadata.updated`. Costs **one**
/// rate-limited API call, which is why it is never called for a whole index:
/// 25 recipes against the unauthenticated limit of 60/hour would be exhausted
/// by two refreshes. The caller asks for one recipe at a time, on demand.
pub(super) fn commit_date(id: &str) -> Result<String> {
    // `per_page=1` — we want the most recent commit touching this path, not its
    // history. GitHub orders newest first.
    let body = get(&format!(
        "https://api.github.com/repos/{REPO}/commits?path=recipes/{id}.yaml&per_page=1"
    ))
    .with_context(|| format!("dating recipe {id}"))?;
    let doc: serde_json::Value =
        serde_json::from_str(&body).context("commits response is not JSON")?;
    let date = doc
        .get(0)
        .and_then(|c| c.get("commit"))
        .and_then(|c| c.get("committer"))
        .and_then(|c| c.get("date"))
        .and_then(|d| d.as_str())
        .with_context(|| format!("no commit found for recipe {id}"))?;
    // ISO-8601 `2026-08-01T12:34:56Z` → the date alone. A time of day is noise
    // for "when was this last updated", and the YAML key is a plain date.
    Ok(date.split('T').next().unwrap_or(date).to_string())
}
