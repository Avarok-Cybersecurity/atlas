// SPDX-License-Identifier: AGPL-3.0-only

//! Fetching recipes from GitHub, with the cache as a first-class answer.
//!
//! **Offline is a normal state, not an error screen.** dgx3 is air-gapped and
//! a laptop on a train is not broken; a Library that is empty without a network
//! is a broken Library. So every read returns whatever is on disk, annotated
//! with its age, and the network only ever *improves* that answer.
//!
//! # Threading
//!
//! **The one rule, for the whole dashboard: the render thread never polls a
//! future. Its only interaction with work happening elsewhere is `try_recv` on
//! a channel.** `.github/workflows/tui-threading.yml` enforces it; this
//! paragraph only explains it.
//!
//! That is the rule, and not "this module is synchronous" — which is what it
//! used to say, and which contradicted `tui/chat.rs` and
//! `atlas-plugin/src/executor.rs`, both of which legitimately spawn tokio
//! tasks and answer over a `std::sync::mpsc`. Two documented contracts that
//! disagree are worse than one that is merely narrow.
//!
//! Within that rule, this module uses blocking `ureq` on plain `std::thread`s,
//! because `ureq` is already in the lock and the async alternative is not: the
//! leanest `reqwest` configuration measured here added **26 crates**, and its
//! default TLS provider added 48 including `aws-lc-sys`, which needs cmake and
//! a C toolchain at build time. For fetching a few KB of YAML that is not a
//! trade worth making, and it would have to be made again for every platform
//! the release workflow targets.
//!
//! So a 20-second fetch stays off the render loop the same way it always did:
//! [`refresh_in_background`] spawns a thread and returns a `Receiver`, plus a
//! cancel flag so quitting mid-refresh does not leave workers running.
//!
//! One request to list, then one per file, fetched concurrently over a shared
//! agent:
//!
//! 1. `GET /repos/{repo}/git/trees/main?recursive=1` — one API call, and it
//!    yields the tree sha.
//! 2. `GET raw.githubusercontent.com/{repo}/{sha}/{path}` per recipe. Pinning
//!    the sha guarantees every file comes from one commit rather than from
//!    whatever `main` pointed at between requests. `raw.` is also not subject
//!    to the API's 60 req/hr unauthenticated limit, so a refresh spends
//!    exactly one rate-limited call.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::Recipe;
use super::fetch_github::{self, try_refresh};

pub(super) const REPO: &str = "Avarok-Cybersecurity/atlas-recipes";
pub(super) const CACHE: &str = "atlas-recipes";
pub(super) const INDEX: &str = "index.json";
/// GitHub rejects a request with no User-Agent.
pub(super) const AGENT: &str = concat!("atlas-spark/", env!("CARGO_PKG_VERSION"));
pub(super) const TIMEOUT: Duration = Duration::from_secs(20);

/// What the Library renders: the recipes, and how fresh they are.
#[derive(Clone, Debug, Default)]
pub struct Index {
    pub recipes: Vec<Recipe>,
    pub tree_sha: String,
    /// Unix seconds when this was fetched from the network. 0 = never.
    pub fetched_at: u64,
    /// Set when the network failed and this came off disk instead.
    pub offline: Option<String>,
    /// Set when the network was REACHED but the result was not good enough to
    /// replace the cache with — some recipe files did not come back, or the
    /// write itself failed.
    ///
    /// Distinct from [`Self::offline`], which means the repository was never
    /// reached at all. Both mean "what is on disk is not what you just asked
    /// for", but only this one can happen on a working network, and it used to
    /// be reported as a clean success: the fetch loop logged unreachable files
    /// at `warn!` and cached whatever did arrive, so a partial fetch silently
    /// replaced a complete cache with a smaller one.
    pub incomplete: Option<String>,
}

impl Index {
    /// How old the data is, for the panel title. `None` when it is live.
    pub fn age_text(&self) -> Option<String> {
        if self.fetched_at == 0 {
            return Some("never fetched".into());
        }
        let now = unix_now();
        let secs = now.saturating_sub(self.fetched_at);
        Some(match secs {
            0..=3599 => format!("{} m old", secs / 60),
            3600..=86399 => format!("{} h old", secs / 3600),
            _ => format!("{} d old", secs / 86400),
        })
    }

    /// The one line the Library puts in its title.
    pub fn status_text(&self) -> String {
        match (&self.offline, self.age_text()) {
            (Some(_), Some(age)) => format!("⚠ {age} — offline"),
            (None, Some(age)) => age,
            (Some(e), None) => format!("⚠ offline — {e}"),
            (None, None) => "up to date".into(),
        }
    }

    /// Why the last fetch failed, phrased for someone who has to fix it.
    ///
    /// The title only has room for "offline", which tells the reader nothing
    /// they cannot already see. The cause matters and differs completely: a
    /// box with no default route needs a proxy, a 403 needs a rate-limit wait,
    /// and a TLS failure needs neither.
    pub fn offline_detail(&self) -> Option<String> {
        let raw = self.offline.as_ref()?;
        let lowered = raw.to_lowercase();
        let hint = if lowered.contains("dns")
            || lowered.contains("resolve")
            || lowered.contains("unreachable")
            || lowered.contains("no route")
        {
            "This machine has no route to github.com. Set HTTPS_PROXY to a host \
             that does — recipes are then fetched through it — or copy \
             ~/.atlas/atlas-recipes/index.json from a machine that can reach it."
        } else if lowered.contains("403") || lowered.contains("rate") {
            "GitHub is rate-limiting this IP. The listing costs one API call per \
             refresh; the cached recipes below are still usable."
        } else if lowered.contains("timed out") || lowered.contains("timeout") {
            "The request timed out. A slow or filtered link will do this; the \
             cached recipes below are still usable."
        } else {
            "The cached recipes below are still usable."
        };
        Some(format!("{raw}. {hint}"))
    }
}

pub(super) fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub(super) fn cache_dir(root: &Path) -> PathBuf {
    root.join(CACHE)
}

/// Read whatever is cached. Never touches the network, so the Library can draw
/// before a fetch has finished — or without one ever succeeding.
pub fn cached(root: &Path) -> Index {
    let path = cache_dir(root).join(INDEX);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        // An index that is not there yet is the ordinary state of a box that
        // has never synced, and an empty Library is the right answer for it.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Index::default(),
        // Anything ELSE is a fact about this machine, not an empty index, and
        // collapsing the two is how a box spends an hour on the wrong problem.
        // Measured case: `$HOME/.atlas` created by uid 1000 while the server
        // runs as uid 996. The index was complete and unreadable; `cached`
        // answered "no recipes", and `bench_selfstart` therefore told the
        // operator to run `spark sync-recipes` to populate a file that was
        // already populated. `parse_cache` below already refuses to render a
        // CORRUPT index as an empty one for exactly this reason; a permission
        // error deserves the same treatment.
        Err(e) => {
            return Index {
                offline: Some(format!("{} could not be read: {e}", path.display())),
                ..Index::default()
            };
        }
    };
    match parse_cache(&text) {
        Ok(index) => index,
        // A corrupt cache is not worth a crash or a modal; the next refresh
        // overwrites it.
        Err(e) => Index {
            offline: Some(format!("cached index unreadable: {e}")),
            ..Index::default()
        },
    }
}

fn parse_cache(text: &str) -> Result<Index> {
    let doc: serde_json::Value = serde_json::from_str(text)?;
    let files = doc
        .get("files")
        .and_then(|f| f.as_object())
        .context("no `files` object")?;
    let mut recipes = Vec::new();
    for (id, content) in files {
        let Some(body) = content.as_str() else {
            bail!("{id} is not text");
        };
        // A single unreadable recipe must not blank the whole Library.
        match Recipe::parse(id.clone(), body) {
            Ok(r) => recipes.push(r),
            Err(e) => tracing::warn!("skipping cached recipe {id}: {e:#}"),
        }
    }
    recipes.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(Index {
        recipes,
        tree_sha: doc
            .get("tree_sha")
            .and_then(|s| s.as_str())
            .unwrap_or_default()
            .to_string(),
        fetched_at: doc.get("fetched_at").and_then(|s| s.as_u64()).unwrap_or(0),
        offline: None,
        incomplete: None,
    })
}

/// Fetch from GitHub, falling back to the cache on any failure.
///
/// Blocking, and safe to call directly only from a thread that is allowed to
/// block for [`TIMEOUT`]. From the UI use [`refresh_in_background`].
pub fn refresh(root: &Path, cancel: &AtomicBool) -> Index {
    refresh_with(root, || try_refresh(root, cancel))
}

/// The fallback rule, with the network injected so it can be tested without
/// one. Any fetch failure serves the cache, annotated with why it is stale.
fn refresh_with(root: &Path, fetch: impl FnOnce() -> Result<Index>) -> Index {
    match fetch() {
        Ok(index) => index,
        Err(e) => {
            let mut fallback = cached(root);
            fallback.offline = Some(one_line(&format!("{e:#}")));
            fallback
        }
    }
}

/// Run [`refresh`] on a dedicated thread, delivering the result over a channel.
///
/// A plain `std::thread` rather than `tokio::spawn_blocking`: this side of the
/// program has no runtime, and borrowing one only to run blocking I/O on it
/// would be exactly the mixing this module avoids. A failed spawn yields a
/// receiver that resolves to the cache, so the caller has no error path.
///
/// The returned flag cancels the refresh. It is checked between files rather
/// than inside one, which is enough: the files are a few KB each, so the wait
/// it removes is the remaining *queue*, not the request in flight. Quitting the
/// dashboard mid-refresh no longer leaves workers running to the 20 s timeout.
pub fn refresh_in_background(root: &Path) -> (std::sync::mpsc::Receiver<Index>, Arc<AtomicBool>) {
    let owned = root.to_path_buf();
    let cancel = Arc::new(AtomicBool::new(false));
    let rx = crate::tui::worker::spawn(
        "atlas-recipes",
        {
            let cancel = Arc::clone(&cancel);
            move || refresh(&owned, &cancel)
        },
        // Still answer, with what is on disk.
        |e| {
            let mut index = cached(root);
            index.offline = Some(format!("fetcher thread unavailable: {e}"));
            index
        },
    );
    (rx, cancel)
}

/// Collapse a multi-line error chain into something a title bar can hold.
fn one_line(s: &str) -> String {
    let flat = s.replace('\n', " ");
    if flat.chars().count() <= 90 {
        return flat;
    }
    flat.chars().take(90).collect::<String>() + "…"
}

#[cfg(test)]
#[path = "fetch_tests.rs"]
mod tests;

/// Ask GitHub when one recipe was last changed, off the render thread.
///
/// The lazy half of the recipe date: `metadata.updated` is authoritative when a
/// recipe carries one, and this fills the gap for the rest. It is deliberately
/// **one recipe at a time, on demand** — see [`fetch_github::commit_date`] for
/// why dating the whole index is not an option.
///
/// Same threading contract as [`refresh_in_background`], and for the same
/// reason: a plain `std::thread` and a `std::sync::mpsc::Receiver` the UI polls
/// on its tick. Nothing here touches the async runtime.
///
/// The recipe id is echoed back in the message because by the time an answer
/// arrives the selection may have moved, and a date applied to whatever happens
/// to be selected then would be silently wrong.
pub fn updated_in_background(id: &str) -> std::sync::mpsc::Receiver<(String, Option<String>)> {
    let owned = id.to_string();
    let fallback_id = id.to_string();
    crate::tui::worker::spawn(
        "atlas-recipe-date",
        move || {
            let date = fetch_github::commit_date(&owned)
                .map_err(|e| {
                    // Offline is normal here, exactly as for the index: a date
                    // we could not look up is an absent row, never an error
                    // screen.
                    tracing::debug!("could not date recipe {owned}: {e:#}");
                })
                .ok();
            (owned, date)
        },
        |_| (fallback_id, None),
    )
}
