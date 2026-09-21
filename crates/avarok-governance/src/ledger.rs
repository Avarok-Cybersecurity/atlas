// SPDX-License-Identifier: AGPL-3.0-only

//! Reading, appending, and materialising a journey.

use std::collections::BTreeSet;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use anyhow::{Context, Result};
use lattice_core::{CollectionConfig, CollectionEngine, Distance, HnswConfig, Point, VectorConfig};

use crate::event::Event;

/// Every event recorded for one pull request, in the order they were written.
#[derive(Debug, Clone, Default)]
pub struct Journey {
    pub events: Vec<Event>,
}

impl Journey {
    /// Deduplicate by [`Event::identity`], keeping the first occurrence.
    ///
    /// The file is a set, not a log: replaying a CI job appends the same
    /// records again, and a reader that counted them twice would report a gate
    /// as having been evaluated more often than it was.
    pub fn deduplicated(mut self) -> Self {
        let mut seen = BTreeSet::new();
        self.events.retain(|e| seen.insert(e.identity()));
        self
    }

    /// Events for one gate id, oldest first.
    pub fn gate_history<'a>(&'a self, gate: &'a str) -> impl Iterator<Item = &'a Event> {
        self.events.iter().filter(
            move |e| matches!(&e.kind, crate::event::EventKind::Gate { id, .. } if id == gate),
        )
    }
}

/// Append one event to a per-PR file, creating it if needed.
///
/// ★ One file per pull request, never a shared one. Two PRs cannot then touch
/// the same path, so the classic shared-state-file collision is designed out
/// rather than resolved. Within a file, records are only appended, so a textual
/// merge is union — declare `governance/*.jsonl merge=union` in
/// `.gitattributes` and concurrent appends stop conflicting entirely.
pub fn append(path: &Path, event: &Event) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let line = serde_json::to_string(event).context("encoding journey event")?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening {} for append", path.display()))?;
    writeln!(file, "{line}").with_context(|| format!("appending to {}", path.display()))
}

/// Read a journey, skipping blank lines.
///
/// A malformed line is an error rather than a silent skip. The ledger's whole
/// value is that it is complete; a reader that quietly dropped what it could
/// not parse would report a partial history as a full one.
pub fn read_all(path: &Path) -> Result<Journey> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut events = Vec::new();
    for (n, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| format!("reading {} line {}", path.display(), n + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        events.push(
            serde_json::from_str(&line)
                .with_context(|| format!("parsing {} line {}", path.display(), n + 1))?,
        );
    }
    Ok(Journey { events })
}

/// Vector width of the materialised collection.
///
/// ★ One, and the vectors are not embeddings. This build materialises the
/// GRAPH only; the vector field exists because the engine requires one, and is
/// filled with a constant. When the embedder is wired in, this becomes the
/// embedding dimension **read from the first live response** — never a
/// hardcoded constant, because a model swap silently changing the width would
/// leave the index returning confident nonsense. Keeping the placeholder
/// obviously degenerate is what stops it being mistaken for a real dimension
/// later.
const PLACEHOLDER_DIM: usize = 1;

/// Build the in-memory graph from a journey.
///
/// Returns an engine holding one node per event plus one per commit, with
/// `observed` edges from commit to event. Nothing is written to disk: the graph
/// is derived data, rebuilt on demand, and committing it would put an
/// unmergeable binary in the merge path.
pub fn materialize(journey: &Journey) -> Result<CollectionEngine> {
    let config = CollectionConfig::new(
        "journey",
        VectorConfig::new(PLACEHOLDER_DIM, Distance::Cosine),
        HnswConfig {
            m: 16,
            m0: 32,
            ml: HnswConfig::recommended_ml(16),
            ef: 100,
            ef_construction: 200,
        },
    )
    .with_relation("observed", 0)
    .with_relation("precedes", 1);

    let mut engine =
        CollectionEngine::new(config).map_err(|e| anyhow::anyhow!("creating collection: {e}"))?;

    // Commits get the low ids so an id is stable as events are appended: a
    // commit's node must not move when a later run adds events, or edges
    // recorded by an earlier materialisation would point somewhere else.
    let mut commits: Vec<&str> = journey.events.iter().map(|e| e.head_sha.as_str()).collect();
    commits.sort_unstable();
    commits.dedup();

    let mut points = Vec::new();
    for (i, sha) in commits.iter().enumerate() {
        points.push(
            Point::new_vector(i as u64, vec![1.0; PLACEHOLDER_DIM])
                .with_field("label", br#""commit""#.to_vec())
                .with_field("sha", serde_json::to_vec(sha).unwrap_or_default()),
        );
    }
    let commit_base = commits.len() as u64;
    for (i, event) in journey.events.iter().enumerate() {
        points.push(
            Point::new_vector(commit_base + i as u64, vec![1.0; PLACEHOLDER_DIM])
                .with_field(
                    "label",
                    serde_json::to_vec(event.node_label()).unwrap_or_default(),
                )
                .with_field("at", serde_json::to_vec(&event.at).unwrap_or_default())
                .with_field("kind", serde_json::to_vec(&event.kind).unwrap_or_default()),
        );
    }
    engine
        .upsert_points(points)
        .map_err(|e| anyhow::anyhow!("upserting journey points: {e}"))?;

    for (i, event) in journey.events.iter().enumerate() {
        let Ok(commit_idx) = commits.binary_search(&event.head_sha.as_str()) else {
            continue;
        };
        engine
            .add_edge(commit_idx as u64, commit_base + i as u64, "observed", 1.0)
            .map_err(|e| anyhow::anyhow!("adding observed edge: {e}"))?;
    }

    Ok(engine)
}

/// The conventional path for a pull request's journey.
pub fn path_for(root: &Path, pr: u64) -> std::path::PathBuf {
    root.join("governance").join(format!("pr-{pr}.jsonl"))
}
