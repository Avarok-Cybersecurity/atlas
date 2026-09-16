// SPDX-License-Identifier: AGPL-3.0-only
//! Benchmark groups: one gate satisfied by several runs.
//!
//! A group is a gate whose measurement is split across N shard runs of the
//! same benchmark that can run on different boxes at the same time. The
//! group id is what `coverage::REQUIRED`, `BENCH.toml` and `pr-taxonomy.json`
//! refer to, and it is the benchmark id every shard runs under.
//!
//! Keeping the group id equal to the old single-benchmark id is deliberate:
//! `bfcl-subset` stays `bfcl-subset`, so `REQUIRED` stays eleven entries, the
//! BENCH.toml thresholds are untouched, and every `_benches` reference keeps
//! resolving. Only the way the number is PRODUCED changes.
//!
//! # Why a group is not just "run them and average"
//!
//! Two rules make the difference between a group that means something and one
//! that quietly reports a different measurement:
//!
//! 1. **All or nothing.** A group with three of four shards present is not
//!    75% measured, it is a DIFFERENT measurement — its aggregate is computed
//!    over a sample set the thresholds were never drawn against. A missing
//!    shard is a named failure, never a pass.
//! 2. **Aggregate over counts, never over scores.** See
//!    [`crate::benchmarks::bfcl::aggregate`] — `score.py` weights
//!    hierarchically, so a mean of member scores is not the whole-set value.
//!
//! Members may be signed by different boxes only because these are
//! `Sensitivity::Correctness` gates, which is measured, not assumed — see
//! [`super::agreement`].
//!
//! # ★ A COMPLETE PARTITION AT ONE COMMIT IS THE ONLY THING THAT SATISFIES A GROUP
//!
//! Owner decision, 2026-09-13: sharded certification is ENABLED and REQUIRED
//! for `bfcl-subset` and `bfcl-subset-echolp`; 2026-09-15: the shard COUNT
//! is not fixed. A shard is the group's own benchmark run with
//! `--param shard=i/n`, filed under the group; the campaign picks `n` from
//! the fleet it has (`spark bench certify --shards N`), and the verdict
//! accepts the newest complete partition the records at the commit form
//! ([`select_partition`]). `check_one` hands every group id straight to
//! `check_group`; a whole-draw record under the group's own id is history,
//! not evidence, and the verdict says so by name when one is all a directory
//! holds. Each shard is judged by every rule a plain gate record is judged by
//! — required subject, completed frame, clean tree, signature — plus the
//! group rules above.
//!
//! # ★ SCORED OPEN, BY DECISION: THE NUMBER IS PARTITION- AND ORDER-DEPENDENT
//!
//! The certified regime keeps **cross-request SSM snapshot reuse ON**. It is
//! not `--hermetic`. That is a choice, made with the consequence measured, and
//! the consequence is this:
//!
//! Everything below makes the ARITHMETIC of a group exact: the shards are a
//! partition, the tallies sum as integers, the hierarchy is applied once. None
//! of that makes the MEASUREMENT order-independent, and on the shipped serve it
//! is not. Measured 2026-09-08 (#936) at one commit, one model, temp 0, seed 42:
//!
//! | configuration | whole (995) vs its own 4 shards |
//! |---|---|
//! | shipped (the certified regime) | **12 samples disagree** |
//! | `ATLAS_NO_TAIL_SPLIT=1` (snapshot producer off) | 4 |
//! | `ATLAS_MARCONI_MIN_TOKENS=1e8` (consumer off) | 2 |
//! | `--hermetic` (#981) | 0 |
//!
//! The cause is cross-request **SSM snapshot reuse**. A snapshot saved by one
//! request enters a shared, globally evicted pool (128 slots / 19392 MB on
//! GB10); a later request restores from whichever eligible anchor happens to
//! be there; restoring at a different depth gives numerically different SSM
//! state; at a near-tied argmax the emitted token flips. Sharding changes
//! eviction pressure because it changes run length, so it changes which anchor
//! a sample gets. The engine is bit-reproducible for a fixed request ORDER —
//! two runs of the same shard at the same commit were byte-identical — so this
//! is entirely an ordering effect, not run-to-run noise.
//!
//! The twelve samples that flipped are named in
//! [`crate::benchmarks::bfcl::sensitive`]; a run warns on each one it scores
//! and reports the count as `known_partition_sensitive`. The floors for both
//! gates are cut from the SHARDED aggregate, never from a whole-draw run, so
//! the bar and the measurement are taken under the same regime.
//!
//! Since 2026-09-15 a campaign may also run consecutive shards on one box
//! against ONE server (`spark benchmark run --serve-reuse`, verified to be
//! the server the shard would have started), so a later shard can find the
//! snapshot pool warm from an earlier one — the same mechanism, one more
//! ordering the number depends on; the record's command line says when it
//! applied.
//!
//! **What this means for anyone extending this module.** A group's aggregate is
//! exact with respect to its shards, and its shards are not guaranteed to
//! reproduce the serial run they stand in for. Do not read a passing group as
//! evidence that sharding is transparent; it is evidence that the shards,
//! run as that partition, clear the floors that were cut from a sharded run.
//!
//! `--hermetic` remains available (`spark serve --hermetic` closes the prefix
//! cache, the snapshot pool and the MTP probe; whole vs its own four shards
//! reads 0 of 995 under it) and is pinned as the SUBJECT of `kat-equality-gate`
//! only. Equality costs score — 84.22 / 84.12 open, 83.92 / 84.22 closed on
//! the whole draw — and the owner chose the open number as the certified one.

/// One shard record, as the partition rule sees it: which slice it measured,
/// at which commit, when, and the caller's handle to the record (an index
/// into whatever list the caller holds).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardRecord {
    pub index: usize,
    pub count: usize,
    pub git_sha: String,
    pub recorded_at: u64,
    /// The caller's handle — `check_group` uses the position in its list.
    pub handle: usize,
}

/// The chosen partition: its shard count, the one commit it was measured
/// at, and the caller's handles, one per index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Partition {
    pub count: usize,
    pub git_sha: String,
    pub handles: Vec<usize>,
}

/// The newest COMPLETE partition among the standing shard records — the
/// `(count, commit)` for which every index `0..count` has a record measured
/// at that commit, newest record per index — or why there is none.
///
/// The shard count is not declared anywhere: it is whatever the campaign
/// that measured the draw chose (`spark bench certify --shards N`, sized to
/// the fleet), and the records SAY what they are (`shard.index` /
/// `shard.count`, written by the driver from the rows it was handed). So
/// the verdict reads the records, groups them by count AND commit, and
/// accepts a group whose indices are all present exactly once. Two complete
/// partitions (a 4-way and an 8-way, or the same count at two commits that
/// both still stand) are both valid measurements; the newer one is the
/// branch's current word, like any other newest-first rule. A partition is
/// never assembled ACROSS commits: a group is one measurement, and a
/// re-run shard at a later commit re-opens the group until its siblings
/// join it there.
///
/// ★ WHY THIS IS NOT REDUNDANT WITH THE `samples` PIN. `samples` is pinned
/// exactly (min == max == 995) and does catch a missing or duplicated
/// SAMPLE. It does not catch a duplicated SHARD: two records of index 2 and
/// none of index 3 still hold ~995 rows between them — shard C twice and
/// shard D never — and every per-subset tally still looks plausible,
/// because the subsets are strided. The total is right and the sample set
/// is wrong, which is precisely the silently-wrong green a gate exists to
/// prevent. Newest-per-index makes a duplicate a re-run, never a double
/// count.
pub fn select_partition(
    group: &'static str,
    shards: &[ShardRecord],
) -> Result<Partition, GroupFault> {
    if shards.is_empty() {
        return Err(GroupFault::Missing {
            group,
            held: Vec::new(),
        });
    }
    let held = held_by(shards);
    // Every complete partition, ranked by the age of its newest record.
    let mut complete: Vec<(u64, &PartitionKey, &PerIndex<'_>)> = held
        .iter()
        .filter(|(key, per_index)| per_index.len() == key.0)
        .map(|(key, per_index)| {
            let newest = per_index.values().map(|s| s.recorded_at).max().unwrap_or(0);
            (newest, key, per_index)
        })
        .collect();
    complete.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.0.cmp(&a.1.0)));
    match complete.into_iter().next() {
        Some((_, (count, git_sha), per_index)) => Ok(Partition {
            count: *count,
            git_sha: git_sha.clone(),
            handles: per_index.values().map(|s| s.handle).collect(),
        }),
        None => Err(GroupFault::Missing {
            group,
            held: held
                .iter()
                .map(|((count, sha), per_index)| {
                    (*count, sha.clone(), per_index.keys().copied().collect())
                })
                .collect(),
        }),
    }
}

/// `(count, commit)`: what a partition is keyed by.
pub type PartitionKey = (usize, String);
/// The newest record per index within one partition.
pub type PerIndex<'a> = std::collections::BTreeMap<usize, &'a ShardRecord>;

/// Per `(count, commit)`, the newest record per index.
pub fn held_by(shards: &[ShardRecord]) -> std::collections::BTreeMap<PartitionKey, PerIndex<'_>> {
    let mut held: std::collections::BTreeMap<PartitionKey, PerIndex<'_>> =
        std::collections::BTreeMap::new();
    for s in shards {
        let slot = held
            .entry((s.count, s.git_sha.clone()))
            .or_default()
            .entry(s.index)
            .or_insert(s);
        if s.recorded_at > slot.recorded_at {
            *slot = s;
        }
    }
    held
}

/// A gate whose measurement is produced by a partition of shard runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchmarkGroup {
    /// The gate id — the one `REQUIRED` and `BENCH.toml` know, and the
    /// benchmark every shard of it is a run of (`--param shard=i/n`).
    pub id: &'static str,
}

/// Every group.
///
/// The ids here are the GATE ids that `coverage::REQUIRED`, `BENCH.toml` and
/// `pr-taxonomy.json` already know — deliberately unchanged, so none of those
/// move. A shard is not a benchmark of its own: it is the group's benchmark
/// run with `--param shard=i/n`, and its record is filed under the group.
pub const GROUPS: &[BenchmarkGroup] = &[
    BenchmarkGroup { id: "bfcl-subset" },
    BenchmarkGroup {
        id: "bfcl-subset-echolp",
    },
];

/// The group a benchmark id names, if any.
pub fn find(id: &str) -> Option<&'static BenchmarkGroup> {
    GROUPS.iter().find(|g| g.id == id)
}

/// Why a group is not satisfied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GroupFault {
    /// No complete partition at one commit. `held` is what IS there, per
    /// shard count and commit: the indices with a standing record.
    Missing {
        /// The group.
        group: &'static str,
        /// `(count, commit, indices present)` for every pair seen.
        held: Vec<(usize, String, Vec<usize>)>,
    },
}

impl std::fmt::Display for GroupFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing { group, held } if held.is_empty() => write!(
                f,
                "{group} is a benchmark group and has no shard record at this commit. \
                 A group is satisfied only by a COMPLETE partition of its draw \
                 (`--param shard=i/n` for every i in 0..n at one commit): an \
                 aggregate over a subset is computed on a sample set the thresholds \
                 were never drawn against, which is a different measurement, not a \
                 partial one."
            ),
            Self::Missing { group, held } => write!(
                f,
                "{group} is a benchmark group and no shard partition is complete at one \
                 commit: {}. A group is satisfied only by a COMPLETE partition of its \
                 draw — every index 0..n once, all at one commit — since an aggregate \
                 over a subset is computed on a sample set the thresholds were never \
                 drawn against, which is a different measurement, not a partial one; \
                 and a group is ONE measurement.",
                held.iter()
                    .map(|(n, sha, idx)| {
                        let missing: Vec<String> = (0..*n)
                            .filter(|i| !idx.contains(i))
                            .map(|i| i.to_string())
                            .collect();
                        format!(
                            "{n}-way at {} holds {:?}, missing {}",
                            sha.chars().take(10).collect::<String>(),
                            idx,
                            missing.join(",")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
        }
    }
}

/// Which id's BENCH.toml entry describes how to RUN this benchmark. A shard
/// is a run of its group's benchmark, so this is the identity function now
/// that shards have no ids of their own; kept as the one place a future
/// alias would be resolved.
pub fn serve_baseline_id(benchmark_id: &str) -> &str {
    benchmark_id
}
