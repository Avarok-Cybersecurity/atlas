// SPDX-License-Identifier: AGPL-3.0-only
//! Benchmark groups: one gate satisfied by several runs.
//!
//! A group is a gate whose measurement is split across N member benchmarks that
//! can run on different boxes at the same time. The group id is what
//! `coverage::REQUIRED`, `BENCH.toml` and `pr-taxonomy.json` refer to; the
//! members are ordinary benchmarks that are NOT required in their own right.
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
//! 1. **All or nothing.** A group with three of four members present is not
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

/// A gate whose measurement is produced by several member runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchmarkGroup {
    /// The gate id — the one `REQUIRED` and `BENCH.toml` know.
    pub id: &'static str,
    /// The member benchmark ids, in shard order. Never required themselves.
    pub members: &'static [&'static str],
}

/// Every group.
///
/// The ids here are the GATE ids that `coverage::REQUIRED`, `BENCH.toml` and
/// `pr-taxonomy.json` already know — deliberately unchanged, so none of those
/// move. The members are ordinary registered benchmarks that are NOT required
/// in their own right.
pub const GROUPS: &[BenchmarkGroup] = &[
    BenchmarkGroup {
        id: "bfcl-subset",
        members: &[
            "bfcl-subset-a",
            "bfcl-subset-b",
            "bfcl-subset-c",
            "bfcl-subset-d",
        ],
    },
    BenchmarkGroup {
        id: "bfcl-subset-echolp",
        members: &[
            "bfcl-subset-echolp-a",
            "bfcl-subset-echolp-b",
            "bfcl-subset-echolp-c",
            "bfcl-subset-echolp-d",
        ],
    },
];

/// The group a benchmark id names, if any.
pub fn find(id: &str) -> Option<&'static BenchmarkGroup> {
    GROUPS.iter().find(|g| g.id == id)
}

/// Is this id a MEMBER of some group? Members must never be treated as
/// required gates in their own right — that would demand eleven-plus records
/// where the group needs one verdict.
pub fn member_of(id: &str) -> Option<&'static BenchmarkGroup> {
    GROUPS.iter().find(|g| g.members.contains(&id))
}

/// What a member contributed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemberRecord {
    /// The member benchmark id.
    pub id: String,
    /// The commit it was measured at.
    pub git_sha: String,
}

/// Why a group is not satisfied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GroupFault {
    /// One or more members have no record at this commit.
    Missing {
        /// The group.
        group: &'static str,
        /// Members with no record, in shard order.
        missing: Vec<&'static str>,
    },
    /// Members measured at different commits. A group is ONE measurement.
    SpansCommits {
        /// The group.
        group: &'static str,
        /// The distinct commits seen, sorted.
        commits: Vec<String>,
    },
    /// A record naming a member this group does not have.
    Foreign {
        /// The group.
        group: &'static str,
        /// The unexpected member id.
        id: String,
    },
}

impl std::fmt::Display for GroupFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing { group, missing } => write!(
                f,
                "{group} is a benchmark group and {} of its members have no record \
                 at this commit ({}). A group is satisfied only when EVERY member \
                 has run: an aggregate over a subset is computed on a sample set \
                 the thresholds were never drawn against, which is a different \
                 measurement, not a partial one.",
                missing.len(),
                missing.join(", ")
            ),
            Self::SpansCommits { group, commits } => write!(
                f,
                "{group}'s members were measured at {} different commits ({}). A \
                 group is ONE measurement; re-run the stragglers at the head you \
                 intend to merge.",
                commits.len(),
                commits.join(", ")
            ),
            Self::Foreign { group, id } => write!(
                f,
                "{id:?} is not a member of {group}; refusing to fold it into the \
                 aggregate."
            ),
        }
    }
}

/// Do these member records satisfy the group's composition rules?
///
/// Composition only — whether the aggregate then CLEARS the thresholds is
/// `scoring::check_record`'s job, on the aggregate this permits building.
pub fn composition_ok(
    group: &'static BenchmarkGroup,
    records: &[MemberRecord],
) -> Result<(), GroupFault> {
    for r in records {
        if !group.members.contains(&r.id.as_str()) {
            return Err(GroupFault::Foreign {
                group: group.id,
                id: r.id.clone(),
            });
        }
    }

    let missing: Vec<&'static str> = group
        .members
        .iter()
        .copied()
        .filter(|m| !records.iter().any(|r| r.id == *m))
        .collect();
    if !missing.is_empty() {
        return Err(GroupFault::Missing {
            group: group.id,
            missing,
        });
    }

    let mut commits: Vec<String> = records.iter().map(|r| r.git_sha.clone()).collect();
    commits.sort();
    commits.dedup();
    if commits.len() > 1 {
        return Err(GroupFault::SpansCommits {
            group: group.id,
            commits,
        });
    }
    Ok(())
}
