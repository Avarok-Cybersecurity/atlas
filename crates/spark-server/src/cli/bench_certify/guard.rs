// SPDX-License-Identifier: AGPL-3.0-only

//! The drift guard `scripts/campaign-guard.sh` implements, in the binary.
//!
//! On 2026-08-28 a ten-gate campaign ran nine hours against a branch; twenty
//! minutes into the last gate a collaborator pushed one 23-line change to a
//! perf path, and every record was invalidated — discovered at the end. This
//! asks, between gates and on a timer during them, whether the branch still
//! describes the tree being measured. It answers a CONTENT question: a push
//! that touches only docs is harmless and must not abort anything.
//!
//! An error is never "safe to continue". A guard that cannot fetch reports
//! that, and the campaign stops rather than guessing.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use atlas_plugin::gate::PERF_PATHS;

/// The two git questions the guard asks, behind a trait so the decision is
/// testable without a repository.
pub trait Git {
    /// Fetch `remote_ref` (`REMOTE/BRANCH`) and return its head sha.
    fn fetch_head(&self, remote_ref: &str) -> Result<String>;
    /// The `PERF_PATHS` files that differ between two commits.
    fn changed_perf_paths(&self, from: &str, to: &str) -> Result<Vec<String>>;
}

/// What the guard found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Drift {
    /// The anchor is still the head of the branch.
    Unmoved,
    /// The branch moved, but nothing the gate looks at changed.
    MovedHarmlessly { head: String },
    /// A perf path changed: whatever is running measures a dead tree.
    PerfPathMoved { head: String, paths: Vec<String> },
}

/// How many consecutive checks may fail to ANSWER before a poller treats
/// the silence as a verdict. The guard runs every [`super::GUARD_EVERY`]
/// (60 s) while a unit is in flight, so this is ~five minutes of a mute
/// network. Stack 1089308's third campaign was aborted 35 minutes in, three
/// units at 60-80 %, by ONE `git fetch` that could not resolve github.com
/// for a moment. "Could not check" is still never "safe": after this many
/// misses in a row the campaign stops exactly as before — it just does not
/// stop on the first one.
pub const BLIND_TICKS_ALLOWED: u32 = 5;

/// What a poller does with one guard result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Judgement {
    /// The branch still describes the tree; carry on.
    Fine,
    /// The guard could not answer, `n` times in a row so far; carry on and
    /// say so.
    Blind(u32),
    /// Hand this to the campaign: it stops.
    Stop(Result<Drift, String>),
}

/// The blind budget, owned by whichever loop polls the guard. Pure.
#[derive(Debug, Default)]
pub struct Poll {
    blind: u32,
}

impl Poll {
    pub fn judge(&mut self, result: Result<Drift, String>) -> Judgement {
        match result {
            Ok(Drift::Unmoved) | Ok(Drift::MovedHarmlessly { .. }) => {
                self.blind = 0;
                Judgement::Fine
            }
            Ok(moved @ Drift::PerfPathMoved { .. }) => {
                self.blind = 0;
                Judgement::Stop(Ok(moved))
            }
            Err(e) => {
                self.blind += 1;
                if self.blind <= BLIND_TICKS_ALLOWED {
                    Judgement::Blind(self.blind)
                } else {
                    Judgement::Stop(Err(format!("{e} — {} checks in a row", self.blind)))
                }
            }
        }
    }
}

/// Ask the two questions. `Err` means the guard could not answer.
pub fn drift(git: &dyn Git, anchor: &str, remote_ref: &str) -> Result<Drift> {
    let head = git.fetch_head(remote_ref)?;
    if head.starts_with(anchor) || anchor.starts_with(&head) {
        return Ok(Drift::Unmoved);
    }
    let paths = git.changed_perf_paths(anchor, &head)?;
    if paths.is_empty() {
        Ok(Drift::MovedHarmlessly { head })
    } else {
        Ok(Drift::PerfPathMoved { head, paths })
    }
}

/// The real thing: `git` in a checkout.
pub struct GitCli {
    pub root: PathBuf,
}

impl GitCli {
    fn run(&self, args: &[&str]) -> Result<String> {
        let out = std::process::Command::new("git")
            .arg("-C")
            .arg(&self.root)
            .args(args)
            .stdin(std::process::Stdio::null())
            .output()
            .with_context(|| format!("running git {args:?}"))?;
        if !out.status.success() {
            bail!(
                "git {} failed: {}",
                args.join(" "),
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }
}

impl Git for GitCli {
    fn fetch_head(&self, remote_ref: &str) -> Result<String> {
        let (remote, branch) = remote_ref
            .split_once('/')
            .with_context(|| format!("guard ref {remote_ref:?} is not REMOTE/BRANCH"))?;
        self.run(&["fetch", "-q", remote, branch])?;
        self.run(&["rev-parse", "FETCH_HEAD"])
    }

    fn changed_perf_paths(&self, from: &str, to: &str) -> Result<Vec<String>> {
        let mut args = vec!["diff", "--name-only", from, to, "--"];
        args.extend(PERF_PATHS);
        Ok(self
            .run(&args)?
            .lines()
            .map(str::to_owned)
            .filter(|l| !l.is_empty())
            .collect())
    }
}

/// `REMOTE/BRANCH` for HEAD's upstream, if it has one.
pub fn upstream_of_head(root: &Path) -> Option<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args([
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ])
        .stdin(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (s.contains('/') && !s.is_empty()).then_some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake {
        head: Result<String, String>,
        changed: Result<Vec<String>, String>,
    }

    impl Git for Fake {
        fn fetch_head(&self, _: &str) -> Result<String> {
            self.head.clone().map_err(|e| anyhow::anyhow!(e))
        }
        fn changed_perf_paths(&self, _: &str, _: &str) -> Result<Vec<String>> {
            self.changed.clone().map_err(|e| anyhow::anyhow!(e))
        }
    }

    #[test]
    fn an_unmoved_branch_is_unmoved_even_with_a_short_anchor() {
        let g = Fake {
            head: Ok("1a0dc88a8c9083bb956bd84cafa2cccbdb8e6e18".into()),
            changed: Ok(vec!["crates/x.rs".into()]),
        };
        assert_eq!(
            drift(&g, "1a0dc88a8c", "avarok/main").unwrap(),
            Drift::Unmoved
        );
    }

    #[test]
    fn a_docs_only_move_is_harmless() {
        let g = Fake {
            head: Ok("bbbb".into()),
            changed: Ok(vec![]),
        };
        assert_eq!(
            drift(&g, "aaaa", "avarok/main").unwrap(),
            Drift::MovedHarmlessly {
                head: "bbbb".into()
            }
        );
    }

    #[test]
    fn a_perf_path_move_names_the_paths() {
        let g = Fake {
            head: Ok("bbbb".into()),
            changed: Ok(vec!["crates/spark-model/src/lib.rs".into()]),
        };
        assert_eq!(
            drift(&g, "aaaa", "avarok/main").unwrap(),
            Drift::PerfPathMoved {
                head: "bbbb".into(),
                paths: vec!["crates/spark-model/src/lib.rs".into()]
            }
        );
    }

    /// A poller retries a mute guard for the blind budget and stops after it;
    /// any answer resets the run. NEGATIVE CONTROL: a perf-path move stops at
    /// once, budget or no budget, and the (budget + 1)-th miss stops too.
    #[test]
    fn a_poller_tolerates_a_mute_guard_for_the_budget_and_no_longer() {
        let mut p = Poll::default();
        for i in 1..=BLIND_TICKS_ALLOWED {
            assert_eq!(p.judge(Err("dns".into())), Judgement::Blind(i));
        }
        assert_eq!(p.judge(Ok(Drift::Unmoved)), Judgement::Fine);
        assert_eq!(p.judge(Err("dns".into())), Judgement::Blind(1));
        for _ in 1..BLIND_TICKS_ALLOWED {
            assert!(matches!(p.judge(Err("dns".into())), Judgement::Blind(_)));
        }
        match p.judge(Err("dns".into())) {
            Judgement::Stop(Err(why)) => assert!(why.contains("6 checks in a row"), "{why}"),
            other => panic!("{other:?}"),
        }
        let moved = Drift::PerfPathMoved {
            head: "b".into(),
            paths: vec!["crates/x.rs".into()],
        };
        let mut fresh = Poll::default();
        assert_eq!(fresh.judge(Ok(moved.clone())), Judgement::Stop(Ok(moved)));
    }

    /// NEGATIVE CONTROL: "could not answer" is an error, never `Unmoved`.
    #[test]
    fn a_fetch_failure_is_an_error_not_a_pass() {
        let g = Fake {
            head: Err("could not fetch".into()),
            changed: Ok(vec![]),
        };
        assert!(drift(&g, "aaaa", "avarok/main").is_err());
        let g = Fake {
            head: Ok("bbbb".into()),
            changed: Err("diff failed".into()),
        };
        assert!(drift(&g, "aaaa", "avarok/main").is_err());
    }
}
