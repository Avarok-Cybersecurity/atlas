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
