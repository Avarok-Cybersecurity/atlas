// SPDX-License-Identifier: AGPL-3.0-only

//! Everything that has to be true BEFORE a campaign spends its first GPU
//! minute, evaluated as a pure function over facts gathered once.
//!
//! Every item here has cost real hours: a record measured from a dirty tree
//! (belongs to no commit), a signer nobody committed (every record unverifiable
//! at the end), a second `spark` on the box (a speed number measured under
//! contention), an anchor that was not HEAD (records naming a tree that was
//! never run). The evaluation is separated from the gathering so each refusal
//! is tested by flipping exactly one fact.

use std::path::Path;

use anyhow::{Context, Result};
use atlas_plugin::gate;

/// What was observed.
#[derive(Clone, Debug, Default)]
pub struct PreflightFacts {
    pub head: String,
    pub anchor: String,
    pub dirty_perf_paths: Vec<String>,
    pub signer: String,
    pub committed_signers: Vec<String>,
    pub atlas_home: String,
    pub atlas_home_writable: bool,
    pub other_spark_pids: Vec<u32>,
    /// `MemAvailable / MemTotal`; `None` when `/proc/meminfo` is unreadable.
    pub free_fraction: Option<f64>,
    /// The class's floor for it (`HARDWARE.toml` `[benchmarks.limits.memory]`)
    /// — the same number the child's self-start applies, so certify cannot
    /// pass a box the child refuses.
    pub min_free_fraction: f64,
    pub guard_ref: Option<String>,
    pub no_guard: bool,
    pub needs_confirmation_units: Vec<&'static str>,
    pub yes: bool,
    /// `--remote-only`: this box runs nothing, so its GPU and memory are
    /// not this campaign's business. Its signer still is — a record placed
    /// from a node is verified here against the same committed set.
    pub remote_only: bool,
}

/// One reason not to start. The text is the remedy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Finding(pub String);

impl std::fmt::Display for Finding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Every reason not to start, or none.
pub fn evaluate(f: &PreflightFacts) -> Vec<Finding> {
    let mut out = Vec::new();
    if f.head != f.anchor {
        out.push(Finding(format!(
            "HEAD is {} but the anchor is {}: check out the anchor, or drop --anchor",
            f.head, f.anchor
        )));
    }
    if !f.dirty_perf_paths.is_empty() {
        out.push(Finding(format!(
            "{} uncommitted PERF_PATHS file(s) ({}): a record measured from this tree \
             belongs to no commit — commit or stash them",
            f.dirty_perf_paths.len(),
            f.dirty_perf_paths.join(", ")
        )));
    }
    if !f.committed_signers.iter().any(|s| s == &f.signer) {
        out.push(Finding(format!(
            "signer {} ({}) is not committed in .github/record-signers/ — every record \
             this campaign writes would fail verification; commit the .pub (or point \
             ATLAS_HOME at an identity that is committed)",
            f.signer, f.atlas_home
        )));
    }
    if !f.atlas_home_writable {
        out.push(Finding(format!(
            "ATLAS_HOME {} is not writable: the run history and the signing identity \
             live there",
            f.atlas_home
        )));
    }
    if !f.other_spark_pids.is_empty() && !f.remote_only {
        out.push(Finding(format!(
            "another spark process is running (pid {}): a gate needs the GPU to itself, \
             and a speed number measured under contention is not a measurement",
            f.other_spark_pids
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    match f.free_fraction {
        _ if f.remote_only => {}
        Some(frac) if frac < f.min_free_fraction => out.push(Finding(format!(
            "only {:.0} % of host memory is available; a self-start needs {:.0} % — \
             something else is holding memory (check `nvidia-smi --query-compute-apps` \
             and `sudo docker ps`)",
            frac * 100.0,
            f.min_free_fraction * 100.0
        ))),
        Some(_) => {}
        None => out.push(Finding(
            "cannot read /proc/meminfo, so the free-memory rule cannot be applied".into(),
        )),
    }
    if f.guard_ref.is_none() && !f.no_guard {
        out.push(Finding(
            "HEAD has no upstream to guard against: pass --guard-ref REMOTE/BRANCH, or \
             --no-guard on a tree nobody else pushes to"
                .into(),
        ));
    }
    if !f.needs_confirmation_units.is_empty() && !f.yes {
        out.push(Finding(format!(
            "{} need(s) --yes: it executes model-authored shell in a sandbox, and that is \
             confirmed, not assumed",
            f.needs_confirmation_units.join(", ")
        )));
    }
    out
}

/// Gather the facts from this machine.
pub fn gather(
    root: &Path,
    anchor: &str,
    guard_ref: Option<String>,
    no_guard: bool,
    needs_confirmation_units: Vec<&'static str>,
    yes: bool,
    remote_only: bool,
    min_free_fraction: f64,
) -> Result<PreflightFacts> {
    let head = gate::git_sha(root)?;
    let dirty_perf_paths = gate::dirty_perf_paths(root)?;
    let store = atlas_plugin::ArtifactStore::discover().context("locating ATLAS_HOME")?;
    let atlas_home = store.root().display().to_string();
    let identity = gate::signing::load_or_create(store.root())
        .with_context(|| format!("loading the signing identity under {atlas_home}"))?;
    let committed_signers = gate::signing::committed_signers(root)?;
    let atlas_home_writable = {
        let probe = store.root().join(".certify-write-probe");
        let ok = std::fs::write(&probe, b"").is_ok();
        let _ = std::fs::remove_file(&probe);
        ok
    };
    Ok(PreflightFacts {
        head,
        anchor: anchor.to_string(),
        dirty_perf_paths,
        signer: identity.fingerprint().to_string(),
        committed_signers,
        atlas_home,
        atlas_home_writable,
        other_spark_pids: other_spark_pids(),
        free_fraction: free_fraction(),
        min_free_fraction,
        guard_ref,
        no_guard,
        needs_confirmation_units,
        yes,
        remote_only,
    })
}

/// Every `spark` process on this box other than ourselves — by exact process
/// name, never by command-line substring (a `-f` match finds the shell that
/// runs this very command).
pub fn other_spark_pids() -> Vec<u32> {
    let me = std::process::id();
    let Ok(out) = std::process::Command::new("pgrep")
        .args(["-x", "spark"])
        .stdin(std::process::Stdio::null())
        .output()
    else {
        return Vec::new();
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().parse::<u32>().ok())
        .filter(|pid| *pid != me)
        .collect()
}

/// `MemAvailable / MemTotal` from `/proc/meminfo`, the same reading the
/// self-start applies.
pub fn free_fraction() -> Option<f64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let field = |name: &str| -> Option<f64> {
        text.lines()
            .find(|l| l.starts_with(name))?
            .split_whitespace()
            .nth(1)?
            .parse::<f64>()
            .ok()
    };
    let total = field("MemTotal:")?;
    let avail = field("MemAvailable:")?;
    (total > 0.0).then(|| avail / total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clean() -> PreflightFacts {
        PreflightFacts {
            head: "abc".into(),
            anchor: "abc".into(),
            dirty_perf_paths: vec![],
            signer: "a27dbc8ed2fc2a31".into(),
            committed_signers: vec!["a27dbc8ed2fc2a31".into()],
            atlas_home: "/x".into(),
            atlas_home_writable: true,
            other_spark_pids: vec![],
            free_fraction: Some(0.95),
            min_free_fraction: 0.85,
            guard_ref: Some("avarok/main".into()),
            no_guard: false,
            needs_confirmation_units: vec![],
            yes: false,
            remote_only: false,
        }
    }

    /// `--remote-only` waives the two findings about THIS box's GPU and
    /// memory and nothing else: the signer rule still applies, because a
    /// record placed from a node is verified here.
    #[test]
    fn remote_only_waives_the_local_box_findings_only() {
        let mut f = clean();
        f.other_spark_pids = vec![4242];
        f.free_fraction = Some(0.10);
        assert_eq!(evaluate(&f).len(), 2);
        f.remote_only = true;
        assert!(evaluate(&f).is_empty(), "{:?}", evaluate(&f));
        // NEGATIVE CONTROL: an uncommitted signer is still refused.
        f.signer = "not-committed".into();
        let v = evaluate(&f);
        assert_eq!(v.len(), 1);
        assert!(v[0].0.contains("record-signers"));
    }

    #[test]
    fn a_clean_box_has_no_findings() {
        assert!(evaluate(&clean()).is_empty());
    }

    /// Each fact flips exactly one finding, and the finding names the remedy.
    #[test]
    fn each_fact_flips_one_finding() {
        type Flip = Box<dyn Fn(&mut PreflightFacts)>;
        let cases: Vec<(&str, Flip, &str)> = vec![
            (
                "anchor",
                Box::new(|f| f.anchor = "def".into()),
                "check out the anchor",
            ),
            (
                "dirty",
                Box::new(|f| f.dirty_perf_paths = vec!["crates/x.rs".into()]),
                "crates/x.rs",
            ),
            (
                "signer",
                Box::new(|f| f.committed_signers.clear()),
                "record-signers",
            ),
            (
                "home",
                Box::new(|f| f.atlas_home_writable = false),
                "not writable",
            ),
            (
                "spark",
                Box::new(|f| f.other_spark_pids = vec![4242]),
                "pid 4242",
            ),
            ("memory", Box::new(|f| f.free_fraction = Some(0.5)), "50 %"),
            (
                "meminfo",
                Box::new(|f| f.free_fraction = None),
                "/proc/meminfo",
            ),
            ("guard", Box::new(|f| f.guard_ref = None), "--guard-ref"),
            (
                "confirm",
                Box::new(|f| f.needs_confirmation_units = vec!["agentic-webserver"]),
                "--yes",
            ),
        ];
        for (name, flip, needle) in cases {
            let mut f = clean();
            flip(&mut f);
            let found = evaluate(&f);
            assert_eq!(found.len(), 1, "{name}: {found:?}");
            assert!(found[0].0.contains(needle), "{name}: {}", found[0]);
        }
    }

    #[test]
    fn no_guard_and_yes_clear_their_findings() {
        let mut f = clean();
        f.guard_ref = None;
        f.no_guard = true;
        f.needs_confirmation_units = vec!["agentic-webserver"];
        f.yes = true;
        assert!(evaluate(&f).is_empty());
    }

    /// The free-memory bar is the class's declared floor, inclusive.
    #[test]
    fn the_memory_threshold_is_the_declared_floor() {
        let mut f = clean();
        f.free_fraction = Some(f.min_free_fraction);
        assert!(evaluate(&f).is_empty());
        f.free_fraction = Some(f.min_free_fraction - 0.001);
        assert_eq!(evaluate(&f).len(), 1);
        // Another class, another floor: the same reading passes at 0.5.
        f.min_free_fraction = 0.5;
        assert!(evaluate(&f).is_empty());
    }
}
