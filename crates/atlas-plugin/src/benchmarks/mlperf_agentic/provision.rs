// SPDX-License-Identifier: AGPL-3.0-only

//! `~/.atlas/artifacts/mlperf-agentic` — the official trajectory file, which
//! DOES NOT EXIST YET anywhere in the world this code can reach.
//!
//! Unlike every other provisioner in this tree, this one cannot download its
//! artifact: mlcommons/endpoints@7935df4 ships an empty `datasets/` directory
//! and its README says the official dataset "can be downloaded from MLCommons
//! storage (link TBD)". There is no URL, no license, no auth story. So this
//! module does the two things that ARE possible today:
//!
//! * fail loudly and specifically when the file is absent — a leg that
//!   reported 0.0 because it had no data would be the vacuous-PASS class the
//!   SSM gate work already had to fix once, and
//! * when an operator has placed the published file here, verify it by
//!   full-file SHA256 before anything reads it — closing, for this leg, the
//!   gap where BFCL computes a dataset digest during provisioning and then
//!   drops it.

use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::artifacts::ArtifactStore;
use crate::plugin::PluginHandle;

pub const PLUGIN_ID: &str = "mlperf-agentic";

/// Where the upstream repo says the dataset will eventually come from,
/// verbatim, so the error below stays quotable when someone greps for it.
pub const UPSTREAM_DATASET_STATUS: &str = "mlcommons/endpoints@7935df4 examples/10_Agentic_Inference/README.md: \"The official \
     MLPerf dataset can be downloaded from MLCommons storage (link TBD)\"";

#[derive(Clone, Debug)]
pub struct Artifacts {
    pub dir: PathBuf,
    pub dataset: PathBuf,
    /// Full-file SHA256 of `dataset`, hex. Recorded into the run summary and
    /// the gate record's dataset fingerprint.
    pub file_sha256: String,
}

/// Locate and verify the dataset. `expected_sha256` is the pin: empty means
/// "record only" — the honest state while no official file exists to pin —
/// and non-empty means the file must hash to exactly that.
pub fn ensure(
    store: &ArtifactStore,
    handle: &PluginHandle,
    expected_sha256: &str,
) -> Result<Artifacts> {
    let dir = store.plugin_dir(PLUGIN_ID)?;
    let dataset = dir.join("dataset.jsonl");
    if !dataset.is_file() {
        // The whole failure mode this leg must not have is "ran anyway":
        // no proxy dataset, no reconstruction from the upstream DeepSWE /
        // Workato sources, no empty-denominator zero score. A score from a
        // different draw is not comparable to anything, and calling one
        // "MLPerf" would be exactly the failure this repo's BENCH notes exist
        // to prevent.
        bail!(
            "the MLPerf Agentic Inference dataset is not provisioned: {} does not exist.\n\
             The official dataset (613 trajectories: 500 Workato workflow + 113 DeepSWE \
             coding, 30,335 client turns) is NOT yet published — {}. There is no download \
             URL, license, or auth story to automate, and this leg deliberately refuses to \
             substitute a proxy or reconstructed dataset: a score from a different draw is \
             comparable to nothing, however official it looks.\n\
             When MLCommons publishes the file: place it at the path above, record its \
             SHA256 as this leg's expected_sha256 parameter (and in BENCH.toml), and run \
             the calibration protocol in the BENCH.toml note before trusting any number.",
            dataset.display(),
            UPSTREAM_DATASET_STATUS,
        );
    }

    let (file_sha256, bytes) = sha256_file(&dataset)?;
    if !expected_sha256.is_empty() && !expected_sha256.eq_ignore_ascii_case(&file_sha256) {
        bail!(
            "{} does not match its pin: sha256 {file_sha256} ({bytes} bytes), expected \
             {expected_sha256}. A replay of the wrong file scores against the wrong ground \
             truth; refusing to run.",
            dataset.display()
        );
    }
    handle.info(format!(
        "dataset {} — {bytes} bytes, sha256 {file_sha256}{}",
        dataset.display(),
        if expected_sha256.is_empty() {
            " (UNPINNED: no expected_sha256 — record-only until the official file ships)"
        } else {
            " (pin verified)"
        }
    ));
    std::fs::write(
        dir.join("dataset_summary.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "file_sha256": file_sha256,
            "bytes": bytes,
        }))? + "\n",
    )?;
    Ok(Artifacts {
        dir,
        dataset,
        file_sha256,
    })
}

/// Hex SHA256 and byte count of a file, streamed — the official file is a
/// whole replay corpus and has no business living in memory twice.
fn sha256_file(path: &Path) -> Result<(String, u64)> {
    use sha2::{Digest, Sha256};
    let mut file =
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("reading {}", path.display()))?;
        if n == 0 {
            break;
        }
        total += n as u64;
        digest.update(&buf[..n]);
    }
    Ok((format!("{:x}", digest.finalize()), total))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("atlas-mlperf-prov-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn sha256_matches_a_known_vector() {
        let dir = tmp("sha");
        let p = dir.join("f");
        std::fs::write(&p, b"abc").unwrap();
        let (sha, n) = sha256_file(&p).unwrap();
        assert_eq!(n, 3);
        assert_eq!(
            sha,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    // The absent-dataset failure itself (message names the path, the TBD
    // upstream status, and the no-proxy rule) is pinned in mod.rs's tests,
    // where a PluginHandle exists to call ensure() with.
}
