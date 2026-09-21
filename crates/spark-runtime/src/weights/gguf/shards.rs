// SPDX-License-Identifier: AGPL-3.0-only

//! Split-GGUF shard-set resolution for the loader in the parent module.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

use super::sidecar;

/// One resolved split-GGUF file set, in `split.no` order.
///
/// A split GGUF is `N` files that share one logical tensor table. llama.cpp
/// writes the full metadata KV block into shard 0 only; every shard carries
/// `split.no`, `split.count` and `split.tensors.count`, and each holds its own
/// slice of the tensors.
pub struct GgufShardSet {
    /// Shard paths ordered by `split.no`. Length 1 for an unsplit file.
    pub paths: Vec<PathBuf>,
    /// `split.tensors.count` from shard 0: the total across the whole set.
    pub total_tensors: usize,
}

/// Resolve every shard of a split GGUF, given the first one.
///
/// WHY THIS IS NOT "glob the directory": a directory can legitimately hold more
/// than one model, an in-progress download, or a leftover shard from a
/// different quant. The split keys are the only authority on what belongs to
/// this file set, so the count comes from `split.count` and membership is
/// checked per shard rather than assumed from the filename.
///
/// Refuses, rather than proceeding, when:
///   * the passed file is not `split.no == 0` (the caller resolved the wrong one)
///   * any sibling named by the convention is missing or unparseable
///   * a sibling disagrees about `split.count` or `split.tensors.count`
///   * the `split.no` values are not exactly `0..count`
///   * the per-shard tensor counts do not sum to `split.tensors.count`
///
/// That last check is the one that matters in practice: a download that lost a
/// shard still leaves every other shard individually valid and checksum-clean,
/// so nothing below the tensor-count sum catches it.
pub fn find_gguf_shards(first: &Path) -> Result<GgufShardSet> {
    let (_f, _m, g) = sidecar::open_gguf(first)?;
    let count = match g.get_u64("split.count") {
        None | Some(0) | Some(1) => {
            return Ok(GgufShardSet {
                paths: vec![first.to_path_buf()],
                total_tensors: g.tensors.len(),
            });
        }
        Some(c) => c as usize,
    };
    let no = g.get_u64("split.no").unwrap_or(0) as usize;
    if no != 0 {
        bail!(
            "GGUF split: {} reports split.no={no}, expected shard 0. \
             Point the loader at the `-00001-of-*` shard.",
            first.display()
        );
    }
    let total_tensors = g
        .get_u64("split.tensors.count")
        .context("GGUF split: shard 0 has 'split.count' but no 'split.tensors.count'")?
        as usize;

    let name = first
        .file_name()
        .and_then(|n| n.to_str())
        .context("GGUF split: shard path has no filename")?;
    // llama.cpp convention: `<stem>-00001-of-00007.gguf`, one-based on disk,
    // zero-based in `split.no`.
    let marker = format!("-00001-of-{count:05}.gguf");
    let stem = name.strip_suffix(&marker).with_context(|| {
        format!(
            "GGUF split: '{name}' declares split.count={count} but does not end in '{marker}'; \
             cannot derive sibling shard names"
        )
    })?;
    let dir = first.parent().unwrap_or_else(|| Path::new("."));

    let mut paths = vec![first.to_path_buf()];
    let mut seen = vec![0usize; count];
    seen[0] = g.tensors.len();
    for (i, slot) in seen.iter_mut().enumerate().skip(1) {
        let p = dir.join(format!("{stem}-{:05}-of-{count:05}.gguf", i + 1));
        if !p.exists() {
            bail!(
                "GGUF split: shard {} of {count} missing: {}",
                i + 1,
                p.display()
            );
        }
        let (_f2, _m2, g2) = sidecar::open_gguf(&p)
            .with_context(|| format!("GGUF split: failed to open shard {}", p.display()))?;
        let c2 = g2.get_u64("split.count").unwrap_or(0) as usize;
        if c2 != count {
            bail!(
                "GGUF split: {} reports split.count={c2}, shard 0 says {count}",
                p.display()
            );
        }
        let n2 = g2.get_u64("split.no").unwrap_or(usize::MAX as u64) as usize;
        if n2 != i {
            bail!(
                "GGUF split: {} reports split.no={n2}, expected {i} from its filename",
                p.display()
            );
        }
        if let Some(t2) = g2.get_u64("split.tensors.count")
            && t2 as usize != total_tensors
        {
            bail!(
                "GGUF split: {} reports split.tensors.count={t2}, shard 0 says {total_tensors}",
                p.display()
            );
        }
        *slot = g2.tensors.len();
        paths.push(p);
    }

    let summed: usize = seen.iter().sum();
    if summed != total_tensors {
        bail!(
            "GGUF split: shards hold {summed} tensors but 'split.tensors.count' is \
             {total_tensors} — the file set is incomplete or mismatched (per-shard: {seen:?})"
        );
    }
    Ok(GgufShardSet {
        paths,
        total_tensors,
    })
}
