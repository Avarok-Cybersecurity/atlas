// SPDX-License-Identifier: AGPL-3.0-only

//! Which files to fetch for a model, and how many bytes that is.
//!
//! Pure: no network, no filesystem. The whole point is that "why is this
//! download three times the size of the model" is decided in one testable
//! function rather than discovered on a metered connection.

/// One file as the Hub describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteFile {
    pub name: String,
    /// Bytes, when the listing declared them. `None` is common: the plain
    /// model endpoint omits sizes, so they come from the tree endpoint.
    pub size: Option<u64>,
}

/// Metadata the loader looks for by exact name.
///
/// Deliberately a fixed list rather than "every .json": a repo can carry
/// evaluation results, quantisation manifests and training configs that the
/// loader never opens.
const METADATA: &[&str] = &[
    "config.json",
    "params.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
];

/// Weight formats Atlas cannot load, and directories that duplicate the model.
///
/// `original/` in particular is why an unfiltered mirror costs double: Llama
/// and Gemma repos ship the reference checkpoint there alongside the
/// safetensors the loader actually reads.
fn is_excluded(name: &str) -> bool {
    const SKIP_DIRS: &[&str] = &["original/", "onnx/", "openvino/", "coreml/", "tflite/"];
    const SKIP_EXT: &[&str] = &[
        ".bin", ".pth", ".pt", ".msgpack", ".h5", ".onnx", ".gguf", ".tflite",
    ];
    SKIP_DIRS.iter().any(|d| name.starts_with(d)) || SKIP_EXT.iter().any(|e| name.ends_with(e))
}

fn is_weight(name: &str) -> bool {
    name.ends_with(".safetensors") || name.ends_with(".safetensors.index.json")
}

/// Does this name stay inside the snapshot directory it is joined onto?
///
/// The name is the Hub's `rfilename`, so it is chosen by whoever published
/// the repo, and `select`'s output is joined straight onto the cache path by
/// the downloader. `Path::join` resolves `..` against the parent and treats
/// an absolute component as a *replacement* for everything to its left, so
/// an unvalidated name is an arbitrary-file-write primitive on the machine
/// doing the pull — enough to overwrite another model's shards.
///
/// Weights legitimately sit in subdirectories, so `/` itself has to be
/// allowed; it is the traversal and the rooting that are rejected.
fn is_contained(name: &str) -> bool {
    // `\` is a separator on Windows and no Hub repo needs it; a `C:` prefix
    // is absolute on the platforms that parse it.
    if name.is_empty() || name.starts_with('/') || name.contains('\\') {
        return false;
    }
    let b = name.as_bytes();
    if b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':' {
        return false;
    }
    // An empty component is `//`, which some joiners re-root on.
    name.split('/')
        .all(|c| !c.is_empty() && c != "." && c != "..")
}

/// Is this file worth downloading?
pub fn wanted(name: &str) -> bool {
    if !is_contained(name) || is_excluded(name) {
        return false;
    }
    // Only top-level metadata: a `subfolder/config.json` belongs to a
    // component the loader resolves separately, if at all.
    is_weight(name) || (!name.contains('/') && METADATA.contains(&name))
}

/// The files to fetch, in the order to fetch them.
///
/// Small files first, then weights ascending. Two reasons: a wrong-model abort
/// costs under a second instead of a shard, and `config.json` — which decides
/// whether the model is loadable at all — lands before gigabytes do.
pub fn select(files: &[RemoteFile]) -> Vec<RemoteFile> {
    let mut out: Vec<RemoteFile> = files.iter().filter(|f| wanted(&f.name)).cloned().collect();
    out.sort_by(|a, b| {
        let key = |f: &RemoteFile| (is_weight(&f.name), f.size.unwrap_or(0), f.name.clone());
        key(a).cmp(&key(b))
    });
    out
}

/// Does this plan contain anything Atlas could actually load?
///
/// A repo publishing only GGUF is a real and common case — the whole plan
/// filters away and the download would "succeed" having fetched a tokenizer.
pub fn has_weights(plan: &[RemoteFile]) -> bool {
    plan.iter().any(|f| f.name.ends_with(".safetensors"))
}

/// Total bytes of a plan, counting only files whose size is known.
pub fn total_bytes(plan: &[RemoteFile]) -> u64 {
    plan.iter().filter_map(|f| f.size).sum()
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
