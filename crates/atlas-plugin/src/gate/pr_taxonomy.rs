// SPDX-License-Identifier: AGPL-3.0-only

//! The PR **intent** taxonomy: a tree the classifier descends one level at a
//! time, and the benchmarks each path implies.
//!
//! # Two taxonomies, and they are not the same thing
//!
//! [`crate::gate::taxon`] walks `kernels/<hw>/<model>/<quant>/`. It is derived from
//! PATHS, needs no model, and is the floor for invalidation. This module is
//! about what a change is FOR — `performance/decode`, `correctness/kv-cache` —
//! which cannot be read off a directory.
//!
//! # ★ `benches` may only ADD. It can never remove.
//!
//! The required set is `path_derived ∪ intent_derived`. The path-derived half
//! (`PERF_PATHS` + the closure hash) stands on its own; nothing here can shrink
//! it. So a MISCLASSIFICATION COSTS GPU MINUTES, NEVER A MISSED REGRESSION.
//!
//! That is the only footing on which a language model belongs near a merge
//! gate. Invert it and the classifier becomes a way to skip tests by writing a
//! misleading PR title — the diff would not even have to lie, only the prose.
//!
//! What it buys is the direction paths cannot see. A scheduler change under
//! `crates/spark-server/` touches no `kernels/`, so every target's closure hash
//! is unchanged and the kernel rungs excuse all of them — yet it can move
//! decode wall badly. `performance/scheduling` pulls in the agentic leg that
//! would otherwise not run.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context, Result, bail};

/// Keys in the JSON that are metadata rather than child nodes.
const RESERVED: [&str; 2] = ["_doc", "_benches"];

/// The parsed tree. Children are ordered, which keeps the prompt the classifier
/// sees stable across runs — an unordered set would reshuffle the option list
/// and make two runs on one PR harder to compare.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub name: String,
    pub benches: Vec<String>,
    pub children: Vec<Node>,
}

impl Node {
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// Load `.github/pr-taxonomy.json` from a repo root.
pub fn load(root: &Path) -> Result<Vec<Node>> {
    let path = root.join(".github/pr-taxonomy.json");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let roots = parse_children(&json)?;
    validate(&roots)?;
    Ok(roots)
}

fn parse_children(value: &serde_json::Value) -> Result<Vec<Node>> {
    let Some(obj) = value.as_object() else {
        bail!("every taxonomy node must be a JSON object");
    };
    let mut out = Vec::new();
    for (key, child) in obj {
        if RESERVED.contains(&key.as_str()) {
            continue;
        }
        // ★ STRICT. The first version used `.as_array().unwrap_or_default()`
        // and `filter_map(as_str)`, so `_benches: "bfcl-subset"` (a bare
        // string) and `[["x"]]` both parsed as EMPTY here while the ci.yml jq
        // read them fine. Two implementations of one function that disagreed —
        // and the Rust half failed in the REMOVING direction, which is exactly
        // what this module's safety property forbids. A typo must be loud.
        let benches = match child.get("_benches") {
            None => Vec::new(),
            Some(serde_json::Value::Array(items)) => {
                let mut v = Vec::with_capacity(items.len());
                for item in items {
                    match item.as_str() {
                        Some(s) => v.push(s.to_string()),
                        None => bail!(
                            "{key}: _benches contains a non-string entry ({item}). \
                             A silently-dropped entry removes a benchmark."
                        ),
                    }
                }
                v
            }
            Some(other) => bail!(
                "{key}: _benches must be an ARRAY of benchmark ids, got {other}. \
                 A bare string parses as empty here while jq reads it, so the two \
                 halves would disagree — in the removing direction."
            ),
        };
        out.push(Node {
            name: key.clone(),
            benches,
            children: parse_children(child)?,
        });
    }
    Ok(out)
}

/// The shape rules the JSON's own `_doc` promises. A rule nothing enforces is
/// a comment.
fn validate(roots: &[Node]) -> Result<()> {
    if roots.len() < 2 {
        bail!("the taxonomy needs at least two roots; one root is not a choice");
    }
    let known: BTreeSet<&str> = super::coverage::REQUIRED.iter().map(|g| g.id).collect();
    fn walk(nodes: &[Node], trail: &str, known: &BTreeSet<&str>) -> Result<()> {
        for n in nodes {
            let here = if trail.is_empty() {
                n.name.clone()
            } else {
                format!("{trail}/{}", n.name)
            };
            if !n
                .name
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
                || n.name.is_empty()
            {
                bail!("{here}: keys must be lowercase kebab-case so a path is a safe label");
            }
            for b in &n.benches {
                if !known.contains(b.as_str()) {
                    bail!(
                        "{here}: _benches names {b:?}, which is not a required benchmark. \
                         A path that selects a benchmark nobody runs is a silent no-op."
                    );
                }
            }
            // A single child is not a choice: asking a model to pick from a set
            // of one wastes a call and manufactures confidence. `descend`
            // auto-follows those, so the tree must not contain them either.
            if n.children.len() == 1 {
                bail!(
                    "{here} has exactly one child ({}). Either give it a sibling or \
                     make {here} a leaf.",
                    n.children[0].name
                );
            }
            walk(&n.children, &here, known)?;
        }
        Ok(())
    }
    walk(roots, "", &known)
}

/// The options a classifier should be offered at `path`, or `None` when `path`
/// is a leaf and the descent is over.
///
/// Returns `Some` only when there is a real choice — a lone child is followed
/// automatically by [`resolve`] rather than put to a model.
pub fn options_at(roots: &[Node], path: &[String]) -> Option<Vec<String>> {
    let node = walk_to(roots, path)?;
    let kids: Vec<String> = node.iter().map(|n| n.name.clone()).collect();
    (kids.len() > 1).then_some(kids)
}

fn walk_to<'a>(roots: &'a [Node], path: &[String]) -> Option<&'a [Node]> {
    let mut level = roots;
    for step in path {
        level = &level.iter().find(|n| &n.name == step)?.children;
    }
    Some(level)
}

/// Every benchmark a path implies: the UNION of `_benches` along it.
///
/// Union, not "the leaf's list": `correctness` implies BFCL for every child,
/// and `correctness/kv-cache` adds both TTFT legs on top. A leaf-only rule
/// would silently drop the ancestor's requirement the moment someone added a
/// leaf and forgot to repeat it.
///
/// An unknown path segment yields what was matched so far rather than an error:
/// this feeds a gate, and a stale label must degrade to *fewer extra* benches,
/// never to a crash that takes the view down.
pub fn benches_for(roots: &[Node], path: &[String]) -> BTreeSet<String> {
    benches_for_matched(roots, path).0
}

/// [`benches_for`], plus HOW MANY leading segments actually matched.
///
/// The degrade-on-unknown-segment rule above is right for a gate and invisible
/// to a human: `performance/decodes` and `performance` return the same set, and
/// nothing says one of them was a typo. Reporting the matched depth lets a
/// caller warn ("matched 1 of 2 segments") without walking the tree a second
/// time — and a second walk is exactly how this module acquired its last bug,
/// when a jq reimplementation of it drifted out of agreement in the *removing*
/// direction.
pub fn benches_for_matched(roots: &[Node], path: &[String]) -> (BTreeSet<String>, usize) {
    let mut out = BTreeSet::new();
    let mut level = roots;
    let mut matched = 0usize;
    for step in path {
        let Some(node) = level.iter().find(|n| &n.name == step) else {
            break;
        };
        out.extend(node.benches.iter().cloned());
        level = &node.children;
        matched += 1;
    }
    (out, matched)
}

/// Follow every forced (single-child) step from `path` downward.
///
/// The tree currently forbids single-child nodes, so this is a no-op today —
/// it exists so that relaxing that rule later cannot silently start asking a
/// model to choose from a set of one.
pub fn resolve(roots: &[Node], path: &[String]) -> Vec<String> {
    let mut out = path.to_vec();
    loop {
        let Some(level) = walk_to(roots, &out) else {
            return out;
        };
        if level.len() == 1 {
            out.push(level[0].name.clone());
        } else {
            return out;
        }
    }
}

/// Is `path` a complete descent — i.e. does it end at a leaf?
pub fn is_complete(roots: &[Node], path: &[String]) -> bool {
    walk_to(roots, path).is_some_and(<[Node]>::is_empty)
}

#[cfg(test)]
#[path = "pr_taxonomy_tests.rs"]
mod pr_taxonomy_tests;
