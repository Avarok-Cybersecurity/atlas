// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel-target resolution: which compiled target serves a checkpoint.
//!
//! Split out of `lib.rs` so the selection RULES are pure functions over
//! plain declarations — unit-testable on a GPU-free host where
//! `ATLAS_SKIP_BUILD=1` leaves `all_ptx_sets()` empty.
//!
//! ## Why `(model_type, hidden_size)` alone is not enough
//!
//! Qwen3.8-27B is architecturally identical to Qwen3.6-27B: same
//! `model_type` (`qwen3_5`), same `hidden_size` (5120), same every numeric
//! config field — the checkpoints differ only in weights. Two kernel
//! targets therefore declare the SAME exact `(qwen3_5, 5120)` pair, and the
//! historical resolver (`.find()` over build-order-sorted targets) would
//! have silently picked whichever sorted first. A silent wrong pick
//! mis-serves the MLPerf-edge flagship's sampling presets and behavior
//! flags, so ambiguity must never resolve by iteration order.
//!
//! ## The rules
//!
//! 1. Exact `(model_type, Some(hidden_size))` declarations beat wildcard
//!    `(model_type, None)` declarations (unchanged).
//! 2. Within the winning tier, if exactly ONE target (by name) matches,
//!    it is selected (unchanged — covers every non-colliding model).
//! 3. If SEVERAL differently-named targets match, the tie is broken by
//!    the checkpoint reference: each colliding target declares explicit
//!    `match_names` needles in its MODEL.toml (`[model] match_names`),
//!    and a candidate survives when any needle is a case-insensitive
//!    substring of any reference (HF id, `--model-name`, resolved model
//!    dir). Exactly one survivor wins — and the selection is explicit,
//!    because the needles are declared per target, not inferred.
//! 4. Anything else — zero or multiple survivors — is
//!    [`TargetResolveError::Ambiguous`]. It never falls through to the
//!    wildcard tier and never picks by order.
//! 5. `--kernel-target <name>` pins resolution to a named target,
//!    bypassing the tie-break — but the pinned target must still declare
//!    compatibility with the checkpoint's `(model_type, hidden_size)`,
//!    otherwise [`TargetResolveError::PinIncompatible`]. This is the
//!    escape hatch for checkpoints whose references carry no identity
//!    (e.g. `--model-from-path /model`).
//!
//! `build.rs` enforces at compile time that every set of differently-named
//! targets sharing a `(model_type, hidden_size)` declaration carries
//! explicit `match_names`, so rule 3 can never reach a colliding target
//! with nothing declared.

use crate::{ModelTypeMatch, TargetPtxSet};

/// The resolution-relevant slice of one compiled target. Borrowed views so
/// tests can drive the rules with synthetic declarations and production
/// wraps `TargetPtxSet`s without copying module blobs.
pub struct ResolveCandidate<'a> {
    /// Kernel-target directory name (`KernelTarget::model`), e.g.
    /// `"qwen3.8-27b"`. Multi-quant builds repeat a name with different
    /// quants; same-name candidates are never ambiguous with each other
    /// (the downstream quant-compat gate arbitrates quant).
    pub name: &'a str,
    /// `[[model_types]]` declarations from MODEL.toml.
    pub type_matches: &'a [ModelTypeMatch],
    /// `[model] match_names` needles from MODEL.toml. Empty when the
    /// target never collides (build.rs enforces presence on collision).
    pub match_names: &'a [&'a str],
}

/// Why resolution could not choose a target. Every variant is a hard error
/// at the call site — resolution must never fall back to iteration order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetResolveError {
    /// More than one differently-named target claims the checkpoint's
    /// `(model_type, hidden_size)` and the reference tie-break did not
    /// leave exactly one.
    Ambiguous {
        model_type: String,
        hidden_size: usize,
        /// `"exact"` or `"wildcard"` — which declaration tier collided.
        tier: &'static str,
        /// Distinct target names in the colliding tier, with their needles.
        candidates: Vec<(String, Vec<String>)>,
        /// The subset whose needles matched a reference (empty = none did).
        matched: Vec<String>,
        /// The checkpoint references that were searched.
        model_refs: Vec<String>,
    },
    /// `--kernel-target` named a target this binary did not compile.
    PinNotFound { pin: String, available: Vec<String> },
    /// `--kernel-target` named a compiled target that does not declare
    /// support for the checkpoint's `(model_type, hidden_size)` — serving
    /// would run another architecture's kernels.
    PinIncompatible {
        pin: String,
        model_type: String,
        hidden_size: usize,
        declared: Vec<String>,
    },
}

impl std::fmt::Display for TargetResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ambiguous {
                model_type,
                hidden_size,
                tier,
                candidates,
                matched,
                model_refs,
            } => {
                let cands = candidates
                    .iter()
                    .map(|(n, needles)| format!("{n} (match_names: {needles:?})"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let outcome = if matched.is_empty() {
                    "no checkpoint reference names any of them".to_string()
                } else {
                    format!("the references match several of them: {matched:?}")
                };
                write!(
                    f,
                    "AMBIGUOUS kernel target: {} compiled targets declare {tier} support for \
                     (model_type '{model_type}', hidden_size {hidden_size}) — [{cands}] — and \
                     {outcome} (references searched: {model_refs:?}). Refusing to pick by build \
                     order. Fix: serve with a model id/path that contains exactly one target's \
                     match_names needle, pin explicitly with --kernel-target <name>, or build \
                     single-target with ATLAS_TARGET_MODEL=<name>.",
                    candidates.len(),
                )
            }
            Self::PinNotFound { pin, available } => write!(
                f,
                "--kernel-target '{pin}' does not name a compiled kernel target \
                 (available: {available:?})"
            ),
            Self::PinIncompatible {
                pin,
                model_type,
                hidden_size,
                declared,
            } => write!(
                f,
                "--kernel-target '{pin}' is compiled but declares no support for this \
                 checkpoint's (model_type '{model_type}', hidden_size {hidden_size}) — it \
                 declares {declared:?}. Serving another architecture's kernels would be \
                 garbage; refusing."
            ),
        }
    }
}

impl std::error::Error for TargetResolveError {}

/// Case-insensitive substring: does any declared needle appear in any
/// checkpoint reference? References are HF ids, `--model-name` values, or
/// resolved model directories — all of which normally embed the model name.
fn needles_hit(match_names: &[&str], refs_lower: &[String]) -> bool {
    match_names.iter().any(|needle| {
        let n = needle.to_lowercase();
        !n.is_empty() && refs_lower.iter().any(|r| r.contains(&n))
    })
}

/// Distinct candidate names in first-seen order for a set of indices.
fn distinct_names<'a>(candidates: &[ResolveCandidate<'a>], idxs: &[usize]) -> Vec<&'a str> {
    let mut names: Vec<&str> = Vec::new();
    for &i in idxs {
        if !names.contains(&candidates[i].name) {
            names.push(candidates[i].name);
        }
    }
    names
}

/// Resolve which candidate serves `(model_type, hidden_size)` for a
/// checkpoint identified by `model_refs`. Returns the index of the winning
/// candidate, `Ok(None)` when nothing declares the pair at all, and
/// [`TargetResolveError::Ambiguous`] when a collision cannot be broken to
/// exactly one target name.
pub fn resolve_target(
    candidates: &[ResolveCandidate<'_>],
    model_type: &str,
    hidden_size: usize,
    model_refs: &[&str],
) -> Result<Option<usize>, TargetResolveError> {
    let refs_lower: Vec<String> = model_refs.iter().map(|r| r.to_lowercase()).collect();

    let tiers: [(&'static str, Option<usize>); 2] =
        [("exact", Some(hidden_size)), ("wildcard", None)];
    for (tier, want_hidden) in tiers {
        let idxs: Vec<usize> = candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                c.type_matches
                    .iter()
                    .any(|m| m.model_type == model_type && m.hidden_size == want_hidden)
            })
            .map(|(i, _)| i)
            .collect();
        if idxs.is_empty() {
            continue;
        }
        let names = distinct_names(candidates, &idxs);
        if names.len() == 1 {
            // Single target name (possibly several quant variants — the
            // quant-compat gate downstream arbitrates those, as before).
            return Ok(Some(idxs[0]));
        }
        // Collision: break the tie on declared match_names vs references.
        let matched: Vec<&str> = names
            .iter()
            .copied()
            .filter(|n| {
                idxs.iter().any(|&i| {
                    candidates[i].name == *n && needles_hit(candidates[i].match_names, &refs_lower)
                })
            })
            .collect();
        if let [winner] = matched.as_slice() {
            let idx = idxs
                .iter()
                .copied()
                .find(|&i| candidates[i].name == *winner)
                .expect("winner name came from idxs");
            return Ok(Some(idx));
        }
        // Zero or several survivors: hard error. Deliberately does NOT
        // fall through to the wildcard tier — a checkpoint that exact-
        // matches colliding targets must be disambiguated, not quietly
        // downgraded to a wildcard target.
        return Err(TargetResolveError::Ambiguous {
            model_type: model_type.to_string(),
            hidden_size,
            tier,
            candidates: names
                .iter()
                .map(|n| {
                    let needles = idxs
                        .iter()
                        .filter(|&&i| candidates[i].name == *n)
                        .flat_map(|&i| candidates[i].match_names.iter().map(|s| s.to_string()))
                        .collect();
                    (n.to_string(), needles)
                })
                .collect(),
            matched: matched.iter().map(|n| n.to_string()).collect(),
            model_refs: model_refs.iter().map(|r| r.to_string()).collect(),
        });
    }
    Ok(None)
}

/// Resolve a `--kernel-target` pin: the named target wins unconditionally
/// over the tie-break, but must exist and must declare the checkpoint's
/// `(model_type, hidden_size)` (exact or wildcard).
pub fn resolve_pinned(
    candidates: &[ResolveCandidate<'_>],
    pin: &str,
    model_type: &str,
    hidden_size: usize,
) -> Result<usize, TargetResolveError> {
    let pinned: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| c.name.eq_ignore_ascii_case(pin))
        .map(|(i, _)| i)
        .collect();
    if pinned.is_empty() {
        let all: Vec<usize> = (0..candidates.len()).collect();
        return Err(TargetResolveError::PinNotFound {
            pin: pin.to_string(),
            available: distinct_names(candidates, &all)
                .into_iter()
                .map(String::from)
                .collect(),
        });
    }
    let compatible = pinned.iter().copied().find(|&i| {
        candidates[i].type_matches.iter().any(|m| {
            m.model_type == model_type
                && (m.hidden_size.is_none() || m.hidden_size == Some(hidden_size))
        })
    });
    compatible.ok_or_else(|| TargetResolveError::PinIncompatible {
        pin: pin.to_string(),
        model_type: model_type.to_string(),
        hidden_size,
        declared: pinned
            .iter()
            .flat_map(|&i| candidates[i].type_matches.iter())
            .map(|m| format!("({}, {:?})", m.model_type, m.hidden_size))
            .collect(),
    })
}

/// Find the PTX module set matching a checkpoint.
///
/// Matching rules (full statement + rationale in this module):
/// 1. Exact match on `(model_type, Some(hidden_size))` beats wildcard
///    `(model_type, None)`.
/// 2. When several differently-named targets declare the same pair (the
///    configs of e.g. Qwen3.6-27B and Qwen3.8-27B are bit-identical), the
///    tie is broken by matching each target's declared `match_names`
///    needles against `model_refs` (HF id, `--model-name`, resolved model
///    dir) — and a tie that does not break to exactly one target is
///    `Err(TargetResolveError::Ambiguous)`, never a build-order pick.
/// 3. `pinned_target` (`--kernel-target`) bypasses the tie-break but must
///    name a compiled target that declares the `(model_type, hidden_size)`.
/// 4. `Ok(None)` if no compiled target declares the pair at all.
pub fn ptx_for_config(
    model_type: &str,
    hidden_size: usize,
    model_refs: &[&str],
    pinned_target: Option<&str>,
) -> Result<Option<TargetPtxSet>, TargetResolveError> {
    let targets = crate::all_ptx_sets();
    let candidates: Vec<ResolveCandidate<'_>> = targets
        .iter()
        .map(|t| ResolveCandidate {
            name: t.target.model,
            type_matches: &t.model_type_matches,
            match_names: t.match_names,
        })
        .collect();
    let idx = match pinned_target {
        Some(pin) => Some(resolve_pinned(&candidates, pin, model_type, hidden_size)?),
        None => resolve_target(&candidates, model_type, hidden_size, model_refs)?,
    };
    drop(candidates);
    Ok(idx.and_then(|i| targets.into_iter().nth(i)))
}

/// The compiled target with exactly this `(model, quant)` identity.
///
/// For consumers that already KNOW the resolved target (the dashboard's
/// kernel table re-reads the target `serve` selected and published) —
/// an exact lookup cannot re-introduce the ambiguity `ptx_for_config`
/// just resolved.
pub fn ptx_for_exact_target(model: &str, quant: &str) -> Option<TargetPtxSet> {
    crate::all_ptx_sets()
        .into_iter()
        .find(|t| t.target.model == model && t.target.quant == quant)
}

#[cfg(test)]
#[path = "resolve_tests.rs"]
mod tests;
