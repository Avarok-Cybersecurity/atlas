// SPDX-License-Identifier: AGPL-3.0-only

//! Native EXL3 dense (GDN + attention) projections — the
//! `ATLAS_EXL3_NATIVE_DENSE=1` extension of the materialize pass (see
//! `exl3_materialize.rs` for the pass itself, `exl3_materialize_moe.rs` for
//! the routed-expert sibling this mirrors).
//!
//! This file owns the dense-family predicates and the per-layer ATOMIC
//! keep-set computation:
//!
//!  * [`exl3_native_dense_families`] / [`check_exl3_native_dense_gates`] —
//!    the env gates. `ATLAS_EXL3_NATIVE_DENSE=1` (requires
//!    `ATLAS_EXL3_NATIVE=1`, hard error otherwise) enables BOTH families;
//!    `ATLAS_EXL3_NATIVE_GDN=0` / `ATLAS_EXL3_NATIVE_ATTN=0` opt one family
//!    back out for A/B. Either sub-gate set without the DENSE gate is a hard
//!    error (fail-loud house style, never a silent ignore).
//!  * [`exl3_native_serves_dense`] — the prefix predicate. Two families:
//!    GDN = `.linear_attn.{in_proj_qkv,in_proj_z,out_proj}`, attention =
//!    `.self_attn.{q,k,v,o}_proj`. `mtp.*` is EXCLUDED (the MTP block is not
//!    wired on this branch and its loaders read BF16), as are
//!    `indexer.index_qk_proj` (K=2 QSA projection, consumer not routed),
//!    `in_proj_a`/`in_proj_b` (ship BF16, interleaved at load) and the
//!    shared expert.
//!  * [`dense_keep_set`] — per-(layer, family) ATOMIC keep-or-materialize:
//!    every ROUTED projection of the family must be present as EXL3 AND
//!    inside [`super::exl3_native_supported`] (K in {2,4} — the GEMV
//!    envelope that keeps small-row launches off the shared-locks GEMM — cb
//!    MCG/MUL1, dims %128), or the WHOLE routed set of that layer
//!    materializes to BF16 exactly as today. No half-native layers: the
//!    loader arms decide "kept" from the projections' `.trellis` still being
//!    in the store, so a partially-kept set would be read half from trellis,
//!    half from a `.weight` that no longer exists.
//!
//! ROUTED vs ALL leaves ([`Exl3DenseFamily::leaves`] vs `all_leaves`): a
//! projection joins the keep-set only in lockstep with its layer-site
//! dispatch arm — a kept-packed tensor with no arm has no `.weight` for the
//! BF16 consumers and fails at the first request. Today the routed set is
//! the WHOLE GDN family (`out_proj`: step 1; `in_proj_qkv`/`in_proj_z` as a
//! shared-A pair into the fused `[Q|K|V|Z]` arena row: step 2 of the design
//! map) and the WHOLE attention family (`q/k/v/o_proj`: step 3 — per-site
//! drop-in arms, q_proj writing the raw `[Q|gate]` interleave ahead of the
//! existing deinterleave). `all_leaves == leaves` for both; the split stays
//! so a future family can land arm-by-arm.

use std::collections::{BTreeMap, HashSet};

use anyhow::{Result, bail};
use spark_runtime::weights::WeightStore;
use spark_runtime::weights::exl3::{Exl3Weight, is_exl3_linear};

/// Which dense families the gates admit. `OFF` (both false) is the default
/// and keeps every dense linear on the materialize path byte-for-byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Exl3DenseFamilies {
    pub gdn: bool,
    pub attn: bool,
}

impl Exl3DenseFamilies {
    pub const OFF: Self = Self {
        gdn: false,
        attn: false,
    };
    pub const ALL: Self = Self {
        gdn: true,
        attn: true,
    };

    pub fn any(self) -> bool {
        self.gdn || self.attn
    }

    pub fn admits(self, family: Exl3DenseFamily) -> bool {
        match family {
            Exl3DenseFamily::Gdn => self.gdn,
            Exl3DenseFamily::Attn => self.attn,
        }
    }
}

/// `ATLAS_EXL3_NATIVE_DENSE=1`: serve the GDN + attention dense projections
/// natively from packed trellis (requires `ATLAS_EXL3_NATIVE=1`). Read per
/// call — load paths only.
pub fn exl3_native_dense_enabled() -> bool {
    std::env::var("ATLAS_EXL3_NATIVE_DENSE").as_deref() == Ok("1")
}

/// The admitted dense families from the environment: the DENSE gate plus
/// the optional per-family `=0` opt-outs. Does NOT validate — call
/// [`check_exl3_native_dense_gates`] first (the materialize pass does).
pub fn exl3_native_dense_families() -> Exl3DenseFamilies {
    exl3_native_dense_families_with(
        exl3_native_dense_enabled(),
        std::env::var("ATLAS_EXL3_NATIVE_GDN").ok().as_deref(),
        std::env::var("ATLAS_EXL3_NATIVE_ATTN").ok().as_deref(),
    )
}

/// Env-independent body of [`exl3_native_dense_families`].
pub fn exl3_native_dense_families_with(
    dense: bool,
    gdn_env: Option<&str>,
    attn_env: Option<&str>,
) -> Exl3DenseFamilies {
    if !dense {
        return Exl3DenseFamilies::OFF;
    }
    Exl3DenseFamilies {
        gdn: gdn_env != Some("0"),
        attn: attn_env != Some("0"),
    }
}

/// Gate-combination validation for the dense gates. The DENSE gate extends
/// the master native gate, and the per-family sub-gates refine the DENSE
/// gate; any of them set without its parent is a misconfiguration that must
/// fail at load, never silently serve the materialized BF16 copies.
pub fn check_exl3_native_dense_gates(
    native: bool,
    dense: bool,
    gdn_env: Option<&str>,
    attn_env: Option<&str>,
) -> Result<()> {
    if dense && !native {
        bail!(
            "ATLAS_EXL3_NATIVE_DENSE=1 requires ATLAS_EXL3_NATIVE=1 (the dense \
             gate extends the native serving set; it cannot enable native \
             serving by itself) — set ATLAS_EXL3_NATIVE=1 or unset \
             ATLAS_EXL3_NATIVE_DENSE"
        );
    }
    if !dense {
        for (name, v) in [
            ("ATLAS_EXL3_NATIVE_GDN", gdn_env),
            ("ATLAS_EXL3_NATIVE_ATTN", attn_env),
        ] {
            if let Some(v) = v {
                bail!(
                    "{name}={v} is set but ATLAS_EXL3_NATIVE_DENSE is not 1 — the \
                     per-family sub-gates only refine the DENSE gate (=0 opts a \
                     family out); set ATLAS_EXL3_NATIVE_DENSE=1 or unset {name}"
                );
            }
        }
    }
    for (name, v, family) in [
        ("ATLAS_EXL3_NATIVE_GDN", gdn_env, Exl3DenseFamily::Gdn),
        ("ATLAS_EXL3_NATIVE_ATTN", attn_env, Exl3DenseFamily::Attn),
    ] {
        if let Some(v) = v
            && v != "0"
            && v != "1"
        {
            bail!("{name}={v}: expected 0 (opt the family out) or 1");
        }
        // An EXPLICIT opt-in to a family with no dispatch arm is a request
        // that cannot be honored — fail at load rather than serve BF16 while
        // the operator believes the family is native.
        if dense && v == Some("1") && !family.routed() {
            bail!(
                "{name}=1 but the {} family has no native dispatch arm routed yet (its \
                 projections materialize to BF16); unset {name} to keep the default",
                family.module(),
            );
        }
    }
    Ok(())
}

/// Say once, at load, which gate-admitted families have no dispatch arm yet
/// and therefore keep materializing (the tally would otherwise read as "0
/// attention layers kept" with no reason attached).
pub fn log_unrouted_dense_families(families: Exl3DenseFamilies) {
    for family in [Exl3DenseFamily::Gdn, Exl3DenseFamily::Attn] {
        if families.admits(family) && !family.routed() {
            tracing::info!(
                "EXL3 native dense: the {} family ({:?}) has no native dispatch arm routed \
                 yet — its projections materialize to BF16 as before",
                family.module(),
                family.all_leaves(),
            );
        }
    }
}

/// The two natively-served dense families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Exl3DenseFamily {
    /// `linear_attn.{in_proj_qkv, in_proj_z, out_proj}` (36 GDN layers).
    Gdn,
    /// `self_attn.{q, k, v, o}_proj` (12 full-attention layers).
    Attn,
}

impl Exl3DenseFamily {
    /// Every projection of the family as the checkpoint packs it.
    pub fn all_leaves(self) -> &'static [&'static str] {
        match self {
            Self::Gdn => &["in_proj_qkv", "in_proj_z", "out_proj"],
            Self::Attn => &["q_proj", "k_proj", "v_proj", "o_proj"],
        }
    }

    /// The projections that HAVE a native layer-site dispatch arm — the
    /// keep-set. ALL of them are kept for a layer or none. Grows in lockstep
    /// with the arms (see the module docs): the whole GDN family
    /// (`Exl3GdnWeights::{in_proj_linear, out_proj_linear}`) and the whole
    /// attention family (`Exl3AttnWeights::{proj, qkv, kv, o_proj}_linear` at
    /// every decode / multi-seq / prefill site).
    pub fn leaves(self) -> &'static [&'static str] {
        match self {
            Self::Gdn => &["in_proj_qkv", "in_proj_z", "out_proj"],
            Self::Attn => &["q_proj", "k_proj", "v_proj", "o_proj"],
        }
    }

    /// Whether any projection of the family has a dispatch arm.
    pub fn routed(self) -> bool {
        !self.leaves().is_empty()
    }

    /// The module segment between the layer prefix and the leaf.
    pub fn module(self) -> &'static str {
        match self {
            Self::Gdn => "linear_attn",
            Self::Attn => "self_attn",
        }
    }

    /// The store prefixes of one layer's family: `{lp}.{module}.{leaf}`.
    pub fn prefixes(self, lp: &str) -> Vec<String> {
        let m = self.module();
        self.leaves()
            .iter()
            .map(|leaf| format!("{lp}.{m}.{leaf}"))
            .collect()
    }
}

/// Classify a trellis prefix: `Some((layer_key, family))` for a dense
/// projection in the natively-served set, `None` otherwise. `layer_key` is
/// everything before `.linear_attn.` / `.self_attn.`.
///
/// Excludes `mtp.*` (not wired on this branch), `indexer.*` (the QSA
/// projection lives under `self_attn.indexer.`, so `self_attn.indexer.
/// index_qk_proj` does not match the `self_attn.<leaf>` shape), and every
/// leaf outside the family lists.
pub fn exl3_dense_prefix_family(prefix: &str) -> Option<(&str, Exl3DenseFamily)> {
    if prefix.starts_with("mtp.") || prefix.contains(".mtp.") {
        return None;
    }
    for family in [Exl3DenseFamily::Gdn, Exl3DenseFamily::Attn] {
        let marker = format!(".{}.", family.module());
        if let Some(i) = prefix.find(&marker) {
            let leaf = &prefix[i + marker.len()..];
            if family.leaves().contains(&leaf) {
                return Some((&prefix[..i], family));
            }
        }
    }
    None
}

/// True for a dense-projection prefix in the natively-served set AND whose
/// family the gates admit.
pub fn exl3_native_serves_dense(prefix: &str, families: Exl3DenseFamilies) -> bool {
    exl3_dense_prefix_family(prefix).is_some_and(|(_, f)| families.admits(f))
}

/// What the dense keep-set decided — for the load log.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct Exl3DenseKeepStats {
    pub gdn_layers_kept: usize,
    pub gdn_layers_materialized: usize,
    pub attn_layers_kept: usize,
    pub attn_layers_materialized: usize,
    /// Resident bytes of the kept-packed dense tensors.
    pub kept_packed_bytes: usize,
    /// What those same tensors WOULD have cost as BF16 `[out, in]` (the
    /// materialize path's format for this set).
    pub bf16_equiv_bytes: usize,
}

/// Compute the set of dense prefixes to KEEP packed, atomically per
/// (layer, family).
///
/// `dense` maps every gate-admitted dense trellis prefix in the store to its
/// resolved [`Exl3Weight`]. A (layer, family) keeps ALL its projections iff
/// every leaf of the family is present in `dense` and every one is inside
/// [`super::exl3_native_supported`]. Otherwise the whole family of that
/// layer materializes (one warn naming the layer, family and reason).
pub(crate) fn dense_keep_set(
    dense: &BTreeMap<String, Exl3Weight>,
) -> (HashSet<String>, Exl3DenseKeepStats) {
    let mut groups: BTreeMap<(&str, Exl3DenseFamily), Vec<(&str, &Exl3Weight)>> = BTreeMap::new();
    for (p, w) in dense {
        if let Some((layer, family)) = exl3_dense_prefix_family(p) {
            groups
                .entry((layer, family))
                .or_default()
                .push((p.as_str(), w));
        }
    }

    let mut keep = HashSet::new();
    let mut stats = Exl3DenseKeepStats::default();
    for ((layer, family), tensors) in &groups {
        let expected = family.leaves().len();
        let reason = if tensors.len() != expected {
            Some(format!(
                "only {} of the {expected} routed {} projections are EXL3 trellis in \
                 the store (the rest ship in another format)",
                tensors.len(),
                family.module(),
            ))
        } else {
            tensors
                .iter()
                .find(|(_, w)| !super::exl3_native_supported(w))
                .map(|(p, w)| {
                    format!(
                        "{p} is outside the dense kernel envelope (K={} cb={:?} \
                         [{}x{}]; need K in {{2,4}}, cb MCG/MUL1, dims %128)",
                        w.k_bits, w.cb, w.in_dim, w.out_dim,
                    )
                })
        };
        match reason {
            Some(reason) => {
                tracing::warn!(
                    "EXL3 native dense: {layer} {} family falls back to BF16 \
                     materialization — {reason}. Atomic per layer: NO projection \
                     of this family is kept packed.",
                    family.module(),
                );
                match family {
                    Exl3DenseFamily::Gdn => stats.gdn_layers_materialized += 1,
                    Exl3DenseFamily::Attn => stats.attn_layers_materialized += 1,
                }
            }
            None => {
                for (p, w) in tensors {
                    keep.insert((*p).to_string());
                    stats.kept_packed_bytes += w.packed_bytes();
                    stats.bf16_equiv_bytes += w.in_dim * w.out_dim * 2;
                }
                match family {
                    Exl3DenseFamily::Gdn => stats.gdn_layers_kept += 1,
                    Exl3DenseFamily::Attn => stats.attn_layers_kept += 1,
                }
            }
        }
    }
    (keep, stats)
}

/// LOADER-side re-derivation of "was this layer's family kept packed?":
/// the gates admit the family AND every projection of the family is still
/// an EXL3 linear in the store. The materialize pass keeps a family only
/// atomically, so the two predicates agree; checking every leaf (not just
/// the first) is defense against a store that was assembled some other way
/// — a half-present family fails the loader's `ensure!` instead of reading
/// a `.weight` that was never written.
pub fn exl3_dense_family_kept(store: &WeightStore, lp: &str, family: Exl3DenseFamily) -> bool {
    if !(family.routed()
        && super::exl3_native_enabled()
        && exl3_native_dense_families().admits(family))
    {
        return false;
    }
    let ps = family.prefixes(lp);
    let n_exl3 = ps.iter().filter(|p| is_exl3_linear(store, p)).count();
    if n_exl3 != 0 && n_exl3 != ps.len() {
        // Not reachable through the materialize pass; say so loudly rather
        // than let the caller take the BF16 arm over a missing `.weight`.
        tracing::error!(
            "EXL3 native dense: {lp} {} family is HALF packed ({n_exl3} of {} \
             projections hold trellis) — the atomic keep-set cannot produce \
             this; the BF16 arm will fail on the missing .weight",
            family.module(),
            ps.len(),
        );
    }
    n_exl3 == ps.len()
}

#[cfg(test)]
#[path = "exl3_materialize_dense_tests.rs"]
mod tests;
