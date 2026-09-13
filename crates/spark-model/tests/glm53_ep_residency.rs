// SPDX-License-Identifier: AGPL-3.0-only

//! Slice-12 GATE 1: **loader-side expert residency, proven without collectives.**
//!
//! Scoped to `LibertAIDAI/GLM-5.3-Flash-NVFP4` snapshot
//! `9e0d74e3cef17f634e84fb8e2223707e02616290` — 120 shards, 113,074 tensors.
//!
//! # What this proves
//!
//! Slice 11 recorded "the weight loader has no expert-residency sharding" after
//! grepping `spark-model/src/weight_loader/`. That was **mis-scoped**: residency
//! is generic and lives one crate over, in
//! [`spark_runtime::weights::SafetensorsLoader::should_skip_tensor`]. This test
//! drives that **real predicate** — not a reimplementation — over the **real**
//! checkpoint tensor index and proves the EP=2 split.
//!
//! # Fixture
//!
//! `name<TAB>bytes<TAB>dtype<TAB>shard`, one row per tensor, produced by reading
//! the safetensors header of all 120 shards on n3 (headers only — no tensor data
//! is read, the checkpoint is never mutated). Not committed (12 MB); the test
//! **skips loudly** when absent. Regenerate with
//! `docs/glm5next/dump_st_index.py`.
//!
//! Path override: `GLM53_TENSOR_INDEX`.
//!
//! # 🪤 Traps this test pins
//!
//! * `should_skip_tensor`'s MTP-replication exemption keys on a leading `mtp.`,
//!   a DeepSeek-style name. **GLM-5.3 has zero such tensors** — its MTP head is
//!   `model.language_model.layers.45.*`. The exemption is a silent no-op here.
//!   Harmless while layer 45 is out of scope; revisit before enabling MTP.
//! * `shared_experts.*` must NOT be sharded. It does not match `parse_expert_index`
//!   (the segment is `shared_experts`, not `experts`), so it replicates. Asserted.

use std::collections::BTreeSet;

use spark_model::weight_loader::glm5_next::classify;
use spark_runtime::weights::{SafetensorsLoader, parse_expert_index};

const NUM_EXPERTS: usize = 288;
const EP_WORLD: usize = 2;

/// Measured over the reference checkpoint, not modelled.
const TEXT_MODEL_BYTES: u64 = 189_071_011_832; // 176.086 GiB
const ROUTED_TOTAL_BYTES: u64 = 171_228_411_648; // 159.469 GiB
const PER_RANK_BYTES: u64 = 103_456_806_008; // 96.352 GiB
const PER_RANK_TENSORS: usize = 55_678;
/// FNV-1a-64 over each rank's newline-joined, sorted kept-tensor names.
const RANK_NAME_DIGEST: [u64; 2] = [0xfc98_f8f8_2517_2351, 0x21a0_2360_cfc0_f023];

const GIB: f64 = (1u64 << 30) as f64;

struct Row {
    name: String,
    bytes: u64,
}

fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// `model.language_model.layers.<i>.` → `i`.
fn layer_of(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("model.language_model.layers.")?;
    rest.split('.').next()?.parse().ok()
}

/// Load the index, or `None` when the fixture is absent (skip, loudly).
fn load_index() -> Option<Vec<Row>> {
    let path = std::env::var("GLM53_TENSOR_INDEX")
        .unwrap_or_else(|_| "/home/msi1/atlas-scratch/ep-residency/glm53_tensor_index.tsv".into());
    let text = std::fs::read_to_string(&path).ok()?;
    let rows: Vec<Row> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split('\t');
            let name = it.next().expect("name").to_string();
            let bytes = it.next().expect("bytes").parse().expect("bytes is u64");
            Row { name, bytes }
        })
        .collect();
    assert_eq!(
        rows.len(),
        113_074,
        "fixture is not the reference checkpoint"
    );
    Some(rows)
}

macro_rules! index_or_skip {
    () => {
        match load_index() {
            Some(r) => r,
            None => {
                eprintln!(
                    "SKIP: GLM-5.3 tensor index absent. Regenerate with \
                     docs/glm5next/dump_st_index.py on a node holding the checkpoint \
                     (n3/n4), or set GLM53_TENSOR_INDEX."
                );
                return;
            }
        }
    };
}

/// Text-model tensors: everything except the vision tower and the layer-45 MTP
/// head. Mirrors the Slice-11 accounting split so the two reconcile.
fn is_text_model_loaded(name: &str) -> bool {
    let role = classify(name).unwrap_or_else(|| panic!("unclassified tensor: {name}"));
    role.is_text_model() && layer_of(name) != Some(45)
}

/// THE gate: disjoint, complete, correctly sized expert residency at EP=2,
/// decided by the real loader predicate.
#[test]
fn ep2_expert_residency_is_disjoint_complete_and_exact() {
    let rows = index_or_skip!();

    let loaders: Vec<SafetensorsLoader> = (0..EP_WORLD)
        .map(|r| SafetensorsLoader::with_ep(r, EP_WORLD, NUM_EXPERTS))
        .collect();

    let mut text_bytes = 0u64;
    let mut routed_total = 0u64;
    let mut kept = [0u64; EP_WORLD];
    let mut routed_kept = [0u64; EP_WORLD];
    let mut owned: [BTreeSet<usize>; EP_WORLD] = Default::default();
    let mut kept_names: [Vec<&str>; EP_WORLD] = Default::default();

    for row in &rows {
        if !is_text_model_loaded(&row.name) {
            continue;
        }
        text_bytes += row.bytes;
        let expert = parse_expert_index(&row.name);
        if expert.is_some() {
            routed_total += row.bytes;
        }
        for r in 0..EP_WORLD {
            if loaders[r].should_skip_tensor(&row.name) {
                continue;
            }
            kept[r] += row.bytes;
            kept_names[r].push(&row.name);
            if let Some(e) = expert {
                routed_kept[r] += row.bytes;
                owned[r].insert(e);
            }
        }
    }

    // Reconciles with the Slice-11 checkpoint accounting.
    assert_eq!(text_bytes, TEXT_MODEL_BYTES, "text-model bytes drifted");
    assert_eq!(
        routed_total, ROUTED_TOTAL_BYTES,
        "routed-expert bytes drifted"
    );

    // Ownership: disjoint, complete, evenly split, canonical absolute ids.
    assert!(
        owned[0].is_disjoint(&owned[1]),
        "routed experts are resident on BOTH ranks: {:?}",
        owned[0]
            .intersection(&owned[1])
            .take(10)
            .collect::<Vec<_>>()
    );
    let union: BTreeSet<usize> = owned[0].union(&owned[1]).copied().collect();
    assert_eq!(
        union,
        (0..NUM_EXPERTS).collect::<BTreeSet<_>>(),
        "union of rank inventories is not all {NUM_EXPERTS} experts"
    );
    assert_eq!(owned[0].len(), 144, "rank 0 expert count");
    assert_eq!(owned[1].len(), 144, "rank 1 expert count");
    // Absolute (global) ids, contiguous per rank — no renumbering to 0..144.
    assert_eq!(
        (*owned[0].first().unwrap(), *owned[0].last().unwrap()),
        (0, 143)
    );
    assert_eq!(
        (*owned[1].first().unwrap(), *owned[1].last().unwrap()),
        (144, 287)
    );

    // Residency: exact bytes, and no duplicate remote residency.
    assert_eq!(
        routed_kept[0] + routed_kept[1],
        routed_total,
        "routed bytes must partition exactly — no duplication, no loss"
    );
    for r in 0..EP_WORLD {
        assert_eq!(kept[r], PER_RANK_BYTES, "rank {r} total residency");
        assert_eq!(
            kept_names[r].len(),
            PER_RANK_TENSORS,
            "rank {r} tensor count"
        );

        // Ownership is hash-provable and reproducible.
        kept_names[r].sort_unstable();
        let digest = fnv1a64(&kept_names[r].join("\n"));
        assert_eq!(
            digest, RANK_NAME_DIGEST[r],
            "rank {r} inventory digest drifted: 0x{digest:016x}"
        );
    }

    let replicated = kept[0] - routed_kept[0];
    eprintln!(
        "GATE 1 PASS — EP=2 residency\n\
         \x20 text model      {:>15} B  {:8.3} GiB\n\
         \x20 routed/rank     {:>15} B  {:8.3} GiB  (144 experts)\n\
         \x20 replicated/rank {:>15} B  {:8.3} GiB\n\
         \x20 TOTAL/rank      {:>15} B  {:8.3} GiB  ({PER_RANK_TENSORS} tensors)",
        text_bytes,
        text_bytes as f64 / GIB,
        routed_kept[0],
        routed_kept[0] as f64 / GIB,
        replicated,
        replicated as f64 / GIB,
        kept[0],
        kept[0] as f64 / GIB,
    );
}

/// EP=1 must load everything — the residency predicate is inert at world size 1.
#[test]
fn ep1_loads_every_expert() {
    let rows = index_or_skip!();
    let loader = SafetensorsLoader::new();
    let skipped = rows
        .iter()
        .filter(|r| loader.should_skip_tensor(&r.name))
        .count();
    assert_eq!(skipped, 0, "EP=1 must not skip any tensor");
}

/// The shared expert and the router are replicated, not sharded. If either got
/// caught by the expert-index parser, every rank would compute a partial shared
/// expert and the post-all-reduce add would be wrong.
#[test]
fn shared_expert_and_router_replicate_on_both_ranks() {
    let rows = index_or_skip!();
    let loaders: Vec<SafetensorsLoader> = (0..EP_WORLD)
        .map(|r| SafetensorsLoader::with_ep(r, EP_WORLD, NUM_EXPERTS))
        .collect();

    let mut shared = 0usize;
    let mut router = 0usize;
    for row in &rows {
        let is_shared = row.name.contains(".shared_experts.");
        let is_router = row.name.contains(".mlp.gate.") || row.name.ends_with(".mlp.gate.weight");
        if !(is_shared || is_router) {
            continue;
        }
        assert!(
            parse_expert_index(&row.name).is_none(),
            "replicated tensor parsed as a routed expert: {}",
            row.name
        );
        for (r, l) in loaders.iter().enumerate() {
            assert!(
                !l.should_skip_tensor(&row.name),
                "rank {r} would skip replicated tensor {}",
                row.name
            );
        }
        if is_shared {
            shared += 1;
        } else {
            router += 1;
        }
    }
    assert!(
        shared > 0,
        "no shared-expert tensors found — fixture wrong?"
    );
    assert!(router > 0, "no router tensors found — fixture wrong?");
}

/// 🪤 GLM-5.3 has no `mtp.`-prefixed tensor, so the loader's MTP-replication
/// exemption never fires. Pinned so it fails loudly if the MTP head is enabled
/// later and silently gets EP-sharded.
#[test]
fn glm53_has_no_mtp_prefixed_tensors() {
    let rows = index_or_skip!();
    let n = rows.iter().filter(|r| r.name.starts_with("mtp.")).count();
    assert_eq!(
        n, 0,
        "GLM-5.3 gained `mtp.`-prefixed tensors — re-check the MTP EP exemption"
    );
    // Layer 45 DOES carry routed experts, and they WOULD be sharded.
    let l45_experts = rows
        .iter()
        .filter(|r| layer_of(&r.name) == Some(45) && parse_expert_index(&r.name).is_some())
        .count();
    assert_eq!(l45_experts, 2_592, "layer-45 routed expert tensor count");
}
