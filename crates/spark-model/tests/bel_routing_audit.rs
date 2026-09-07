// SPDX-License-Identifier: AGPL-3.0-only

//! Every MoE routing path either masks the router or refuses to run.
//!
//! Boot-time expert loading (`--expert-category`) leaves NULL entries in the
//! dense expert pointer table for experts it did not load. The router mask
//! is what keeps top-k from naming one. A routing path that selects experts
//! without applying the mask, and without refusing, hands a kernel a null
//! pointer — an illegal memory access that kills the serve, with nothing in
//! the log connecting it to the flag that caused it.
//!
//! The compiler cannot check this: applying the mask is a call, not a type.
//! So this scans the sources the way `kernel_shadow_detector.rs` and
//! `moe_site_wiring.rs` do, and fails when a file selects experts without
//! being on one side of the partition.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn moe_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("layers")
        .join("moe")
}

/// Files that select experts: they call a top-k or hash-route op.
fn routing_files() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(moe_dir()).expect("moe module dir") {
        let p = e.unwrap().path();
        if p.extension().and_then(|x| x.to_str()) != Some("rs") {
            continue;
        }
        let name = p.file_name().unwrap().to_str().unwrap().to_string();
        let text = std::fs::read_to_string(&p).unwrap();
        // Strip comment lines so prose naming a kernel does not count as a
        // call — the mistake that made the first version of this vacuous.
        let code: String = text
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        if code.contains("ops::moe_topk_") || code.contains("ops::moe_hash_route") {
            out.push((name, code));
        }
    }
    out.sort();
    out
}

#[test]
fn every_routing_path_masks_or_refuses() {
    let files = routing_files();
    assert!(
        files.len() >= 8,
        "the scan stopped finding routing files — it would pass vacuously: {:?}",
        files.iter().map(|(n, _)| n).collect::<Vec<_>>()
    );

    let mut unprotected = Vec::new();
    for (name, code) in &files {
        // Two ways to be masked, both on the same buffer the top-k reads:
        //   * the file applies the mask itself;
        //   * it obtains its logits from `batched_gate_logits`, which masks
        //     the whole [n, num_experts] block before returning it;
        //   * `forward_prefill_router.rs` holds shared bias-dispatch helpers
        //     its callers invoke BETWEEN their own mask and their top-k.
        let masked = code.contains("apply_bel_mask")
            || code.contains("batched_gate_logits(")
            || name == "forward_prefill_router.rs";
        let refuses = code.contains("bel_guard");
        if !masked && !refuses {
            unprotected.push(name.clone());
        }
    }
    assert!(
        unprotected.is_empty(),
        "these MoE routing paths neither apply the BEL router mask nor refuse under it, so \
         --expert-category could let them select an expert with no weights loaded: {unprotected:?}"
    );
}

#[test]
fn the_masked_and_refusing_sets_do_not_overlap() {
    // A path that both masks and refuses is a contradiction: either the mask
    // works there and the refusal is dead, or it does not and the mask is a
    // lie. Either way a reader cannot tell which is true.
    let both: Vec<String> = routing_files()
        .into_iter()
        .filter(|(_, code)| code.contains("apply_bel_mask") && code.contains("bel_guard"))
        .map(|(n, _)| n)
        .collect();
    assert!(
        both.is_empty(),
        "these files both mask and refuse — pick one: {both:?}"
    );
}

#[test]
fn the_masked_paths_are_the_ones_bel_claims_to_support() {
    // BEL's documented v1 scope: the prefill paths, single-token decode, and
    // the batched decode gate. If this set changes, the flag's own docs and
    // the boot-time validation have to change with it — this test is what
    // makes that a decision rather than a drift.
    let masked: BTreeSet<String> = routing_files()
        .into_iter()
        .filter(|(_, code)| code.contains("apply_bel_mask"))
        .map(|(n, _)| n)
        .collect();
    // Files that mask DIRECTLY. `forward_batched.rs` is masked one hop away,
    // inside `batched_gate_logits`, and is covered by the partition test.
    let expected: BTreeSet<String> = [
        "forward.rs",
        "forward_prefill.rs",
        "forward_prefill_bf16.rs",
        "forward_prefill_fp8.rs",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(
        masked, expected,
        "the set of BEL-masked routing paths changed; update the --expert-category docs and \
         the boot validation to match"
    );
}
