// SPDX-License-Identifier: AGPL-3.0-only

//! Pins the sigmoid-routing kernel's shared-memory bounds to their Rust mirrors,
//! and blocks the shadow drift that made those bounds wrong in the first place.
//!
//! `kernels/gb10/common/moe_topk_sigmoid.cu` sizes its top-K staging arrays from
//! `#define MAX_TOP_K` / `#define MAX_EXPERTS`. Two model directories shipped
//! their own copy of that file with `MAX_TOP_K` lowered 32 -> 24 while
//! Nemotron-Super-120B routes `num_experts_per_tok = 22` — two slots of headroom,
//! no host-side check, and a device-side loop bounded by the expert count rather
//! than by the array. The same shadows had also dropped the lower-index-wins
//! tie-break the common file carries, so one model's decode and its batched
//! prefill could route identical logits to different experts.
//!
//! Both shadows are gone; the whole family builds from `common/` again. These
//! tests exist so the next copy has to justify itself.

use std::path::{Path, PathBuf};

use spark_model::layers::ops::{MOE_TOPK_SIGMOID_MAX_EXPERTS, MOE_TOPK_SIGMOID_MAX_TOP_K};

const KERNEL: &str = "gb10/common/moe_topk_sigmoid.cu";

fn kernels_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/spark-model is two levels below the workspace root")
        .join("kernels")
}

/// `#define <name> <integer>`, ignoring any trailing comment.
fn define(text: &str, name: &str) -> usize {
    let needle = format!("#define {name} ");
    let line = text
        .lines()
        .find(|l| l.trim_start().starts_with(&needle))
        .unwrap_or_else(|| panic!("{KERNEL} no longer defines {name}"));
    line.trim_start()[needle.len()..]
        .split_whitespace()
        .next()
        .and_then(|t| t.parse().ok())
        .unwrap_or_else(|| panic!("{name} in {KERNEL} is not a plain integer: {line}"))
}

/// The Rust constants the load-time checks compare against are a MIRROR. If the
/// kernel's arrays shrink and the mirror does not, the check waves through a
/// config the kernel cannot hold — which is precisely the shape of the bug.
#[test]
fn rust_bounds_mirror_the_kernel_defines() {
    let text = std::fs::read_to_string(kernels_root().join(KERNEL)).unwrap();
    assert_eq!(
        define(&text, "MAX_TOP_K"),
        MOE_TOPK_SIGMOID_MAX_TOP_K,
        "MAX_TOP_K in {KERNEL} and MOE_TOPK_SIGMOID_MAX_TOP_K must move together"
    );
    assert_eq!(
        define(&text, "MAX_EXPERTS"),
        MOE_TOPK_SIGMOID_MAX_EXPERTS,
        "MAX_EXPERTS in {KERNEL} and MOE_TOPK_SIGMOID_MAX_EXPERTS must move together"
    );
}

/// A model directory that shadows this kernel gets its own copy of the bounds
/// AND its own copy of the tie-break, and nothing compares the two. Ban the
/// shadow rather than trying to diff it: the kernel has no model-specific
/// content, so a copy is drift by construction.
#[test]
fn no_model_directory_shadows_the_sigmoid_routing_kernel() {
    let root = kernels_root();
    let mut shadows = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.file_name().is_some_and(|n| n == "moe_topk_sigmoid.cu")
                && path.parent().is_some_and(|p| !p.ends_with("common"))
            {
                shadows.push(
                    path.strip_prefix(&root)
                        .unwrap_or(path.as_path())
                        .display()
                        .to_string(),
                );
            }
        }
    }
    shadows.sort();
    assert!(
        shadows.is_empty(),
        "moe_topk_sigmoid.cu belongs only in a `common/` directory; these copies \
         will drift in their MAX_TOP_K and their tie-break exactly as the two \
         Nemotron ones did: {shadows:?}"
    );
}

/// Every checkpoint the repo declares must fit the bounds the kernel can hold.
/// Nemotron-Super-120B is the tightest at 22 of 32, and the Nemotron and
/// Step-3.7 families sit exactly on the 512-expert ceiling. Compile-time rather
/// than a `#[test]`, so lowering a bound below a shipped config cannot even
/// build — which is the failure mode the 24-cap shadows had.
const _: () = assert!(
    MOE_TOPK_SIGMOID_MAX_TOP_K >= 22,
    "Nemotron-Super-120B-A12B routes num_experts_per_tok=22"
);
const _: () = assert!(
    MOE_TOPK_SIGMOID_MAX_EXPERTS >= 512,
    "the Nemotron and Step-3.7 families declare n_routed_experts=512"
);
