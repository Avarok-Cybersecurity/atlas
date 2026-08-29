// SPDX-License-Identifier: AGPL-3.0-only

//! Pins GLM-5.3's DSA decode to the kernel it actually resolves at serve time, and blocks
//! the shadow/duplicate drift that a second copy of an MLA kernel would reintroduce.
//!
//! History: `mla_paged_decode{,_fp8}.cu` lived only under `deepseek-v4-flash/nvfp4/`, so no
//! other target could resolve them. Byte-identical copies were added to `common/` to make
//! them reachable — V4's originals then SHADOWED the common pair, which meant two copies of
//! the same kernel that nothing kept in step. GLM never called them anyway: its DSA layer
//! resolves its own NoPE selected-index sparse decode. The copies are gone; these tests
//! exist so the next copy has to justify itself.
//!
//! Mirrors the idiom in `moe_topk_sigmoid_bounds.rs`.

use std::path::{Path, PathBuf};

use spark_model::layers::glm5next_dsa::KERNEL_KV_LORA_DIM;
use spark_model::layers::glm5next_dsa::attend::DSA_DECODE_MODULE;

fn kernels_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/spark-model is two levels below the workspace root")
        .join("kernels")
}

/// `#define <name> <integer>`, ignoring any trailing comment.
fn define(text: &str, name: &str, whose: &str) -> usize {
    let needle = format!("#define {name} ");
    let line = text
        .lines()
        .find(|l| l.trim_start().starts_with(&needle))
        .unwrap_or_else(|| panic!("{whose} no longer defines {name}"));
    line.trim_start()[needle.len()..]
        .split_whitespace()
        .next()
        .and_then(|t| t.parse().ok())
        .unwrap_or_else(|| panic!("{name} in {whose} is not a plain integer: {line}"))
}

/// THE PRODUCTION RESOLUTION PATH. `Glm5NextDsaDecodeKernel::resolve` asks for
/// `glm5next_dsa_mla_decode_fp8`, and unlisted `.cu` files take their file stem as the
/// module name. If the file is renamed or moved out of the GLM target without updating the
/// resolve site, serving fails at load — this fails in CI instead.
#[test]
fn the_dsa_decode_entry_point_exists_in_the_glm_target() {
    // The module name IS the file stem for unlisted `.cu` files, so derive the path from
    // the constant the resolve site uses rather than restating it.
    let cu = kernels_root().join(format!("gb10/glm-5.3-flash/nvfp4/{DSA_DECODE_MODULE}.cu"));
    let src = std::fs::read_to_string(&cu).unwrap_or_else(|e| {
        panic!("GLM's DSA decode kernel is missing at {cu:?} ({e}); `Glm5NextDsaDecodeKernel::resolve` would fail at load")
    });
    assert!(
        src.contains("glm5next_dsa_mla_decode_fp8"),
        "{cu:?} no longer defines the entry point `Glm5NextDsaDecodeKernel::resolve` asks \
         for (`glm5next_dsa_mla_decode_fp8`). The module name is the file stem, so a rename \
         here is a load-time failure at serve."
    );
}

/// The Rust constant the config check compares against is a MIRROR of the kernel's
/// `#define`. If the kernel's latent width changes and the mirror does not, the check waves
/// through a config the kernel cannot hold — the same shape as the bug it guards.
#[test]
fn rust_kv_lora_mirror_matches_the_glm_kernel_define() {
    let cu = kernels_root().join(format!("gb10/glm-5.3-flash/nvfp4/{DSA_DECODE_MODULE}.cu"));
    let src = std::fs::read_to_string(&cu).expect("GLM DSA decode kernel readable");
    assert_eq!(
        define(&src, "GLM_KV_LORA_DIM", "glm5next_dsa_mla_decode.cu"),
        KERNEL_KV_LORA_DIM,
        "GLM_KV_LORA_DIM and KERNEL_KV_LORA_DIM disagree; Glm5NextDsaConfig::validate would \
         admit a checkpoint the kernel reads at the wrong width."
    );
}

/// No second copy of the V4 MLA paged-decode kernels. `common/` merges into every target, so
/// a copy there is shadowed by any model directory that also has one — two files, one name,
/// nothing keeping them in step.
#[test]
fn no_duplicate_mla_paged_decode_kernel() {
    let root = kernels_root();
    for stem in ["mla_paged_decode.cu", "mla_paged_decode_fp8.cu"] {
        let mut found: Vec<String> = Vec::new();
        let mut stack = vec![root.clone()];
        while let Some(dir) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&dir) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.file_name().and_then(|s| s.to_str()) == Some(stem) {
                    found.push(
                        p.strip_prefix(&root).unwrap_or(&p).display().to_string(),
                    );
                }
            }
        }
        found.sort();
        assert_eq!(
            found.len(),
            1,
            "{stem} must exist in exactly one place; found {found:?}. A `common/` copy plus a \
             model-directory copy is a shadow: the model's copy wins and the two silently \
             diverge. Promote or delete — do not fork."
        );
    }
}
