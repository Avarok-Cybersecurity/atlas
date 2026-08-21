// SPDX-License-Identifier: AGPL-3.0-only

//! The two refutations at the top are the reason this module exists: both pass
//! trivially against a hash of the resolved file SET, and both are real shapes
//! in `kernels/` today.

use super::*;

fn tmp() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "atlas-closure-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("common")).unwrap();
    std::fs::create_dir_all(dir.join("model/nvfp4")).unwrap();
    dir
}

fn write(p: &Path, s: &str) {
    std::fs::write(p, s).unwrap();
}

fn inputs(sources: Vec<PathBuf>) -> ClosureInputs {
    ClosureInputs {
        sources,
        configs: Vec::new(),
        flags: vec!["--fmad=false".into()],
        arch: "sm_121a".into(),
        compiler: "nvcc 13.0.2".into(),
    }
}

/// ★ REFUTATION 1 — a shadow file that `#include`s what it shadows.
///
/// Real shape: `kernels/gb10/qwen3.6-27b/nvfp4/inferspark_prefill_paged_indirect.cu`
/// contains `#include "../../common/inferspark_prefill_paged_indirect.cu"`.
/// A set-membership hash says the model shadows that stem and is therefore
/// immune to a change in the common copy — while the edited bytes are compiled
/// straight into its kernel. Fails open, silently.
#[test]
fn editing_a_common_file_a_shadow_includes_changes_the_hash() {
    let d = tmp();
    let common = d.join("common/prefill.cu");
    let shadow = d.join("model/nvfp4/prefill.cu");
    write(&common, "__global__ void k() { int tile = 64; }\n");
    write(
        &shadow,
        "#include \"../../common/prefill.cu\"\n// model tweak\n",
    );

    let before = hash(&d, &inputs(vec![shadow.clone()])).unwrap();
    write(&common, "__global__ void k() { int tile = 128; }\n");
    let after = hash(&d, &inputs(vec![shadow.clone()])).unwrap();

    assert_ne!(
        before, after,
        "a shadow that INCLUDES the common file must not look immune to \
         a change in it — this is the fail-open a file-set hash produces"
    );
}

/// ★ REFUTATION 2 — headers are in no `.cu` set at all.
///
/// The resolver matches `*.cu` non-recursively, so `common/*.cuh` is invisible
/// to it. Editing the header carrying `BR64` would invalidate nothing.
#[test]
fn editing_an_included_header_changes_the_hash() {
    let d = tmp();
    let header = d.join("common/compute.cuh");
    let src = d.join("common/attn.cu");
    write(&header, "#define BR64 64\n");
    write(&src, "#include \"compute.cuh\"\n__global__ void a() {}\n");

    let before = hash(&d, &inputs(vec![src.clone()])).unwrap();
    write(&header, "#define BR64 128\n");
    let after = hash(&d, &inputs(vec![src.clone()])).unwrap();

    assert_ne!(before, after, "a header edit must change the hash");
}

/// A genuine shadow — one that does NOT include the common file — keeps its
/// hash when the common copy changes. This is the whole point: the targets that
/// really are insulated stop paying for shared-kernel edits.
#[test]
fn a_true_shadow_is_insulated_from_the_common_file() {
    let d = tmp();
    let common = d.join("common/gemm.cu");
    let shadow = d.join("model/nvfp4/gemm.cu");
    write(&common, "__global__ void g() { /* generic */ }\n");
    write(
        &shadow,
        "__global__ void g() { /* hand-tuned, standalone */ }\n",
    );

    let before = hash(&d, &inputs(vec![shadow.clone()])).unwrap();
    write(&common, "__global__ void g() { /* generic, edited */ }\n");
    let after = hash(&d, &inputs(vec![shadow.clone()])).unwrap();

    assert_eq!(
        before, after,
        "a shadow that does not include the common file is genuinely unaffected"
    );
}

#[test]
fn transitive_includes_are_followed() {
    let d = tmp();
    write(&d.join("common/deep.cuh"), "#define X 1\n");
    write(&d.join("common/mid.cuh"), "#include \"deep.cuh\"\n");
    let src = d.join("common/top.cu");
    write(&src, "#include \"mid.cuh\"\n");

    let before = hash(&d, &inputs(vec![src.clone()])).unwrap();
    write(&d.join("common/deep.cuh"), "#define X 2\n");
    assert_ne!(
        before,
        hash(&d, &inputs(vec![src])).unwrap(),
        "two levels deep"
    );
}

/// ★ An unresolvable include is RECORDED, not fatal.
///
/// It was fatal at first. The tree then showed the rule cost more than it
/// bought: the only two unresolvable includes in `kernels/` are dead `#if`
/// arms naming files that exist nowhere, and failing on them denied an
/// attestation to the one target whose benchmarks cost 3.5 GPU-hours. See
/// `hash_with_report`.
#[test]
fn an_unresolvable_include_is_recorded_rather_than_fatal() {
    let d = tmp();
    let src = d.join("common/x.cu");
    write(
        &src,
        "#include \"nowhere/absent.cuh\"\n__global__ void k() {}\n",
    );
    let closure = hash_with_report(&d, &inputs(vec![src])).expect("must not fail");
    assert_eq!(closure.unresolved.len(), 1, "{:?}", closure.unresolved);
    let entry = closure.unresolved.iter().next().unwrap();
    assert!(entry.contains("nowhere/absent.cuh"), "{entry}");
    assert!(
        entry.starts_with("common/x.cu"),
        "the report must be repo-relative, or two checkouts disagree: {entry}"
    );
}

/// Becoming resolvable — or ceasing to be — must move the digest even though
/// the including file's own bytes are untouched.
#[test]
fn resolving_a_previously_missing_include_changes_the_hash() {
    let d = tmp();
    let src = d.join("common/x.cu");
    write(&src, "#include \"later.cuh\"\n__global__ void k() {}\n");
    let before = hash(&d, &inputs(vec![src.clone()])).unwrap();

    write(&d.join("common/later.cuh"), "#define L 1\n");
    let after = hash(&d, &inputs(vec![src.clone()])).unwrap();
    assert_ne!(before, after, "the file appearing must move the hash");

    // And once it exists, its CONTENT is inside the hash like any other.
    write(&d.join("common/later.cuh"), "#define L 2\n");
    assert_ne!(after, hash(&d, &inputs(vec![src])).unwrap());
}

/// Two different files naming the same missing header are two distinct facts,
/// so the report is keyed by the including file rather than by the include.
#[test]
fn unresolved_entries_are_keyed_by_the_including_file() {
    let d = tmp();
    let a = d.join("common/a.cu");
    let b = d.join("common/b.cu");
    write(&a, "#include \"gone.cuh\"\n");
    write(&b, "#include \"gone.cuh\"\n");
    let closure = hash_with_report(&d, &inputs(vec![a, b])).unwrap();
    assert_eq!(closure.unresolved.len(), 2, "{:?}", closure.unresolved);
}

/// A preprocessor conditional is not evaluated, so an include in a branch this
/// build never takes is still walked. That over-includes — costing re-runs,
/// not soundness — and the doc says so rather than implying `#if` is understood.
#[test]
fn includes_inside_untaken_conditionals_are_still_walked() {
    let d = tmp();
    let src = d.join("common/x.cu");
    write(&d.join("common/dead.cuh"), "#define D 1\n");
    write(
        &src,
        "#if defined(NEVER)\n#include \"dead.cuh\"\n#endif\n__global__ void k() {}\n",
    );
    let before = hash(&d, &inputs(vec![src.clone()])).unwrap();
    write(&d.join("common/dead.cuh"), "#define D 2\n");
    assert_ne!(
        before,
        hash(&d, &inputs(vec![src])).unwrap(),
        "an untaken branch is over-included, which is the safe direction"
    );
}

/// Angle-bracket includes are toolchain headers; they are covered by the
/// recorded compiler version, not by content, and must not be chased.
#[test]
fn angle_bracket_includes_are_not_followed() {
    let d = tmp();
    let src = d.join("common/x.cu");
    write(&src, "#include <cuda_fp16.h>\n__global__ void k() {}\n");
    assert!(hash(&d, &inputs(vec![src])).is_ok());
}

/// A commented-out include is not compiled, so hashing it would make a comment
/// edit look like a source change.
#[test]
fn commented_out_includes_are_ignored() {
    let d = tmp();
    let src = d.join("common/x.cu");
    write(&src, "// #include \"absent.cuh\"\n__global__ void k() {}\n");
    assert!(
        hash(&d, &inputs(vec![src])).is_ok(),
        "a commented include must not be resolved, let alone demanded"
    );
}

/// Mutually-including headers must terminate. Include guards make this normal.
#[test]
fn include_cycles_terminate() {
    let d = tmp();
    write(&d.join("common/a.cuh"), "#include \"b.cuh\"\n");
    write(&d.join("common/b.cuh"), "#include \"a.cuh\"\n");
    let src = d.join("common/x.cu");
    write(&src, "#include \"a.cuh\"\n");
    assert!(hash(&d, &inputs(vec![src])).is_ok());
}

/// Flags, arch and compiler change the emitted code without touching a source
/// byte, so each must move the hash on its own.
#[test]
fn non_source_inputs_each_move_the_hash() {
    let d = tmp();
    let src = d.join("common/x.cu");
    write(&src, "__global__ void k() {}\n");
    let base = hash(&d, &inputs(vec![src.clone()])).unwrap();

    let mut flags = inputs(vec![src.clone()]);
    flags.flags.push("-O3".into());
    assert_ne!(base, hash(&d, &flags).unwrap(), "nvcc flags");

    let mut arch = inputs(vec![src.clone()]);
    arch.arch = "sm_120a".into();
    assert_ne!(base, hash(&d, &arch).unwrap(), "arch");

    let mut cc = inputs(vec![src.clone()]);
    cc.compiler = "nvcc 12.9.0".into();
    assert_ne!(base, hash(&d, &cc).unwrap(), "compiler version");

    let mut cfg = inputs(vec![src.clone()]);
    let toml = d.join("model/MODEL.toml");
    write(&toml, "[behavior]\nthinking_default = true\n");
    cfg.configs.push(toml.clone());
    let with_cfg = hash(&d, &cfg).unwrap();
    assert_ne!(base, with_cfg, "config presence");
    write(&toml, "[behavior]\nthinking_default = false\n");
    assert_ne!(with_cfg, hash(&d, &cfg).unwrap(), "config CONTENT");
}

/// ★ Identical bytes at a different stem is a different compile.
///
/// The stem decides which common file a source shadows, so content alone is not
/// the identity — the name is hashed too.
#[test]
fn identical_content_under_a_different_name_hashes_differently() {
    let d = tmp();
    let a = d.join("common/one.cu");
    let b = d.join("common/two.cu");
    write(&a, "__global__ void k() {}\n");
    write(&b, "__global__ void k() {}\n");
    assert_ne!(
        hash(&d, &inputs(vec![a])).unwrap(),
        hash(&d, &inputs(vec![b])).unwrap()
    );
}

/// Deterministic: same inputs, same digest, regardless of the order sources are
/// listed in. Without this two machines would disagree about the same commit.
#[test]
fn the_hash_is_order_independent_and_repeatable() {
    let d = tmp();
    let a = d.join("common/a.cu");
    let b = d.join("common/b.cu");
    write(&a, "__global__ void a() {}\n");
    write(&b, "__global__ void b() {}\n");

    let one = hash(&d, &inputs(vec![a.clone(), b.clone()])).unwrap();
    let two = hash(&d, &inputs(vec![b, a])).unwrap();
    assert_eq!(one, two, "source order must not matter");
    assert_eq!(one.len(), 64, "sha256 hex");
}

/// The digest is over repo-RELATIVE paths, so two checkouts of one commit in
/// different directories agree.
#[test]
fn the_hash_does_not_depend_on_the_checkout_location() {
    let d1 = tmp();
    let d2 = d1.parent().unwrap().join(format!(
        "{}-elsewhere",
        d1.file_name().unwrap().to_string_lossy()
    ));
    let _ = std::fs::remove_dir_all(&d2);
    std::fs::create_dir_all(d2.join("common")).unwrap();

    write(&d1.join("common/x.cu"), "__global__ void k() {}\n");
    write(&d2.join("common/x.cu"), "__global__ void k() {}\n");
    write(&d1.join("model/MODEL.toml"), "[model]\nname = \"same\"\n");
    std::fs::create_dir_all(d2.join("model")).unwrap();
    write(&d2.join("model/MODEL.toml"), "[model]\nname = \"same\"\n");

    // Use a lexical alias for one checkout root. `expand` canonicalizes source
    // files, so the root must be normalized to the same form before paths are
    // made relative. This reproduces aliases such as macOS `/var` ->
    // `/private/var` without depending on the host platform.
    let aliased_d1 = d1.join("model/..");
    let mut first = inputs(vec![aliased_d1.join("common/x.cu")]);
    first.configs.push(aliased_d1.join("model/MODEL.toml"));
    let mut second = inputs(vec![d2.join("common/x.cu")]);
    second.configs.push(d2.join("model/MODEL.toml"));

    assert_eq!(
        hash(&aliased_d1, &first).unwrap(),
        hash(&d2, &second).unwrap(),
        "the same commit checked out twice must hash the same"
    );
}
