// SPDX-License-Identifier: AGPL-3.0-only

// Build script for atlas-rdma. It compiles the one-sided RDMA verbs C shim
// and decides `cfg(atlas_rdma_verbs)` for the whole workspace:
//
//   * cfg ON  ⇔ target_os != macos AND !skip_build()   (unchanged semantics)
//   * skip_build() honours ATLAS_SKIP_BUILD / SKIP_ATLAS_BUILD ∈ {1,true,TRUE}
//     — the CPU/CI convention for hosts without rdma-core dev headers. There
//     is deliberately NO header probing: on Linux with the skip unset and
//     headers missing, cc fails the build loudly.
//
// Cross-crate propagation: `cargo:rustc-cfg` does NOT cross crate boundaries,
// so when the shim builds we also publish `cargo:has_verbs=1`. Thanks to the
// `links = "atlas_rdma_shim"` key, cargo exposes that to the build scripts of
// DIRECT dependents as `DEP_ATLAS_RDMA_SHIM_HAS_VERBS`. A direct dependent's
// build.rs (e.g. spark-storage, in a follow-up PR) reads it and re-emits
// `cargo:rustc-cfg=atlas_rdma_verbs` for its own gated modules. Any crate that
// grows a real `#[cfg(atlas_rdma_verbs)]` gate needs its own DIRECT dep on
// atlas-rdma plus the same re-emit (DEP_ vars are invisible to transitive
// dependents).

fn main() {
    // Unconditional and FIRST (before any early return): the lint config must
    // know the cfg name even when the cfg is off — `unexpected_cfgs` is a hard
    // error under `[workspace.lints.rust] warnings = "deny"`. (This also fixes
    // the latent macOS ordering landmine the old spark-storage build.rs had,
    // where the macOS arm returned before the check-cfg line.)
    println!("cargo:rustc-check-cfg=cfg(atlas_rdma_verbs)");
    println!("cargo:rerun-if-env-changed=ATLAS_SKIP_BUILD");
    println!("cargo:rerun-if-env-changed=SKIP_ATLAS_BUILD");
    println!("cargo:rerun-if-changed=src/rdma_shim.c");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ATLAS_NO_RDMA");

    // RDMA-only opt-out: gate off the verbs shim (no libibverbs, no C compile)
    // WITHOUT the nvcc/CUDA side effects that ATLAS_SKIP_BUILD carries in
    // spark-storage. Downstream RDMA code compiles to its `not(atlas_rdma_verbs)`
    // stubs; the NVMe/host tiers are unaffected. (Proves the "NVMe-only, no RDMA"
    // build profile; candidate for a proper `rdma` cargo feature upstream.)
    if std::env::var("ATLAS_NO_RDMA").as_deref() == Ok("1") {
        return;
    }

    // Apple Silicon hosts have no rdma-core; the verbs module compiles out.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        return;
    }
    if skip_build() {
        return;
    }

    // Compile `src/rdma_shim.c` and link libibverbs. `cc` emits the
    // static-lib link directives (rustc-link-lib=static=atlas_rdma_shim +
    // link-search into OUR OUT_DIR); static/dylib link flags propagate to
    // final binaries transitively, so downstream crates need nothing.
    cc::Build::new()
        .file("src/rdma_shim.c")
        .opt_level(2)
        .warnings(true)
        .compile("atlas_rdma_shim");
    println!("cargo:rustc-link-lib=dylib=ibverbs");
    // Gates our own `verbs` module.
    println!("cargo:rustc-cfg=atlas_rdma_verbs");
    // The cross-crate signal: DEP_ATLAS_RDMA_SHIM_HAS_VERBS in direct
    // dependents' build scripts. Emitted only when ON (presence test
    // downstream — matches the boolean semantics the cfg always had).
    println!("cargo:has_verbs=1");
}

/// Verbatim spark-storage semantics: ATLAS_SKIP_BUILD / SKIP_ATLAS_BUILD in
/// {"1","true","TRUE"} suppresses the shim compile and the cfg.
fn skip_build() -> bool {
    let truthy = |key: &str| {
        matches!(
            std::env::var(key).ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE")
        )
    };
    truthy("ATLAS_SKIP_BUILD") || truthy("SKIP_ATLAS_BUILD")
}
