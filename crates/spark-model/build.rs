// SPDX-License-Identifier: AGPL-3.0-only

//! Forward the active target HARDWARE.toml `[hardware]` gating flags to this
//! crate as compile-time env vars (`cargo:rustc-env`).
//!
//! Why this exists: `atlas-kernels/build.rs` already parses the same
//! `HARDWARE.toml` and emits these vars, but `cargo:rustc-env` is
//! *crate-local* — it only reaches the crate that owns the `build.rs`. The
//! `option_env!("ATLAS_HW_*")` reads live in `spark-model` (e.g.
//! `weight_loader/qwen35_dense.rs`, `layers/qwen3_attention/prefill/paged_attn.rs`),
//! so without this they were always `None` and the gates were dead code.
//!
//! SSOT: the authoritative value lives only in `kernels/<hw>/HARDWARE.toml`.
//! This build script does not duplicate it — it resolves the same file (via
//! the same `ATLAS_TARGET_HW` env + `kernels/<hw>/` layout as atlas-kernels)
//! and re-emits the flags verbatim.
//!
//! PCND: a flag is emitted ONLY when the key is explicitly `true` in the
//! TOML. Absent key (e.g. NVIDIA/GB10) => no env emitted => `option_env!`
//! is `None` => unchanged behaviour. No implicit defaults.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=ATLAS_TARGET_HW");

    // Mirror atlas-kernels: workspace_root/kernels/<ATLAS_TARGET_HW|gb10>/HARDWARE.toml.
    // CARGO_MANIFEST_DIR = <workspace>/crates/spark-model.
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("CARGO_MANIFEST_DIR has <workspace>/crates/<crate> layout");
    let hw = env::var("ATLAS_TARGET_HW").unwrap_or_else(|_| "gb10".into());
    let hw_toml_path = workspace_root
        .join("kernels")
        .join(&hw)
        .join("HARDWARE.toml");
    println!("cargo:rerun-if-changed={}", hw_toml_path.display());

    // Missing HARDWARE.toml is not fatal here (atlas-kernels owns that error);
    // simply emit nothing so option_env! stays None.
    let Ok(text) = std::fs::read_to_string(&hw_toml_path) else {
        return;
    };
    let toml: toml::Value = match toml::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            // atlas-kernels build.rs panics on bad TOML; here we stay quiet
            // (it will fail there) but surface a warning for diagnosability.
            println!(
                "cargo:warning=spark-model: ignoring unparseable {}: {e}",
                hw_toml_path.display()
            );
            return;
        }
    };
    let Some(hardware) = toml.get("hardware") else {
        return;
    };

    // List of (TOML key, emitted env var). Additive: each is emitted only
    // when present-and-true. Keep in sync with the option_env! read sites.
    const FLAGS: &[(&str, &str)] = &[
        (
            "disable_fp8_ssm_prefill",
            "ATLAS_HW_DISABLE_FP8_SSM_PREFILL",
        ),
        ("force_br32_prefill", "ATLAS_HW_FORCE_BR32"),
    ];
    for (key, env_name) in FLAGS {
        if hardware
            .get(*key)
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            println!("cargo:rustc-env={env_name}=true");
        }
    }
}
