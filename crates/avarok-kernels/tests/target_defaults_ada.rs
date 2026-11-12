// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the CHECKED-IN `kernels/ada/HARDWARE.toml` `[defaults]`
//! table, parsed with the real build-script parser.
//!
//! Moved out of `tests/target_defaults.rs` unchanged (that file had crossed
//! the 500-line cap); the companion tests and the reasons this is an
//! integration test rather than a build-script unit test are there.

#[allow(dead_code)]
#[path = "../build_defaults.rs"]
mod build_defaults;

use build_defaults::{Defaults, baseline, read_defaults, read_sm_count};

use std::path::PathBuf;

fn kernels_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/avarok-kernels is two levels below the workspace root")
        .join("kernels")
}

fn declared(hw: &str) -> Defaults {
    read_defaults(&kernels_root(), hw)
}

/// Ada declares the conservative table apart from the one row it has an
/// end-to-end receipt for: the BF16 dense-FFN prefill on cuBLASLt (RTX 4060 Ti,
/// 3,921-token prompt: 0.8B 8,199 -> 13,113 tok/s, 4B 795 -> 3,677 tok/s). No
/// Hopper or GB10 row arrives by resemblance. Its SM count is the reference
/// part's, the RTX 4060 Ti 16 GB (AD106, 34 SMs).
#[test]
fn ada_declares_conservative_defaults_apart_from_its_measured_rows() {
    let d = declared("ada");
    assert!(d.ffn_bf16_cublas_prefill, "the measured ada prefill win");
    assert!(
        d.attn_gated_bf16_batchm,
        "bit-identical on ada: same kernel source, --fmad=false"
    );
    assert!(d.memory_fit_consumer, "the 16 GB consumer-card memory fits");
    let normalised = Defaults {
        ffn_bf16_cublas_prefill: false,
        attn_gated_bf16_batchm: false,
        memory_fit_consumer: false,
        ..d
    };
    assert_eq!(normalised, baseline("ada"));
    assert_eq!(read_sm_count(&kernels_root(), "ada"), 34);
    for hw in ["gb10", "hopper", "b200", "b300"] {
        assert!(!declared(hw).ffn_bf16_cublas_prefill, "{hw}");
        assert!(!declared(hw).attn_gated_bf16_batchm, "{hw}");
        assert!(!declared(hw).memory_fit_consumer, "{hw}");
    }
    // The GDN verify GEMM row has no receipt on any target, ada included.
    for hw in ["ada", "gb10", "hopper", "b200", "b300"] {
        assert!(!declared(hw).gdn_batched_gemm, "{hw}");
    }
}
