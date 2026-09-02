// SPDX-License-Identifier: AGPL-3.0-only

//! The K (bits/weight) envelope of the compiled `exl3_matmul` module and the
//! load-time probe name list — split from `exl3_matmul.rs` for the 500-LoC
//! cap. ONE definition of "which K has instances" for every native arm
//! (lm_head, GDN/attention dense, the MoE decode tier); the weight_map
//! keep-predicates name these constants rather than mirroring the sets.

use anyhow::Result;

use super::{ensure_k_cb, resolve_gemm_shape};

/// The K (bits/weight) values with compiled gemm + mgemm instances
/// (`exl3_matmul.cu`, `EXL3_GEMM_SET`). K=1 and K=7 are NOT instantiated:
/// no shipped Qwen3.8-Flash-Next branch uses K=1, and K=7 (5.05bpw dense,
/// 6.05bpw `mtp.fc`) stays excluded until a kernel exists for it.
pub const EXL3_GEMM_K_BITS: [u32; 6] = [2, 3, 4, 5, 6, 8];

/// `true` when gemm/mgemm instances exist for this K ([`EXL3_GEMM_K_BITS`]).
pub fn exl3_gemm_serves_k(k_bits: u32) -> bool {
    EXL3_GEMM_K_BITS.contains(&k_bits)
}

/// `true` when the small-row GEMV tier has instances for this K
/// (`EXL3_GEMV_SET` is instantiated for K in 2..=4 only). A `rows <= 8`
/// projection at any other K goes straight to the f32-C GEMM — the callers
/// (`exl3_dense_linear`, `Exl3LmHead`) skip the GEMV explicitly, and
/// `exl3_gemv` itself returns `Ok(false)` rather than asking the module for
/// a name that does not exist.
pub fn exl3_gemv_serves_k(k_bits: u32) -> bool {
    (2..=4).contains(&k_bits)
}

/// Every `exl3_matmul` instance name a dense `[k -> n]` projection at
/// `(k_bits, cb)` can dispatch to through `exl3_gemm` / `exl3_gemv`: the
/// GEMM shape the Blackwell heuristic resolves for this geometry (f16 and
/// f32 C), the universal shape-2 fallback, and — only for the K the GEMV tier
/// is instantiated at — the four GEMV configs x two C dtypes. Load-time
/// probes resolve exactly this list so a missing module / JIT failure / name
/// rule drift is paid at load, never on the first request. Errors when the
/// geometry has no compatible GEMM shape (n % 128 != 0) or K has no
/// instances.
pub fn exl3_dense_kernel_names(k: usize, n: usize, k_bits: u32, cb: u32) -> Result<Vec<String>> {
    ensure_k_cb(k_bits, cb)?;
    let picked = resolve_gemm_shape(k, n, k_bits, false, 1, 1, None)?;
    let mut names = Vec::new();
    let mut shapes = vec![picked];
    if picked != 2 {
        shapes.push(2);
    }
    for shape in shapes {
        for suf in ["f16", "f32"] {
            names.push(format!("exl3_gemm_k{k_bits}_cb{cb}_sh{shape}_{suf}"));
        }
    }
    if exl3_gemv_serves_k(k_bits) {
        for mmode in [0, 1] {
            for cfg in [0, 1] {
                for suf in ["f16", "f32"] {
                    names.push(format!(
                        "exl3_gemv_k{k_bits}_cb{cb}_m{mmode}_cfg{cfg}_{suf}"
                    ));
                }
            }
        }
    }
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_envelopes_match_the_instantiated_sets() {
        for k in [2, 3, 4, 5, 6, 8] {
            assert!(exl3_gemm_serves_k(k), "K={k} has gemm/mgemm instances");
            assert!(ensure_k_cb(k, 2).is_ok());
        }
        for k in [0, 1, 7, 9] {
            assert!(!exl3_gemm_serves_k(k));
            assert!(ensure_k_cb(k, 2).is_err());
        }
        // GEMV is instantiated for K in 2..=4 only.
        for k in [2, 3, 4] {
            assert!(exl3_gemv_serves_k(k));
        }
        for k in [5, 6, 8] {
            assert!(!exl3_gemv_serves_k(k), "K={k} must skip the GEMV tier");
        }
        assert!(ensure_k_cb(4, 0).is_err(), "cb0 has no instances");
    }

    #[test]
    fn dense_kernel_names_follow_the_dispatch_rule() {
        // lm_head geometry at K=6: heuristic -> shape 4 (n % 512, n > 16384),
        // plus the universal shape-2 fallback; NO gemv names at K=6.
        let names = exl3_dense_kernel_names(2560, 248320, 6, 2).unwrap();
        assert_eq!(
            names,
            vec![
                "exl3_gemm_k6_cb2_sh4_f16",
                "exl3_gemm_k6_cb2_sh4_f32",
                "exl3_gemm_k6_cb2_sh2_f16",
                "exl3_gemm_k6_cb2_sh2_f32",
            ]
        );
        // GDN qkv at K=3: shape 3 + shape 2 + the eight gemv instances.
        let names = exl3_dense_kernel_names(2560, 10240, 3, 2).unwrap();
        assert_eq!(names.len(), 4 + 8);
        assert_eq!(names[0], "exl3_gemm_k3_cb2_sh3_f16");
        assert_eq!(names[2], "exl3_gemm_k3_cb2_sh2_f16");
        assert!(
            names[4..]
                .iter()
                .all(|n| n.starts_with("exl3_gemv_k3_cb2_m"))
        );
        // attention k/v at K=6: heuristic itself lands on shape 2 -> two
        // names, no duplicates, no gemv.
        let names = exl3_dense_kernel_names(2560, 512, 6, 1).unwrap();
        assert_eq!(
            names,
            vec!["exl3_gemm_k6_cb1_sh2_f16", "exl3_gemm_k6_cb1_sh2_f32"]
        );
        // K=8 (6.05bpw dense): gemm-only; the K>=7 heuristic branch lands on
        // shape 2 for the GDN qkv width (n=10240 > 8192 and <= 32768).
        let names = exl3_dense_kernel_names(2560, 10240, 8, 2).unwrap();
        assert_eq!(
            names,
            vec!["exl3_gemm_k8_cb2_sh2_f16", "exl3_gemm_k8_cb2_sh2_f32"]
        );
        // K=7 has no instances; n % 128 != 0 has no shape.
        assert!(exl3_dense_kernel_names(2560, 10240, 7, 2).is_err());
        assert!(exl3_dense_kernel_names(2560, 10200, 4, 2).is_err());
    }
}
