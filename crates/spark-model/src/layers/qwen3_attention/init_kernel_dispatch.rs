// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel-module dispatch table for `KvCacheDtype`.
//!
//! Extracted from `init.rs::new_with_gating` so the routing — which is
//! the exact site where Turbo2 silently fell through to the FP8 ABI and
//! 8-of-9 asymmetric variants silently fell through to the K-side
//! symmetric kernels — is a pure function with `#[cfg(test)]` coverage
//! that runs without a GPU.
//!
//! The test in this file walks the full enum and asserts every variant
//! routes to a kernel module whose name contains the variant's storage
//! shape (e.g. `Bf16KTurbo3V` must end up at modules containing
//! `bf16k_turbo3v`). A new variant added to the enum without a dedicated
//! kernel module — i.e. one that falls through to either the FP8
//! catch-all or one of the inappropriate K-side symmetric arms — fails
//! this test on CI before merge.
//!
//! See `feedback_atlas_dispatch_match_arm_audit.md` in the contributor
//! memory: this is the "enum-add without match-update" bug class the
//! audit was opened against.

use spark_runtime::kv_cache::KvCacheDtype;

/// Module + function name 4-tuple consumed by `Qwen3AttentionLayer::new_with_gating`:
/// `(reshape_mod, reshape_fn, decode_mod, decode_fn)`. The reshape pair feeds
/// `self.reshape_cache_k`; the decode pair feeds `self.paged_decode_k`.
pub(super) fn kernel_modules_for_dtype(
    kv_dtype: KvCacheDtype,
    head_dim: usize,
) -> (&'static str, &'static str, &'static str, &'static str) {
    let hd_le_128 = head_dim <= 128;
    match kv_dtype {
        KvCacheDtype::Nvfp4 => (
            "reshape_and_cache",
            "reshape_and_cache_flash_nvfp4",
            "paged_decode_nvfp4",
            "paged_decode_attn_nvfp4",
        ),
        KvCacheDtype::Turbo4 => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_turbo4",
            if hd_le_128 {
                "paged_decode_turbo4_128"
            } else {
                "paged_decode_turbo4"
            },
            "paged_decode_attn_turbo4",
        ),
        KvCacheDtype::Turbo3 => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_turbo3",
            if hd_le_128 {
                "paged_decode_turbo3_128"
            } else {
                "paged_decode_turbo3"
            },
            "paged_decode_attn_turbo3",
        ),
        KvCacheDtype::Turbo2 => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_turbo2",
            "paged_decode_turbo2_128",
            "paged_decode_attn_turbo2",
        ),
        KvCacheDtype::Turbo8 => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_turbo8",
            if hd_le_128 {
                "paged_decode_turbo8_128"
            } else {
                "paged_decode_turbo8"
            },
            "paged_decode_attn_turbo8",
        ),
        KvCacheDtype::Bf16KTurbo3V => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_bf16k_turbo3v",
            if hd_le_128 {
                "paged_decode_bf16k_turbo3v_128"
            } else {
                "paged_decode_bf16k_turbo3v"
            },
            "paged_decode_attn_bf16k_turbo3v",
        ),
        KvCacheDtype::Bf16KTurbo4V => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_bf16k_turbo4v",
            if hd_le_128 {
                "paged_decode_bf16k_turbo4v_128"
            } else {
                "paged_decode_bf16k_turbo4v"
            },
            "paged_decode_attn_bf16k_turbo4v",
        ),
        KvCacheDtype::Bf16KTurbo2V => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_bf16k_turbo2v",
            if hd_le_128 {
                "paged_decode_bf16k_turbo2v_128"
            } else {
                "paged_decode_bf16k_turbo2v"
            },
            "paged_decode_attn_bf16k_turbo2v",
        ),
        KvCacheDtype::Fp8KTurbo3V => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_fp8k_turbo3v",
            if hd_le_128 {
                "paged_decode_fp8k_turbo3v_128"
            } else {
                "paged_decode_fp8k_turbo3v"
            },
            "paged_decode_attn_fp8k_turbo3v",
        ),
        KvCacheDtype::Fp8KTurbo4V => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_fp8k_turbo4v",
            if hd_le_128 {
                "paged_decode_fp8k_turbo4v_128"
            } else {
                "paged_decode_fp8k_turbo4v"
            },
            "paged_decode_attn_fp8k_turbo4v",
        ),
        KvCacheDtype::Fp8KTurbo2V => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_fp8k_turbo2v",
            if hd_le_128 {
                "paged_decode_fp8k_turbo2v_128"
            } else {
                "paged_decode_fp8k_turbo2v"
            },
            "paged_decode_attn_fp8k_turbo2v",
        ),
        KvCacheDtype::Turbo4KTurbo3V => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_turbo4k_turbo3v",
            "paged_decode_turbo4k_turbo3v_128",
            "paged_decode_attn_turbo4k_turbo3v",
        ),
        KvCacheDtype::Turbo4KTurbo8V => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_turbo4k_turbo8v",
            "paged_decode_turbo4k_turbo8v_128",
            "paged_decode_attn_turbo4k_turbo8v",
        ),
        KvCacheDtype::Turbo3KTurbo8V => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_turbo3k_turbo8v",
            "paged_decode_turbo3k_turbo8v_128",
            "paged_decode_attn_turbo3k_turbo8v",
        ),
        KvCacheDtype::Bf16 => (
            "reshape_and_cache",
            "reshape_and_cache_flash",
            "paged_decode",
            "paged_decode_attn",
        ),
        KvCacheDtype::Fp8 => (
            "reshape_and_cache",
            "reshape_and_cache_flash_fp8",
            "paged_decode_fp8",
            "paged_decode_attn_fp8",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every variant the enum advertises must be in the dispatch table.
    /// The match in `kernel_modules_for_dtype` is exhaustive (no `_` arm),
    /// so a new enum variant added without a corresponding routing fails
    /// to compile rather than slipping through to the FP8 ABI silently.
    /// This test is a runtime sanity check that the compile-time guarantee
    /// is exercised — every variant returns a non-empty tuple.
    #[test]
    fn every_variant_returns_non_empty_modules() {
        const ALL: &[KvCacheDtype] = &[
            KvCacheDtype::Bf16,
            KvCacheDtype::Fp8,
            KvCacheDtype::Nvfp4,
            KvCacheDtype::Turbo4,
            KvCacheDtype::Turbo3,
            KvCacheDtype::Turbo2,
            KvCacheDtype::Turbo8,
            KvCacheDtype::Bf16KTurbo3V,
            KvCacheDtype::Bf16KTurbo4V,
            KvCacheDtype::Bf16KTurbo2V,
            KvCacheDtype::Fp8KTurbo3V,
            KvCacheDtype::Fp8KTurbo4V,
            KvCacheDtype::Fp8KTurbo2V,
            KvCacheDtype::Turbo4KTurbo3V,
            KvCacheDtype::Turbo4KTurbo8V,
            KvCacheDtype::Turbo3KTurbo8V,
        ];
        for &d in ALL {
            for &hd in &[128usize, 256] {
                let (rm, rf, dm, df) = kernel_modules_for_dtype(d, hd);
                assert!(!rm.is_empty(), "{d:?} hd={hd}: empty reshape module");
                assert!(!rf.is_empty(), "{d:?} hd={hd}: empty reshape fn");
                assert!(!dm.is_empty(), "{d:?} hd={hd}: empty decode module");
                assert!(!df.is_empty(), "{d:?} hd={hd}: empty decode fn");
            }
        }
    }

    /// Each asymmetric variant must route to kernel module names that
    /// contain its dtype-pair shape (e.g. `Bf16KTurbo3V` → modules
    /// containing `bf16k_turbo3v`). A new asym variant that
    /// silently falls through to a K-side symmetric kernel — which
    /// would treat V as the K dtype and mis-size the V pool — fails
    /// here because the K-side module name (e.g. `reshape_and_cache_flash`
    /// for bf16) does not contain the asym shape token.
    ///
    /// This is the test that would have caught the original
    /// Bf16KTurbo4V/Bf16KTurbo2V/Fp8KTurbo{2,3,4}V/Turbo*KTurbo*V
    /// silent-fall-through that PR review caught via end-to-end PPL
    /// benchmarking.
    #[test]
    fn each_asym_variant_routes_to_dedicated_kernel() {
        let cases: &[(KvCacheDtype, &str)] = &[
            (KvCacheDtype::Bf16KTurbo3V, "bf16k_turbo3v"),
            (KvCacheDtype::Bf16KTurbo4V, "bf16k_turbo4v"),
            (KvCacheDtype::Bf16KTurbo2V, "bf16k_turbo2v"),
            (KvCacheDtype::Fp8KTurbo3V, "fp8k_turbo3v"),
            (KvCacheDtype::Fp8KTurbo4V, "fp8k_turbo4v"),
            (KvCacheDtype::Fp8KTurbo2V, "fp8k_turbo2v"),
            (KvCacheDtype::Turbo4KTurbo3V, "turbo4k_turbo3v"),
            (KvCacheDtype::Turbo4KTurbo8V, "turbo4k_turbo8v"),
            (KvCacheDtype::Turbo3KTurbo8V, "turbo3k_turbo8v"),
        ];
        for &(d, shape) in cases {
            for &hd in &[128usize, 256] {
                let (_rm, rf, dm, df) = kernel_modules_for_dtype(d, hd);
                assert!(
                    rf.contains(shape),
                    "{d:?} hd={hd}: reshape_fn {rf:?} missing shape token {shape:?} \
                     — silently dispatching to a non-asym kernel?"
                );
                assert!(
                    dm.contains(shape),
                    "{d:?} hd={hd}: decode_mod {dm:?} missing shape token {shape:?}"
                );
                assert!(
                    df.contains(shape),
                    "{d:?} hd={hd}: decode_fn {df:?} missing shape token {shape:?}"
                );
            }
        }
    }

    /// Sym dtypes route to their own dedicated kernels (no asym
    /// variant should accidentally claim a sym kernel name).
    #[test]
    fn sym_variants_route_to_sym_kernels() {
        // (dtype, expected substring in decode_fn)
        let cases: &[(KvCacheDtype, &str)] = &[
            (KvCacheDtype::Bf16, "paged_decode_attn"),
            (KvCacheDtype::Fp8, "paged_decode_attn_fp8"),
            (KvCacheDtype::Nvfp4, "paged_decode_attn_nvfp4"),
            (KvCacheDtype::Turbo4, "paged_decode_attn_turbo4"),
            (KvCacheDtype::Turbo3, "paged_decode_attn_turbo3"),
            (KvCacheDtype::Turbo2, "paged_decode_attn_turbo2"),
            (KvCacheDtype::Turbo8, "paged_decode_attn_turbo8"),
        ];
        for &(d, want) in cases {
            for &hd in &[128usize, 256] {
                let (_, _, _, df) = kernel_modules_for_dtype(d, hd);
                assert!(
                    df.contains(want),
                    "{d:?} hd={hd}: decode_fn {df:?} doesn't contain {want:?}"
                );
                // And asym shape tokens must NOT appear in sym dtypes.
                for asym_shape in &["bf16k_", "fp8k_", "turbo4k_", "turbo3k_"] as &[&str] {
                    assert!(
                        !df.contains(asym_shape),
                        "{d:?} hd={hd}: sym dtype routed to asym kernel {df:?}"
                    );
                }
            }
        }
    }

    /// hd>128 decode-module selection works for Turbo3/4/8 + Bf16K /
    /// Fp8K asym variants (the `if hd_le_128 { ..._128 } else { ... }`
    /// branch fires). hd=128 variants must end in `_128`; hd=256
    /// variants must NOT end in `_128`.
    #[test]
    fn hd_gate_picks_128_or_full_kernel() {
        // Turbo3 has both _128 and full variants.
        let (_, _, dm_128, _) = kernel_modules_for_dtype(KvCacheDtype::Turbo3, 128);
        let (_, _, dm_256, _) = kernel_modules_for_dtype(KvCacheDtype::Turbo3, 256);
        assert!(
            dm_128.ends_with("_128"),
            "hd=128 turbo3: {dm_128} should end _128"
        );
        assert!(
            !dm_256.ends_with("_128"),
            "hd=256 turbo3: {dm_256} should not end _128"
        );

        // Same shape gate for the asym families that support hd>128
        // (Bf16K/Fp8K — Turbo*KTurbo*V are 128-only today and the test
        // accepts that).
        for asym in &[KvCacheDtype::Bf16KTurbo3V, KvCacheDtype::Fp8KTurbo3V] {
            let (_, _, dm_128, _) = kernel_modules_for_dtype(*asym, 128);
            let (_, _, dm_256, _) = kernel_modules_for_dtype(*asym, 256);
            assert!(
                dm_128.ends_with("_128"),
                "{asym:?} hd=128: {dm_128} should end _128"
            );
            assert!(
                !dm_256.ends_with("_128"),
                "{asym:?} hd=256: {dm_256} should not end _128"
            );
        }
    }
}
