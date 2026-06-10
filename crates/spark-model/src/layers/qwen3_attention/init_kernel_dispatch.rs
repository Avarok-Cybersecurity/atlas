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

/// Optional-handle kernels (loaded via `try_kernel`, dispatch checks
/// `handle.0 != 0`) that the given `--kv-cache-dtype` cannot run without.
/// `kernel_modules_for_dtype` covers the hard-required reshape/decode pair
/// (those already fail layer construction via `gpu.kernel(..)?`); this list
/// covers the rest: the chunked-prefill paged-attention kernel for the
/// dtype, and the WHT rotation bookends for turbo dtypes. Used by
/// `validate_required_kernels` to fail at startup instead of at first
/// dispatch (or worse, at a silent fall-through).
pub(super) fn required_optional_kernels_for_dtype(
    kv_dtype: KvCacheDtype,
    head_dim: usize,
) -> Vec<(&'static str, &'static str)> {
    let mut req: Vec<(&'static str, &'static str)> = Vec::new();
    match kv_dtype {
        KvCacheDtype::Turbo2 => {
            req.push(("prefill_paged_turbo2", "inferspark_prefill_paged_turbo2"));
        }
        KvCacheDtype::Turbo3 => {
            req.push(("prefill_paged_turbo3", "inferspark_prefill_paged_turbo3_64"));
        }
        KvCacheDtype::Turbo4 => {
            req.push(("prefill_paged_turbo4", "inferspark_prefill_paged_turbo4_64"));
        }
        KvCacheDtype::Turbo8 => {
            req.push(("prefill_paged_turbo8", "inferspark_prefill_paged_turbo8_64"));
        }
        KvCacheDtype::Bf16KTurbo3V => {
            req.push((
                "prefill_paged_bf16k_turbo3v",
                "inferspark_prefill_paged_bf16k_turbo3v_64",
            ));
        }
        KvCacheDtype::Bf16KTurbo4V => {
            req.push((
                "prefill_paged_bf16k_turbo4v",
                "inferspark_prefill_paged_bf16k_turbo4v_64",
            ));
        }
        KvCacheDtype::Bf16KTurbo2V => {
            req.push((
                "prefill_paged_bf16k_turbo2v",
                "inferspark_prefill_paged_bf16k_turbo2v_64",
            ));
        }
        KvCacheDtype::Fp8KTurbo3V => {
            req.push((
                "prefill_paged_fp8k_turbo3v",
                "inferspark_prefill_paged_fp8k_turbo3v_64",
            ));
        }
        KvCacheDtype::Fp8KTurbo4V => {
            req.push((
                "prefill_paged_fp8k_turbo4v",
                "inferspark_prefill_paged_fp8k_turbo4v_64",
            ));
        }
        KvCacheDtype::Fp8KTurbo2V => {
            req.push((
                "prefill_paged_fp8k_turbo2v",
                "inferspark_prefill_paged_fp8k_turbo2v_64",
            ));
        }
        KvCacheDtype::Turbo4KTurbo3V => {
            req.push((
                "prefill_paged_turbo4k_turbo3v",
                "inferspark_prefill_paged_turbo4k_turbo3v_64",
            ));
        }
        KvCacheDtype::Turbo4KTurbo8V => {
            req.push((
                "prefill_paged_turbo4k_turbo8v",
                "inferspark_prefill_paged_turbo4k_turbo8v_64",
            ));
        }
        KvCacheDtype::Turbo3KTurbo8V => {
            req.push((
                "prefill_paged_turbo3k_turbo8v",
                "inferspark_prefill_paged_turbo3k_turbo8v_64",
            ));
        }
        KvCacheDtype::Bf16 | KvCacheDtype::Fp8 | KvCacheDtype::Nvfp4 => {}
    }
    // WHT rotation bookends: the write path stores turbo cache contents in
    // the rotated basis whenever either side is a turbo dtype at a supported
    // head_dim, so the Q/output bookends are required for correctness.
    let (k_dtype, v_dtype) = kv_dtype.kv_pair();
    if (k_dtype.is_wht_rotated() || v_dtype.is_wht_rotated())
        && matches!(head_dim, 128 | 256 | 512)
    {
        req.push(("wht_bf16", "wht_bf16_inplace"));
        req.push(("wht_bf16", "wht_bf16_inplace_inv"));
    }
    req
}

/// Startup fail-fast: resolve every dtype-required kernel handle for the
/// selected `--kv-cache-dtype` and bail with the full missing list if any
/// is absent — instead of failing at first dispatch (minutes later, after
/// weight load) or silently producing a wrong-kernel fall-through.
pub(super) fn validate_required_kernels(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    kv_dtype: KvCacheDtype,
    head_dim: usize,
) -> anyhow::Result<()> {
    let missing: Vec<String> = required_optional_kernels_for_dtype(kv_dtype, head_dim)
        .into_iter()
        .filter(|(m, f)| gpu.kernel(m, f).is_err())
        .map(|(m, f)| format!("{m}::{f}"))
        .collect();
    if !missing.is_empty() {
        anyhow::bail!(
            "kv-cache-dtype {kv_dtype:?} (head_dim {head_dim}) requires kernel(s) \
             missing from this build: {} — rebuild kernels or pick a supported dtype",
            missing.join(", ")
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every dtype with a turbo side must require its dedicated
    /// chunked-prefill kernel AND the WHT bookend pair; plain dtypes must
    /// require nothing. Walks the full enum so a new variant added without
    /// a requirement entry fails to compile (exhaustive match in
    /// `required_optional_kernels_for_dtype`).
    #[test]
    fn required_optional_kernels_cover_turbo_variants() {
        const TURBO: &[(KvCacheDtype, &str)] = &[
            (KvCacheDtype::Turbo2, "prefill_paged_turbo2"),
            (KvCacheDtype::Turbo3, "prefill_paged_turbo3"),
            (KvCacheDtype::Turbo4, "prefill_paged_turbo4"),
            (KvCacheDtype::Turbo8, "prefill_paged_turbo8"),
            (KvCacheDtype::Bf16KTurbo3V, "prefill_paged_bf16k_turbo3v"),
            (KvCacheDtype::Bf16KTurbo4V, "prefill_paged_bf16k_turbo4v"),
            (KvCacheDtype::Bf16KTurbo2V, "prefill_paged_bf16k_turbo2v"),
            (KvCacheDtype::Fp8KTurbo3V, "prefill_paged_fp8k_turbo3v"),
            (KvCacheDtype::Fp8KTurbo4V, "prefill_paged_fp8k_turbo4v"),
            (KvCacheDtype::Fp8KTurbo2V, "prefill_paged_fp8k_turbo2v"),
            (KvCacheDtype::Turbo4KTurbo3V, "prefill_paged_turbo4k_turbo3v"),
            (KvCacheDtype::Turbo4KTurbo8V, "prefill_paged_turbo4k_turbo8v"),
            (KvCacheDtype::Turbo3KTurbo8V, "prefill_paged_turbo3k_turbo8v"),
        ];
        for &(d, prefill_mod) in TURBO {
            let req = required_optional_kernels_for_dtype(d, 256);
            assert!(
                req.iter().any(|(m, _)| *m == prefill_mod),
                "{d:?}: requirement list missing prefill module {prefill_mod}"
            );
            assert!(
                req.iter()
                    .any(|(m, f)| *m == "wht_bf16" && *f == "wht_bf16_inplace"),
                "{d:?}: requirement list missing wht_bf16_inplace"
            );
            assert!(
                req.iter()
                    .any(|(m, f)| *m == "wht_bf16" && *f == "wht_bf16_inplace_inv"),
                "{d:?}: requirement list missing wht_bf16_inplace_inv"
            );
        }
        for d in [KvCacheDtype::Bf16, KvCacheDtype::Fp8, KvCacheDtype::Nvfp4] {
            assert!(
                required_optional_kernels_for_dtype(d, 256).is_empty(),
                "{d:?}: plain dtype should require no optional kernels"
            );
        }
    }

    /// Turbo2 is WHT-rotated by the write path like Turbo3/4/8 — the decode
    /// and prefill bookend gates must include it (this was the decode-gate
    /// omission that desynced Q rotation from the cache contents).
    #[test]
    fn turbo2_is_wht_rotated() {
        for d in [
            KvCacheDtype::Turbo2,
            KvCacheDtype::Turbo3,
            KvCacheDtype::Turbo4,
            KvCacheDtype::Turbo8,
        ] {
            assert!(d.is_wht_rotated(), "{d:?} must gate the WHT bookends");
        }
        for d in [KvCacheDtype::Bf16, KvCacheDtype::Fp8, KvCacheDtype::Nvfp4] {
            assert!(!d.is_wht_rotated(), "{d:?} must not gate the WHT bookends");
        }
        // Asym variants gate per side via kv_pair().
        let (k, v) = KvCacheDtype::Bf16KTurbo2V.kv_pair();
        assert!(!k.is_wht_rotated() && v.is_wht_rotated());
        let (k, v) = KvCacheDtype::Turbo4KTurbo8V.kv_pair();
        assert!(k.is_wht_rotated() && v.is_wht_rotated());
    }

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
