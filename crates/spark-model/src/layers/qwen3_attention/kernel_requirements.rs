// SPDX-License-Identifier: AGPL-3.0-only

//! Startup requirement table for `KvCacheDtype` — the optional kernel
//! handles each dtype's dispatch arms cannot run without.
//!
//! Sibling of `init_kernel_dispatch.rs` (which routes the hard-required
//! reshape/decode pair): this file covers the `try_kernel` optional-handle
//! class — the chunked-prefill paged-attention kernel for the dtype and the
//! WHT rotation bookends for turbo dtypes — so `spark serve` can fail at
//! startup with the full missing list instead of at first dispatch.

use spark_runtime::kv_cache::KvCacheDtype;

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
    if (k_dtype.is_wht_rotated() || v_dtype.is_wht_rotated()) && matches!(head_dim, 128 | 256 | 512)
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
            (
                KvCacheDtype::Turbo4KTurbo3V,
                "prefill_paged_turbo4k_turbo3v",
            ),
            (
                KvCacheDtype::Turbo4KTurbo8V,
                "prefill_paged_turbo4k_turbo8v",
            ),
            (
                KvCacheDtype::Turbo3KTurbo8V,
                "prefill_paged_turbo3k_turbo8v",
            ),
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
}
