// SPDX-License-Identifier: AGPL-3.0-only

//! GEMM-path selection, resolved once and then **carried**.
//!
//! These flags used to be nine `OnceLock` statics that read `ATLAS_*` at first
//! touch. A static is the wrong home for them twice over:
//!
//! * **It outlives the model whose flags it encodes.** Swap to a model whose
//!   recipe sets different levers and the process keeps serving the previous
//!   model's dispatch decisions — silently, because a cached `bool` has no way
//!   to say it is stale.
//! * **It hides a dependency.** A function that reads the environment through a
//!   static takes no argument that says so, cannot be tested with a different
//!   configuration without mutating the process, and gives the compiler nothing
//!   to check.
//!
//! Carrying it on [`crate::layer::ForwardContext`] — which already reaches
//! every dispatch site — fixes both. The value is resolved once when the model
//! is built, borrowed for the duration of that model's run, and dropped with
//! it. If a future context is missed, the build fails; there is no runtime
//! check to forget.

/// Which GEMM implementation each projection takes.
///
/// Plain `Copy` data, resolved from the environment at model construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmDispatch {
    /// Block-scaled FP8 prefill (per-128-block weight scales + per-token
    /// activation scales). The DEFAULT for block-scaled FP8 checkpoints since
    /// 2026-06-17: it matches vLLM's per-block precision and avoids the
    /// single-scale path, whose collapse of per-block dynamic range pushed
    /// long-context tool-arg decode into the FP8 argmax-flip regime (B1 drift
    /// gauge ~1400 → ~100 once block-scaled prefill is on).
    /// Opt out with `ATLAS_FP8_SINGLE_SCALE=1` — diagnostic/fallback only.
    pub fp8_blockscaled_prefill: bool,
    /// cuBLASLt BF16 GEMM. The hand-written mma.sync projection GEMMs reach
    /// only ~30% of the cuBLAS bf16 ceiling on GB10.
    pub cublas_gemm: bool,
    /// Native-FP8 cuBLASLt GEMM.
    pub cublas_fp8: bool,
    /// CUTLASS BF16 GEMM, scoped to dense projections using the same FP8→BF16
    /// cached dequant as cuBLASLt.
    pub cutlass_gemm: bool,
    /// Native CUTLASS NVFP4 GEMM: quantizes activations to CUTLASS NVFP4 and
    /// consumes transposed Atlas NVFP4 weights after repacking scales into the
    /// CUTLASS SM120 layout. Implies every per-projection NVFP4 flag below.
    pub cutlass_nvfp4_gemm: bool,
    pub cutlass_nvfp4_qkvz: bool,
    pub cutlass_nvfp4_attn_q: bool,
    pub cutlass_nvfp4_attn_kv: bool,
    pub cutlass_nvfp4_attn_o: bool,
    pub cutlass_nvfp4_ssm_out: bool,
    /// `ATLAS_W4A16_VARIANT` — 1/2/3 pin a kernel variant, 0 = auto (v2).
    /// A dispatch decision like every other field here, so it belongs on the
    /// struct the forward pass already carries rather than in a `OnceLock`
    /// that would pin the first model's choice.
    pub w4a16_variant: u8,
}

fn from_values(mut value: impl FnMut(&str) -> Option<String>) -> GemmDispatch {
    fn on(value: &mut impl FnMut(&str) -> Option<String>, var: &str) -> bool {
        value(var).as_deref() == Some("1")
    }

    let all_nvfp4 = on(&mut value, "ATLAS_CUTLASS_NVFP4_GEMM");
    GemmDispatch {
        w4a16_variant: match value("ATLAS_W4A16_VARIANT").as_deref() {
            Some("v1") => 1,
            Some("v2") => 2,
            Some("v3") => 3,
            _ => 0,
        },
        fp8_blockscaled_prefill: !on(&mut value, "ATLAS_FP8_SINGLE_SCALE"),
        cublas_gemm: on(&mut value, "ATLAS_CUBLAS_GEMM"),
        cublas_fp8: on(&mut value, "ATLAS_CUBLAS_FP8"),
        cutlass_gemm: on(&mut value, "ATLAS_CUTLASS_GEMM"),
        cutlass_nvfp4_gemm: all_nvfp4,
        cutlass_nvfp4_qkvz: all_nvfp4 || on(&mut value, "ATLAS_CUTLASS_NVFP4_QKVZ"),
        cutlass_nvfp4_attn_q: all_nvfp4 || on(&mut value, "ATLAS_CUTLASS_NVFP4_ATTN_Q"),
        cutlass_nvfp4_attn_kv: all_nvfp4 || on(&mut value, "ATLAS_CUTLASS_NVFP4_ATTN_KV"),
        cutlass_nvfp4_attn_o: all_nvfp4 || on(&mut value, "ATLAS_CUTLASS_NVFP4_ATTN_O"),
        // Deliberately NOT implied by the umbrella flag.
        cutlass_nvfp4_ssm_out: on(&mut value, "ATLAS_CUTLASS_NVFP4_SSM_OUT"),
    }
}

impl GemmDispatch {
    /// Resolve from the environment. Called once, when the model is built.
    pub fn from_env() -> Self {
        from_values(|var| std::env::var(var).ok())
    }

    /// Everything off, block-scaled FP8 prefill on — the shape a build with no
    /// `ATLAS_*` set in the environment resolves to. Tests construct a context
    /// with this instead of mutating the process environment.
    pub fn defaults() -> Self {
        Self {
            w4a16_variant: 0,
            fp8_blockscaled_prefill: true,
            cublas_gemm: false,
            cublas_fp8: false,
            cutlass_gemm: false,
            cutlass_nvfp4_gemm: false,
            cutlass_nvfp4_qkvz: false,
            cutlass_nvfp4_attn_q: false,
            cutlass_nvfp4_attn_kv: false,
            cutlass_nvfp4_attn_o: false,
            cutlass_nvfp4_ssm_out: false,
        }
    }

    /// NVFP4 attention Q/K/V enabled for the named projection.
    pub fn cutlass_nvfp4_attn_qkv(&self, label: &str) -> bool {
        match label {
            "q_proj" => self.cutlass_nvfp4_attn_q,
            "k_proj" | "v_proj" => self.cutlass_nvfp4_attn_kv,
            _ => self.cutlass_nvfp4_gemm,
        }
    }
}

impl Default for GemmDispatch {
    fn default() -> Self {
        Self::defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn resolve(values: &[(&str, &str)]) -> GemmDispatch {
        let values: HashMap<_, _> = values.iter().copied().collect();
        from_values(|name| values.get(name).map(|value| (*value).to_owned()))
    }

    #[test]
    fn defaults_have_only_blockscaled_prefill_on() {
        let d = GemmDispatch::defaults();
        assert_eq!(
            resolve(&[]),
            d,
            "absent environment uses the public default"
        );
        assert_eq!(
            d,
            GemmDispatch {
                fp8_blockscaled_prefill: true,
                cublas_gemm: false,
                cublas_fp8: false,
                cutlass_gemm: false,
                cutlass_nvfp4_gemm: false,
                cutlass_nvfp4_qkvz: false,
                cutlass_nvfp4_attn_q: false,
                cutlass_nvfp4_attn_kv: false,
                cutlass_nvfp4_attn_o: false,
                cutlass_nvfp4_ssm_out: false,
                w4a16_variant: 0,
            }
        );
    }

    #[test]
    fn the_umbrella_flag_implies_the_per_projection_ones() {
        let d = resolve(&[("ATLAS_CUTLASS_NVFP4_GEMM", "1")]);
        assert!(d.cutlass_nvfp4_gemm);
        assert!(d.cutlass_nvfp4_qkvz);
        assert!(d.cutlass_nvfp4_attn_qkv("q_proj"));
        assert!(d.cutlass_nvfp4_attn_qkv("k_proj"));
        assert!(d.cutlass_nvfp4_attn_qkv("v_proj"));
        assert!(d.cutlass_nvfp4_attn_o);
        // SSM-out was never implied by the umbrella flag.
        assert!(!d.cutlass_nvfp4_ssm_out);
    }

    #[test]
    fn per_projection_flags_are_independent() {
        let cases = [
            (
                "ATLAS_CUTLASS_NVFP4_QKVZ",
                [true, false, false, false, false],
            ),
            (
                "ATLAS_CUTLASS_NVFP4_ATTN_Q",
                [false, true, false, false, false],
            ),
            (
                "ATLAS_CUTLASS_NVFP4_ATTN_KV",
                [false, false, true, false, false],
            ),
            (
                "ATLAS_CUTLASS_NVFP4_ATTN_O",
                [false, false, false, true, false],
            ),
            (
                "ATLAS_CUTLASS_NVFP4_SSM_OUT",
                [false, false, false, false, true],
            ),
        ];
        for (name, expected) in cases {
            let d = resolve(&[(name, "1")]);
            assert_eq!(
                [
                    d.cutlass_nvfp4_qkvz,
                    d.cutlass_nvfp4_attn_q,
                    d.cutlass_nvfp4_attn_kv,
                    d.cutlass_nvfp4_attn_o,
                    d.cutlass_nvfp4_ssm_out,
                ],
                expected,
                "{name} must not enable a neighboring projection"
            );
        }
    }

    #[test]
    fn non_nvfp4_flags_map_independently_and_single_scale_is_inverted() {
        let cases = [
            ("ATLAS_CUBLAS_GEMM", [true, false, false]),
            ("ATLAS_CUBLAS_FP8", [false, true, false]),
            ("ATLAS_CUTLASS_GEMM", [false, false, true]),
        ];
        for (name, expected) in cases {
            let d = resolve(&[(name, "1")]);
            assert_eq!(
                [d.cublas_gemm, d.cublas_fp8, d.cutlass_gemm],
                expected,
                "{name} must not enable a neighboring GEMM path"
            );
            assert!(d.fp8_blockscaled_prefill);
        }
        assert!(!resolve(&[("ATLAS_FP8_SINGLE_SCALE", "1")]).fp8_blockscaled_prefill);
        assert!(
            !resolve(&[("ATLAS_CUBLAS_GEMM", "true")]).cublas_gemm,
            "dispatch booleans accept only the documented literal 1"
        );
    }

    #[test]
    fn w4a16_variants_accept_only_documented_spellings() {
        for (value, expected) in [
            ("v1", 1),
            ("v2", 2),
            ("v3", 3),
            ("1", 0),
            ("V1", 0),
            ("unknown", 0),
        ] {
            assert_eq!(
                resolve(&[("ATLAS_W4A16_VARIANT", value)]).w4a16_variant,
                expected,
                "value {value}"
            );
        }
    }

    #[test]
    fn an_unknown_projection_label_falls_back_to_the_umbrella_flag() {
        assert!(!GemmDispatch::defaults().cutlass_nvfp4_attn_qkv("mystery"));
        let d = GemmDispatch {
            cutlass_nvfp4_gemm: true,
            ..GemmDispatch::defaults()
        };
        assert!(d.cutlass_nvfp4_attn_qkv("mystery"));
    }
}
