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

/// Which projection FAMILIES take a cuBLASLt GEMM arm.
///
/// WHY a set and not a `bool` — H100, 2026-09-11, `Qwen/Qwen3.8-27B-FP8`
/// native FP8, tip `5f78270dc`, "config B" of the round-3 receipt.
/// `ATLAS_CUBLAS_GEMM=1` resolved to ONE global boolean, so the variable that
/// arms #917's dense-FFN W8A8 fast path ALSO armed the SSM `in_proj_qkvz` arm,
/// which materialised a cached BF16 dequant of the fused QKVZ weight
/// (`[10240,5120] + [6144,5120]` x 2 B = `167772160` bytes per layer, ~10.3
/// GiB over 48 SSM layers) outside the buffer ledger. One 28-token prefill ate
/// 6120 MiB and died at layer 36 with `cuMemAlloc_v2 ... status 2`. The FFN arm
/// under test could not be exercised end-to-end on an 80 GB card, and no knob
/// separated the two. The CUTLASS family next door already spells its
/// projections out (`ATLAS_CUTLASS_NVFP4_QKVZ`, `..._ATTN_Q`, `..._ATTN_KV`,
/// `..._ATTN_O`, `..._SSM_OUT`); this gives cuBLASLt the same property in one
/// variable instead of five.
///
/// GRAMMAR — `ATLAS_CUBLAS_GEMM=<token>[,<token>]*`, ASCII-case-insensitive,
/// whitespace around a token ignored:
///
/// | token | meaning |
/// |---|---|
/// | `ffn` | dense-FFN + MoE shared-expert projections |
/// | `attn` | attention Q/K/V, O and the output gate |
/// | `ssm` | SSM/GDN `in_proj_qkvz` |
/// | `head` | LM / MTP head (see [`CublasScope::head`]) |
/// | `all`, `1`, `true` | every family — the pre-2026-09-11 spelling |
/// | `off`, `0`, `false`, empty | the empty set |
///
/// The result is the UNION of the tokens, so `off` adds nothing rather than
/// clearing what another token armed (`ffn,off` is `ffn`). Unknown tokens are
/// dropped with a warning and never widen the set: a typo must not silently arm
/// an arm, which is the exact failure above.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CublasScope {
    /// Dense-FFN gate/up/down and the MoE shared expert — #917's W8A8 arm.
    pub ffn: bool,
    /// Attention Q/K/V, O projection and the output gate.
    pub attn: bool,
    /// SSM/GDN fused `in_proj_qkvz` prefill projection.
    pub ssm: bool,
    /// LM / MTP head. Parsed, carried and covered by `all`, but NO dispatch
    /// site reads it yet — setting `head` is inert today. Named anyway so the
    /// grammar is the complete family list and `all` has a fixed meaning; a
    /// lever that silently drops a spelling is worse than one that documents
    /// an unclaimed slot.
    pub head: bool,
}

impl CublasScope {
    /// No family armed — the shape an absent or `off` `ATLAS_CUBLAS_GEMM`
    /// resolves to, and the default for every build.
    pub const OFF: Self = Self {
        ffn: false,
        attn: false,
        ssm: false,
        head: false,
    };

    /// Every family — what `all` / `1` / `true` resolve to.
    pub const ALL: Self = Self {
        ffn: true,
        attn: true,
        ssm: true,
        head: true,
    };

    /// Whether any family is armed (for the resolved-set log line).
    pub fn any(&self) -> bool {
        self.ffn || self.attn || self.ssm || self.head
    }
}

/// Parse the [`CublasScope`] grammar. Returns the resolved set plus the tokens
/// that matched nothing, so the caller can warn with the operator's own
/// spelling. Pure — the environment read and the logging both live in
/// [`GemmDispatch::from_env`], which is what makes the table testable.
pub fn parse_cublas_scope(raw: Option<&str>) -> (CublasScope, Vec<String>) {
    let mut scope = CublasScope::OFF;
    let mut unknown = Vec::new();
    let Some(raw) = raw else {
        return (scope, unknown);
    };
    for token in raw.split(',') {
        match token.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "false" | "off" => {}
            "1" | "true" | "all" => scope = CublasScope::ALL,
            "ffn" => scope.ffn = true,
            "attn" => scope.attn = true,
            "ssm" => scope.ssm = true,
            "head" => scope.head = true,
            other => unknown.push(other.to_owned()),
        }
    }
    (scope, unknown)
}

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
    /// Which projection families take a cuBLASLt GEMM arm
    /// (`ATLAS_CUBLAS_GEMM`). The hand-written mma.sync projection GEMMs reach
    /// only ~30% of the cuBLAS bf16 ceiling on GB10, which is why the arms
    /// exist; [`CublasScope`] is why they are no longer all one switch.
    pub cublas: CublasScope,
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
        // OVERRIDDEN by `from_env` from `[defaults] cublas_gemm_scope`; kept
        // so the grammar table below stays pure in the variable alone.
        cublas: parse_cublas_scope(value("ATLAS_CUBLAS_GEMM").as_deref()).0,
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
    /// Resolve from the compiled target's defaults and the environment.
    /// Called once, when the model is built.
    ///
    /// `cublas` comes from [`super::target_defaults::resolved`] — it is the one
    /// field here whose default differs by hardware (`kernels/hopper` declares
    /// `ffn,ssm,attn`; `kernels/gb10` declares `off`, cuBLASLt not being the
    /// GB10 path). The rest are diagnostics and A/B arms with no target
    /// opinion, and still read the environment directly.
    pub fn from_env() -> Self {
        let raw = std::env::var("ATLAS_CUBLAS_GEMM").ok();
        let levers = super::target_defaults::resolved();
        let resolved = GemmDispatch {
            cublas: levers.cublas.value,
            ..from_values(|var| std::env::var(var).ok())
        };
        log_cublas_scope(raw.as_deref(), resolved.cublas);
        resolved
    }

    /// Everything off, block-scaled FP8 prefill on — the shape a build with no
    /// `ATLAS_*` set in the environment resolves to. Tests construct a context
    /// with this instead of mutating the process environment.
    pub fn defaults() -> Self {
        Self {
            w4a16_variant: 0,
            fp8_blockscaled_prefill: true,
            cublas: CublasScope::OFF,
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

/// Say once, at model build, which cuBLASLt arms an `ATLAS_CUBLAS_GEMM` value
/// actually armed.
///
/// A scoped lever is only an improvement if the operator can SEE the scope it
/// resolved to: the failure this replaces was invisible until a `cuMemAlloc_v2`
/// error named a layer 36 nobody had aimed at. Silent when the variable is
/// unset — a serve that never asked for cuBLASLt should not narrate it.
fn log_cublas_scope(raw: Option<&str>, scope: CublasScope) {
    let Some(raw) = raw else {
        // Silent: the target's declaration is reported once, with every other
        // lever, by `target_defaults::summary_line()`.
        return;
    };
    let (_, unknown) = parse_cublas_scope(Some(raw));
    if !unknown.is_empty() {
        tracing::warn!(
            "ATLAS_CUBLAS_GEMM={raw:?}: ignoring unknown families [{}]. The grammar is a \
             comma-separated subset of all|ffn|attn|ssm|head|off (1/true = all).",
            unknown.join(", ")
        );
    }
    tracing::info!(
        "[atlas] ATLAS_CUBLAS_GEMM={raw:?} -> cuBLASLt arms ffn={} attn={} ssm={} head={} \
         (head has no consumer yet){}",
        scope.ffn,
        scope.attn,
        scope.ssm,
        scope.head,
        if scope.any() {
            ""
        } else {
            " — no arm enabled"
        }
    );
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
                cublas: CublasScope::OFF,
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
                [d.cublas.any(), d.cublas_fp8, d.cutlass_gemm],
                expected,
                "{name} must not enable a neighboring GEMM path"
            );
            assert!(d.fp8_blockscaled_prefill);
        }
        assert!(!resolve(&[("ATLAS_FP8_SINGLE_SCALE", "1")]).fp8_blockscaled_prefill);
    }

    // ───────────────── ATLAS_CUBLAS_GEMM scope grammar ─────────────────

    fn scope(raw: &str) -> CublasScope {
        resolve(&[("ATLAS_CUBLAS_GEMM", raw)]).cublas
    }

    /// The whole table, in one place, as the doc comment on [`CublasScope`]
    /// states it. `1`/`true` keep meaning "every arm" so a pre-2026-09-11
    /// launch script is unchanged; every other spelling is new.
    #[test]
    fn the_scope_grammar_maps_each_spelling_to_its_family_set() {
        let f = |ffn, attn, ssm, head| CublasScope {
            ffn,
            attn,
            ssm,
            head,
        };
        let cases: [(&str, CublasScope); 13] = [
            ("all", CublasScope::ALL),
            ("1", CublasScope::ALL),
            ("true", CublasScope::ALL),
            ("ALL", CublasScope::ALL),
            ("off", CublasScope::OFF),
            ("0", CublasScope::OFF),
            ("false", CublasScope::OFF),
            ("", CublasScope::OFF),
            ("ffn", f(true, false, false, false)),
            ("attn", f(false, true, false, false)),
            ("ssm", f(false, false, true, false)),
            ("head", f(false, false, false, true)),
            ("ffn,attn", f(true, true, false, false)),
        ];
        for (raw, expected) in cases {
            assert_eq!(scope(raw), expected, "ATLAS_CUBLAS_GEMM={raw:?}");
        }
        // Whitespace is an operator typing a list, not a new family.
        assert_eq!(scope(" ffn , ssm "), f(true, false, true, false));
        // Union, so `off` subtracts nothing — documented, and the alternative
        // (a clearing token) makes the meaning depend on token order.
        assert_eq!(scope("ffn,off"), f(true, false, false, false));
    }

    /// A typo must not widen the set. The whole point of the change is that
    /// arming an unintended family costs 10.3 GiB of unledgered weight copies;
    /// `ATLAS_CUBLAS_GEMM=fnn` resolving to `all` would reintroduce it.
    #[test]
    fn unknown_families_are_dropped_and_reported_never_widening_the_set() {
        assert_eq!(scope("junk"), CublasScope::OFF);
        assert_eq!(
            scope("ffn,junk"),
            CublasScope {
                ffn: true,
                ..CublasScope::OFF
            }
        );
        let (resolved, unknown) = parse_cublas_scope(Some("ffn, FNN ,bogus"));
        assert_eq!(
            resolved,
            CublasScope {
                ffn: true,
                ..CublasScope::OFF
            }
        );
        assert_eq!(
            unknown,
            vec!["fnn".to_owned(), "bogus".to_owned()],
            "the warning must name what the operator typed, lowercased"
        );
    }

    /// An absent variable is not the same input as `off`, and both must land
    /// on the empty set without allocating an "unknown token" for the caller
    /// to warn about.
    #[test]
    fn an_absent_variable_resolves_to_the_empty_set_silently() {
        assert_eq!(parse_cublas_scope(None), (CublasScope::OFF, Vec::new()));
        assert_eq!(
            parse_cublas_scope(Some("off")),
            (CublasScope::OFF, Vec::new())
        );
        assert!(!CublasScope::OFF.any());
        assert!(CublasScope::ALL.any());
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
