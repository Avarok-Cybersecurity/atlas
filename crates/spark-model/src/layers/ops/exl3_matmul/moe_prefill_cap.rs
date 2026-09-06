// SPDX-License-Identifier: AGPL-3.0-only

//! The fused EXL3 MoE prefill tier's PER-EXPERT ROW CAP — resolution and
//! slab arithmetic, pure (no GPU, no env inside the resolver) so the sizing
//! and dispatch decisions are unit-testable. Split from `moe_prefill.rs` on
//! the 500-LoC cap.
//!
//! ## What the cap is
//!
//! The vendored `exl3_moe` kernel (`exl3_vendor/exl3_moe_kernel.cuh`) takes
//! `max_tokens_per_expert` as a RUNTIME argument: it is the height of the
//! per-group temp slabs (`temp_state_{g,u}[C, rows, hidden]`,
//! `temp_intermediate_{g,u}[C, rows, inter]`) and nothing else — the kernel's
//! GEMM loops walk any count in 16-row M tiles, and an expert whose sorted
//! row count exceeds the cap is skipped (ticket-free) for the host's
//! overflow tier (`moe_prefill_overflow.rs`). Upstream derives the value from
//! `temp_state_g.shape[1]`; vllm-exl3 sizes that slab at 2048 rows; Atlas
//! shipped 128 (upstream's `TEMP_ROWS_FUSED`) until 2026-09-05.
//!
//! ## Why 128 was the wrong default for serving
//!
//! At the canonical 4096-token prefill chunk over 512 experts at top-10 there
//! are 40,960 sorted slots — a MEAN of 80 rows/expert with a heavy routing
//! tail, so on every full chunk many experts exceed 128 and take the
//! overflow tier: per expert, per 1024-row chunk, FIVE host-issued launches
//! (gather, gate `exl3_gemm`, up `exl3_gemm`, SiLU·mul, down `exl3_gemm`, plus
//! the slot store), three of them cooperative grids of <= 48 blocks that
//! serialize behind one another, all behind the fused launch. Raising the
//! cap keeps those experts inside the ONE fused launch, whose ticket
//! scheduler keeps ~C expert groups busy. HYPOTHESIS until the GPU A/B
//! (`.research/exl3_decode_perf/ab_moe_row_cap.sh`) runs: nothing here is
//! measured.
//!
//! ## Why the default is 1024 (arithmetic, not measurement)
//!
//! Temp-slab bytes are `C * rows * (hidden + inter) * 2 B * 2 slabs`
//! ([`exl3_moe_temp_slab_bytes`]). GB10 has 48 SMs -> C = 6; qwen4_exp has
//! hidden 2560, inter 640:
//!
//! | rows | slab bytes | vs the 419 MB deterministic slot slab |
//! |-----:|-----------:|--------------------------------------:|
//! |  128 |    9.8 MB  |  2.3 % |
//! |  512 |   39.3 MB  |  9.4 % |
//! | 1024 |   78.6 MB  | 18.8 % |
//! | 2048 |  157.3 MB  | 37.5 % |
//! | 4096 |  314.6 MB  | 75.0 % |
//!
//! 1024 is 12.8x the 80-row mean — beyond it the tail is routing pathology
//! (one expert holding a quarter of a chunk's tokens), where an 8-SM group
//! walking 64 sixteen-row passes serially is the load-balance risk the
//! review flagged, and the overflow tier's 48-block cooperative GEMMs are the
//! better fit. It also equals `EXL3_MOE_OVERFLOW_CHUNK_ROWS`, so an expert
//! that still overflows always fills at least one whole overflow chunk. The
//! slab cost stays under a fifth of the slot slab the deterministic epilogue
//! already pays. vllm-exl3's 2048 is reachable through the env knob for the
//! A/B; the default can move once measured.
//!
//! ## The kernel's own bound
//!
//! There is no compile-time maximum in the kernel. Its per-group slab offsets
//! are 32-bit `int` arithmetic — `group_idx * max_tokens_per_expert *
//! hidden_dim` and `128 * warp_idx` with `warp_idx < count * hidden_dim /
//! 128` — so `C * rows * max(hidden, inter)` must stay below `i32::MAX`
//! ([`exl3_moe_row_cap_kernel_max`]); the resolver clamps to it loudly. It
//! also clamps to the token-batch cap `t_cap`: a token appears at most once
//! per expert, so no expert can receive more than `t_cap` rows in one batch
//! and a larger cap only buys idle slab.
//!
//! ## Knobs
//!
//! * `ATLAS_EXL3_MOE_ROWS_PER_EXPERT=<rows>` — numeric override.
//! * `ATLAS_NO_EXL3_MOE_WIDE_ROWS` — KILL SWITCH (house convention: presence
//!   check, `=0` is not off): pins the legacy 128-row cap, and WINS over the
//!   numeric knob (an emergency lever must not lose to a stray variable). This
//!   is the A/B's only variable.
//!
//! ## Numerics contract
//!
//! Same kernel, same grid policy, same deterministic per-slot epilogue for
//! every expert the fused tier serves; the cap only changes WHICH experts it
//! serves. An expert with `128 < count <= cap` moves from the overflow tier's
//! cooperative `exl3_gemm` (its own tile shape and split-K) onto the fused
//! kernel's 16x32x128 MoE tile: same trellis decode, same f16 activation
//! precision, but a different fp32 accumulation ORDER — a ONE-TIME bit
//! change for those experts, never a run-to-run one (the per-slot epilogue
//! keeps prefill bit-reproducible on both arms). A batch in which no expert
//! exceeds 128 rows must produce BYTE-IDENTICAL output on both arms.

/// Default per-expert row cap (temp-slab height) — see the module docs.
pub const EXL3_MOE_ROWS_PER_EXPERT_DEFAULT: usize = 1024;

/// The pre-2026-09-05 cap (upstream's `TEMP_ROWS_FUSED`); what the kill
/// switch restores, and the A/B control arm.
pub const EXL3_MOE_ROWS_PER_EXPERT_LEGACY: usize = 128;

/// Floor: one MoE M tile (`MOE_TILESIZE_M`). Below it the fused tier would
/// serve nothing but single-slot experts.
pub const EXL3_MOE_ROWS_PER_EXPERT_MIN: usize = 16;

/// Numeric override of the cap.
pub const EXL3_MOE_ROWS_PER_EXPERT_ENV: &str = "ATLAS_EXL3_MOE_ROWS_PER_EXPERT";

/// Kill switch (presence): pin [`EXL3_MOE_ROWS_PER_EXPERT_LEGACY`].
pub const EXL3_MOE_WIDE_ROWS_KILL_ENV: &str = "ATLAS_NO_EXL3_MOE_WIDE_ROWS";

/// Where the resolved cap came from (logged at model build).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exl3MoeRowCapSource {
    Default,
    Env,
    KillSwitch,
}

/// The geometry the cap is resolved against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exl3MoeRowCapGeometry {
    /// Token-batch cap of the prefill tier (`pf_t_cap`).
    pub t_cap: usize,
    pub hidden: usize,
    pub inter: usize,
    /// Temp-slab count C (= sm_count / 8, clamped to 1..=64).
    pub concurrency: usize,
}

/// The resolved cap plus everything the caller must log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Exl3MoeRowCap {
    /// Rows per expert the kernel is launched with (= temp-slab height).
    pub rows: usize,
    pub source: Exl3MoeRowCapSource,
    /// Loud clamps / rejected inputs — each is a `warn!` at model build.
    pub warnings: Vec<String>,
}

/// Bytes of the four fused temp slabs at this cap: two `[C, rows, hidden]`
/// f16 slabs plus two `[C, rows, inter]` f16 slabs.
pub fn exl3_moe_temp_slab_bytes(
    concurrency: usize,
    rows: usize,
    hidden: usize,
    inter: usize,
) -> usize {
    concurrency * rows * (hidden + inter) * 2 * 2
}

/// The kernel's own bound: the largest cap whose in-kernel `int` slab
/// arithmetic (`C * rows * max(hidden, inter)`) cannot overflow, rounded
/// down to a multiple of [`EXL3_MOE_ROWS_PER_EXPERT_MIN`].
pub fn exl3_moe_row_cap_kernel_max(concurrency: usize, hidden: usize, inter: usize) -> usize {
    let per_row = concurrency.max(1) * hidden.max(inter).max(1);
    let raw = (i32::MAX as usize) / per_row;
    (raw / EXL3_MOE_ROWS_PER_EXPERT_MIN) * EXL3_MOE_ROWS_PER_EXPERT_MIN
}

/// Which tier serves an expert with `count` sorted rows under `cap`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exl3MoeExpertTier {
    /// `count == 0`: neither tier touches it.
    Idle,
    /// `0 < count <= cap`: the fused kernel.
    Fused,
    /// `count > cap`: the chunked overflow tier.
    Overflow,
}

/// The per-expert dispatch decision the host and the kernel must agree on
/// (the kernel's `if (token_count > max_tokens_per_expert) continue;`).
pub fn exl3_moe_expert_tier(count: usize, cap: usize) -> Exl3MoeExpertTier {
    if count == 0 {
        Exl3MoeExpertTier::Idle
    } else if count <= cap {
        Exl3MoeExpertTier::Fused
    } else {
        Exl3MoeExpertTier::Overflow
    }
}

/// Whether a batch of `s = t * top_k` slots needs the host-sync D2H of
/// `expert_offsets`: only when SOME expert could exceed the cap. With
/// `s <= cap` no expert can, the fused kernel covers everything and upstream's
/// no-sync shortcut (`num_active = -1`, default grid) applies. Raising the cap
/// therefore also removes the sync for short batches (HYPOTHESIS: a TTFT win
/// at MTP-verify/short-prompt shapes; unmeasured).
pub fn exl3_moe_needs_host_sync(s: usize, cap: usize) -> bool {
    s > cap
}

/// Resolve the cap from the two knobs against the geometry. Pure: the caller
/// reads the environment ([`exl3_moe_row_cap_from_env`]) so this is testable.
pub fn resolve_exl3_moe_row_cap(
    kill_present: bool,
    env_value: Option<&str>,
    geom: Exl3MoeRowCapGeometry,
) -> Exl3MoeRowCap {
    let mut warnings = Vec::new();
    let (requested, source) = if kill_present {
        if env_value.is_some() {
            warnings.push(format!(
                "{EXL3_MOE_WIDE_ROWS_KILL_ENV} is set, so {EXL3_MOE_ROWS_PER_EXPERT_ENV}={} is \
                 IGNORED — the kill switch pins the legacy {EXL3_MOE_ROWS_PER_EXPERT_LEGACY}-row cap",
                env_value.unwrap_or_default(),
            ));
        }
        (
            EXL3_MOE_ROWS_PER_EXPERT_LEGACY,
            Exl3MoeRowCapSource::KillSwitch,
        )
    } else {
        match env_value {
            None => (
                EXL3_MOE_ROWS_PER_EXPERT_DEFAULT,
                Exl3MoeRowCapSource::Default,
            ),
            Some(v) => match v.trim().parse::<usize>() {
                Ok(n) if n >= 1 => (n, Exl3MoeRowCapSource::Env),
                _ => {
                    warnings.push(format!(
                        "{EXL3_MOE_ROWS_PER_EXPERT_ENV}={v:?} is not a positive row count — using \
                         the default {EXL3_MOE_ROWS_PER_EXPERT_DEFAULT}"
                    ));
                    (
                        EXL3_MOE_ROWS_PER_EXPERT_DEFAULT,
                        Exl3MoeRowCapSource::Default,
                    )
                }
            },
        }
    };

    let mut rows = requested;
    if rows < EXL3_MOE_ROWS_PER_EXPERT_MIN {
        warnings.push(format!(
            "EXL3 MoE rows-per-expert {rows} is below one M tile — clamped to \
             {EXL3_MOE_ROWS_PER_EXPERT_MIN}"
        ));
        rows = EXL3_MOE_ROWS_PER_EXPERT_MIN;
    }
    if rows > geom.t_cap {
        warnings.push(format!(
            "EXL3 MoE rows-per-expert {rows} exceeds the prefill token-batch cap {} (no expert \
             can receive more rows than tokens in a batch) — clamped to {}",
            geom.t_cap, geom.t_cap
        ));
        rows = geom.t_cap;
    }
    let kmax = exl3_moe_row_cap_kernel_max(geom.concurrency, geom.hidden, geom.inter);
    if rows > kmax {
        warnings.push(format!(
            "EXL3 MoE rows-per-expert {rows} exceeds the fused kernel's 32-bit slab-index bound \
             {kmax} at C={} hidden={} inter={} — clamped",
            geom.concurrency, geom.hidden, geom.inter
        ));
        rows = kmax;
    }
    Exl3MoeRowCap {
        rows: rows.max(1),
        source,
        warnings,
    }
}

/// Read the two knobs from the environment and resolve (model build only —
/// nothing on the hot path consults the environment; the slab height does).
pub fn exl3_moe_row_cap_from_env(geom: Exl3MoeRowCapGeometry) -> Exl3MoeRowCap {
    let kill = std::env::var_os(EXL3_MOE_WIDE_ROWS_KILL_ENV).is_some();
    let env = std::env::var(EXL3_MOE_ROWS_PER_EXPERT_ENV).ok();
    resolve_exl3_moe_row_cap(kill, env.as_deref(), geom)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// qwen4_exp on one GB10: 48 SMs -> C = 6, hidden 2560, inter 640,
    /// the 4096-token default batch.
    const GB10_QWEN4: Exl3MoeRowCapGeometry = Exl3MoeRowCapGeometry {
        t_cap: 4096,
        hidden: 2560,
        inter: 640,
        concurrency: 6,
    };

    #[test]
    fn default_is_1024_from_nothing_set() {
        let r = resolve_exl3_moe_row_cap(false, None, GB10_QWEN4);
        assert_eq!(r.rows, EXL3_MOE_ROWS_PER_EXPERT_DEFAULT);
        assert_eq!(r.rows, 1024);
        assert_eq!(r.source, Exl3MoeRowCapSource::Default);
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
    }

    #[test]
    fn env_overrides_and_vllm_exl3s_2048_is_reachable() {
        let r = resolve_exl3_moe_row_cap(false, Some("2048"), GB10_QWEN4);
        assert_eq!((r.rows, r.source), (2048, Exl3MoeRowCapSource::Env));
        assert!(r.warnings.is_empty());
        let r = resolve_exl3_moe_row_cap(false, Some(" 128 "), GB10_QWEN4);
        assert_eq!(r.rows, 128);
    }

    /// The kill switch is a PRESENCE check — `=0` is not off — and it wins
    /// over the numeric knob, loudly.
    #[test]
    fn kill_switch_pins_legacy_128_and_beats_the_numeric_knob() {
        let r = resolve_exl3_moe_row_cap(true, None, GB10_QWEN4);
        assert_eq!(
            (r.rows, r.source),
            (
                EXL3_MOE_ROWS_PER_EXPERT_LEGACY,
                Exl3MoeRowCapSource::KillSwitch
            )
        );
        assert_eq!(r.rows, 128);
        assert!(r.warnings.is_empty());
        let r = resolve_exl3_moe_row_cap(true, Some("2048"), GB10_QWEN4);
        assert_eq!(r.rows, 128);
        assert_eq!(r.warnings.len(), 1, "{:?}", r.warnings);
        assert!(r.warnings[0].contains("IGNORED"));
    }

    #[test]
    fn garbage_env_falls_back_to_the_default_with_a_warning() {
        for bad in ["", "abc", "0", "-5", "12.5"] {
            let r = resolve_exl3_moe_row_cap(false, Some(bad), GB10_QWEN4);
            assert_eq!(r.rows, EXL3_MOE_ROWS_PER_EXPERT_DEFAULT, "{bad:?}");
            assert_eq!(r.source, Exl3MoeRowCapSource::Default);
            assert_eq!(r.warnings.len(), 1, "{bad:?}: {:?}", r.warnings);
        }
    }

    #[test]
    fn clamps_to_the_token_batch_cap_and_the_tile_floor() {
        let r = resolve_exl3_moe_row_cap(false, Some("100000"), GB10_QWEN4);
        assert_eq!(r.rows, 4096);
        assert!(r.warnings.iter().any(|w| w.contains("token-batch cap")));
        let r = resolve_exl3_moe_row_cap(false, Some("3"), GB10_QWEN4);
        assert_eq!(r.rows, EXL3_MOE_ROWS_PER_EXPERT_MIN);
        assert!(r.warnings.iter().any(|w| w.contains("M tile")));
        // A batch cap smaller than the default clamps the default too.
        let small = Exl3MoeRowCapGeometry {
            t_cap: 192,
            ..GB10_QWEN4
        };
        let r = resolve_exl3_moe_row_cap(false, None, small);
        assert_eq!(r.rows, 192);
    }

    /// The kernel's `int` slab arithmetic: `C * rows * max(hidden, inter)`
    /// must fit in i32. At the qwen4_exp shape the bound is far above any
    /// sane cap (~139K rows); at a pathological geometry it bites, loudly.
    #[test]
    fn kernel_int32_bound_is_enforced() {
        assert_eq!(exl3_moe_row_cap_kernel_max(6, 2560, 640), 139_808);
        assert!(exl3_moe_row_cap_kernel_max(6, 2560, 640) > 4096);
        // C=64 (the kernel's MOE_MAX_GROUPS), hidden 16384: 2047 rows max.
        let wide = Exl3MoeRowCapGeometry {
            t_cap: 1 << 20,
            hidden: 16384,
            inter: 4096,
            concurrency: 64,
        };
        let kmax = exl3_moe_row_cap_kernel_max(64, 16384, 4096);
        assert_eq!(kmax, 2032);
        assert!((kmax as u64) * 64 * 16384 <= i32::MAX as u64);
        let r = resolve_exl3_moe_row_cap(false, Some("8192"), wide);
        assert_eq!(r.rows, kmax);
        assert!(r.warnings.iter().any(|w| w.contains("32-bit")));
        // inter > hidden geometries use the larger of the two.
        assert_eq!(
            exl3_moe_row_cap_kernel_max(1, 128, 4096),
            exl3_moe_row_cap_kernel_max(1, 4096, 128)
        );
    }

    /// The module-doc table, pinned: C=6, hidden 2560, inter 640.
    #[test]
    fn slab_bytes_match_the_documented_arithmetic() {
        let b = |rows| exl3_moe_temp_slab_bytes(6, rows, 2560, 640);
        assert_eq!(b(128), 9_830_400);
        assert_eq!(b(512), 39_321_600);
        assert_eq!(b(1024), 78_643_200);
        assert_eq!(b(2048), 157_286_400);
        assert_eq!(b(4096), 314_572_800);
        // Under a fifth of the deterministic slot slab (4096 x 10 x 2560 x 4).
        let slot_slab = 4096 * 10 * 2560 * 4;
        assert!(b(EXL3_MOE_ROWS_PER_EXPERT_DEFAULT) * 5 < slot_slab);
    }

    /// Host and kernel agree on the tier boundary: `count <= cap` fused,
    /// `count > cap` overflow, zero idle — at BOTH caps.
    #[test]
    fn tier_boundary_is_inclusive_at_the_cap() {
        for cap in [
            EXL3_MOE_ROWS_PER_EXPERT_LEGACY,
            EXL3_MOE_ROWS_PER_EXPERT_DEFAULT,
            2048,
        ] {
            assert_eq!(exl3_moe_expert_tier(0, cap), Exl3MoeExpertTier::Idle);
            assert_eq!(exl3_moe_expert_tier(1, cap), Exl3MoeExpertTier::Fused);
            assert_eq!(exl3_moe_expert_tier(cap, cap), Exl3MoeExpertTier::Fused);
            assert_eq!(
                exl3_moe_expert_tier(cap + 1, cap),
                Exl3MoeExpertTier::Overflow
            );
        }
        // The lever's whole point: a 460-row expert overflowed at 128 and
        // stays fused at 1024; a 1500-row one still overflows at 1024.
        assert_eq!(exl3_moe_expert_tier(460, 128), Exl3MoeExpertTier::Overflow);
        assert_eq!(exl3_moe_expert_tier(460, 1024), Exl3MoeExpertTier::Fused);
        assert_eq!(
            exl3_moe_expert_tier(1500, 1024),
            Exl3MoeExpertTier::Overflow
        );
    }

    /// The no-sync shortcut follows the cap: `s <= cap` needs no D2H because
    /// no expert can exceed the cap.
    #[test]
    fn host_sync_follows_the_cap() {
        assert!(!exl3_moe_needs_host_sync(128, 128));
        assert!(exl3_moe_needs_host_sync(129, 128));
        assert!(!exl3_moe_needs_host_sync(1024, 1024));
        assert!(exl3_moe_needs_host_sync(1025, 1024));
        // 60 tokens x top-10 = 600 slots: synced at 128, shortcut at 1024.
        assert!(exl3_moe_needs_host_sync(600, 128));
        assert!(!exl3_moe_needs_host_sync(600, 1024));
    }
}
