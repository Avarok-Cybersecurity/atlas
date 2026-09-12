// SPDX-License-Identifier: AGPL-3.0-only

//! The 5..=32-row native-FP8 dense-FFN decode tier on TENSOR CORES —
//! `w8a16_gemm_m16`, behind `ATLAS_FFN_M16_TC` (#927).
//!
//! WHY. Measured on 1xH100, 2026-09-11, Qwen/Qwen3.8-27B-FP8, tip
//! `2962cfed7`: at a decode batch of 16 the step is **86.7 ms**, of which the
//! 48 SSM layers are 63.3 ms and the dense FFN inside them is **63%**
//! (~833 us/layer). The tier that serves those widths today,
//! `w8a16_gemv_batch16` (rungs 2-3 of `dense_ffn_batch16_decode.rs`), is
//! bit-exact but **FP32-FMA-bound** at M=16, not bandwidth-bound:
//!
//! | shape | batch16 GEMV @ M=16 | HBM3 |
//! |---|---|---|
//! | gate/up N=17408 K=5120 | 0.260 ms / **342 GB/s** | ~3,000 GB/s |
//! | down    N=5120 K=17408 | 0.330 ms / **270 GB/s** | ~3,000 GB/s |
//!
//! An 89 MB FP8 weight matrix should stream in ~30 us. The GEMV spends ~37 ALU
//! ops per weight BYTE (16 scalar FFMA across the 16 rows, 16 BF16->FP32
//! converts, a LUT lookup, a scale multiply), which caps it near 350 GB/s no
//! matter how fast the DRAM is. `w8a16_gemm_m16` replaces those 16 FFMA with
//! one `mma.sync.m16n8k16` lane-slot — the M tile IS 16 rows, so nothing is
//! padded, which is the whole difference from the tile GEMMs that pad M to 128
//! and waste 7/8 of every tile — and cuts the dequant to ~2 instructions per
//! byte. Target: >= 1,500 GB/s at M=16, >= 1,000 GB/s at M=8.
//!
//! 🔴 NUMERICS — THIS ARM REASSOCIATES; THE BATCH16 ARM DOES NOT.
//! `w8a16_gemv_batch16` reduces each output in ONE FP32 accumulator walked in
//! strict K order, which makes it bit-identical to the scalar `w8a16_gemv` that
//! M=1 decode runs. An MMA reduces 16 K-products in the tensor core's own order
//! first, so THIS arm is not. Its contract is <= 2 BF16 ULP per element, with
//! the 128-K block scale still folded once per block onto an FP32 outer
//! accumulator (the two-level fold, preserved exactly). That is a seam, and it
//! is why the lever exists and defaults OFF.
//!
//! It is not a NEW seam, though: the arm the FFN reached at these widths BEFORE
//! #927 was `w8a16_gemm_n128_m128` / `w8a16_gemm_pipelined`, both m16n8k16 MMA
//! kernels with exactly this reassociation. Turning the lever on returns 5..=32
//! to MMA numerics while keeping the ONE-weight-pass property #927 bought.
//!
//! ARM ORDER in `w8_gemm!` (`dense_ffn.rs`) with the lever ON:
//!   1. `m <= 4`     -> `w8a16_gemv_batch4`                (bit-exact)
//!   2. `m` 5..=16   -> `w8a16_gemm_m16`                   (here, MMA)
//!   3. `m` 17..=32  -> `w8a16_gemm_m16` x2 halves         (here, MMA)
//!   4. `m` 5..=16   -> `w8a16_gemv_batch16`               (bit-exact)
//!   5. `m` 17..=32  -> `w8a16_gemv_batch16` x2 halves     (bit-exact)
//!   6. W8A8 block-scaled prefill (#917/#928)
//!   7. transposed / pipelined / base W8A16 tile GEMMs
//! With the lever OFF (the default) rungs 2-3 vanish and the ladder is exactly
//! what #927 shipped.
//!
//! 17..=32 runs the kernel TWICE on contiguous row halves, for the same reason
//! `batch16_decode.rs` does: the FFN activations and outputs are contiguous
//! `[m, k]` / `[m, n]`, so a half is a plain byte offset, and two weight passes
//! still beat one M-padded MMA tile at these widths.
//!
//! ── THE LEVER IS SPLIT PER PROJECTION FAMILY (round 6) ─────────────────────
//! Round 6's serving A/B on 1xH100 (2026-09-11, bs16, `ATLAS_MS_PROFILE`) with
//! the single old lever turned the WHOLE route on at once and measured two
//! opposite results in one number:
//!
//! | phase | tier | Δ step time |
//! |---|---|---|
//! | attention | QKV + o_proj (`w8a16_gemm_m16{,_strided}`) | **−21.7%** |
//! | SSM layers | dense FFN arm (`w8a16_gemm_m16`) | **+13.7%** |
//! | net | | **+5.2%** |
//!
//! One lever could only ship both or neither, so the win was unbuyable. The
//! grammar is now three presence-based variables, all default OFF:
//!
//! | variable | turns on |
//! |---|---|
//! | `ATLAS_ATTN_M16_TC` | the QKV and o_proj tiers |
//! | `ATLAS_FFN_M16_TC` | the dense-FFN arm (rungs 2-3 above) |
//! | `ATLAS_M16_TC` | BOTH — the umbrella, i.e. round 6's behaviour |
//!
//! ⚠ `ATLAS_FFN_M16_TC=1` MEANS SOMETHING NARROWER THAN IT DID IN ROUND 6.
//! Before this commit it was the only lever and it reached all three tiers;
//! round 6's serve J and its +5.2% were measured with it. The recipe that
//! reproduces round 6 is now `ATLAS_M16_TC=1`. The recipe that buys the
//! attention win WITHOUT the FFN loss — the point of the split — is
//! `ATLAS_ATTN_M16_TC=1` alone.
//!
//! ── WHY THE FFN ARM LOSES WHERE THE ATTENTION TIERS WIN (HYPOTHESIS) ───────
//! Same kernel, same M, same weight format; the one thing that differs is N,
//! and therefore the grid:
//!
//! | tier | N | CTAs at `N_TILE=32` |
//! |---|---|---|
//! | o_proj | 5120 | 160 |
//! | QKV (K, V) | 1024 | 32 |
//! | QKV (Q) | 6144 | 192 |
//! | **dense FFN gate/up** | **17408** | **544** |
//! | dense FFN down | 5120 | 160 |
//!
//! The kernel is 4 warps at 19,456 B of smem under `__launch_bounds__(128, 4)`,
//! so an H100 SM holds 4 CTAs and the machine holds 132 × 4 = **528**. Every
//! attention tier fits inside one partial wave and runs at full occupancy from
//! the first instruction. gate/up at 544 is **one full wave plus a 16-CTA
//! tail**: 3% of the work costs a second wave's worth of launch, prologue and
//! HBM-latency ramp, none of which is overlapped with anything, because by then
//! 116 SMs are idle. That is the leading explanation for a tier that beats
//! `w8a16_gemv_batch16` 3.71× in the microtest (which times ONE shape in
//! isolation, with no tail to pay) and still loses 13.7% in the serve.
//!
//! A second, non-exclusive explanation: at `N_TILE=32` each 16-row A tile is
//! read by twice as many CTAs as at 64, and gate/up's 89 MB weight evicts A
//! from a 50 MB L2 between passes, so the "A stays L2-resident" claim in the
//! kernel header — which holds comfortably at N=1024 — may not hold at
//! N=17408.
//!
//! Both hypotheses predict the same fix, which is why `ATLAS_FFN_M16_TC_NTILE`
//! exists: `=64` selects `w8a16_gemm_m16_n64`, taking gate/up to 272 CTAs
//! (inside one wave) and doubling A reuse. Default stays 32 — the tile with the
//! receipt. NEITHER hypothesis has been measured; the A/B that settles it is
//! `ATLAS_FFN_M16_TC=1 ATLAS_FFN_M16_TC_NTILE=64` against
//! `ATLAS_FFN_M16_TC=1` on the same serve.
//!
//! ── THE ROUND-6 M=32 RED CELL WAS THE ORACLE, NOT THE SPLIT ────────────────
//! Round 6's microtest reported `gate/up M=32` at `max_ulp 28`, 5 of 557,056
//! elements over the 2-ULP budget, `sign_flips 0`, `rel_rms 4.2e-5`, while
//! `down M=32` and every M ≤ 16 cell was green. It was read as a possible
//! row/pitch defect in the two-halves rung. It is not: a host simulation of the
//! exact geometry (`dense_ffn_m16_tc_m32_tests.rs`) reproduces the signature —
//! 5 over-budget elements, none in rows 0..15 — with NO offset arithmetic at
//! all. Every one of them is an output that cancelled to |ref| between 5.7e-6
//! and 1.6e-4 against a reference RMS of 39.1, i.e. to ~1e-7..4e-6 of the
//! matrix scale, where one FP32 accumulation rounding spans hundreds of ordinal
//! BF16 ULP. M=32 trips it and M=16 does not because M=32 samples twice the
//! outputs; gate/up trips it and down does not because gate/up has 3.4× the
//! columns. The fix is in the oracle's comparison (a mixed absolute/relative
//! criterion), not here — see `examples/native_fp8_ffn_m16_tc_microtest.rs`.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

/// The tier's NUMERICS CONTRACT — the one comparison the GPU oracle
/// (`examples/native_fp8_ffn_m16_tc_microtest.rs`) and the host simulation both
/// evaluate, so a receipt and a unit test cannot be grading different things.
#[path = "dense_ffn_m16_tc_oracle.rs"]
pub mod oracle;

pub use oracle::{
    M16_TC_ACC_FLOOR_MARGIN, M16_TC_MAX_ULP, bf16_ord, m16_tc_acc_floor, within_m16_tc_budget,
};

use super::DenseFfnLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::Fp8Weight;
use spark_runtime::gpu::KernelHandle;

/// Which projection families the tensor-core tier serves, and at what CTA
/// width. SSOT for the whole lever grammar; every call site resolves it ONCE at
/// construction into a field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct M16TcLevers {
    /// The dense-FFN arm (`ATLAS_FFN_M16_TC`).
    pub ffn: bool,
    /// The multi-seq FP8 QKV tier and the FP8 o_proj tier
    /// (`ATLAS_ATTN_M16_TC`).
    pub attn: bool,
    /// CTA N width for the FFN arm: 32 (default) or 64
    /// (`ATLAS_FFN_M16_TC_NTILE=64`). Attention always runs 32 — the wide tile
    /// has no strided twin and its N is already CTA-starved.
    pub ffn_n_tile: u32,
}

/// The grammar, as a pure function of the three variables' PRESENCE plus the
/// N-tile string — so the rule is testable without touching the process
/// environment.
///
/// Presence rather than `=1` everywhere (the N tile aside, which needs a
/// value): it keeps every A/B recipe a bare `VAR=1` prefix with no "=0 means
/// on" trap, the same contract `ATLAS_FFN_NO_BATCH16` uses next door. All three
/// default OFF, which is the opposite polarity to that kill switch and
/// deliberately so: it is an operator's escape hatch from a shipped default,
/// these are opt-ins to a route that trades #927's bit-exactness for bandwidth.
///
/// An unrecognised `ATLAS_FFN_M16_TC_NTILE` falls back to 32 rather than
/// failing the boot: the tile is a perf A/B knob, and the route log says which
/// one actually ran.
pub(crate) fn resolve_m16_tc_levers(ffn: bool, attn: bool, n_tile: Option<&str>) -> M16TcLevers {
    M16TcLevers {
        ffn,
        attn,
        ffn_n_tile: match n_tile {
            Some("64") => ops::W8A16_GEMM_M16_N_TILE_WIDE,
            _ => ops::W8A16_GEMM_M16_N_TILE,
        },
    }
}

/// The resolved levers for this process.
///
/// `OnceLock`-cached for the same reason the batch16 switch is: the selector
/// runs per projection per layer per step, and a per-call `var_os` could change
/// the captured launch set across CUDA-graph replays.
pub fn m16_tc_levers() -> M16TcLevers {
    static ON: std::sync::OnceLock<M16TcLevers> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let n_tile = std::env::var("ATLAS_FFN_M16_TC_NTILE").ok();
        resolve_m16_tc_levers(
            // ★ THE TARGET'S DECLARATION, environment second. `ffn_m16_tc` is
            // a `[defaults]` row (`kernels/<hw>/HARDWARE.toml`), so an H100
            // serve reproduces round 6's verdict — the FFN arm OFF — with an
            // empty environment, and `ATLAS_FFN_M16_TC` / `ATLAS_M16_TC`
            // remain the A/B. Both variables are folded in by the resolver,
            // not here: an umbrella that could also DISARM a declaration would
            // make the recipe depend on export order.
            ops::target_defaults::resolved().ffn_m16_tc.value,
            std::env::var_os("ATLAS_ATTN_M16_TC").is_some()
                || std::env::var_os("ATLAS_M16_TC").is_some(),
            n_tile.as_deref(),
        )
    })
}

/// How the tensor-core tier serves `m` rows, or `None` when it does not claim
/// them. Mirrors `Batch16Plan` so the two ladders read the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum M16TcPlan {
    /// One launch covering rows `0..m` (m <= 16 = the kernel's M tile).
    Single,
    /// Two launches on contiguous row halves: rows `0..first`, then
    /// `first..m`. `first` is `ceil(m/2)`, so both halves are <= 16 for every
    /// m <= 32 and the FIRST half is the wider one (m=17 -> 9 + 8).
    Halves { first: u32 },
}

/// The whole selection rule, as a pure function of the row count, the reduction
/// depth, the handle's presence and the lever.
///
/// `k` is part of the rule and not an `ensure!` at the call site: the kernel
/// indexes `block_scale[n_block * (K/128) + k/128]`, so a K that is not a whole
/// number of 128-wide scale blocks has no correct scale to fold and the tier
/// must DECLINE rather than launch and be wrong. Every Atlas FP8 FFN shape
/// satisfies it (Qwen3.8-27B: 5120 and 17408), but a model whose hidden size is
/// not a multiple of 128 would otherwise fall off this cliff silently.
pub(crate) fn m16_tc_plan(m: u32, k: u32, loaded: bool, enabled: bool) -> Option<M16TcPlan> {
    if !enabled || !loaded || !k.is_multiple_of(128) {
        return None;
    }
    match m {
        5..=16 => Some(M16TcPlan::Single),
        // Both halves must be <= 16, the kernel's M tile. `div_ceil` puts the
        // odd row in the first half; the split changes no row's arithmetic.
        17..=32 => Some(M16TcPlan::Halves {
            first: m.div_ceil(2),
        }),
        _ => None,
    }
}

/// Which contiguous instantiation the FFN arm launches, given the requested
/// tile and which entry points this shadow actually carries.
///
/// A shadow built before the wide arm existed has no `w8a16_gemm_m16_n64`, so
/// `ATLAS_FFN_M16_TC_NTILE=64` must fall back to the 32-wide kernel rather than
/// launch a zero handle. The fallback is silent by design — the route log names
/// the tile that ran.
pub(crate) fn m16_tc_kernel(
    n_tile: u32,
    narrow: KernelHandle,
    wide: KernelHandle,
) -> (ops::ContiguousM16Gemm, KernelHandle, u32) {
    if n_tile == ops::W8A16_GEMM_M16_N_TILE_WIDE && wide.0 != 0 {
        (
            ops::w8a16_gemm_m16_n64,
            wide,
            ops::W8A16_GEMM_M16_N_TILE_WIDE,
        )
    } else {
        (ops::w8a16_gemm_m16, narrow, ops::W8A16_GEMM_M16_N_TILE)
    }
}

impl DenseFfnLayer {
    /// The plan for `m` rows at reduction depth `k` on THIS layer — handle
    /// presence plus the lever.
    pub(crate) fn ffn_m16_tc_plan(&self, m: u32, k: u32) -> Option<M16TcPlan> {
        m16_tc_plan(m, k, self.w8a16_gemm_m16_k.0 != 0, self.m16_tc)
    }

    /// Run one dense-FFN projection through `w8a16_gemm_m16`.
    ///
    /// `input` is `[m, k]` BF16 and `out` is `[m, n]` BF16, both CONTIGUOUS,
    /// which is what makes the `Halves` plan a pair of byte offsets rather than
    /// a strided launch. (`ops::w8a16_gemm_m16_strided` is the tool when a
    /// caller's rows are not contiguous; the attention QKV path uses it.)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn w8a16_m16_tc_proj(
        &self,
        ctx: &ForwardContext,
        plan: M16TcPlan,
        w: &Fp8Weight,
        input: DevicePtr,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        let (gemm, kernel, n_tile) = m16_tc_kernel(
            self.m16_tc_n_tile,
            self.w8a16_gemm_m16_k,
            self.w8a16_gemm_m16_n64_k,
        );
        self.log_m16_tc_route(ctx, plan, n_tile);
        const BF16: usize = 2;
        let launch = |rows: u32, first: u32| {
            gemm(
                ctx.gpu,
                kernel,
                input.offset(first as usize * k as usize * BF16),
                w.weight,
                w.row_scale,
                out.offset(first as usize * n as usize * BF16),
                rows,
                n,
                k,
                stream,
            )
        };
        match plan {
            M16TcPlan::Single => launch(m, 0),
            M16TcPlan::Halves { first } => {
                launch(first, 0)?;
                launch(m - first, first)
            }
        }
    }

    /// Log-once latch, in the same `log:ffn_*` shape the other dense-FFN route
    /// logs use. It is worth a line because this arm is the one that is NOT
    /// bit-identical to the M=1 decode path: a TPOT report or a parity
    /// complaint at 5..=32 rows needs to say which of the two tiers ran.
    fn log_m16_tc_route(&self, ctx: &ForwardContext, plan: M16TcPlan, n_tile: u32) {
        if ctx.stats.once("log:ffn_m16_tc_decode") {
            let how = match plan {
                M16TcPlan::Single => "one launch",
                M16TcPlan::Halves { .. } => "two launches on contiguous row halves",
            };
            let asked = self.m16_tc_n_tile;
            tracing::info!(
                "[atlas] dense FFN decode: ATLAS_FFN_M16_TC — tensor-core w8a16_gemm_m16 \
                 N_TILE={n_tile} (asked {asked}) ({how}) for 5..=32 rows, ahead of \
                 w8a16_gemv_batch16. One weight pass, m16n8k16 MMA, so outputs are \
                 REASSOCIATED vs the scalar w8a16_gemv (<= 2 BF16 ULP), unlike the batch16 \
                 tier. This lever no longer reaches the attention tiers — that is \
                 ATLAS_ATTN_M16_TC, and ATLAS_M16_TC is both. Unset it to restore the \
                 bit-exact tier (#927)."
            );
        }
    }
}

#[cfg(test)]
#[path = "dense_ffn_m16_tc_lever_tests.rs"]
mod lever_tests;

#[cfg(test)]
#[path = "dense_ffn_m16_tc_tests.rs"]
mod tests;

/// The host simulation that settles round 6's `gate/up M=32` red cell: it
/// reproduces the two-halves geometry and the oracle's comparison on the CPU,
/// with no GPU and no offset arithmetic to get wrong.
#[cfg(test)]
#[path = "dense_ffn_m16_tc_m32_tests.rs"]
mod m32_tests;
