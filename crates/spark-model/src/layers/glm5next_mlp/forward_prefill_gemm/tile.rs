// SPDX-License-Identifier: AGPL-3.0-only

//! Which grouped-GEMM tile geometry the routed prefill launches, and the width-floor /
//! exact-tiles / enable env levers that gate it.
//!
//! Split out of `forward_prefill_gemm.rs` to keep that file under the 500-LoC cap; see
//! `dispatch.rs` for the launch that consumes [`GemmTile`] and these levers.

/// `M_TILE` of the BASE `moe_w4a16_grouped_gemm_ptrtable`. 🪤 COUPLED to `#define M_TILE 64`
/// in `kernels/gb10/common/moe_w4a16_grouped_gemm.cu`; `max_m_tiles` is counted in these.
/// A tile variant carries its own — see [`GemmTile`].
#[cfg(test)]
pub(crate) const GROUPED_M_TILE: usize = 64;

/// Which grouped-GEMM tile geometry the routed prefill launches,
/// `AVAROK_GLM_MOE_GEMM_TILE=<name>`.
///
/// 🔴 MEASURED 2026-09-22 on n1 (one GB10), `examples/glm5next_moe_grouped_tile_bench.rs`,
/// the REAL production shape (288 experts / 144 local under EP=2, `top_k = 8`, gate+up
/// `N=2048 K=4096`, down `N=4096 K=2048`, 679.5 MB of expert weight per sweep, rows=256
/// routing). Every row is BYTE-IDENTICAL to `base`; the harness asserts that rather than an
/// error bar, because the production claim is `sha8 d44c9251` unchanged.
///
/// | tile | gate/up GB/s | down GB/s | % of 273 | vs base |
/// |---|---:|---:|---:|---:|
/// | `base` (M64 N64 K16) | 43.2 | 46.1 | 15.8 / 16.9 % | 1.00x |
/// | `k32` (M64 N64 K32) | 57.6 | 59.8 | 21 % | ~1.3x |
/// | `k64` (M64 N64 K64) | 58.4 | 63.1 | 21–23 % | ~1.37x |
/// | `m16_k64` (M16 N64 K64) | 72.9 | 73.0 | 27 % | ~1.6x |
/// | `alkm_m16_k128` (+arith LUT, k-major fetch) | 138.7 | 140.6 | 51 % | ~3.1x |
/// | **`bt_m16_k128`** (+transposed staging) | **156.2** | **161.8** | **57–59 %** | **3.5x** |
///
/// 🔴 And the denominator that makes those percentages mean something: a PURE coalesced
/// stream of exactly these bytes, no dequant and no mma, measures **237–240 GB/s (87 %)** on
/// this part in this harness (`moe_w4a16_grouped_stream_probe`). So `bt_m16_k128` is at
/// **66 % of what the memory system actually delivers here**, and the 273 GB/s datasheet
/// figure is not the reachable bar.
///
/// 🪤 What the three winning changes were, in order of size — none of them the M tile the
/// profile pointed at, which is why they were measured rather than argued:
///  1. **The `__constant__` E2M1 table.** A constant-memory read broadcasts ONE address per
///     replay; 32 lanes holding up to 16 distinct nibbles cost up to 16 replays per lookup,
///     and there is one lookup per weight element (1.2e9 per launch). Replacing it with
///     integer bit assembly is worth **1.7x on its own** (72.9 → 123.6 GB/s at M16 K64).
///  2. **Transposed staged B tile** — 16 BF16 become two 16-byte shared stores instead of
///     16 two-byte ones, and the mma's `b0`/`b1` become one aligned 32-bit read: **1.12x**.
///  3. **`M_TILE` 64 → 16** with the four warps splitting N instead of M: **1.6x** at the
///     base dequant, and it is the padding fix the profile predicted — but only the third
///     largest of the three, and worth far less than the table.
///
/// 🪤 Staging MORE k stops paying once the shared-memory footprint costs a resident CTA:
/// K256 is slower than K128 at every other setting, and at `M_TILE = 64` K128 loses to K64.
/// This kernel is latency-bound, not bandwidth-bound, until the dequant cost is removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GemmTile {
    /// Kernel entry point in the `moe_w4a16` module.
    pub name: &'static str,
    /// 🪤 `max_m_tiles` and the worst-case bound are counted in THIS, not in 64. A kernel
    /// launched with a grid height computed against the wrong tile silently drops every row
    /// past the first tile of any expert.
    pub m_tile: usize,
    /// 🪤 grid.x is counted in THIS. Too small computes the left of the output twice and
    /// never writes the right.
    pub n_tile: u32,
    /// Block threads = warps x 32.
    pub threads: u32,
}

/// The tile this branch ships by default: measured 3.5x the base at rows=256,
/// byte-identical. `AVAROK_GLM_MOE_GEMM_TILE=base` restores the prior behaviour exactly.
pub(crate) const DEFAULT_GEMM_TILE: GemmTile = GemmTile {
    name: "moe_w4a16_grouped_gemm_ptrtable_bt_m16_k128",
    m_tile: 16,
    n_tile: 64,
    threads: 128,
};

/// Every tile the dispatch will accept. Kept small on purpose: these are the shapes the
/// microbench actually measured on GLM's own geometry, not the full instantiation list.
pub(crate) const GEMM_TILES: &[GemmTile] = &[
    // 🪤 index 0 is the BASE tile and the fallback; `select_gemm_tile("base")` and the
    // PTX-missing path both name it by position.
    GemmTile {
        name: "moe_w4a16_grouped_gemm_ptrtable",
        m_tile: 64,
        n_tile: 64,
        threads: 128,
    },
    GemmTile {
        name: "moe_w4a16_grouped_gemm_ptrtable_k32",
        m_tile: 64,
        n_tile: 64,
        threads: 128,
    },
    GemmTile {
        name: "moe_w4a16_grouped_gemm_ptrtable_k64",
        m_tile: 64,
        n_tile: 64,
        threads: 128,
    },
    GemmTile {
        name: "moe_w4a16_grouped_gemm_ptrtable_m16_k64",
        m_tile: 16,
        n_tile: 64,
        threads: 128,
    },
    GemmTile {
        name: "moe_w4a16_grouped_gemm_ptrtable_alkm_m16_k128",
        m_tile: 16,
        n_tile: 64,
        threads: 128,
    },
    DEFAULT_GEMM_TILE,
    GemmTile {
        name: "moe_w4a16_grouped_gemm_ptrtable_bt_m16_n128_k128",
        m_tile: 16,
        n_tile: 128,
        threads: 256,
    },
];

/// Resolve `AVAROK_GLM_MOE_GEMM_TILE` to one of [`GEMM_TILES`]. The value is the kernel's
/// suffix (`base`, `k32`, `k64`, `m16_k64`, `alkm_m16_k128`, `bt_m16_k128`,
/// `bt_m16_n128_k128`) or the full kernel name.
///
/// 🪤 Unknown names FAIL LOUD at resolve time. A typo that silently fell back to the base
/// tile would turn an A/B arm into a duplicate of its control and read as "no difference".
pub(crate) fn select_gemm_tile(v: &str) -> Option<GemmTile> {
    let v = v.trim();
    if v.eq_ignore_ascii_case("base") || v.is_empty() {
        return Some(GEMM_TILES[0]);
    }
    GEMM_TILES
        .iter()
        .copied()
        .find(|t| t.name == v || t.name.strip_prefix("moe_w4a16_grouped_gemm_ptrtable_") == Some(v))
}

/// The tile in force for this process. Read once — this sits on the per-layer path.
pub(crate) fn gemm_tile() -> GemmTile {
    static T: std::sync::OnceLock<GemmTile> = std::sync::OnceLock::new();
    *T.get_or_init(|| match std::env::var("AVAROK_GLM_MOE_GEMM_TILE") {
        Ok(v) => match select_gemm_tile(&v) {
            Some(t) => {
                tracing::warn!(
                    "GLM routed-MoE prefill grouped GEMM tile overridden to `{}` \
                     (M_TILE {}, N_TILE {}, {} threads); default is `{}`",
                    t.name,
                    t.m_tile,
                    t.n_tile,
                    t.threads,
                    DEFAULT_GEMM_TILE.name
                );
                t
            }
            None => {
                tracing::error!(
                    "AVAROK_GLM_MOE_GEMM_TILE=`{v}` is not a known tile — using the default \
                     `{}`. Known: {:?}",
                    DEFAULT_GEMM_TILE.name,
                    GEMM_TILES.iter().map(|t| t.name).collect::<Vec<_>>()
                );
                DEFAULT_GEMM_TILE
            }
        },
        Err(_) => DEFAULT_GEMM_TILE,
    })
}

/// Route the routed-expert **prefill** through the grouped GEMM.
/// `AVAROK_GLM_MOE_PREFILL_GEMM=0` restores the row-batched GEMV path exactly.
///
/// 🔬 Default ON in this branch so the A/B is one env flip on ONE image — the same lever
/// shape as `AVAROK_GLM_MOE_ROW_BATCH_MAX`. Read once: this sits on the per-layer path.
pub(crate) fn prefill_gemm_enabled() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let off = std::env::var("AVAROK_GLM_MOE_PREFILL_GEMM").as_deref() == Ok("0");
        if off {
            tracing::warn!(
                "GLM routed-MoE prefill grouped GEMM DISABLED \
                 (AVAROK_GLM_MOE_PREFILL_GEMM=0) — row-batched GEMV path"
            );
        }
        !off
    })
}

/// Narrowest prefill sub-chunk the grouped GEMM may take, `AVAROK_GLM_MOE_PREFILL_GEMM_MIN_ROWS`.
///
/// 🔴 MEASURED, and the reason this gate exists at all. n1/n2, ONE image per round,
/// community ckpt, 5,400-token prompt, spec-off, max-seq-len 131072, transport gate PROVEN
/// RoCE on both rails, every cell a matched arm on the same binary:
///
/// | sub-chunk | GEMV (production) | grouped GEMM | ratio |
/// |---|---:|---:|---:|
/// | 16 rows (the shipping default) | **63.05 tok/s** | 25.14 tok/s | **0.40x** |
/// | 64 rows | 77.79 tok/s | 57.71 tok/s | 0.74x |
/// | 128 rows | 80.73 tok/s | **88.17 tok/s** | **1.09x** |
/// | 256 rows | 82.14 tok/s | **128.10 tok/s** | **1.56x** |
///
/// The crossover is between 64 and 128, so THE FLOOR IS 128. It was first shipped at 64 on
/// the reasoning that 64 sits between the measured-bad 16 and the measured-good 256; the
/// 64- and 128-row arms were then run and 64 turned out to be on the LOSING side (0.74x).
/// That is what the sweep was for.
///
/// Why the narrow widths lose: 16 rows at `top_k = 8` is 128 routed slots over 288 experts,
/// so the expert union a chunk touches is barely smaller than the 8-row GEMV's own — there
/// is no weight traffic to save — while every active expert still gets an `M_TILE = 64` tile
/// holding ~2.6 real rows. At 256 rows the same chunk touches essentially this rank's WHOLE
/// local expert set once: ~5x less weight traffic per token, and the win appears.
///
/// 🪤 NOT the host sync. The obvious suspect was the per-layer `expert_offsets` D2H, which
/// drains the stream `ceil(5400/16) * 42 = 14,196` times at a 16-row chunk. Measured with
/// [`prefill_gemm_exact_tiles`] off — every one of those drains removed — the 16-row arm went
/// 25.14 -> 25.56 tok/s, **+1.7 %**. The sync is not the cost; the tile geometry is. Anyone
/// tempted to fix narrow widths by deferring the sync should read that number first.
///
/// 🪤 A default-ON grouped path with no width floor would have silently regressed the
/// shipping serve by 2.5x, because `PREFILL_ROWS` defaults to 16. The floor is what makes
/// "default ON" safe, and it is an env lever so the crossover can be re-swept without a
/// rebuild.
pub(crate) fn prefill_gemm_min_rows() -> usize {
    static M: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        let m = std::env::var("AVAROK_GLM_MOE_PREFILL_GEMM_MIN_ROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_GEMM_MIN_ROWS)
            .max(1);
        if m != DEFAULT_GEMM_MIN_ROWS {
            tracing::warn!(
                "GLM routed-MoE prefill grouped GEMM width floor overridden to {m} rows \
                 (default {DEFAULT_GEMM_MIN_ROWS}; MEASURED 0.40x at 16, 0.74x at 64, 1.09x at 128, 1.56x at 256)"
            );
        }
        m
    })
}

/// See [`prefill_gemm_min_rows`] for the measurement this number comes from.
pub(crate) const DEFAULT_GEMM_MIN_ROWS: usize = 128;

/// Read the REAL expert histogram to size the grid, `AVAROK_GLM_MOE_PREFILL_GEMM_EXACT_TILES=0`
/// to use the worst-case bound instead and skip the host read entirely.
///
/// 🔬 A discriminator, not a tuning knob. The exact bound costs one `copy_d2h_on_stream` per
/// routed layer per sub-chunk, and that call DRAINS THE STREAM — at a 16-row sub-chunk over a
/// 5,400-token prompt that is `338 * 42 = 14,196` full host-GPU round trips. Turning it off
/// trades those drains for a grid that is `ceil(rows*top_k/64)`x taller and almost entirely
/// early-exit. Flipping this against a fixed arm separates "the sync is the cost" from "the
/// tile padding is the cost" in ONE image instead of guessing.
pub(crate) fn prefill_gemm_exact_tiles() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let off = std::env::var("AVAROK_GLM_MOE_PREFILL_GEMM_EXACT_TILES").as_deref() == Ok("0");
        if off {
            tracing::warn!(
                "GLM routed-MoE prefill grouped GEMM: WORST-CASE grid height \
                 (AVAROK_GLM_MOE_PREFILL_GEMM_EXACT_TILES=0) — no per-layer expert_offsets D2H"
            );
        }
        !off
    })
}

/// Tiles of [`GROUPED_M_TILE`] the busiest expert needs, from the host copy of
/// `expert_offsets`.
///
/// 🔴 Why this is not the worst case. The kernel's grid is
/// `(ceil(N/64), max_m_tiles, num_experts)` and every CTA past an expert's real row count
/// early-exits. The safe bound — one expert takes every routed slot — is
/// `ceil(rows * top_k / 64)`: at `rows = 256`, `top_k = 8` that is 32, i.e. a
/// `32 x 32 x 288 = 294,912`-CTA launch of which ~97 % do nothing, three times per layer.
/// Reading the REAL offsets makes it `1 x 32 x 288`. 288 experts over 2,048 routed slots
/// average 7.1 rows each, so one tile is the normal answer.
///
/// 🪤 Cannot truncate: the result is the max over the ACTUAL per-expert counts, and it is
/// clamped to the worst case only as an upper bound. `layers::moe`'s prefill makes the same
/// trade (`moe_prefill_exact_tiles`, default-on for NVFP4, measured -120.7 ms cold TTFT).
pub(crate) fn max_m_tiles_from_offsets(offsets: &[i32], worst_case: u32, m_tile: usize) -> u32 {
    let mut prev = 0i32;
    let mut max_rows = 0i32;
    for &cur in offsets.iter().skip(1) {
        max_rows = max_rows.max(cur - prev);
        prev = cur;
    }
    (max_rows.max(0) as u32)
        .div_ceil(m_tile.max(1) as u32)
        .max(1)
        .min(worst_case.max(1))
}
