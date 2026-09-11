// SPDX-License-Identifier: AGPL-3.0-only

//! HOST SIMULATION of the INDEX ALGEBRA of the two Hopper GDN prefill remnant
//! twins, plus the lever grammar that selects them (#928).
//!
//! Neither kernel can run anywhere this repository's CI or its GB10 boxes can
//! reach — they exist only under `kernels/hopper` and need an sm_90a device —
//! so everything about them that is decidable WITHOUT a GPU is decided here,
//! and the arithmetic half is in `ssm_gdn_remnants_numerics_tests.rs`.
//!
//! What is at risk is not the arithmetic; an `mma.sync` accumulates in f32
//! whatever it is handed. It is the maps that decide WHAT it is handed, because
//! both twins moved from "one thread owns one column, loop over rows" to "one
//! lane owns a scattered C fragment", so every quantity is now addressed by
//! warp and lane rather than by a loop variable:
//!
//!   1. `fwd_o`'s output fragment, 4 m-tiles x 4 n-quarters over [64][128];
//!   2. `fwd_o`'s Gram fragment, 4 x 4 tiles of 16 over [64][64], which is also
//!      where the causal mask and the decay fold now happen — the parent left
//!      the `l > i` half holding raw Gram values because its consumer looped
//!      `l <= i`, and an MMA has no such loop bound;
//!   3. `wu`'s L-build fragment, the same 16x16 tiling, which must cover the
//!      four DIAGONAL blocks exactly once for the T_jj inverses to be defined;
//!   4. `wu`'s per-warp [64][16] panel and its transposed publish, which is the
//!      single most plausible slip in the blocked solve and leaves every shape
//!      and bound legal;
//!   5. the padded 136/72/24 row strides, which exist to make the fragment
//!      reads bank-conflict-free and which silently read the wrong column if
//!      the launcher and the kernel disagree about the footprint.

use super::{
    GDN_FWD_O_HOPPER_SMEM, GDN_HOPPER_CHUNK, GDN_HOPPER_DIM, GDN_HOPPER_REMNANT_BLOCK,
    GDN_WU_HOPPER_SMEM, gdn_hopper_remnant_pick, gdn_hopper_remnant_reject,
};
use spark_runtime::gpu::KernelHandle;

const KD: usize = 128;
const VD: usize = 128;
const C: usize = 64;
const SW: usize = 136; // q / k / S^T padded row stride, in bf16 elements
const SC: usize = 72; //  Gram limbs / uc^T / L limbs
const SX: usize = 24; //  the 16-column triangular-solve panel
const THREADS: usize = 512;

// ── the four fragment maps, as the kernels spell them ──────────────────────

/// `fwd_o` output / `q.S^T` slot `(tid, nt, e)` -> the `(i, v)` it holds.
/// Kernel: `m_base = (warp & 3) * 16`, `n_base = (warp >> 2) * 32`, NT = 4.
fn fwd_o_slot(tid: usize, nt: usize, e: usize) -> (usize, usize) {
    let (warp, lane) = (tid >> 5, tid & 31);
    let (grp, q4) = (lane >> 2, lane & 3);
    let i = (warp & 3) * 16 + grp + if e >= 2 { 8 } else { 0 };
    let v = (warp >> 2) * 32 + nt * 8 + q4 * 2 + (e & 1);
    (i, v)
}

/// The 16x16 Gram slot `(tid, nt, e)` -> `(row, col)`. Shared by `fwd_o`'s
/// `kq` fold and `wu`'s `L` build: `m_base = (warp & 3) * 16`,
/// `n_base = (warp >> 2) * 16`, NT = 2.
fn gram_slot(tid: usize, nt: usize, e: usize) -> (usize, usize) {
    let (warp, lane) = (tid >> 5, tid & 31);
    let (grp, q4) = (lane >> 2, lane & 3);
    let r = (warp & 3) * 16 + grp + if e >= 2 { 8 } else { 0 };
    let c = (warp >> 2) * 16 + nt * 8 + q4 * 2 + (e & 1);
    (r, c)
}

/// `wu`'s right-hand-side / output slot `(tid, mt, nt, e)` -> `(i, col)`, with
/// `col` LOCAL to the warp's 16-column panel. `solve` and the panel base come
/// from the warp index and are checked separately.
fn wu_slot(tid: usize, mt: usize, nt: usize, e: usize) -> (usize, usize) {
    let lane = tid & 31;
    let (grp, q4) = (lane >> 2, lane & 3);
    let i = mt * 16 + grp + if e >= 2 { 8 } else { 0 };
    let col = nt * 8 + q4 * 2 + (e & 1);
    (i, col)
}

/// `wuh_publish` slot `(lane, nt, e)` -> the `(col, row)` it writes, i.e. the
/// TRANSPOSE that turns a C fragment into the next MMA's `.col` operand.
fn publish_slot(lane: usize, nt: usize, e: usize) -> (usize, usize) {
    let (grp, q4) = (lane >> 2, lane & 3);
    let r = grp + if e >= 2 { 8 } else { 0 };
    let cl = nt * 8 + q4 * 2 + (e & 1);
    (cl, r)
}

// ── 1. every map is a bijection onto the tile it claims ────────────────────

#[test]
fn the_fwd_o_output_map_tiles_every_token_column_once() {
    let mut seen = vec![0u8; C * VD];
    for tid in 0..THREADS {
        for nt in 0..4 {
            for e in 0..4 {
                let (i, v) = fwd_o_slot(tid, nt, e);
                assert!(i < C && v < VD, "tid={tid} nt={nt} e={e} -> ({i},{v})");
                seen[i * VD + v] += 1;
            }
        }
    }
    assert!(
        seen.iter().all(|&c| c == 1),
        "the 4 m-tile x 4 n-quarter split over 16 warps must neither overlap \
         nor leave a hole; holes={} duplicates={}",
        seen.iter().filter(|&&c| c == 0).count(),
        seen.iter().filter(|&&c| c > 1).count()
    );
    // 16 f32 accumulator registers per thread — the whole [64][128] output
    // divided by the block, which is what says the twin needs no smem for it.
    assert_eq!(4 * 4, C * VD / THREADS);
}

#[test]
fn the_gram_map_tiles_the_square_once_and_covers_every_diagonal_block() {
    let mut seen = vec![0u8; C * C];
    let mut diag = vec![0u8; 4 * 16 * 16];
    for tid in 0..THREADS {
        for nt in 0..2 {
            for e in 0..4 {
                let (r, c) = gram_slot(tid, nt, e);
                assert!(r < C && c < C);
                seen[r * C + c] += 1;
                // `wu` copies an element into `Ld` exactly when its row and
                // column share a 16-block; without full coverage of those four
                // blocks a T_jj is built from uninitialised shared memory.
                if (r >> 4) == (c >> 4) {
                    diag[(r >> 4) * 256 + (r & 15) * 16 + (c & 15)] += 1;
                }
            }
        }
    }
    assert!(
        seen.iter().all(|&c| c == 1),
        "holes={} duplicates={}",
        seen.iter().filter(|&&c| c == 0).count(),
        seen.iter().filter(|&&c| c > 1).count()
    );
    assert!(
        diag.iter().all(|&c| c == 1),
        "the four 16x16 diagonal blocks must be written exactly once each"
    );
}

#[test]
fn the_wu_panel_map_tiles_sixty_four_by_sixteen_once() {
    let mut seen = vec![0u8; C * 16];
    for lane in 0..32 {
        for mt in 0..4 {
            for nt in 0..2 {
                for e in 0..4 {
                    let (i, col) = wu_slot(lane, mt, nt, e);
                    assert!(i < C && col < 16);
                    seen[i * 16 + col] += 1;
                }
            }
        }
    }
    assert!(
        seen.iter().all(|&c| c == 1),
        "one warp's [64][16] panel must be exactly its 32 lanes x 4 m-tiles \
         x 2 n-tiles x 4 elements; holes={} duplicates={}",
        seen.iter().filter(|&&c| c == 0).count(),
        seen.iter().filter(|&&c| c > 1).count()
    );
    // 32 f32 registers of panel per thread, and 16 warps x 16 columns covers
    // both 128-column solves — the claim the kernel header makes about why
    // nothing lands in local memory.
    assert_eq!(4 * 2 * 4, C * 16 / 32);
    assert_eq!(16 * 16, VD + KD);
}

#[test]
fn the_publish_map_is_the_transpose_and_a_bijection() {
    let mut seen = vec![0u8; 16 * 16];
    for lane in 0..32 {
        for nt in 0..2 {
            for e in 0..4 {
                let (cl, r) = publish_slot(lane, nt, e);
                // The published element must be the SAME element the C
                // fragment holds, only at [col][row] instead of [row][col].
                let (i, col) = wu_slot(lane, 0, nt, e);
                assert_eq!((cl, r), (col, i), "publish is not the transpose");
                assert!(cl * SX + r < 16 * SX, "panel overflow");
                seen[cl * 16 + r] += 1;
            }
        }
    }
    assert!(seen.iter().all(|&c| c == 1), "the panel must be tiled once");
}

// ── 2. every padded address stays inside the buffer the launcher sized ─────

#[test]
fn every_padded_address_stays_inside_its_buffer() {
    for tid in 0..THREADS {
        for nt in 0..4 {
            for e in 0..4 {
                let (i, v) = fwd_o_slot(tid, nt, e);
                assert!(i * SW + KD - 1 < C * SW, "sq[i][k] overflow");
                assert!(v * SW + KD - 1 < VD * SW, "S^T[v][k] overflow");
                assert!(v * SC + C - 1 < VD * SC, "uc^T[v][l] overflow");
            }
        }
        for nt in 0..2 {
            for e in 0..4 {
                let (r, c) = gram_slot(tid, nt, e);
                assert!(r * SC + c < C * SC, "kq limb / L limb overflow");
                if (r >> 4) == (c >> 4) {
                    assert!(r * SX + (c & 15) < C * SX, "Ld / Tf overflow");
                }
            }
        }
    }
    // The MMA A-operand reads 4 bytes at element `row * stride + col`, so that
    // index must be EVEN or the load is misaligned. Every stride here is even
    // and every column offset is `q*2` or `q*2 + 8`.
    for s in [SW, SC, SX] {
        assert_eq!(
            s % 2,
            0,
            "operand row stride {s} must keep 4-byte alignment"
        );
    }

    // ...and the launcher's byte counts must be the sums the kernels lay out.
    assert_eq!(
        GDN_FWD_O_HOPPER_SMEM as usize,
        2 * (C * SW * 2) + VD * SW * 2 + VD * SC * 2 + C * SC * 2 + C * 4
    );
    assert_eq!(GDN_FWD_O_HOPPER_SMEM, 97_536);
    assert_eq!(
        GDN_WU_HOPPER_SMEM as usize,
        C * SW * 2
            + 2 * (C * SX * 4)
            + 2 * (C * SC * 2)
            + 2 * (C * SX * 2)
            + 2 * (16 * 16 * SX * 2)
            + C * 4
    );
    assert_eq!(GDN_WU_HOPPER_SMEM, 79_104);
}

/// `fwd_o`'s `kq` lo limb ALIASES the dead `sk`; that is only sound while it
/// fits, and it is why the twin's footprint is under the parent's 98 816 B.
/// Two CTAs must also fit H100's 228 KB, which is the whole point of that
/// kernel's `__launch_bounds__(512, 2)`. Compile-time, because all three are
/// properties of constants rather than of any run.
const _: () = assert!(
    C * SC * 2 <= C * SW * 2,
    "the kq lo limb does not fit in sk"
);
const _: () = assert!(GDN_FWD_O_HOPPER_SMEM < 98_816);
const _: () = assert!(2 * GDN_FWD_O_HOPPER_SMEM <= 228 * 1024);

// ── 3. the lever grammar ───────────────────────────────────────────────────

const PARENT: KernelHandle = KernelHandle(7);
const TWIN: KernelHandle = KernelHandle(9);
const ABSENT: KernelHandle = KernelHandle(0);

/// Production geometry (Qwen3.8-27B: kd = vd = 128, chunk 64) is the one
/// combination that must be accepted.
#[test]
fn the_production_geometry_is_accepted() {
    assert_eq!(
        gdn_hopper_remnant_reject(true, false, true, 128, 128, 64),
        None
    );
}

#[test]
fn the_lever_is_off_until_asked_for() {
    assert_eq!(
        gdn_hopper_remnant_reject(false, false, true, 128, 128, 64),
        Some("not requested"),
        "the TC prefill family must default OFF — the parents stay the default"
    );
}

/// The twins read the SPINE'S lever, not a second one of their own.
///
/// This is the bit the launcher hands `gdn_hopper_remnants`, taken here the
/// same way `gdn_prefill_fla` takes it. It pins two things that a fresh
/// `std::env::var("ATLAS_GDN_PREFILL_TC").is_ok()` at this layer would break,
/// and which nothing else in the suite would notice:
///
///  * `ATLAS_GDN_PREFILL_TC=0` is an explicit OFF under the 2026-09-11
///    grammar. A presence check would read it as ON and run a prefill whose
///    twins were enabled and whose state spine was not — neither leg of the
///    A/B the variable exists for;
///  * the COMPILED TARGET's `[defaults] gdn_prefill_tc` is what applies when
///    the variable is absent, so a target that one day declares it true gets
///    the whole family, with no launch script to remember.
///
/// The assertion is the declared default of every target shipped today
/// (`kernels/*/HARDWARE.toml`), which `atlas-kernels`' own
/// `target_defaults.rs` pins against the TOML; what is under test HERE is that
/// the remnants are wired to THAT value.
#[test]
fn the_twins_read_the_spines_resolved_lever() {
    let spine = crate::layers::ops::target_defaults::resolved()
        .gdn_prefill_tc
        .value;
    assert!(
        !spine,
        "no target declares [defaults] gdn_prefill_tc true yet; if one now does,          this test should assert the twins follow it rather than be deleted"
    );
    assert_eq!(
        gdn_hopper_remnant_reject(spine, false, true, 128, 128, 64),
        Some("not requested"),
        "the twins must follow the spine's resolved lever, not a lever of their own"
    );
}

/// Each refusal NAMES itself, so an A/B that silently fell back cannot be
/// mistaken for a lever with no effect (PR #296).
#[test]
fn every_refusal_names_its_guard() {
    for (case, want) in [
        (
            gdn_hopper_remnant_reject(true, true, true, 128, 128, 64),
            "ATLAS_NO_GDN_PREFILL_TC_REMNANTS=1 pins wu/fwd_o to their parents",
        ),
        (
            gdn_hopper_remnant_reject(true, false, false, 128, 128, 64),
            "kernel absent from this image (kernels/hopper only)",
        ),
        (
            gdn_hopper_remnant_reject(true, false, true, 64, 128, 64),
            "head/chunk differs from the compile-time tile (K_DIM=V_DIM=128, CHUNK=64)",
        ),
        (
            gdn_hopper_remnant_reject(true, false, true, 128, 64, 64),
            "head/chunk differs from the compile-time tile (K_DIM=V_DIM=128, CHUNK=64)",
        ),
        (
            gdn_hopper_remnant_reject(true, false, true, 128, 128, 32),
            "head/chunk differs from the compile-time tile (K_DIM=V_DIM=128, CHUNK=64)",
        ),
    ] {
        assert_eq!(case, Some(want));
    }
}

/// A REJECTION MUST BE THE PRE-#928 LAUNCH, byte for byte: the parent handle,
/// the parent's block size, the parent's shared memory. Anything else and the
/// kill switch is not a control arm.
#[test]
fn a_rejected_pick_is_exactly_the_parent_launch() {
    for (killed, present) in [(true, true), (false, false)] {
        let twin = if present { TWIN } else { ABSENT };
        let p = gdn_hopper_remnant_pick(
            true,
            killed,
            PARENT,
            256,
            33_024,
            twin,
            GDN_WU_HOPPER_SMEM,
            128,
            128,
            64,
        );
        assert_eq!(p.kernel.0, PARENT.0);
        assert_eq!(p.block, 256);
        assert_eq!(p.smem, 33_024);
        assert!(p.reject.is_some(), "a parent launch must name its guard");
    }
}

#[test]
fn an_accepted_pick_carries_the_twins_block_and_footprint() {
    let p = gdn_hopper_remnant_pick(
        true,
        false,
        PARENT,
        512,
        98_816,
        TWIN,
        GDN_FWD_O_HOPPER_SMEM,
        128,
        128,
        64,
    );
    assert_eq!(p.kernel.0, TWIN.0);
    assert_eq!(p.block, GDN_HOPPER_REMNANT_BLOCK);
    assert_eq!(p.block, 512, "both twins are 16 warps, every one computing");
    assert_eq!(p.smem, GDN_FWD_O_HOPPER_SMEM);
    assert_eq!(p.reject, None);
}

#[test]
fn the_tile_constants_match_the_kernels() {
    assert_eq!(GDN_HOPPER_DIM, 128);
    assert_eq!(GDN_HOPPER_CHUNK, 64);
    assert_eq!(GDN_HOPPER_CHUNK as usize, C);
    assert_eq!(GDN_HOPPER_DIM as usize, KD);
}
