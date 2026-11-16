// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: a host simulation of the index maps of
//! `kernels/ada/common/gdn_chunk_tc_ada.cu`: every fragment-ownership, store
//! and swizzle map must be a bijection over its tile, and the two operand
//! permutations (W's column interleave, K's row order) must hand each lane the
//! TF32 A-fragment pair (t, t + 4). The arithmetic of each map is the kernel's,
//! line for line; a hole or a double write here is a wrong element on the GPU.

// Index loops mirror the kernel's index arithmetic.
#![allow(clippy::needless_range_loop)]

fn assert_bijection(
    cells: impl IntoIterator<Item = (usize, usize)>,
    rows: usize,
    cols: usize,
    what: &str,
) {
    let mut seen = vec![false; rows * cols];
    let mut n = 0;
    for (r, c) in cells {
        assert!(
            r < rows && c < cols,
            "{what}: ({r}, {c}) out of the {rows}x{cols} tile"
        );
        assert!(!seen[r * cols + c], "{what}: ({r}, {c}) written twice");
        seen[r * cols + c] = true;
        n += 1;
    }
    assert_eq!(n, rows * cols, "{what}: holes");
}

/// Rows of an m16n8 accumulator fragment element `e` for lane group `lane_g`.
fn frag_row(base: usize, lane: usize, e: usize) -> usize {
    base + (lane >> 2) + if e >= 2 { 8 } else { 0 }
}

/// Columns of an m16n8 accumulator fragment element `e` for lane `lane`.
fn frag_col(base: usize, lane: usize, e: usize) -> usize {
    base + 2 * (lane & 3) + (e & 1)
}

#[test]
fn state_fragments_own_each_state_element_once() {
    // sp::state_body Phase 0 / Z: strip, warp = dt * 4 + s_idx, lane, m, e.
    let mut cells = Vec::new();
    for strip in 0..8 {
        for warp in 0..8 {
            let (s_idx, dt) = (warp % 4, warp / 4);
            for lane in 0..32 {
                for m in 0..2 {
                    for e in 0..4 {
                        let k = frag_row(s_idx * 32 + m * 16, lane, e);
                        let d = frag_col(strip * 16 + dt * 8, lane, e);
                        cells.push((k, d));
                    }
                }
            }
        }
    }
    assert_bijection(cells, 128, 128, "state fragments");
}

#[test]
fn h_chunk_and_v_new_stores_cover_their_blocks() {
    // Phase B coop write: strip, sit, v < 128 -> d = strip*16 + v/8, k = sit*32 + (v%8)*4 + j.
    let mut hc = Vec::new();
    for strip in 0..8 {
        for sit in 0..4 {
            for v in 0..128 {
                for j in 0..4 {
                    hc.push((strip * 16 + v / 8, sit * 32 + (v % 8) * 4 + j));
                }
            }
        }
    }
    assert_bijection(hc, 128, 128, "h_chunk [d][k]");
    // Phase D: row = s_idx*16 + lane_g (+8), col = strip*16 + dt*8 + 2*lane_t (+1).
    let mut vn = Vec::new();
    for strip in 0..8 {
        for warp in 0..8 {
            let (s_idx, dt) = (warp % 4, warp / 4);
            for lane in 0..32 {
                for e in 0..4 {
                    vn.push((
                        frag_row(s_idx * 16, lane, e),
                        frag_col(strip * 16 + dt * 8, lane, e),
                    ));
                }
            }
        }
    }
    assert_bijection(vn, 64, 128, "v_new [t][d]");
}

#[test]
fn prepare_and_output_stores_cover_their_tiles() {
    // compute_store_wu_panel: panel, warp, v < 32 -> row = warp*16 + v/2,
    // col = panel*16 + (v%2)*8 + j.
    let mut wu = Vec::new();
    for panel in 0..8 {
        for warp in 0..4 {
            for v in 0..32 {
                for j in 0..8 {
                    wu.push((warp * 16 + v / 2, panel * 16 + (v % 2) * 8 + j));
                }
            }
        }
    }
    assert_bijection(wu, 64, 128, "W/U [t][k]");
    // output_job: panel, warp, lane, nt, e -> row = warp*16 + lane_g (+8),
    // col = panel*16 + nt*8 + 2*lane_t (+1).
    let mut out = Vec::new();
    for panel in 0..8 {
        for warp in 0..4 {
            for lane in 0..32 {
                for nt in 0..2 {
                    for e in 0..4 {
                        out.push((
                            frag_row(warp * 16, lane, e),
                            frag_col(panel * 16 + nt * 8, lane, e),
                        ));
                    }
                }
            }
        }
    }
    assert_bijection(out, 64, 128, "output [t][d]");
}

#[test]
fn w_interleave_feeds_the_tf32_a_fragment() {
    // compute_store_wu_panel<true>'s 32-bit word shuffle over 8 BF16 columns.
    let x: [u32; 8] = [0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17];
    let pk = |lo: u32, hi: u32| lo | (hi << 16);
    let (px, py, pz, pw) = (
        pk(x[0], x[1]),
        pk(x[2], x[3]),
        pk(x[4], x[5]),
        pk(x[6], x[7]),
    );
    let words = [
        (px & 0x0000_ffff) | (pz << 16),
        (px >> 16) | (pz & 0xffff_0000),
        (py & 0x0000_ffff) | (pw << 16),
        (py >> 16) | (pw & 0xffff_0000),
    ];
    let stored: Vec<u32> = words.iter().flat_map(|w| [w & 0xffff, w >> 16]).collect();
    assert_eq!(stored, vec![x[0], x[4], x[1], x[5], x[2], x[6], x[3], x[7]]);
    // ldmatrix hands lane_t the pair at (2t, 2t+1) = logical columns (t, t+4):
    // unpack low -> a0 (col t), high -> a2 (col t+4), the m16n8k8 A layout.
    for t in 0..4 {
        assert_eq!((stored[2 * t], stored[2 * t + 1]), (x[t], x[t + 4]));
    }
}

#[test]
fn k_row_permutation_pairs_t_with_t_plus_4() {
    // Bf16KView::physical_row within each group of eight token rows.
    let phys = |r: usize| (r & !7) + ((r & 3) << 1) + ((r & 7) >> 2);
    let mut inverse = [usize::MAX; 8];
    for r in 0..8 {
        assert_eq!(inverse[phys(r)], usize::MAX, "row {r} collides");
        inverse[phys(r)] = r;
    }
    // ldmatrix.x2.trans packs physical rows (2t, 2t+1) into one register.
    for t in 0..4 {
        assert_eq!((inverse[2 * t], inverse[2 * t + 1]), (t, t + 4));
    }
    assert_eq!(phys(13), 8 + phys(5));
}

#[test]
fn swizzles_are_row_permutations() {
    fn check(cols: usize, f: impl Fn(usize, usize) -> usize, what: &str) {
        for row in 0..64 {
            let mut seen = vec![false; cols];
            for col in 0..cols {
                let c = f(row, col);
                assert!(c < cols && !seen[c], "{what}: row {row} col {col}");
                seen[c] = true;
            }
        }
    }
    // SmemTile<STRIDE>.
    check(64, |r, c| c ^ (((r & 3) << 3) | (r & 4)), "SmemTile<64>");
    check(16, |r, c| c ^ (((r >> 1) & 3) << 2), "SmemTile<16>");
    // Bf16Tile<STRIDE>.
    for s in [16usize, 32, 128] {
        check(s, move |r, c| c ^ ((r & (s / 8 - 1)) << 3), "Bf16Tile");
    }
    // SnapView (32 floats per value row).
    check(32, |r, c| c ^ ((r & 7) << 2), "SnapView");
    // Bf16WView / Bf16KView column swizzle over 128.
    check(
        128,
        |r, c| (c & !63) + ((((c & 63) >> 3) ^ (r & 7)) << 3) + (c & 7),
        "W/K view",
    );
    // The WU output stage's 16-wide swizzle.
    check(
        16,
        |r, c| (((c >> 3) ^ (r & 1)) << 3) | (c & 7),
        "WU output stage",
    );
}
