// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only guard binding the MoE grouped-GEMM launch geometry to the kernel's
//! N-tile width.
//!
//! The two facts that must agree:
//!   * Rust: every caller of `moe_w4a16_grouped_gemm_ptrtable` and
//!     `..._ptrtable_t` goes through an `ops::` wrapper that launches
//!     `grid.x = ceil(n_out / 128)`.
//!   * CUDA: those two entries in `kernels/gb10/common/moe_w4a16_grouped_gemm.cu`
//!     must therefore own 128 columns per CTA.
//!
//! They silently diverged: the kernel used `N_TILE 64` while the launcher
//! divided by 128, so the top half of the N range was never written and stayed
//! at the caller's memset zero. On Nemotron-H Lightning-30B that halved the
//! routed-expert contribution in live prefill (grouped UP wrote 960 of 1856
//! columns, grouped DOWN 1344 of 2688). Nothing in CI noticed, because the
//! divisor and the tile width lived in different languages.
//!
//! This test asserts both halves and cross-checks them against each other, so
//! changing one without the other fails in CPU-only CI.
//!
//! Bit-exact numerics for the retuned kernel live in the GPU oracle
//! `examples/moe_grouped_ntile_microtest.rs` (byte-parity vs the 64-wide
//! kernel on its own correct grid, plus a half-zero negative control).

use spark_model::layers::ops;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

/// The divisor every `ops::moe_w4a16_grouped_gemm_ptrtable*` wrapper uses for
/// `grid.x`, and therefore the N-tile width the CUDA entries must have.
const PTRTABLE_N_TILE: u32 = 128;

/// Nemotron-H Lightning-30B A3B routed-expert shapes.
const LIGHTNING_UP_N: u32 = 1856; // ceil(1856/128) = 15
const LIGHTNING_DOWN_N: u32 = 2688; // ceil(2688/128) = 21

const KERNEL_SRC: &str = include_str!("../../../kernels/gb10/common/moe_w4a16_grouped_gemm.cu");

fn dummy(gpu: &MockGpuBackend) -> DevicePtr {
    gpu.alloc(256).unwrap()
}

/// Launch both pointer-table wrappers and return their observed `grid.x`.
///
/// Both entries (`_ptrtable` and `_ptrtable_t`) reach the GPU through these
/// two wrappers, so covering both covers every production launch of the two
/// kernels this fix retunes.
fn observed_grid_x(n_out: u32) -> Vec<u32> {
    let gpu = MockGpuBackend::new();
    let g: &dyn GpuBackend = &gpu;
    let k = KernelHandle(1);
    let p = dummy(&gpu);
    let (ne, kk, mt, s) = (128u32, 2688u32, 1u32, 0u64);
    ops::moe_w4a16_grouped_gemm_ptrtable(g, k, p, p, p, p, p, p, p, ne, n_out, kk, mt, s).unwrap();
    ops::moe_w4a16_grouped_gemm_ptrtable_n128(g, k, p, p, p, p, p, p, p, ne, n_out, kk, mt, s)
        .unwrap();
    gpu.launches_snapshot().iter().map(|l| l.grid[0]).collect()
}

#[test]
fn ptrtable_wrappers_launch_a_128_wide_n_grid() {
    for n_out in [LIGHTNING_UP_N, LIGHTNING_DOWN_N, 128, 129, 1024] {
        let want = n_out.div_ceil(PTRTABLE_N_TILE);
        for (i, got) in observed_grid_x(n_out).into_iter().enumerate() {
            assert_eq!(
                got, want,
                "wrapper #{i} launched grid.x={got} for n_out={n_out}; the CUDA \
                 ptrtable entries own {PTRTABLE_N_TILE} columns per CTA, so it must be {want}"
            );
        }
    }
    // The exact numbers the defect got wrong, spelled out so a regression is
    // legible in the failure message rather than arithmetic.
    assert_eq!(LIGHTNING_UP_N.div_ceil(PTRTABLE_N_TILE), 15);
    assert_eq!(LIGHTNING_DOWN_N.div_ceil(PTRTABLE_N_TILE), 21);
}

#[test]
fn common_ptrtable_entries_use_the_matching_cuda_tile_width() {
    // The tile width the CUDA side actually compiles with.
    let define = KERNEL_SRC
        .lines()
        .find_map(|l| l.strip_prefix("#define N_TILE_PT "))
        .expect("kernels/gb10/common/moe_w4a16_grouped_gemm.cu must define N_TILE_PT");
    let n_tile_pt: u32 = define.trim().parse().expect("N_TILE_PT must be an integer");
    assert_eq!(
        n_tile_pt, PTRTABLE_N_TILE,
        "N_TILE_PT ({n_tile_pt}) disagrees with the grid.x divisor the ops:: \
         wrappers use ({PTRTABLE_N_TILE}) — the launcher would leave \
         (N_out - grid.x*{n_tile_pt}) output columns unwritten"
    );

    // Both pointer-table entries must derive their column origin from
    // N_TILE_PT, and neither may still use the legacy 64-wide N_TILE.
    let bodies: Vec<&str> = KERNEL_SRC
        .split("extern \"C\" __global__ void moe_w4a16_grouped_gemm")
        .collect();
    for name in ["_ptrtable(", "_ptrtable_t("] {
        let body = bodies
            .iter()
            .find(|b| b.starts_with(name))
            .unwrap_or_else(|| panic!("entry moe_w4a16_grouped_gemm{name} not found"));
        assert!(
            body.contains("cta_n = blockIdx.x * N_TILE_PT"),
            "moe_w4a16_grouped_gemm{name} must take its column origin from N_TILE_PT"
        );
        assert!(
            !body.contains("N_TILE +") && !body.contains("* N_TILE;"),
            "moe_w4a16_grouped_gemm{name} still references the 64-wide N_TILE"
        );
    }

    // The legacy stacked-buffer entry is deliberately left at N_TILE 64: its
    // only caller (atlas-spark-bench/benches/moe.rs) launches the matching
    // 64-wide grid, and the GPU oracle uses it as the unmodified reference.
    assert!(
        KERNEL_SRC.contains("#define N_TILE 64"),
        "the legacy moe_w4a16_grouped_gemm entry must stay 64-wide"
    );
}
