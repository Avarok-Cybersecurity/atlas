// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: unit test of the RetroInfer k-means kernel source: its `#define` constants
//! match the host's geometry, the assign kernel's static shared memory fits sm_89, and every
//! entry point the launcher looks up is defined (split from `tests.rs` for the 500-line cap).

use super::geometry::{self, TILES};

#[test]
fn kernel_constants_match_and_static_smem_fits() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/ada/common/retro_kmeans_ada.cu"
    );
    let src = std::fs::read_to_string(path).expect("retro_kmeans_ada.cu");
    let define = |name: &str| -> u32 {
        let line = src
            .lines()
            .find(|l| l.starts_with(&format!("#define {name} ")))
            .unwrap_or_else(|| panic!("#define {name} missing"));
        line.split_whitespace()
            .nth(2)
            .and_then(|v| v.parse().ok())
            .expect("numeric define")
    };
    assert_eq!(define("RK_D"), geometry::HEAD_DIM);
    assert_eq!(define("RK_TILES"), TILES);
    assert_eq!(define("RK_MEAN_CHUNKS"), geometry::MEAN_CHUNKS);
    assert_eq!(define("RK_ASSIGN_ROWS"), 128);
    // retro_km_assign: the centroid chunk plus the mean, against ptxas's 48 KiB sm_89 ceiling.
    let smem = define("RK_ASSIGN_CHUNK") * (geometry::HEAD_DIM + 8) * 2 + geometry::HEAD_DIM * 4;
    assert!(smem <= 48 * 1024, "{smem}");
    for e in [
        "retro_km_mean_partial(",
        "retro_km_mean_final(",
        "retro_km_init(",
        "retro_km_assign(",
        "retro_km_count(",
        "retro_km_scan(",
        "retro_km_scatter(",
        "retro_km_update(",
    ] {
        assert!(src.contains(e), "{e}");
    }
}
