// SPDX-License-Identifier: AGPL-3.0-only

//! Bit-identity of the fused up-GEMM+mix arm against the stock arm.
//! Split from `hyper_connection_lowrank_tests.rs` (500-LoC cap).

use super::*;

/// Deterministic pseudo-random BF16 bytes. A bit-identity test compares the two
/// arms against EACH OTHER, so it needs well-conditioned inputs, not the real
/// checkpoint — which is why this file does not touch `ATLAS_HC_TEST_DATA`
/// (whose generator, `bench/qwen4_exp`, is not in this tree) and can therefore
/// run anywhere there is a GPU.
fn lcg_bf16(n: usize, seed: u32) -> Vec<u8> {
    let mut s = seed | 1;
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        // ~N(0,1)-ish in [-1, 1), then to BF16 by truncating the f32 mantissa.
        let v = ((s >> 8) as f32 / (1u32 << 23) as f32) - 1.0;
        out.extend_from_slice(&(v.to_bits() >> 16).to_le_bytes()[..2]);
    }
    out
}

fn lcg_f32(n: usize, seed: u32) -> Vec<u8> {
    let mut s = seed | 1;
    let mut out = Vec::with_capacity(n * 4);
    for _ in 0..n {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let v = ((s >> 8) as f32 / (1u32 << 23) as f32) - 1.0;
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    out
}

/// The fused up-GEMM+mix arm must be BIT-IDENTICAL to the stock arm, and this
/// is the only thing that proves it.
///
/// `hc_pre_gemm_matches_reference` cannot: it compares against goldens at
/// `tol_for` (5% of reference RMS — three orders looser than a 1-ulp slip), and
/// its `TILE=12` gives `big_t=96`, under `DM_M_TILE=128`, so `cta_m` is always
/// 0 and the fused epilogue's row arithmetic never executes. This test uses
/// `T=288` so `grid.y == 3` and `cta_m` takes non-zero values INCLUDING a
/// partial final tile, runs BOTH arms in one process through the explicit-bool
/// entry point (the env reader is a `OnceLock`, so a process can only ever see
/// one arm through it), and asserts RAW BYTES.
///
/// Any non-zero diff is a revert, not a tolerance discussion: the entire case
/// for the fusion is that it removes bytes without changing numbers.
#[test]
#[ignore]
fn hc_pre_gemm_fused_up_mix_is_bit_identical() {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4").expect(
        "qwen3.8-flash-next/nvfp4 is not in this build — \
         build with ATLAS_TARGET_MODEL='*' or =qwen3.8-flash-next",
    );
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();

    // Shipping geometry, and a T that is NOT a multiple of the 128-row M-tile.
    let (h, hc, rank) = (2560usize, 4usize, 320usize);
    let big_t = 288usize;
    let hc_dim = hc * h;
    assert!(big_t > 128, "must exceed DM_M_TILE so cta_m is exercised");

    // The fused kernel must be present, or this test would silently compare the
    // stock arm against itself and pass for the wrong reason.
    let k_um = crate::layers::try_kernel(g, "hyper_connection", "hc_pre_up_mix");
    let k_ig = crate::layers::try_kernel(g, "hyper_connection", "hc_inj_gate");
    let k_di = crate::layers::try_kernel(g, "hyper_connection", "hc_down_inj");
    assert!(
        k_um.0 != 0 && k_ig.0 != 0 && k_di.0 != 0,
        "hc_pre_up_mix / hc_inj_gate / hc_down_inj missing from the module — \
         the fused arms would silently degrade and this test would be vacuous"
    );

    let streams = upload(g, &lcg_f32(big_t * hc_dim, 0x51ED));
    let w = HcLowRank {
        norm_w: upload(g, &lcg_bf16(hc_dim, 0xA11CE)),
        down_w: upload(g, &lcg_bf16(rank * hc_dim, 0xD0)),
        up_w: upload(g, &lcg_bf16(hc_dim * rank, 0x0FF)),
        inject_w: upload(g, &lcg_bf16(hc * hc_dim, 0x1B7)),
        rank,
    };
    let scratch = g.alloc(big_t * (2 * hc_dim + rank + hc) * 2).unwrap();

    // Separate destinations per arm, so a MISSING write shows up as a diff
    // rather than as a stale value left behind by the previous arm.
    let y_a = g.alloc(big_t * h * 2).unwrap();
    let y_b = g.alloc(big_t * h * 2).unwrap();
    let inj_a = g.alloc(big_t * hc * 4).unwrap();
    let inj_b = g.alloc(big_t * hc * 4).unwrap();

    // Reference arm first, then EVERY fused combination against it. Running
    // all four in one process is the whole point: the env readers are
    // `OnceLock`s, so a process can only ever observe a single arm through the
    // production wrapper. The `both` arm is what ships, but the two singles
    // are here so a failure says WHICH fusion broke rather than just "the
    // fused path".
    let go = |up: bool, di: bool, y, inj| {
        crate::layers::ops::hyper_connection_lowrank::gemm::hc_pre_gemm_fused(
            g,
            streams,
            &w,
            y,
            inj,
            scratch,
            big_t as u32,
            h as u32,
            hc as u32,
            1e-6,
            /* inject */ true,
            /* use_cublas */ false,
            /* row_exact */ false,
            up,
            di,
            stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
    };

    go(false, false, y_a, inj_a);
    let ya = download_raw(g, y_a, big_t * h * 2);
    let ia = download_raw(g, inj_a, big_t * hc * 4);
    // A run where the reference output is all zeros would pass vacuously.
    assert!(
        ya.iter().any(|&b| b != 0),
        "stock arm produced an all-zero y_out — the comparison would be vacuous"
    );
    assert!(
        ia.iter().any(|&b| b != 0),
        "stock arm produced an all-zero inj_out — the comparison would be vacuous"
    );

    for (up, di, name) in [
        (true, false, "up_mix"),
        (false, true, "down_inj"),
        (true, true, "both"),
    ] {
        go(up, di, y_b, inj_b);
        let yb = download_raw(g, y_b, big_t * h * 2);
        let ib = download_raw(g, inj_b, big_t * hc * 4);

        let ydiff = ya.iter().zip(&yb).filter(|(x, y)| x != y).count();
        let idiff = ia.iter().zip(&ib).filter(|(x, y)| x != y).count();
        println!(
            "{name} vs stock (T={big_t}): y bytes differing {ydiff}/{}, inj {idiff}/{}",
            ya.len(),
            ia.len()
        );
        assert_eq!(ydiff, 0, "{name}: not bit-identical in y_out");
        assert_eq!(idiff, 0, "{name}: not bit-identical in inj_out");
    }
}
