// SPDX-License-Identifier: AGPL-3.0-only

//! Bit-identity of the fused mHC arms against the stock arms.
//! Split from `hyper_connection_lowrank_tests.rs` (500-LoC cap).

use super::*;
use crate::layers::ops::hc_post_lowrank;
use crate::layers::ops::hyper_connection_post_fold::HcDeferredPost;

/// Deterministic pseudo-random BF16 bytes. A bit-identity test compares the two
/// arms against EACH OTHER, so it needs well-conditioned inputs, not the real
/// checkpoint — which is why this file does not touch `AVAROK_HC_TEST_DATA`
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
    let set = avarok_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4").expect(
        "qwen3.8-flash-next/nvfp4 is not in this build — \
         build with AVAROK_TARGET_MODEL='*' or =qwen3.8-flash-next",
    );
    let gpu =
        spark_runtime::cuda_backend::AvarokCudaBackend::new(0, &set.modules).expect("CUDA backend");
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
            /* deferred */ None,
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

/// The folded `hc_post` must be BIT-IDENTICAL to launching the real `hc_post`
/// and then the stock stage — in the HIGHWAY ITSELF, not just in the collapse's
/// outputs. The highway is what `hc_post` writes, so a fold that got `y_out`
/// right and `streams` wrong would still corrupt every later layer.
///
/// `T = 2248` breaks three things at once if they are wrong:
///   * it CROSSES the 2048-token slab (two passes), so the per-slab
///     `block_out.offset(t0*H*2)` and `inj.offset(t0*hc*4)` arithmetic runs at a
///     non-zero `t0`;
///   * `2248 % 128 == 8`, a partial final M-tile in the down/up GEMMs;
///   * `H = 2560` at block 1024 is two full register slots plus a HALF one
///     (`tid < 512`), so the stage kernel's registered head has a partial
///     stride.
///
/// Both arms run in ONE process through the explicit-deferred entry point: the
/// env readers are `OnceLock`s, so a process can only ever observe one arm
/// through `hc_fuse_post()`.
///
/// Any non-zero diff is a revert, not a tolerance discussion.
#[test]
#[ignore]
fn hc_post_folded_into_stage_is_bit_identical() {
    let set = avarok_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4").expect(
        "qwen3.8-flash-next/nvfp4 is not in this build — \
         build with AVAROK_TARGET_MODEL='*' or =qwen3.8-flash-next",
    );
    let gpu =
        spark_runtime::cuda_backend::AvarokCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();

    let (h, hc, rank) = (2560usize, 4usize, 320usize);
    let big_t = 2248usize;
    let hc_dim = hc * h;
    assert!(
        big_t > spark_runtime::buffers::hc_gemm_slab(),
        "T must cross the slab so the per-slab block_out/inj offsets are exercised"
    );

    // Present, or the fold would silently degrade and this test would compare
    // the stock arm against itself.
    let k_fold = crate::layers::try_kernel(g, "hyper_connection", "hc_pre_stage_bf16_post");
    assert!(
        k_fold.0 != 0,
        "hc_pre_stage_bf16_post missing from the module — the fold would degrade \
         and this test would be vacuous"
    );
    let k_post = g.kernel("hyper_connection", "hc_post").unwrap();

    let pristine = lcg_f32(big_t * hc_dim, 0x51ED);
    let block_out = upload(g, &lcg_bf16(big_t * h, 0xB10C));
    let inj_src = lcg_f32(big_t * hc, 0x1A5);
    let w = HcLowRank {
        norm_w: upload(g, &lcg_bf16(hc_dim, 0xA11CE)),
        down_w: upload(g, &lcg_bf16(rank * hc_dim, 0xD0)),
        up_w: upload(g, &lcg_bf16(hc_dim * rank, 0x0FF)),
        inject_w: upload(g, &lcg_bf16(hc * hc_dim, 0x1B7)),
        rank,
    };
    let scratch = g.alloc(big_t * (2 * hc_dim + rank + hc) * 2).unwrap();

    let collapse = |streams, y, inj_out, deferred, up: bool, di: bool| {
        crate::layers::ops::hyper_connection_lowrank::gemm::hc_pre_gemm_fused(
            g,
            streams,
            &w,
            y,
            inj_out,
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
            deferred,
            stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
    };

    for (up, di, name) in [(false, false, "stock"), (true, true, "up_mix+down_inj")] {
        // ── Reference: the real hc_post, then the stock stage ──
        let s_a = upload(g, &pristine);
        let inj_a_in = upload(g, &inj_src);
        let y_a = g.alloc(big_t * h * 2).unwrap();
        let inj_a = g.alloc(big_t * hc * 4).unwrap();
        hc_post_lowrank(
            g,
            k_post,
            block_out,
            s_a,
            inj_a_in,
            s_a,
            big_t as u32,
            h as u32,
            hc as u32,
            stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        collapse(s_a, y_a, inj_a, None, up, di);

        let s_ref = download_raw(g, s_a, big_t * hc_dim * 4);
        let y_ref = download_raw(g, y_a, big_t * h * 2);
        let i_ref = download_raw(g, inj_a, big_t * hc * 4);

        // Vacuity guards. The last is the one specific to THIS fusion: were
        // `hc_post` a no-op, the fold would match trivially.
        assert!(y_ref.iter().any(|&b| b != 0), "{name}: y_out all zero");
        assert!(i_ref.iter().any(|&b| b != 0), "{name}: inj_out all zero");
        assert_ne!(
            s_ref, pristine,
            "{name}: hc_post did not change the highway — the comparison is vacuous"
        );

        // ── Arm 1: folded, with a DISTINCT injection buffer ──
        let s_b = upload(g, &pristine);
        let inj_b_in = upload(g, &inj_src);
        let y_b = g.alloc(big_t * h * 2).unwrap();
        let inj_b = g.alloc(big_t * hc * 4).unwrap();
        collapse(
            s_b,
            y_b,
            inj_b,
            Some(HcDeferredPost::new(block_out, inj_b_in)),
            up,
            di,
        );

        // ── Arm 2: folded, with the deferred injection vector ALIASING
        // `inj_out` — which is what production does, because both sites write
        // `ctx.buffers.hc_post()`. Crossing the slab is what makes this
        // meaningful: if the stage read `inj` AFTER the mix had overwritten it,
        // slab 2 would diverge here and nowhere else.
        let s_c = upload(g, &pristine);
        let inj_c = upload(g, &inj_src);
        let y_c = g.alloc(big_t * h * 2).unwrap();
        collapse(
            s_c,
            y_c,
            inj_c,
            Some(HcDeferredPost::new(block_out, inj_c)),
            up,
            di,
        );

        for (tag, s, y, i) in [
            ("distinct-inj", s_b, y_b, inj_b),
            ("aliased-inj", s_c, y_c, inj_c),
        ] {
            let sd = download_raw(g, s, big_t * hc_dim * 4);
            let yd = download_raw(g, y, big_t * h * 2);
            let id = download_raw(g, i, big_t * hc * 4);
            let sdiff = s_ref.iter().zip(&sd).filter(|(x, y)| x != y).count();
            let ydiff = y_ref.iter().zip(&yd).filter(|(x, y)| x != y).count();
            let idiff = i_ref.iter().zip(&id).filter(|(x, y)| x != y).count();
            println!(
                "{name}/{tag} (T={big_t}): streams {sdiff}/{}, y {ydiff}/{}, inj {idiff}/{}",
                s_ref.len(),
                y_ref.len(),
                i_ref.len()
            );
            assert_eq!(sdiff, 0, "{name}/{tag}: highway not bit-identical");
            assert_eq!(ydiff, 0, "{name}/{tag}: y_out not bit-identical");
            assert_eq!(idiff, 0, "{name}/{tag}: inj_out not bit-identical");
        }
    }
}
