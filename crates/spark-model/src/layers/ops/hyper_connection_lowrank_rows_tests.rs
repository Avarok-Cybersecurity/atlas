// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! Parity and microbench tests for the decode-rows arm (`hyper_connection_lowrank_rows.rs`),
//! split out of `hyper_connection_lowrank_tests.rs` under the 500-line cap. Run with
//! `--test-threads=1` and `ATLAS_HC_TEST_DATA` set (cuBLASLt concurrency artifact otherwise).

use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::hyper_connection_lowrank_tests::*;

/// The decode-rows arm (default on, `ATLAS_HC_DECODE_ROWS=0` to disable; T <= 8): `hc_pre_stage` +
/// `hc_dec_down` + `hc_dec_up`, held to the split arm's TIGHT bound at the
/// fixture's T=8 and again at T=3 (the MTP two-draft verify width), where the
/// first three tokens of the fixture are an exact prefix golden because the
/// collapse is per-token independent.
#[test]
#[ignore]
fn hc_rows_matches_reference() {
    let f = Fixture::load();
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4").expect(
        "qwen3.8-flash-next/nvfp4 is not in this build — \
         build with ATLAS_TARGET_MODEL='*' or =qwen3.8-flash-next",
    );
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let (t, h, hc) = (f.tokens, f.h, f.hc);
    for name in ["hc_pre_stage", "hc_dec_down", "hc_dec_up"] {
        let k = g.kernel("hyper_connection", name).unwrap();
        assert!(k.0 != 0, "{name} resolved to handle 0");
    }
    assert!(
        super::hyper_connection_lowrank_rows::hc_decode_rows_shape_ok(
            t as u32,
            h as u32,
            hc as u32,
            f.rank as u32
        ),
        "fixture shape is outside the decode-rows contract"
    );
    let streams = upload(g, &f.bytes("streams"));
    let y_out = g.alloc(t * h * 2).unwrap();
    let inj_out = g.alloc(t * hc * 4).unwrap();
    let scratch = g.alloc(64 * (hc * h + f.rank) * 4).unwrap();

    for rows in [t, 3usize.min(t)] {
        println!("decode-rows arm at T={rows}:");
        for site in ["attn", "mlp"] {
            let w = site_weights(g, &f, site, true);
            let want_mixed: Vec<f32> = f.f32s(&format!("{site}_mixed"))[..rows * h].to_vec();
            let want_inj: Vec<f32> = f.f32s(&format!("{site}_inj"))[..rows * hc].to_vec();
            super::hyper_connection_lowrank_rows::hc_pre_rows(
                g,
                streams,
                &w,
                y_out,
                inj_out,
                scratch,
                rows as u32,
                h as u32,
                hc as u32,
                f.eps,
                true,
                stream,
            )
            .unwrap();
            g.synchronize(stream).unwrap();
            println!("{site}_hyper_connection (rows arm, T={rows}):");
            compare(
                "mixed_input",
                &download_bf16(g, y_out, rows * h),
                &want_mixed,
                tol_for(&want_mixed),
            );
            compare(
                "injection_weights",
                &download_f32(g, inj_out, rows * hc),
                &want_inj,
                tol_for(&want_inj),
            );
        }
        let w_head = site_weights(g, &f, "head", false);
        let want_head: Vec<f32> = f.f32s("head_mixed")[..rows * h].to_vec();
        super::hyper_connection_lowrank_rows::hc_pre_rows(
            g,
            streams,
            &w_head,
            y_out,
            DevicePtr::NULL,
            scratch,
            rows as u32,
            h as u32,
            hc as u32,
            f.eps,
            false,
            stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        println!("hyper_connection_mixer (rows arm, T={rows}):");
        compare(
            "mixed_input",
            &download_bf16(g, y_out, rows * h),
            &want_head,
            tol_for(&want_head),
        );
    }
}

/// Row-exactness of the decode-rows arm: every row of a T=3 launch pair must
/// be BYTE-IDENTICAL to the same row computed alone at T=1. The kernels keep
/// one accumulator per token and reduce each in the same lane order whatever
/// T is, so a K-row verify body that batches the attention layers' sites at
/// T=K (verify_rows_hc.rs) reproduces the one-row bodies bit for bit at the
/// hyper-connection sites.
#[test]
#[ignore]
fn hc_rows_t3_rows_equal_t1_rows() {
    let f = Fixture::load();
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4").expect(
        "qwen3.8-flash-next/nvfp4 is not in this build — \
         build with ATLAS_TARGET_MODEL='*' or =qwen3.8-flash-next",
    );
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let (t, h, hc) = (f.tokens, f.h, f.hc);
    let rows = 3usize.min(t);
    let streams = upload(g, &f.bytes("streams"));
    let y_t3 = g.alloc(rows * h * 2).unwrap();
    let inj_t3 = g.alloc(rows * hc * 4).unwrap();
    let y_t1 = g.alloc(h * 2).unwrap();
    let inj_t1 = g.alloc(hc * 4).unwrap();
    let scratch = g.alloc(64 * (hc * h + f.rank) * 4).unwrap();
    for site in ["attn", "mlp"] {
        let w = site_weights(g, &f, site, true);
        super::hyper_connection_lowrank_rows::hc_pre_rows(
            g,
            streams,
            &w,
            y_t3,
            inj_t3,
            scratch,
            rows as u32,
            h as u32,
            hc as u32,
            f.eps,
            true,
            stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        let mut y3 = vec![0u8; rows * h * 2];
        let mut i3 = vec![0u8; rows * hc * 4];
        g.copy_d2h(y_t3, &mut y3).unwrap();
        g.copy_d2h(inj_t3, &mut i3).unwrap();
        for r in 0..rows {
            super::hyper_connection_lowrank_rows::hc_pre_rows(
                g,
                streams.offset(r * hc * h * 4),
                &w,
                y_t1,
                inj_t1,
                scratch,
                1,
                h as u32,
                hc as u32,
                f.eps,
                true,
                stream,
            )
            .unwrap();
            g.synchronize(stream).unwrap();
            let mut y1 = vec![0u8; h * 2];
            let mut i1 = vec![0u8; hc * 4];
            g.copy_d2h(y_t1, &mut y1).unwrap();
            g.copy_d2h(inj_t1, &mut i1).unwrap();
            assert!(
                y3[r * h * 2..(r + 1) * h * 2] == y1[..],
                "{site}: mixed_input row {r} differs T=3 vs T=1"
            );
            assert!(
                i3[r * hc * 4..(r + 1) * hc * 4] == i1[..],
                "{site}: injection row {r} differs T=3 vs T=1"
            );
        }
        println!("{site}: T=3 rows byte-identical to T=1 rows ({rows} rows)");
    }
}

/// Microbench of the decode-rows arm (2026-09-09): wall time per
/// `hc_pre_rows` call (hc_pre_stage + hc_dec_down + hc_dec_up) at T=3 on the
/// attn site, 500 launches after a warmup. A number to compare kernel
/// variants against, beside the reference test above.
#[test]
#[ignore]
fn hc_rows_microbench() {
    let f = Fixture::load();
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 is not in this build");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let (h, hc) = (f.h, f.hc);
    let streams = upload(g, &f.bytes("streams"));
    let y_out = g.alloc(8 * h * 2).unwrap();
    let inj_out = g.alloc(8 * hc * 4).unwrap();
    let scratch = g.alloc(64 * (hc * h + f.rank) * 4).unwrap();
    // 32 device copies of the site's weights (~10.5 MB each = 336 MB, more
    // than the L2) cycled launch to launch, so the numbers are DRAM-streaming
    // like the 104 sites of a real step, not L2-warm re-reads of one site.
    let copies: Vec<_> = (0..32).map(|_| site_weights(g, &f, "attn", true)).collect();
    for rows in [3u32, 1u32] {
        for w in copies.iter().take(8) {
            super::hyper_connection_lowrank_rows::hc_pre_rows(
                g, streams, w, y_out, inj_out, scratch, rows, h as u32, hc as u32, f.eps, true,
                stream,
            )
            .unwrap();
        }
        g.synchronize(stream).unwrap();
        let n = 512;
        let t0 = std::time::Instant::now();
        for i in 0..n {
            let w = &copies[i % copies.len()];
            super::hyper_connection_lowrank_rows::hc_pre_rows(
                g, streams, w, y_out, inj_out, scratch, rows, h as u32, hc as u32, f.eps, true,
                stream,
            )
            .unwrap();
        }
        g.synchronize(stream).unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
        let mb = (2.0 * f.rank as f64 * (hc * h) as f64 * 2.0 + (hc * hc * h) as f64 * 2.0) / 1e6;
        println!(
            "hc_pre_rows attn site T={rows}: {us:.1} us per call (3 launches, {mb:.1} MB of weights = {:.0} GB/s)",
            mb / us * 1e3
        );
    }
}
