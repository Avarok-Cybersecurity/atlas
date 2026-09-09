// SPDX-License-Identifier: AGPL-3.0-only

//! GPU parity for the Qwen low-rank mHC against the reference module.
//!
//! `hyper_connection.cu` and `hyper_connection_lowrank.rs` were written from
//! `modeling_qwen4_exp.py` and, until this file existed, compared to nothing.
//! They are about to be wired into all 48 layers of a 180B model whose every
//! failure mode is fluent-and-wrong, so they get held to a number first.
//!
//! Fixtures come from `bench/qwen4_exp/hc_golden.py`, which runs the REAL
//! `Qwen4ExpTextGatedResidual` on REAL checkpoint weights — transformers
//! 5.16.1 ships `qwen4_exp` natively and is byte-identical to the vendored
//! `bench/qwen4_exp/ref/modeling_qwen4_exp.py`.
//!
//! GPU test: `#[ignore]` per repo convention (CI is CPU-only). Run with
//! ```text
//! ATLAS_HC_TEST_DATA=/tank/atlas-testdata/qwen4exp_hc \
//!   cargo test -p spark-model --release hc_lowrank -- --ignored --nocapture
//! ```

use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;
use crate::layers::qwen3_attention::HcLowRank;

#[path = "hyper_connection_verify_tests.rs"]
mod verify;

struct Fixture {
    dir: String,
    hc: usize,
    h: usize,
    rank: usize,
    eps: f32,
    tokens: usize,
}

impl Fixture {
    fn load() -> Self {
        let dir = std::env::var("ATLAS_HC_TEST_DATA").expect(
            "set ATLAS_HC_TEST_DATA — generate with \
             `python3 -u bench/qwen4_exp/hc_golden.py --bin-dir <dir>`",
        );
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/meta.json")).unwrap())
                .unwrap();
        // The fixture records which RMSNorm convention it was generated
        // under. `Qwen4ExpTextRMSNorm` is offset-from-1 while the
        // `RMSNormGated` beside it in the same block is not, so a fixture
        // regenerated against the other form would silently retune the
        // tolerance instead of failing. Refuse it by name.
        assert_eq!(
            meta["norm_convention"].as_str().unwrap(),
            "normed * (1.0 + weight)",
            "fixture was generated under a different norm convention"
        );
        Self {
            dir,
            hc: meta["hc_count"].as_u64().unwrap() as usize,
            h: meta["hidden_size"].as_u64().unwrap() as usize,
            rank: meta["hc_lowrank"].as_u64().unwrap() as usize,
            eps: meta["rms_norm_eps"].as_f64().unwrap() as f32,
            tokens: meta["num_tokens"].as_u64().unwrap() as usize,
        }
    }

    fn bytes(&self, name: &str) -> Vec<u8> {
        let p = format!("{}/{name}.bin", self.dir);
        std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
    }

    fn f32s(&self, name: &str) -> Vec<f32> {
        self.bytes(name)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = g.alloc(bytes.len()).unwrap();
    g.copy_h2d_async(bytes, p, g.default_stream()).unwrap();
    p
}

fn download_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut raw = vec![0u8; n * 2];
    g.copy_d2h(p, &mut raw).unwrap();
    raw.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn download_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut raw = vec![0u8; n * 4];
    g.copy_d2h(p, &mut raw).unwrap();
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Max absolute difference, cosine similarity, and the reference's own scale
/// — reported together because either alone hides a failure. A near-null
/// output has a tiny max-abs against a small reference; a correctly-shaped
/// but mis-scaled one has cosine 1.0.
fn compare(label: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let mut max_abs = 0.0f32;
    let mut dot = 0.0f64;
    let (mut ng, mut nw) = (0.0f64, 0.0f64);
    let mut worst = 0usize;
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
            worst = i;
        }
        dot += a as f64 * b as f64;
        ng += a as f64 * a as f64;
        nw += b as f64 * b as f64;
    }
    let cos = dot / (ng.sqrt() * nw.sqrt()).max(1e-30);
    let rms = (nw / want.len() as f64).sqrt();
    println!(
        "  {label:<22} max|diff|={max_abs:.4e} cos={cos:.9} \
         ref_rms={rms:.4e}  worst[{worst}] got={:.5} want={:.5}",
        got[worst], want[worst]
    );
    assert!(
        max_abs <= tol,
        "{label}: max|diff| {max_abs:.4e} exceeds {tol:.4e}"
    );
    assert!(cos > 0.9999, "{label}: cosine {cos:.9} — shape differs");
}

/// One site's low-rank weights, uploaded.
fn site_weights(g: &dyn GpuBackend, f: &Fixture, site: &str, inject: bool) -> HcLowRank {
    HcLowRank {
        norm_w: upload(g, &f.bytes(&format!("{site}_w_hc_norm"))),
        down_w: upload(g, &f.bytes(&format!("{site}_w_down"))),
        up_w: upload(g, &f.bytes(&format!("{site}_w_up"))),
        inject_w: if inject {
            upload(g, &f.bytes(&format!("{site}_w_inject")))
        } else {
            DevicePtr::NULL
        },
        rank: f.rank,
    }
}

/// BF16 carries ~8 mantissa bits, so a stored output is good to ~4e-3
/// relative. These outputs run to |x| ~ 1-13, and the kernel accumulates a
/// 10240-term dot product in FP32 in a different order than torch does.
/// Tolerances are set from the reference's own RMS rather than a constant,
/// and are tight enough that both defects this harness was built for — a
/// global RMS (max|diff| 15.2) and a dropped offset-from-1 (4.65) — fail by
/// three orders of magnitude.
fn tol_for(ref_vals: &[f32]) -> f32 {
    let rms = (ref_vals.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / ref_vals.len() as f64)
        .sqrt() as f32;
    (rms * 0.05).max(1e-3)
}

/// The small-T cuBLASLt arm shares the prefill GEMM formulation (BF16-staged
/// `normed`, tensor-core dots) but cuBLASLt's reduction order differs from
/// `dense_gemm_bf16_pipelined`'s, so its worst element sits at ~9% of the
/// reference RMS where the split/tile arms sit under 5% — same cos grade
/// (>=0.999996) the prefill GEMM shipped with. Held to 12% + the cosine gate.
fn tol_gemm(ref_vals: &[f32]) -> f32 {
    let rms = (ref_vals.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / ref_vals.len() as f64)
        .sqrt() as f32;
    (rms * 0.12).max(1e-3)
}

#[test]
#[ignore]
fn hc_lowrank_matches_reference() {
    let f = Fixture::load();
    // NOT `ptx_modules()`. In a wildcard (`ATLAS_TARGET_MODEL=*`) build that
    // is an alias for target 0, and `hyper_connection` in some other target's
    // set is DeepSeek-V4's Sinkhorn kernel — a different argument list behind
    // the same name, i.e. a segfault or, worse, plausible numbers. Ask for
    // this target by identity.
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4").expect(
        "qwen3.8-flash-next/nvfp4 is not in this build — \
         build with ATLAS_TARGET_MODEL='*' or =qwen3.8-flash-next",
    );
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let (t, h, hc) = (f.tokens, f.h, f.hc);
    println!(
        "hc={hc} hidden={h} rank={} tokens={t} eps={:e}",
        f.rank, f.eps
    );

    let k_pre = g.kernel("hyper_connection", "hc_pre").unwrap();
    let k_head = g.kernel("hyper_connection", "hc_head").unwrap();
    let k_post = g.kernel("hyper_connection", "hc_post").unwrap();
    for (name, k) in [("hc_pre", k_pre), ("hc_head", k_head), ("hc_post", k_post)] {
        assert!(
            k.0 != 0,
            "{name} resolved to handle 0 — the qwen3.8-flash-next shadow is \
             not the one loaded, so this would be testing DeepSeek's kernel"
        );
    }

    let streams = upload(g, &f.bytes("streams"));
    let y_out = g.alloc(t * h * 2).unwrap();
    let inj_out = g.alloc(t * hc * 4).unwrap();
    // Sized as the BufferArena sizes it (64-token ceiling): big enough for
    // BOTH small-T layouts (split f32 [64, hc*h + rank]; the cuBLASLt GEMM
    // layout is smaller at lay = 8).
    let scratch = g.alloc(64 * (hc * h + f.rank) * 4).unwrap();

    // ── hc_pre: the per-layer collapse, both sites, BOTH small-T arms ──
    // The public entry routes small T through the cuBLASLt GEMM arm
    // (`tol_gemm` — see its docs); the split arm is called directly and
    // held to the original tight bound.
    for site in ["attn", "mlp"] {
        let w = site_weights(g, &f, site, true);
        let want_mixed = f.f32s(&format!("{site}_mixed"));
        let want_inj = f.f32s(&format!("{site}_inj"));

        ops::hc_pre_lowrank(
            g, k_pre, streams, &w, y_out, inj_out, scratch, t as u32, h as u32, hc as u32, f.eps,
            stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        println!("{site}_hyper_connection (cublas arm):");
        compare(
            "mixed_input",
            &download_bf16(g, y_out, t * h),
            &want_mixed,
            tol_gemm(&want_mixed),
        );
        compare(
            "injection_weights",
            &download_f32(g, inj_out, t * hc),
            &want_inj,
            tol_gemm(&want_inj),
        );

        super::hyper_connection_lowrank::hc_pre_split(
            g, streams, &w, y_out, inj_out, scratch, t as u32, h as u32, hc as u32, f.eps, true,
            stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        println!("{site}_hyper_connection (split arm):");
        compare(
            "mixed_input",
            &download_bf16(g, y_out, t * h),
            &want_mixed,
            tol_for(&want_mixed),
        );
        compare(
            "injection_weights",
            &download_f32(g, inj_out, t * hc),
            &want_inj,
            tol_for(&want_inj),
        );
    }

    // ── hc_head: the model-level mixer, `use_combine=False`. This IS the
    //    model's final norm — the checkpoint ships no `model.norm.weight`.
    let w_head = site_weights(g, &f, "head", false);
    ops::hc_head_lowrank(
        g, k_head, streams, &w_head, y_out, scratch, t as u32, h as u32, hc as u32, f.eps, stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    let want_head = f.f32s("head_mixed");
    println!("hyper_connection_mixer (cublas arm):");
    compare(
        "mixed_input",
        &download_bf16(g, y_out, t * h),
        &want_head,
        tol_gemm(&want_head),
    );

    // Split arm of the head, tight bound.
    super::hyper_connection_lowrank::hc_pre_split(
        g,
        streams,
        &w_head,
        y_out,
        DevicePtr::NULL,
        scratch,
        t as u32,
        h as u32,
        hc as u32,
        f.eps,
        false,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    println!("hyper_connection_mixer (split arm):");
    compare(
        "mixed_input",
        &download_bf16(g, y_out, t * h),
        &want_head,
        tol_for(&want_head),
    );

    // ── hc_post: inject a block output back into every stream ──
    let block_out = upload(g, &f.bytes("post_block_out"));
    let inj = upload(
        g,
        &f.f32s("attn_inj")
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let post_out = g.alloc(t * hc * h * 4).unwrap();
    ops::hc_post_lowrank(
        g, k_post, block_out, streams, inj, post_out, t as u32, h as u32, hc as u32, stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    let want_post = f.f32s("post_expected");
    println!("hc_post:");
    compare(
        "residual",
        &download_f32(g, post_out, t * hc * h),
        &want_post,
        tol_for(&want_post),
    );
}

// provenance-id: 526f6e616c6420522e205374657369616b
/// The decode-rows arm (`ATLAS_HC_DECODE_ROWS=1`, T <= 8): `hc_pre_stage` +
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
        super::hyper_connection_lowrank::hc_decode_rows_shape_ok(t as u32, h as u32, hc as u32, f.rank as u32),
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
            super::hyper_connection_lowrank::hc_pre_rows(
                g, streams, &w, y_out, inj_out, scratch, rows as u32, h as u32, hc as u32, f.eps,
                true, stream,
            )
            .unwrap();
            g.synchronize(stream).unwrap();
            println!("{site}_hyper_connection (rows arm, T={rows}):");
            compare("mixed_input", &download_bf16(g, y_out, rows * h), &want_mixed, tol_for(&want_mixed));
            compare("injection_weights", &download_f32(g, inj_out, rows * hc), &want_inj, tol_for(&want_inj));
        }
        let w_head = site_weights(g, &f, "head", false);
        let want_head: Vec<f32> = f.f32s("head_mixed")[..rows * h].to_vec();
        super::hyper_connection_lowrank::hc_pre_rows(
            g, streams, &w_head, y_out, DevicePtr::NULL, scratch, rows as u32, h as u32, hc as u32,
            f.eps, false, stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        println!("hyper_connection_mixer (rows arm, T={rows}):");
        compare("mixed_input", &download_bf16(g, y_out, rows * h), &want_head, tol_for(&want_head));
    }
}

/// The GEMM-path collapse (T > 64) against the SAME reference goldens: hc_pre
/// is per-token independent (per-stream RMS, rank projection, gates — no
/// cross-token term), so tiling the T=8 fixture 12x to T=96 is an EXACT
/// large-T golden: every replica must reproduce the reference outputs. This
/// routes through `hc_pre_gemm` (stage_bf16 + dense_gemm x3 + mix), which
/// rounds `normed` to BF16 — covered by the same rms-relative tolerances.
#[test]
#[ignore]
fn hc_pre_gemm_matches_reference() {
    const TILE: usize = 12;
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
    let big_t = t * TILE;
    assert!(
        big_t > 64,
        "tiled fixture must exceed the split-path ceiling"
    );

    let k_pre = g.kernel("hyper_connection", "hc_pre").unwrap();
    for name in ["hc_pre_stage_bf16", "hc_silu_scale", "hc_pre_mix"] {
        let k = g.kernel("hyper_connection", name).unwrap();
        assert!(k.0 != 0, "{name} resolved to handle 0");
    }

    let stream_bytes = f.bytes("streams");
    let tiled: Vec<u8> = stream_bytes
        .iter()
        .copied()
        .cycle()
        .take(stream_bytes.len() * TILE)
        .collect();
    let streams = upload(g, &tiled);
    let y_out = g.alloc(big_t * h * 2).unwrap();
    let inj_out = g.alloc(big_t * hc * 4).unwrap();
    // Sized exactly as sizes.rs sizes it for m = big_t (< 2048): the GEMM
    // layout L = min(T, 2048) = big_t.
    let scratch = g.alloc(big_t * (2 * hc * h + f.rank + hc) * 2).unwrap();

    for site in ["attn", "mlp"] {
        let w = site_weights(g, &f, site, true);
        ops::hc_pre_lowrank(
            g,
            k_pre,
            streams,
            &w,
            y_out,
            inj_out,
            scratch,
            big_t as u32,
            h as u32,
            hc as u32,
            f.eps,
            stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();

        let want_mixed: Vec<f32> = {
            let one = f.f32s(&format!("{site}_mixed"));
            one.iter().copied().cycle().take(one.len() * TILE).collect()
        };
        let want_inj: Vec<f32> = {
            let one = f.f32s(&format!("{site}_inj"));
            one.iter().copied().cycle().take(one.len() * TILE).collect()
        };
        println!("{site}_hyper_connection (GEMM path, T={big_t}):");
        compare(
            "mixed_input",
            &download_bf16(g, y_out, big_t * h),
            &want_mixed,
            tol_for(&want_mixed),
        );
        compare(
            "injection_weights",
            &download_f32(g, inj_out, big_t * hc),
            &want_inj,
            tol_for(&want_inj),
        );
    }

    // The model-level mixer takes the same GEMM path with inject=false.
    let k_head = g.kernel("hyper_connection", "hc_head").unwrap();
    let w_head = site_weights(g, &f, "head", false);
    ops::hc_head_lowrank(
        g,
        k_head,
        streams,
        &w_head,
        y_out,
        scratch,
        big_t as u32,
        h as u32,
        hc as u32,
        f.eps,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    let want_head: Vec<f32> = {
        let one = f.f32s("head_mixed");
        one.iter().copied().cycle().take(one.len() * TILE).collect()
    };
    println!("hyper_connection_mixer (GEMM path, T={big_t}):");
    compare(
        "mixed_input",
        &download_bf16(g, y_out, big_t * h),
        &want_head,
        tol_for(&want_head),
    );
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
        super::hyper_connection_lowrank::hc_pre_rows(
            g, streams, &w, y_t3, inj_t3, scratch, rows as u32, h as u32, hc as u32, f.eps, true, stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        let mut y3 = vec![0u8; rows * h * 2];
        let mut i3 = vec![0u8; rows * hc * 4];
        g.copy_d2h(y_t3, &mut y3).unwrap();
        g.copy_d2h(inj_t3, &mut i3).unwrap();
        for r in 0..rows {
            super::hyper_connection_lowrank::hc_pre_rows(
                g, streams.offset(r * hc * h * 4), &w, y_t1, inj_t1, scratch, 1, h as u32, hc as u32,
                f.eps, true, stream,
            )
            .unwrap();
            g.synchronize(stream).unwrap();
            let mut y1 = vec![0u8; h * 2];
            let mut i1 = vec![0u8; hc * 4];
            g.copy_d2h(y_t1, &mut y1).unwrap();
            g.copy_d2h(inj_t1, &mut i1).unwrap();
            assert!(y3[r * h * 2..(r + 1) * h * 2] == y1[..], "{site}: mixed_input row {r} differs T=3 vs T=1");
            assert!(i3[r * hc * 4..(r + 1) * hc * 4] == i1[..], "{site}: injection row {r} differs T=3 vs T=1");
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
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4").expect(
        "qwen3.8-flash-next/nvfp4 is not in this build",
    );
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
            super::hyper_connection_lowrank::hc_pre_rows(
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
            super::hyper_connection_lowrank::hc_pre_rows(
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
