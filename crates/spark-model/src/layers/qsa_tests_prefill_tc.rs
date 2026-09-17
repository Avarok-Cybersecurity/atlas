// SPDX-License-Identifier: AGPL-3.0-only

//! The tensor-core prefill-attention cases, split out of
//! `qsa_tests_prefill.rs` to keep it under the 500-line cap.

use super::*;

/// Stage 2B, tensor-core twin: `qsa_prefill_attn_tc` vs the same CPU
/// reference, at the geometry it is gated to (hd 256, nq <= 16, nkv == 1 —
/// a TP=2 rank of qwen4_exp). `topk * ratio` is deliberately larger than the
/// kernel's 64-token tile so the online-softmax carry across tiles, the
/// rescale of the O accumulator and the partial last tile are all exercised.
///
/// The bar is the scalar kernel's own: cos > 0.999. P goes through bf16 to
/// reach the PV mma (the scalar path keeps it f32), so bit-equality is not
/// expected — if this ever fails, split P into hi/lo the way
/// `qsa_score_rows_tc` splits q rather than reverting the kernel.
#[test]
#[ignore]
fn qsa_prefill_attn_tc_matches_cpu() {
    let set = avarok_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with AVAROK_TARGET_MODEL='*'");
    let gpu =
        spark_runtime::cuda_backend::AvarokCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();

    // nkv = 1 and nq = 12: the TP=2 rank shape, and the reason one CTA per
    // row is correct (every head reads the same KV row).
    let (rows, nq, nkv, hd, ratio, topk, bs) =
        (5usize, 12usize, 1usize, 256usize, 4usize, 24usize, 16usize);
    assert!(ops::qsa_prefill_attn_tc_ok(
        nq as u32, nkv as u32, hd as u32
    ));
    // complete = (pos+1)/ratio must be >= topk so every list entry is a real
    // block, as it is in production (this kernel only runs past the inert bound).
    let first_pos = 101usize; // complete = 25 > topk = 24; tail = 2 at row 0
    let n_pos = first_pos + rows;
    let pages = n_pos.div_ceil(bs);

    let mut seed = 0x2468au32;
    let mut nextf = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((seed >> 8) as f32 / (1 << 24) as f32) - 0.5
    };
    let bf = |v: f32| -> u16 { (v.to_bits() >> 16) as u16 };
    let unbf = |u: u16| -> f32 { f32::from_bits((u as u32) << 16) };

    let q_host: Vec<u16> = (0..rows * nq * hd).map(|_| bf(nextf())).collect();
    let kv_elems = pages * bs * nkv * hd;
    let k_host: Vec<u16> = (0..kv_elems).map(|_| bf(nextf())).collect();
    let v_host: Vec<u16> = (0..kv_elems).map(|_| bf(nextf())).collect();
    let lists_host: Vec<i32> = (0..rows)
        .flat_map(|r| (0..topk as i32).map(move |i| (i * 3 + r as i32) % 25))
        .collect();

    let as_bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let q_dev = upload(g, &as_bytes(&q_host));
    let k_dev = upload(g, &as_bytes(&k_host));
    let v_dev = upload(g, &as_bytes(&v_host));
    let lists_dev = upload(
        g,
        &lists_host
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    let ident: Vec<u8> = (0..pages as i32).flat_map(|v| v.to_le_bytes()).collect();
    let table = upload(g, &ident);
    let out_dev = g.alloc(rows * nq * hd * 2).unwrap();
    let scale = 1.0 / (hd as f32).sqrt();

    // BOTH tiles, against the same CPU reference. Until 2026-09-15 this test
    // only ever loaded `qsa_prefill_attn_tc` (TB 64) — the VERIFY tile — so the
    // TB-16 tile that actually serves prefill had NO correctness test at all.
    // That gap hid a real one: `nc = warp*8 + gid` reaches 63 regardless of TB,
    // so at TB 16 the QK mma indexes n-columns past the tile and the store
    // drops them. When K moved to a row-contiguous layout that stray index
    // became a row index and ran off the end of smem (CUDA 700). The TB-64 arm
    // could not see it, because at TB 64 nc never exceeds the tile.
    for (kname, tb16) in [
        ("qsa_prefill_attn_tc", false),
        ("qsa_prefill_attn_tc_tb16", true),
    ] {
        let k = g.kernel("qsa_indexer", kname).unwrap();
        ops::qsa_prefill_attn_tc(
            g,
            k,
            q_dev,
            k_dev,
            v_dev,
            table,
            lists_dev,
            out_dev,
            rows as u32,
            first_pos as u32,
            topk as u32,
            ratio as u32,
            bs as u32,
            nq as u32,
            nkv as u32,
            hd as u32,
            scale,
            tb16,
            stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        let got = dl_bf16(g, out_dev, rows * nq * hd);

        let group = nq / nkv;
        let mut worst_cos = 1.0f64;
        for r in 0..rows {
            let pos = first_pos + r;
            let complete = (pos + 1) / ratio;
            let tail = (pos + 1) - complete * ratio;
            let mut toks: Vec<usize> = lists_host[r * topk..(r + 1) * topk]
                .iter()
                .flat_map(|&b| (0..ratio).map(move |i| b as usize * ratio + i))
                .collect();
            toks.extend(complete * ratio..complete * ratio + tail);
            for h in 0..nq {
                let kvh = h / group;
                let qv: Vec<f32> = (0..hd)
                    .map(|d| unbf(q_host[(r * nq + h) * hd + d]))
                    .collect();
                let scores: Vec<f32> = toks
                    .iter()
                    .map(|&t| {
                        let base = (t * nkv + kvh) * hd;
                        (0..hd).map(|d| qv[d] * unbf(k_host[base + d])).sum::<f32>() * scale
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f32::MIN, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                let l: f32 = exps.iter().sum();
                let mut refv = vec![0.0f32; hd];
                for (i, &t) in toks.iter().enumerate() {
                    let base = (t * nkv + kvh) * hd;
                    let w = exps[i] / l;
                    for d in 0..hd {
                        refv[d] += w * unbf(v_host[base + d]);
                    }
                }
                let gv = &got[(r * nq + h) * hd..(r * nq + h + 1) * hd];
                let dot: f64 = gv
                    .iter()
                    .zip(&refv)
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum();
                let ng: f64 = gv.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
                let nr: f64 = refv.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
                let cos = dot / (ng * nr).max(1e-30);
                worst_cos = worst_cos.min(cos);
            }
        }
        println!("{kname} vs CPU: worst cos = {worst_cos:.9}");
        assert!(
            worst_cos > 0.999,
            "{kname}: TC attention kernel diverges: {worst_cos}"
        );
    }
}

/// Minimal repro for the dense chunk-0 flash zeroing rows past ~1280 at
/// qwen4_exp geometry (nq=24, nkv=2, hd=256, causal, seq 2809). Synthetic
/// q/k/v, CPU reference at probe rows. If this passes, the corruption is in
/// the K/V staging upstream of the kernel, not the kernel.
#[test]
#[ignore]
fn flash64_long_seq_rows_repro() {
    let set = avarok_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with AVAROK_TARGET_MODEL='*'");
    let gpu =
        spark_runtime::cuda_backend::AvarokCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let k = g
        .kernel("inferspark_prefill", "inferspark_prefill_64")
        .unwrap();

    let (n, nq, nkv, hd) = (2809usize, 24usize, 2usize, 256usize);
    let mut seed = 0xBEEFu32;
    let mut nextf = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((seed >> 8) as f32 / (1 << 24) as f32) - 0.5
    };
    let bf = |v: f32| -> u16 { (v.to_bits() >> 16) as u16 };
    let unbf = |u: u16| -> f32 { f32::from_bits((u as u32) << 16) };
    let q_host: Vec<u16> = (0..n * nq * hd).map(|_| bf(nextf())).collect();
    let k_host: Vec<u16> = (0..n * nkv * hd).map(|_| bf(nextf())).collect();
    let v_host: Vec<u16> = (0..n * nkv * hd).map(|_| bf(nextf())).collect();
    let as_bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let q_dev = upload(g, &as_bytes(&q_host));
    let k_dev = upload(g, &as_bytes(&k_host));
    let v_dev = upload(g, &as_bytes(&v_host));
    let out_dev = g.alloc(n * nq * hd * 2).unwrap();
    // Poison the output so unwritten rows are detectable.
    let poison = vec![0x3Fu8; n * nq * hd * 2];
    g.copy_h2d_async(&poison, out_dev, stream).unwrap();
    let scale = 1.0 / (hd as f32).sqrt();

    ops::prefill_attention_64(
        g, k, q_dev, k_dev, v_dev, out_dev, n as u32, 1, nq as u32, nkv as u32, hd as u32, scale,
        true, 0, stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    let got = dl_bf16(g, out_dev, n * nq * hd);

    let group = nq / nkv;
    for &row in &[100usize, 1024, 1200, 1279, 1280, 1290, 1500, 2051, 2808] {
        // CPU reference for head 0 only (cheap).
        let h = 0usize;
        let kvh = h / group;
        let qv: Vec<f32> = (0..hd)
            .map(|d| unbf(q_host[(row * nq + h) * hd + d]))
            .collect();
        let mut m = f32::MIN;
        let scores: Vec<f32> = (0..=row)
            .map(|t| {
                let base = (t * nkv + kvh) * hd;
                let s: f32 = (0..hd).map(|d| qv[d] * unbf(k_host[base + d])).sum::<f32>() * scale;
                m = m.max(s);
                s
            })
            .collect();
        let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let l: f32 = exps.iter().sum();
        let mut refv = vec![0.0f32; hd];
        for (t, e) in exps.iter().enumerate() {
            let base = (t * nkv + kvh) * hd;
            let w = e / l;
            for d in 0..hd {
                refv[d] += w * unbf(v_host[base + d]);
            }
        }
        let gv = &got[(row * nq + h) * hd..(row * nq + h) * hd + hd];
        let dot: f64 = gv
            .iter()
            .zip(&refv)
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let ng: f64 = gv.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let nr: f64 = refv.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (ng * nr).max(1e-30);
        println!("  flash64 row {row:>4}: cos={cos:.6} |got|={ng:.4} |ref|={nr:.4}");
    }
}
