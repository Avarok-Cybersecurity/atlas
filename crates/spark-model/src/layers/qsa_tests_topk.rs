// SPDX-License-Identifier: AGPL-3.0-only

//! Bit-determinism + host parity for the device PREFILL top-k
//! (`qsa_topk_rows`), whose output list feeds `qsa_prefill_attn` DIRECTLY —
//! no `qsa_expand_sel` re-sort in between, unlike decode. That consumer's
//! online softmax accumulates in list order, so the ORDER of `out[]`, not
//! just the set, must be a pure function of the scores: the pre-fix kernel
//! emitted in atomicAdd race order and was the second temp-0 prefill
//! nondeterminism source (the first was the MoE prefill epilogue).
//!
//! Needs no fixtures — scores are synthesised, including the tie-heavy
//! shapes (relu floors most blocks to exactly 0.0, so ties dominate).
//! Run with
//! `cargo test -p spark-model --release qsa_prefill_topk -- --ignored --nocapture`.

use crate::layers::ops;
use spark_runtime::gpu::GpuBackend;

fn synth(kind: &str, n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32
    };
    match kind {
        "distinct" => (0..n).map(|_| 0.05 + 40.0 * next()).collect(),
        // The production shape: most blocks relu'd flat to 0.0.
        "relu_floor" => (0..n)
            .map(|_| if next() < 0.6 { 0.0 } else { 30.0 * next() })
            .collect(),
        // Everything tied — the set AND the order are decided entirely by
        // the lower-index tie-break.
        "all_tied" => vec![7.5; n],
        _ => unreachable!(),
    }
}

/// Runs the select 8 times over identical device-resident scores and asserts
/// (a) every run's `out[]` is BIT-identical to the first, (b) the list is
/// strictly ascending (the canonical order the fix guarantees), and (c) the
/// set matches the host reference `prefill_select` semantics — top-k by
/// score, ties broken by lower index. (a) and (b) fail on the pre-fix
/// kernel; (c) pins the fix as a reorder, not a re-select.
#[test]
#[ignore]
fn qsa_prefill_topk_bit_deterministic() {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL='*'");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let k = g.kernel("qsa_indexer", "qsa_topk_rows").unwrap();

    // Production shape: ratio 4, block_topk 512 (budget 2048). first_pos puts
    // every row's visible prefix well past the inert bound (complete > topk),
    // and per-row `complete` varies so the boundary column is exercised.
    let (ratio, topk, rows) = (4usize, 512usize, 64usize);
    let first_pos = 8191usize; // complete = 2048..=2063
    let stride = (first_pos + rows + 1).div_ceil(ratio);

    let lists = g.alloc(rows * topk * 4).unwrap();
    let scores = g.alloc(rows * stride * 4).unwrap();
    for kind in ["relu_floor", "all_tied", "distinct"] {
        let mut host = vec![0f32; rows * stride];
        for r in 0..rows {
            host[r * stride..(r + 1) * stride].copy_from_slice(&synth(
                kind,
                stride,
                0xD0_0D + r as u64,
            ));
        }
        let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
        g.copy_h2d_async(&bytes, scores, stream).unwrap();

        let mut runs: Vec<Vec<i32>> = Vec::new();
        for _ in 0..8 {
            // Poison between runs so a partial write shows up as a diff.
            g.copy_h2d_async(&vec![0xEEu8; rows * topk * 4], lists, stream)
                .unwrap();
            ops::qsa_topk_rows(
                g,
                k,
                scores,
                lists,
                rows as u32,
                stride as u32,
                topk as u32,
                first_pos as u32,
                ratio as u32,
                stream,
            )
            .unwrap();
            g.synchronize(stream).unwrap();
            let mut raw = vec![0u8; rows * topk * 4];
            g.copy_d2h(lists, &mut raw).unwrap();
            runs.push(
                raw.chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            );
        }
        for (i, run) in runs.iter().enumerate().skip(1) {
            assert_eq!(
                run, &runs[0],
                "{kind}: run {i} differs from run 0 — the emit order is racing"
            );
        }
        for r in 0..rows {
            let complete = (first_pos + r + 1) / ratio;
            let got = &runs[0][r * topk..(r + 1) * topk];
            assert!(
                got.windows(2).all(|w| w[0] < w[1]),
                "{kind} row {r}: out[] not strictly ascending"
            );
            // Host reference: sort by (-score, index), take k — the same
            // tie-break `prefill_select`'s host path uses.
            let sc = &host[r * stride..r * stride + complete];
            let mut order: Vec<u32> = (0..complete as u32).collect();
            order.sort_by(|&a, &b| {
                sc[b as usize]
                    .partial_cmp(&sc[a as usize])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            let mut want: Vec<i32> = order[..topk].iter().map(|&b| b as i32).collect();
            want.sort_unstable();
            assert_eq!(got, &want[..], "{kind} row {r}: selected set != host");
        }
    }
    println!("qsa prefill top-k: 8/8 bit-identical, ascending, host-set parity OK");
}
