// SPDX-License-Identifier: AGPL-3.0-only

//! GPU parity for the ON-DEVICE decode selection: `qsa_topk_rows` +
//! `qsa_expand_sel` must reproduce the host top-k/sort/expand byte for byte.
//!
//! Needs no fixtures — the scores are synthesised, including the tie-heavy
//! shapes that matter (relu floors every non-attending block to exactly 0.0).
//! The exhaustive sweep lives in
//! `examples/qsa_decode_select_parity.rs`; this is the leg that runs under
//! `cargo test -p spark-model --release qsa_decode -- --ignored --nocapture`.

use super::super::QsaIndexer;
use spark_runtime::gpu::GpuBackend;

fn synth(kind: &str, complete: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32
    };
    match kind {
        "distinct" => (0..complete).map(|_| 0.05 + 40.0 * next()).collect(),
        // The production shape: most blocks relu'd flat to 0.0.
        "relu_floor" => (0..complete)
            .map(|_| if next() < 0.6 { 0.0 } else { 30.0 * next() })
            .collect(),
        // Everything tied — the selection is decided entirely by the
        // lower-index tie-break, for every candidate.
        "all_tied" => vec![7.5; complete],
        _ => unreachable!(),
    }
}

#[test]
#[ignore]
fn qsa_decode_select_device_matches_host() {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL='*'");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();

    let (hd, hidden, ratio, budget) = (128usize, 2560usize, 4usize, 2048usize);
    let topk = budget / ratio;
    let a = |n: usize| g.alloc(n).unwrap();
    let qsa = QsaIndexer::new(
        a((4 + 1) * hd * hidden * 2),
        a(hd * 2),
        a(hd * 2),
        4,
        hd,
        ratio,
        budget,
        64,
        1.0e7,
        1.0e-6,
        hidden,
        2,
        256,
        g,
    )
    .unwrap();

    let mut cases = 0usize;
    for kind in ["distinct", "relu_floor", "all_tied"] {
        // One block past the early-out, and a real serving shape (8192
        // blocks = a 32K visible prefix).
        for (i, &complete) in [topk + 1, 8192usize].iter().enumerate() {
            let sc = synth(kind, complete, 0x51A7_C0DE + i as u64);
            for tail in 0..ratio {
                let visible = complete * ratio + tail;
                let (dev, host, dev_len, host_len) = qsa
                    .decode_select_parity_probe(&sc, visible, 16, g, stream)
                    .unwrap();
                assert_eq!(
                    dev_len,
                    (budget + tail) as u32,
                    "{kind} complete={complete} tail={tail}: seq_len"
                );
                assert_eq!(host_len, dev_len, "host seq_len");
                assert_eq!(
                    dev, host,
                    "{kind} complete={complete} tail={tail}: sel[] device != host"
                );
                assert!(
                    dev.windows(2).all(|w| w[0] < w[1]),
                    "{kind} complete={complete} tail={tail}: sel[] not ascending"
                );
                cases += 1;
            }
        }
    }
    println!("qsa decode selection parity OK: {cases} cases");
}
