// SPDX-License-Identifier: AGPL-3.0-only
//! Parity leg for the ON-DEVICE QSA decode selection.
//!
//! `decode_select` used to D2H every block score, sort on the CPU, expand the
//! chosen blocks into token indices and H2D the result — once per QSA layer
//! per sequence per decode step. The device path is `qsa_topk_rows` (radix
//! select) followed by `qsa_expand_sel` (shared-memory ascending sort +
//! expansion + seq_len + identity table). This driver runs BOTH over the same
//! device-resident scores and asserts the produced `sel[]` and `n_sel` are
//! byte-identical.
//!
//! Ties matter and are the common case: scores are `sum_h relu(q.k)`, so a
//! block whose every head dot-product is negative scores exactly 0.0. Half
//! the cases here are built to be tie-dominated.
//!
//! ```text
//! cargo run --release -p spark-model --example qsa_decode_select_parity
//! ```
use anyhow::Result;
use spark_model::layers::qsa::QsaIndexer;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

struct Rng(u64);
impl Rng {
    fn n(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn u(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.n() >> 40) as f32 / (1u64 << 24) as f32)
    }
}

/// Score generators. `relu` floors at 0.0, so exact ties are ordinary.
fn scores(kind: &str, complete: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed);
    match kind {
        // Broad continuous range: ties essentially absent.
        "distinct" => (0..complete).map(|_| r.u(0.05, 40.0)).collect(),
        // Realistic: most blocks fully relu'd to zero, a live minority.
        "relu_floor" => (0..complete)
            .map(|_| {
                if r.u(0.0, 1.0) < 0.6 {
                    0.0
                } else {
                    r.u(0.0, 30.0)
                }
            })
            .collect(),
        // Every score identical — the top-k is decided purely by the
        // lower-index tie-break, for all `complete` candidates.
        "all_tied" => vec![7.5; complete],
        // A plateau that straddles the cutoff: the k-th value has far more
        // holders than remaining slots, so the tie walk must pick the lowest
        // indices and nothing else.
        "cutoff_plateau" => (0..complete)
            .map(|i| if i % 3 == 0 { 12.0 } else { r.u(0.0, 12.0) })
            .collect(),
        // Coarse quantisation: many exact repeats away from the floor too.
        "quantised" => (0..complete).map(|_| (r.u(0.0, 8.0)).round()).collect(),
        _ => unreachable!(),
    }
}

fn main() -> Result<()> {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL='*' (or =qwen3.8-flash-next)");
    let be = AtlasCudaBackend::new(0, &set.modules)?;
    let gpu: &dyn GpuBackend = &be;
    let stream = gpu.default_stream();

    // Indexer geometry: production values (n_heads 4, hd 128, ratio 4,
    // budget 2048 -> block_topk 512, the shared-memory sort's maximum), and a
    // second one at budget 2044 -> block_topk 511 so the bitonic padding to a
    // power of two is exercised rather than skipped.
    let hd = 128usize;
    let hidden = 2560usize;
    let dummy = |n: usize| gpu.alloc(n).unwrap();
    let build = |budget: usize| -> Result<QsaIndexer> {
        QsaIndexer::new(
            dummy(5 * hd * hidden * 2),
            dummy(hd * 2),
            dummy(hd * 2),
            /* n_heads */ 4,
            hd,
            /* ratio */ 4,
            budget,
            /* rot */ 64,
            /* theta */ 1.0e7,
            /* eps */ 1.0e-6,
            hidden,
            /* nkv_attn */ 2,
            /* hd_attn */ 256,
            gpu,
        )
    };

    let kinds = [
        "distinct",
        "relu_floor",
        "all_tied",
        "cutoff_plateau",
        "quantised",
    ];
    let block_size = 16u32;
    let mut cases = 0usize;
    let mut seed = 0x51A7_C0DEu64;

    for budget in [2048usize, 2044] {
        let qsa = build(budget)?;
        let topk = budget / 4;
        // `complete` sweeps from one block past the early-out (the tightest
        // case the top-k sees) up to a real serving shape: 8192 blocks is a
        // 32K visible prefix, the default ATLAS_QSA_MAX_TOKENS.
        for &complete in &[topk + 1, topk + 7, 1024usize, 4096, 8192] {
            if complete <= topk {
                continue;
            }
            for kind in kinds {
                let sc = scores(kind, complete, seed);
                seed = seed.wrapping_add(0x0123_4567_89AB_CDEF);
                // Every tail length 0..ratio: the expansion appends
                // complete*ratio..visible after the selected blocks.
                for tail in 0..4usize {
                    let visible = complete * 4 + tail;
                    let (dev, host, dev_len, host_len) =
                        qsa.decode_select_parity_probe(&sc, visible, block_size, gpu, stream)?;
                    let want_n = (budget + tail) as u32;
                    assert_eq!(
                        dev_len, want_n,
                        "{kind} budget={budget} complete={complete} tail={tail}: \
                         device seq_len {dev_len} != {want_n}"
                    );
                    assert_eq!(host_len, want_n, "host seq_len");
                    assert_eq!(dev.len(), want_n as usize, "{kind}: device sel length");
                    if dev != host {
                        let bad = dev.iter().zip(&host).position(|(a, b)| a != b).unwrap_or(0);
                        panic!(
                            "{kind} budget={budget} complete={complete} tail={tail}: \
                             sel[] differs at {bad} — device {} vs host {}",
                            dev[bad], host[bad]
                        );
                    }
                    // The host path sorts the blocks before expanding; assert
                    // the property directly so a future rewrite that keeps
                    // both paths wrong in the same way still fails here.
                    assert!(
                        dev.windows(2).all(|w| w[0] < w[1]),
                        "{kind} budget={budget} complete={complete} tail={tail}: \
                         sel[] not strictly ascending"
                    );
                    cases += 1;
                }
            }
        }
    }
    println!("qsa decode selection parity OK: {cases} cases, device == host");
    Ok(())
}
