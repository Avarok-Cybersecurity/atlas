// SPDX-License-Identifier: AGPL-3.0-only
// Gate design after TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// docs/kernel-contract.md:1-10 (planted controls: "a run where a control passes is a failed run";
// five-run byte repeat; tile-edge shapes) and docs/recipes/qwen3.5-2b-native.md (one-row against
// multi-row execution). Apache-2.0; see TENSORFOLD-ATTRIBUTION.md. Changes: no code copied; the gates are
// written for Atlas's affine route.

//! GPU oracle and timings for the Ada affine 4-bit route (port lane L17a): the quantizer, pack, group
//! sums and dequant kernels (`kernels/ada/common/affine_pack_ada.cu`) and the lane GEMV
//! (`kernels/ada/common/affine4_lane_ada.cu`), through `port::affine` and its `W4Linear` face.
//!
//! WHAT THIS PINS (each gate prints PASS/FAIL, each planted control CONTROL-OK; any FAIL, or any control
//! that passes its gate, exits non-zero):
//!  * the quantizer's words, scales and biases equal to the host mirror (`port::affine::reference`) on
//!    random and edge groups (flat, one outlier, all negative, a range under the step floor, ties), at
//!    groups 32/64/128 with the clip search off and on; the non-finite flag names the first bad group;
//!    control: a round-half-to-even quantizer differs on the tie row;
//!  * the tiled pack and the scale transpose against the host, and the pack's round trip; control: a
//!    pack with two nibble slots swapped;
//!  * the dequantizer and the group sums byte for byte;
//!  * per decode shape N in {9216, 2560, 12288, 4096, 129} x K in {2560, 9216, 4096} at group 64: the
//!    device tiles equal to the host pack; the GEMV within the float64 bound (`reference_gemv.rs`) at
//!    M = 1, 2, 4, 5, 8, 16 (control: the last group dropped); row r at every M bit-equal to row r at
//!    M = 1 (control: the slice partials re-added in reverse order); the persistent body against the
//!    split form within the bound; a five-run byte repeat (control: an arrival-order slice sum);
//!    `decode_pair` (one `_seg` launch) and `_many` byte-equal to separate calls; the same decode gates
//!    repeat at groups 32 and 128 on the three `--quick` shapes.
//!
//! TIMINGS (`timing.rs`): DRAM-bound GB/s at M = 1 and 4 for the 4B decode shapes against `fp8_gemv_ada`,
//! with the L17a bar (gate+up M = 1 at or above 85% of FP8's GB/s) printed as BAR PASS / BAR MISS.
//!
//! Run on the Ada card, in a build of the ada target:
//!     AVAROK_TARGET_HW=ada CUDARC_CUDA_VERSION=13000 AVAROK_NO_RDMA=1 cargo run --release -p spark-model \
//!       --example affine_ada_microtest --features cuda,gpu-examples
//! `--gates-only` skips the timing rows; `--quick` runs the decode gates on three shapes only.

#[path = "affine_ada_microtest/gates.rs"]
mod gates;
#[path = "affine_ada_microtest/shape.rs"]
mod shape;
#[path = "affine_ada_microtest/timing.rs"]
mod timing;
#[path = "affine_ada_microtest/util.rs"]
mod util;

use anyhow::{Result, ensure};
use spark_model::port::affine::AffineKernels;
use spark_runtime::cuda_backend::AvarokCudaBackend;

/// The decode shapes of the plan: N x K.
const NS: [usize; 5] = [9216, 2560, 12288, 4096, 129];
const KS: [usize; 3] = [2560, 9216, 4096];
/// `--quick`: one split-heavy, one unsplit and one ragged shape.
const QUICK: [(usize, usize); 3] = [(2560, 9216), (12288, 2560), (129, 4096)];

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let gates_only = args.iter().any(|a| a == "--gates-only");
    let quick = args.iter().any(|a| a == "--quick");
    let gpu = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    ensure!(
        AffineKernels::available(&gpu),
        "the affine modules are absent: build with AVAROK_TARGET_HW=ada"
    );
    let k = AffineKernels::resolve(&gpu)?;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut r = util::Lcg(0x0a1f_4e89_2026_1112);
    let mut t = util::Tally::default();

    println!("load-side kernels:");
    gates::quantizer(&gpu, &k, &mut r, &mut t)?;
    gates::layout(&gpu, &k, &mut r, &mut t)?;

    println!("decode GEMV:");
    let shapes: Vec<(usize, usize)> = if quick {
        QUICK.to_vec()
    } else {
        NS.iter()
            .flat_map(|&n| KS.iter().map(move |&kk| (n, kk)))
            .collect()
    };
    for (n, kk) in shapes {
        shape::run(&gpu, &mut r, n, kk, shape::GROUP, threads, &mut t)?;
    }
    // The group-32 and group-128 instantiations of the same kernels: the same gates on the QUICK shapes
    // (a split-heavy, an unsplit and a ragged one).
    println!("decode GEMV, groups 32 and 128:");
    for group in [32u16, 128] {
        for (n, kk) in QUICK {
            shape::run(&gpu, &mut r, n, kk, group, threads, &mut t)?;
        }
    }

    let bar = if gates_only {
        None
    } else {
        let fk = timing::Fp8Kernels::resolve(&gpu)?;
        Some(timing::rows(&gpu, &fk, &mut r)?)
    };
    println!(
        "summary: {} gates and controls, {} failed; bar: {}",
        t.gates,
        t.fails,
        match bar {
            None => "not run",
            Some(true) => "PASS",
            Some(false) => "MISS",
        }
    );
    ensure!(t.fails == 0, "{} gate(s) or control(s) failed", t.fails);
    println!("all gates PASS");
    Ok(())
}
