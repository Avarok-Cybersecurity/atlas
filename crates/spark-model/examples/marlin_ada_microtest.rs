// SPDX-License-Identifier: AGPL-3.0-only
// Gate shapes after vLLM (https://github.com/vllm-project/vllm, commit 8a36f08),
// tests/kernels/quantization/test_marlin_gemm.py (the M sweep around the 16-row tiles and the
// GPTQ/AWQ round trips against a dequantized reference), Copyright contributors to the vLLM project,
// Apache-2.0. Changes: no code copied; the gates, the oracle bound and the row-invariance checks are
// written for Atlas's `W4Linear` contract.

//! GPU oracle and timings for the Marlin W4A16 route (port lanes L18a kernels, L18b host): the 30
//! entry points of `kernels/ada/common/marlin_ada.cu` through `port::marlin` and its `W4Linear` face.
//!
//! WHAT THIS PINS (each gate prints `PASS ...` or `FAIL ...`, each planted control `CONTROL-OK ...`;
//! any FAIL, or a control that passes its gate, exits non-zero):
//!  * every entry point once, on its full row tile, within 2 BF16 ulps plus the FP32 term-magnitude
//!    bound of a float64 CPU product of the dequantized weight (`port::marlin::reference`); the PASS
//!    line names the entry (a GPU measurement script that is not part of this repository counts
//!    the 30 names); the lock array back at zero;
//!  * decode (the row-invariant policy) at M = 1, 8, 16, 17, 32, 64 on the 4B FFN shapes, GPTQ and
//!    AWQ types, with a planted control (a reference missing its last group must fail the bound);
//!  * row invariance: row r at every M in 1..=64 byte-equal to row r at M = 64 and to the row decoded
//!    alone; vLLM's own selector at M = 1 and 17 is printed for contrast (`info`);
//!  * a five-run byte repeat; `prefill` at M = 65, 200, 513 (vLLM's M split) within the bound;
//!  * the opt-in RTN route from a device BF16 weight bit-equal to the CPU quantizer, decode within the
//!    bound; synthetic GPTQ (symmetric, v1 asymmetric) and AWQ tensor sets through `from_gptq` and
//!    `from_awq` bit-equal to the CPU dequant and within the bound.
//!
//! TIMINGS (`info timing ...`): decode GB/s at M = 1..64 on the 4B FFN and attention shapes.
//!
//! Run on the Ada card, in a build of the ada target:
//!     AVAROK_TARGET_HW=ada CUDARC_CUDA_VERSION=13000 AVAROK_NO_RDMA=1 cargo run --release -p spark-model \
//!       --example marlin_ada_microtest --features cuda,gpu-examples
//! `--gates-only` skips the timing rows; `--quick` runs the decode and prefill gates on one shape.

#[path = "marlin_ada_microtest/gates.rs"]
mod gates;
#[path = "marlin_ada_microtest/loads.rs"]
mod loads;
#[path = "marlin_ada_microtest/util.rs"]
mod util;

use anyhow::{Result, ensure};
use spark_model::port::marlin::{self, config};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::GpuBackend;

/// The 4B decode shapes (FFN gate/up, FFN down), N x K.
const SHAPES: [(usize, usize); 2] = [(9216, 2560), (2560, 9216)];

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let gates_only = args.iter().any(|a| a == "--gates-only");
    let quick = args.iter().any(|a| a == "--quick");
    let gpu = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    ensure!(
        marlin::available(&gpu),
        "the {} module is absent: build with AVAROK_TARGET_HW=ada",
        marlin::kernels::MODULE
    );
    let sms = gpu.sm_count()? as usize;
    println!(
        "info device sms={sms} max_shared_mem={}",
        config::ADA_MAX_SHARED_MEM_OPTIN
    );
    let s = util::scratch(&gpu, config::c_tmp_bytes(64, sms))?;
    let mut r = util::Lcg(0x4d61_726c_696e_2026);
    let mut t = util::Tally::default();

    println!("entry points:");
    gates::entries(&gpu, &mut r, &s, &mut t)?;

    println!("decode and prefill:");
    let shapes: &[(usize, usize)] = if quick { &SHAPES[..1] } else { &SHAPES };
    for &(n, k) in shapes {
        for zp in [false, true] {
            gates::decode(&gpu, &mut r, &s, &mut t, n, k, zp)?;
        }
        gates::prefill(&gpu, &mut r, &s, &mut t, n, k, false)?;
    }

    println!("load routes:");
    loads::rtn(&gpu, &mut r, &s, &mut t)?;
    loads::checkpoints(&gpu, &mut r, &s, &mut t)?;

    if !gates_only {
        println!("timing:");
        loads::timing(&gpu, &mut r, &s)?;
    }
    gpu.free(s.ws)?;
    println!(
        "summary: {} gates and controls, {} failed",
        t.gates, t.fails
    );
    ensure!(t.fails == 0, "{} gate(s) or control(s) failed", t.fails);
    println!("all gates PASS");
    Ok(())
}
