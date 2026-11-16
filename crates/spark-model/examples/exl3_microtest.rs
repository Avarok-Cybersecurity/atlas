// SPDX-License-Identifier: AGPL-3.0-only
// Gate design after TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// docs/kernel-contract.md:1-10 (planted controls: "a run where a control passes is a failed run";
// five-run byte repeat) and zig/src/core/exl3_projection_gpu_test.zig (GPU decode against the scalar
// format oracle). Apache-2.0; see EXL3-ATTRIBUTION.md. Changes: no code copied; the gates are written for
// Atlas's EXL3 route.

//! GPU oracle and timings for the Ada EXL3 trellis route (port lane L20b, row `weights_exl3`): the input
//! rotation and the BF16 dequant (`kernels/ada/common/exl3_hadamard_ada.cu`) and the decode-row linear
//! (`kernels/ada/common/exl3_linear_ada.cu`), through `port::exl3` and its `W4Linear` face.
//!
//! WHAT THIS PINS (each gate prints PASS/FAIL, each planted control CONTROL-OK; any FAIL, or any control
//! that passes its gate, exits non-zero):
//!  * dequant: for every served width and codebook (30), the GPU BF16 dequant BYTE FOR BYTE equal to the
//!    host mirror (`port::exl3::reference_math::dequant_bf16_mirror`) on random tiles, plus one large
//!    matrix; control: one tile's words flipped on the device only;
//!  * rotation: the rotated fp16 input left in the workspace byte for byte equal to `rot_in_mirror`;
//!    control: the mirror with the input scales reversed;
//!  * per shape (K x N of the 4B and 27B FFNs and a split-heavy one) and width (2, 3, 3.5, 4 bits on the
//!    three codebooks): the linear within the FP64 bound (`gemv_row_ref`) at M = 1, 2, 4, 5, 8, 16
//!    (decode) and 17, 48, 128 (prefill); control: the reference without the last k tile; row r at M = 16
//!    and M = 128 bit-equal to row r computed alone (control: row 1 against row 0); a five-run byte
//!    repeat; the prefill declines at M = 129;
//!  * real tensors: TensorFold's public-checkpoint fixture block (`ATLAS_EXL3_ORACLE=<exl3_oracle.bin>`)
//!    and, with `--real-dir <shards> --prefix <tensor prefix>`, one group of a real EXL3 checkpoint:
//!    GPU dequant byte-equal to the mirror, and for the real group the linear gates above.
//!
//! TIMINGS: GB/s of trellis bytes at M = 1 and 4 on the 27B FFN shapes at 3.0, 3.5 and 4.0 bpw (INFO).
//!
//! Run on the Ada card, in a build of the ada target:
//!     AVAROK_TARGET_HW=ada CUDARC_CUDA_VERSION=13000 AVAROK_NO_RDMA=1 cargo run --release -p spark-model \
//!       --example exl3_microtest --features cuda,gpu-examples -- [--gates-only] [--quick] \
//!       [--real-dir DIR --prefix PREFIX]

#[path = "exl3_microtest/real.rs"]
mod real;
#[path = "exl3_microtest/util.rs"]
mod util;

use anyhow::{Context, Result, ensure};
use spark_model::port::exl3::format::{Exl3Codebook, Exl3Parts, Exl3Scales, all_widths};
use spark_model::port::exl3::reference::unpack;
use spark_model::port::exl3::reference_math::{dequant_bf16_mirror, gemv_row_ref, rot_in_mirror};
use spark_model::port::exl3::{Exl3Linear, exl3_of, kernels, release};
use spark_model::port::w4::{W4Box, W4Scratch};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use util::{
    Lcg, Matrix, Tally, download_u16, refs_par, scratch, time_us, unequal, unpack_par, upload,
};

const DECODE_MS: [usize; 6] = [1, 2, 4, 5, 8, 16];
const PREFILL_MS: [usize; 3] = [17, 48, 128];
/// `(k, n)`: the 4B FFN, the 27B FFN, and a narrow output that splits K eight ways.
const SHAPES: [(usize, usize); 5] = [
    (2560, 9216),
    (9216, 2560),
    (5120, 17408),
    (17408, 5120),
    (4096, 1024),
];
const SHAPES_QUICK: [(usize, usize); 2] = [(9216, 2560), (4096, 1024)];
const WIDTHS: [(u16, Exl3Codebook); 6] = [
    (6, Exl3Codebook::Mul1),
    (7, Exl3Codebook::Mul1),
    (8, Exl3Codebook::Mul1),
    (8, Exl3Codebook::Mcg),
    (8, Exl3Codebook::Inst3),
    (4, Exl3Codebook::Inst3),
];
const WIDTHS_QUICK: [(u16, Exl3Codebook); 2] = [(7, Exl3Codebook::Mul1), (8, Exl3Codebook::Mcg)];
const ROWS: usize = 128;

type Bytes3 = (Vec<u8>, Vec<u8>, Vec<u8>);

fn parts<'a>(m: &Matrix, b: &'a Bytes3) -> Exl3Parts<'a> {
    Exl3Parts {
        k: m.k,
        n: m.n,
        halves: m.halves,
        codebook: m.cb,
        trellis: &b.0,
        su: Exl3Scales::F16(&b.1),
        sv: Exl3Scales::F16(&b.2),
        bias: None,
    }
}

fn dequant_gates(gpu: &dyn GpuBackend, r: &mut Lcg, t: &mut Tally) -> Result<()> {
    let mut bad_widths = 0;
    let mut detail = String::new();
    for (halves, cb) in all_widths() {
        let m = Matrix::random(r, 256, 256, halves, cb);
        let b = m.bytes();
        let w = Exl3Linear::upload(gpu, &parts(&m, &b), 0)?;
        let out = gpu.alloc(256 * 256 * 2)?;
        w.dequant_bf16(gpu, out, 0)?;
        let got = download_u16(gpu, out, 256 * 256)?;
        let want = dequant_bf16_mirror(
            &unpack(halves, cb, &m.trellis, 256, 256),
            256,
            256,
            &m.suh,
            &m.svh,
        );
        let bad = unequal(&got, &want);
        if bad > 0 {
            bad_widths += 1;
            detail.push_str(&format!(" {halves}/{cb:?}:{bad}"));
        }
        gpu.free(out)?;
        release(gpu, &*w)?;
    }
    t.gate(
        "dequant == host mirror, every width x codebook (30)",
        bad_widths,
        &detail,
    );

    // One large matrix, and the planted control on it.
    let m = Matrix::random(r, 2560, 1024, 7, Exl3Codebook::Mul1);
    let b = m.bytes();
    let w = Exl3Linear::upload(gpu, &parts(&m, &b), 0)?;
    let out = gpu.alloc(1024 * 2560 * 2)?;
    w.dequant_bf16(gpu, out, 0)?;
    let want = dequant_bf16_mirror(&unpack_par(&m), 2560, 1024, &m.suh, &m.svh);
    t.gate(
        "dequant == host mirror, [2560 -> 1024] 3.5 bpw",
        unequal(&download_u16(gpu, out, 1024 * 2560)?, &want),
        "",
    );
    let dev = exl3_of(&*w).context("an EXL3 weight")?;
    let flipped: Vec<u8> = b.0[..56 * 2].iter().map(|x| !x).collect();
    gpu.copy_h2d(&flipped, dev.trellis)?;
    w.dequant_bf16(gpu, out, 0)?;
    t.control(
        "dequant with tile 0 flipped on the device",
        unequal(&download_u16(gpu, out, 1024 * 2560)?, &want),
        "",
    );
    gpu.free(out)?;
    release(gpu, &*w)?;
    Ok(())
}

/// Run `rows` rows (decode or prefill) and return the bf16 output bits.
fn run(
    gpu: &dyn GpuBackend,
    w: &W4Box,
    x: DevicePtr,
    rows: usize,
    out: DevicePtr,
    s: &W4Scratch,
) -> Result<Vec<u16>> {
    if rows <= w.max_decode_rows() {
        w.decode(gpu, x, rows, out, s, 0)?;
    } else {
        ensure!(
            w.prefill(gpu, x, rows, out, s, 0)?,
            "prefill of {rows} rows declined"
        );
    }
    download_u16(gpu, out, rows * w.n())
}

fn linear_gates(
    gpu: &dyn GpuBackend,
    m: &Matrix,
    xb: &[u16],
    label: &str,
    t: &mut Tally,
) -> Result<()> {
    let (k, n) = (m.k, m.n);
    let b = m.bytes();
    let w = Exl3Linear::upload(gpu, &parts(m, &b), 0)?;
    let s = scratch(gpu, w.scratch_bytes(ROWS))?;
    let x = upload(gpu, &spark_model::port::exl3::format::u16_bytes(xb))?;
    let out = gpu.alloc(ROWS * n * 2)?;
    let wq = unpack_par(m);
    let xh = rot_in_mirror(xb, ROWS, k, &m.suh);
    let refs = refs_par(&xh, ROWS, &wq, k, n, &m.svh);
    let plan = exl3_of(&*w).context("an EXL3 weight")?.plan;
    println!(
        " {label}: [{k} -> {n}] {} half-bits {:?}, plan sk={} wk={}",
        m.halves, m.cb, plan.sk, plan.wk
    );

    let mut outs = Vec::new();
    for &rows in DECODE_MS.iter().chain(&PREFILL_MS) {
        let got = run(gpu, &w, x, rows, out, &s)?;
        let bad: usize = (0..rows)
            .map(|r| refs[r].violations(&got[r * n..(r + 1) * n]))
            .sum();
        t.gate(&format!("{label} M={rows} within the FP64 bound"), bad, "");
        if rows == 16 {
            let got_xh = download_u16(gpu, s.ws, 16 * k)?;
            t.gate(
                &format!("{label} rotation == mirror"),
                unequal(&got_xh, &xh[..16 * k]),
                "16 rows",
            );
            let mut rev = m.suh.clone();
            rev.reverse();
            t.control(
                &format!("{label} rotation against reversed scales"),
                unequal(&got_xh, &rot_in_mirror(&xb[..16 * k], 16, k, &rev)),
                "",
            );
        }
        outs.push((rows, got));
    }
    let short = {
        let mut z = xh[..k].to_vec();
        z[k - 16..].fill(0);
        gemv_row_ref(&z, &wq, k, n, &m.svh)
    };
    t.control(
        &format!("{label} M=1 against the reference without the last k tile"),
        short.violations(&outs[0].1),
        "",
    );

    // Row invariance: row r alone (x offset to row r) against rows of the M = 16 and M = 128 calls.
    let wide: Vec<&(usize, Vec<u16>)> = outs
        .iter()
        .filter(|(rows, _)| *rows == 16 || *rows == 128)
        .collect();
    let mut bad = 0;
    let mut alone0 = Vec::new();
    for r in [0usize, 1, 7, 15, 17, 127] {
        let one = run(gpu, &w, x.offset(r * k * 2), 1, out, &s)?;
        if r == 0 {
            alone0 = one.clone();
        }
        for (rows, got) in &wide {
            if r < *rows {
                bad += unequal(&one, &got[r * n..(r + 1) * n]);
            }
        }
    }
    t.gate(
        &format!("{label} row r alone == row r at M=16 and M=128"),
        bad,
        "rows 0 1 7 15 17 127",
    );
    let m16 = &wide[0].1;
    t.control(
        &format!("{label} row 1 at M=16 against row 0 alone"),
        unequal(&m16[n..2 * n], &alone0),
        "",
    );

    let mut rep = 0;
    for _ in 0..5 {
        rep += unequal(&run(gpu, &w, x, 16, out, &s)?, m16);
    }
    t.gate(&format!("{label} five-run repeat at M=16"), rep, "");
    t.gate(
        &format!("{label} prefill declines at M=129"),
        usize::from(w.prefill(gpu, x, 129, out, &s, 0)?),
        "",
    );

    gpu.free(x)?;
    gpu.free(out)?;
    gpu.free(s.ws)?;
    release(gpu, &*w)?;
    Ok(())
}

fn timing(gpu: &dyn GpuBackend, r: &mut Lcg, t: &mut Tally) -> Result<()> {
    for (k, n) in [(5120, 17408), (17408, 5120), (2560, 9216)] {
        for halves in [6u16, 7, 8] {
            let m = Matrix::random(r, k, n, halves, Exl3Codebook::Mul1);
            let b = m.bytes();
            let w = Exl3Linear::upload(gpu, &parts(&m, &b), 0)?;
            let s = scratch(gpu, w.scratch_bytes(4))?;
            let x = upload(
                gpu,
                &spark_model::port::exl3::format::u16_bytes(&r.bf16s(4 * k)),
            )?;
            let out = gpu.alloc(4 * n * 2)?;
            for rows in [1usize, 4] {
                let us = time_us(gpu, || w.decode(gpu, x, rows, out, &s, 0))?;
                let gbs = b.0.len() as f64 / us / 1e3;
                t.info(
                    &format!(
                        "decode [{k} -> {n}] {:.1} bpw M={rows}",
                        f64::from(halves) / 2.0
                    ),
                    &format!(
                        "{us:.1} us, {gbs:.1} GB/s of codes ({:.0}% of 288)",
                        gbs / 2.88
                    ),
                );
            }
            for p in [x, out, s.ws] {
                gpu.free(p)?;
            }
            release(gpu, &*w)?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |f: &str| args.iter().any(|a| a == f);
    let value = |f: &str| {
        args.iter()
            .position(|a| a == f)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let (gates_only, quick) = (flag("--gates-only"), flag("--quick"));
    let gpu = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    ensure!(
        kernels::available(&gpu),
        "the EXL3 modules are absent: build with AVAROK_TARGET_HW=ada"
    );
    let mut r = Lcg(0x0e8c_3ada_2026_1112);
    let mut t = Tally::default();

    println!("dequant:");
    dequant_gates(&gpu, &mut r, &mut t)?;

    println!("decode-row linear:");
    let shapes: &[(usize, usize)] = if quick { &SHAPES_QUICK } else { &SHAPES };
    let widths: &[(u16, Exl3Codebook)] = if quick { &WIDTHS_QUICK } else { &WIDTHS };
    for &(k, n) in shapes {
        for &(halves, cb) in widths {
            let m = Matrix::random(&mut r, k, n, halves, cb);
            let xb = r.bf16s(ROWS * k);
            linear_gates(&gpu, &m, &xb, &format!("{k}x{n}/{halves}/{cb:?}"), &mut t)?;
        }
    }

    println!("real tensors:");
    real::fixture_gate(&gpu, &mut t)?;
    if let (Some(dir), Some(prefix)) = (value("--real-dir"), value("--prefix")) {
        let g = real::load_group(std::path::Path::new(&dir), &prefix)?;
        real::dequant_gate(&gpu, &g, &prefix, &mut t)?;
        let p = g.parts();
        let m = Matrix {
            k: g.k,
            n: g.n,
            halves: g.halves,
            cb: g.cb,
            trellis: p.trellis_u16(),
            suh: p.suh_bits(),
            svh: p.svh_bits(),
        };
        let xb = r.bf16s(ROWS * g.k);
        linear_gates(&gpu, &m, &xb, &prefix, &mut t)?;
    } else {
        t.info("real group", "no --real-dir/--prefix, skipped");
    }

    if !gates_only {
        println!("timing:");
        timing(&gpu, &mut r, &mut t)?;
    }
    println!(
        "summary: {} gates and controls, {} failed",
        t.gates, t.fails
    );
    ensure!(t.fails == 0, "{} gate(s) or control(s) failed", t.fails);
    println!("all gates PASS");
    Ok(())
}
