// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the affine microtest's decode gates on one `N x K` shape at group 64, through the
//! real `W4Linear` path: tiles equal to the host pack, the GEMV within the float64 bound at M = 1, 2, 4,
//! 5, 8, 16, row r at every M bit-equal to row r at M = 1, the persistent body against the split form,
//! a five-run byte repeat, and `decode_pair` / `_many` byte-equal to separate calls. Every gate has a
//! planted control that must fail it.

use anyhow::Result;
use spark_model::port::affine::reference::{
    GemvRef, Mlx, bf16_of, count_outside, gemv_f64, pack_tiles, quantize_mlx, transpose_pad16,
};
use spark_model::port::affine::{AffineWeight, affine_of, kernels, quantize_from_bf16};
use spark_model::port::w4::{W4Box, W4Opts, W4Scratch};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::util::{
    Lcg, Out, Tally, bits, download_f32, download_u16, download_u32, u16_bytes, unequal, upload,
};

pub const ROWS: [usize; 6] = [1, 2, 4, 5, 8, 16];
/// The group the plan's shape list runs at; groups 32 and 128 repeat the gates on a few shapes.
pub const GROUP: u16 = 64;

fn opts(group: u16) -> W4Opts {
    W4Opts {
        bits: 4,
        group,
        clip_search: 0,
        int8_prefill: false,
    }
}

/// A weight quantized on the device from fresh values, plus its host mirror.
pub fn weight(
    gpu: &dyn GpuBackend,
    r: &mut Lcg,
    n: usize,
    k: usize,
    group: u16,
) -> Result<(W4Box, Mlx)> {
    let v = r.bf16s(n * k, 0.02);
    let src = upload(gpu, &u16_bytes(&bits(&v)))?;
    let b = quantize_from_bf16(gpu, src, n, k, &opts(group), 0)?;
    gpu.free(src)?;
    let host = quantize_mlx(&v, n, k, group as usize, 0).expect("finite weights");
    Ok((b, host))
}

/// The slice partials a split launch left in `work`, re-added in `order` (a permutation of the slices)
/// and rounded to bf16: what the split kernel would store if it added the slices in that order.
fn recombine(part: &[f32], w: &AffineWeight, rows: usize, order: &[usize]) -> Vec<u16> {
    let (n, sk) = (w.n, w.sk());
    let mut out = vec![0u16; rows * n];
    for r in 0..rows {
        for c in 0..n {
            let (tile, col) = (c / 64, c % 64);
            let at = |s: usize| part[((tile * sk + s) * 16 + r) * 64 + col];
            let mut t = at(order[0]);
            for &s in &order[1..] {
                t += at(s);
            }
            out[r * n + c] = bf16_of(t);
        }
    }
    out
}

pub fn run(
    gpu: &dyn GpuBackend,
    r: &mut Lcg,
    n: usize,
    k: usize,
    group: u16,
    threads: usize,
    t: &mut Tally,
) -> Result<()> {
    println!("  shape N={n} K={k} (group {group})");
    let g = group as usize;
    let (b, host) = weight(gpu, r, n, k, group)?;
    let w = affine_of(&*b).expect("an affine box");
    let (sk, kg, npad) = (w.sk(), w.kg(), w.npad);

    // Tiles: the quantizer and the pack together, against the host.
    let tiles = download_u32(gpu, w.words, npad * k / 8)?;
    let (st, bt) = (
        download_u16(gpu, w.scales, kg * npad)?,
        download_u16(gpu, w.biases, kg * npad)?,
    );
    let bad = unequal(&tiles, &pack_tiles(&host.words, n, k, g))
        + unequal(&st, &transpose_pad16(&host.scales, n, kg, npad))
        + unequal(&bt, &transpose_pad16(&host.biases, n, kg, npad));
    t.gate(
        "tiles",
        bad,
        "(codes, scales, biases vs host quantize + pack)",
    );

    let x = bits(&r.bf16s(16 * k, 1.0));
    let xd = upload(gpu, &u16_bytes(&x))?;
    let refr: GemvRef = gemv_f64(&x, 16, &host, threads);
    let s = W4Scratch {
        ws: gpu.alloc(b.scratch_bytes(16))?,
        ws_bytes: b.scratch_bytes(16),
        tickets: DevicePtr::NULL,
        xq: DevicePtr::NULL,
        xs: DevicePtr::NULL,
    };
    let sums = w.sums_bytes(16);
    let out16 = Out::new(gpu, 16 * n)?;

    // M = 16 first, so its slice partials are still in the workspace for the controls.
    b.decode(gpu, xd, 16, out16.ptr(), &s, 0)?;
    let full = out16.read(gpu)?;
    let part = if sk > 1 {
        download_f32(gpu, s.ws.offset(sums), w.tiles() * sk * 16 * 64)?
    } else {
        Vec::new()
    };
    let (bad, worst) = count_outside(&full, &refr, k, g, sk);
    t.gate(
        "gemv M=16 within bound",
        bad,
        &format!("(sk {sk}, worst {worst:.3} of the bound)"),
    );

    // PLANTED CONTROL: the last group of every input row zeroed must leave the bound.
    let mut planted = x.clone();
    for row in 0..16 {
        planted[row * k + k - g..(row + 1) * k].fill(0);
    }
    let pd = upload(gpu, &u16_bytes(&planted))?;
    let po = Out::new(gpu, 16 * n)?;
    b.decode(gpu, pd, 16, po.ptr(), &s, 0)?;
    let (pbad, _) = count_outside(&po.read(gpu)?, &refr, k, g, sk);
    t.control("dropped last group", pbad, "(vs the full reference)");

    // Row invariance: row r alone (M = 1) and row r inside every M.
    let mut single = Vec::with_capacity(16 * n);
    let o1 = Out::new(gpu, n)?;
    for row in 0..16 {
        o1.reset(gpu)?;
        b.decode(gpu, xd.offset(row * k * 2), 1, o1.ptr(), &s, 0)?;
        single.extend(o1.read(gpu)?);
    }
    let (mut inv, mut bound_bad) = (0, 0);
    for m in ROWS {
        let om = Out::new(gpu, m * n)?;
        b.decode(gpu, xd, m, om.ptr(), &s, 0)?;
        let got = om.read(gpu)?;
        inv += unequal(&got, &single[..m * n]);
        bound_bad += count_outside(&got, &refr, k, g, sk).0;
        om.free(gpu)?;
    }
    t.gate("gemv M=1,2,4,5,8,16 within bound", bound_bad, "");
    t.gate(
        "row r at M=k == row r at M=1",
        inv,
        &format!("(M in {ROWS:?})"),
    );
    if sk > 1 {
        // PLANTED CONTROL: the same partials added in reverse slice order (bits that would depend on
        // how a launch splits its work) must differ from the kernel's.
        let rev: Vec<usize> = (0..sk).rev().collect();
        t.control(
            "reverse slice order",
            unequal(&recombine(&part, &w, 16, &rev), &full),
            "",
        );
        let fwd: Vec<usize> = (0..sk).collect();
        t.gate(
            "partials in slice order == stored",
            unequal(&recombine(&part, &w, 16, &fwd), &full),
            "",
        );
    }

    // The persistent body (every slice in one CTA) against the split form.
    if sk > 1 {
        let xs = gpu.alloc(16 * kg * 4)?;
        kernels::group_sums(gpu, &w.kernels, xd, 16, k, group, xs, 0)?;
        let ou = Out::new(gpu, 16 * n)?;
        kernels::gemv_unsplit(gpu, xd, xs, &w, ou.ptr(), 16, 0)?;
        let got = ou.read(gpu)?;
        let (ub, _) = count_outside(&got, &refr, k, g, sk);
        t.gate(
            "unsplit within bound",
            ub,
            &format!("(unequal to split: {})", unequal(&got, &full)),
        );
        ou.free(gpu)?;
        gpu.free(xs)?;
    }

    // Five runs, byte for byte.
    let mut repeat = 0;
    for _ in 0..5 {
        out16.reset(gpu)?;
        b.decode(gpu, xd, 16, out16.ptr(), &s, 0)?;
        repeat += unequal(&out16.read(gpu)?, &full);
    }
    t.gate("five-run byte repeat", repeat, "");
    if sk > 1 {
        // PLANTED CONTROL: an atomics-style sum, whose slice order is whatever order CTAs finish in.
        let mut runs = Vec::new();
        for _ in 0..5 {
            let mut order: Vec<usize> = (0..sk).collect();
            for i in (1..sk).rev() {
                order.swap(i, r.below(i + 1));
            }
            runs.push(recombine(&part, &w, 16, &order));
        }
        let differ = runs
            .iter()
            .map(|v| unequal(v, &full))
            .filter(|d| *d > 0)
            .count();
        t.control(
            "arrival-order split sum",
            differ,
            "(runs differing from the kernel)",
        );
    }

    // decode_pair (one _seg launch) and _many against separate calls, on a second projection of the
    // same input.
    let (b2, _) = weight(gpu, r, n, k, group)?;
    let w2 = affine_of(&*b2).expect("an affine box");
    let rows = 4;
    let o: Vec<Out> = (0..6)
        .map(|_| Out::new(gpu, rows * n))
        .collect::<Result<_>>()?;
    b.decode(gpu, xd, rows, o[0].ptr(), &s, 0)?;
    b2.decode(gpu, xd, rows, o[1].ptr(), &s, 0)?;
    b.decode_pair(&*b2, gpu, xd, rows, o[2].ptr(), o[3].ptr(), &s, 0)?;
    let xs = gpu.alloc(rows * kg * 4)?;
    kernels::group_sums(gpu, &w.kernels, xd, rows, k, group, xs, 0)?;
    kernels::gemv_many(gpu, xd, xs, &[(&w, o[4].ptr()), (&w2, o[5].ptr())], rows, 0)?;
    let v: Vec<Vec<u16>> = o.iter().map(|x| x.read(gpu)).collect::<Result<_>>()?;
    let bad = unequal(&v[2], &v[0])
        + unequal(&v[3], &v[1])
        + unequal(&v[4], &v[0])
        + unequal(&v[5], &v[1]);
    t.gate("decode_pair and _many == separate decodes", bad, "(M=4)");
    t.control(
        "pair member vs the other projection",
        unequal(&v[2], &v[1]),
        "",
    );

    for x in o.into_iter().chain([out16, po, o1]) {
        x.free(gpu)?;
    }
    for p in [xd, pd, xs, s.ws] {
        gpu.free(p)?;
    }
    for bx in [&b, &b2] {
        spark_model::port::affine::release(gpu, &**bx)?;
    }
    Ok(())
}
