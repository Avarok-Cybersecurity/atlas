// SPDX-License-Identifier: AGPL-3.0-only

//! Phase C: precompute fused Q-absorption matrix W_QK_absorbed
//! [nq*kv_lora, q_lora] on the CPU, upload BF16.

use anyhow::Result;

use super::super::gpu_alloc_or_managed;
use super::ctx::MistralLayerCtx;
use crate::weight_map::DenseWeight;

/// BF16 bit pattern → FP32 (exact widen: BF16 is the high half of an FP32).
#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Widen a little-endian BF16 byte buffer to FP32, once.
///
/// The naive absorption loop re-decoded both operands *inside* the innermost
/// accumulation — `n_kv · kv_lora · q_lora · nope` times, 4.8e9 decodes per
/// LongCat sublayer, each a bounds-checked two-byte slice read. Widening both
/// inputs up front costs one linear pass and turns the inner loop into plain
/// FP32 arithmetic over contiguous slices.
fn widen_bf16(buf: &[u8]) -> Vec<f32> {
    buf.chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// How many worker threads the absorption GEMM may use.
///
/// `ATLAS_MLA_ABSORB_THREADS` overrides (1 = fully sequential, which is the
/// escape hatch if this is ever suspected of a numerics change — it is not,
/// see `absorb_rows`, but the knob costs nothing).
fn absorb_threads(rows: usize) -> usize {
    let want = std::env::var("ATLAS_MLA_ABSORB_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });
    want.clamp(1, rows.max(1))
}

/// Compute one contiguous span of W_QK rows.
///
/// `out` is `[rows_here, q_lora]`, covering absorbed rows
/// `row0 .. row0 + rows_here`, where row `r` is head `r / kv_lora`, latent
/// index `r % kv_lora`.
///
/// **Accumulation order is load-bearing.** The reference form is
///
/// ```text
/// out[r][l] = Σ_{p=0}^{nope-1}  wqb[(n*hd + p)*q_lora + l] * wuk[(n*kv_lora + lkv)*nope + p]
/// ```
///
/// summed in ascending `p`, each product rounded to FP32 before the add. This
/// version hoists `l` into the inner position so the `wqb` row is walked
/// contiguously, but every `out[..][l]` accumulator still visits exactly the
/// same products in exactly the same ascending-`p` order, starting from `0.0`.
/// Per-`l` accumulators are independent, so vectorising across `l` cannot
/// reassociate a sum. Rust performs no FP contraction by default, so the
/// `mul` + `add` stay separate in both forms. The result is therefore
/// **bit-identical** to the reference — asserted by
/// `absorb_rows_is_bit_exact_vs_reference` below.
#[allow(clippy::too_many_arguments)]
fn absorb_rows(
    out: &mut [f32],
    row0: usize,
    wqb: &[f32],
    wuk: &[f32],
    kv_lora: usize,
    q_lora: usize,
    nope: usize,
    hd: usize,
) {
    for (i, acc) in out.chunks_exact_mut(q_lora).enumerate() {
        let row = row0 + i;
        let n = row / kv_lora;
        let lkv = row % kv_lora;
        acc.fill(0.0);
        let wuk_row = &wuk[(n * kv_lora + lkv) * nope..][..nope];
        for (p, &w) in wuk_row.iter().enumerate() {
            let wqb_row = &wqb[(n * hd + p) * q_lora..][..q_lora];
            for (a, &q) in acc.iter_mut().zip(wqb_row.iter()) {
                *a += q * w;
            }
        }
    }
}

/// W_QK_absorbed as FP32 `[n_kv * kv_lora, q_lora]`, row-parallel.
///
/// Each worker owns a disjoint span of output rows, so the split changes
/// nothing about any individual dot product — see `absorb_rows`.
#[allow(clippy::too_many_arguments)]
fn absorb_qk(
    wqb: &[f32],
    wuk: &[f32],
    n_kv: usize,
    kv_lora: usize,
    q_lora: usize,
    nope: usize,
    hd: usize,
) -> Vec<f32> {
    let rows = n_kv * kv_lora;
    let mut out = vec![0.0f32; rows * q_lora];
    let threads = absorb_threads(rows);
    if threads <= 1 {
        absorb_rows(&mut out, 0, wqb, wuk, kv_lora, q_lora, nope, hd);
        return out;
    }
    let rows_per = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for (chunk_idx, chunk) in out.chunks_mut(rows_per * q_lora).enumerate() {
            let row0 = chunk_idx * rows_per;
            scope.spawn(move || {
                absorb_rows(chunk, row0, wqb, wuk, kv_lora, q_lora, nope, hd);
            });
        }
    });
    out
}

pub(crate) fn build_w_qk_absorbed(ctx: &mut MistralLayerCtx<'_>) -> Result<()> {
    // Load-time phase timer — see the note in `phase_per_head`. The GEMM here
    // is O(n_kv · kv_lora · q_lora · nope) FP32 multiply-adds (LongCat:
    // 32·512·1536·192 = 4.8e9 per sublayer), which used to be where the bulk
    // of a LongCat layer's build time went.
    let t_phase = std::time::Instant::now();
    let n_kv = ctx.n_kv;
    let n_heads = ctx.n_heads;
    let kv_lora = ctx.kv_lora;
    let q_lora = ctx.q_lora;
    let nope = ctx.nope;
    let hd = ctx.hd;
    let bf16 = ctx.bf16;
    let gpu = ctx.gpu;

    let wq_b = ctx.wq_b.as_ref().expect("phase A");
    let w_uk_t = ctx.w_uk_t.as_ref().expect("phase B");

    let wqk_size = n_kv * kv_lora * q_lora * bf16;
    let wqk_ptr = gpu_alloc_or_managed(gpu, wqk_size)?;
    {
        // Read wq_b[n_heads*hd, q_lora] from GPU.
        let wqb_bytes = n_heads * hd * q_lora * bf16;
        let mut wqb_buf = vec![0u8; wqb_bytes];
        let t = std::time::Instant::now();
        gpu.copy_d2h(wq_b.weight, &mut wqb_buf)?;
        // Read W_UK[n_heads, kv_lora, nope] from GPU (transposed layout).
        let wuk_bytes = n_kv * kv_lora * nope * bf16;
        let mut wuk_buf = vec![0u8; wuk_bytes];
        gpu.copy_d2h(w_uk_t.weight, &mut wuk_buf)?;
        let t_d2h = t.elapsed();

        // Compute W_QK[n, kv_lora, q_lora] on CPU in FP32.
        let t = std::time::Instant::now();
        let wqb_f32 = widen_bf16(&wqb_buf);
        let wuk_f32 = widen_bf16(&wuk_buf);
        let t_widen = t.elapsed();
        let t = std::time::Instant::now();
        let threads = absorb_threads(n_kv * kv_lora);
        let wqk_f32 = absorb_qk(&wqb_f32, &wuk_f32, n_kv, kv_lora, q_lora, nope, hd);
        let t_gemm = t.elapsed();

        // Truncate FP32 → BF16 into one preallocated buffer. The `flat_map`
        // + `to_le_bytes().to_vec()` this replaces heap-allocated a 2-byte
        // `Vec` PER ELEMENT — 25.2M mallocs per LongCat sublayer, which cost
        // ~177 ms against ~14 ms of actual upload. Same bytes: still the high
        // half of the FP32, little-endian.
        let t = std::time::Instant::now();
        let mut wqk_bf16 = vec![0u8; wqk_f32.len() * 2];
        for (dst, &v) in wqk_bf16.chunks_exact_mut(2).zip(wqk_f32.iter()) {
            dst.copy_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
        }
        let t_convert = t.elapsed();
        let t = std::time::Instant::now();
        gpu.copy_h2d(&wqk_bf16, wqk_ptr)?;
        let t_h2d = t.elapsed();
        tracing::info!(
            "MLA phase C (W_QK_absorbed) L{}: total={:.1}ms | d2h ({:.1} MB)={:.1}ms, \
             bf16-widen={:.1}ms, cpu-gemm ({:.2}e9 fp32 MACs, {threads} threads)={:.1}ms, \
             bf16-convert={:.1}ms, h2d ({:.1} MB)={:.1}ms",
            ctx.layer_idx,
            t_phase.elapsed().as_secs_f64() * 1e3,
            (wqb_bytes + wuk_bytes) as f64 / 1e6,
            t_d2h.as_secs_f64() * 1e3,
            t_widen.as_secs_f64() * 1e3,
            (n_kv * kv_lora * q_lora * nope) as f64 / 1e9,
            t_gemm.as_secs_f64() * 1e3,
            t_convert.as_secs_f64() * 1e3,
            wqk_size as f64 / 1e6,
            t_h2d.as_secs_f64() * 1e3,
        );
        if ctx.layer_idx == 0 {
            tracing::info!(
                "W_QK_absorbed: [{}, {}] ({:.1} MB per layer)",
                n_kv * kv_lora,
                q_lora,
                wqk_size as f64 / 1e6
            );
        }
    }
    ctx.w_qk_absorbed = Some(DenseWeight { weight: wqk_ptr });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pre-optimization loop, verbatim, as the bit-exactness oracle.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        wqb_buf: &[u8],
        wuk_buf: &[u8],
        n_kv: usize,
        kv_lora: usize,
        q_lora: usize,
        nope: usize,
        hd: usize,
    ) -> Vec<f32> {
        let mut wqk_f32 = vec![0.0f32; n_kv * kv_lora * q_lora];
        let to_f32 = |buf: &[u8], idx: usize| -> f32 {
            let bits = u16::from_le_bytes([buf[idx * 2], buf[idx * 2 + 1]]);
            f32::from_bits((bits as u32) << 16)
        };
        for n in 0..n_kv {
            for lkv in 0..kv_lora {
                for l in 0..q_lora {
                    let mut sum = 0.0f32;
                    for p in 0..nope {
                        let wqb_val = to_f32(wqb_buf, (n * hd + p) * q_lora + l);
                        let wuk_val = to_f32(wuk_buf, n * kv_lora * nope + lkv * nope + p);
                        sum += wqb_val * wuk_val;
                    }
                    wqk_f32[(n * kv_lora + lkv) * q_lora + l] = sum;
                }
            }
        }
        wqk_f32
    }

    /// xorshift64* — deterministic, no dev-dependency.
    fn rng(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        *state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// BF16 bit patterns spread over a realistic weight magnitude range,
    /// including zeros and both signs. Exponents are kept inside the normal
    /// range so the test is about accumulation order, not denormal handling.
    fn bf16_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        let mut out = Vec::with_capacity(n * 2);
        for _ in 0..n {
            let r = rng(&mut state);
            let sign = ((r >> 40) & 1) as u16;
            // exponent 0x68..0x87 → magnitudes ~1e-7 .. ~2e2
            let exp = 0x68u16 + ((r >> 8) & 0x1f) as u16;
            let man = (r & 0x7f) as u16;
            let bits = if r & 0xff == 0 {
                0 // sprinkle exact zeros
            } else {
                (sign << 15) | (exp << 7) | man
            };
            out.extend_from_slice(&bits.to_le_bytes());
        }
        out
    }

    /// The parallel, `l`-inner absorption must be BIT-IDENTICAL to the
    /// original scalar triple loop — this weight feeds the absorbed MLA
    /// attention path, so a one-ulp drift is a silent numerics change.
    #[test]
    fn absorb_rows_is_bit_exact_vs_reference() {
        // LongCat-shaped but small enough for a unit test: the shape that
        // matters is `nope` (the accumulation length) and `q_lora` (the
        // vectorised axis), both kept non-round to catch tail handling.
        for &(n_kv, kv_lora, q_lora, nope, hd) in &[
            (2usize, 5usize, 13usize, 7usize, 11usize),
            (3, 8, 33, 17, 24),
            (4, 16, 64, 32, 48),
            (1, 3, 1, 192, 256),
        ] {
            let wqb = bf16_bytes(n_kv * hd * q_lora, 0x1234_5678 ^ (nope as u64));
            let wuk = bf16_bytes(n_kv * kv_lora * nope, 0x9abc_def0 ^ (hd as u64));
            let want = reference(&wqb, &wuk, n_kv, kv_lora, q_lora, nope, hd);

            let wqb_f32 = widen_bf16(&wqb);
            let wuk_f32 = widen_bf16(&wuk);

            // Every thread count must give the same bytes, including the
            // sequential path and a split that does not divide the row count.
            for threads in [1usize, 2, 3, 7, 64] {
                let rows = n_kv * kv_lora;
                let mut got = vec![0.0f32; rows * q_lora];
                let t = threads.clamp(1, rows);
                let rows_per = rows.div_ceil(t);
                std::thread::scope(|scope| {
                    for (ci, chunk) in got.chunks_mut(rows_per * q_lora).enumerate() {
                        let row0 = ci * rows_per;
                        let (a, b) = (&wqb_f32, &wuk_f32);
                        scope.spawn(move || {
                            absorb_rows(chunk, row0, a, b, kv_lora, q_lora, nope, hd);
                        });
                    }
                });
                for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "row-parallel absorption diverged at element {i} \
                         (shape n_kv={n_kv} kv_lora={kv_lora} q_lora={q_lora} \
                         nope={nope} hd={hd}, threads={threads}): {g:?} vs {w:?}",
                    );
                }
            }
        }
    }

    /// `absorb_qk` (the function the loader actually calls, thread count taken
    /// from the machine) must match the reference too.
    #[test]
    fn absorb_qk_matches_reference() {
        let (n_kv, kv_lora, q_lora, nope, hd) = (4usize, 16usize, 40usize, 24usize, 32usize);
        let wqb = bf16_bytes(n_kv * hd * q_lora, 0xfeed_face);
        let wuk = bf16_bytes(n_kv * kv_lora * nope, 0x0bad_c0de);
        let want = reference(&wqb, &wuk, n_kv, kv_lora, q_lora, nope, hd);
        let got = absorb_qk(
            &widen_bf16(&wqb),
            &widen_bf16(&wuk),
            n_kv,
            kv_lora,
            q_lora,
            nope,
            hd,
        );
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "absorb_qk diverged at {i}");
        }
    }

    /// Wall-clock A/B of the absorption GEMM at LongCat-Flash-Lite's real
    /// per-sublayer shape, CPU-only (no GPU, no model, no checkpoint). This is
    /// the phase that dominated LongCat load time; run it to re-measure after
    /// touching `absorb_rows`:
    ///
    /// ```text
    /// cargo test --release -p spark-model --lib -- --ignored --nocapture absorb_qk_bench
    /// ```
    #[test]
    #[ignore = "wall-clock benchmark: seconds per run, not a correctness gate"]
    fn absorb_qk_bench() {
        // LongCat-Flash-Lite: 32 heads, kv_lora 512, q_lora 1536, padded
        // nope 192 (padded head 256 − rope 64), padded hd 256.
        let (n_kv, kv_lora, q_lora, nope, hd) = (32usize, 512usize, 1536usize, 192usize, 256usize);
        let wqb = bf16_bytes(n_kv * hd * q_lora, 0x5eed_1234);
        let wuk = bf16_bytes(n_kv * kv_lora * nope, 0x5eed_5678);
        let macs = (n_kv * kv_lora * q_lora * nope) as f64;

        let t = std::time::Instant::now();
        let want = reference(&wqb, &wuk, n_kv, kv_lora, q_lora, nope, hd);
        let t_ref = t.elapsed();

        let t = std::time::Instant::now();
        let wqb_f32 = widen_bf16(&wqb);
        let wuk_f32 = widen_bf16(&wuk);
        let t_widen = t.elapsed();
        let t = std::time::Instant::now();
        let got = absorb_qk(&wqb_f32, &wuk_f32, n_kv, kv_lora, q_lora, nope, hd);
        let t_new = t.elapsed();

        assert_eq!(got.len(), want.len());
        let diffs = got
            .iter()
            .zip(want.iter())
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count();
        println!(
            "absorb_qk @ LongCat shape ({:.2}e9 MACs, {} threads):\n  \
             reference (scalar, bf16-decode-in-loop): {t_ref:?}\n  \
             widen: {t_widen:?}   new (row-parallel, l-inner): {t_new:?}\n  \
             speedup gemm-only {:.1}x, including widen {:.1}x\n  \
             bit-mismatched elements: {diffs}",
            macs / 1e9,
            absorb_threads(n_kv * kv_lora),
            t_ref.as_secs_f64() / t_new.as_secs_f64(),
            t_ref.as_secs_f64() / (t_new + t_widen).as_secs_f64(),
        );
        // The FP32→BF16 pack was, after the GEMM fix, the biggest remaining
        // item in this phase. A/B it here too.
        let t = std::time::Instant::now();
        let old_pack: Vec<u8> = got
            .iter()
            .flat_map(|&v| {
                let bits = (v.to_bits() >> 16) as u16;
                bits.to_le_bytes().to_vec()
            })
            .collect();
        let t_pack_old = t.elapsed();
        let t = std::time::Instant::now();
        let mut new_pack = vec![0u8; got.len() * 2];
        for (dst, &v) in new_pack.chunks_exact_mut(2).zip(got.iter()) {
            dst.copy_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
        }
        let t_pack_new = t.elapsed();
        println!(
            "  bf16 pack ({} elems): flat_map+to_vec {t_pack_old:?} -> preallocated \
             {t_pack_new:?} ({:.1}x), bytes equal: {}",
            got.len(),
            t_pack_old.as_secs_f64() / t_pack_new.as_secs_f64(),
            old_pack == new_pack,
        );
        assert_eq!(old_pack, new_pack, "bf16 pack must stay byte-identical");

        assert_eq!(diffs, 0, "benchmark output must stay bit-exact");
    }

    /// The FP32→BF16 truncation must produce the exact bytes the old
    /// `flat_map(|v| ...to_le_bytes().to_vec())` produced — the rewrite was
    /// purely about removing 25.2M per-element heap allocations, not about
    /// changing a single output byte.
    #[test]
    fn bf16_truncate_matches_old_flat_map() {
        // Include the awkward cases the truncation has to round-trip:
        // signed zeros, denormals, infinities, NaN, and the exact halfway
        // patterns where a *rounding* implementation would differ from this
        // truncating one.
        let mut vals: Vec<f32> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(0x3f80_8000),
            f32::from_bits(0x3f80_7fff),
            f32::from_bits(0x0000_0001),
        ];
        let mut state = 0xdead_beefu64;
        for _ in 0..4096 {
            vals.push(f32::from_bits(rng(&mut state) as u32));
        }

        let want: Vec<u8> = vals
            .iter()
            .flat_map(|&v| {
                let bits = (v.to_bits() >> 16) as u16;
                bits.to_le_bytes().to_vec()
            })
            .collect();

        let mut got = vec![0u8; vals.len() * 2];
        for (dst, &v) in got.chunks_exact_mut(2).zip(vals.iter()) {
            dst.copy_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
        }

        assert_eq!(
            got, want,
            "bf16 truncation bytes diverged from the old form"
        );
    }

    /// The BF16 widen is the exact high-half reinterpretation the loader's
    /// old inline closure did.
    #[test]
    fn widen_bf16_is_exact_high_half() {
        let bytes = bf16_bytes(1024, 0xa5a5_a5a5);
        let widened = widen_bf16(&bytes);
        for (i, w) in widened.iter().enumerate() {
            let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
            assert_eq!(w.to_bits(), (bits as u32) << 16);
        }
    }
}
