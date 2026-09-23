// SPDX-License-Identifier: AGPL-3.0-only
//! The ROUTING leg of the cross-row grouped MoE decode, on Qwen3.6-35B-A3B
//! shapes (hidden 2048, 256 experts, top-8): the per-token router
//! (`dense_gemv_bf16` + `moe_topk_softmax`, what `MoeLayer::forward` runs for
//! one row — the MTP drafter's `forward_one`) versus the batched router
//! (`dense_gemm_bf16` + `moe_topk_softmax_batched`, what
//! `forward_fp8_grouped_decode` runs for n rows — the batched propose).
//!
//! The expert-dispatch oracle (`fp8_moe_grouped_decode_microtest`) feeds both
//! legs ONE routing and proves the dispatch bit-identical; this is the leg it
//! leaves out. The two routers differ in FP32 summation order, so their BF16
//! logits can differ by rounding and a razor-margin top-k choice can flip —
//! the only way a batched draft can differ from a per-seq one. The bar here
//! is therefore exact about WHEN a difference is admissible:
//!
//! * the two logit rows agree to within `LOGIT_TOL` everywhere;
//! * where the selected expert SETS differ, the per-token row's k-th/(k+1)-th
//!   logit margin is no wider than twice the observed logit discrepancy on
//!   that row — a flip must be explained by rounding, never by a wrong index;
//! * where the sets agree, every expert's softmax weight agrees to `W_TOL`.
//!
//! A flip that clears the bar is REPORTED (count + margin) so the acceptance
//! effect of the lever can be reasoned about; one that does not is a failure.
//! The oracle refuses two known-bad mutations (a far-off expert substituted
//! into a row; a perturbed weight) so it cannot pass vacuously.
//!
//! Run (GB10):
//!   cargo run --release -p spark-model --features cuda,gpu-examples \
//!     --example fp8_moe_grouped_routing_microtest
use anyhow::{Result, ensure};
use half::bf16;
use spark_model::layers::ops;
use spark_model::weight_map::DenseWeight;
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const H: usize = 2048;
const E: usize = 256;
const TOP_K: usize = 8;
const MAX_M: usize = 32;
/// Two BF16 ulps at |logit| ~ 1..2 (ulp = 2^-7 there): the GEMV and the GEMM
/// accumulate the same 2048 products in different orders in FP32 and round
/// once to BF16, so anything beyond a 1-ulp disagreement is a wrong GEMM.
const LOGIT_TOL: f32 = 2.0 / 128.0;
/// Softmax over logits that agree to LOGIT_TOL, normalised over 8 experts.
const W_TOL: f32 = 2e-3;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn bf16_bytes(rng: &mut Rng, n: usize, scale: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| {
            bf16::from_f32(((rng.next() % 2049) as f32 - 1024.0) / 1024.0 * scale)
                .to_bits()
                .to_le_bytes()
        })
        .collect()
}

fn read_bf16(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut buf = vec![0u8; n * 2];
    gpu.copy_d2h(ptr, &mut buf)?;
    Ok(buf
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect())
}

fn read_u32(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize) -> Result<Vec<u32>> {
    let mut buf = vec![0u8; n * 4];
    gpu.copy_d2h(ptr, &mut buf)?;
    Ok(buf
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn read_f32(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize) -> Result<Vec<f32>> {
    Ok(read_u32(gpu, ptr, n)?
        .into_iter()
        .map(f32::from_bits)
        .collect())
}

/// One router's output for m rows, host side.
struct Routing {
    logits: Vec<f32>,
    idx: Vec<u32>,
    w: Vec<f32>,
}

/// The oracle. Returns the number of admissible flips (rows whose expert sets
/// differ within the rounding margin) or the first inadmissible difference.
fn check(per_token: &Routing, batched: &Routing, m: usize) -> Result<usize> {
    ensure!(per_token.logits.len() >= m * E && batched.logits.len() >= m * E);
    ensure!(per_token.idx.len() >= m * TOP_K && batched.idx.len() >= m * TOP_K);
    let mut flips = 0usize;
    for t in 0..m {
        let lt = &per_token.logits[t * E..(t + 1) * E];
        let lb = &batched.logits[t * E..(t + 1) * E];
        let mut disc = 0f32;
        for (a, b) in lt.iter().zip(lb) {
            ensure!(a.is_finite() && b.is_finite(), "row {t}: nonfinite logit");
            disc = disc.max((a - b).abs());
        }
        ensure!(
            disc <= LOGIT_TOL,
            "row {t}: batched router logits differ from per-token by {disc} (> {LOGIT_TOL})"
        );
        let mut st: Vec<u32> = per_token.idx[t * TOP_K..(t + 1) * TOP_K].to_vec();
        let mut sb: Vec<u32> = batched.idx[t * TOP_K..(t + 1) * TOP_K].to_vec();
        st.sort_unstable();
        sb.sort_unstable();
        ensure!(
            st.windows(2).all(|p| p[0] != p[1]) && sb.windows(2).all(|p| p[0] != p[1]),
            "row {t}: duplicate expert in a top-k set"
        );
        ensure!(
            st.iter().chain(sb.iter()).all(|&e| (e as usize) < E),
            "row {t}: expert id out of range"
        );
        if st != sb {
            // Admissible only as a rounding flip: the per-token row's k-th
            // margin must be inside the observed discrepancy.
            let mut sorted = lt.to_vec();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let margin = sorted[TOP_K - 1] - sorted[TOP_K];
            ensure!(
                margin <= 2.0 * disc,
                "row {t}: expert sets differ ({st:?} vs {sb:?}) but the k-th margin {margin} \
                 exceeds twice the logit discrepancy {disc} — not a rounding flip"
            );
            flips += 1;
            continue;
        }
        // Same set: weights must agree per expert (slot order may differ).
        for k in 0..TOP_K {
            let e = per_token.idx[t * TOP_K + k];
            let wt = per_token.w[t * TOP_K + k];
            let kb = (0..TOP_K)
                .find(|&j| batched.idx[t * TOP_K + j] == e)
                .expect("set equality checked");
            let wb = batched.w[t * TOP_K + kb];
            ensure!(
                wt.is_finite() && wb.is_finite(),
                "row {t}: nonfinite weight"
            );
            ensure!(
                (wt - wb).abs() <= W_TOL,
                "row {t} expert {e}: weight {wb} (batched) vs {wt} (per-token)"
            );
        }
    }
    Ok(flips)
}

fn main() -> Result<()> {
    let gpu = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let gemv_k = gpu.kernel("gemv", "dense_gemv_bf16")?;
    let gemm_k = gpu.kernel("gemm", "dense_gemm_bf16")?;
    let topk_k = gpu.kernel("moe_topk", "moe_topk_softmax")?;
    let topk_batched_k = gpu.kernel("moe_topk", "moe_topk_softmax_batched")?;
    let mut rng = Rng(0x726f_7574_6539_2026);

    // Router [E, H] BF16 at the drafter's `mlp.gate` scale; inputs [MAX_M, H].
    let gate = DenseWeight {
        weight: upload(&gpu, &bf16_bytes(&mut rng, E * H, 0.05))?,
    };
    let input = upload(&gpu, &bf16_bytes(&mut rng, MAX_M * H, 1.0))?;
    let logits_t = gpu.alloc(MAX_M * E * 2)?;
    let logits_b = gpu.alloc(MAX_M * E * 2)?;
    let idx_t = gpu.alloc(MAX_M * TOP_K * 4)?;
    let w_t = gpu.alloc(MAX_M * TOP_K * 4)?;
    let idx_b = gpu.alloc(MAX_M * TOP_K * 4)?;
    let w_b = gpu.alloc(MAX_M * TOP_K * 4)?;

    let mut failures = 0usize;
    let mut first = true;
    for normalize in [true, false] {
        for m in [2usize, 4, 8, 16, 32] {
            for t in 0..m {
                ops::dense_gemv(
                    &gpu,
                    gemv_k,
                    input.offset(t * H * 2),
                    &gate,
                    logits_t.offset(t * E * 2),
                    E as u32,
                    H as u32,
                    0,
                )?;
                ops::moe_topk_softmax(
                    &gpu,
                    topk_k,
                    logits_t.offset(t * E * 2),
                    idx_t.offset(t * TOP_K * 4),
                    w_t.offset(t * TOP_K * 4),
                    E as u32,
                    TOP_K as u32,
                    normalize,
                    0,
                )?;
            }
            ops::dense_gemm(
                &gpu, gemm_k, input, &gate, logits_b, m as u32, E as u32, H as u32, 0,
            )?;
            ops::moe_topk_softmax_batched(
                &gpu,
                topk_batched_k,
                logits_b,
                idx_b,
                w_b,
                E as u32,
                TOP_K as u32,
                normalize,
                m as u32,
                0,
            )?;
            gpu.synchronize(0)?;
            let per_token = Routing {
                logits: read_bf16(&gpu, logits_t, m * E)?,
                idx: read_u32(&gpu, idx_t, m * TOP_K)?,
                w: read_f32(&gpu, w_t, m * TOP_K)?,
            };
            let batched = Routing {
                logits: read_bf16(&gpu, logits_b, m * E)?,
                idx: read_u32(&gpu, idx_b, m * TOP_K)?,
                w: read_f32(&gpu, w_b, m * TOP_K)?,
            };

            if first {
                // Known-bad 1: substitute the row's LOWEST-logit expert into
                // slot 0 of the batched routing — a wrong index, not a flip.
                let mut bad = Routing {
                    logits: batched.logits.clone(),
                    idx: batched.idx.clone(),
                    w: batched.w.clone(),
                };
                let lowest = (0..E)
                    .min_by(|&a, &b| {
                        per_token.logits[a]
                            .partial_cmp(&per_token.logits[b])
                            .unwrap()
                    })
                    .unwrap() as u32;
                bad.idx[0] = lowest;
                let err = check(&per_token, &bad, m)
                    .expect_err("a far-off expert was admitted as a rounding flip");
                println!("KNOWN_BAD wrong-expert: refused: {err}");
                // Known-bad 2: a perturbed weight on an agreeing set.
                let mut bad = Routing {
                    logits: batched.logits.clone(),
                    idx: batched.idx.clone(),
                    w: batched.w.clone(),
                };
                bad.w[0] += 10.0 * W_TOL;
                match check(&per_token, &bad, m) {
                    Err(err) => println!("KNOWN_BAD weight: refused: {err}"),
                    Ok(_) => {
                        // Row 0 flipped on this draw, so its weights are not
                        // compared; that is the oracle's contract, not a hole.
                        let s0: std::collections::BTreeSet<u32> =
                            per_token.idx[..TOP_K].iter().copied().collect();
                        let b0: std::collections::BTreeSet<u32> =
                            batched.idx[..TOP_K].iter().copied().collect();
                        ensure!(s0 != b0, "perturbed weight admitted on an agreeing row");
                        println!(
                            "KNOWN_BAD weight: row 0 flipped on this draw (weights uncompared)"
                        );
                    }
                }
                first = false;
            }

            match check(&per_token, &batched, m) {
                Ok(flips) => println!(
                    "normalize={normalize:5} M={m:2}: logits agree within {LOGIT_TOL:.4}; \
                     expert sets {} rows identical, {flips} rounding flip(s) — OK",
                    m - flips
                ),
                Err(e) => {
                    println!("FAIL normalize={normalize} M={m}: {e}");
                    failures += 1;
                }
            }
        }
    }
    ensure!(failures == 0, "{failures} case(s) failed");
    println!("ALL PASS: batched router == per-token router up to admissible rounding flips");
    Ok(())
}
