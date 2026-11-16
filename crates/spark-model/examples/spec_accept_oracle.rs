// SPDX-License-Identifier: AGPL-3.0-only
//! GPU oracle for device-side speculative acceptance (lane L12, row `mtp_device_accept`):
//! `kernels/ada/common/spec_accept_sample_ada.cu` against its CPU model
//! (`spark_model::port::spec_accept::reference`) on fixed logits at the Qwen3.5 vocabulary.
//!
//! * Greedy configurations (masks, the EOS floor, repetition, presence and frequency penalties,
//!   biases, two regimes, exact ties, a NaN): every pick must equal the reference. Any difference is
//!   a failure (exit 1): the greedy path claims bit-identity with the host pipeline.
//! * Sampled configurations (top-k + top-p, full vocabulary, uncapped top-p, min-p, with and without
//!   a draft): `DRAWS` launches with fresh keys. Reported: the share of picks equal to the reference
//!   pick for the same key (the kernel's parallel sums and `expf` may move a pick that sits on a
//!   rounding boundary; below 99% is a failure), and a chi-square of the device picks against the
//!   exact support distribution (above the 1e-6 critical value is a failure).
//! * Timing: the median wall time of one K=4 launch, upload and readback included.
//!
//!   AVAROK_SKIP_BUILD=0 cargo run -p spark-model --release --features cuda,gpu-examples \
//!       --example spec_accept_oracle
//!
//! Env: DRAWS (default 10000), VOCAB (default 248320).

use anyhow::{Context, Result, bail};
use spark_model::port::spec_accept::{
    AcceptMode, AcceptRequest, EditDesc, MODULE, NO_DRAFT, SampleParams, SpecAcceptState, launch,
    reference, status,
};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::GpuBackend;

const K: usize = 4;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// f32 to BF16 bits, round to nearest even (NaN kept quiet).
fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) as u16) | 0x40;
    }
    let round = 0x7FFF + ((b >> 16) & 1);
    ((b.wrapping_add(round)) >> 16) as u16
}

/// K rows of logits: a Zipf-like falloff over a hashed rank, BF16, with planted ties and one NaN.
fn rows(vocab: usize, seed: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(K * vocab * 2);
    for r in 0..K {
        for i in 0..vocab {
            let h = ((i as u64) ^ (seed.wrapping_mul(31) + r as u64))
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .rotate_left(23);
            let rank = (h % vocab as u64) as f32 + 1.0;
            let mut v = 14.0 - 2.2 * rank.ln();
            if i == 1000 + r || i == 2000 + r {
                v = 15.0; // an exact tie for first place
            }
            if r == 2 && i == 77 {
                v = f32::NAN;
            }
            out.extend_from_slice(&bf16_bits(v).to_le_bytes());
        }
    }
    out
}

fn edits(vocab: usize, regime: usize) -> Vec<EditDesc> {
    let mut e = vec![
        EditDesc {
            token: 1000,
            mask: regime == 0,
            ..Default::default()
        },
        EditDesc {
            token: 1001,
            eos: true,
            ..Default::default()
        },
        EditDesc {
            token: 2000,
            rep_count: 3,
            delta: Some(0.7),
            ..Default::default()
        },
        EditDesc {
            token: 2001,
            sub12: true,
            ..Default::default()
        },
        EditDesc {
            token: 5000,
            bias: [-8.0, 0.25],
            n_bias: 2,
            ..Default::default()
        },
    ];
    for t in (10_000..(10_000 + 600).min(vocab as u32)).step_by(3) {
        e.push(EditDesc {
            token: t,
            rep_count: 1 + t % 4,
            delta: Some(0.1 * (t % 5) as f32),
            ..Default::default()
        });
    }
    e.retain(|d| (d.token as usize) < vocab);
    e.sort_by_key(|d| d.token);
    e
}

fn request<'a>(
    logits: spark_runtime::gpu::DevicePtr,
    vocab: usize,
    mode: AcceptMode,
    regimes: &'a [Vec<EditDesc>],
    drafts: &'a [u32],
    key: u64,
) -> AcceptRequest<'a> {
    AcceptRequest {
        logits,
        vocab,
        n_pos: K,
        drafts,
        mode,
        rng_key: key,
        pos_base: 1234,
        eos_until: 2,
        rep_penalty: 1.2,
        regimes,
    }
}

/// Upper chi-square quantile by Wilson-Hilferty, at z standard deviations.
fn chi2_crit(df: usize, z: f64) -> f64 {
    let d = df as f64;
    let a = 2.0 / (9.0 * d);
    d * (1.0 - a + z * a.sqrt()).powi(3)
}

fn main() -> Result<()> {
    let draws = env_usize("DRAWS", 10_000);
    let vocab = env_usize("VOCAB", 248_320);
    let gpu = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    if !gpu.has_module(MODULE) {
        bail!("{MODULE} is not in this build's kernel set (Ada only)");
    }
    let host = rows(vocab, 7);
    let dev = gpu.alloc(host.len())?;
    gpu.copy_h2d(&host, dev)?;
    let state = SpecAcceptState::default();
    let regimes = vec![edits(vocab, 0), edits(vocab, 1)];
    let mut failures = 0usize;

    // ── Greedy: bit-identical to the reference ──
    let r = request(dev, vocab, AcceptMode::Greedy, &regimes, &[], 0);
    let got = launch::run(&state, &gpu, &r)?.context("kernel unavailable")?;
    let want = reference::run(&host, &r);
    if got != want {
        failures += 1;
        println!("greedy: MISMATCH device {got:?} reference {want:?}");
    } else {
        println!("greedy: {} jobs equal to the reference", got.len());
    }

    // ── Sampled ──
    let configs = [
        (
            "top_k20_top_p0.95_T0.6",
            SampleParams {
                temperature: 0.6,
                top_k: 20,
                top_p: 0.95,
                min_p: 0.0,
            },
        ),
        (
            "full_T1.0",
            SampleParams {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
            },
        ),
        (
            "top_p0.5_T0.3",
            SampleParams {
                temperature: 0.3,
                top_k: 0,
                top_p: 0.5,
                min_p: 0.0,
            },
        ),
        (
            "min_p0.2_T0.5",
            SampleParams {
                temperature: 0.5,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.2,
            },
        ),
        (
            "all_k8_p0.8_m0.05_T0.8",
            SampleParams {
                temperature: 0.8,
                top_k: 8,
                top_p: 0.8,
                min_p: 0.05,
            },
        ),
    ];
    let one = vec![regimes[0].clone()];
    for (name, p) in configs {
        for drafts in [vec![], vec![1001u32, 2000, NO_DRAFT, 5000]] {
            let mode = AcceptMode::Sample(p);
            let probe = request(dev, vocab, mode, &one, &drafts, 0);
            let rows_f32: Vec<Vec<f32>> = (0..K)
                .map(|pos| {
                    reference::processed_row(
                        &host[pos * vocab * 2..(pos + 1) * vocab * 2],
                        vocab,
                        &one[0],
                        pos as u32,
                        probe.eos_until,
                        probe.rep_penalty,
                    )
                })
                .collect();
            let mut equal = 0usize;
            let mut fallbacks = 0usize;
            let mut stat_sum = 0.0f64;
            let mut df_sum = 0usize;
            let mut counts: Vec<std::collections::HashMap<u32, u64>> = vec![Default::default(); K];
            for n in 0..draws {
                let req = request(dev, vocab, mode, &one, &drafts, 0x5EED_0000 + n as u64);
                let got = launch::run(&state, &gpu, &req)?.context("kernel unavailable")?;
                let want = reference::run(&host, &req);
                for pos in 0..K {
                    if got[pos].status & status::FALLBACK != 0 {
                        fallbacks += 1;
                        continue;
                    }
                    if got[pos].pick == want[pos].pick {
                        equal += 1;
                    }
                    *counts[pos].entry(got[pos].pick).or_default() += 1;
                }
            }
            for (pos, c) in counts.iter().enumerate() {
                let Ok(s) = reference::support(&rows_f32[pos], &p) else {
                    continue;
                };
                let total: f64 = s.w.iter().map(|&w| w as f64).sum();
                let n_pos: u64 = c.values().sum();
                let (mut stat, mut bins, mut pool_o, mut pool_e) = (0.0f64, 0usize, 0.0f64, 0.0f64);
                for (&id, &w) in s.ids.iter().zip(&s.w) {
                    let e = w as f64 / total * n_pos as f64;
                    let o = *c.get(&id).unwrap_or(&0) as f64;
                    if e >= 5.0 {
                        stat += (o - e).powi(2) / e;
                        bins += 1;
                    } else {
                        pool_o += o;
                        pool_e += e;
                    }
                }
                let outside: u64 = c
                    .iter()
                    .filter(|(id, _)| !s.ids.contains(id))
                    .map(|(_, v)| *v)
                    .sum();
                if outside > 0 {
                    failures += 1;
                    println!("{name} pos {pos}: {outside} picks outside the support");
                }
                if pool_e >= 5.0 {
                    stat += (pool_o - pool_e).powi(2) / pool_e;
                    bins += 1;
                }
                let df = bins.saturating_sub(1).max(1);
                stat_sum += stat;
                df_sum += df;
                if stat > chi2_crit(df, 4.75) {
                    failures += 1;
                    println!("{name} pos {pos}: chi2 {stat:.1} over {df} df FAILS");
                }
            }
            let picked = draws * K - fallbacks;
            let share = if picked == 0 {
                0.0
            } else {
                equal as f64 / picked as f64
            };
            if share < 0.99 && picked > 0 {
                failures += 1;
            }
            println!(
                "{name} drafts={}: {picked} picks, {:.3}% equal to the reference, {fallbacks} fallbacks, chi2 {stat_sum:.1} over {df_sum} df",
                !drafts.is_empty(),
                share * 100.0
            );
        }
    }

    // ── Timing: one K=4 launch with its upload and readback ──
    let p = configs[0].1;
    let mut times = Vec::with_capacity(200);
    for n in 0..200u64 {
        let req = request(dev, vocab, AcceptMode::Sample(p), &one, &[], n);
        let t = std::time::Instant::now();
        let _ = launch::run(&state, &gpu, &req)?;
        times.push(t.elapsed().as_secs_f64() * 1e6);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    println!(
        "timing: K={K} vocab {vocab} median {:.1} us, p90 {:.1} us",
        times[100], times[180]
    );

    gpu.free(dev)?;
    if failures > 0 {
        println!("spec_accept_oracle: {failures} failure(s)");
        std::process::exit(1);
    }
    println!("spec_accept_oracle: PASS");
    Ok(())
}
