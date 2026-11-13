// SPDX-License-Identifier: AGPL-3.0-only
// Translated for Atlas from microsoft/RetrievalAttention (RetroInfer), MIT License,
// Copyright (c) Microsoft Corporation. Upstream: cache_hub/kmeans.py (segment_k_means and its
// Triton kernels) and cache_hub/retroinfer_cache.py:533-554 (centering), commit 03f912c. Changes:
// a CPU mirror of Atlas's deterministic CUDA build (`retro_kmeans_ada.cu`) rather than of the
// Triton atomics: the same reduction orders and roundings, a stable CSR, absolute positions.

//! Responsibility: the CPU mirror of the GPU index build, stage by stage, used by the unit tests
//! and by the GPU microtest as the reference each kernel is compared with. Every stage reproduces
//! the kernel's arithmetic bit for bit (f32 operations in the same order, BF16 round to nearest
//! even) except [`assign`], whose dot products the tensor cores accumulate in their own order: there
//! a near tie may resolve the other way, which the microtest judges with [`dot`].

use super::geometry::{HEAD_DIM, Job, MEAN_CHUNKS};

const D: usize = HEAD_DIM as usize;

/// Round an f32 to BF16 (nearest even) and back.
pub fn bf16(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

/// BF16 bits of an f32 (nearest even).
pub fn bf16_bits(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}

/// The f32 value of BF16 bits.
pub fn from_bf16_bits(b: u16) -> f32 {
    half::bf16::from_bits(b).to_f32()
}

/// An FP8 E4M3 byte dequantized with a per-tensor scale (the kernels' `rk_fp8(b) * scale`).
pub fn from_fp8(b: u8, scale: f32) -> f32 {
    spark_runtime::kv_dequant::e4m3_lut()[usize::from(b)] * scale
}

/// The warp butterfly sum of 32 lane partials (`__shfl_xor_sync`, offsets 16 down to 1); every
/// lane ends with these bits.
pub fn butterfly_sum(mut p: [f32; 32]) -> f32 {
    for off in [16usize, 8, 4, 2, 1] {
        let q = p;
        for l in 0..32 {
            p[l] = q[l] + q[l ^ off];
        }
    }
    p[0]
}

/// Sequential f32 dot product (the tie judge for [`assign`]).
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |s, (x, y)| s + x * y)
}

/// The key mean of one head (`retro_km_mean_partial` + `_final`): 64 ascending chunk sums, summed
/// in chunk order, divided, rounded to BF16. `keys` is `[n_tokens][256]`.
pub fn mean(keys: &[f32], n_tokens: u32) -> Vec<f32> {
    let n = n_tokens as usize;
    let chunk = n.div_ceil(MEAN_CHUNKS as usize);
    let mut sum = vec![0.0f32; D];
    for c in 0..MEAN_CHUNKS as usize {
        let (b, e) = ((c * chunk).min(n), ((c + 1) * chunk).min(n));
        let mut part = vec![0.0f32; D];
        for t in b..e {
            for d in 0..D {
                part[d] += keys[t * D + d];
            }
        }
        for d in 0..D {
            sum[d] += part[d];
        }
    }
    sum.iter().map(|&s| bf16(s / n_tokens as f32)).collect()
}

/// Centered keys `bf16(k - mean)`, `[n_tokens][256]`.
pub fn center(keys: &[f32], mean: &[f32]) -> Vec<f32> {
    keys.iter()
        .enumerate()
        .map(|(i, &k)| bf16(k - mean[i % D]))
        .collect()
}

/// Uniform-stride initial centroids `[nc][256]` (`retro_km_init`).
pub fn init(centered: &[f32], g: &Job) -> Vec<f32> {
    let mut out = vec![0.0f32; g.n_centroids as usize * D];
    for i in 0..g.n_centroids {
        let t = super::geometry::init_index(i, g.n_tokens, g.n_centroids) as usize;
        out[i as usize * D..(i as usize + 1) * D].copy_from_slice(&centered[t * D..(t + 1) * D]);
    }
    out
}

/// One assignment pass (`retro_km_assign`), f32 sequential dots, lowest index on a tie, -1 for a
/// token outside every training segment. `segmented` as in `launch::assign`.
pub fn assign(centered: &[f32], cent: &[f32], g: &Job, segmented: bool) -> Vec<i32> {
    let (seg_tokens, seg_cents, n_seg) = if segmented {
        (g.seg_tokens, g.seg_centroids, g.n_segments)
    } else {
        (g.n_tokens, g.n_centroids, 1)
    };
    let n = g.n_tokens as usize;
    let mut out = vec![-1i32; n];
    let threads = std::thread::available_parallelism()
        .map_or(4, |p| p.get())
        .min(32);
    let per = n.div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        for (ci, chunk) in out.chunks_mut(per).enumerate() {
            sc.spawn(move || {
                for (j, slot) in chunk.iter_mut().enumerate() {
                    let t = (ci * per + j) as u32;
                    if t >= n_seg * seg_tokens {
                        continue;
                    }
                    let lo = (t / seg_tokens) * seg_cents;
                    let row = &centered[t as usize * D..(t as usize + 1) * D];
                    let mut best = f32::NEG_INFINITY;
                    let mut bi = -1i32;
                    for c in lo..lo + seg_cents {
                        let v = dot(row, &cent[c as usize * D..(c as usize + 1) * D]);
                        if v > best {
                            best = v;
                            bi = c as i32;
                        }
                    }
                    *slot = bi;
                }
            });
        }
    });
    out
}

/// The stable CSR of an assignment: `(cluster_start, cluster_size, members)` with members the
/// absolute positions `start + t`, clusters in id order, members ascending.
pub fn csr(assign: &[i32], nc: u32, start: u32) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let mut size = vec![0i32; nc as usize];
    for &a in assign {
        if a >= 0 {
            size[a as usize] += 1;
        }
    }
    let mut cstart = vec![0i32; nc as usize];
    let mut run = 0i32;
    for c in 0..nc as usize {
        cstart[c] = run;
        run += size[c];
    }
    let mut cursor = cstart.clone();
    let mut members = vec![0i32; run as usize];
    for (t, &a) in assign.iter().enumerate() {
        if a >= 0 {
            let slot = &mut cursor[a as usize];
            members[*slot as usize] = (start as usize + t) as i32;
            *slot += 1;
        }
    }
    (cstart, size, members)
}

/// One centroid update (`retro_km_update`) for one head, in place. `values` is required on the
/// final pass (`[n_tokens][256]`, dequantized); `vsum` receives the BF16 value sums then.
pub fn update(
    centered: &[f32],
    values: Option<&[f32]>,
    mean: &[f32],
    cent: &mut [f32],
    vsum: &mut [f32],
    csr: (&[i32], &[i32], &[i32]),
    start: u32,
    final_pass: bool,
) {
    let (cstart, csize, members) = csr;
    for c in 0..csize.len() {
        let cnt = csize[c];
        let first = cstart[c] as usize;
        let mut acc = [0.0f32; D];
        let mut vacc = [0.0f32; D];
        for k in 0..cnt as usize {
            let t = members[first + k] as usize - start as usize;
            for d in 0..D {
                acc[d] += centered[t * D + d];
            }
            if final_pass {
                let v = values.expect("final pass needs the values");
                for d in 0..D {
                    vacc[d] += v[t * D + d];
                }
            }
        }
        let out = &mut cent[c * D..(c + 1) * D];
        if cnt > 0 {
            let mut x = [0.0f32; D];
            for d in 0..D {
                x[d] = acc[d] / cnt as f32;
            }
            if !final_pass {
                let mut lanes = [0.0f32; 32];
                for (l, lane) in lanes.iter_mut().enumerate() {
                    for j in 0..8 {
                        *lane += x[l * 8 + j] * x[l * 8 + j];
                    }
                }
                let ss = butterfly_sum(lanes);
                if ss > 0.0 {
                    let nrm = ss.sqrt();
                    for v in &mut x {
                        *v /= nrm;
                    }
                }
                for d in 0..D {
                    out[d] = bf16(x[d]);
                }
            } else {
                for d in 0..D {
                    out[d] = bf16(bf16(x[d]) + mean[d]);
                }
            }
        } else if final_pass {
            for d in 0..D {
                out[d] = bf16(out[d] + mean[d]);
            }
        }
        if final_pass {
            for d in 0..D {
                vsum[c * D + d] = if cnt > 0 { bf16(vacc[d]) } else { 0.0 };
            }
        }
    }
}

/// One head's index as the GPU build leaves it.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadIndex {
    /// `[nc][256]` BF16 values.
    pub centroids: Vec<f32>,
    /// `[nc][256]` BF16 values.
    pub value_sum: Vec<f32>,
    pub cluster_start: Vec<i32>,
    pub cluster_size: Vec<i32>,
    /// Absolute positions, `[n_tokens]`.
    pub cluster_tokens: Vec<i32>,
    /// The final pass's assignment.
    pub assign: Vec<i32>,
    /// The key mean.
    pub mean: Vec<f32>,
}

/// The whole build of one head from the job's keys and values (`[n_tokens][256]` each,
/// dequantized f32): mean, init, the normalized segmented passes, the final global pass. CSR
/// starts are local (0 = the job's first slot), members absolute positions.
pub fn build_head(keys: &[f32], values: &[f32], g: &Job) -> HeadIndex {
    let m = mean(keys, g.n_tokens);
    let centered = center(keys, &m);
    let mut cent = init(&centered, g);
    let mut vsum = vec![0.0f32; g.n_centroids as usize * D];
    let iters = g.iters.max(1);
    let mut last = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for it in 0..iters {
        let final_pass = it + 1 == iters;
        let a = assign(&centered, &cent, g, !final_pass);
        let (cs, sz, mem) = csr(&a, g.n_centroids, g.first_pos);
        let vals = if final_pass { Some(values) } else { None };
        update(
            &centered,
            vals,
            &m,
            &mut cent,
            &mut vsum,
            (&cs, &sz, &mem),
            g.first_pos,
            final_pass,
        );
        if final_pass {
            last = (cs, sz, mem, a);
        }
    }
    let (cluster_start, cluster_size, cluster_tokens, assign) = last;
    HeadIndex {
        centroids: cent,
        value_sum: vsum,
        cluster_start,
        cluster_size,
        cluster_tokens,
        assign,
        mean: m,
    }
}

/// [`build_head`] for every head, one thread each. `heads[h] = (keys, values)`.
pub fn build(heads: &[(Vec<f32>, Vec<f32>)], g: &Job) -> Vec<HeadIndex> {
    std::thread::scope(|sc| {
        let jobs: Vec<_> = heads
            .iter()
            .map(|(k, v)| sc.spawn(move || build_head(k, v, g)))
            .collect();
        jobs.into_iter()
            .map(|j| j.join().expect("retro reference build thread panicked"))
            .collect()
    })
}
