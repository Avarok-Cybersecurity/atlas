// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: unit tests of the RetroInfer estimation-attention reference (port lane
//! L34b): equality with L33's oracle (`oracle_eval::retro_attention` over
//! `oracle_eval::exact_positions`) for decode and for trimmed verify rows, the identical-key
//! cluster case against plain dense softmax, and the degenerate cases (large logits, nothing
//! attended, empty and out-of-range estimation ids). Fixtures are the parent module's.

use super::super::attn::{self, HostAttnIn, reference_attention};
use super::super::gather::HostExact;
use super::super::oracle::HeadKv;
use super::super::oracle_eval;
use super::{
    CAPC, Fx, HD, KVH, LEN, Lcg, QH, RET, SCALE, bits, fx, gather_run, oracle_zones, zones_dev,
};

type ZonesDev = (Vec<i32>, Vec<i32>, Vec<i32>);

fn attn_in<'a>(
    f: &'a Fx,
    ex: &'a HostExact,
    q: &'a [u16],
    z: &'a ZonesDev,
    rows: usize,
) -> HostAttnIn<'a> {
    HostAttnIn {
        q,
        q_row_stride: QH * HD,
        rows,
        row_trim: 0,
        q_heads: QH,
        kv_heads: KVH,
        scale: SCALE,
        exact: ex,
        est_ids: &z.2,
        counts: &z.1,
        es_cap: 6,
        centroids: &f.cent,
        value_sum: &f.vsum,
        cluster_size: &f.size,
        cap_centroids: CAPC,
    }
}

fn max_err(got: &[f32], want: &[f64]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(&a, &b)| (f64::from(a) - b).abs())
        .fold(0.0, f64::max)
}

#[test]
fn attention_matches_the_oracle_for_decode_and_trimmed_verify_rows() {
    for (seed, gain) in [(21u64, 1.0f32), (22, 6.0)] {
        let f = fx(seed, gain);
        let z = zones_dev(&RET, None);
        let rows = 3usize;
        let lens: Vec<i32> = (0..rows as i32).map(|r| LEN as i32 - 2 + r).collect();
        let ex = gather_run(&f, &lens, 512, &z.0, &z.1);
        let mut r = Lcg(seed + 100);
        let q: Vec<u16> = (0..rows * QH * HD).map(|_| bits(r.unit() * gain)).collect();
        let (out, lse) = reference_attention(&attn_in(&f, &ex, &q, &z, rows));
        for row in 0..rows {
            for j in 0..QH {
                let h = j / (QH / KVH);
                let kv = HeadKv {
                    k: &f.keys[h],
                    v: &f.vals[h],
                    head_dim: HD,
                };
                let len = lens[row] as u32;
                let zh = oracle_zones(h);
                let (pos, _) =
                    oracle_eval::exact_positions(&f.idx[h], &zh, &f.counts, len, Some(512));
                let at = (row * QH + j) * HD;
                let qf: Vec<f32> = q[at..at + HD].iter().map(|&b| attn::bf16_f32(b)).collect();
                let (want, want_lse) =
                    oracle_eval::retro_attention(kv, &f.idx[h], &pos, &zh.estimation, &qf, SCALE);
                let err = max_err(&out[at..at + HD], &want);
                assert!(err < 1e-5, "row {row} head {j} gain {gain}: {err:e}");
                let lse_err = (f64::from(lse[row * QH + j]) - want_lse).abs();
                assert!(
                    lse_err < 1e-4 * want_lse.abs().max(1.0),
                    "row {row} head {j}: LSE {lse_err:e}"
                );
            }
        }
    }
}

/// Softmax over `toks` (key, value) for query head `j` of row 0, in f64.
fn dense(q: &[u16], j: usize, toks: &[(&[f32], &[f32])]) -> Vec<f64> {
    let qf: Vec<f64> = q[j * HD..][..HD]
        .iter()
        .map(|&b| f64::from(attn::bf16_f32(b)))
        .collect();
    let dot = |k: &[f32]| {
        qf.iter()
            .zip(k)
            .map(|(a, &b)| a * f64::from(b))
            .sum::<f64>()
            / 16.0
    };
    let s: Vec<f64> = toks.iter().map(|t| dot(t.0)).collect();
    let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
    let den: f64 = w.iter().sum();
    (0..HD)
        .map(|d| {
            toks.iter()
                .zip(&w)
                .map(|(t, w)| w * f64::from(t.1[d]))
                .sum::<f64>()
                / den
        })
        .collect()
}

#[test]
fn identical_key_clusters_equal_dense_attention_over_their_members() {
    // Every member of the cluster shares its key: the estimation term is then exact.
    let mut f = fx(7, 1.0);
    let mut ex = HostExact {
        valid: vec![0; KVH],
        ..HostExact::filled(KVH as u32, 4, 0, -1)
    };
    let mut r = Lcg(70);
    let key: Vec<f32> = (0..HD).map(|_| r.unit()).collect();
    let vals: Vec<Vec<f32>> = (0..5)
        .map(|_| (0..HD).map(|_| r.unit()).collect())
        .collect();
    let row = 2; // cluster 2 of head 0, five members
    f.size[row] = 5;
    for d in 0..HD {
        f.cent[row * HD + d] = bits(key[d]);
        f.vsum[row * HD + d] = bits(vals.iter().map(|v| v[d]).sum());
    }
    let vs: Vec<f32> = (0..HD)
        .map(|d| attn::bf16_f32(f.vsum[row * HD + d]))
        .collect();
    // One exact entry beside it.
    ex.valid[0] = 1;
    ex.pos[0] = 0;
    for d in 0..HD {
        ex.k[d] = bits(vals[0][d]);
        ex.v[d] = bits(key[d]);
    }
    let mut z = zones_dev(&RET, None);
    z.1[1] = 1; // head 0 estimation: one cluster
    z.2[0] = 2;
    let q: Vec<u16> = (0..QH * HD).map(|_| bits(r.unit() * 3.0)).collect();
    let (out, _) = reference_attention(&attn_in(&f, &ex, &q, &z, 1));
    // Dense equivalent: the exact token, then five tokens of `key` whose values sum to `vs`.
    let mean: Vec<f32> = vs.iter().map(|x| x / 5.0).collect();
    let k0: Vec<f32> = (0..HD).map(|d| attn::bf16_f32(ex.k[d])).collect();
    let v0: Vec<f32> = (0..HD).map(|d| attn::bf16_f32(ex.v[d])).collect();
    let mut toks: Vec<(&[f32], &[f32])> = vec![(&k0[..], &v0[..])];
    toks.extend((0..5).map(|_| (&key[..], &mean[..])));
    let err = max_err(&out[..HD], &dense(&q, 0, &toks));
    assert!(err < 1e-6, "{err:e}");
}

#[test]
fn attention_degenerate_cases() {
    let f = fx(8, 1.0);
    let z = zones_dev(&RET, None);
    let ex = gather_run(&f, &[LEN as i32], 512, &z.0, &z.1);
    let mut r = Lcg(5);
    // Large logits stay finite.
    let q: Vec<u16> = (0..QH * HD).map(|_| bits(r.unit() * 3000.0)).collect();
    let (out, lse) = reference_attention(&attn_in(&f, &ex, &q, &z, 1));
    assert!(out.iter().chain(&lse).all(|x| x.is_finite()));
    // Nothing attended: everything trimmed and no estimation clusters.
    let mut none = zones_dev(&RET, None);
    none.1.iter_mut().for_each(|c| *c = 0);
    let mut trimmed = attn_in(&f, &ex, &q, &none, 1);
    trimmed.row_trim = 10_000;
    let (out, lse) = reference_attention(&trimmed);
    assert!(out.iter().all(|&x| x == 0.0));
    assert!(lse.iter().all(|&x| x == f32::NEG_INFINITY));
    // Empty and out-of-range estimation ids change nothing.
    let mut bad = zones_dev(&RET, None);
    bad.2[0] = 20; // an empty cluster
    bad.2[1] = CAPC as i32 + 9;
    let mut base = zones_dev(&RET, None);
    base.2[0] = -1;
    base.2[1] = -1;
    let got = reference_attention(&attn_in(&f, &ex, &q, &bad, 1));
    assert_eq!(got, reference_attention(&attn_in(&f, &ex, &q, &base, 1)));
}
