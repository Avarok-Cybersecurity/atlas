// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: unit tests of the RetroInfer gather and estimation attention references
//! (port lane L34b) against L33's oracle (`oracle_eval::exact_positions`,
//! `oracle_eval::retro_attention`): entry order, whole-cluster drops, verify-row trims, the
//! combined formula; plus the properties the oracle does not cover (FP8 dequant bits,
//! unreadable tokens, sentinels after the valid length, invalid ids). This file holds the
//! shared fixtures and the gather tests; the attention tests are in `kernels_tests_attn.rs`
//! and the launch ABI tests in `kernels_tests_abi.rs`. The GPU side of the same comparisons
//! is `examples/retro_attn_microtest.rs`.

use super::attn;
use super::gather::{self, HostExact, HostLists, HostPaged, HostZones};
use super::oracle::HeadIndex;
use super::oracle_eval::{self, Zones};
use super::state::IndexCounts;
use crate::layers::qwen3_attention::sparse_seam::KvDtypeTag;

#[path = "kernels_tests_abi.rs"]
mod abi;
#[path = "kernels_tests_attn.rs"]
mod attn_tests;

const HD: usize = 256;
const KVH: usize = 2;
const BS: usize = 16;
const LEN: u32 = 300;
const CAPC: usize = 24;
const QH: usize = 8;
const SCALE: f32 = 1.0 / 16.0;

pub(super) struct Lcg(pub u64);
impl Lcg {
    pub fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
    /// A BF16-representable value in [-1, 1).
    pub fn unit(&mut self) -> f32 {
        let x = self.next() as f32 / (1u64 << 31) as f32 * 2.0 - 1.0;
        attn::bf16_f32(bits(x))
    }
}

pub(super) fn bits(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}

/// Keys, values, a k-means-shaped index and the device layouts of both, for every head.
struct Fx {
    keys: Vec<Vec<f32>>,
    vals: Vec<Vec<f32>>,
    idx: Vec<HeadIndex>,
    counts: IndexCounts,
    pool_k: Vec<u8>,
    pool_v: Vec<u8>,
    table: Vec<i32>,
    offsets: Vec<i32>,
    tokens: Vec<i32>,
    cent: Vec<u16>,
    vsum: Vec<u16>,
    size: Vec<i32>,
}

/// Token `t` of `[sink, indexed_end)` goes to cluster `(floor(sqrt(t - sink)) + h) % 16`, so
/// cluster sizes vary (1 to 29); clusters 16..24 stay empty.
fn fx(seed: u64, gain: f32) -> Fx {
    let (sink, ie) = (4u32, LEN - 64);
    let mut r = Lcg(seed);
    let mut rand = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| attn::bf16_f32(bits(r.unit() * gain)))
            .collect()
    };
    let keys: Vec<Vec<f32>> = (0..KVH).map(|_| rand(LEN as usize * HD)).collect();
    let vals: Vec<Vec<f32>> = (0..KVH).map(|_| rand(LEN as usize * HD)).collect();
    let blocks = LEN as usize / BS + 2;
    // A scrambled but one-to-one table: the multiplier must be coprime to `blocks` (20), or
    // several logical blocks share one physical block and later writes overwrite earlier rows.
    let table: Vec<i32> = (0..blocks).map(|l| ((l * 7 + 1) % blocks) as i32).collect();
    let mut seen = table.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(
        seen.len(),
        blocks,
        "the fixture block table must be one-to-one"
    );
    let stride = BS * KVH * HD * 2;
    let (mut pool_k, mut pool_v) = (vec![0u8; blocks * stride], vec![0u8; blocks * stride]);
    let (mut offsets, mut tokens) = (Vec::new(), vec![0i32; KVH * LEN as usize]);
    let (mut cent, mut vsum) = (vec![0u16; KVH * CAPC * HD], vec![0u16; KVH * CAPC * HD]);
    let (mut size, mut idx) = (vec![0i32; KVH * CAPC], Vec::new());
    for h in 0..KVH {
        for t in 0..LEN as usize {
            let at = table[t / BS] as usize * stride + ((t % BS) * KVH + h) * HD * 2;
            for d in 0..HD {
                pool_k[at + 2 * d..at + 2 * d + 2]
                    .copy_from_slice(&bits(keys[h][t * HD + d]).to_le_bytes());
                pool_v[at + 2 * d..at + 2 * d + 2]
                    .copy_from_slice(&bits(vals[h][t * HD + d]).to_le_bytes());
            }
        }
        let mut lists = vec![Vec::new(); CAPC];
        for t in sink..ie {
            lists[((f64::from(t - sink).sqrt() as usize) + h) % 16].push(t);
        }
        let mut hi = HeadIndex {
            head_dim: HD,
            ..HeadIndex::default()
        };
        offsets.push(0);
        let mut at = 0usize;
        for (c, l) in lists.iter().enumerate() {
            let row = h * CAPC + c;
            for d in 0..HD {
                let ks: f32 = l.iter().map(|&t| keys[h][t as usize * HD + d]).sum();
                let vs: f32 = l.iter().map(|&t| vals[h][t as usize * HD + d]).sum();
                let mean = if l.is_empty() {
                    0.0
                } else {
                    ks / l.len() as f32
                };
                cent[row * HD + d] = bits(mean);
                vsum[row * HD + d] = bits(vs);
                hi.centroids.push(attn::bf16_f32(bits(mean)));
                hi.value_sum.push(attn::bf16_f32(bits(vs)));
            }
            size[row] = l.len() as i32;
            hi.sizes.push(l.len() as u32);
            for &t in l {
                tokens[h * LEN as usize + at] = t as i32;
                at += 1;
            }
            offsets.push(at as i32);
        }
        hi.lists = lists;
        idx.push(hi);
    }
    let counts = IndexCounts {
        indexed_end: ie,
        sink,
        n_centroids: CAPC as u32,
        ..IndexCounts::default()
    };
    Fx {
        keys,
        vals,
        idx,
        counts,
        pool_k,
        pool_v,
        table,
        offsets,
        tokens,
        cent,
        vsum,
        size,
    }
}

/// Retrieval ids per head (rank order) and the zone-list arrays (`cap` 6, counts pairs).
const RET: [[i32; 5]; 2] = [[12, 3, 14, 1, 20], [9, 0, 15, 4, 2]];
const EST: [[i32; 6]; 2] = [[0, 2, 4, 5, 6, 7], [1, 3, 5, 6, 7, 8]];

fn zones_dev(ret: &[[i32; 5]; 2], extra: Option<i32>) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let mut ids = vec![-1i32; KVH * 6];
    let mut counts = vec![0i32; KVH * 2];
    let mut est = vec![-1i32; KVH * 6];
    for h in 0..KVH {
        ids[h * 6..h * 6 + 5].copy_from_slice(&ret[h]);
        if let Some(x) = extra {
            ids[h * 6 + 5] = x;
        }
        est[h * 6..h * 6 + 6].copy_from_slice(&EST[h]);
        counts[h * 2] = if extra.is_some() { 6 } else { 5 };
        counts[h * 2 + 1] = 6;
    }
    (ids, counts, est)
}

fn oracle_zones(h: usize) -> Zones {
    Zones {
        retrieval: RET[h].iter().map(|&c| c as u32).collect(),
        estimation: EST[h].iter().map(|&c| c as u32).collect(),
    }
}

fn gather_run(f: &Fx, lens: &[i32], cap: u32, ids: &[i32], counts: &[i32]) -> HostExact {
    let header = f.counts.to_words();
    let p = HostPaged {
        k_pool: &f.pool_k,
        v_pool: &f.pool_v,
        dtype: KvDtypeTag::Bf16,
        block_size: BS as u32,
        kv_heads: KVH as u32,
        block_stride: BS * KVH * HD * 2,
        k_scale: 1.0,
        v_scale: 1.0,
        block_table: &f.table,
        row_lens: lens,
    };
    let l = HostLists {
        header: &header,
        offsets: &f.offsets,
        tokens: &f.tokens,
        cap_centroids: CAPC as u32,
        cap_tokens: LEN,
    };
    let z = HostZones {
        ids,
        counts,
        cap: 6,
    };
    let mut out = HostExact::filled(KVH as u32, cap, 0xABCD, -9);
    gather::reference_gather(&p, &l, &z, &mut out);
    out
}

#[test]
fn gather_matches_the_oracle_order_drops_and_rows() {
    let f = fx(11, 1.0);
    let (ids, counts, _) = zones_dev(&RET, None);
    for cap in [512u32, 98, 70] {
        let out = gather_run(&f, &[LEN as i32], cap, &ids, &counts);
        for h in 0..KVH {
            let (want, dropped) = oracle_eval::exact_positions(
                &f.idx[h],
                &oracle_zones(h),
                &f.counts,
                LEN,
                Some(cap as usize),
            );
            let n = out.valid[h] as usize;
            let got: Vec<u32> = out.pos[h * cap as usize..][..n]
                .iter()
                .map(|&p| p as u32)
                .collect();
            assert_eq!(got, want, "cap {cap} head {h}");
            assert_eq!(out.dropped[h] as u32, dropped, "cap {cap} head {h}");
            for (e, &t) in want.iter().enumerate() {
                let at = (h * cap as usize + e) * HD;
                let key: Vec<u16> = f.keys[h][t as usize * HD..][..HD]
                    .iter()
                    .map(|&x| bits(x))
                    .collect();
                let val: Vec<u16> = f.vals[h][t as usize * HD..][..HD]
                    .iter()
                    .map(|&x| bits(x))
                    .collect();
                assert_eq!(out.k[at..at + HD], key[..], "K h{h} e{e}");
                assert_eq!(out.v[at..at + HD], val[..], "V h{h} e{e}");
            }
            assert!(
                out.pos[h * cap as usize + n..(h + 1) * cap as usize]
                    .iter()
                    .all(|&p| p == -9)
            );
        }
    }
    // cap 98 leaves room 30 on head 0: cluster 12 (25) fits, 3 (7) and 14 (29) drop, 1 (3) fits.
    let out = gather_run(&f, &[LEN as i32], 98, &ids, &counts);
    assert_eq!(out.dropped[0], 36);
}

#[test]
fn gather_skips_invalid_ids_like_the_oracle_without_them() {
    let f = fx(5, 1.0);
    let (ids, counts, _) = zones_dev(&RET, Some(CAPC as i32 + 3));
    let out = gather_run(&f, &[LEN as i32], 512, &ids, &counts);
    let (ids2, counts2, _) = zones_dev(&RET, None);
    assert_eq!(out, gather_run(&f, &[LEN as i32], 512, &ids2, &counts2));
}

#[test]
fn gather_fp8_rows_equal_kv_dequant_and_unreadable_tokens_are_zero() {
    let f = fx(3, 1.0);
    let header = f.counts.to_words();
    let mut r = Lcg(9);
    let stride = BS * KVH * HD;
    let mut fp8 = |n: usize| -> Vec<u8> {
        (0..n)
            .map(|_| match r.next() as u8 {
                b if b & 0x7f == 0x7f => 0x30,
                b => b,
            })
            .collect()
    };
    let (k8, v8) = (fp8(f.table.len() * stride), fp8(f.table.len() * stride));
    let mut table = f.table.clone();
    table[0] = -1; // the sink block is unreadable
    let p = HostPaged {
        k_pool: &k8,
        v_pool: &v8,
        dtype: KvDtypeTag::Fp8,
        block_size: BS as u32,
        kv_heads: KVH as u32,
        block_stride: stride,
        k_scale: 0.5,
        v_scale: 0.25,
        block_table: &table,
        row_lens: &[LEN as i32],
    };
    let l = HostLists {
        header: &header,
        offsets: &f.offsets,
        tokens: &f.tokens,
        cap_centroids: CAPC as u32,
        cap_tokens: LEN,
    };
    let (ids, counts, _) = zones_dev(&RET, None);
    let z = HostZones {
        ids: &ids,
        counts: &counts,
        cap: 6,
    };
    let mut out = HostExact::filled(KVH as u32, 512, 0xABCD, -9);
    gather::reference_gather(&p, &l, &z, &mut out);
    for h in 0..KVH {
        let (want, _) = gather::exact_positions(&p, &l, &z, h, 512);
        for (e, &t) in want.iter().enumerate() {
            let at = (h * 512 + e) * HD;
            if t < BS as i32 {
                assert_eq!(out.pos[h * 512 + e], -1);
                assert!(out.k[at..at + HD].iter().all(|&b| b == 0));
                continue;
            }
            let src = table[t as usize / BS] as usize * stride + ((t as usize % BS) * KVH + h) * HD;
            for (pool, scale, got) in [(&k8, 0.5, &out.k), (&v8, 0.25, &out.v)] {
                let mut row = vec![half::bf16::ZERO; HD];
                spark_runtime::kv_dequant::dequant_fp8_to_bf16(
                    &pool[src..src + HD],
                    scale,
                    &mut row,
                );
                assert!(
                    row.iter()
                        .zip(&got[at..at + HD])
                        .all(|(a, &b)| a.to_bits() == b)
                );
            }
        }
    }
}
