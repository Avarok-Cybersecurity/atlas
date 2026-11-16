// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the page sketch's CPU checks (`kv_cache::sketch`): the f64 RoPE inverse
//! and the pre-RoPE page mean at positions to 200K with the Qwen3.5 rotary shape, the
//! count-weighted merge of 8 page means equal to the block mean, an f32 mirror of the
//! sketch kernel's arithmetic against the oracle, the job planner, the tail span, and
//! the buffers on the mock backend (exact allocation, launch geometry, uploaded jobs,
//! refusals, nothing allocated when the module is absent).

use super::oracle::{mean_rows, merge_means, page_mean_pre_rope, rope_forward, rope_inverse};
use super::tests::Rng;
use crate::gpu::mock::{MockArg, MockGpuBackend};
use crate::gpu::{DevicePtr, GpuBackend};
use crate::kv_cache::KvCacheDtype;
use crate::kv_cache::sketch::{
    KERNEL_MODULE, SketchBuffers, SketchConfig, SketchJob, SketchLayer, SketchView, TailBatch,
    WrittenRange, blocks_for_budget, plan_jobs, tail_pages_span,
};

const THETA: f64 = 10_000_000.0;
const ROT: usize = 64;
const HD: usize = 256;

fn vec_normal(rng: &mut Rng, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.normal()).collect()
}

#[test]
fn rope_inverse_undoes_forward_up_to_200k() {
    let mut rng = Rng::new(21);
    for pos in [0u32, 1, 15, 4_095, 65_537, 199_999] {
        let x = vec_normal(&mut rng, HD);
        let mut y = x.clone();
        rope_forward(&mut y, pos, ROT, THETA);
        if pos > 0 {
            assert!(
                y[..ROT].iter().zip(&x).any(|(a, b)| (a - b).abs() > 1e-6),
                "rotated"
            );
        }
        assert_eq!(&y[ROT..], &x[ROT..], "dims past rotary_dim pass through");
        rope_inverse(&mut y, pos, ROT, THETA);
        let err = y
            .iter()
            .zip(&x)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-9, "pos {pos}: {err:e}");
    }
}

#[test]
fn the_pre_rope_page_mean_is_the_mean_of_the_unrotated_keys() {
    let mut rng = Rng::new(22);
    for (pos0, fill) in [(0u32, 16usize), (199_984, 16), (12_345 * 16, 9)] {
        let pre: Vec<f64> = vec_normal(&mut rng, fill * HD);
        let mut post = pre.clone();
        for t in 0..fill {
            rope_forward(&mut post[t * HD..(t + 1) * HD], pos0 + t as u32, ROT, THETA);
        }
        let got = page_mean_pre_rope(&post, fill, HD, pos0, ROT, THETA);
        let want = mean_rows(&pre, fill, HD);
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-9, "pos0 {pos0}: {err:e}");
    }
}

#[test]
fn merging_eight_page_means_by_count_is_the_block_mean() {
    let mut rng = Rng::new(23);
    for _ in 0..50 {
        let counts: Vec<u32> = (0..8)
            .map(|i| {
                if i == 7 {
                    1 + rng.below(16) as u32
                } else {
                    rng.below(17) as u32
                }
            })
            .collect();
        let tokens: Vec<Vec<f64>> = counts
            .iter()
            .map(|&c| vec_normal(&mut rng, c as usize * 32))
            .collect();
        let pages: Vec<Vec<f64>> = tokens
            .iter()
            .zip(&counts)
            .map(|(t, &c)| mean_rows(t, c as usize, 32))
            .collect();
        let refs: Vec<&[f64]> = pages.iter().map(|p| p.as_slice()).collect();
        let merged = merge_means(&refs, &counts).expect("tail page has tokens");
        let all: Vec<f64> = tokens.concat();
        let want = mean_rows(&all, counts.iter().sum::<u32>() as usize, 32);
        let err = merged
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-12, "{err:e}");
    }
    assert!(
        merge_means(&[&[1.0][..]], &[0]).is_none(),
        "no tokens, no mean"
    );
}

/// The sketch kernel's arithmetic in f32 (`kps_sketch_page`), from BF16 stored keys.
fn kernel_mirror(stored: &[f32], fill: usize, pos0: u32) -> Vec<f32> {
    let half = ROT / 2;
    (0..HD)
        .map(|d| {
            let mut acc = 0.0f32;
            for o in 0..fill {
                let y = stored[o * HD + d];
                if d < ROT {
                    let (pair, partner, d0) = if d < half {
                        (d, d + half, true)
                    } else {
                        (d - half, d - half, false)
                    };
                    let freq = (1.0 / THETA.powf((2 * pair) as f64 / ROT as f64)) as f32;
                    let angle = (pos0 + o as u32) as f32 * freq;
                    let (s, c) = angle.sin_cos();
                    let yp = stored[o * HD + partner];
                    acc += if d0 { y * c + yp * s } else { y * c - yp * s };
                } else {
                    acc += y;
                }
            }
            half::bf16::from_f32(acc / fill as f32).to_f32()
        })
        .collect()
}

#[test]
fn the_kernel_arithmetic_matches_the_oracle_within_bf16_tolerance() {
    let mut rng = Rng::new(24);
    for (pos0, fill) in [(0u32, 16usize), (65_536, 16), (199_984, 7)] {
        let pre: Vec<f64> = vec_normal(&mut rng, fill * HD);
        let mut post = pre.clone();
        for t in 0..fill {
            rope_forward(&mut post[t * HD..(t + 1) * HD], pos0 + t as u32, ROT, THETA);
        }
        let stored: Vec<f32> = post
            .iter()
            .map(|&v| half::bf16::from_f64(v).to_f32())
            .collect();
        let got = kernel_mirror(&stored, fill, pos0);
        let want = mean_rows(&pre, fill, HD);
        let num: f64 = got
            .iter()
            .zip(&want)
            .map(|(&g, w)| (f64::from(g) - w).powi(2))
            .sum();
        let den: f64 = want.iter().map(|w| w * w).sum();
        assert!(
            (num / den).sqrt() < 1e-2,
            "pos0 {pos0}: {:e}",
            (num / den).sqrt()
        );
    }
}

#[test]
fn jobs_cover_every_page_a_written_row_lands_in() {
    let table_v: Vec<u32> = (100..140).collect();
    let table = table_v.as_slice();
    let ranges = [
        WrittenRange {
            first_pos: 10,
            n_new: 30,
            host_table: Some(table),
        },
        WrittenRange {
            first_pos: 47,
            n_new: 1,
            host_table: None,
        },
        WrittenRange {
            first_pos: 0,
            n_new: 0,
            host_table: None,
        },
        WrittenRange {
            first_pos: 630,
            n_new: 20,
            host_table: Some(table),
        },
    ];
    let (jobs, uncovered) = plan_jobs(&ranges, 16, true);
    let want = [
        SketchJob {
            block: 100,
            stream: 0,
            logical_page: 0,
            fill: 16,
        },
        SketchJob {
            block: 101,
            stream: 0,
            logical_page: 1,
            fill: 16,
        },
        SketchJob {
            block: 102,
            stream: 0,
            logical_page: 2,
            fill: 8,
        },
        SketchJob {
            block: -1,
            stream: 1,
            logical_page: 2,
            fill: 16,
        },
        SketchJob {
            block: 139,
            stream: 3,
            logical_page: 39,
            fill: 16,
        },
    ];
    assert_eq!(jobs, want);
    // Positions 640..650 fall in page 40, past the 40-entry host table.
    assert_eq!(uncovered, 10);
    let (jobs, uncovered) = plan_jobs(&ranges[1..2], 16, false);
    assert!(jobs.is_empty());
    assert_eq!(uncovered, 1, "no host table and no device tables");
}

#[test]
fn tail_span_counts_the_pages_n_tokens_can_touch() {
    assert_eq!(tail_pages_span(0, 16), 0);
    assert_eq!(tail_pages_span(1, 16), 1);
    assert_eq!(tail_pages_span(2, 16), 2);
    assert_eq!(tail_pages_span(16, 16), 2);
    assert_eq!(tail_pages_span(17, 16), 2);
    assert_eq!(tail_pages_span(18, 16), 3);
    for n in 1..64u32 {
        for first in 0..48u32 {
            let touched = (first + n - 1) / 16 - first / 16 + 1;
            assert!(touched <= tail_pages_span(n, 16), "n {n} first {first}");
        }
    }
}

fn cfg_4b(num_blocks: usize) -> SketchConfig {
    SketchConfig {
        num_blocks,
        attn_layers: 8,
        kv_heads: 4,
        head_dim: 256,
        block_size: 16,
        rotary_dim: 64,
        rope_theta: 1.0e7,
        max_jobs: 4,
    }
}

#[test]
fn the_4b_sketch_costs_16_kib_per_page() {
    assert_eq!(SketchConfig::bytes_per_block(8, 4, 256), 16 * 1024 + 1);
    let c = cfg_4b(1000);
    assert_eq!(c.means_bytes(), 1000 * 16 * 1024);
    assert_eq!(c.total_bytes(), 1000 * (16 * 1024 + 1) + 4 * 16);
    assert_eq!(
        blocks_for_budget(1 << 30, 128 << 10, 16 * 1024 + 1),
        (1 << 30) / (144 * 1024 + 1)
    );
    assert!(
        SketchConfig {
            block_size: 256,
            ..c
        }
        .validate()
        .is_err(),
        "u8 counts"
    );
    assert!(
        SketchConfig {
            rotary_dim: 63,
            ..c
        }
        .validate()
        .is_err()
    );
    assert!(
        SketchConfig {
            rotary_dim: 512,
            ..c
        }
        .validate()
        .is_err()
    );
    assert!(c.validate().is_ok());
}

fn layer(dtype: KvCacheDtype, pool: DevicePtr) -> SketchLayer {
    SketchLayer {
        attn_layer: 3,
        k_pool: pool,
        dtype,
        block_stride_bytes: 128 << 10,
        kv_scale: 0.5,
    }
}

#[test]
fn buffers_allocate_exactly_and_launch_the_planned_geometry() {
    let gpu = MockGpuBackend::new();
    let c = cfg_4b(64);
    let sk = SketchBuffers::new(&gpu, c).unwrap();
    assert_eq!(gpu.live_alloc_count(), 3);
    assert_eq!(gpu.live_bytes(), Some(c.total_bytes()));
    assert!(
        gpu.read_alloc(sk.counts_ptr())
            .unwrap()
            .iter()
            .all(|&b| b == 0),
        "counts start at 0"
    );
    let pool = gpu.alloc(64).unwrap();
    let jobs: Vec<SketchJob> = (0..10)
        .map(|i| SketchJob {
            block: i,
            stream: 0,
            logical_page: i as u32,
            fill: 16,
        })
        .collect();
    let n = sk
        .update_jobs(
            &gpu,
            &layer(KvCacheDtype::Bf16, pool),
            &jobs,
            DevicePtr::NULL,
            0,
            7,
        )
        .unwrap();
    assert_eq!(n, 3, "10 jobs in launches of at most 4");
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 3);
    assert_eq!(l[0].grid, [4, 4, 1]);
    assert_eq!(l[2].grid, [2, 4, 1]);
    assert_eq!(l[0].block, [256, 1, 1]);
    assert_eq!(l[0].stream, 7);
    assert_eq!(l[0].args[0], MockArg::Buffer(pool));
    assert_eq!(l[0].args.len(), 17);
    // The last upload holds jobs 8 and 9.
    let words = gpu
        .read_alloc(DevicePtr(l[0].args[3].clone().buffer()))
        .unwrap();
    let w = |i: usize| i32::from_le_bytes(words[4 * i..4 * i + 4].try_into().unwrap());
    assert_eq!((w(0), w(1), w(2), w(3), w(4)), (8, 0, 8, 16, 9));
    let t = TailBatch {
        block_table: pool,
        seq_lens: pool,
        max_blocks_per_seq: 8,
        n_rows: 8,
        rows_per_seq: 4,
        n_new: 4,
    };
    sk.update_tail(&gpu, &layer(KvCacheDtype::Fp8, pool), &t, 7)
        .unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 4);
    assert_eq!(l[3].grid, [2 * 2, 4, 1], "2 sequences x 2 pages");
    assert_eq!(l[3].args.len(), 20);
    let view_pages = [5u32, 9];
    let v = sk.view(&view_pages);
    assert_eq!(
        (v.layers(), v.kv_heads(), v.head_dim(), v.page_tokens()),
        (8, 4, 256, 16)
    );
    assert_eq!(v.pages(), &view_pages);
    assert_eq!(v.means(), sk.means_ptr());
    sk.free(&gpu).unwrap();
    assert_eq!(gpu.live_alloc_count(), 1, "only the test's pool is left");
}

#[test]
fn unsupported_inputs_are_refused_without_a_launch() {
    let gpu = MockGpuBackend::new();
    let sk = SketchBuffers::new(&gpu, cfg_4b(8)).unwrap();
    let job = [SketchJob {
        block: 0,
        stream: 0,
        logical_page: 0,
        fill: 1,
    }];
    let turbo = layer(KvCacheDtype::Turbo4, DevicePtr::NULL);
    assert!(
        sk.update_jobs(&gpu, &turbo, &job, DevicePtr::NULL, 0, 0)
            .is_err()
    );
    let far = SketchLayer {
        attn_layer: 8,
        ..layer(KvCacheDtype::Bf16, DevicePtr::NULL)
    };
    assert!(
        sk.update_jobs(&gpu, &far, &job, DevicePtr::NULL, 0, 0)
            .is_err()
    );
    let t = TailBatch {
        block_table: DevicePtr::NULL,
        seq_lens: DevicePtr::NULL,
        max_blocks_per_seq: 1,
        n_rows: 3,
        rows_per_seq: 2,
        n_new: 2,
    };
    assert!(
        sk.update_tail(&gpu, &layer(KvCacheDtype::Bf16, DevicePtr::NULL), &t, 0)
            .is_err()
    );
    assert_eq!(
        sk.update_jobs(&gpu, &far, &[], DevicePtr::NULL, 0, 0)
            .unwrap(),
        0
    );
    assert!(gpu.launches_snapshot().is_empty());
    assert_eq!(sk.invalidate(&gpu, &[1, 2], 0).unwrap(), 1);
}

#[test]
fn nothing_is_allocated_where_the_module_is_absent() {
    let gpu = MockGpuBackend::new();
    gpu.mark_module_absent(KERNEL_MODULE);
    let err = SketchBuffers::new(&gpu, cfg_4b(8)).err().expect("refused");
    assert!(err.to_string().contains(KERNEL_MODULE), "{err}");
    assert_eq!(gpu.live_alloc_count(), 0);
    assert!(gpu.kernel_lookups_snapshot().is_empty());
    assert!(gpu.launches_snapshot().is_empty());
}

trait BufferArg {
    fn buffer(self) -> u64;
}

impl BufferArg for MockArg {
    fn buffer(self) -> u64 {
        match self {
            MockArg::Buffer(p) => p.0,
            MockArg::Bytes(_) => panic!("not a buffer argument"),
        }
    }
}
