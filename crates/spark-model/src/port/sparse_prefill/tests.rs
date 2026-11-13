// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU tests of the block-sparse prefill host side: knob resolution and refusals, the
//! selection rule (kernel procedure against the definition), launch geometry, scratch sizes, the
//! eligibility verdicts, the entry table and the `/metrics` text.

use std::collections::HashMap;

use spark_runtime::gpu::DevicePtr;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::config::{Config, Knobs, resolve};
use super::launch::{self, Call, Decline, Geometry, KvKind, eligibility, entry_names};
use super::select::{INF_BITS, cut_bits, forced, mismatches, select_f32, select_greedy};
use super::{
    BLOCK, DEFAULT_MIN_TOKENS, DEFAULT_STRIDE, DEFAULT_THRESHOLD, ENV_MIN_TOKENS, ENV_STRIDE,
    ENV_THRESHOLD,
};

fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
    let m: HashMap<String, String> = pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    move |k| m.get(k).cloned()
}

fn on(threshold: f32) -> Config {
    Config {
        enabled: true,
        min_tokens: 1024,
        threshold,
        stride: 8,
    }
}

fn call() -> Call {
    Call {
        q: DevicePtr(0x1000),
        out: DevicePtr(0x2000),
        k_pool: DevicePtr(0x3000),
        v_pool: DevicePtr(0x4000),
        block_table: DevicePtr(0x5000),
        q_len: 2048,
        kv_len: 34_816,
        q_offset: 32_768,
        nq: 16,
        nkv: 4,
        hd: 256,
        block_size: 16,
        cache_stride: 16 * 4 * 256,
        kv: KvKind::Bf16,
        k_scale: 1.0,
        v_scale: 1.0,
        inv_sqrt_d: 1.0 / 16.0,
        sliding_window: 0,
        graph_capture: false,
        seam_active: false,
        layer: 3,
    }
}

#[test]
fn defaults_flags_and_environment_resolve_in_order() {
    let c = resolve(false, &Knobs::default(), env(&[])).unwrap();
    assert_eq!(c, Config::OFF);
    assert_eq!(
        (c.min_tokens, c.threshold, c.stride),
        (DEFAULT_MIN_TOKENS, DEFAULT_THRESHOLD, DEFAULT_STRIDE)
    );
    let e = env(&[
        (ENV_MIN_TOKENS, "65536"),
        (ENV_THRESHOLD, "0.95"),
        (ENV_STRIDE, "16"),
    ]);
    let c = resolve(true, &Knobs::default(), &e).unwrap();
    assert_eq!(
        (c.enabled, c.min_tokens, c.threshold, c.stride),
        (true, 65_536, 0.95, 16)
    );
    let cli = Knobs {
        min_tokens: Some(4096),
        threshold: Some(0.8),
        stride: Some(4),
    };
    let c = resolve(true, &cli, &e).unwrap();
    assert_eq!(
        (c.min_tokens, c.threshold, c.stride),
        (4096, 0.8, 4),
        "the flag wins over the environment"
    );
    let blank = env(&[(ENV_STRIDE, "  ")]);
    assert_eq!(
        resolve(false, &Knobs::default(), blank).unwrap().stride,
        DEFAULT_STRIDE
    );
}

#[test]
fn out_of_range_and_unparsable_values_are_startup_errors() {
    let bad_cli = [
        Knobs {
            min_tokens: Some(0),
            ..Knobs::default()
        },
        Knobs {
            threshold: Some(0.0),
            ..Knobs::default()
        },
        Knobs {
            threshold: Some(1.5),
            ..Knobs::default()
        },
        Knobs {
            threshold: Some(f32::NAN),
            ..Knobs::default()
        },
        Knobs {
            stride: Some(5),
            ..Knobs::default()
        },
        Knobs {
            stride: Some(32),
            ..Knobs::default()
        },
    ];
    for k in bad_cli {
        assert!(resolve(false, &k, env(&[])).is_err(), "{k:?}");
    }
    for (name, v) in [
        (ENV_MIN_TOKENS, "lots"),
        (ENV_THRESHOLD, "0.9x"),
        (ENV_STRIDE, "-8"),
        (ENV_STRIDE, "2"),
    ] {
        let err = resolve(false, &Knobs::default(), env(&[(name, v)])).unwrap_err();
        assert!(
            format!("{err:#}").contains("PREFILL_SPARSE") || format!("{err:#}").contains("stride"),
            "{err:#}"
        );
    }
    // Threshold 1.0 is legal: it is the identity gate.
    let k = Knobs {
        threshold: Some(1.0),
        ..Knobs::default()
    };
    assert!(resolve(true, &k, env(&[])).unwrap().keeps_all());
}

#[test]
fn the_kernel_procedure_matches_the_definition_on_exact_masses() {
    // Dyadic masses (multiples of 1/1024) sum exactly in FP32, so the bit search and the sorted greedy
    // must agree block for block, ties included.
    let mut seed = 0x5EED_u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    for case in 0..400 {
        let n = 1 + (next() % 60) as usize;
        let mut units: Vec<u32> = (0..n).map(|_| next() % 40).collect();
        if case % 7 == 0 {
            units.iter_mut().for_each(|u| *u = 3); // all ties
        }
        let total: u32 = units.iter().sum::<u32>().max(1);
        // Normalize to a sum of 1024 units so the masses sum to exactly 1.
        let mut scaled: Vec<u32> = units.iter().map(|u| u * 1024 / total).collect();
        let short = 1024 - scaled.iter().sum::<u32>();
        scaled[0] += short;
        let m32: Vec<f32> = scaled.iter().map(|u| *u as f32 / 1024.0).collect();
        let m64: Vec<f64> = m32.iter().map(|x| *x as f64).collect();
        for thr in [0.25f32, 0.5, 0.75, 0.9, 0.984375] {
            assert_eq!(
                select_f32(&m32, thr),
                select_greedy(&m64, thr as f64),
                "case {case} n {n} thr {thr} {scaled:?}"
            );
        }
    }
}

#[test]
fn forced_blocks_threshold_one_and_the_cut_extremes() {
    let jd = 9;
    let kept: Vec<usize> = (0..=jd).filter(|&j| forced(j, jd)).collect();
    assert_eq!(kept, vec![0, 8, 9]);
    assert!(forced(0, 0) && forced(0, 1) && forced(1, 1));
    let mass = vec![0.5f32, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.13, 0.30];
    // Forced already hold 0.93: nothing else at 0.9, the cut is +inf.
    assert_eq!(cut_bits(&mass, 0.9), INF_BITS);
    assert_eq!(select_f32(&mass, 0.9).iter().filter(|x| **x).count(), 3);
    assert!(select_f32(&mass, 1.0).iter().all(|x| *x));
    // A shortfall (the non-forced blocks cannot reach the threshold, as FP32 rounding can make happen
    // just below 1.0) keeps everything: cut 0.
    let short = vec![0.05f32; 10];
    assert_eq!(cut_bits(&short, 0.999_999), 0);
    assert!(select_f32(&short, 0.999_999).iter().all(|x| *x));
}

#[test]
fn a_higher_threshold_keeps_a_superset() {
    let mass: Vec<f64> = (0..50)
        .map(|j| ((j * 37 % 11) as f64 + 0.5) / 300.0)
        .collect();
    let s: f64 = mass.iter().sum();
    let mass: Vec<f64> = mass.iter().map(|m| m / s).collect();
    let mut prev = select_greedy(&mass, 0.1);
    for t in [0.3, 0.5, 0.7, 0.9, 0.95, 0.99] {
        let cur = select_greedy(&mass, t);
        assert!(
            prev.iter().zip(&cur).all(|(p, c)| !*p || *c),
            "threshold {t}"
        );
        let kept: f64 = (0..mass.len()).filter(|&j| cur[j]).map(|j| mass[j]).sum();
        assert!(kept >= t - 1e-12, "threshold {t} kept {kept}");
        prev = cur;
    }
}

#[test]
fn mismatches_only_count_blocks_at_the_boundary_as_benign() {
    // Forced: 0, 6, 7. The reference keeps 1 and 2: last kept 0.2, next candidate 0.1.
    let mass = vec![0.3, 0.25, 0.2, 0.1, 0.05, 0.02, 0.03, 0.05];
    let a = vec![true, true, true, false, false, false, true, true];
    let mut at_edge = a.clone();
    at_edge[3] = true;
    assert_eq!(mismatches(&a, &at_edge, &mass, 0.01), (1, 0));
    let mut far = a.clone();
    far[5] = true; // 0.02: far below the boundary
    assert_eq!(mismatches(&a, &far, &mass, 0.05), (1, 1));
    let mut dropped = a.clone();
    dropped[1] = false; // 0.25: far above the boundary
    assert_eq!(mismatches(&a, &dropped, &mass, 0.03), (1, 1));
    assert_eq!(mismatches(&a, &a, &mass, 0.0), (0, 0));
}

#[test]
fn geometry_of_aligned_and_unaligned_chunks() {
    let c = call();
    let g = Geometry::new(&c);
    assert_eq!(g.q_block_lo, 32_768 / BLOCK);
    assert_eq!(g.n_qb, 16);
    assert_eq!(g.nkb, 34_816 / BLOCK);
    assert_eq!(g.mask_words, (34_816 / BLOCK).div_ceil(32));
    assert_eq!(g.items, 16 * 16);
    assert_eq!(g.n_tiles, 32);
    // A chunk that starts and ends mid-block (prefix-cache hits land on KV-block boundaries).
    let mut u = call();
    u.q_offset = 33_000;
    u.q_len = 1000;
    u.kv_len = 34_000;
    let g = Geometry::new(&u);
    assert_eq!(g.q_block_lo, 33_000 / 128);
    assert_eq!(g.n_qb, (33_999 / 128) - (33_000 / 128) + 1);
    let tile0: u32 = 33_000 / 64 * 64;
    assert_eq!(g.n_tiles, (34_000 - tile0).div_ceil(64));
    assert_eq!(g.nkb, 34_000u32.div_ceil(128));
    // Scratch: one (R + 1) x nkb FP32 slot per scoring CTA.
    let grid = g.score_grid(34, 256);
    assert_eq!(grid, 34u32.min(g.items));
    assert_eq!(
        g.score_scratch_bytes(grid, 8),
        grid as usize * 17 * g.nkb as usize * 4
    );
    assert_eq!(g.mask_bytes(), g.items as usize * g.mask_words as usize * 4);
    assert_eq!(Geometry::new(&call()).score_grid(34, 128), 68);
}

#[test]
fn shared_memory_mirrors_the_kernel_tiles_and_fits_the_ada_opt_in() {
    const OPTIN: u32 = 101_376;
    assert_eq!(launch::score_smem(256), 128 * 264 * 2 + 32 * 4 + 8 * 4 + 8);
    assert_eq!(launch::attn_smem(256), (64 + 64) * 264 * 2);
    assert_eq!(launch::attn_smem(128), (64 + 128) * 136 * 2);
    for hd in super::HEAD_DIMS {
        assert!(
            launch::score_smem(hd) <= OPTIN && launch::attn_smem(hd) <= OPTIN,
            "hd {hd}"
        );
    }
}

#[test]
fn every_decline_names_its_guard_and_the_row_off_check_comes_first() {
    let c = call();
    assert_eq!(eligibility(&Config::OFF, &c), Err(Decline::Off));
    let mut weird = c;
    weird.kv = KvKind::Other;
    weird.graph_capture = true;
    assert_eq!(
        eligibility(&Config::OFF, &weird),
        Err(Decline::Off),
        "row off is decided first"
    );
    assert_eq!(eligibility(&on(1.0), &c), Err(Decline::KeepAll));
    assert_eq!(eligibility(&on(0.9), &c), Ok(()));
    let mut cases: Vec<(Call, Decline)> = Vec::new();
    let mut x = call();
    x.kv_len = 1000;
    cases.push((x, Decline::Empty));
    let mut x = call();
    x.q_len = 0;
    cases.push((x, Decline::Empty));
    let mut x = call();
    (x.q_offset, x.q_len, x.kv_len) = (0, 512, 512);
    cases.push((x, Decline::Short));
    let mut x = call();
    x.graph_capture = true;
    cases.push((x, Decline::Capture));
    let mut x = call();
    x.seam_active = true;
    cases.push((x, Decline::Seam));
    let mut x = call();
    x.kv = KvKind::Other;
    cases.push((x, Decline::KvDtype));
    let mut x = call();
    x.hd = 512;
    cases.push((x, Decline::HeadDim));
    let mut x = call();
    x.sliding_window = 4096;
    cases.push((x, Decline::SlidingWindow));
    let mut x = call();
    x.nkv = 3;
    cases.push((x, Decline::Gqa));
    for (x, want) in cases {
        assert_eq!(eligibility(&on(0.9), &x), Err(want), "{x:?}");
    }
    let mut f = call();
    f.kv = KvKind::Fp8;
    assert_eq!(eligibility(&on(0.9), &f), Ok(()));
}

#[test]
fn the_entry_table_covers_every_compiled_combination_once() {
    let mut seen = std::collections::HashSet::new();
    for kv in [KvKind::Bf16, KvKind::Fp8] {
        for hd in super::HEAD_DIMS {
            for s in super::STRIDES {
                let (score, attn) = entry_names(kv, hd, s).expect("compiled");
                let tag = if kv == KvKind::Fp8 { "fp8" } else { "bf16" };
                assert_eq!(score, format!("prefill_sparse_score_ada_{tag}_hd{hd}_s{s}"));
                assert_eq!(attn, format!("prefill_sparse_attn_ada_{tag}_hd{hd}"));
                assert!(seen.insert(score));
            }
        }
    }
    assert_eq!(seen.len(), 12);
    assert!(entry_names(KvKind::Other, 256, 8).is_none());
    assert!(entry_names(KvKind::Bf16, 64, 8).is_none());
    assert!(entry_names(KvKind::Bf16, 256, 2).is_none());
}

#[test]
fn run_launches_score_then_attention_through_the_op_cache() {
    let gpu = MockGpuBackend::new();
    // Row off: no lookup, no launch, no allocation.
    assert!(!launch::run(&gpu, &Config::OFF, &call(), 7).unwrap());
    assert!(gpu.kernel_lookups_snapshot().is_empty() && gpu.launches_snapshot().is_empty());
    // Below the minimum length: the same.
    let mut short = call();
    (short.q_offset, short.q_len, short.kv_len) = (0, 512, 512);
    assert!(!launch::run(&gpu, &on(0.9), &short, 7).unwrap());
    assert!(gpu.kernel_lookups_snapshot().is_empty());

    assert!(launch::run(&gpu, &on(0.9), &call(), 7).unwrap());
    let lookups = gpu.kernel_lookups_snapshot();
    let want = vec![
        (
            launch::MODULE.to_string(),
            "prefill_sparse_score_ada_bf16_hd256_s8".to_string(),
        ),
        (
            launch::MODULE.to_string(),
            "prefill_sparse_attn_ada_bf16_hd256".to_string(),
        ),
    ];
    assert_eq!(lookups, want);
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 2, "score then attention");
    let g = Geometry::new(&call());
    assert_eq!(
        l[0].grid,
        [g.score_grid(48, 256), 1, 1],
        "mock sm_count is 48"
    );
    assert_eq!(l[0].block, [launch::SCORE_THREADS, 1, 1]);
    assert_eq!(l[0].shared_mem, launch::score_smem(256));
    assert_eq!(l[1].grid, [g.n_tiles, 16, 1]);
    assert_eq!(l[1].block, [launch::ATTN_THREADS, 1, 1]);
    assert_eq!(l[1].shared_mem, launch::attn_smem(256));
    assert!(l.iter().all(|x| x.stream == 7));
    // The handles are memoized on the backend: a second chunk looks nothing up.
    assert!(launch::run(&gpu, &on(0.9), &call(), 7).unwrap());
    assert_eq!(gpu.kernel_lookups_snapshot().len(), 2);
    assert_eq!(gpu.launches_snapshot().len(), 4);
}

#[test]
fn an_absent_module_declines_without_a_lookup() {
    let gpu = MockGpuBackend::new();
    gpu.mark_module_absent(launch::MODULE);
    assert!(!launch::run(&gpu, &on(0.9), &call(), 0).unwrap());
    assert!(gpu.kernel_lookups_snapshot().is_empty());
    assert!(gpu.launches_snapshot().is_empty());
}

#[test]
fn the_metrics_text_has_both_series() {
    let mut s = String::new();
    super::metrics::render_values(&mut s, 7, 1.25, 4);
    assert!(s.contains(
        "# TYPE atlas_prefill_sparse_layers_total counter\natlas_prefill_sparse_layers_total 7\n"
    ));
    assert!(s.contains("# TYPE atlas_prefill_sparse_blocks_kept_ratio summary\n"));
    assert!(s.contains("atlas_prefill_sparse_blocks_kept_ratio_sum 1.25\n"));
    assert!(s.contains("atlas_prefill_sparse_blocks_kept_ratio_count 4\n"));
    let mut z = String::new();
    super::metrics::render_values(&mut z, 0, 0.0, 0);
    assert!(z.contains("atlas_prefill_sparse_layers_total 0\n") && z.contains("_count 0\n"));
}
