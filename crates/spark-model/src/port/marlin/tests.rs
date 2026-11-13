// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU tests of the Marlin configuration rules (`config.rs`) and the instance table
//! (`kernels.rs`): cache sizes worked by hand, selection order, the M split, the decode policy and
//! its planted contrast (vLLM's selector changes the configuration with M), and the kernel lookups.
#![allow(clippy::useless_vec)]

use std::collections::HashSet;

use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::MarlinType;
use super::config::*;
use super::kernels::{self, KernelKey, MarlinKernels, entry_name, instances};

fn any(_t: usize, _m8: bool, _c: &ThreadConfig) -> bool {
    true
}

/// The availability of the pruned Ada instance table for one type at g128.
fn table(btype: MarlinType) -> impl Fn(usize, bool, &ThreadConfig) -> bool {
    let keys: HashSet<KernelKey> = instances()
        .into_iter()
        .filter(|k| k.btype == btype)
        .collect();
    move |t, m8, c| keys.contains(&KernelKey::of(btype, Some(8), t, m8, c))
}

const FFN_UP: (usize, usize) = (9216, 2560);
const FFN_DOWN: (usize, usize) = (2560, 9216);
const SMS: usize = 34;

#[test]
fn kernel_cache_sizes_match_the_hand_computation() {
    let p = Problem::of(FFN_UP.0, FFN_UP.1, 128, false);
    // tb 128x128, tb_m 16: tmp = max(32768, 4352, 4352 + 256) = 32768; A = 16384; scales = 1024.
    assert_eq!(
        get_kernel_cache_size(&SMALL_BATCH_THREAD_CONFIGS[0], 1, &p, 4),
        50_176
    );
    // tb 64x256, tb_m 64: tmp = 33792; A = 32768; scales = 2048.
    assert_eq!(
        get_kernel_cache_size(&LARGE_BATCH_THREAD_CONFIGS[0], 4, &p, 4),
        68_608
    );
    // tb 128x64, tb_m 64 (the decode configuration at its widest tile): 16384 + 65536 + 512.
    assert_eq!(get_kernel_cache_size(&UNDERFILL_CONFIG, 4, &p, 4), 82_432);
    // tb 64x128, tb_m 64: 17408 + 32768 + 1024.
    assert_eq!(get_kernel_cache_size(&DECODE_CONFIGS[1], 4, &p, 4), 51_200);
    // Zero points add a quarter of the scale cache at 4 bits.
    let pz = Problem::of(FFN_UP.0, FFN_UP.1, 128, true);
    assert_eq!(
        get_kernel_cache_size(&SMALL_BATCH_THREAD_CONFIGS[0], 1, &pz, 4),
        50_176 + 256
    );
    assert_eq!(
        get_scales_cache_size(&LARGE_BATCH_THREAD_CONFIGS[0], &p, 4),
        2_048
    );
    // One group: one scale row per tile whatever thread_k is.
    let pc = Problem::of(256, 128, 128, false);
    assert_eq!(pc.group_size, None);
    assert_eq!(pc.group_blocks(), None);
    assert_eq!(
        get_scales_cache_size(&SMALL_BATCH_THREAD_CONFIGS[0], &pc, 4),
        1_024
    );
    assert_eq!(p.group_blocks(), Some(8));
}

#[test]
fn validity_follows_divisibility_minimums_and_shared_memory() {
    let p = Problem::of(FFN_UP.0, FFN_UP.1, 128, false);
    for th in SMALL_BATCH_THREAD_CONFIGS
        .iter()
        .chain(&LARGE_BATCH_THREAD_CONFIGS)
    {
        assert!(
            is_valid_config(th, 1, &p, 4, ADA_MAX_SHARED_MEM_OPTIN),
            "{th:?}"
        );
    }
    // N = 192 is not a multiple of 128: only the 64-wide tile divides it.
    let narrow = Problem::of(192, 256, 128, false);
    assert!(!is_valid_config(
        &SMALL_BATCH_THREAD_CONFIGS[0],
        1,
        &narrow,
        4,
        ADA_MAX_SHARED_MEM_OPTIN
    ));
    assert!(is_valid_config(
        &UNDERFILL_CONFIG,
        1,
        &narrow,
        4,
        ADA_MAX_SHARED_MEM_OPTIN
    ));
    // Below the minimum thread count or tile width.
    let small = ThreadConfig {
        thread_k: 64,
        thread_n: 64,
        num_threads: 64,
    };
    assert!(!is_valid_config(&small, 1, &p, 4, ADA_MAX_SHARED_MEM_OPTIN));
    let thin = ThreadConfig {
        thread_k: 32,
        thread_n: 128,
        num_threads: 128,
    };
    assert!(!is_valid_config(&thin, 1, &p, 4, ADA_MAX_SHARED_MEM_OPTIN));
    // Shared memory: the 82,432-byte tile fits 101,376 but not 80 KiB.
    assert!(is_valid_config(
        &UNDERFILL_CONFIG,
        4,
        &p,
        4,
        ADA_MAX_SHARED_MEM_OPTIN
    ));
    assert!(!is_valid_config(&UNDERFILL_CONFIG, 4, &p, 4, 80 * 1024));
}

#[test]
fn selection_takes_the_first_fitting_instantiated_config() {
    let p = Problem::of(FFN_UP.0, FFN_UP.1, 128, false);
    let small = determine_exec_config(&p, 1, true, 4, ADA_MAX_SHARED_MEM_OPTIN, &any);
    assert_eq!(small, Some(SMALL_BATCH_THREAD_CONFIGS[0]));
    let large = determine_exec_config(&p, 2, false, 4, ADA_MAX_SHARED_MEM_OPTIN, &any);
    assert_eq!(large, Some(LARGE_BATCH_THREAD_CONFIGS[0]));
    // An instance missing from the build is skipped, as `MarlinDefault` is upstream.
    let no256 = |_t: usize, _m8: bool, c: &ThreadConfig| c.num_threads != 256;
    let p2 = determine_exec_config(&p, 1, false, 4, ADA_MAX_SHARED_MEM_OPTIN, &no256);
    assert_eq!(p2, Some(SMALL_BATCH_THREAD_CONFIGS[1]));
    let narrow = Problem::of(192, 256, 128, false);
    let n2 = determine_exec_config(&narrow, 1, false, 4, ADA_MAX_SHARED_MEM_OPTIN, &any);
    assert_eq!(n2, Some(UNDERFILL_CONFIG));
    let none = |_t: usize, _m8: bool, _c: &ThreadConfig| false;
    assert_eq!(
        determine_exec_config(&p, 1, false, 4, ADA_MAX_SHARED_MEM_OPTIN, &none),
        None
    );
}

#[test]
fn the_m_split_matches_marlin_mm() {
    let p = Problem::of(FFN_UP.0, FFN_UP.1, 128, false);
    let one = plan_launches(&p, 1, SMS, ADA_MAX_SHARED_MEM_OPTIN, &any).unwrap();
    assert_eq!(
        one,
        [Chunk {
            row_offset: 0,
            rows: 1,
            thread_m_blocks: 1,
            m_block_size_8: true,
            cfg: SMALL_BATCH_THREAD_CONFIGS[0],
            blocks: SMS,
            smem: ADA_MAX_SHARED_MEM_OPTIN,
        }]
    );
    // N > 4096: at most 16 64-row slices a launch; 200 = 192 + 8.
    let c = plan_launches(&p, 200, SMS, ADA_MAX_SHARED_MEM_OPTIN, &any).unwrap();
    let shape: Vec<_> = c
        .iter()
        .map(|c| (c.row_offset, c.rows, c.thread_m_blocks, c.m_block_size_8))
        .collect();
    assert_eq!(shape, [(0, 192, 4, false), (192, 8, 1, true)]);
    assert_eq!(c[0].cfg, LARGE_BATCH_THREAD_CONFIGS[0]);
    // N <= 4096: up to 128 slices; 10000 = 8192 + 1792 + 16.
    let d = Problem::of(FFN_DOWN.0, FFN_DOWN.1, 128, false);
    let c = plan_launches(&d, 10_000, SMS, ADA_MAX_SHARED_MEM_OPTIN, &any).unwrap();
    let rows: Vec<_> = c.iter().map(|c| (c.row_offset, c.rows)).collect();
    assert_eq!(rows, [(0, 8192), (8192, 1792), (9984, 16)]);
    assert!(!c[2].m_block_size_8 && c[2].thread_m_blocks == 1);
}

#[test]
fn the_m_split_uses_the_m48_tile_and_steps_down_without_it() {
    // marlin.cu:307 asks for min(ceil(40 / 16), 4) = 3 blocks, which L18a instantiates (m48).
    let p = Problem::of(FFN_UP.0, FFN_UP.1, 128, false);
    let avail = table(MarlinType::U4B8);
    let c = plan_launches(&p, 40, SMS, ADA_MAX_SHARED_MEM_OPTIN, &avail).unwrap();
    let shape: Vec<_> = c
        .iter()
        .map(|c| (c.row_offset, c.rows, c.thread_m_blocks, c.m_block_size_8))
        .collect();
    assert_eq!(shape, [(0, 40, 3, false)]);
    assert_eq!(c[0].cfg, LARGE_BATCH_THREAD_CONFIGS[0]);
    // A build without thread_m_blocks 3: 40 rows become 32 (two blocks) + 8, as marlin_mm's
    // `max_thread_m_blocks--` loop does.
    let no3 = |t: usize, m8: bool, cfg: &ThreadConfig| t != 3 && avail(t, m8, cfg);
    let c = plan_launches(&p, 40, SMS, ADA_MAX_SHARED_MEM_OPTIN, &no3).unwrap();
    let shape: Vec<_> = c
        .iter()
        .map(|c| (c.row_offset, c.rows, c.thread_m_blocks, c.m_block_size_8))
        .collect();
    assert_eq!(shape, [(0, 32, 2, false), (32, 8, 1, true)]);
}

#[test]
fn an_underfilled_grid_switches_to_the_narrow_tile() {
    // 1024 / 128 * 1 * 4 = 32 tiles <= 34 SMs.
    let p = Problem::of(1024, 2560, 128, false);
    let c = plan_launches(&p, 1, SMS, ADA_MAX_SHARED_MEM_OPTIN, &any).unwrap();
    assert_eq!(c[0].cfg, UNDERFILL_CONFIG);
    // With more SMs than tiles the same holds; with fewer it does not.
    let c = plan_launches(&p, 1, 16, ADA_MAX_SHARED_MEM_OPTIN, &any).unwrap();
    assert_eq!(c[0].cfg, SMALL_BATCH_THREAD_CONFIGS[0]);
}

#[test]
fn every_plan_covers_m_with_contiguous_chunks() {
    let avail = table(MarlinType::U4);
    for (n, k) in [FFN_UP, FFN_DOWN, (1024, 2560), (256, 512)] {
        let p = Problem::of(n, k, 128, true);
        for m in 1..=300 {
            let c = plan_launches(&p, m, SMS, ADA_MAX_SHARED_MEM_OPTIN, &avail).unwrap();
            let mut next = 0;
            for ch in &c {
                assert_eq!(ch.row_offset, next, "[{n}x{k}] m={m}");
                let tile = if ch.m_block_size_8 {
                    8
                } else {
                    ch.thread_m_blocks * 16
                };
                assert!(
                    ch.rows <= tile || ch.rows.is_multiple_of(tile),
                    "[{n}x{k}] m={m} {ch:?}"
                );
                assert!(
                    avail(ch.thread_m_blocks, ch.m_block_size_8, &ch.cfg),
                    "{ch:?}"
                );
                next += ch.rows;
            }
            assert_eq!(next, m, "[{n}x{k}] m={m}");
        }
    }
}

#[test]
fn the_decode_policy_fixes_one_config_for_every_row_count() {
    let avail = table(MarlinType::U4B8);
    for (n, k) in [FFN_UP, FFN_DOWN, (1024, 2560)] {
        let p = Problem::of(n, k, 128, false);
        let cfg =
            invariant_decode_config(&p, &kernels::DECODE_TMBS, ADA_MAX_SHARED_MEM_OPTIN, &avail)
                .expect("a decode config");
        assert_eq!(cfg, UNDERFILL_CONFIG, "[{n}x{k}]");
        for rows in 1..=64 {
            let c = decode_chunk(
                rows,
                cfg,
                &kernels::DECODE_TMBS,
                SMS,
                ADA_MAX_SHARED_MEM_OPTIN,
            )
            .unwrap();
            assert_eq!(c.cfg, cfg);
            assert!(!c.m_block_size_8, "rows={rows}");
            assert!(c.thread_m_blocks * 16 >= rows);
            assert!(avail(c.thread_m_blocks, false, &c.cfg));
            assert_eq!((c.blocks, c.smem), (SMS, ADA_MAX_SHARED_MEM_OPTIN));
        }
        assert!(
            decode_chunk(
                65,
                cfg,
                &kernels::DECODE_TMBS,
                SMS,
                ADA_MAX_SHARED_MEM_OPTIN
            )
            .is_err()
        );
    }
    // K = 192 rules out thread_k 128: the 64 x 128 tile is the one left.
    let p = Problem::of(256, 192, 64, false);
    let cfg = invariant_decode_config(&p, &kernels::DECODE_TMBS, ADA_MAX_SHARED_MEM_OPTIN, &any);
    assert_eq!(cfg, Some(DECODE_CONFIGS[1]));
    // No instance at all: no config.
    let none = |_t: usize, _m8: bool, _c: &ThreadConfig| false;
    assert_eq!(
        invariant_decode_config(&p, &kernels::DECODE_TMBS, ADA_MAX_SHARED_MEM_OPTIN, &none),
        None
    );
}

/// The contrast that justifies the decode policy: vLLM's selector picks a different tile (and the
/// m8 path) at M = 1 than at M = 17, so its row bits would depend on M.
#[test]
fn the_vllm_selector_is_not_row_invariant() {
    let p = Problem::of(FFN_UP.0, FFN_UP.1, 128, false);
    let avail = table(MarlinType::U4B8);
    let a = plan_launches(&p, 1, SMS, ADA_MAX_SHARED_MEM_OPTIN, &avail).unwrap();
    let b = plan_launches(&p, 17, SMS, ADA_MAX_SHARED_MEM_OPTIN, &avail).unwrap();
    assert_ne!(
        (a[0].cfg, a[0].m_block_size_8),
        (b[0].cfg, b[0].m_block_size_8)
    );
}

#[test]
fn decode_tiles_and_buffers() {
    let t = &kernels::DECODE_TMBS;
    assert_eq!(decode_thread_m_blocks(1, t), Some(1));
    assert_eq!(decode_thread_m_blocks(16, t), Some(1));
    assert_eq!(decode_thread_m_blocks(17, t), Some(2));
    assert_eq!(decode_thread_m_blocks(32, t), Some(2));
    assert_eq!(decode_thread_m_blocks(33, t), Some(3));
    assert_eq!(decode_thread_m_blocks(48, t), Some(3));
    assert_eq!(decode_thread_m_blocks(49, t), Some(4));
    assert_eq!(decode_thread_m_blocks(64, t), Some(4));
    assert_eq!(decode_thread_m_blocks(65, t), None);
    assert_eq!(c_tmp_bytes(1, SMS), 35 * 16 * 256 * 4);
    assert_eq!(c_tmp_bytes(17, SMS), 35 * 32 * 256 * 4);
    assert_eq!(c_tmp_bytes(40, SMS), 35 * 48 * 256 * 4);
    assert_eq!(c_tmp_bytes(64, SMS), 35 * 64 * 256 * 4);
    assert_eq!(c_tmp_bytes(4096, SMS), c_tmp_bytes(64, SMS));
    assert_eq!(locks_bytes(SMS), SMS * 4 * 4);
}

#[test]
fn supported_shapes() {
    assert!(supports_shape(FFN_UP.0, FFN_UP.1, 128).is_ok());
    assert!(supports_shape(FFN_DOWN.0, FFN_DOWN.1, 128).is_ok());
    assert!(supports_shape(248_320, 2560, 128).is_ok());
    assert!(supports_shape(256, 192, 64).is_ok());
    assert!(supports_shape(100, 2560, 128).is_err());
    assert!(supports_shape(192, 192, 64).is_err());
    assert!(supports_shape(256, 2560, 100).is_err());
    assert!(supports_shape(0, 2560, 128).is_err());
}

#[test]
fn the_instance_table_is_the_pruned_generator_set() {
    let all = instances();
    assert_eq!(all.len(), kernels::ENTRY_COUNT);
    assert_eq!(all.len(), 30);
    let names: HashSet<String> = all.iter().map(entry_name).collect();
    assert_eq!(names.len(), all.len(), "entry names are unique");
    for k in &all {
        if k.threads == 256 {
            let expect = if k.thread_m_blocks <= 1 {
                (8, 8)
            } else {
                (4, 16)
            };
            assert_eq!((k.thread_k_blocks, k.thread_n_blocks), expect, "{k:?}");
        }
        assert!(!k.m_block_size_8 || k.thread_m_blocks == 1);
        assert_eq!(k.group_blocks, Some(8));
    }
    let key = KernelKey::of(
        MarlinType::U4B8,
        Some(8),
        1,
        true,
        &SMALL_BATCH_THREAD_CONFIGS[0],
    );
    assert_eq!(
        entry_name(&key),
        "marlin_ada_bf16_u4b8_g128_m8_k128_n128_t256"
    );
    let key = KernelKey::of(
        MarlinType::U4,
        Some(8),
        4,
        false,
        &LARGE_BATCH_THREAD_CONFIGS[0],
    );
    assert_eq!(
        entry_name(&key),
        "marlin_ada_bf16_u4_g128_m64_k64_n256_t256"
    );
    let key = KernelKey::of(MarlinType::U4B8, Some(8), 3, false, &UNDERFILL_CONFIG);
    assert_eq!(
        entry_name(&key),
        "marlin_ada_bf16_u4b8_g128_m48_k128_n64_t128"
    );
    // Every decode tile of both decode configurations is in the table.
    for b in kernels::TYPES {
        for &t in &kernels::DECODE_TMBS {
            for cfg in &DECODE_CONFIGS {
                assert!(
                    all.contains(&KernelKey::of(b, Some(8), t, false, cfg)),
                    "{b:?} {t} {cfg:?}"
                );
            }
        }
    }
    assert!(kernels::group_supported(128, 2560));
    assert!(!kernels::group_supported(64, 2560));
    assert!(
        !kernels::group_supported(128, 128),
        "one group needs the channelwise kernel"
    );
}

#[test]
fn resolution_looks_up_exactly_the_family_and_nothing_without_the_module() {
    let gpu = MockGpuBackend::new();
    let k = MarlinKernels::resolve(&gpu, MarlinType::U4, Some(8)).unwrap();
    let looked = gpu.kernel_lookups_snapshot();
    assert_eq!(looked.len(), 15);
    assert!(
        looked
            .iter()
            .all(|(m, f)| m == kernels::MODULE && f.starts_with("marlin_ada_bf16_u4_g128_"))
    );
    assert!(k.has(4, false, &UNDERFILL_CONFIG));
    assert!(k.has(3, false, &UNDERFILL_CONFIG));
    assert!(
        !k.has(2, true, &UNDERFILL_CONFIG),
        "the 8-row tile exists for one M block only"
    );

    let absent = MockGpuBackend::new();
    absent.mark_module_absent(kernels::MODULE);
    assert!(!super::available(&absent));
    assert!(MarlinKernels::resolve(&absent, MarlinType::U4B8, Some(8)).is_err());
    assert!(absent.kernel_lookups_snapshot().is_empty());

    let partial = MockGpuBackend::new();
    let key = KernelKey::of(MarlinType::U4B8, Some(8), 2, false, &UNDERFILL_CONFIG);
    partial.deny_kernel(kernels::MODULE, &entry_name(&key));
    let e = MarlinKernels::resolve(&partial, MarlinType::U4B8, Some(8))
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("marlin_ada_bf16_u4b8_g128_m32_k128_n64_t128"),
        "{e}"
    );
    assert!(partial.has_module(kernels::MODULE));
}

/// One `MARLIN_ADA_ENTRY(name, weight, threads, m_blocks, n_blocks, k_blocks, m8)` line of
/// `marlin_ada.cu`, as a key.
fn parse_entry(line: &str) -> Option<(String, KernelKey)> {
    let body = line
        .trim()
        .strip_prefix("MARLIN_ADA_ENTRY(")?
        .strip_suffix(')')?;
    let f: Vec<&str> = body.split(',').map(str::trim).collect();
    if f.len() != 7 {
        return None;
    }
    let btype = match f[1] {
        "vllm::kU4B8Id" => MarlinType::U4B8,
        "vllm::kU4Id" => MarlinType::U4,
        _ => return None,
    };
    let key = KernelKey {
        btype,
        group_blocks: Some(8),
        threads: f[2].parse().ok()?,
        thread_m_blocks: f[3].parse().ok()?,
        thread_n_blocks: f[4].parse().ok()?,
        thread_k_blocks: f[5].parse().ok()?,
        m_block_size_8: f[6] == "true",
    };
    Some((f[0].to_string(), key))
}

/// The cross-lane oracle: the instance table and its names are exactly lane L18a's export list,
/// read from `kernels/ada/common/marlin_ada.cu` (the name, and the template arguments behind it).
#[test]
fn the_instance_table_is_marlin_ada_cu() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/ada/common/marlin_ada.cu");
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let exported: Vec<(String, KernelKey)> = src
        .lines()
        .filter(|l| l.starts_with("MARLIN_ADA_ENTRY("))
        .map(|l| parse_entry(l).unwrap_or_else(|| panic!("unparsed entry line: {l}")))
        .collect();
    assert_eq!(exported.len(), kernels::ENTRY_COUNT);
    for (name, key) in &exported {
        assert_eq!(
            &entry_name(key),
            name,
            "the name encodes the template arguments"
        );
    }
    let ours: HashSet<KernelKey> = instances().into_iter().collect();
    let theirs: HashSet<KernelKey> = exported.iter().map(|(_, k)| *k).collect();
    assert_eq!(ours, theirs);
    // Planted control: a key the build does not carry (the 8-row tile at two M blocks) parses and is
    // absent, and its name differs from the line's name.
    let bent = parse_entry(
        "MARLIN_ADA_ENTRY(marlin_ada_bf16_u4_g128_m32_k64_n128_t128, vllm::kU4Id, 128, 2, 8, 4, true)",
    )
    .unwrap();
    assert!(!theirs.contains(&bent.1), "the m8 flag is part of the key");
    assert_ne!(entry_name(&bent.1), bent.0);
}
