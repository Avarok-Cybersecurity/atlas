// SPDX-License-Identifier: AGPL-3.0-only
// Translated for Atlas from vLLM (https://github.com/vllm-project/vllm, commit 8a36f08),
// csrc/quantization/marlin/marlin.cu:69-238 (`small_batch_thread_configs`, `large_batch_thread_configs`,
// `get_scales_cache_size`, `get_kernel_cache_size`, `is_valid_config`, `determine_exec_config`) and
// :240-403 (the M split of `marlin_mm`), with the constants of marlin.cuh:21-31.
// Copyright (C) Marlin.2024 Elias Frantar, Apache-2.0, adapted from https://github.com/IST-DASLab/marlin;
// Copyright contributors to the vLLM project, Apache-2.0.
// Changes: C++ to Rust with unsigned arithmetic (vLLM's -1 sentinels become `Option`); the kernel
// selector (`get_marlin_kernel` returning `MarlinDefault`) becomes an availability callback over the
// pruned Ada instance set; float zero points, the atomic-add reduce and the Turing 2-stage path are
// dropped (not instantiated); the `max_shared_mem` device query is the sm_89 opt-in constant; the
// row-invariant decode policy (`invariant_decode_config`, `decode_thread_m_blocks`) is new.

//! Responsibility: which Marlin kernel instance and launch geometry serve a problem. Pure arithmetic,
//! no GPU: every rule is pinned by the tests in `tests.rs`.
//!
//! Two policies:
//! * [`plan_launches`] is vLLM's `marlin_mm` loop: the M split into chunks of at most
//!   `16 * max_par` 64-row slices and `determine_exec_config` per chunk, whose thread configuration
//!   changes with M. Used by `prefill`.
//! * [`invariant_decode_config`] + [`decode_thread_m_blocks`] is the decode policy: one 128-thread
//!   configuration that is instantiated for every `thread_m_blocks`, fixed per weight, and never the
//!   `m_block_size_8` tile. With one launch of `prob_m <= 64` the kernel's `parallel` is 1 and the K
//!   partition (`iters`, `slice_*`) depends only on the shape, the configuration and the grid
//!   (`marlin_template.h:328-358`), so a row's bits do not depend on the row count.

use anyhow::{Result, bail, ensure};

/// `default_threads` (`marlin.cuh:21`).
pub const DEFAULT_THREADS: usize = 256;
/// `pipe_stages` (`marlin.cuh:23`): 4 stages on sm_80 and newer.
pub const PIPE_STAGES: usize = 4;
/// `min_thread_n` (`marlin.cuh:26`).
pub const MIN_THREAD_N: usize = 64;
/// `min_thread_k` (`marlin.cuh:27`).
pub const MIN_THREAD_K: usize = 64;
/// `max_thread_n` (`marlin.cuh:28`): the widest N tile, which sizes the FP32 reduce buffer.
pub const MAX_THREAD_N: usize = 256;
/// `tile_size` (`marlin.cuh:30`).
pub const TILE_SIZE: usize = 16;
/// `max_par` (`marlin.cuh:31`): 64-row slices one launch takes for a wide N.
pub const MAX_PAR: usize = 16;
/// The largest `thread_m_blocks` (64 rows a CTA).
pub const MAX_THREAD_M_BLOCKS: usize = 4;
/// Lock slots per SM (vLLM's `MARLIN_MAX_BLOCKS_PER_SM`, `marlin_utils.py:30`).
pub const MAX_BLOCKS_PER_SM: usize = 4;
/// `cudaDevAttrMaxSharedMemoryPerBlockOptin` on sm_86 and sm_89 (101,376 bytes, the value
/// `kernels/ada/common/prefill_dyn_smem.cuh` asserts against). Atlas's backend has no device
/// attribute query; this module only serves the Ada build.
pub const ADA_MAX_SHARED_MEM_OPTIN: usize = 101_376;
/// The margin `determine_exec_config` keeps below the opt-in when it ranks configurations.
pub const SELECT_SMEM_MARGIN: usize = 512;

/// One thread-block tile: `thread_k x thread_n` weights, `num_threads` threads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ThreadConfig {
    pub thread_k: usize,
    pub thread_n: usize,
    pub num_threads: usize,
}

const fn tc(thread_k: usize, thread_n: usize, num_threads: usize) -> ThreadConfig {
    ThreadConfig {
        thread_k,
        thread_n,
        num_threads,
    }
}

/// `small_batch_thread_configs` (`marlin.cu:75-81`), in priority order.
pub const SMALL_BATCH_THREAD_CONFIGS: [ThreadConfig; 3] =
    [tc(128, 128, 256), tc(64, 128, 128), tc(128, 64, 128)];
/// `large_batch_thread_configs` (`marlin.cu:83-89`), in priority order.
pub const LARGE_BATCH_THREAD_CONFIGS: [ThreadConfig; 3] =
    [tc(64, 256, 256), tc(64, 128, 128), tc(128, 64, 128)];
/// The configuration `marlin_mm` switches to when the grid would be under-filled (`marlin.cu:331-336`).
pub const UNDERFILL_CONFIG: ThreadConfig = tc(128, 64, 128);
/// The decode policy's candidates, in preference order: the two 128-thread configurations, which the
/// kernel generator instantiates for every `thread_m_blocks` (`generate_kernels.py:153-161`).
pub const DECODE_CONFIGS: [ThreadConfig; 2] = [tc(128, 64, 128), tc(64, 128, 128)];

/// The GEMM a launch computes: `[m, k] x [k, n]` with 4-bit B.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Problem {
    pub n: usize,
    pub k: usize,
    pub num_bits: usize,
    /// `None`: one scale per output column over all of K (vLLM's `group_size == -1`).
    pub group_size: Option<usize>,
    pub has_zp: bool,
}

impl Problem {
    /// The problem of a weight of `groups = k / group` groups (one group is the channelwise kernel).
    pub fn of(n: usize, k: usize, group: usize, has_zp: bool) -> Self {
        Self {
            n,
            k,
            num_bits: 4,
            group_size: if group >= k { None } else { Some(group) },
            has_zp,
        }
    }

    /// The kernel's `group_blocks` (`group_size / 16`, or `None` for one group).
    pub fn group_blocks(&self) -> Option<usize> {
        self.group_size.map(|g| g / 16)
    }
}

/// `get_scales_cache_size` (`marlin.cu:96-112`): shared bytes of the scale pipeline.
pub fn get_scales_cache_size(th: &ThreadConfig, p: &Problem, stages: usize) -> usize {
    let tb_groups = match p.group_size {
        None => 1,
        Some(g) => th.thread_k.div_ceil(g),
    };
    tb_groups * th.thread_n * 2 * stages
}

/// `get_kernel_cache_size` (`marlin.cu:114-147`): shared bytes of one CTA (float zero points are not
/// instantiated, so their branch is gone).
pub fn get_kernel_cache_size(
    th: &ThreadConfig,
    thread_m_blocks: usize,
    p: &Problem,
    stages: usize,
) -> usize {
    let pack_factor = 32 / p.num_bits;
    let tb_k = th.thread_k;
    let tb_n = th.thread_n;
    let tb_m = thread_m_blocks * 16;
    let sh_a_size = stages * (tb_m * tb_k) * 2;
    let sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
    let sh_red_size = tb_m * (tb_n + 8) * 2;
    let sh_bias_size = tb_n * 2;
    let tmp_size = sh_b_size.min(sh_red_size) + sh_bias_size;
    let tmp_size = sh_b_size.max(sh_red_size).max(tmp_size);
    let sh_s_size = get_scales_cache_size(th, p, stages);
    let sh_zp_size = if p.has_zp {
        match p.num_bits {
            4 => sh_s_size / 4,
            8 => sh_s_size / 2,
            _ => 0,
        }
    } else {
        0
    };
    tmp_size + sh_a_size + sh_s_size + sh_zp_size
}

/// `is_valid_config` (`marlin.cu:149-179`).
pub fn is_valid_config(
    th: &ThreadConfig,
    thread_m_blocks: usize,
    p: &Problem,
    stages: usize,
    max_shared_mem: usize,
) -> bool {
    if th.thread_k == 0 || th.thread_n == 0 || th.num_threads == 0 {
        return false;
    }
    if !p.k.is_multiple_of(th.thread_k) || !p.n.is_multiple_of(th.thread_n) {
        return false;
    }
    if th.thread_n < MIN_THREAD_N || th.thread_k < MIN_THREAD_K {
        return false;
    }
    if th.num_threads < 128 {
        return false;
    }
    get_kernel_cache_size(th, thread_m_blocks, p, stages) <= max_shared_mem
}

/// Whether the build carries the instance `(thread_m_blocks, m_block_size_8, config)` for the
/// problem's type and group (the role of `get_marlin_kernel(..) != MarlinDefault`).
pub type Avail<'a> = &'a dyn Fn(usize, bool, &ThreadConfig) -> bool;

/// `determine_exec_config` (`marlin.cu:197-238`): the first configuration of the batch class that
/// fits the shared memory (minus [`SELECT_SMEM_MARGIN`]) and has an instance. `None` is vLLM's
/// `{-1, -1, -1}`.
pub fn determine_exec_config(
    p: &Problem,
    thread_m_blocks: usize,
    m_block_size_8: bool,
    stages: usize,
    max_shared_mem: usize,
    avail: Avail<'_>,
) -> Option<ThreadConfig> {
    let configs = if thread_m_blocks > 1 {
        &LARGE_BATCH_THREAD_CONFIGS
    } else {
        &SMALL_BATCH_THREAD_CONFIGS
    };
    let budget = max_shared_mem.saturating_sub(SELECT_SMEM_MARGIN);
    configs
        .iter()
        .find(|th| {
            is_valid_config(th, thread_m_blocks, p, stages, budget)
                && avail(thread_m_blocks, m_block_size_8, th)
        })
        .copied()
}

/// One kernel launch of a plan: rows `row_offset..row_offset + rows` of the activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub row_offset: usize,
    pub rows: usize,
    pub thread_m_blocks: usize,
    pub m_block_size_8: bool,
    pub cfg: ThreadConfig,
    /// Grid size (`sms * blocks_per_sm`, `blocks_per_sm` is always 1 here).
    pub blocks: usize,
    /// Dynamic shared bytes of the launch, also the kernel's `max_shared_mem` argument.
    pub smem: usize,
}

/// The `marlin_mm` loop (`marlin.cu:293-402`) with automatic configuration: the M split and one
/// [`Chunk`] per launch. Errors where vLLM's `STD_TORCH_CHECK` would.
pub fn plan_launches(
    p: &Problem,
    prob_m: usize,
    sms: usize,
    max_shared_mem: usize,
    avail: Avail<'_>,
) -> Result<Vec<Chunk>> {
    ensure!(
        prob_m > 0 && p.n > 0 && p.k > 0,
        "marlin: invalid MNK = [{prob_m}, {}, {}]",
        p.n,
        p.k
    );
    if let Some(g) = p.group_size {
        ensure!(
            g.is_multiple_of(16) && p.k.is_multiple_of(g),
            "marlin: K={} is not divisible into groups of {g}",
            p.k
        );
    }
    let stages = PIPE_STAGES;
    let max_par = if p.n <= 4096 { MAX_PAR * 8 } else { MAX_PAR };
    let max_shared_mem_new = max_shared_mem;
    let mut rest_m = prob_m;
    let mut row = 0;
    let mut max_tmb = MAX_THREAD_M_BLOCKS;
    let mut out = Vec::new();
    while rest_m > 0 {
        let par_count = (rest_m / (max_tmb * 16)).min(max_par);
        let prob_m_split = if par_count > 0 {
            par_count * max_tmb * 16
        } else {
            rest_m
        };
        let tmb = prob_m_split.div_ceil(16).min(max_tmb);
        let m8 = prob_m_split <= 8;
        let mut cfg = determine_exec_config(p, tmb, m8, stages, max_shared_mem, avail);
        if let Some(c) = cfg {
            let tiles = p.n / c.thread_n * prob_m_split.div_ceil(tmb * 16) * 4;
            if tiles <= sms
                && is_valid_config(&UNDERFILL_CONFIG, tmb, p, stages, max_shared_mem_new)
                && avail(tmb, m8, &UNDERFILL_CONFIG)
            {
                cfg = Some(UNDERFILL_CONFIG);
            }
        }
        let Some(cfg) = cfg else {
            if max_tmb > 1 {
                max_tmb -= 1;
                continue;
            }
            bail!(
                "marlin: no instantiated thread config for MNK = [{prob_m}, {}, {}], \
                 prob_m_split = {prob_m_split}, group = {:?}",
                p.n,
                p.k,
                p.group_size
            );
        };
        ensure!(
            is_valid_config(&cfg, tmb, p, stages, max_shared_mem_new),
            "marlin: invalid thread config {cfg:?} at thread_m_blocks = {tmb} for MNK = [{prob_m}, {}, {}]",
            p.n,
            p.k
        );
        out.push(Chunk {
            row_offset: row,
            rows: prob_m_split,
            thread_m_blocks: tmb,
            m_block_size_8: m8,
            cfg,
            blocks: sms,
            smem: max_shared_mem_new,
        });
        row += prob_m_split;
        rest_m -= prob_m_split;
    }
    Ok(out)
}

/// The decode policy's `thread_m_blocks` for `rows`: the smallest instantiated value whose tile holds
/// every row (`decode_tmbs` lists the instantiated values, ascending). `None` above the largest.
pub fn decode_thread_m_blocks(rows: usize, decode_tmbs: &[usize]) -> Option<usize> {
    decode_tmbs.iter().copied().find(|&t| t * 16 >= rows)
}

/// The decode policy's one configuration for a weight: the first of [`DECODE_CONFIGS`] that is
/// valid and instantiated (without `m_block_size_8`) at every `thread_m_blocks` in `decode_tmbs`.
pub fn invariant_decode_config(
    p: &Problem,
    decode_tmbs: &[usize],
    max_shared_mem: usize,
    avail: Avail<'_>,
) -> Option<ThreadConfig> {
    let budget = max_shared_mem.saturating_sub(SELECT_SMEM_MARGIN);
    DECODE_CONFIGS.iter().copied().find(|th| {
        !decode_tmbs.is_empty()
            && decode_tmbs
                .iter()
                .all(|&t| is_valid_config(th, t, p, PIPE_STAGES, budget) && avail(t, false, th))
    })
}

/// The decode launch of `rows` under the decode policy.
pub fn decode_chunk(
    rows: usize,
    cfg: ThreadConfig,
    decode_tmbs: &[usize],
    sms: usize,
    max_shared_mem: usize,
) -> Result<Chunk> {
    let Some(tmb) = decode_thread_m_blocks(rows, decode_tmbs) else {
        bail!(
            "marlin: decode of {rows} rows has no instantiated thread_m_blocks in {decode_tmbs:?}"
        );
    };
    Ok(Chunk {
        row_offset: 0,
        rows,
        thread_m_blocks: tmb,
        m_block_size_8: false,
        cfg,
        blocks: sms,
        smem: max_shared_mem,
    })
}

/// Bytes of the FP32 global-reduce buffer (`C_tmp`) a call of at most `rows` rows needs:
/// `(sms + 1) * min(round16(rows), 64) * max_thread_n` floats. vLLM sizes it with `sms` slots
/// (`marlin.cu:528-541`); one spare slot covers the `locks_off` increment of the last stripe.
pub fn c_tmp_bytes(rows: usize, sms: usize) -> usize {
    let m_block = (rows.div_ceil(16) * 16).clamp(16, 64);
    (sms + 1) * m_block * MAX_THREAD_N * 4
}

/// Bytes of a weight's lock array (`sms * MAX_BLOCKS_PER_SM` i32, zero at rest).
pub fn locks_bytes(sms: usize) -> usize {
    sms * MAX_BLOCKS_PER_SM * 4
}

/// The shapes the Marlin GEMM and repack serve without padding (vLLM pads; Atlas refuses and keeps the
/// layer on its other route): N a multiple of 64, K of 16 and of the group, and one of the thread-tile
/// families `(n % 64, k % 128)` or `(n % 128, k % 64)` (`marlin_utils.py:221-243`).
pub fn supports_shape(n: usize, k: usize, group: usize) -> std::result::Result<(), String> {
    if n == 0 || k == 0 {
        return Err(format!("empty shape [{n}x{k}]"));
    }
    if group == 0 || !group.is_multiple_of(16) || !k.is_multiple_of(group) {
        return Err(format!("K={k} does not split into groups of {group}"));
    }
    if !n.is_multiple_of(MIN_THREAD_N) {
        return Err(format!(
            "N={n} is not a multiple of {MIN_THREAD_N} (min_thread_n)"
        ));
    }
    let family_a = k.is_multiple_of(128);
    let family_b = n.is_multiple_of(128) && k.is_multiple_of(MIN_THREAD_K);
    if !(family_a || family_b) {
        return Err(format!(
            "[{n}x{k}] fits neither thread-tile family (n%64 with k%128, or n%128 with k%64)"
        ));
    }
    Ok(())
}
