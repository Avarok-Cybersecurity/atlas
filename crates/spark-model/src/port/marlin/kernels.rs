// SPDX-License-Identifier: AGPL-3.0-only
// The instance set follows vLLM (https://github.com/vllm-project/vllm, commit 8a36f08),
// csrc/quantization/marlin/generate_kernels.py:53-180 (thread configs, thread_m_blocks, the 256-thread
// pruning rule) and the kernel parameter list of kernel.h:10-18 (`MARLIN_KERNEL_PARAMS`). Copyright
// contributors to the vLLM project, Apache-2.0; Copyright (C) Marlin.2024 Elias Frantar, Apache-2.0.
// Changes: the generator loop restated in Rust over the pruned Ada set of lane L18a (BF16 activations,
// kU4B8 and kU4 with zero points, group_blocks 8, thread_m_blocks 0.5/1/2/3/4, stages 4) and bound to
// the `extern "C"` entry names of kernels/ada/common/marlin_ada.cu; the three bool parameters are `int`
// in that ABI; the launch goes through `GpuBackend::launch_typed`.

//! Responsibility: the Marlin kernel instances this build carries ([`instances`]), their entry names
//! ([`entry_name`]), the resolved handles ([`MarlinKernels`]) and the one launcher ([`launch`]).
//!
//! The instance table is the API shared with lane L18a: `kernels/ada/common/marlin_ada.cu` exports
//! exactly the 30 names [`entry_name`] gives for [`instances`]
//! (`marlin_ada_bf16_<u4b8|u4>_g128_m<8|16|32|48|64>_k<thread_k>_n<thread_n>_t<threads>`), each an
//! `extern "C" __global__` with vLLM's `MARLIN_KERNEL_PARAMS` list where `has_bias`,
//! `use_atomic_add` and `use_fp32_reduce` are `int` (17 parameters, every scalar 4 bytes). The CPU
//! test `crates/avarok-kernels/tests/marlin_ada_entries.rs` (lane L18a) pins the same names from the
//! source side. Handles are resolved only on a target that compiles the module
//! (`try_target_kernel`), only for names in the table, and only when a weight is built, so a row-off
//! serve issues no lookup.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use anyhow::{Result, bail, ensure};
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};

use super::MarlinType;
use super::config::ThreadConfig;

/// The module (`kernels/ada/common/marlin_ada.cu`, lane L18a).
pub const MODULE: &str = "marlin_ada";
/// The weight types instantiated.
pub const TYPES: [MarlinType; 2] = [MarlinType::U4, MarlinType::U4B8];
/// The `group_blocks` instantiated (8 = groups of 128).
pub const GROUP_BLOCKS: [usize; 1] = [8];
/// `THREAD_CONFIGS` of the generator, `(thread_k, thread_n, threads)`.
pub const GEN_THREAD_CONFIGS: [(usize, usize, usize); 4] = [
    (128, 128, 256),
    (64, 256, 256),
    (64, 128, 128),
    (128, 64, 128),
];
/// `thread_m_blocks` instantiated as `(thread_m_blocks, m_block_size_8)`: vLLM's 0.5, 1, 2, 3, 4.
/// Lane L18a keeps 3 (`m48`) because `marlin_mm` asks for it at 33 to 48 rows (`marlin.cu:307`).
pub const GEN_THREAD_M_BLOCKS: [(usize, bool); 5] =
    [(1, true), (1, false), (2, false), (3, false), (4, false)];
/// The `thread_m_blocks` the decode policy may use (ascending, `m_block_size_8` never).
pub const DECODE_TMBS: [usize; 4] = [1, 2, 3, 4];
/// The number of kernel parameters (`MARLIN_KERNEL_PARAMS`).
pub const KERNEL_PARAMS: usize = 17;
/// The number of entry points `marlin_ada.cu` exports (2 types x 5 M tiles x 3 configurations).
pub const ENTRY_COUNT: usize = 30;

/// One kernel instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KernelKey {
    pub btype: MarlinType,
    /// `None` is the channelwise kernel (`group_blocks == -1`).
    pub group_blocks: Option<usize>,
    pub thread_m_blocks: usize,
    pub m_block_size_8: bool,
    pub thread_n_blocks: usize,
    pub thread_k_blocks: usize,
    pub threads: usize,
}

impl KernelKey {
    /// The key of `cfg` at this type, group and M tile.
    pub fn of(
        btype: MarlinType,
        group_blocks: Option<usize>,
        thread_m_blocks: usize,
        m_block_size_8: bool,
        cfg: &ThreadConfig,
    ) -> Self {
        Self {
            btype,
            group_blocks,
            thread_m_blocks,
            m_block_size_8,
            thread_n_blocks: cfg.thread_n / 16,
            thread_k_blocks: cfg.thread_k / 16,
            threads: cfg.num_threads,
        }
    }

    /// Rows of the CTA tile (8 for `m_block_size_8`, else `16 * thread_m_blocks`).
    pub fn m_rows(&self) -> usize {
        if self.m_block_size_8 {
            8
        } else {
            16 * self.thread_m_blocks
        }
    }
}

/// The `extern "C"` name of an instance in `marlin_ada.cu`'s scheme, e.g.
/// `marlin_ada_bf16_u4b8_g128_m16_k128_n128_t256`: the group size in inputs (`gch` would be the
/// channelwise kernel, which is not instantiated), `m` the CTA's row tile (8, 16, 32, 48 or 64), then
/// `thread_k`, `thread_n` and the thread count.
pub fn entry_name(key: &KernelKey) -> String {
    let g = match key.group_blocks {
        Some(g) => format!("g{}", g * 16),
        None => "gch".to_string(),
    };
    format!(
        "marlin_ada_bf16_{}_{g}_m{}_k{}_n{}_t{}",
        key.btype.label(),
        key.m_rows(),
        key.thread_k_blocks * 16,
        key.thread_n_blocks * 16,
        key.threads
    )
}

/// Every instance of the build, in the generator's order (`generate_kernels.py:148-161`): for each
/// type, group and M tile, each thread configuration, where the 256-thread configurations are kept
/// only as `(128, 128)` for one M block and `(64, 256)` for more. [`ENTRY_COUNT`] of them.
pub fn instances() -> Vec<KernelKey> {
    let mut out = Vec::new();
    for &btype in &TYPES {
        for &gb in &GROUP_BLOCKS {
            for &(tmb, m8) in &GEN_THREAD_M_BLOCKS {
                for &(tk, tn, threads) in &GEN_THREAD_CONFIGS {
                    if threads == 256 {
                        let one_block = tmb <= 1;
                        if one_block && (tk, tn) != (128, 128) {
                            continue;
                        }
                        if !one_block && (tk, tn) != (64, 256) {
                            continue;
                        }
                    }
                    out.push(KernelKey {
                        btype,
                        group_blocks: Some(gb),
                        thread_m_blocks: tmb,
                        m_block_size_8: m8,
                        thread_n_blocks: tn / 16,
                        thread_k_blocks: tk / 16,
                        threads,
                    });
                }
            }
        }
    }
    out
}

/// True when some instance serves groups of `group` at `k` (a single group would need the
/// channelwise kernel, which is not instantiated).
pub fn group_supported(group: usize, k: usize) -> bool {
    if group == 0 || group >= k || !group.is_multiple_of(16) {
        return false;
    }
    GROUP_BLOCKS.contains(&(group / 16))
}

/// The resolved handles of one `(type, group_blocks)` family.
#[derive(Debug)]
pub struct MarlinKernels {
    pub btype: MarlinType,
    pub group_blocks: Option<usize>,
    handles: HashMap<KernelKey, KernelHandle>,
}

impl MarlinKernels {
    /// Build from explicit handles (the tests use this to model a partial instance set).
    pub fn from_handles(
        btype: MarlinType,
        group_blocks: Option<usize>,
        handles: HashMap<KernelKey, KernelHandle>,
    ) -> Self {
        Self {
            btype,
            group_blocks,
            handles,
        }
    }

    /// Resolve every instance of the family. Errors when the module is absent (no lookup is issued)
    /// or when the module lacks a name of the table (the table and `marlin_ada.cu` disagree).
    pub fn resolve(
        gpu: &dyn GpuBackend,
        btype: MarlinType,
        group_blocks: Option<usize>,
    ) -> Result<Self> {
        ensure!(
            gpu.has_module(MODULE),
            "marlin: module {MODULE} is not compiled for this target (the Marlin route needs the Ada build)"
        );
        let mut handles = HashMap::new();
        let mut missing = Vec::new();
        for key in instances()
            .into_iter()
            .filter(|k| k.btype == btype && k.group_blocks == group_blocks)
        {
            let name = entry_name(&key);
            let h = crate::layers::try_target_kernel(gpu, MODULE, &name);
            if h.0 == 0 {
                missing.push(name);
            } else {
                handles.insert(key, h);
            }
        }
        if !missing.is_empty() {
            bail!(
                "marlin: {MODULE} lacks {} entry point(s) of the instance table, first {}",
                missing.len(),
                missing[0]
            );
        }
        ensure!(
            !handles.is_empty(),
            "marlin: no {btype:?} instance at group_blocks {group_blocks:?}"
        );
        Ok(Self {
            btype,
            group_blocks,
            handles,
        })
    }

    /// The handle of an instance, if the build carries it.
    pub fn get(&self, key: &KernelKey) -> Option<KernelHandle> {
        self.handles.get(key).copied()
    }

    /// Whether `(thread_m_blocks, m_block_size_8, cfg)` is instantiated for this family.
    pub fn has(&self, thread_m_blocks: usize, m_block_size_8: bool, cfg: &ThreadConfig) -> bool {
        let key = KernelKey::of(
            self.btype,
            self.group_blocks,
            thread_m_blocks,
            m_block_size_8,
            cfg,
        );
        self.handles.contains_key(&key)
    }
}

type CacheKey = (usize, MarlinType, Option<usize>);

fn cache() -> &'static Mutex<HashMap<CacheKey, Arc<MarlinKernels>>> {
    static C: OnceLock<Mutex<HashMap<CacheKey, Arc<MarlinKernels>>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// [`MarlinKernels::resolve`], once per backend and family (a model builds hundreds of weights). The
/// module check runs on every call, so a backend without the module never hits a stale entry.
pub fn resolved(
    gpu: &dyn GpuBackend,
    btype: MarlinType,
    group_blocks: Option<usize>,
) -> Result<Arc<MarlinKernels>> {
    ensure!(
        gpu.has_module(MODULE),
        "marlin: module {MODULE} is not compiled for this target (the Marlin route needs the Ada build)"
    );
    let id = std::ptr::from_ref(gpu).cast::<()>().addr();
    let key = (id, btype, group_blocks);
    if let Some(k) = cache().lock().get(&key) {
        return Ok(k.clone());
    }
    let k = Arc::new(MarlinKernels::resolve(gpu, btype, group_blocks)?);
    cache().lock().insert(key, k.clone());
    Ok(k)
}

/// The pointer and size arguments of one launch (`MARLIN_KERNEL_PARAMS` order).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchArgs {
    /// `[prob_m, lda]` BF16 activations, 16-byte aligned.
    pub a: DevicePtr,
    /// Repacked codes.
    pub b: DevicePtr,
    /// `[prob_m, prob_n]` BF16 output.
    pub c: DevicePtr,
    /// FP32 reduce buffer (`config::c_tmp_bytes`).
    pub c_tmp: DevicePtr,
    /// Permuted BF16 scales.
    pub scales: DevicePtr,
    /// Packed zero points, or NULL for `kU4B8`.
    pub zp: DevicePtr,
    pub prob_m: usize,
    pub prob_n: usize,
    pub prob_k: usize,
    pub lda: usize,
    /// The weight's lock array (zero at rest; the kernel returns every lock to zero).
    pub locks: DevicePtr,
    pub use_fp32_reduce: bool,
}

fn int_arg(v: usize, what: &str) -> Result<[u8; 4]> {
    let i = i32::try_from(v)
        .map_err(|_| anyhow::anyhow!("marlin: {what} = {v} does not fit an int"))?;
    Ok(i.to_le_bytes())
}

/// Launch one instance: `blocks` CTAs of `threads`, `smem` dynamic shared bytes (also passed as the
/// kernel's `max_shared_mem`; above 48 KiB the registry opts the function in). No bias, no global
/// scale, no atomic-add reduce (so the launch is deterministic). The three flags are 4-byte `int`s,
/// as `marlin_ada.cu` exports them. Capture-safe: no allocation, no synchronization.
pub fn launch(
    gpu: &dyn GpuBackend,
    func: KernelHandle,
    blocks: usize,
    threads: usize,
    smem: usize,
    a: &LaunchArgs,
    stream: u64,
) -> Result<()> {
    ensure!(func.0 != 0, "marlin: launch of an unresolved instance");
    let prob_m = int_arg(a.prob_m, "prob_m")?;
    let prob_n = int_arg(a.prob_n, "prob_n")?;
    let prob_k = int_arg(a.prob_k, "prob_k")?;
    let lda = int_arg(a.lda, "lda")?;
    let max_smem = int_arg(smem, "max_shared_mem")?;
    let no = 0i32.to_le_bytes();
    let fp32 = i32::from(a.use_fp32_reduce).to_le_bytes();
    let args: [KernelArg<'_>; KERNEL_PARAMS] = [
        KernelArg::Buffer(a.a),
        KernelArg::Buffer(a.b),
        KernelArg::Buffer(a.c),
        KernelArg::Buffer(a.c_tmp),
        KernelArg::Buffer(DevicePtr::NULL), // b_bias_ptr
        KernelArg::Buffer(a.scales),
        KernelArg::Buffer(DevicePtr::NULL), // global_scale_ptr (NVFP4 only)
        KernelArg::Buffer(a.zp),
        KernelArg::Bytes(&prob_m),
        KernelArg::Bytes(&prob_n),
        KernelArg::Bytes(&prob_k),
        KernelArg::Bytes(&lda),
        KernelArg::Buffer(a.locks),
        KernelArg::Bytes(&no), // has_bias
        KernelArg::Bytes(&no), // use_atomic_add
        KernelArg::Bytes(&fp32),
        KernelArg::Bytes(&max_smem),
    ];
    let grid = [u32::try_from(blocks)?, 1, 1];
    let block = [u32::try_from(threads)?, 1, 1];
    gpu.launch_typed(func, grid, block, u32::try_from(smem)?, stream, &args)?;
    if gpu.debug_sync_kernels() {
        gpu.synchronize(stream).map_err(|e| {
            anyhow::anyhow!(
                "AVAROK_DEBUG_SYNC_KERNELS: GPU fault after a Marlin launch (grid {grid:?}, block \
                 {block:?}, smem {smem}, m {}, n {}, k {}): {e}",
                a.prob_m,
                a.prob_n,
                a.prob_k
            )
        })?;
    }
    Ok(())
}
