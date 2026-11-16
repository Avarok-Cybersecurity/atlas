// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the host ABI of `kernels/ada/common/lfm3_shortconv_ada.cu`:
//! module and entry names, launch geometry and argument order of its three
//! entries, and their resolution. Public so the GPU microtest
//! (`examples/lfm3_shortconv_microtest.rs`) launches exactly what the layer
//! launches instead of restating it.
//!
//! The module is compiled only by `kernels/ada` (it is a real file in
//! `kernels/ada/common/`, registered in the `[kernels] overrides` list of
//! `kernels/ada/HARDWARE.toml`). No other target carries it, and only an LFM3
//! model resolves it, so every other model and target issues no lookup.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Module the kernel file compiles to (its file stem).
pub const MODULE: &str = "lfm3_shortconv_ada";
pub const ENTRY_DECODE: &str = "lfm3_shortconv_decode_bf16";
pub const ENTRY_PREFILL: &str = "lfm3_shortconv_prefill_bf16";
pub const ENTRY_PREFILL_STATE: &str = "lfm3_shortconv_prefill_state_bf16";

/// Threads per block (one channel each); mirrors `LFM3_THREADS`.
pub const THREADS: u32 = 128;
/// Tokens per prefill block row; mirrors `LFM3_TILE_T`.
pub const TILE_T: u32 = 64;
/// Sequences per decode launch (pointer parameters `s0..s15`); mirrors
/// `LFM3_DECODE_SEQS`.
pub const DECODE_SEQS: usize = 16;

/// The three resolved entries.
#[derive(Clone, Copy, Debug)]
pub struct ConvKernels {
    pub decode: KernelHandle,
    pub prefill: KernelHandle,
    pub prefill_state: KernelHandle,
}

impl ConvKernels {
    /// Resolve the entries. Fails when the module is not in the image: an
    /// LFM3 checkpoint can only be served on a target that compiles the conv
    /// kernels (today `kernels/ada`), and failing at load is the honest answer
    /// rather than a first-token fault.
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        anyhow::ensure!(
            gpu.has_module(MODULE),
            "LFM3 conv kernels: module `{MODULE}` is not in this kernel image \
             (LFM3 runs only on a target that compiles kernels/ada/common/lfm3_shortconv_ada.cu; \
             build with AVAROK_TARGET_HW=ada)"
        );
        Ok(Self {
            decode: gpu.kernel(MODULE, ENTRY_DECODE)?,
            prefill: gpu.kernel(MODULE, ENTRY_PREFILL)?,
            prefill_state: gpu.kernel(MODULE, ENTRY_PREFILL_STATE)?,
        })
    }
}

/// How the tokens of a prefill launch map onto sequences.
#[derive(Clone, Copy, Debug)]
pub enum SeqLayout {
    /// The whole chunk is one sequence whose window is at this pointer.
    Single(DevicePtr),
    /// Packed sequences: `cu_seqlens` is `[num_seqs + 1]` i32 on the device
    /// (prefix sums, `cu[0] == 0`, `cu[num_seqs] == total_tokens`), and
    /// `state_table` is `[num_seqs]` u64 window pointers on the device.
    Packed {
        cu_seqlens: DevicePtr,
        state_table: DevicePtr,
        num_seqs: u32,
    },
}

impl SeqLayout {
    /// `(cu, table, state0, num_seqs)` as the kernels take them.
    fn args(self) -> (DevicePtr, DevicePtr, DevicePtr, u32) {
        match self {
            Self::Single(state) => (DevicePtr::NULL, DevicePtr::NULL, state, 1),
            Self::Packed {
                cu_seqlens,
                state_table,
                num_seqs,
            } => (cu_seqlens, state_table, DevicePtr::NULL, num_seqs),
        }
    }
}

/// Row strides and width shared by every launch, in BF16 elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Strides {
    /// Channels (`hidden`).
    pub dim: u32,
    /// Elements between consecutive BCx rows (`>= 3 * dim`).
    pub bcx: u32,
    /// Elements between consecutive y rows (`>= dim`).
    pub y: u32,
}

impl Strides {
    /// Dense rows: BCx `[rows, 3 * dim]`, y `[rows, dim]`.
    pub fn dense(dim: u32) -> Self {
        Self {
            dim,
            bcx: 3 * dim,
            y: dim,
        }
    }
}

/// Grid of one decode launch of `rows` sequences.
pub fn decode_grid(dim: u32, rows: u32) -> [u32; 3] {
    [div_ceil(dim, THREADS), rows, 1]
}

/// Grid of the prefill y kernel.
pub fn prefill_grid(dim: u32, total_tokens: u32) -> [u32; 3] {
    [div_ceil(dim, THREADS), div_ceil(total_tokens, TILE_T), 1]
}

/// Grid of the final-state kernel.
pub fn state_grid(dim: u32, num_seqs: u32) -> [u32; 3] {
    [div_ceil(dim, THREADS), num_seqs, 1]
}

/// One decode token for each of `states.len()` sequences: BCx row `r` and y
/// row `r` belong to `states[r]`, whose window shifts in place. Launches once
/// per group of [`DECODE_SEQS`] rows. A NULL state pointer skips its row.
#[allow(clippy::too_many_arguments)]
pub fn decode(
    gpu: &dyn GpuBackend,
    k: &ConvKernels,
    bcx: DevicePtr,
    weight: DevicePtr,
    y: DevicePtr,
    strides: Strides,
    states: &[DevicePtr],
    stream: u64,
) -> Result<()> {
    for (g, group) in states.chunks(DECODE_SEQS).enumerate() {
        let row0 = g * DECODE_SEQS;
        let mut l = KernelLaunch::new(gpu, k.decode)
            .grid(decode_grid(strides.dim, group.len() as u32))
            .block([THREADS, 1, 1])
            .arg_ptr(bcx.offset(row0 * strides.bcx as usize * 2))
            .arg_ptr(weight)
            .arg_ptr(y.offset(row0 * strides.y as usize * 2))
            .arg_u32(strides.dim)
            .arg_u32(strides.bcx)
            .arg_u32(strides.y)
            .arg_u32(group.len() as u32);
        for slot in 0..DECODE_SEQS {
            l = l.arg_ptr(group.get(slot).copied().unwrap_or(DevicePtr::NULL));
        }
        l.launch(stream)?;
    }
    Ok(())
}

/// The y half of a prefill over `total_tokens` token-major rows. Reads the
/// initial windows; never writes them (that is [`prefill_state`]).
#[allow(clippy::too_many_arguments)]
pub fn prefill(
    gpu: &dyn GpuBackend,
    k: &ConvKernels,
    bcx: DevicePtr,
    weight: DevicePtr,
    y: DevicePtr,
    layout: SeqLayout,
    total_tokens: u32,
    strides: Strides,
    stream: u64,
) -> Result<()> {
    if total_tokens == 0 {
        return Ok(());
    }
    let (cu, table, state0, num_seqs) = layout.args();
    KernelLaunch::new(gpu, k.prefill)
        .grid(prefill_grid(strides.dim, total_tokens))
        .block([THREADS, 1, 1])
        .arg_ptr(bcx)
        .arg_ptr(weight)
        .arg_ptr(y)
        .arg_ptr(cu)
        .arg_ptr(table)
        .arg_ptr(state0)
        .arg_u32(total_tokens)
        .arg_u32(num_seqs)
        .arg_u32(strides.dim)
        .arg_u32(strides.bcx)
        .arg_u32(strides.y)
        .launch(stream)
}

/// The final windows of a prefill, in place. Must run after [`prefill`] on
/// the same stream.
pub fn prefill_state(
    gpu: &dyn GpuBackend,
    k: &ConvKernels,
    bcx: DevicePtr,
    layout: SeqLayout,
    total_tokens: u32,
    strides: Strides,
    stream: u64,
) -> Result<()> {
    let (cu, table, state0, num_seqs) = layout.args();
    if num_seqs == 0 {
        return Ok(());
    }
    KernelLaunch::new(gpu, k.prefill_state)
        .grid(state_grid(strides.dim, num_seqs))
        .block([THREADS, 1, 1])
        .arg_ptr(bcx)
        .arg_ptr(cu)
        .arg_ptr(table)
        .arg_ptr(state0)
        .arg_u32(total_tokens)
        .arg_u32(num_seqs)
        .arg_u32(strides.dim)
        .arg_u32(strides.bcx)
        .launch(stream)
}

/// Both prefill launches in order: y from the initial windows, then the
/// windows advanced past the chunk.
#[allow(clippy::too_many_arguments)]
pub fn prefill_chunk(
    gpu: &dyn GpuBackend,
    k: &ConvKernels,
    bcx: DevicePtr,
    weight: DevicePtr,
    y: DevicePtr,
    layout: SeqLayout,
    total_tokens: u32,
    strides: Strides,
    stream: u64,
) -> Result<()> {
    prefill(
        gpu,
        k,
        bcx,
        weight,
        y,
        layout,
        total_tokens,
        strides,
        stream,
    )?;
    prefill_state(gpu, k, bcx, layout, total_tokens, strides, stream)
}
