// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// GPU-side argmax over BF16 logits.
///
/// Finds the index of the maximum value, writes a single u32 to `out`.
///
/// Kernel: `argmax_bf16(logits, out, n)`
/// Grid: (1, 1, 1)  Block: (1024, 1, 1)
pub fn argmax_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    out: DevicePtr,
    vocab_size: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(out)
        .arg_u32(vocab_size)
        .launch(stream)
}

/// Batched argmax: ONE launch, one block per row, instead of n serial launches of
/// the single-row `argmax_bf16` (which is a one-CTA reduction and so uses 1 of 48
/// SMs). Byte-identical — each block runs the identical per-row body.
#[allow(clippy::too_many_arguments)]
pub fn argmax_bf16_batch(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    out: DevicePtr,
    vocab_size: u32,
    n_rows: u32,
    row_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n_rows, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(out)
        .arg_u32(vocab_size)
        .arg_u32(row_stride)
        .launch(stream)
}

/// Batched argmax that ALSO writes each row's top-1 log-probability
/// (`out_logprob[row] = log softmax(row)[argmax]`, FP32), computed by online
/// softmax in the same pass — same bandwidth as `argmax_bf16_batch`, same
/// index semantics.
///
/// Consumer: D-Cut verification-depth pruning, whose ranking key is the prefix
/// SUM of these log-probabilities (= the log of the prefix product of survival
/// probabilities). Separate kernel so every existing `argmax_bf16_batch` caller
/// stays byte-identical and an unresolved handle is a silent 0 the caller gates
/// on.
#[allow(clippy::too_many_arguments)]
pub fn argmax_bf16_batch_lp(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    out: DevicePtr,
    out_logprob: DevicePtr,
    vocab_size: u32,
    n_rows: u32,
    row_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n_rows, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(out)
        .arg_ptr(out_logprob)
        .arg_u32(vocab_size)
        .arg_u32(row_stride)
        .launch(stream)
}

/// GPU-side argmax + embedding lookup — eliminates D2H sync in MTP propose.
///
/// Reads the argmax result from `argmax_out`, looks up the embedding row
/// from `embed_table`, and writes it to `embed_out`. Also copies the token
/// ID to `token_id_out` for deferred CPU readback.
pub fn embed_from_argmax(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    argmax_out: DevicePtr,
    embed_table: DevicePtr,
    embed_out: DevicePtr,
    token_id_out: DevicePtr,
    hidden_size: u32,
    stream: u64,
) -> Result<()> {
    let grid_x = hidden_size.div_ceil(256);
    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(argmax_out)
        .arg_ptr(embed_table)
        .arg_ptr(embed_out)
        .arg_ptr(token_id_out)
        .arg_u32(hidden_size)
        .launch(stream)
}

/// Batched embedding: gather N rows from embedding table in one launch.
///
/// Replaces N individual D2D copies with a single kernel.
/// `token_ids_dev` must point to `[num_tokens]` u32 on device.
///
/// Kernel: `batched_embed(token_ids, embed_table, output, hidden_size)`
/// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
pub fn batched_embed(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    token_ids_dev: DevicePtr,
    embed_table: DevicePtr,
    output: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(token_ids_dev)
        .arg_ptr(embed_table)
        .arg_ptr(output)
        .arg_u32(hidden_size)
        .launch(stream)
}

/// FP8-table variant of [`batched_embed`]: rows are FP8 E4M3 bytes with a
/// per-row f32 dequant scale (the `quantize_bf16_to_fp8` layout); the
/// kernel dequantizes on read and writes BF16 rows.
///
/// Kernel: `batched_embed_fp8(token_ids, table, row_scale, output, hidden)`
/// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
pub fn batched_embed_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    token_ids_dev: DevicePtr,
    embed_table: DevicePtr,
    row_scale: DevicePtr,
    output: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(token_ids_dev)
        .arg_ptr(embed_table)
        .arg_ptr(row_scale)
        .arg_ptr(output)
        .arg_u32(hidden_size)
        .launch(stream)
}

// ── MoE routing ──────────────────────────────────────────────────
