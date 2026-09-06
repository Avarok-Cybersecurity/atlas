// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Fused 3-token GDN decode (K=3 speculative verification).
///
/// Processes exactly 3 tokens through GDN in a single kernel launch.
/// Saves 2 intermediate H states (H_1, H_2) for rollback on draft rejection.
/// 4 passes vs 6 for 3× sequential decode.
///
/// Kernel: `gated_delta_rule_chunk3(h_state, query, key, value, gate, beta,
///          output, h_inter0, h_inter1, batch_size, num_k_heads,
///          num_v_heads, k_dim, v_dim, qk_stride, v_stride, gb_stride)`
/// Grid: (num_v_heads, batch, 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_chunk3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    h_state_inter0: DevicePtr,
    h_state_inter1: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_ptr(h_state_inter0)
        .arg_ptr(h_state_inter1)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .launch(stream)
}

/// WY-chunkwise 2-token GDN decode (2-pass algorithm).
///
/// Drop-in replacement for `gdn_decode_chunk2`. Computes both H^T @ k_t
/// dot products in a single pass over H, then applies WY algebraic correction.
/// 2 passes vs 3, reducing memory traffic by 33%.
///
/// Grid: (num_v_heads, batch, 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_wy2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    h_state_intermediate: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    // false = contiguous state bases indexed by (b*num_v_heads+vh);
    // true  = device pointer TABLES, one entry per sequence. See
    // `gdn_decode_wy4` for the full rationale — contiguous is only correct at
    // batch_size==1 because the intermediate's pool stride is
    // num_intermediates x h_state's.
    state_is_table: bool,
    stream: u64,
) -> Result<()> {
    // HARD guard, not debug_assert: this compiles out in release, which is
    // exactly where the corruption would be silent.
    anyhow::ensure!(
        state_is_table || batch_size == 1,
        "gdn_decode_wy2: contiguous state addressing is only valid at \
         batch_size==1 (got {batch_size}) — the intermediate's pool stride is \
         num_intermediates x h_state's, so sequence 1's Hi0 would land on \
         sequence 0's Hi1. Stage pointer tables and pass state_is_table=true."
    );
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_ptr(h_state_intermediate)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .arg_u32(u32::from(state_is_table))
        .launch(stream)
}

/// WY-chunkwise 3-token GDN decode (2-pass algorithm).
///
/// Drop-in replacement for `gdn_decode_chunk3`. All 3 H^T @ k_t dot products
/// computed in a single pass. 2 passes vs 4, reducing memory traffic by 50%.
///
/// Grid: (num_v_heads, batch, 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_wy3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    h_state_inter0: DevicePtr,
    h_state_inter1: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    // false = contiguous state bases indexed by (b*num_v_heads+vh);
    // true  = device pointer TABLES, one entry per sequence. See
    // `gdn_decode_wy4` for the full rationale — contiguous is only correct at
    // batch_size==1 because the intermediates' pool stride is
    // num_intermediates x h_state's.
    state_is_table: bool,
    stream: u64,
) -> Result<()> {
    // HARD guard, not debug_assert: this compiles out in release, which is
    // exactly where the corruption would be silent.
    anyhow::ensure!(
        state_is_table || batch_size == 1,
        "gdn_decode_wy3: contiguous state addressing is only valid at \
         batch_size==1 (got {batch_size}) — the intermediates' pool stride is \
         num_intermediates x h_state's, so sequence 1's Hi0 would land on \
         sequence 0's Hi1. Stage pointer tables and pass state_is_table=true."
    );
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_ptr(h_state_inter0)
        .arg_ptr(h_state_inter1)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .arg_u32(u32::from(state_is_table))
        .launch(stream)
}

/// WY-chunkwise 4-token GDN decode (2-pass algorithm).
///
/// All 4 H^T @ k_t dot products computed in a single pass, then WY correction
/// derives v_new values. Second pass applies all 4 state updates + outputs.
/// 2 passes vs 5, reducing memory traffic by 60%.
///
/// Grid: (num_v_heads, batch, 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_wy4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    h_state_inter0: DevicePtr,
    h_state_inter1: DevicePtr,
    h_state_inter2: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    // false = contiguous state bases indexed by (b*num_v_heads+vh);
    // true  = device pointer TABLES, one entry per sequence.
    //
    // Contiguous is only correct at batch_size==1: it assumes the intermediates
    // share h_state's batch stride, but the pool's intermediate stride is
    // num_intermediates x larger, so at n>1 sequence 1's Hi0 lands on sequence
    // 0's Hi1 — silent cross-sequence rollback corruption. Pass true with staged
    // tables for any batched verify. false is byte-identical to the old kernel.
    state_is_table: bool,
    stream: u64,
) -> Result<()> {
    // HARD guard, not debug_assert: this compiles out in release, which is
    // exactly where the corruption would be silent.
    anyhow::ensure!(
        state_is_table || batch_size == 1,
        "gdn_decode_wy4: contiguous state addressing is only valid at \
         batch_size==1 (got {batch_size}) — the intermediates' pool stride is \
         num_intermediates x h_state's, so sequence 1's Hi0 would land on \
         sequence 0's Hi1. Stage pointer tables and pass state_is_table=true."
    );
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_ptr(h_state_inter0)
        .arg_ptr(h_state_inter1)
        .arg_ptr(h_state_inter2)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .arg_u32(u32::from(state_is_table))
        .launch(stream)
}

/// WY-Chunkwise Gated Delta Rule, pool-layout intermediates — K-generic
/// launch shared by the K=17 DFlash verify (`gated_delta_rule_wy17`) and the
/// chain-verify K∈{5..8} instantiations (`gated_delta_rule_wy5..wy8`, one
/// templated source `gated_delta_rule_wyn.cu`). K is compile-time in the
/// kernel; the caller selects it via the `kernel` handle. Computes K H·k dot
/// products in 1 pass over H, applies WY algebraic correction over K tokens
/// (K*(K-1)/2 inter-token k-dots), then applies K state updates in a second
/// fused pass writing Hi_0..Hi_{K-2} + final H.
///
/// `h_state_inter_base` points to a contiguous pool of (K-1) intermediate
/// H states per (layer, slot). Each Hi_t is at
/// `h_state_inter_base + t * inter_stride_floats` (per (b, vh) sub-region).
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_wyn(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    h_state_inter_base: DevicePtr,
    inter_stride_floats: u32,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    stream: u64,
) -> Result<()> {
    // `gdn_decode_wyn`'s kernel hardcodes the CONTIGUOUS state stride
    // ((b*num_v_heads+vh)*hv) for BOTH h_state and the intermediates. That is
    // wrong for the intermediates, whose pool stride is num_intermediates x
    // larger — at batch_size>1 sequence 1's Hi0 lands on sequence 0's Hi1,
    // silently corrupting cross-sequence rollback. Only wy4 has the
    // `state_is_table` pointer-table form that sidesteps this. Refuse rather
    // than corrupt; port the table form (see gated_delta_rule_wy4.cu) before
    // enabling a batched verify at K<4.
    anyhow::ensure!(
        batch_size == 1,
        "gdn_decode_wyn: contiguous state addressing is only valid at batch_size==1 \
         (got {batch_size}); port the wy4 `state_is_table` pointer-table form first"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_ptr(h_state_inter_base)
        .arg_u32(inter_stride_floats)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .launch(stream)
}

/// Cross-sequence batched-verify launch of the wyN pointer-table twins
/// (`gated_delta_rule_wy{5..16}_table` / `_f16_table`): `state_is_table`
/// is compiled into the symbol, so the argument list is identical to
/// [`gdn_decode_wyn`] — but the STATE arguments mean something different
/// and batching is the point:
/// - `h_tables`   = the layer's staged table slice: slab 0 holds one
///   per-sequence H base pointer per entry (`VERIFY_WY_TABLE_SEQS` slots).
/// - `hi_tables`  = slab 1 (Hi0); consecutive Hi slabs follow at
///   `VERIFY_WY_TABLE_SEQS`-entry stride.
/// - `slab_entry_stride` MUST be `VERIFY_WY_TABLE_SEQS as u32` — the
///   kernel reinterprets the contiguous-form stride argument as "pointer
///   entries between Hi slabs".
///
/// Tables are staged by `upload_verify_wy_tables` (verify_e2.rs) into a
/// fixed buffer refreshed pre-replay, so the launch is CUDA-graph-stable.
#[allow(clippy::too_many_arguments)]
// provenance-id: 526f6e616c6420522e205374657369616b
#[allow(dead_code)] // seam 2026-09-01: consumer = the dispatcher 5..=16 arm, next change
pub fn gdn_decode_wyn_table(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_tables: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    hi_tables: DevicePtr,
    slab_entry_stride: u32,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        !h_tables.is_null() && !hi_tables.is_null(),
        "gdn_decode_wyn_table: staged pointer tables are required (NULL table \
         means upload_verify_wy_tables declined — run the per-sequence loop)"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .arg_ptr(h_tables)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_ptr(hi_tables)
        .arg_u32(slab_entry_stride)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .launch(stream)
}

/// Fused 2-token conv1d sliding window update + SiLU.
///
/// Each thread handles one channel independently. The 2-token dependency
/// (token 1's window includes token 0's input) is resolved in registers.
/// Saves intermediate conv_state (after token 0) for rollback.
///
/// Kernel: `causal_conv1d_update_chunk2(conv_state, input, weight, bias,
///          output, conv_state_intermediate, batch, dim, d_conv)`
/// Grid: (ceil(dim/256), batch, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_chunk2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    conv_state_intermediate: DevicePtr,
    d_inner: u32,
    d_conv: u32,
    batch_size: u32,
    stream: u64,
) -> Result<()> {
    let bias_ptr = DevicePtr::NULL;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(d_inner, 256), batch_size, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(bias_ptr)
        .arg_ptr(output)
        .arg_ptr(conv_state_intermediate)
        .arg_u32(batch_size)
        .arg_u32(d_inner)
        .arg_u32(d_conv)
        .launch(stream)
}

// ── Activations / Element-wise ─────────────────────────────────────

/// Write-on-accept K=4 verify (`gated_delta_rule_wy4_woa`): byte-identical
/// `output` to `gdn_decode_wy4`, writes NO state, stashes the per-row update
/// terms for `gdn_wy4_fold`. Table form only. Grid (num_v_heads, batch), 128.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_wy4_woa(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_table: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    stash: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    stash_seq_floats: u32,
    engaged_flag: DevicePtr,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        !h_table.is_null() && !stash.is_null() && !engaged_flag.is_null(),
        "gdn_decode_wy4_woa: null table/stash/flag"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .shared_mem(k_dim * v_dim * 4)
        .arg_ptr(h_table)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_ptr(stash)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .arg_u32(stash_seq_floats)
        .arg_ptr(engaged_flag)
        .launch(stream)
}

/// Post-verdict fold (`gated_delta_rule_wy4_fold`): applies rows 0..na_tab[b]
/// of the stashed updates to H, one read + one write per state.
#[allow(clippy::too_many_arguments)]
pub fn gdn_wy4_fold(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_table: DevicePtr,
    stash: DevicePtr,
    na_tab: DevicePtr,
    hi_tables: DevicePtr,
    slab_entries: u32,
    engaged_flag: DevicePtr,
    k_rows: u32,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    stash_seq_floats: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .arg_ptr(h_table)
        .arg_ptr(stash)
        .arg_ptr(na_tab)
        .arg_ptr(hi_tables)
        .arg_u32(slab_entries)
        .arg_ptr(engaged_flag)
        .arg_u32(k_rows)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(stash_seq_floats)
        .launch(stream)
}

/// Reset a layer's write-on-accept engaged word (after its fold, same stream).
pub fn gdn_wy4_flag_clear(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    flag: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(flag)
        .launch(stream)
}
