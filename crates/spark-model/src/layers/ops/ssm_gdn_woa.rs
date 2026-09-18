// SPDX-License-Identifier: AGPL-3.0-only

//! Launchers for the GDN write-on-accept K=4 verify pair
//! (`kernels/gb10/common/gated_delta_rule_wy4_woa.cu`): the twin, its
//! post-verdict fold, and the engaged-word clear. Split from `ssm_gdn_b.rs`
//! to keep that file under the 500-LoC cap.
//!
//! provenance-id: 526f6e616c6420522e205374657369616b

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

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

/// Post-verdict fold (`gated_delta_rule_wy4_fold`): applies rows `0..na_tab[b]`
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

/// Reset a layer's write-on-accept engaged word (at the top of a requesting
/// batched verify, same stream and capture as the launches that follow).
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
