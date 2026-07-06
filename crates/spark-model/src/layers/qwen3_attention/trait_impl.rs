// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::Qwen3AttentionLayer;
use crate::layer::{
    BatchedAttnMetadata, EmptyLayerState, ForwardContext, LayerState, TransformerLayer,
};
use crate::layers::FfnComponent;

mod decode_inner;
mod multi_seq;
mod prefill_inner;

/// Debug: read back BF16 GPU tensor and compute L2 norm + first 4 values.
pub(super) fn diag_norm(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    n_elements: usize,
    stream: u64,
    label: &str,
) {
    let _ = gpu.synchronize(stream);
    let mut buf = vec![0u16; n_elements];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n_elements * 2) };
    if gpu.copy_d2h(ptr, bytes).is_err() {
        return;
    }
    let vals: Vec<f32> = buf
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();
    let norm: f32 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
    let max_abs: f32 = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let f4 = if vals.len() >= 4 {
        format!(
            "[{:.4},{:.4},{:.4},{:.4}]",
            vals[0], vals[1], vals[2], vals[3]
        )
    } else {
        format!("{:?}", &vals[..vals.len().min(4)])
    };
    tracing::info!("DIAG {label}: norm={norm:.4} max={max_abs:.4} first4={f4} n={n_elements}");
}

/// Debug: read back FP32 GPU tensor and compute L2 norm + first 4 values.
/// Used by the DeepSeek-V4 multi-seq decode diagnostic path (post/comb-attn
/// holographic tensors are FP32). V4-only — no non-V4 caller.
pub fn diag_norm_f32(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    n_elements: usize,
    stream: u64,
    label: &str,
) {
    let _ = gpu.synchronize(stream);
    let mut buf = vec![0f32; n_elements];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n_elements * 4) };
    if gpu.copy_d2h(ptr, bytes).is_err() {
        return;
    }
    let norm: f32 = buf.iter().map(|v| v * v).sum::<f32>().sqrt();
    let max_abs: f32 = buf.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let f4 = if buf.len() >= 4 {
        format!("[{:.4},{:.4},{:.4},{:.4}]", buf[0], buf[1], buf[2], buf[3])
    } else {
        format!("{:?}", &buf[..buf.len().min(4)])
    };
    tracing::info!(
        "DIAG {label}: norm={norm:.4} max={max_abs:.4} first4={f4} n={n_elements} (FP32)"
    );
}

/// V4-Flash cache-line dump (Fable5 write-path A/B, 2026-07-05). Set
/// ATLAS_V4_CACHEDUMP_POS=<p> to dump the raw 576-wide cache line for absolute
/// position `p` at layers 0/21/42 — the assembled BF16 line (pre-quant) AND the
/// FP8 pool bytes (post write_kv_cache). `tag` = "prefill" or "decode" (which
/// writer ran). Files: /tmp/v4dump/{tag}-L{layer}-{bf16,fp8}.txt. Compare a
/// prefill-writer run vs a decode-writer run at the same p to localize the bug
/// to the write path (structured diff) vs the kernel read side (byte-identical).
pub(super) fn v4_cache_dump(
    gpu: &dyn GpuBackend,
    kv_cache: &PagedKvCache,
    layer_idx: usize,
    positions: DevicePtr,
    n_tokens: usize,
    bf16_line: DevicePtr,
    mla_cache_dim: usize,
    block_table: DevicePtr,
    block_size: u32,
    tag: &str,
    stream: u64,
) {
    // Host-ref BF16 key history (independent of the CACHEDUMP gate): append EVERY
    // written token's bf16 assembled line so an offline host-ref has exact BF16
    // keys for all positions. Layer 0 only. Runs on both prefill + decode writes.
    if std::env::var("ATLAS_V4_HOSTREF_POS").is_ok() && layer_idx == 0 && n_tokens > 0 {
        let _ = gpu.synchronize(stream);
        let mut hp = vec![0u32; n_tokens];
        let hpb = unsafe { std::slice::from_raw_parts_mut(hp.as_mut_ptr() as *mut u8, n_tokens * 4) };
        let mut hb = vec![0u16; n_tokens * mla_cache_dim];
        let hbb = unsafe {
            std::slice::from_raw_parts_mut(hb.as_mut_ptr() as *mut u8, n_tokens * mla_cache_dim * 2)
        };
        if gpu.copy_d2h(positions, hpb).is_ok() && gpu.copy_d2h(bf16_line, hbb).is_ok() {
            let _ = std::fs::create_dir_all("/tmp/v4hostref");
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("/tmp/v4hostref/keys_bf16.txt")
            {
                for t in 0..n_tokens {
                    let vals: String = (0..mla_cache_dim)
                        .map(|d| format!("{:.6}", f32::from_bits((hb[t * mla_cache_dim + d] as u32) << 16)))
                        .collect::<Vec<_>>()
                        .join(" ");
                    let _ = writeln!(f, "{} {vals}", hp[t]);
                }
            }
        }
    }
    let target: i64 = match std::env::var("ATLAS_V4_CACHEDUMP_POS") {
        Ok(v) => v.parse().unwrap_or(-1),
        _ => -1,
    };
    if target < 0 || !matches!(layer_idx, 0 | 21 | 42) || n_tokens == 0 {
        return;
    }
    let _ = gpu.synchronize(stream);
    // Find the token in this buffer whose absolute position == target.
    let mut pos = vec![0u32; n_tokens];
    let pb = unsafe { std::slice::from_raw_parts_mut(pos.as_mut_ptr() as *mut u8, n_tokens * 4) };
    if gpu.copy_d2h(positions, pb).is_err() {
        return;
    }
    let ti = match pos.iter().position(|&p| p as i64 == target) {
        Some(i) => i,
        None => return,
    };
    let dir = "/tmp/v4dump";
    let _ = std::fs::create_dir_all(dir);
    // BF16 assembled line (pre write_kv_cache quant).
    let mut b16 = vec![0u16; mla_cache_dim];
    let bb = unsafe { std::slice::from_raw_parts_mut(b16.as_mut_ptr() as *mut u8, mla_cache_dim * 2) };
    if gpu
        .copy_d2h(bf16_line.offset(ti * mla_cache_dim * 2), bb)
        .is_ok()
    {
        let s: String = b16
            .iter()
            .map(|&b| format!("{:.6}", f32::from_bits((b as u32) << 16)))
            .collect::<Vec<_>>()
            .join(" ");
        let _ = std::fs::write(format!("{dir}/{tag}-L{layer_idx}-bf16.txt"), s);
    }
    // FP8 pool bytes (post write_kv_cache). Paged: physical_block = block_table[
    // pos/block_size]; within = pos%block_size; token_stride = mla_cache_dim (1
    // byte/elem fp8, num_kv_heads=1). Mirrors mla_paged_decode_fp8.cu addressing.
    let block_idx = (target as usize) / (block_size as usize);
    let within = (target as usize) % (block_size as usize);
    let mut bt = [0u8; 4];
    if gpu
        .copy_d2h(block_table.offset(block_idx * 4), &mut bt)
        .is_ok()
    {
        let phys = i32::from_le_bytes(bt) as usize;
        let stride = kv_cache.block_stride_bytes_for_layer(layer_idx) as usize;
        let off = phys * stride + within * mla_cache_dim;
        let mut fp8 = vec![0u8; mla_cache_dim];
        if gpu
            .copy_d2h(kv_cache.k_pool_ptr(layer_idx).offset(off), &mut fp8)
            .is_ok()
        {
            let s: String = fp8.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(" ");
            let _ = std::fs::write(format!("{dir}/{tag}-L{layer_idx}-fp8.txt"), s);
        }
    }
    tracing::info!("V4CACHEDUMP {tag} L{layer_idx} pos{target} token_index={ti} dumped");
}

/// V4-Flash prefill-Q dump (Fable5 2026-07-05): dump the prefill-built Q (post-rope,
/// head 0) for the token at absolute position ATLAS_V4_HOSTREF_POS, layer 0. Compared
/// offline against the decode-built Q (v4_hostref_dump's q.txt) — if they differ, the
/// decode Q chain (q_b_norm → rope extract/writeback on reused scratch) is the bug.
#[allow(clippy::too_many_arguments)]
pub(super) fn v4_prefill_q_dump(
    gpu: &dyn GpuBackend,
    layer_idx: usize,
    positions: DevicePtr,
    n_tokens: usize,
    q_full: DevicePtr,
    per_token_stride: usize, // nq*hd_mla
    hd_mla: usize,           // 512 (head 0 = first hd_mla of each token)
    stream: u64,
) {
    let target: i64 = match std::env::var("ATLAS_V4_HOSTREF_POS") {
        Ok(v) => v.parse().unwrap_or(-1),
        _ => -1,
    };
    if target < 0 || layer_idx != 0 || n_tokens == 0 {
        return;
    }
    let _ = gpu.synchronize(stream);
    let mut pos = vec![0u32; n_tokens];
    let pb = unsafe { std::slice::from_raw_parts_mut(pos.as_mut_ptr() as *mut u8, n_tokens * 4) };
    if gpu.copy_d2h(positions, pb).is_err() {
        return;
    }
    let ti = match pos.iter().position(|&p| p as i64 == target) {
        Some(i) => i,
        None => return,
    };
    let mut q = vec![0u16; hd_mla];
    let qb = unsafe { std::slice::from_raw_parts_mut(q.as_mut_ptr() as *mut u8, hd_mla * 2) };
    if gpu
        .copy_d2h(q_full.offset(ti * per_token_stride * 2), qb)
        .is_ok()
    {
        let _ = std::fs::create_dir_all("/tmp/v4hostref");
        let s: String = q
            .iter()
            .map(|&x| format!("{:.6}", f32::from_bits((x as u32) << 16)))
            .collect::<Vec<_>>()
            .join(" ");
        let _ = std::fs::write("/tmp/v4hostref/prefill_q.txt", s);
        tracing::info!("V4PREFILLQ pos{target} token_index={ti} dumped");
    }
}

/// V4-Flash hidden/token-ID dump (Fable5 2026-07-05): at HOSTREF pos, layer 0, dump the
/// token ID (input_ids) AND the raw `hidden` (embedding = hc_expand input) for the target
/// token, BEFORE hc_expand. `base_pos` = absolute position of token index 0 in this pass
/// (decode: seq_len-1; prefill: seq_len_start). IDs match + hidden match => measurements
/// aligned; IDs match + hidden differ => embedding/hidden-source bug; IDs differ => the
/// pos-N A/Bs compared different tokens (artifact).
#[allow(clippy::too_many_arguments)]
pub(super) fn v4_hidden_dump(
    gpu: &dyn GpuBackend,
    tag: &str,
    layer_idx: usize,
    base_pos: usize,
    n_tokens: usize,
    token_ids: DevicePtr,
    hidden: DevicePtr,
    h: usize,
    stream: u64,
) {
    let target: i64 = match std::env::var("ATLAS_V4_HOSTREF_POS") {
        Ok(v) => v.parse().unwrap_or(-1),
        _ => -1,
    };
    if target < 0 || layer_idx != 0 {
        return;
    }
    if (target as usize) < base_pos {
        return;
    }
    let ti = target as usize - base_pos;
    if ti >= n_tokens {
        return;
    }
    let _ = gpu.synchronize(stream);
    let dir = "/tmp/v4hostref";
    let _ = std::fs::create_dir_all(dir);
    let mut idb = [0u8; 4];
    if !token_ids.is_null() && gpu.copy_d2h(token_ids.offset(ti * 4), &mut idb).is_ok() {
        let _ = std::fs::write(format!("{dir}/{tag}-tokid-L0.txt"), u32::from_le_bytes(idb).to_string());
    }
    let mut b = vec![0u16; h];
    let bb = unsafe { std::slice::from_raw_parts_mut(b.as_mut_ptr() as *mut u8, h * 2) };
    if gpu.copy_d2h(hidden.offset(ti * h * 2), bb).is_ok() {
        let s: String = b
            .iter()
            .map(|&x| format!("{:.6}", f32::from_bits((x as u32) << 16)))
            .collect::<Vec<_>>()
            .join(" ");
        let _ = std::fs::write(format!("{dir}/{tag}-hidden-L0.txt"), s);
    }
    tracing::info!("V4HIDDEN {tag} L0 pos{target} ti{ti} base{base_pos} dumped");
}

/// V4-Flash mHC-input dump (Fable5 2026-07-05): at HOSTREF pos, layers 0 + 21, dump
/// the layer input `normed` (post hc_pre + input_norm) AND the raw `residual` (pre-mix
/// hidden) for the target token. Compare decode vs prefill: residual differs => upstream
/// stream diverged; residual same but normed differs => hc_pre mix (mHC state) diverged.
/// L0-identical/L21-different => accumulation (state-update); L0-different => handoff bug.
#[allow(clippy::too_many_arguments)]
pub(super) fn v4_mhc_dump(
    gpu: &dyn GpuBackend,
    tag: &str,
    layer_idx: usize,
    positions: DevicePtr,
    n_tokens: usize,
    normed: DevicePtr,
    residual: DevicePtr,
    h: usize,
    stream: u64,
) {
    let target: i64 = match std::env::var("ATLAS_V4_HOSTREF_POS") {
        Ok(v) => v.parse().unwrap_or(-1),
        _ => -1,
    };
    if target < 0 || !matches!(layer_idx, 0 | 21) || n_tokens == 0 {
        return;
    }
    let _ = gpu.synchronize(stream);
    let mut pos = vec![0u32; n_tokens];
    let pb = unsafe { std::slice::from_raw_parts_mut(pos.as_mut_ptr() as *mut u8, n_tokens * 4) };
    if gpu.copy_d2h(positions, pb).is_err() {
        return;
    }
    let ti = match pos.iter().position(|&p| p as i64 == target) {
        Some(i) => i,
        None => return,
    };
    let dir = "/tmp/v4hostref";
    let _ = std::fs::create_dir_all(dir);
    let dump = |ptr: DevicePtr, name: &str| {
        let mut b = vec![0u16; h];
        let bb = unsafe { std::slice::from_raw_parts_mut(b.as_mut_ptr() as *mut u8, h * 2) };
        if gpu.copy_d2h(ptr.offset(ti * h * 2), bb).is_ok() {
            let s: String = b
                .iter()
                .map(|&x| format!("{:.6}", f32::from_bits((x as u32) << 16)))
                .collect::<Vec<_>>()
                .join(" ");
            let _ = std::fs::write(format!("{dir}/{tag}-{name}-L{layer_idx}.txt"), s);
        }
    };
    dump(normed, "normed");
    if !residual.is_null() {
        dump(residual, "resid");
    }
    tracing::info!("V4MHC {tag} L{layer_idx} pos{target} ti{ti} dumped (normed,resid)");
}

/// V4-Flash host-reference dump (Fable5 discriminator #3, 2026-07-05). At decode
/// position ATLAS_V4_HOSTREF_POS (layer 0), dump everything an offline f64 host-ref
/// needs to recompute head-0 attention: Q(post-rope), kernel attn_out, ALL FP8 pool
/// key lines 0..pos, sink, inv_sqrt_d. Compare host-ref (FP8 & BF16 keys) vs kernel.
#[allow(clippy::too_many_arguments)]
pub(super) fn v4_hostref_dump(
    gpu: &dyn GpuBackend,
    kv_cache: &PagedKvCache,
    layer_idx: usize,
    positions: DevicePtr,
    q_out: DevicePtr,
    attn_out: DevicePtr,
    hd: usize,
    block_table: DevicePtr,
    block_size: u32,
    sink: DevicePtr,
    inv_sqrt_d: f32,
    stream: u64,
) {
    let target: i64 = match std::env::var("ATLAS_V4_HOSTREF_POS") {
        Ok(v) => v.parse().unwrap_or(-1),
        _ => -1,
    };
    if target < 0 || layer_idx != 0 {
        return;
    }
    let _ = gpu.synchronize(stream);
    let mut p0 = [0u8; 4];
    if gpu.copy_d2h(positions, &mut p0).is_err() {
        return;
    }
    if u32::from_le_bytes(p0) as i64 != target {
        return;
    }
    let dir = "/tmp/v4hostref";
    let _ = std::fs::create_dir_all(dir);
    let dump_bf16 = |ptr: DevicePtr, n: usize, name: &str| {
        let mut b = vec![0u16; n];
        let bb = unsafe { std::slice::from_raw_parts_mut(b.as_mut_ptr() as *mut u8, n * 2) };
        if gpu.copy_d2h(ptr, bb).is_ok() {
            let s: String = b
                .iter()
                .map(|&x| format!("{:.6}", f32::from_bits((x as u32) << 16)))
                .collect::<Vec<_>>()
                .join(" ");
            let _ = std::fs::write(format!("{dir}/{name}.txt"), s);
        }
    };
    // Head 0: Q post-rope [0..hd] and kernel attn_out [0..hd].
    dump_bf16(q_out, hd, "q");
    dump_bf16(attn_out, hd, "attn_out_kernel");
    // Sink[0].
    let mut sk = [0u8; 2];
    let sink0 = if !sink.is_null() && gpu.copy_d2h(sink, &mut sk).is_ok() {
        f32::from_bits((u16::from_le_bytes(sk) as u32) << 16)
    } else {
        f32::NEG_INFINITY
    };
    let _ = std::fs::write(
        format!("{dir}/meta.txt"),
        format!("inv_sqrt_d {inv_sqrt_d}\nsink0 {sink0}\nseq {}\nblock_size {block_size}\nhd {hd}\n", target + 1),
    );
    // FP8 pool key lines 0..=target (K==V), via block_table.
    use std::io::Write;
    if let Ok(mut f) = std::fs::File::create(format!("{dir}/keys_fp8.txt")) {
        let stride = kv_cache.block_stride_bytes_for_layer(layer_idx) as usize;
        for j in 0..=(target as usize) {
            let block_idx = j / block_size as usize;
            let within = j % block_size as usize;
            let mut bt = [0u8; 4];
            if gpu.copy_d2h(block_table.offset(block_idx * 4), &mut bt).is_err() {
                continue;
            }
            let phys = i32::from_le_bytes(bt) as usize;
            let mut line = vec![0u8; 576];
            if gpu
                .copy_d2h(kv_cache.k_pool_ptr(layer_idx).offset(phys * stride + within * 576), &mut line)
                .is_ok()
            {
                let vals: String = line.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(" ");
                let _ = writeln!(f, "{j} {vals}");
            }
        }
    }
    tracing::info!("V4HOSTREF pos{target} dumped (q, attn_out_kernel, keys_fp8, keys_bf16, meta)");
}

/// Gemma-4 diagnostic gate. Set ATLAS_DIAG_GEMMA4=1 to enable per-layer
/// hidden-state norm dumps in the decode path. Heavy (one d2h copy per
pub(super) fn gemma4_diag_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_DIAG_GEMMA4").ok().as_deref(),
            Some("1") | Some("true")
        )
    })
}

impl TransformerLayer for Qwen3AttentionLayer {
    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_inner(
            hidden,
            residual,
            state,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_inner(
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len_start,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            kv_write_start,
            None, // batched_meta — single-stream
            ctx,
            stream,
        )
    }

    /// Q12 Path B: batched-mode attention prefill via `prefill_inner` with
    /// `batched_meta = Some`. The model-level `prefill_attn_batched_layer`
    /// calls this method. Per-stream block_table is unused under batched
    /// mode (block_table_ptrs from batched_meta carries them); we still
    /// pass an empty Vec to satisfy the signature.
    fn prefill_inner_batched_q12(
        &self,
        hidden_stacked: DevicePtr,
        residual_stacked: DevicePtr,
        num_tokens: usize,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        batched_meta: &BatchedAttnMetadata,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let mut empty_state = EmptyLayerState;
        let mut empty_block_table: Vec<u32> = Vec::new();
        let mut empty_disk_block_ids: Vec<u32> = Vec::new();
        let mut empty_disk_last: Vec<u32> = Vec::new();
        self.prefill_inner(
            hidden_stacked,
            residual_stacked,
            num_tokens,
            &mut empty_state,
            kv_cache,
            seq_len_start,
            &mut empty_block_table,
            &mut empty_disk_block_ids,
            &mut empty_disk_last,
            0,
            Some(batched_meta),
            ctx,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_multi_seq_inner(
            hidden,
            residual,
            num_seqs,
            states,
            kv_cache,
            seq_lens,
            block_tables,
            ctx,
            stream,
        )
    }

    fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(EmptyLayerState))
    }

    fn transpose_moe_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_for_prefill(gpu, config)?;
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.transpose_for_prefill(gpu, config)?;
        }
        Ok(())
    }

    fn transpose_moe_gate_up_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_gate_up_for_prefill(gpu, config)?;
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.transpose_gate_up_for_prefill(gpu, config)?;
        }
        Ok(())
    }

    fn set_moe_down_transpose_scratch(
        &mut self,
        scratch_packed: DevicePtr,
        scratch_scale: DevicePtr,
        packed_ptrs_t: DevicePtr,
        scale_ptrs_t: DevicePtr,
    ) {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.set_down_transpose_scratch(
                scratch_packed,
                scratch_scale,
                packed_ptrs_t,
                scale_ptrs_t,
            );
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.set_down_transpose_scratch(
                scratch_packed,
                scratch_scale,
                packed_ptrs_t,
                scale_ptrs_t,
            );
        }
    }

    fn transpose_moe_for_prefill_unified(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_for_prefill_unified(gpu, config)?;
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.transpose_for_prefill_unified(gpu, config)?;
        }
        Ok(())
    }

    fn transpose_moe_for_prefill_hybrid(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_for_prefill_hybrid(gpu, config)?;
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.transpose_for_prefill_hybrid(gpu, config)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    #[test]
    fn test_alloc_state_returns_empty() {
        let gpu = MockGpuBackend::new();
        assert!(gpu.kernel("norm", "rms_norm").is_ok());
        assert!(gpu.kernel("rope", "rope_forward").is_ok());
        assert!(
            gpu.kernel("paged_decode_fp8", "paged_decode_attn_fp8")
                .is_ok()
        );
    }
}
