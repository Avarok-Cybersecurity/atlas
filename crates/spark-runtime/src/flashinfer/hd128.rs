// SPDX-License-Identifier: AGPL-3.0-only
//! head_dim=128 (Laguna) FlashInfer ragged-prefill wrapper. Split from
//! `flashinfer.rs` (500-LoC cap); a child module so it shares the private
//! workspace singleton.

use anyhow::{Result, bail};

#[cfg(atlas_flashinfer)]
use std::ffi::c_void;

#[cfg(atlas_flashinfer)]
use super::workspaces;

#[cfg(atlas_flashinfer)]
unsafe extern "C" {
    // head_dim=128 (Laguna). Same ABI as hd256 plus `window_left` after
    // `causal`: -1 = full causal attention, otherwise sliding_window - 1.
    #[allow(clippy::too_many_arguments)]
    fn atlas_fi_ragged_prefill_bf16_hd128(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        o: *mut c_void,
        qo_indptr_h: *const i32,
        kv_indptr_h: *const i32,
        qo_indptr_d: *const i32,
        kv_indptr_d: *const i32,
        batch: u32,
        total_qo_rows: u32,
        total_kv_rows: u32,
        num_qo_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        sm_scale: f32,
        causal: i32,
        window_left: i32,
        float_ws: *mut c_void,
        float_ws_bytes: usize,
        int_ws: *mut c_void,
        int_ws_bytes: usize,
        pinned_int_ws: *mut c_void,
        pinned_int_ws_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

/// Ragged batched prefill attention, BF16, **head_dim=128** (Laguna), GQA.
///
/// Same contract as [`super::ragged_prefill_bf16_hd256`] but with a sliding-window
/// bound. `sliding_window` is the Atlas convention (mask when `q - k >= w`,
/// see `kernels/gb10/common/inferspark_prefill.cu`); pass `None` for the
/// full-attention layers. It is converted to FlashInfer's `window_left`
/// (= `w - 1`) internally.
///
/// Passing the window is not just a correctness requirement for Laguna's 36
/// sliding-window-512 layers — FlashInfer's scheduler uses it to skip
/// out-of-window KV tiles entirely, so it is also where the speedup comes from.
#[allow(clippy::too_many_arguments)]
pub fn ragged_prefill_bf16_hd128(
    q: u64,
    k: u64,
    v: u64,
    o: u64,
    qo_indptr_h: &[i32],
    kv_indptr_h: &[i32],
    qo_indptr_d: u64,
    kv_indptr_d: u64,
    batch: u32,
    total_qo_rows: u32,
    total_kv_rows: u32,
    num_qo_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    sm_scale: f32,
    causal: bool,
    sliding_window: Option<u32>,
    stream: u64,
) -> Result<()> {
    #[cfg(atlas_flashinfer)]
    {
        if head_dim != 128 {
            bail!("ragged_prefill_bf16_hd128 is head_dim=128 only (got {head_dim})");
        }
        if qo_indptr_h.len() != (batch + 1) as usize || kv_indptr_h.len() != (batch + 1) as usize {
            bail!("indptr host slices must be batch+1 long");
        }
        // Atlas masks at (q - k) >= w; FlashInfer keeps kv within window_left of
        // the query, so window_left = w - 1. w == 0 means "no window".
        let window_left: i32 = match sliding_window {
            Some(w) if w > 0 => (w - 1) as i32,
            _ => -1,
        };
        let ws = workspaces()?;
        let st = unsafe {
            atlas_fi_ragged_prefill_bf16_hd128(
                q as *const c_void,
                k as *const c_void,
                v as *const c_void,
                o as *mut c_void,
                qo_indptr_h.as_ptr(),
                kv_indptr_h.as_ptr(),
                qo_indptr_d as *const i32,
                kv_indptr_d as *const i32,
                batch,
                total_qo_rows,
                total_kv_rows,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                sm_scale,
                if causal { 1 } else { 0 },
                window_left,
                ws.float_ws as *mut c_void,
                ws.float_sz,
                ws.int_ws as *mut c_void,
                ws.int_sz,
                ws.pinned_int_ws as *mut c_void,
                ws.pinned_sz,
                stream as *mut c_void,
            )
        };
        if st != 0 {
            bail!(
                "FlashInfer ragged prefill hd128 failed: status {st} \
                 (batch={batch}, qo={total_qo_rows}, window_left={window_left})"
            );
        }
        Ok(())
    }
    #[cfg(not(atlas_flashinfer))]
    {
        let _ = (
            q,
            k,
            v,
            o,
            qo_indptr_h,
            kv_indptr_h,
            qo_indptr_d,
            kv_indptr_d,
            batch,
            total_qo_rows,
            total_kv_rows,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            sm_scale,
            causal,
            sliding_window,
            stream,
        );
        bail!("FlashInfer support was not built; set FLASHINFER_HOME when building")
    }
}
