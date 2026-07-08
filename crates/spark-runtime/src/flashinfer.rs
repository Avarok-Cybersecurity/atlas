// SPDX-License-Identifier: AGPL-3.0-only
//! Host-callable FlashInfer ragged/varlen prefill attention FFI (GB10/sm_121).
//!
//! FlashInfer's `BatchPrefillWithRaggedKVCacheDispatched` is a FlashAttention-2
//! SM80-class kernel (mma.sync/ldmatrix/cp.async) that codegens for sm_121f. We
//! wrap it host-side exactly like the CUTLASS object: nvcc compiles
//! `cuda/flashinfer_ragged_prefill.cu` to a static lib in `build.rs` (gated on
//! `FLASHINFER_HOME`), this module declares the `extern "C"` ABI, and callers
//! pass `u64` device pointers + a `cudaStream_t` as `u64`.
//!
//! Purpose: batch N requests' attention into ONE varlen launch (q_indptr/
//! kv_indptr ragged offsets) so cross-request prefill scales — the missing
//! piece behind Atlas's flat ~3880 tok/s prefill at any concurrency.

use anyhow::{Result, bail};

#[cfg(atlas_flashinfer)]
use std::ffi::c_void;
#[cfg(atlas_flashinfer)]
use std::sync::OnceLock;

#[cfg(atlas_flashinfer)]
unsafe extern "C" {
    // Available for diagnostics; the wrapper uses fixed workspace budgets instead.
    #[allow(dead_code)]
    fn atlas_fi_ragged_prefill_workspace_sizes(
        max_batch: u32,
        max_total_qo_rows: u32,
        num_qo_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        float_ws_bytes_out: *mut usize,
        int_ws_bytes_out: *mut usize,
        pinned_int_ws_bytes_out: *mut usize,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn atlas_fi_ragged_prefill_bf16_hd256(
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
        float_ws: *mut c_void,
        float_ws_bytes: usize,
        int_ws: *mut c_void,
        int_ws_bytes: usize,
        pinned_int_ws: *mut c_void,
        pinned_int_ws_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    #[cfg(atlas_flashinfer)]
    fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
    #[cfg(atlas_flashinfer)]
    fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
}

/// Whether the FlashInfer wrapper was compiled in (FLASHINFER_HOME was set at build).
pub fn available() -> bool {
    cfg!(atlas_flashinfer)
}

// Persistent workspaces, sized for a generous max config and reused across calls
// (FlashInfer plans into these each call; they don't carry state between calls).
#[cfg(atlas_flashinfer)]
struct Workspaces {
    float_ws: u64,
    int_ws: u64,
    pinned_int_ws: u64,
    float_sz: usize,
    int_sz: usize,
    pinned_sz: usize,
}
#[cfg(atlas_flashinfer)]
unsafe impl Send for Workspaces {}
#[cfg(atlas_flashinfer)]
unsafe impl Sync for Workspaces {}
#[cfg(atlas_flashinfer)]
static WS: OnceLock<Workspaces> = OnceLock::new();

// Max config the persistent workspaces are sized for. Generous upper bounds for
// Holo serving (>= any realistic concurrent-prefill batch). num heads/head_dim
// are fixed by the model.
#[cfg(atlas_flashinfer)]
const MAX_BATCH: u32 = 16;
#[cfg(atlas_flashinfer)]
const MAX_TOTAL_QO_ROWS: u32 = 16 * 16384;
#[cfg(atlas_flashinfer)]
const N_QO_HEADS: u32 = 16;
#[cfg(atlas_flashinfer)]
const N_KV_HEADS: u32 = 2;
#[cfg(atlas_flashinfer)]
const HEAD_DIM: u32 = 256;

// FlashInfer's float workspace is a BUDGET PrefillPlan splits KV within (it
// plans to fit, not a hard requirement) — vLLM uses ~128MB. The int/pinned
// workspaces hold the scheduler metadata arrays (request/tile indices), bounded
// by tile count. Fixed generous budgets; PrefillPlan adapts within them.
#[cfg(atlas_flashinfer)]
const FLOAT_WS_BYTES: usize = 256 << 20; // 256 MB
#[cfg(atlas_flashinfer)]
const INT_WS_BYTES: usize = 64 << 20; // 64 MB
#[cfg(atlas_flashinfer)]
const PINNED_WS_BYTES: usize = 64 << 20; // 64 MB

#[cfg(atlas_flashinfer)]
fn workspaces() -> Result<&'static Workspaces> {
    if let Some(w) = WS.get() {
        return Ok(w);
    }
    let _ = (
        MAX_BATCH,
        MAX_TOTAL_QO_ROWS,
        N_QO_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
    );
    let (fsz, isz, psz) = (FLOAT_WS_BYTES, INT_WS_BYTES, PINNED_WS_BYTES);
    let mut float_ws = 0u64;
    let mut int_ws = 0u64;
    let mut pinned = std::ptr::null_mut::<c_void>();
    unsafe {
        let s1 = cuMemAlloc_v2(&mut float_ws, fsz.max(1));
        if s1 != 0 {
            bail!("cuMemAlloc FlashInfer float ws ({fsz}B) failed: {s1}");
        }
        let s2 = cuMemAlloc_v2(&mut int_ws, isz.max(1));
        if s2 != 0 {
            bail!("cuMemAlloc FlashInfer int ws ({isz}B) failed: {s2}");
        }
        let s3 = cudaHostAlloc(&mut pinned, psz.max(1), 0);
        if s3 != 0 {
            bail!("cudaHostAlloc FlashInfer pinned int ws ({psz}B) failed: {s3}");
        }
    }
    let _ = WS.set(Workspaces {
        float_ws,
        int_ws,
        pinned_int_ws: pinned as u64,
        float_sz: fsz,
        int_sz: isz,
        pinned_sz: psz,
    });
    Ok(WS.get().unwrap())
}

/// Ragged batched prefill attention (BF16, head_dim=256, GQA, causal selectable).
///
/// `q`/`o`: `[total_qo_rows, num_qo_heads, 256]` BF16 device; `k`/`v`:
/// `[total_kv_rows, num_kv_heads, 256]` BF16 device. `qo_indptr`/`kv_indptr` are
/// `[batch+1]` int32 prefix-sum offsets — provided both on host (`*_h`, for the
/// scheduler plan) and as device copies (`*_d`, read by the kernel).
#[allow(clippy::too_many_arguments)]
pub fn ragged_prefill_bf16_hd256(
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
    stream: u64,
) -> Result<()> {
    #[cfg(atlas_flashinfer)]
    {
        if head_dim != HEAD_DIM {
            bail!("FlashInfer wrapper is head_dim=256 only (got {head_dim})");
        }
        if qo_indptr_h.len() != (batch + 1) as usize || kv_indptr_h.len() != (batch + 1) as usize {
            bail!("indptr host slices must be batch+1 long");
        }
        let ws = workspaces()?;
        let st = unsafe {
            atlas_fi_ragged_prefill_bf16_hd256(
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
                "FlashInfer ragged prefill failed: status {st} (batch={batch}, qo={total_qo_rows})"
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
            stream,
        );
        bail!("FlashInfer support was not built; set FLASHINFER_HOME when building")
    }
}

#[cfg(all(test, atlas_flashinfer))]
mod tests {
    use super::*;
    use std::ffi::c_void;

    const H2D: i32 = 1;
    const D2H: i32 = 2;
    unsafe extern "C" {
        fn cudaMalloc(p: *mut *mut c_void, n: usize) -> i32;
        fn cudaFree(p: *mut c_void) -> i32;
        fn cudaMemcpy(d: *mut c_void, s: *const c_void, n: usize, k: i32) -> i32;
        fn cudaDeviceSynchronize() -> i32;
    }
    fn f32_to_bf16(x: f32) -> u16 {
        let b = x.to_bits();
        ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
    }
    fn bf16_to_f32(x: u16) -> f32 {
        f32::from_bits((x as u32) << 16)
    }
    unsafe fn dev<T>(data: &[T]) -> u64 {
        let bytes = std::mem::size_of_val(data);
        let mut p = std::ptr::null_mut();
        assert_eq!(unsafe { cudaMalloc(&mut p, bytes.max(1)) }, 0);
        assert_eq!(
            unsafe { cudaMemcpy(p, data.as_ptr() as *const c_void, bytes, H2D) },
            0
        );
        p as u64
    }

    #[test]
    #[ignore = "requires a free CUDA device + FLASHINFER_HOME build"]
    #[allow(clippy::needless_range_loop)] // numerical reference: index-parallel loops read clearest
    fn flashinfer_ragged_prefill_matches_cpu_reference() {
        // 2 ragged requests (6 + 10 tokens), GQA 4 qo / 2 kv heads, hd=256, causal.
        const HD: usize = 256;
        const NQO: usize = 4;
        const NKV: usize = 2;
        let lens = [6usize, 10usize];
        let qo_indptr: Vec<i32> = {
            let mut v = vec![0i32];
            for &l in &lens {
                v.push(v.last().unwrap() + l as i32);
            }
            v
        };
        let kv_indptr = qo_indptr.clone();
        let total: usize = lens.iter().sum();
        let sm_scale = 1.0f32 / (HD as f32).sqrt();

        // Deterministic pseudo-random bf16 inputs.
        let rnd = |seed: u64| -> f32 {
            let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            x ^= x >> 31;
            x = x.wrapping_mul(0xBF58476D1CE4E5B9);
            ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.5
        };
        let q: Vec<u16> = (0..total * NQO * HD)
            .map(|i| f32_to_bf16(rnd(i as u64)))
            .collect();
        let k: Vec<u16> = (0..total * NKV * HD)
            .map(|i| f32_to_bf16(rnd(i as u64 ^ 0x1111)))
            .collect();
        let v: Vec<u16> = (0..total * NKV * HD)
            .map(|i| f32_to_bf16(rnd(i as u64 ^ 0x2222)))
            .collect();
        let mut o = vec![0u16; total * NQO * HD];

        let (q_d, k_d, v_d, o_d, qo_d, kv_d);
        unsafe {
            q_d = dev(&q);
            k_d = dev(&k);
            v_d = dev(&v);
            o_d = dev(&o);
            qo_d = dev(&qo_indptr);
            kv_d = dev(&kv_indptr);
        }

        ragged_prefill_bf16_hd256(
            q_d,
            k_d,
            v_d,
            o_d,
            &qo_indptr,
            &kv_indptr,
            qo_d,
            kv_d,
            lens.len() as u32,
            total as u32,
            total as u32,
            NQO as u32,
            NKV as u32,
            HD as u32,
            sm_scale,
            true,
            0,
        )
        .unwrap();
        unsafe {
            assert_eq!(cudaDeviceSynchronize(), 0);
            assert_eq!(
                cudaMemcpy(
                    o.as_mut_ptr() as *mut c_void,
                    o_d as *const c_void,
                    o.len() * 2,
                    D2H
                ),
                0
            );
        }

        // CPU reference: per-request causal GQA attention.
        let group = NQO / NKV;
        let qf = |r: usize, h: usize, d: usize| bf16_to_f32(q[(r * NQO + h) * HD + d]);
        let kf = |r: usize, kh: usize, d: usize| bf16_to_f32(k[(r * NKV + kh) * HD + d]);
        let vf = |r: usize, kh: usize, d: usize| bf16_to_f32(v[(r * NKV + kh) * HD + d]);
        let mut max_rel = 0.0f64;
        let mut worst_cos = 1.0f64;
        for (b, &len) in lens.iter().enumerate() {
            let start = qo_indptr[b] as usize;
            for qi in 0..len {
                for h in 0..NQO {
                    let kh = h / group;
                    let mut scores = vec![0f32; qi + 1];
                    for j in 0..=qi {
                        let mut s = 0.0f32;
                        for d in 0..HD {
                            s += qf(start + qi, h, d) * kf(start + j, kh, d);
                        }
                        scores[j] = s * sm_scale;
                    }
                    let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
                    let mut den = 0.0f32;
                    for s in &mut scores {
                        *s = (*s - mx).exp();
                        den += *s;
                    }
                    let mut out_ref = vec![0f32; HD];
                    for (j, &p) in scores.iter().enumerate() {
                        let w = p / den;
                        for d in 0..HD {
                            out_ref[d] += w * vf(start + j, kh, d);
                        }
                    }
                    let mut dot = 0.0f64;
                    let mut na = 0.0f64;
                    let mut nb = 0.0f64;
                    for d in 0..HD {
                        let g = bf16_to_f32(o[(start + qi) * NQO * HD + h * HD + d]) as f64;
                        let r = out_ref[d] as f64;
                        dot += g * r;
                        na += g * g;
                        nb += r * r;
                        max_rel = max_rel.max((g - r).abs() / (r.abs() + 1e-3));
                    }
                    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
                    worst_cos = worst_cos.min(cos);
                }
            }
        }
        unsafe {
            for p in [q_d, k_d, v_d, o_d, qo_d, kv_d] {
                cudaFree(p as *mut c_void);
            }
        }
        eprintln!("FLASHINFER_RAGGED worst_cos={worst_cos:.6} max_rel={max_rel:.4}");
        assert!(
            worst_cos > 0.99,
            "FlashInfer ragged prefill diverges from CPU ref: cos {worst_cos}"
        );
    }
}
