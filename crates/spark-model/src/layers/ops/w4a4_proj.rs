// SPDX-License-Identifier: AGPL-3.0-only

//! OPT-IN W4A4 small-M projection on FP4 tensor cores (`w4a4_gemv_mx.cu`,
//! module `w4a4_gemv_mx`), under `--w4a4-downcast true` (default false).
//!
//! # Why
//!
//! bf16 tensor-core MMA power on GB10 scales with LIVE rows (TEP 2026-09-23:
//! `w4a16_gemv_tc8` draws 46/51/59 W at 1/4/8 rows). The block-scale FP4 MMA
//! (`kind::mxf4nvf4`) consumes the checkpoint's NVFP4 weight bytes and E4M3
//! group scales directly, with no dequant, and moves 4x fewer operand bits per
//! MAC. vLLM runs its NVFP4 layers this way at every M. The activations are
//! quantised per row to NVFP4 first (dynamic per-row global scale; see the
//! .cu header), so the numerics become W4A4. That is a real accuracy change,
//! which is why this is opt-in.
//!
//! # Scope
//!
//! Projection sites only, via [`nvfp4_proj_small_m`]: GDN qkvz/out_proj,
//! attention q/k/v/o, dense-FFN gate/up/down at 1..=32 rows. The lm_head keeps
//! `w4a16_gemv_batchm`, because logits are the most quantisation-sensitive
//! output.
//!
//! # Scratch
//!
//! The quantised activations need `[M, K/2] + [M, K/16] + [M] f32` of device
//! scratch. One buffer is allocated per backend by [`prepare`] at model build
//! (called from `W4a16BatchmTiers::resolve`), never on the decode path, so a
//! CUDA-graph capture never sees an allocation. Launches are stream-ordered
//! on the caller's stream. Two projections on DIFFERENT streams at once would
//! race on it, and no dense-27B site does that.

use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

/// Rows the widest entry (`w4a4_gemv_mx32`) covers.
pub const W4A4_MAX_M: u32 = 32;
/// Largest K the scratch is sized for (every 27B projection is <= 17408).
pub const W4A4_MAX_K: u32 = 32768;

/// `--w4a4-downcast`: THE one reader of the W4A4 activation-downcast choice.
/// It covers both halves: the projection path in this module, and the
/// dense-FFN small-M steps on the W4A4 MMQ (`DenseFfn::forward_k2/k3/km`).
/// There is NO environment fallback. The serve publishes it before the model
/// is built, and anything that reads it first (a test, an example) sees the
/// default, false.
static W4A4_DOWNCAST: OnceLock<bool> = OnceLock::new();

/// Publish `--w4a4-downcast`. Returns the value in force: the first
/// publication wins, and a caller seeing a different value must warn.
pub fn set_w4a4_downcast_from_cli(on: bool) -> bool {
    let _ = W4A4_DOWNCAST.set(on);
    *W4A4_DOWNCAST.get().expect("just set")
}

/// `--w4a4-downcast` in force? False unless the serve published true.
pub fn w4a4_downcast_enabled() -> bool {
    *W4A4_DOWNCAST.get_or_init(|| false)
}

#[derive(Clone, Copy)]
struct W4a4State {
    quant: KernelHandle,
    mx8: KernelHandle,
    mx16: KernelHandle,
    mx32: KernelHandle,
    aq: DevicePtr,
    a_scale: DevicePtr,
    a_gs: DevicePtr,
    /// W4A16 reference output for [`audit_enabled`] (null otherwise).
    audit_ref: DevicePtr,
}

/// Largest projection N the audit buffer holds (the FFN gate+up, 34816).
const AUDIT_MAX_N: usize = 65536;

/// `AVAROK_W4A4_PROJ_AUDIT` (non-empty): DIAGNOSTIC ONLY. Every eager (not
/// graph-captured) W4A4 projection also runs the W4A16 path, synchronises,
/// and accumulates `||y_w4a4 - y_w4a16|| / ||y_w4a16||` per CALL SITE, logged
/// every 64 samples. It costs a sync per launch, so run it only with graphs
/// off (`AVAROK_NO_MTP_VERIFY_GRAPHS=1 AVAROK_NO_DECODE_GRAPHS_MULTISEQ=1`).
fn audit_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("AVAROK_W4A4_PROJ_AUDIT").is_some_and(|v| !v.is_empty()))
}

fn cache() -> &'static Mutex<Vec<(usize, Option<W4a4State>)>> {
    static CACHE: OnceLock<Mutex<Vec<(usize, Option<W4a4State>)>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn key(gpu: &dyn GpuBackend) -> usize {
    gpu as *const dyn GpuBackend as *const () as usize
}

/// Resolve the kernels and allocate the scratch ONCE per backend, when the
/// lever is on. Call at model build, never inside a graph capture.
pub fn prepare(gpu: &dyn GpuBackend) -> Result<()> {
    if !w4a4_downcast_enabled() {
        return Ok(());
    }
    let mut guard = cache().lock().unwrap_or_else(|p| p.into_inner());
    if guard.iter().any(|(k, _)| *k == key(gpu)) {
        return Ok(());
    }
    let h = |f: &str| crate::layers::try_kernel(gpu, "w4a4_gemv_mx", f);
    let (quant, mx8, mx16, mx32) = (
        h("w4a4_quant_rows"),
        h("w4a4_gemv_mx8"),
        h("w4a4_gemv_mx16"),
        h("w4a4_gemv_mx32"),
    );
    let state = if [quant, mx8, mx16, mx32].iter().all(|k| k.0 != 0) {
        let (m, k) = (W4A4_MAX_M as usize, W4A4_MAX_K as usize);
        Some(W4a4State {
            quant,
            mx8,
            mx16,
            mx32,
            aq: gpu.alloc(m * k / 2)?,
            a_scale: gpu.alloc(m * k / 16)?,
            a_gs: gpu.alloc(m * 4)?,
            audit_ref: if audit_enabled() {
                gpu.alloc(m * AUDIT_MAX_N * 2)?
            } else {
                DevicePtr::NULL
            },
        })
    } else {
        tracing::warn!(
            "--w4a4-downcast is on but the w4a4_gemv_mx kernels are not in this target; projections stay W4A16"
        );
        None
    };
    guard.push((key(gpu), state));
    Ok(())
}

fn state(gpu: &dyn GpuBackend) -> Option<W4a4State> {
    let guard = cache().lock().unwrap_or_else(|p| p.into_inner());
    guard
        .iter()
        .find(|(k, _)| *k == key(gpu))
        .and_then(|(_, s)| *s)
}

/// Row edge of the projection sites' narrow arms (GDN qkvz/out_proj, both
/// dense-FFN arms): 32 under `--w4a4-downcast` (the `w4a4_gemv_mx32` reach,
/// with `w4a16_gemv_batch16/32` as the W4A16 fallback handles), else the
/// W4A16 edge [`super::gemv_tc::narrow_gemv_max_rows`]. The lm_head verify
/// arm deliberately keeps the W4A16 edge.
pub fn proj_max_rows() -> u32 {
    if w4a4_downcast_enabled() {
        W4A4_MAX_M
    } else {
        super::gemv_tc::narrow_gemv_max_rows()
    }
}

/// PURE: may the W4A4 path serve this launch?
pub fn w4a4_route(m: u32, n: u32, k: u32, enabled: bool) -> bool {
    enabled
        && (1..=W4A4_MAX_M).contains(&m)
        && n > 0
        && k > 0
        && k.is_multiple_of(64)
        && k <= W4A4_MAX_K
}

/// The projection launcher: W4A4 FP4 MMA when opted in and prepared, else
/// the W4A16 `w4a16_gemv_batchm` (which itself prefers the tensor-core GEMV).
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub fn nvfp4_proj_small_m(
    gpu: &dyn GpuBackend,
    batch_kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    proj(
        gpu,
        batch_kernel,
        input,
        weight,
        output,
        m,
        n,
        k,
        stream,
        false,
    )
}

/// [`nvfp4_proj_small_m`] for a projection whose `input` is BYTE-FOR-BYTE the
/// input of the immediately preceding projection on this stream (attention
/// k/v after q, FFN up after gate). It skips re-quantising when the previous
/// W4A4 launch quantised exactly `(input, m, k)` on this stream, and otherwise
/// quantises as usual, so a wrong claim about the ADDRESS can never read a
/// stale quantisation. The CALLER guarantees the CONTENTS did not change in
/// between.
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub fn nvfp4_proj_small_m_same_input(
    gpu: &dyn GpuBackend,
    batch_kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    proj(
        gpu,
        batch_kernel,
        input,
        weight,
        output,
        m,
        n,
        k,
        stream,
        true,
    )
}

/// What the scratch currently holds: (backend, input address, m, k, stream).
type QuantKey = (usize, u64, u32, u32, u64);

fn last_quant() -> &'static Mutex<Option<QuantKey>> {
    static LAST: OnceLock<Mutex<Option<QuantKey>>> = OnceLock::new();
    LAST.get_or_init(|| Mutex::new(None))
}

#[allow(clippy::too_many_arguments)]
#[track_caller]
fn proj(
    gpu: &dyn GpuBackend,
    batch_kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
    same_input: bool,
) -> Result<()> {
    if w4a4_route(m, n, k, w4a4_downcast_enabled())
        && let Some(s) = state(gpu)
    {
        let want: QuantKey = (key(gpu), input.0, m, k, stream);
        let mut last = last_quant().lock().unwrap_or_else(|p| p.into_inner());
        if !(same_input && *last == Some(want)) {
            KernelLaunch::new(gpu, s.quant)
                .grid([m, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(input)
                .arg_ptr(s.aq)
                .arg_ptr(s.a_scale)
                .arg_ptr(s.a_gs)
                .arg_u32(k)
                .launch(stream)?;
            *last = Some(want);
        }
        drop(last);
        let mx = if m <= 8 {
            s.mx8
        } else if m <= 16 {
            s.mx16
        } else {
            s.mx32
        };
        KernelLaunch::new(gpu, mx)
            .grid([div_ceil(n, 16), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.aq)
            .arg_ptr(s.a_scale)
            .arg_ptr(s.a_gs)
            .arg_ptr(weight.weight)
            .arg_ptr(weight.weight_scale)
            .arg_f32(weight.weight_scale_2)
            .arg_ptr(output)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream)?;
        if audit_enabled()
            && !s.audit_ref.is_null()
            && (n as usize) <= AUDIT_MAX_N
            && !gpu.stream_is_capturing(stream)
        {
            let site = std::panic::Location::caller();
            super::w4a16_gemv_batchm(
                gpu,
                batch_kernel,
                input,
                weight,
                s.audit_ref,
                m,
                n,
                k,
                stream,
            )?;
            audit(
                gpu,
                output,
                s.audit_ref,
                (m * n) as usize,
                stream,
                site,
                n,
                k,
            )?;
        }
        return Ok(());
    }
    super::w4a16_gemv_batchm(gpu, batch_kernel, input, weight, output, m, n, k, stream)
}

/// Per-site relative-error accumulator for [`audit_enabled`].
#[allow(clippy::too_many_arguments)]
fn audit(
    gpu: &dyn GpuBackend,
    got: DevicePtr,
    reference: DevicePtr,
    elems: usize,
    stream: u64,
    site: &'static std::panic::Location<'static>,
    n: u32,
    k: u32,
) -> Result<()> {
    type Acc = std::collections::HashMap<String, (u64, f64, f64)>;
    static ACC: OnceLock<Mutex<Acc>> = OnceLock::new();
    gpu.synchronize(stream)?;
    let mut a = vec![0u8; elems * 2];
    let mut b = vec![0u8; elems * 2];
    gpu.copy_d2h(got, &mut a)?;
    gpu.copy_d2h(reference, &mut b)?;
    let bf = |x: &[u8]| f32::from_bits((u16::from_le_bytes([x[0], x[1]]) as u32) << 16) as f64;
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
        let (x, y) = (bf(x), bf(y));
        num += (x - y) * (x - y);
        den += y * y;
    }
    let rel = if den > 0.0 { (num / den).sqrt() } else { 0.0 };
    let key = format!("{}:{} N={n} K={k}", site.file(), site.line());
    let mut acc = ACC
        .get_or_init(|| Mutex::new(Acc::new()))
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let e = acc.entry(key.clone()).or_insert((0, 0.0, 0.0));
    e.0 += 1;
    e.1 += rel;
    e.2 = e.2.max(rel);
    if e.0.is_multiple_of(64) {
        tracing::info!(
            "W4A4_AUDIT site={key} samples={} mean_rel={:.5} max_rel={:.5}",
            e.0,
            e.1 / e.0 as f64,
            e.2
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_every_27b_projection_shape_up_to_32_rows() {
        for k in [5120u32, 6144, 17408] {
            for m in 1..=32 {
                assert!(w4a4_route(m, 5120, k, true), "m={m} k={k}");
            }
            assert!(!w4a4_route(33, 5120, k, true), "33 rows exceed mx32");
            assert!(!w4a4_route(4, 5120, k, false), "opt-in");
        }
    }

    #[test]
    fn declines_what_the_kernel_or_scratch_cannot_hold() {
        assert!(!w4a4_route(4, 5120, 5120 + 32, true), "K % 64");
        assert!(!w4a4_route(4, 5120, W4A4_MAX_K + 64, true), "scratch K");
        assert!(!w4a4_route(0, 5120, 5120, true));
    }
}
