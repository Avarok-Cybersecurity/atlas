// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: [`GgufQ4Linear`], the keep-packed GGUF IQ4_XS / Q4_K / Q6_K implementation of `W4Linear`
//! (lane L19), its constructors, and the registry `decode_pair` uses to
//! recognise a GGUF partner.
//!
//! Decode (rows 1..=8): the activation rows are quantized to `block_q8_1` into `W4Scratch::ws`, then one
//! mmvq launch on the raw super-blocks. Prefill (any M): the activations are quantized to the MMQ layout
//! into `ws` and the vendored MMQ engine runs; when `ws` is too small the call declines (`Ok(false)`) and
//! the caller dequantizes ([`W4Linear::dequant_bf16`]) and runs cuBLASLt. `W4Scratch::tickets`, `xq` and
//! `xs` are not used. Row r's bits never depend on the row count: every row is quantized on its own and
//! accumulated in an order fixed by the kernel, not by M.
//!
//! Memory: a weight made by [`from_gguf_tensor`] or [`fuse_rows`] owns its device buffer and [`release`]
//! frees it; a weight made by [`from_device_blocks`] or [`from_weight_tensor`] borrows the weight store's
//! buffer, and [`release`] leaves that buffer to the store.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result, bail, ensure};
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::dequant_cpu::GgmlType;
use spark_runtime::weights::{WeightDtype, WeightTensor};

use super::launch::{self, Kernels, MAX_DECODE_ROWS, MmvqShape, Q4Type, QK_K};
use crate::port::w4::{W4Box, W4Format, W4Linear, W4Scratch};

/// One packed weight: where its blocks are, its shape and its kernels.
#[derive(Clone, Copy, Debug)]
pub struct GgufQ4Weight {
    pub ty: Q4Type,
    /// `[n][k / 256]` raw super-blocks, row-major.
    pub blocks: DevicePtr,
    pub n: usize,
    pub k: usize,
    /// True when this weight allocated `blocks` (freed by [`release`]).
    pub owned: bool,
    pub kernels: Kernels,
}

impl GgufQ4Weight {
    pub fn device_bytes(&self) -> usize {
        launch::weight_bytes(self.ty, self.n, self.k)
    }

    /// The GEMV on activations already quantized to `block_q8_1` rows.
    fn gemv(
        &self,
        gpu: &dyn GpuBackend,
        y_q8: DevicePtr,
        rows: usize,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        launch::mmvq(
            gpu,
            &self.kernels,
            MmvqShape::Block,
            self.blocks,
            y_q8,
            out,
            self.n,
            self.k,
            rows,
            stream,
        )
    }
}

/// The packed type of a ggml type, for the three types this route serves.
pub fn q4_type_of(ty: GgmlType) -> Result<Q4Type> {
    Ok(match ty {
        GgmlType::Iq4Xs => Q4Type::Iq4Xs,
        GgmlType::Q4K => Q4Type::Q4K,
        GgmlType::Q6K => Q4Type::Q6K,
        other => {
            bail!("gguf_q4: ggml type {other:?} is not served keep-packed (IQ4_XS, Q4_K, Q6_K are)")
        }
    })
}

/// The packed type of a weight-store dtype, if it is one of the three.
pub fn q4_type_of_dtype(dtype: WeightDtype) -> Option<Q4Type> {
    match dtype {
        WeightDtype::Iq4Xs => Some(Q4Type::Iq4Xs),
        WeightDtype::Q4K => Some(Q4Type::Q4K),
        WeightDtype::Q6K => Some(Q4Type::Q6K),
        _ => None,
    }
}

/// `n >= 1`, `k` a positive multiple of 256 and every grid dimension within `u32`.
fn check_shape(n: usize, k: usize) -> Result<()> {
    ensure!(n >= 1, "gguf_q4: empty weight (n = 0)");
    ensure!(
        k >= QK_K && k.is_multiple_of(QK_K),
        "gguf_q4: k = {k} is not a positive multiple of {QK_K}"
    );
    ensure!(
        u32::try_from(n * (k / QK_K)).is_ok(),
        "gguf_q4: [{n}, {k}] has more super-blocks than one launch addresses"
    );
    Ok(())
}

/// The `W4Linear` face of a [`GgufQ4Weight`].
#[derive(Debug)]
pub struct GgufQ4Linear {
    w: GgufQ4Weight,
}

/// Live GGUF weights by the address of their `GgufQ4Linear` (the data pointer of the `W4Box`).
fn registry() -> &'static Mutex<HashMap<usize, GgufQ4Weight>> {
    static R: OnceLock<Mutex<HashMap<usize, GgufQ4Weight>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn key_of(w: &dyn W4Linear) -> usize {
    w as *const dyn W4Linear as *const () as usize
}

/// The descriptor behind `w`, if `w` is a live [`GgufQ4Linear`].
pub fn gguf_of(w: &dyn W4Linear) -> Option<GgufQ4Weight> {
    registry().lock().get(&key_of(w)).copied()
}

impl GgufQ4Linear {
    /// Box `w` as a `W4Box` and register it as a `decode_pair` partner.
    pub fn boxed(w: GgufQ4Weight) -> W4Box {
        let arc = Arc::new(Self { w });
        registry()
            .lock()
            .insert(Arc::as_ptr(&arc) as *const () as usize, w);
        arc
    }

    pub fn weight(&self) -> &GgufQ4Weight {
        &self.w
    }
}

impl Drop for GgufQ4Linear {
    fn drop(&mut self) {
        registry()
            .lock()
            .remove(&(self as *const Self as *const () as usize));
    }
}

/// Free the device buffer of an owned GGUF `W4Box` and forget it. Returns `false` (and frees nothing) when
/// `w` is not a live GGUF weight. A borrowed weight is forgotten but its store buffer is left alone. The
/// caller guarantees no launch still reads the weight and that it is not used again.
pub fn release(gpu: &dyn GpuBackend, w: &dyn W4Linear) -> Result<bool> {
    let Some(d) = registry().lock().remove(&key_of(w)) else {
        return Ok(false);
    };
    if d.owned {
        gpu.free(d.blocks)?;
    }
    Ok(true)
}

/// Upload a tensor's raw GGUF bytes (`[n, k]`, row-major super-blocks, as the file stores them) and wrap
/// them. The upload is complete when this returns (`bytes` may be dropped).
pub fn from_gguf_tensor(
    gpu: &dyn GpuBackend,
    ty: GgmlType,
    bytes: &[u8],
    n: usize,
    k: usize,
    stream: u64,
) -> Result<W4Box> {
    let q = q4_type_of(ty)?;
    check_shape(n, k)?;
    let want = launch::weight_bytes(q, n, k);
    ensure!(
        bytes.len() == want,
        "gguf_q4: {q:?} [{n}, {k}] needs {want} bytes, got {}",
        bytes.len()
    );
    let kernels = Kernels::resolve(gpu, q)?;
    let blocks = gpu.alloc(want)?;
    let upload = gpu
        .copy_h2d_async(bytes, blocks, stream)
        .and_then(|()| gpu.synchronize(stream));
    if let Err(e) = upload {
        gpu.free(blocks)?;
        return Err(e.context("gguf_q4: weight upload"));
    }
    Ok(GgufQ4Linear::boxed(GgufQ4Weight {
        ty: q,
        blocks,
        n,
        k,
        owned: true,
        kernels,
    }))
}

/// Wrap raw blocks already on the device (for example the weight store's keep-packed tensor). The weight
/// borrows `blocks`; the caller keeps it alive for the weight's life.
pub fn from_device_blocks(
    gpu: &dyn GpuBackend,
    ty: GgmlType,
    blocks: DevicePtr,
    n: usize,
    k: usize,
) -> Result<W4Box> {
    let q = q4_type_of(ty)?;
    check_shape(n, k)?;
    ensure!(!blocks.is_null(), "gguf_q4: null weight pointer");
    let kernels = Kernels::resolve(gpu, q)?;
    Ok(GgufQ4Linear::boxed(GgufQ4Weight {
        ty: q,
        blocks,
        n,
        k,
        owned: false,
        kernels,
    }))
}

/// The weight store's keep-packed tensor (`WeightDtype::{Iq4Xs, Q4K, Q6K}`, HF shape `[n, k]`) as a
/// `W4Box` borrowing the store's buffer. `Ok(None)` for any other dtype.
pub fn from_weight_tensor(gpu: &dyn GpuBackend, t: &WeightTensor) -> Result<Option<W4Box>> {
    let Some(q) = q4_type_of_dtype(t.dtype) else {
        return Ok(None);
    };
    let &[n, k] = t.shape.as_slice() else {
        bail!("gguf_q4: packed tensor shape {:?} is not 2-D", t.shape);
    };
    let ty = match q {
        Q4Type::Iq4Xs => GgmlType::Iq4Xs,
        Q4Type::Q4K => GgmlType::Q4K,
        Q4Type::Q6K => GgmlType::Q6K,
    };
    from_device_blocks(gpu, ty, t.ptr, n, k).map(Some)
}

/// Concatenate packed `[n_i, k]` row blocks of one type into one owned `[sum n_i, k]` weight (the fused
/// GDN `in_proj_qkvz` of `in_proj_qkv` and `in_proj_z`, or a fused gate+up). Rows are whole super-blocks,
/// so the result is the bytes a fused tensor would have had on disk. Copies are device to device.
pub fn fuse_rows(
    gpu: &dyn GpuBackend,
    ty: GgmlType,
    parts: &[(DevicePtr, usize)],
    k: usize,
    stream: u64,
) -> Result<W4Box> {
    let q = q4_type_of(ty)?;
    ensure!(!parts.is_empty(), "gguf_q4: fuse_rows of nothing");
    let n: usize = parts.iter().map(|p| p.1).sum();
    check_shape(n, k)?;
    let kernels = Kernels::resolve(gpu, q)?;
    let blocks = gpu.alloc(launch::weight_bytes(q, n, k))?;
    let mut off = 0usize;
    for &(src, rows) in parts {
        let bytes = launch::weight_bytes(q, rows, k);
        if let Err(e) = gpu.copy_d2d_async(src, blocks.offset(off), bytes, stream) {
            gpu.free(blocks)?;
            return Err(e.context("gguf_q4: fuse_rows copy"));
        }
        off += bytes;
    }
    gpu.synchronize(stream)?;
    Ok(GgufQ4Linear::boxed(GgufQ4Weight {
        ty: q,
        blocks,
        n,
        k,
        owned: true,
        kernels,
    }))
}

/// `ws` holds at least `need` bytes.
fn ws_holds(s: &W4Scratch, need: usize) -> bool {
    !s.ws.is_null() && s.ws_bytes >= need
}

impl W4Linear for GgufQ4Linear {
    fn format(&self) -> W4Format {
        match self.w.ty {
            Q4Type::Iq4Xs => W4Format::GgufIq4Xs,
            Q4Type::Q4K => W4Format::GgufQ4K,
            Q4Type::Q6K => W4Format::GgufQ6K,
        }
    }

    fn n(&self) -> usize {
        self.w.n
    }

    fn k(&self) -> usize {
        self.w.k
    }

    fn device_bytes(&self) -> usize {
        self.w.device_bytes()
    }

    fn max_decode_rows(&self) -> usize {
        MAX_DECODE_ROWS
    }

    fn scratch_bytes(&self, max_rows: usize) -> usize {
        launch::scratch_bytes(self.w.k, max_rows)
    }

    fn decode(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        rows: usize,
        out: DevicePtr,
        s: &W4Scratch,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            (1..=MAX_DECODE_ROWS).contains(&rows),
            "gguf_q4 decode: rows = {rows} outside 1..={MAX_DECODE_ROWS}"
        );
        let need = launch::q8_rows_bytes(rows, self.w.k);
        ensure!(
            ws_holds(s, need),
            "gguf_q4 decode: workspace holds {} bytes, {need} needed",
            s.ws_bytes
        );
        launch::q8_rows(gpu, &self.w.kernels, x, s.ws, rows, self.w.k, stream)?;
        self.w.gemv(gpu, s.ws, rows, out, stream)
    }

    fn prefill(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        m: usize,
        out: DevicePtr,
        s: &W4Scratch,
        stream: u64,
    ) -> Result<bool> {
        ensure!(m >= 1, "gguf_q4 prefill: m = 0");
        if !ws_holds(s, launch::mmq_act_bytes(m, self.w.k)) {
            // The caller dequantizes and runs its BF16 GEMM instead.
            return Ok(false);
        }
        let k = self.w.kernels;
        launch::mmq_act(gpu, &k, x, s.ws, m, self.w.k, stream)?;
        launch::mmq_gemm(
            gpu,
            &k,
            s.ws,
            self.w.blocks,
            out,
            m,
            self.w.n,
            self.w.k,
            stream,
        )?;
        Ok(true)
    }

    fn decode_pair(
        &self,
        other: &dyn W4Linear,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        rows: usize,
        out_a: DevicePtr,
        out_b: DevicePtr,
        s: &W4Scratch,
        stream: u64,
    ) -> Result<()> {
        // A GGUF partner over the same input width shares one activation quantization; the bytes of both
        // outputs are those of two `decode` calls, since the quantization is a pure function of x.
        match gguf_of(other) {
            Some(o) if o.k == self.w.k && (1..=MAX_DECODE_ROWS).contains(&rows) => {
                let need = launch::q8_rows_bytes(rows, self.w.k);
                ensure!(
                    ws_holds(s, need),
                    "gguf_q4 decode_pair: workspace holds {} bytes, {need} needed",
                    s.ws_bytes
                );
                launch::q8_rows(gpu, &self.w.kernels, x, s.ws, rows, self.w.k, stream)?;
                self.w.gemv(gpu, s.ws, rows, out_a, stream)?;
                o.gemv(gpu, s.ws, rows, out_b, stream)
            }
            _ => {
                self.decode(gpu, x, rows, out_a, s, stream)?;
                other.decode(gpu, x, rows, out_b, s, stream)
            }
        }
    }

    fn dequant_bf16(&self, gpu: &dyn GpuBackend, out: DevicePtr, stream: u64) -> Result<()> {
        launch::dequant(
            gpu,
            &self.w.kernels,
            self.w.ty,
            self.w.blocks,
            out,
            self.w.n,
            self.w.k,
            stream,
        )
        .context("gguf_q4 dequant_bf16")
    }
}
