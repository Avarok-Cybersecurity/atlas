// SPDX-License-Identifier: AGPL-3.0-only
// Translated for Atlas from vLLM (https://github.com/vllm-project/vllm, commit 8a36f08),
// csrc/quantization/marlin/marlin.cu:240-403 (`marlin_mm`: the per-chunk activation and output
// offsets, the launch) and vllm/model_executor/layers/quantization/utils/marlin_utils.py:692-740
// (`apply_gptq_marlin_linear`: the FP32 reduce default, no atomic add). Copyright contributors to the
// vLLM project, Apache-2.0; Copyright (C) Marlin.2024 Elias Frantar, Apache-2.0.
// Changes: an implementation of Atlas's `W4Linear` contract; decode uses the
// row-invariant policy of `config.rs`; `dequant_bf16` is a host reconstruction of the device image.

//! Responsibility: [`MarlinLinear`], the Marlin int4 implementation of `W4Linear`, and [`release`].
//!
//! * `decode` (rows 1..=64): one launch under the decode policy (one configuration per weight, no
//!   `m_block_size_8`), so row `r`'s bits do not depend on `rows`. Capture-safe.
//! * `prefill` (any `m`): vLLM's `marlin_mm` loop (configuration per chunk, `m_block_size_8` for a
//!   tail of at most 8 rows). Declines (`Ok(false)`) only when the scratch cannot hold the FP32 reduce
//!   buffer or no instance serves the shape. Capture-safe.
//! * `dequant_bf16`: reads the device image back, inverts the repack on the host and writes the
//!   `[n, k]` BF16 weight, bit-exact to the kernel's B operand. Host round trip: never captured, used
//!   as the oracle and as the decline fallback.
//!
//! The scratch is `W4Scratch::ws` only (the FP32 reduce buffer); `tickets`, `xq` and `xs` are unused:
//! the locks are the weight's own buffer.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use anyhow::{Result, ensure};
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::config::{self, Chunk};
use super::kernels::{self, KernelKey, LaunchArgs};
use super::repack;
use super::weight::MarlinWeight;
use super::{MAX_DECODE_ROWS, MarlinType};
use crate::port::w4::{W4Box, W4Format, W4Linear, W4Scratch};

/// Which configuration rule a call follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodePolicy {
    /// The weight's fixed decode configuration, no `m_block_size_8` (row invariant).
    RowInvariant,
    /// vLLM's per-M selection (`plan_launches`), as `prefill` uses. Not row invariant.
    Vllm,
}

/// Live Marlin weights by the address of their `MarlinLinear`, for [`release`].
fn registry() -> &'static Mutex<HashMap<usize, MarlinWeight>> {
    static R: OnceLock<Mutex<HashMap<usize, MarlinWeight>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn key_of(w: &dyn W4Linear) -> usize {
    std::ptr::from_ref(w).cast::<()>().addr()
}

/// Free the device buffers of a Marlin `W4Box` and forget it. `Ok(false)` (nothing freed) when `w` is
/// not a live Marlin weight, so a second call is harmless. The caller guarantees no launch still reads
/// the weight and that it is not called again.
pub fn release(gpu: &dyn GpuBackend, w: &dyn W4Linear) -> Result<bool> {
    let Some(d) = registry().lock().remove(&key_of(w)) else {
        return Ok(false);
    };
    d.release(gpu)?;
    Ok(true)
}

/// Forget a Marlin `W4Box` without freeing it: its buffers now belong to someone else (a loader's
/// weight ledger), so a later [`release`] of it is a no-op. False when `w` was not registered.
pub fn forget(w: &dyn W4Linear) -> bool {
    registry().lock().remove(&key_of(w)).is_some()
}

/// The `W4Linear` face of a [`MarlinWeight`].
#[derive(Debug)]
pub struct MarlinLinear {
    w: MarlinWeight,
}

impl MarlinLinear {
    pub fn new(w: MarlinWeight) -> Self {
        Self { w }
    }

    /// Box as a `W4Box` and register for [`release`]. A boxed weight never moves (`W4Box` is an
    /// `Arc`), so its address is a stable key.
    pub fn boxed(self) -> W4Box {
        let w = self.w.clone();
        let arc = Arc::new(self);
        registry()
            .lock()
            .insert(Arc::as_ptr(&arc).cast::<()>().addr(), w);
        arc
    }

    /// The device weight.
    pub fn weight(&self) -> &MarlinWeight {
        &self.w
    }

    /// Free the device buffers of an unboxed weight (the microtest's copies).
    pub fn release_unboxed(self, gpu: &dyn GpuBackend) -> Result<()> {
        self.w.release(gpu)
    }

    /// The launches `rows` rows take under `policy`.
    pub fn plan(&self, rows: usize, policy: DecodePolicy) -> Result<Vec<Chunk>> {
        let w = &self.w;
        match policy {
            DecodePolicy::RowInvariant => Ok(vec![config::decode_chunk(
                rows,
                w.decode_cfg,
                &kernels::DECODE_TMBS,
                w.sms,
                w.max_shared_mem,
            )?]),
            DecodePolicy::Vllm => {
                let avail = |t: usize, m8: bool, c: &config::ThreadConfig| w.kernels.has(t, m8, c);
                config::plan_launches(&w.problem(), rows, w.sms, w.max_shared_mem, &avail)
            }
        }
    }

    /// Run the rows of `x` that `chunks` cover (stream-ordered, capture-safe). The chunks come from
    /// [`MarlinLinear::plan`], or, in the microtest, name one instance each so every entry point is
    /// launched; `s.ws` must hold `scratch_bytes` of the largest chunk.
    pub fn run_plan(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        out: DevicePtr,
        chunks: &[Chunk],
        s: &W4Scratch,
        stream: u64,
    ) -> Result<()> {
        let w = &self.w;
        for c in chunks {
            let key = KernelKey::of(
                w.btype,
                w.problem().group_blocks(),
                c.thread_m_blocks,
                c.m_block_size_8,
                &c.cfg,
            );
            let Some(h) = w.kernels.get(&key) else {
                anyhow::bail!(
                    "marlin: instance {} is not in this build",
                    kernels::entry_name(&key)
                );
            };
            let args = LaunchArgs {
                a: x.offset(c.row_offset * w.k * 2),
                b: w.words,
                c: out.offset(c.row_offset * w.n * 2),
                c_tmp: s.ws,
                scales: w.scales,
                zp: w.zeros,
                prob_m: c.rows,
                prob_n: w.n,
                prob_k: w.k,
                lda: w.k,
                locks: w.locks,
                use_fp32_reduce: true,
            };
            kernels::launch(gpu, h, c.blocks, c.cfg.num_threads, c.smem, &args, stream)?;
        }
        Ok(())
    }

    /// `rows` rows under an explicit policy (the microtest compares the two).
    pub fn run_rows(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        rows: usize,
        out: DevicePtr,
        s: &W4Scratch,
        policy: DecodePolicy,
        stream: u64,
    ) -> Result<()> {
        ensure!(rows > 0, "marlin: a call with no rows");
        check_io(x, out)?;
        let need = self.w.scratch_bytes(rows);
        ensure!(
            !s.ws.is_null() && s.ws_bytes >= need,
            "marlin: [{}x{}] at {rows} rows needs {need} scratch bytes, the workspace has {}",
            self.w.n,
            self.w.k,
            s.ws_bytes
        );
        let chunks = self.plan(rows, policy)?;
        self.run_plan(gpu, x, out, &chunks, s, stream)
    }
}

impl Drop for MarlinLinear {
    fn drop(&mut self) {
        registry()
            .lock()
            .remove(&std::ptr::from_ref(self).cast::<()>().addr());
    }
}

/// The kernel loads A and stores C as 16-byte vectors.
fn check_io(x: DevicePtr, out: DevicePtr) -> Result<()> {
    ensure!(
        x.0.is_multiple_of(16),
        "marlin: the input must be 16-byte aligned, got {:#x}",
        x.0
    );
    ensure!(
        out.0.is_multiple_of(16),
        "marlin: the output must be 16-byte aligned, got {:#x}",
        out.0
    );
    Ok(())
}

impl W4Linear for MarlinLinear {
    fn format(&self) -> W4Format {
        W4Format::MarlinInt4 {
            group: u16::try_from(self.w.group).unwrap_or(u16::MAX),
            zero_point: self.w.btype == MarlinType::U4,
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
        self.w.scratch_bytes(max_rows)
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
            "marlin: decode takes 1..={MAX_DECODE_ROWS} rows, got {rows}"
        );
        self.run_rows(gpu, x, rows, out, s, DecodePolicy::RowInvariant, stream)
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
        ensure!(m > 0, "marlin: a prefill with no rows");
        check_io(x, out)?;
        if s.ws.is_null() || s.ws_bytes < self.w.scratch_bytes(m) {
            tracing::debug!(
                "marlin: prefill of {m} rows declined, the workspace holds {} of {} bytes",
                s.ws_bytes,
                self.w.scratch_bytes(m)
            );
            return Ok(false);
        }
        let chunks = match self.plan(m, DecodePolicy::Vllm) {
            Ok(c) => c,
            Err(e) => {
                tracing::debug!("marlin: prefill of {m} rows declined: {e}");
                return Ok(false);
            }
        };
        self.run_plan(gpu, x, out, &chunks, s, stream)?;
        Ok(true)
    }

    fn dequant_bf16(&self, gpu: &dyn GpuBackend, out: DevicePtr, stream: u64) -> Result<()> {
        let host = self.w.download(gpu, stream)?;
        let logical = repack::from_host(&host)?;
        let nk = logical.dequant_nk();
        let bytes: Vec<u8> = nk.iter().flat_map(|v| v.to_le_bytes()).collect();
        gpu.copy_h2d_async(&bytes, out, stream)?;
        gpu.synchronize(stream)
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}
