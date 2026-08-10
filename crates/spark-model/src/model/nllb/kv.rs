// SPDX-License-Identifier: AGPL-3.0-only

//! Per-sequence KV for the served NLLB model. An encoder-decoder sequence owns
//! two kinds of KV, neither of which fits the scheduler's paged block cache:
//! the **cross-attention K/V** (computed once from the encoder over the source,
//! fixed size `enc_len·d`, reused unchanged every decode step) and the **decoder
//! self-attention K/V** (grows one row per decoded token, capped at
//! `cache_rows`). Both live here, keyed by `SequenceState.slot_idx`, allocated
//! at `alloc_sequence`/`prefill` and freed at `free_sequence`.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// One sequence's decoder KV (self-attn, growing) + cross-attn KV (fixed).
pub(super) struct NllbSeqKv {
    /// Decoder self-attention K per layer, `[cache_rows, d]` bf16.
    pub self_k: Vec<DevicePtr>,
    pub self_v: Vec<DevicePtr>,
    /// Cross-attention K/V per layer, `[enc_len, d]` bf16 (empty until prefill).
    pub cross_k: Vec<DevicePtr>,
    pub cross_v: Vec<DevicePtr>,
    /// Encoder length behind the cross-KV (`0` until prefill fills it).
    pub enc_len: usize,
    /// Next decoder row to write (number of decoder tokens processed so far).
    pub dec_pos: usize,
    cache_rows: usize,
}

impl NllbSeqKv {
    /// Allocate the fixed-size decoder self-attn KV; cross-KV stays empty until
    /// `fill_cross` runs at prefill (its size depends on the source length).
    pub(super) fn new(
        gpu: &dyn GpuBackend,
        dec_layers: usize,
        cache_rows: usize,
        d: usize,
    ) -> Result<Self> {
        let mut self_k = Vec::with_capacity(dec_layers);
        let mut self_v = Vec::with_capacity(dec_layers);
        for _ in 0..dec_layers {
            self_k.push(gpu.alloc(cache_rows * d * 2)?);
            self_v.push(gpu.alloc(cache_rows * d * 2)?);
        }
        Ok(Self {
            self_k,
            self_v,
            cross_k: Vec::new(),
            cross_v: Vec::new(),
            enc_len: 0,
            dec_pos: 0,
            cache_rows,
        })
    }

    /// (Re)allocate the cross-attn KV for a source of `enc_len` rows. Frees any
    /// previous cross buffers first (a slot may be reused across requests).
    pub(super) fn alloc_cross(
        &mut self,
        gpu: &dyn GpuBackend,
        dec_layers: usize,
        enc_len: usize,
        d: usize,
    ) -> Result<()> {
        self.free_cross(gpu)?;
        for _ in 0..dec_layers {
            self.cross_k.push(gpu.alloc(enc_len * d * 2)?);
            self.cross_v.push(gpu.alloc(enc_len * d * 2)?);
        }
        self.enc_len = enc_len;
        self.dec_pos = 0;
        Ok(())
    }

    /// Guard: the self-attn KV cache cannot hold another decoder row.
    pub(super) fn ensure_room(&self) -> Result<()> {
        if self.dec_pos >= self.cache_rows {
            bail!(
                "nllb: decoder length exceeded cache_rows={} — raise --max-model-len",
                self.cache_rows
            );
        }
        Ok(())
    }

    fn free_cross(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        for p in self.cross_k.drain(..).chain(self.cross_v.drain(..)) {
            gpu.free(p)?;
        }
        self.enc_len = 0;
        Ok(())
    }

    /// Free every device buffer this sequence owns.
    pub(super) fn free(mut self, gpu: &dyn GpuBackend) -> Result<()> {
        self.free_cross(gpu)?;
        for p in self.self_k.drain(..).chain(self.self_v.drain(..)) {
            gpu.free(p)?;
        }
        Ok(())
    }
}
