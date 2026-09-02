// SPDX-License-Identifier: AGPL-3.0-only

//! PLE n-gram row gather: host-side slot resolve / NVMe fault-in
//! (`gather_host`) and the graph-safe kernel half (`gather_embed`).
//! Split from `layer.rs` for the ≤500 LoC cap — a CHILD module (like
//! `aux_state.rs`) because these read `PleLayer`'s private fields.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{NgramRowFormat, PleLayer};
use crate::layers::ngram_embed::NgramTable;
use crate::layers::ops;

impl PleLayer {
    /// Resolve row ids to cache slots and gather them into `self.emb`.
    ///
    /// `T * ngram_heads` rows of `head_dim` land contiguously, which IS the
    /// `[T, ngram_heads * head_dim]` concatenation the projections expect —
    /// so `batched_embed` needs no PLE-specific variant.
    pub(super) fn gather(
        &self,
        ids: &[u64],
        num_tokens: usize,
        heads: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let table_va = self.gather_host(ids, gpu, stream)?;
        self.gather_embed(table_va, num_tokens, heads, gpu, stream)
    }

    /// The HOST half of `gather`: NVMe fault-in + slot upload into the
    /// stable `slots_dev` buffer. Capture-illegal (pageable H2D), so under
    /// CUDA graphs it runs from `prestage` BEFORE replay/capture. Returns
    /// the table's device VA for the kernel half.
    pub(super) fn gather_host(
        &self,
        ids: &[u64],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<u64> {
        let mut table = self
            .table
            .lock()
            .map_err(|_| anyhow::anyhow!("PLE table mutex poisoned"))?;
        let table_va = match &mut *table {
            #[cfg(feature = "cuda")]
            NgramTable::Cached(cache) => {
                // Host resolves row -> slot (the ids are host-side anyway) and
                // faults missing rows off NVMe into the pinned, GPU-addressable
                // arena. The gather kernel then reads the arena BY SLOT.
                let mut slots = Vec::with_capacity(ids.len());
                let (h0, m0, _) = cache.stats();
                let t0 = std::time::Instant::now();
                cache.resolve(ids, &mut slots)?;
                // Prefill-scale gathers log the fault profile at info: the
                // misses are SERIAL blocking preads today (QD=1 under this
                // mutex), so miss-count x latency IS the prefill stall.
                // Decode-scale (16 ids) stays at debug.
                let (h1, m1, _) = cache.stats();
                let (dh, dm) = (h1 - h0, m1 - m0);
                let us = t0.elapsed().as_micros();
                if ids.len() > 64 {
                    tracing::info!(
                        "PLE gather: {} ids, {dh} hits / {dm} misses, resolve {us}us",
                        ids.len()
                    );
                } else {
                    tracing::debug!(
                        "PLE gather: {} ids, {dh} hits / {dm} misses, resolve {us}us",
                        ids.len()
                    );
                }
                let bytes: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
                gpu.copy_h2d_async(&bytes, self.slots_dev, stream)?;
                let va = cache.table_dev_va()?;
                // ⚠ KNOWN. `end_batch`'s contract is "call once the gather has
                // been ISSUED"; this releases the pins before `gather_embed` runs
                // the kernel, which under CUDA graphs is a replay later. A next
                // chunk's `resolve` could evict one of these slots and fault new
                // bytes in from the HOST, which is not stream-ordered, and the
                // in-flight kernel would gather the wrong row. Reaching a
                // just-used slot needs the CLOCK hand around inside one resolve —
                // order 65_536 misses against a 32_768-id chunk: close enough to
                // matter later, not reachable now. Moving the release also changes
                // when pins drop on every error path, and a leaked pin exhausts
                // the cache — worse than the race. `NgramEmbeddings` does it in
                // the documented order; copy that, with a prefill-scale test.
                cache.end_batch();
                DevicePtr(va)
            }
            NgramTable::Bf16(w) => {
                // Fully resident table (small fixtures / tests): the "slot" IS
                // the row id, so upload the ids truncated to u32.
                let bytes: Vec<u8> = ids.iter().flat_map(|r| (*r as u32).to_le_bytes()).collect();
                gpu.copy_h2d_async(&bytes, self.slots_dev, stream)?;
                w.weight
            }
            NgramTable::Fp8(_) => anyhow::bail!(
                "PLE: FP8 n-gram tables are not wired. This checkpoint ships BF16 \
                 rows, which are both simpler and more accurate (on LongCat, BF16 \
                 measured 0.0050 error vs FP8's 0.0247)."
            ),
        };
        Ok(table_va.0)
    }

    /// The KERNEL half of `gather`: reads `slots_dev` and the table arena —
    /// both stable device addresses — so it is graph-capture-safe.
    pub(super) fn gather_embed(
        &self,
        table_va: u64,
        num_tokens: usize,
        heads: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        match self.ngram_format {
            NgramRowFormat::Bf16 => ops::batched_embed(
                gpu,
                self.embed_k,
                self.slots_dev,
                DevicePtr(table_va),
                self.emb,
                (num_tokens * heads) as u32,
                self.head_dim as u32,
                stream,
            )
            .context("PLE row gather"),
            // The arena holds RAW exl3_ngram_trellis rows; the kernel decodes
            // (mul1 + row scale + per-head bias) and writes BF16. Row order is
            // [tokens, heads] contiguous, so the kernel derives the head from
            // row_index % heads.
            NgramRowFormat::Exl3 { k_bits, head_bias } => ops::batched_embed_exl3(
                gpu,
                self.embed_exl3_k,
                self.slots_dev,
                DevicePtr(table_va),
                head_bias,
                self.emb,
                (num_tokens * heads) as u32,
                heads as u32,
                k_bits,
                stream,
            )
            .context("PLE exl3 row gather"),
        }
    }
}
