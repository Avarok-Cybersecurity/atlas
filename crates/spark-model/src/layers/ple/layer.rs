// SPDX-License-Identifier: AGPL-3.0-only

//! The PLE layer: ids -> NVMe row gather -> projections -> gate -> dilated
//! conv -> highway add.
//!
//! Runs on ONE model layer (layer 1 here) and injects into the `hc_mult`-wide
//! hyper-connection highway BEFORE that layer's attention hyper-connection,
//! matching `Qwen4ExpTextDecoderLayer.forward`'s
//! `hidden_states = hidden_states + self.ple(...)`.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::ids::{PleIdDims, ple_ngram_ids};
use crate::layer::ForwardContext;
use crate::layers::ngram_embed::NgramTable;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// How the n-gram table's arena rows decode into BF16 embedding rows.
#[derive(Debug, Clone, Copy)]
pub enum NgramRowFormat {
    /// Rows are plain BF16 `[head_dim]` — the historical layout; the plain
    /// `batched_embed` gather copies them.
    Bf16,
    /// Rows are packed `exl3_ngram_trellis` (`1 + 160*K/16` u16 words:
    /// fp16 row scale + mul1-codebook trellis ring); `batched_embed_exl3`
    /// decodes on gather and adds the per-head bias.
    Exl3 {
        k_bits: u32,
        /// Device `[num_heads, 160]` f16 bias, added after the row scale.
        head_bias: DevicePtr,
    },
}

/// Per-SEQUENCE carry: the dilated conv's 9 steps and the token history the
/// id hash needs. Owned by the sequence's [`crate::layer::SsmLayerState`]
/// (Avarok #753 item B: concurrency needs one of these per in-flight
/// sequence, not a layer singleton).
pub struct PleSeqState {
    /// `[(k-1)*dilation, channels]` FP32, device.
    conv: DevicePtr,
    /// The last `context_len` token ids, EOS-filled at a sequence start.
    history: Vec<u32>,
    /// Set by `prestage`: the n-gram table's device VA, recorded when the
    /// step's host work (hash + fault-in + slot upload) already ran BEFORE
    /// graph replay/capture. `forward` consumes it and enqueues kernels only.
    prestaged_va: Option<u64>,
    /// The last VA `prestage` staged, never cleared. `rearm` restores it when
    /// a failed capture attempt re-runs the step eagerly: the slots are still
    /// in `slots_dev` and history has already advanced, so re-hashing would
    /// double-count the token — re-arming is the only correct recovery.
    last_staged_va: u64,
}

pub struct PleLayer {
    dims: PleIdDims,
    head_dim: usize,
    hidden: usize,
    hc_mult: usize,
    state_len: usize,
    k_size: usize,
    dilation: usize,
    eps: f32,

    key_proj: DenseWeight,
    value_proj: DenseWeight,
    norm_key: DenseWeight,
    norm_query: DenseWeight,
    norm_conv: DenseWeight,
    conv1d: DenseWeight,
    /// Behind a mutex because the NVMe cache RESOLVES (and faults, and
    /// evicts) on the forward path, which needs `&mut`, while layers are
    /// invoked through `&self`.
    table: std::sync::Mutex<NgramTable>,

    embed_k: KernelHandle,
    embed_exl3_k: KernelHandle,
    gemm_k: KernelHandle,
    gate_k: KernelHandle,
    conv_k: KernelHandle,
    add_k: KernelHandle,

    /// How the arena's row bytes decode into BF16 embedding rows.
    ngram_format: NgramRowFormat,

    /// Scratch, sized once for `max_tokens`.
    emb: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gated: DevicePtr,
    gated_normed: DevicePtr,
    out: DevicePtr,
    slots_dev: DevicePtr,
    max_tokens: usize,
}

impl PleLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dims: PleIdDims,
        head_dim: usize,
        hidden: usize,
        hc_mult: usize,
        k_size: usize,
        dilation: usize,
        eps: f32,
        weights: PleWeights,
        table: NgramTable,
        ngram_format: NgramRowFormat,
        max_tokens: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        dims.validate()?;
        let heads = dims.ngram_heads();
        anyhow::ensure!(
            heads * head_dim == hidden,
            "PLE: {heads} heads x {head_dim} dims = {} != ple_embed_dim {hidden}. \
             The head slices are CONCATENATED (not summed as LongCat's are), so \
             this product is the embedding width and a mismatch means the \
             geometry is not what we think.",
            heads * head_dim
        );
        if let NgramRowFormat::Exl3 { k_bits, head_bias } = ngram_format {
            anyhow::ensure!(
                (1..=8).contains(&k_bits) && head_bias != DevicePtr::NULL,
                "PLE: exl3 ngram format needs K in 1..=8 (got {k_bits}) and a \
                 head_bias tensor"
            );
            anyhow::ensure!(
                head_dim == 160,
                "PLE: exl3_ngram_trellis rows are fixed at 160 dims, checkpoint \
                 says head_dim={head_dim}"
            );
        }
        let c = hc_mult * hidden;
        let state_len = (k_size - 1) * dilation;
        Ok(Self {
            dims,
            head_dim,
            hidden,
            hc_mult,
            state_len,
            k_size,
            dilation,
            eps,
            key_proj: weights.key_proj,
            value_proj: weights.value_proj,
            norm_key: weights.norm_key,
            norm_query: weights.norm_query,
            norm_conv: weights.norm_conv,
            conv1d: weights.conv1d,
            table: std::sync::Mutex::new(table),
            ngram_format,
            embed_k: gpu.kernel("embed_from_argmax", "batched_embed")?,
            embed_exl3_k: gpu.kernel("embed_from_argmax", "batched_embed_exl3")?,
            gemm_k: gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?,
            gate_k: gpu.kernel("ple", "ple_gate")?,
            conv_k: gpu.kernel("ple", "ple_conv")?,
            add_k: gpu.kernel("ple", "ple_add_highway")?,
            emb: gpu.alloc(max_tokens * hidden * 2)?,
            key: gpu.alloc(max_tokens * c * 2)?,
            value: gpu.alloc(max_tokens * hidden * 2)?,
            gated: gpu.alloc(max_tokens * c * 4)?,
            gated_normed: gpu.alloc(max_tokens * c * 4)?,
            out: gpu.alloc(max_tokens * c * 4)?,
            slots_dev: gpu.alloc(max_tokens * heads * 4)?,
            max_tokens,
        })
    }

    /// Allocate one sequence's PLE carry (conv buffer + empty history).
    /// `reset` runs on first use (`fresh`), so contents start undefined.
    pub fn new_seq_state(&self, gpu: &dyn GpuBackend) -> Result<PleSeqState> {
        Ok(PleSeqState {
            conv: gpu.alloc(self.state_len * self.hc_mult * self.hidden * 4)?,
            history: Vec::new(),
            prestaged_va: None,
            last_staged_va: 0,
        })
    }

    /// Release one sequence's PLE carry.
    ///
    /// Same shape of defect as the QSA indexer carry: `conv` is a bare
    /// `DevicePtr`, so dropping `PleSeqState` frees nothing. Individually
    /// small (~147 KB) and below the 32 MB allocation-trace threshold, which
    /// is exactly why it stayed invisible — but it is one per SSM layer (36
    /// on qwen4_exp) per sequence, and it never comes back.
    ///
    /// Idempotent: `conv` is nulled once freed.
    pub fn release_seq_state(&self, st: &mut PleSeqState, gpu: &dyn GpuBackend) -> Result<()> {
        if st.conv.is_null() {
            return Ok(());
        }
        let r = gpu.free(st.conv);
        st.conv = DevicePtr(0);
        st.history.clear();
        st.prestaged_va = None;
        st.last_staged_va = 0;
        r
    }

    // `reset` + `prestage` live in `aux_state.rs` (≤500 LoC split).

    // Marconi aux-state (snapshot_aux / restore_aux) moved to
    // `aux_state.rs` (≤500 LoC split).

    /// Restore the prestaged state after a failed CUDA-graph capture attempt
    /// (the eager replay re-runs `forward`, which consumed `prestaged_va`).
    pub fn rearm(&self, st: &mut PleSeqState) {
        if st.last_staged_va != 0 {
            st.prestaged_va = Some(st.last_staged_va);
        }
    }

    /// Inject into `highway` `[T, hc_mult*hidden]` FP32, in place.
    ///
    /// `fresh` starts a new sequence (prefill from position 0).
    /// One highway ROW with an explicit id — the multi-seq decode entry
    /// (`ctx.host_token_ids` holds the whole batch; the caller slices).
    pub fn forward_row(
        &self,
        st: &mut PleSeqState,
        highway_row: DevicePtr,
        ids: &[u32],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_with_ids(st, highway_row, 1, false, Some(ids), ctx, stream)
    }

    pub fn forward(
        &self,
        st: &mut PleSeqState,
        highway: DevicePtr,
        num_tokens: usize,
        fresh: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_with_ids(st, highway, num_tokens, fresh, None, ctx, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with_ids(
        &self,
        st: &mut PleSeqState,
        highway: DevicePtr,
        num_tokens: usize,
        fresh: bool,
        ids_override: Option<&[u32]>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            num_tokens <= self.max_tokens,
            "PLE: {num_tokens} tokens exceeds the {} this layer was sized for. \
             Raise ATLAS_PLE_MAX_TOKENS (costs tokens*10240*14 bytes of \
             scratch) or lower the prefill chunk size.",
            self.max_tokens
        );
        let c = self.hc_mult * self.hidden;
        let heads = self.dims.ngram_heads();
        let gpu = ctx.gpu;

        // The ids are a pure function of TOKEN IDS, computed on the host.
        // Prefer `ctx.host_token_ids` — the very slice the caller uploaded
        // into the device buffer — over reading the device copy back: the D2H
        // was a synchronous round trip per DECODE STEP for bytes the caller
        // had in hand, and inside a CUDA-graph capture region it is a
        // capture-unsupported op (STREAM_CAPTURE_INVALIDATED, 901).
        let tokens: Vec<u32> = if let Some(ov) = ids_override {
            anyhow::ensure!(ov.len() == num_tokens, "PLE: ids_override length");
            ov.to_vec()
        } else if let Some(host) = ctx.host_token_ids {
            anyhow::ensure!(
                host.len() >= num_tokens,
                "PLE: host_token_ids has {} ids for {num_tokens} tokens",
                host.len()
            );
            host[..num_tokens].to_vec()
        } else {
            // Fallback for passes that did not thread the host slice. Never
            // legal under capture — refuse rather than invalidate the graph.
            anyhow::ensure!(
                !ctx.graph_capture,
                "PLE: no host_token_ids and a D2H readback is \
                 capture-unsupported; thread the host ids through this pass"
            );
            let tok_dev = ctx.token_ids.ok_or_else(|| {
                anyhow::anyhow!(
                    "PLE needs token ids (host or device); this pass staged \
                     neither"
                )
            })?;
            let mut raw = vec![0u8; num_tokens * 4];
            gpu.copy_d2h(tok_dev, &mut raw)?;
            raw.chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };

        // A prestage staged for a REPLAYED decode step is consumed by the
        // graph, not by this forward (which never runs on replay) — so a
        // `Some` here on a prefill call is ordinary leftover from the
        // previous request's last replayed step, not an error. Only a
        // single-token, non-fresh decode WITHOUT an ids override may
        // consume it: the multi-seq path (`forward_row`) never prestages,
        // so a `Some` there is always the single-seq path's leftover for
        // the PREVIOUS token — consuming it injected the prior token's
        // n-gram rows AND skipped the history advance, shifting every
        // later hash window by one. That was the mixed-tick corruption
        // (one wrong token, then a permanently degraded tail) that forced
        // the hc_mixed_decode_veto; `.take()` still clears the leftover.
        let prestaged = st
            .prestaged_va
            .take()
            .filter(|_| num_tokens == 1 && !fresh && ids_override.is_none());
        if fresh || st.history.len() != self.dims.context_len() {
            self.reset(st, gpu, stream)?;
        }

        if let Some(table_va) = prestaged {
            // The host half already ran from `decode_prestage`, before graph
            // replay/capture: slots sit in `slots_dev`, history has advanced.
            // Only the capture-safe kernel half remains.
            self.gather_embed(table_va, num_tokens, heads, gpu, stream)?;
        } else {
            anyhow::ensure!(
                !ctx.graph_capture,
                "PLE: un-prestaged forward inside CUDA graph capture — the \
                 pageable slot upload would invalidate the recording (901); \
                 the scheduler must call decode_prestage every step"
            );
            // history ++ tokens, hashed together, then keep the new tokens'
            // rows — the same slice the reference takes with
            // `[:, -input_ids.shape[1]:]`.
            let mut window = st.history.clone();
            window.extend_from_slice(&tokens);
            let all = ple_ngram_ids(&self.dims, &window);
            let rows = &all[all.len() - num_tokens..];
            let flat: Vec<u64> = rows.iter().flat_map(|r| r.iter().copied()).collect();

            self.gather(&flat, num_tokens, heads, gpu, stream)?;

            // Carry the last `context_len` tokens for the next step.
            let keep = self.dims.context_len();
            st.history = window[window.len() - keep..].to_vec();
        }

        // Projections off the concatenated n-gram embedding.
        //
        // `dense_gemm_bf16_pipelined`, NOT `dense_gemm`: the ops wrapper and
        // the kernel are a PAIR. `dense_gemm` launches grid
        // [ceil(n,16), ceil(m,16)] block 16x16 for the scalar kernel, while
        // the pipelined one wants [ceil(n,128), ceil(m,128)] block 256.
        // Handing the pipelined kernel to the scalar launcher reads far out of
        // bounds and produced NaN through the whole highway.
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.gemm_k,
            self.emb,
            &self.key_proj,
            self.key,
            num_tokens as u32,
            c as u32,
            self.hidden as u32,
            stream,
        )
        .context("PLE key_proj")?;
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.gemm_k,
            self.emb,
            &self.value_proj,
            self.value,
            num_tokens as u32,
            self.hidden as u32,
            self.hidden as u32,
            stream,
        )
        .context("PLE value_proj")?;

        ops::ple_gate(
            gpu,
            self.gate_k,
            highway,
            self.key,
            self.value,
            self.norm_query.weight,
            self.norm_key.weight,
            self.norm_conv.weight,
            self.gated,
            self.gated_normed,
            num_tokens as u32,
            self.hidden as u32,
            self.hc_mult as u32,
            self.eps,
            stream,
        )?;
        ops::ple_conv(
            gpu,
            self.conv_k,
            self.gated_normed,
            self.gated,
            self.conv1d.weight,
            st.conv,
            self.out,
            num_tokens as u32,
            c as u32,
            self.k_size as u32,
            self.dilation as u32,
            stream,
        )?;
        ops::ple_add_highway(
            gpu,
            self.add_k,
            self.out,
            highway,
            (num_tokens * c) as u32,
            stream,
        )?;

        Ok(())
    }

    /// Resolve row ids to cache slots and gather them into `self.emb`.
    ///
    /// `T * ngram_heads` rows of `head_dim` land contiguously, which IS the
    /// `[T, ngram_heads * head_dim]` concatenation the projections expect —
    /// so `batched_embed` needs no PLE-specific variant.
    fn gather(
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
    fn gather_host(&self, ids: &[u64], gpu: &dyn GpuBackend, stream: u64) -> Result<u64> {
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
    fn gather_embed(
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

/// The dense weights of one PLE site.
pub struct PleWeights {
    pub key_proj: DenseWeight,
    pub value_proj: DenseWeight,
    pub norm_key: DenseWeight,
    pub norm_query: DenseWeight,
    pub norm_conv: DenseWeight,
    pub conv1d: DenseWeight,
}

// Child module (not sibling): the aux fns read PleLayer private
// fields, and only a CHILD module sees them. Same #[path] trick
// qsa.rs uses for its tests.
#[path = "aux_state.rs"]
mod aux_state;
