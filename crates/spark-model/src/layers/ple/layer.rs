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

    // `gather` / `gather_host` / `gather_embed` live in `gather.rs` (≤500 LoC
    // split; child module so they keep reading the private fields above).
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
#[path = "gather.rs"]
mod gather;
