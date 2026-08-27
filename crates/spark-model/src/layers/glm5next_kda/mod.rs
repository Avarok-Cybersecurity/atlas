// SPDX-License-Identifier: AGPL-3.0-only
//! GLM-5.3-Flash **KDA layer** — one integrated Atlas layer, Slice 6.
//!
//! Wires the three NEW Slice-3/4/5 kernels (`kda_gate`, `kda_recurrent`, `kda_chunk`) to the
//! REUSE conv/L2/GEMM path and the ADAPT `o_norm`, into the single forward that
//! `Glm5NextTextLinearAttention` performs:
//!
//! ```text
//! q|k|v_proj -> conv1d + SiLU -> L2(q,k only) -> kda_gate / beta
//!            -> kda_chunk (prefill) | kda_recurrent (decode)
//!            -> o_norm(core, g_b(g_a(h))) -> o_proj
//! ```
//!
//! Scope is deliberately ONE layer: no scheduler, no cache, no MoE/DSA, no 34-layer wiring.
//!
//! # What binding this layer proved
//!
//! * **Nothing in a KDA block is quantised.** All 15 layer-0 `self_attn` tensors are BF16 except
//!   `A_log` / `dt_bias`, which are F32 — matching the Slice-3 kernel signatures exactly. There
//!   is no NVFP4 dequant and no NVFP4 GEMM anywhere on this path, so a real-checkpoint oracle
//!   for a KDA layer *is* the production numerics.
//! * 🪤 **The checkpoint splits the conv, HF fuses it.** HF holds one depthwise
//!   `nn.Conv1d(conv_dim)`; the checkpoint stores `q_conv1d` / `k_conv1d` / `v_conv1d`, each
//!   `[8192, 1, 4]`. Binding is `concat([q, k, v], dim=0)` **in that order** — the same order as
//!   `mixed_qkv = cat([q_proj, k_proj, v_proj])`. Reordering is silent.
//! * 🪤 **`squeeze(1)` is a shape-only fix.** `[dim, 1, ks]` and `[dim, ks]` have identical
//!   row-major bytes, so the trap is in shape interpretation, never in data movement — but a
//!   loader that trusts `shape.len() == 2` will reject the tensor outright.
//! * 🪤 **`o_norm` is ADAPT, not REUSE.** Every Atlas gated RMSNorm applies **SiLU** to the gate
//!   (`kernels/gb10/common/rms_norm.cu`); GLM's `Glm5NextTextRMSNormGated` sets
//!   `activation = "sigmoid"`. Slice 2D classified this as REUSE — corrected here by
//!   `kda_o_norm_gated_*` in `kernels/gb10/common/kda_layer_ops.cu`.
//! * **Conv state widths differ.** HF keeps `kernel - 1 = 3` slots, Atlas keeps 4 and shifts
//!   left before convolving, so `HF_state[0..3] == Atlas_state[1..4]` and Atlas slot 0 is a
//!   don't-care (proven in the Slice-6 conv gate by poisoning it with 1e6 for a 0.0 delta).
//! * **The two conv paths are different kernels.** Decode fuses conv+SiLU+L2
//!   (`causal_conv1d_update_l2norm`); prefill does conv+SiLU only
//!   (`causal_conv1d_update_prefill`) and needs a separate `l2_norm_bf16` over q|k.
//!
//! Config values (`gate_lower_bound`, `rms_norm_eps`, `hidden_act`) are carried in
//! [`KdaLayerConfig`] and must be READ from the checkpoint — vLLM agrees with HF on this
//! checkpoint only by coincidence.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// Dynamic shared memory available with no `cuFuncSetAttribute` opt-in in `AtlasCudaBackend`.
/// GB10 reports `sharedMemPerBlockOptin = 101376`, which is real but unreachable here — see
/// blocker 12. Caps `kda_chunk_scan` at `C <= 32` for `D = 128`.
pub const SMEM_CEILING: usize = 49_152;

const BLOCK: u32 = 128;

/// KDA geometry. Production GLM-5.3-Flash: `hidden 4096 / heads 64 / head_dim 128 / kernel 4`.
#[derive(Clone, Copy, Debug)]
pub struct KdaLayerDims {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
    /// Chunk width for the prefill scan. A tiling parameter only — Slice 5 verified identical
    /// results at C = 2..32 — but bounded by [`SMEM_CEILING`].
    pub chunk: usize,
}

impl KdaLayerDims {
    pub fn qkv_dim(&self) -> usize {
        self.heads * self.head_dim
    }
    /// q | k | v concatenated: what the fused depthwise conv sees.
    pub fn conv_dim(&self) -> usize {
        3 * self.qkv_dim()
    }
    /// Only q | k are L2-normalised. V never is.
    pub fn qk_channels(&self) -> usize {
        2 * self.qkv_dim()
    }
    pub fn smem_prepare(&self) -> usize {
        (self.chunk * self.head_dim + self.chunk * self.chunk + self.chunk) * 4
    }
    pub fn smem_scan(&self) -> usize {
        (2 * self.chunk * self.head_dim + self.chunk * self.chunk) * 4
    }
    fn check(&self) -> Result<()> {
        if !self.qk_channels().is_multiple_of(256) {
            bail!("causal_conv1d_update_l2norm requires qk_channels % 256 == 0");
        }
        if self.head_dim != 128 {
            bail!("the fused conv+L2 kernel hardcodes 2 heads per 256-thread block");
        }
        let (p, s) = (self.smem_prepare(), self.smem_scan());
        if p > SMEM_CEILING || s > SMEM_CEILING {
            bail!(
                "chunk={} needs {p}/{s} B shared, ceiling {SMEM_CEILING}",
                self.chunk
            );
        }
        Ok(())
    }
}

/// Values that MUST come from the checkpoint config, never from a library default.
#[derive(Clone, Copy, Debug)]
pub struct KdaLayerConfig {
    /// `linear_attn_config.gate_lower_bound`. vLLM looks up the legacy key `lower_bound`,
    /// misses it, and falls back to a default that happens to match here.
    pub lower_bound: f32,
    /// `rms_norm_eps`, used by `o_norm`. vLLM never passes it.
    pub rms_eps: f32,
    /// Fixed by the FLA convention (`sqrt(sum + eps)`, not `max(norm, eps)`), not by config.
    pub l2_eps: f32,
}

impl Default for KdaLayerConfig {
    fn default() -> Self {
        Self {
            lower_bound: -5.0,
            rms_eps: 1e-5,
            l2_eps: 1e-6,
        }
    }
}

/// One KDA block's device weights. Torch `Linear` layout `[out, in]`, BF16, except the two
/// F32 gate parameters. There is **no `Z` tensor** — the output gate is low-rank `g_a`/`g_b`.
pub struct KdaLayerWeights {
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    /// `[conv_dim, kernel]` BF16 = `concat([q_conv1d, k_conv1d, v_conv1d]).squeeze(1)`.
    pub conv: DenseWeight,
    pub f_a: DenseWeight,
    pub f_b: DenseWeight,
    /// `[heads * head_dim]` F32 — per **channel**.
    pub dt_bias: DevicePtr,
    /// `[heads]` F32 — per **head**. The asymmetry with `dt_bias` is the highest-risk line.
    pub a_log: DevicePtr,
    pub b_proj: DenseWeight,
    pub g_a: DenseWeight,
    pub g_b: DenseWeight,
    /// `[head_dim]` BF16.
    pub o_norm: DenseWeight,
    pub o_proj: DenseWeight,
}

/// Every kernel the layer launches. Resolved with `kernel()` (not `try_kernel`) so a missing
/// entry point is a hard error rather than a silent fallback.
#[derive(Clone, Copy)]
pub struct KdaLayerKernels {
    pub gemm: KernelHandle,
    pub conv_decode: KernelHandle,
    pub conv_prefill: KernelHandle,
    pub l2: KernelHandle,
    pub gate: KernelHandle,
    pub chunk_prepare: KernelHandle,
    pub chunk_scan: KernelHandle,
    pub recurrent: KernelHandle,
    pub o_norm: KernelHandle,
    pub split_widen: KernelHandle,
    pub sigmoid: KernelHandle,
    pub fill: KernelHandle,
    pub pack: KernelHandle,
}

impl KdaLayerKernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            // Scalar strict-order BF16 GEMM: `C = A @ B^T`, no reassociation.
            gemm: gpu.kernel("gemm", "dense_gemm_bf16")?,
            conv_decode: gpu.kernel("causal_conv1d", "causal_conv1d_update_l2norm")?,
            conv_prefill: gpu.kernel("causal_conv1d", "causal_conv1d_update_prefill")?,
            l2: gpu.kernel("norm", "l2_norm_bf16")?,
            gate: gpu.kernel("kda_gate", "kda_gate_bf16")?,
            chunk_prepare: gpu.kernel("kda_chunk", "kda_chunk_prepare")?,
            chunk_scan: gpu.kernel("kda_chunk", "kda_chunk_scan")?,
            recurrent: gpu.kernel("kda_recurrent", "kda_recurrent_decode_bf16")?,
            o_norm: gpu.kernel("kda_layer_ops", "kda_o_norm_gated_bf16")?,
            split_widen: gpu.kernel("kda_layer_ops", "kda_split_widen")?,
            sigmoid: gpu.kernel("kda_layer_ops", "kda_sigmoid_bf16_f32")?,
            fill: gpu.kernel("kda_layer_ops", "kda_fill_f32")?,
            pack: gpu.kernel("kda_layer_ops", "kda_pack_qkv_bf16")?,
        })
    }
}

/// Every observable stage of one forward, kept so the oracle can be compared stage by stage
/// rather than only at the layer output. Buffers are owned by the caller's allocator.
#[derive(Clone, Copy, Debug)]
pub struct KdaStages {
    /// `[3, T, qkv_dim]` BF16 — the three projections as `dense_gemm_bf16` writes them. Kept
    /// separate because that kernel's output row stride is `N`, so aiming three GEMMs at
    /// offsets inside one `[T, 3*qkv]` buffer makes them overwrite each other for `T > 1` —
    /// and is silently correct at `T = 1`, where the two layouts coincide.
    pub qkv_parts: DevicePtr,
    /// `[T, conv_dim]` BF16 — `cat([q_proj, k_proj, v_proj])` per token, pre-conv.
    pub qkv_proj: DevicePtr,
    /// `[T, conv_dim]` BF16 — post conv + SiLU. On the decode path L2 is already fused in.
    pub conv_out: DevicePtr,
    /// `[T_pad, qkv_dim]` F32 — post-L2 q, post-L2 k, raw v. Prefill only.
    pub q_f32: DevicePtr,
    pub k_f32: DevicePtr,
    pub v_f32: DevicePtr,
    /// `[T, heads, head_dim]` F32 — the bounded log-decay from `kda_gate`.
    pub gate: DevicePtr,
    /// `[T, heads]` F32 — already sigmoided.
    pub beta: DevicePtr,
    /// `[T_pad, heads, head_dim]` F32 — KDA core output, pre-norm.
    pub core: DevicePtr,
    /// `[heads, head_dim, head_dim]` F32, K-major — carried recurrent state, updated in place.
    pub state: DevicePtr,
    /// `[T, qkv_dim]` BF16 — `f_b(f_a(hidden))`, the low-rank forget-gate projection that
    /// `kda_gate` turns into the bounded log-decay. Kept separate from `out_gate`: the two are
    /// the same shape and the same kind of low-rank pair, and aliasing them is silent.
    pub g_raw: DevicePtr,
    /// `[T, qkv_dim]` BF16 — `g_b(g_a(hidden))`, the low-rank output gate.
    pub out_gate: DevicePtr,
    /// `[T, qkv_dim]` BF16 — after gated RMSNorm.
    pub o_norm_out: DevicePtr,
    /// `[T, hidden]` BF16 — layer output.
    pub final_out: DevicePtr,
    /// `[conv_dim, kernel]` F32 — Atlas's 4-wide conv state, updated in place.
    pub conv_state: DevicePtr,
    pub t_pad: usize,
}

/// One bound KDA layer.
pub struct KdaLayer {
    pub dims: KdaLayerDims,
    pub cfg: KdaLayerConfig,
    pub w: KdaLayerWeights,
    pub k: KdaLayerKernels,
}

impl KdaLayer {
    pub fn new(
        dims: KdaLayerDims,
        cfg: KdaLayerConfig,
        w: KdaLayerWeights,
        k: KdaLayerKernels,
    ) -> Result<Self> {
        dims.check()?;
        Ok(Self { dims, cfg, w, k })
    }

    fn gemm(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &DenseWeight,
        out: DevicePtr,
        m: usize,
        n: usize,
        kk: usize,
        stream: u64,
    ) -> Result<()> {
        ops::dense_gemm(
            gpu,
            self.k.gemm,
            input,
            weight,
            out,
            m as u32,
            n as u32,
            kk as u32,
            stream,
        )
    }

    /// Projections + gate + beta + output gate — identical on both paths, and all driven by the
    /// RAW hidden state, never by the post-conv activations.
    #[allow(clippy::too_many_arguments)]
    fn front_end(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        t: usize,
        scratch_lowrank: DevicePtr,
        scratch_beta_bf16: DevicePtr,
        s: &KdaStages,
        stream: u64,
    ) -> Result<()> {
        let d = self.dims;
        let (hid, qkv, hd) = (d.hidden, d.qkv_dim(), d.head_dim);

        // Three separate [T, qkv] GEMMs, then one pack into the q|k|v-per-token layout.
        for (i, w) in [&self.w.q_proj, &self.w.k_proj, &self.w.v_proj]
            .into_iter()
            .enumerate()
        {
            self.gemm(
                gpu,
                hidden,
                w,
                s.qkv_parts.offset(i * t * qkv * 2),
                t,
                qkv,
                hid,
                stream,
            )?;
        }
        KernelLaunch::new(gpu, self.k.pack)
            .grid([div_ceil(qkv as u32, 256), t as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(s.qkv_parts)
            .arg_ptr(s.qkv_parts.offset(t * qkv * 2))
            .arg_ptr(s.qkv_parts.offset(2 * t * qkv * 2))
            .arg_ptr(s.qkv_proj)
            .arg_u32(t as u32)
            .arg_u32(qkv as u32)
            .launch(stream)?;
        // Low-rank forget gate: hidden -> head_dim -> heads*head_dim, then the BOUNDED law.
        self.gemm(
            gpu,
            hidden,
            &self.w.f_a,
            scratch_lowrank,
            t,
            hd,
            hid,
            stream,
        )?;
        self.gemm(
            gpu,
            scratch_lowrank,
            &self.w.f_b,
            s.g_raw,
            t,
            qkv,
            hd,
            stream,
        )?;
        KernelLaunch::new(gpu, self.k.gate)
            .grid([(t * d.heads) as u32, 1, 1])
            .block([BLOCK, 1, 1])
            .arg_ptr(s.g_raw)
            .arg_ptr(self.w.dt_bias)
            .arg_ptr(self.w.a_log)
            .arg_ptr(s.gate)
            .arg_u32(t as u32)
            .arg_u32(d.heads as u32)
            .arg_u32(hd as u32)
            .arg_f32(self.cfg.lower_bound)
            .launch(stream)?;

        // beta = sigmoid(b_proj(hidden)); the KDA kernels take it already sigmoided.
        self.gemm(
            gpu,
            hidden,
            &self.w.b_proj,
            scratch_beta_bf16,
            t,
            d.heads,
            hid,
            stream,
        )?;
        let n = t * d.heads;
        KernelLaunch::new(gpu, self.k.sigmoid)
            .grid([div_ceil(n as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(scratch_beta_bf16)
            .arg_ptr(s.beta)
            .arg_u32(n as u32)
            .launch(stream)?;

        // Low-rank OUTPUT gate — no `Z` tensor exists in a KDA checkpoint.
        self.gemm(
            gpu,
            hidden,
            &self.w.g_a,
            scratch_lowrank,
            t,
            hd,
            hid,
            stream,
        )?;
        self.gemm(
            gpu,
            scratch_lowrank,
            &self.w.g_b,
            s.out_gate,
            t,
            qkv,
            hd,
            stream,
        )
    }

    /// `o_norm` (sigmoid-gated, strict FP32) then `o_proj`.
    fn back_end(&self, gpu: &dyn GpuBackend, t: usize, s: &KdaStages, stream: u64) -> Result<()> {
        let d = self.dims;
        KernelLaunch::new(gpu, self.k.o_norm)
            .grid([(t * d.heads) as u32, 1, 1])
            .block([d.head_dim as u32, 1, 1])
            .arg_ptr(s.core)
            .arg_ptr(s.out_gate)
            .arg_ptr(self.w.o_norm.weight)
            .arg_ptr(s.o_norm_out)
            .arg_u32(d.head_dim as u32)
            .arg_f32(self.cfg.rms_eps)
            .launch(stream)?;
        self.gemm(
            gpu,
            s.o_norm_out,
            &self.w.o_proj,
            s.final_out,
            t,
            d.hidden,
            d.qkv_dim(),
            stream,
        )
    }

    /// Single-token decode. Conv fuses SiLU + L2, so no separate L2 launch — and `q`/`k` reach
    /// `kda_recurrent` already normalised, which is exactly the pre-normalised contract that
    /// kernel takes. Re-normalising here would silently restore bf16 rounding (Slice 4).
    #[allow(clippy::too_many_arguments)]
    pub fn decode(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        s: &KdaStages,
        scratch_lowrank: DevicePtr,
        scratch_beta_bf16: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let d = self.dims;
        let (qkv, cd) = (d.qkv_dim(), d.conv_dim());
        self.front_end(
            gpu,
            hidden,
            1,
            scratch_lowrank,
            scratch_beta_bf16,
            s,
            stream,
        )?;

        ops::conv1d_update_l2norm(
            gpu,
            self.k.conv_decode,
            s.conv_state,
            s.qkv_proj,
            &self.w.conv,
            s.conv_out,
            cd as u32,
            d.conv_kernel as u32,
            1,
            d.qk_channels() as u32,
            d.head_dim as u32,
            self.cfg.l2_eps,
            stream,
        )?;

        KernelLaunch::new(gpu, self.k.recurrent)
            .grid([d.heads as u32, 1, 1])
            .block([BLOCK.min(d.head_dim as u32), 1, 1])
            .shared_mem((3 * d.head_dim * 4) as u32)
            .arg_ptr(s.conv_out)
            .arg_ptr(s.conv_out.offset(qkv * 2))
            .arg_ptr(s.conv_out.offset(qkv * 4))
            .arg_ptr(s.gate)
            .arg_ptr(s.beta)
            .arg_ptr(s.state)
            .arg_ptr(s.core)
            .arg_u32(d.heads as u32)
            .arg_u32(d.head_dim as u32)
            .arg_f32(1.0 / (d.head_dim as f32).sqrt())
            .launch(stream)?;

        self.back_end(gpu, 1, s, stream)
    }

    /// Chunked prefill over `t` tokens from the carried conv + recurrent state.
    ///
    /// `pad_fill` primes the padded q/k/v tails. Zero is the production value; the microtest
    /// passes poison to prove `kda_chunk_*` self-guards — the Slice-5 bug wrote correct outputs
    /// while putting the carried state off by 1.6e13.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        t: usize,
        s: &KdaStages,
        scratch_lowrank: DevicePtr,
        scratch_beta_bf16: DevicePtr,
        chunk_gc: DevicePtr,
        chunk_u: DevicePtr,
        chunk_w: DevicePtr,
        pad_fill: f32,
        stream: u64,
    ) -> Result<()> {
        let d = self.dims;
        let (qkv, cd, hd) = (d.qkv_dim(), d.conv_dim(), d.head_dim);
        let nchunks = t.div_ceil(d.chunk);
        let tp = nchunks * d.chunk;
        if tp != s.t_pad {
            bail!(
                "stage buffers sized for T_pad={} but this prefill needs {tp}",
                s.t_pad
            );
        }
        self.front_end(
            gpu,
            hidden,
            t,
            scratch_lowrank,
            scratch_beta_bf16,
            s,
            stream,
        )?;

        // Prefill conv is conv + SiLU ONLY — L2 is a separate launch over q|k.
        KernelLaunch::new(gpu, self.k.conv_prefill)
            .grid([div_ceil(cd as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.conv_state)
            .arg_ptr(s.qkv_proj)
            .arg_ptr(self.w.conv.weight)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(s.conv_out)
            .arg_u32(cd as u32)
            .arg_u32(d.conv_kernel as u32)
            .arg_u32(t as u32)
            .arg_u32(cd as u32)
            .arg_u32(cd as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.k.l2)
            .grid([(d.qk_channels() / hd) as u32, t as u32, 1])
            .block([hd as u32, 1, 1])
            .arg_ptr(s.conv_out)
            .arg_u32(hd as u32)
            .arg_f32(self.cfg.l2_eps)
            .arg_u32(cd as u32)
            .launch(stream)?;

        // Prime the padded tails, then de-interleave + widen only the REAL tokens.
        for p in [s.q_f32, s.k_f32, s.v_f32] {
            KernelLaunch::new(gpu, self.k.fill)
                .grid([div_ceil((tp * qkv) as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(p)
                .arg_u32((tp * qkv) as u32)
                .arg_f32(pad_fill)
                .launch(stream)?;
        }
        KernelLaunch::new(gpu, self.k.split_widen)
            .grid([div_ceil(qkv as u32, 256), t as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(s.conv_out)
            .arg_ptr(s.q_f32)
            .arg_ptr(s.k_f32)
            .arg_ptr(s.v_f32)
            .arg_u32(t as u32)
            .arg_u32(qkv as u32)
            .launch(stream)?;

        KernelLaunch::new(gpu, self.k.chunk_prepare)
            .grid([nchunks as u32, d.heads as u32, 1])
            .block([BLOCK, 1, 1])
            .shared_mem(d.smem_prepare() as u32)
            .arg_ptr(s.k_f32)
            .arg_ptr(s.v_f32)
            .arg_ptr(s.gate)
            .arg_ptr(s.beta)
            .arg_ptr(chunk_gc)
            .arg_ptr(chunk_u)
            .arg_ptr(chunk_w)
            .arg_u32(d.heads as u32)
            .arg_u32(hd as u32)
            .arg_u32(d.chunk as u32)
            .arg_u32(t as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.k.chunk_scan)
            .grid([d.heads as u32, 1, 1])
            .block([BLOCK, 1, 1])
            .shared_mem(d.smem_scan() as u32)
            .arg_ptr(s.q_f32)
            .arg_ptr(s.k_f32)
            .arg_ptr(chunk_gc)
            .arg_ptr(chunk_u)
            .arg_ptr(chunk_w)
            .arg_ptr(s.state)
            .arg_ptr(s.core)
            .arg_u32(d.heads as u32)
            .arg_u32(hd as u32)
            .arg_u32(d.chunk as u32)
            .arg_u32(nchunks as u32)
            .arg_u32(t as u32)
            .arg_f32(1.0 / (hd as f32).sqrt())
            .launch(stream)?;

        self.back_end(gpu, t, s, stream)
    }
}
