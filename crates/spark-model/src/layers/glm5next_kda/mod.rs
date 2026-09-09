// SPDX-License-Identifier: AGPL-3.0-only
//! GLM-5.3-Flash **KDA attention block** — the reusable production component.
//!
//! One [`Glm5NextKdaLayer`] is a fully bound KDA `self_attn` block. Any of the checkpoint's
//! **34** KDA layers instantiates through the same path; the family is structurally uniform
//! (one distinct name/shape/dtype signature across all 34 — see [`binding`]).
//!
//! ```text
//! q|k|v_proj -> pack -> conv1d + SiLU -> L2(q,k only) -> kda_gate / sigmoid(b_proj)
//!            -> kda_chunk (prefill) | kda_recurrent (decode)
//!            -> sigmoid-gated RMSNorm(o_norm, g_b(g_a(h))) -> o_proj
//! ```
//!
//! Scope: the attention block only. No DSA/MLA, no MoE/dense FFN, no mHC hyper-connection, no
//! scheduler or cache integration. **The model is not loadable on this alone.**
//!
//! # Facts this component encodes (proven, do not re-derive)
//!
//! * **Nothing in a KDA block is quantised.** All 15 `self_attn` tensors are BF16 except `A_log`
//!   and `dt_bias`, which are F32 — verified across all 34 blocks, 0 artefacts. There is no NVFP4
//!   dequant and no NVFP4 GEMM anywhere on this path, so a real-checkpoint KDA oracle *is* the
//!   production numerics.
//! * 🪤 **The checkpoint SPLITS the conv; HF FUSES it.** HF holds one depthwise
//!   `nn.Conv1d(conv_dim)`; the checkpoint stores `q_conv1d`/`k_conv1d`/`v_conv1d`, each
//!   `[qkv, 1, kernel]`. Binding is `concat([q, k, v], dim=0)` **in that order** — the same order
//!   as `mixed_qkv = cat([q_proj, k_proj, v_proj])`. Reordering is silent.
//! * 🪤 **`squeeze(1)` is a shape-only fix.** `[dim, 1, ks]` and `[dim, ks]` have identical
//!   row-major bytes, so nothing moves — but a loader trusting `shape.len() == 2` rejects the
//!   tensor outright. [`binding`] asserts rank 3 and squeezes exactly once.
//! * 🪤 **`o_norm` is ADAPT, not REUSE.** Every Atlas gated RMSNorm applies **SiLU** to the gate
//!   (`kernels/gb10/common/rms_norm.cu`); GLM's `Glm5NextTextRMSNormGated` sets
//!   `activation = "sigmoid"`. Shapes, dtypes and launch geometry all agree, which is why it was
//!   first mis-classified. Hence `kda_o_norm_gated_*` in `kernels/gb10/common/kda_layer_ops.cu`.
//! * 🪤 **`dense_gemm_bf16` writes `C[row * N + col]`** — its output row stride is `N` and there
//!   is no caller-supplied output stride, so the three q/k/v projections **cannot** be aimed at
//!   offsets inside one `[T, 3*qkv]` buffer. They would overwrite each other for `T > 1`, while
//!   being silently correct at `T = 1`. Hence the separate parts buffer plus `kda_pack_qkv_bf16`.
//! * **Conv state widths differ.** HF keeps `kernel - 1` slots, Atlas keeps `kernel` and shifts
//!   left before convolving, so `HF[0..k-1] == Atlas[1..k]` and Atlas slot 0 is a don't-care.
//! * **Decode conv fuses L2; prefill does not** and needs a separate `l2_norm_bf16` over q|k.
//!   q/k are normalised **exactly once**; **V never**.
//! * **The recurrent state is FP32 by reference semantics**, not Atlas policy — HF stores it via
//!   `.to(torch.float32)` and vLLM's `kda_state_dtype` hardcodes fp32.

pub mod binding;
pub mod tp;
/// Applying the TP plan: the shard copies, as an upstream adapter so the binder is untouched.
pub mod tp_bind;

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// Dynamic shared memory available with no `cuFuncSetAttribute` opt-in in `AtlasCudaBackend`.
/// GB10 reports `sharedMemPerBlockOptin = 101376`, real but unreachable from Atlas today.
pub const SMEM_CEILING: usize = 49_152;

const BLOCK: u32 = 128;

/// Geometry and the config values that MUST be read from the checkpoint.
///
/// `gate_lower_bound`, `rms_norm_eps` and `hidden_act` coincide with a library default on
/// GLM-5.3-Flash **by accident** — vLLM looks up the legacy key `lower_bound`, misses it, and
/// falls back to a matching `-5.0`; it never passes `rms_norm_eps`; it hardcodes `"silu"`.
/// Inheriting any of those defaults is a latent bug on the next checkpoint.
#[derive(Clone, Copy, Debug)]
pub struct Glm5NextKdaConfig {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
    /// `linear_attn_config.gate_lower_bound`.
    pub gate_lower_bound: f32,
    /// `rms_norm_eps`, consumed by `o_norm`.
    pub rms_norm_eps: f32,
    /// FLA convention `1/sqrt(sum + eps)`, not `max(norm, eps)`. Not a config key.
    pub l2_eps: f32,
    /// Prefill tiling width. A tiling parameter only — results are identical at `C = 2..32`, and
    /// agree with HF's `C = 64` — but bounded by [`SMEM_CEILING`].
    pub chunk: usize,
}

impl Glm5NextKdaConfig {
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
    /// FP32 `[heads, head_dim, head_dim]`, K-major. Mandatory dtype, not a policy choice.
    pub fn recurrent_state_elems(&self) -> usize {
        self.heads * self.head_dim * self.head_dim
    }
    /// FP32 `[conv_dim, conv_kernel]` — Atlas's width, one slot wider than HF's.
    pub fn conv_state_elems(&self) -> usize {
        self.conv_dim() * self.conv_kernel
    }
    pub fn smem_prepare(&self) -> usize {
        (self.chunk * self.head_dim + self.chunk * self.chunk + self.chunk) * 4
    }
    pub fn smem_scan(&self) -> usize {
        (2 * self.chunk * self.head_dim + self.chunk * self.chunk) * 4
    }

    pub fn validate(&self) -> Result<()> {
        if !self.qk_channels().is_multiple_of(256) {
            bail!("causal_conv1d_update_l2norm requires qk_channels % 256 == 0");
        }
        if self.head_dim != 128 {
            bail!("the fused conv+L2 kernel hardcodes 2 heads per 256-thread block");
        }
        if self.conv_kernel > 4 {
            bail!("the conv kernels keep the sliding window in 4 registers");
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

/// One KDA block's device weights. Torch `Linear` layout `[out, in]`, BF16, except the two F32
/// gate parameters. There is **no `Z` tensor** — the output gate is low-rank `g_a`/`g_b`.
pub struct Glm5NextKdaWeights {
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    /// `[conv_dim, conv_kernel]` BF16 = `concat([q, k, v]).squeeze(1)`.
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

/// Every kernel the block launches. Resolved with `kernel()` (not `try_kernel`) so a missing
/// entry point is a hard error rather than a silent fallback. Shareable across all 34 layers.
/// V-columns one block of the 1R+1W recurrent kernel owns. One warp: 32 threads, and at
/// `head_dim = 128` a 17.9 KiB scratch that leaves two blocks resident per SM.
const KDA_V_PER_BLOCK: usize = 32;
/// Shared memory the launcher will request without opting in past the default limit.
const KDA_SMEM_BUDGET: usize = 48 * 1024;

/// `ATLAS_GLM_KDA_NO_SMEM=1` restores the 2R+2W recurrent kernel. Read once — this is on
/// the per-layer decode path.
fn kda_no_smem() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("ATLAS_GLM_KDA_NO_SMEM").as_deref() == Ok("1"))
}

#[derive(Clone, Copy)]
pub struct Glm5NextKdaKernels {
    pub gemm: KernelHandle,
    /// 🔴 The DECODE weight kernel. `dense_gemm_bf16` tiles 16x16 over (N, M); at M=1 the
    /// grid collapses and it measured 58 GB/s against a 254 GB/s part — 32 % of the whole
    /// GLM decode step (2026-08-28 profile). Every KDA projection is M=1 at decode.
    pub gemv: KernelHandle,
    /// 🔴 The BATCHED weight kernel, `2 ..= 8` rows in ONE weight sweep. This is what makes a
    /// K-token speculative verify cost one pass over q/k/v/f_a/f_b/g_a/g_b/o instead of K.
    /// `0` on a backend without it — [`ops::dense_mm_bf16`] then falls back to the tile GEMM.
    pub gemv_batchm: KernelHandle,
    pub conv_decode: KernelHandle,
    pub conv_prefill: KernelHandle,
    pub l2: KernelHandle,
    pub gate: KernelHandle,
    pub chunk_prepare: KernelHandle,
    pub chunk_scan: KernelHandle,
    pub recurrent: KernelHandle,
    /// 1R+1W sibling of `recurrent`, **bit-identical**: the decayed state column lives in
    /// shared memory between the two passes instead of being re-read from global.
    /// `try_kernel` — a target without it falls back to the 2R+2W kernel.
    pub recurrent_smem: KernelHandle,
    pub o_norm: KernelHandle,
    pub split_widen: KernelHandle,
    pub sigmoid: KernelHandle,
    pub fill: KernelHandle,
    pub pack: KernelHandle,
}

impl Glm5NextKdaKernels {
    pub const ENTRY_POINTS: usize = 14;

    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            // Scalar strict-order BF16 GEMM: `C = A @ B^T`, no reassociation.
            gemm: gpu.kernel("gemm", "dense_gemm_bf16")?,
            gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            gemv_batchm: crate::layers::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_batchm",
            ),
            conv_decode: gpu.kernel("causal_conv1d", "causal_conv1d_update_l2norm")?,
            conv_prefill: gpu.kernel("causal_conv1d", "causal_conv1d_update_prefill")?,
            l2: gpu.kernel("norm", "l2_norm_bf16")?,
            gate: gpu.kernel("kda_gate", "kda_gate_bf16")?,
            chunk_prepare: gpu.kernel("kda_chunk", "kda_chunk_prepare")?,
            chunk_scan: gpu.kernel("kda_chunk", "kda_chunk_scan")?,
            recurrent: gpu.kernel("kda_recurrent", "kda_recurrent_decode_bf16")?,
            recurrent_smem: crate::layers::try_kernel(
                gpu,
                "kda_recurrent",
                "kda_recurrent_decode_bf16_smem",
            ),
            o_norm: gpu.kernel("kda_layer_ops", "kda_o_norm_gated_bf16")?,
            split_widen: gpu.kernel("kda_layer_ops", "kda_split_widen")?,
            sigmoid: gpu.kernel("kda_layer_ops", "kda_sigmoid_bf16_f32")?,
            fill: gpu.kernel("kda_layer_ops", "kda_fill_f32")?,
            pack: gpu.kernel("kda_layer_ops", "kda_pack_qkv_bf16")?,
        })
    }
}

/// The per-sequence state a KDA layer carries. Both buffers are read-modify-write.
///
/// 🪤 The recurrent element is **FP32 by reference semantics** — HF stores it via
/// `last_recurrent_state.to(torch.float32)` and vLLM's `kda_state_dtype` returns
/// `(conv_dtype, torch.float32)` regardless of `mamba_cache_dtype`. `--ssm-h-dtype f16` is not
/// available to KDA without deviating from the reference.
///
/// 🪤 The conv buffer is Atlas's **`conv_kernel`-wide** convention, one slot wider than HF's
/// `conv_kernel - 1`: Atlas shifts left before convolving, so slot 0 is shifted out and never
/// participates. `HF[0..k-1] == Atlas[1..k]` pre-shift. Any code moving state between the two
/// conventions must apply that offset.
#[derive(Clone, Copy, Debug)]
pub struct KdaSeqState {
    /// `[conv_dim, conv_kernel]` FP32.
    pub conv: DevicePtr,
    /// `[heads, head_dim, head_dim]` FP32, K-major.
    pub recurrent: DevicePtr,
}

/// Scratch owned by the forward path. Sized once for `max_tokens` and reused across layers —
/// the whole KDA family shares one workspace because every block has identical geometry.
///
/// The intermediate buffers are `pub` on purpose: the numeric oracle compares stage by stage, and
/// a residual quoted only at the layer output cannot separate a kernel bug from a rounding floor.
pub struct Glm5NextKdaWorkspace {
    /// `[3, T, qkv]` BF16 — the three projections as `dense_gemm_bf16` writes them.
    pub qkv_parts: DevicePtr,
    /// `[T, conv_dim]` BF16 — q|k|v per token, pre-conv.
    pub qkv_proj: DevicePtr,
    /// `[T, conv_dim]` BF16 — post conv + SiLU. On the decode path L2 is already fused in.
    pub conv_out: DevicePtr,
    /// `[T_pad, qkv]` FP32 — post-L2 q, post-L2 k, raw v. Prefill only.
    pub q_f32: DevicePtr,
    pub k_f32: DevicePtr,
    pub v_f32: DevicePtr,
    /// `[T_pad, heads, head_dim]` FP32 — bounded log-decay from `kda_gate`.
    pub gate: DevicePtr,
    /// `[T_pad, heads]` FP32 — already sigmoided.
    pub beta: DevicePtr,
    /// `[T_pad, heads, head_dim]` FP32 — KDA core output, pre-norm.
    pub core: DevicePtr,
    /// `[T, qkv]` BF16 — `f_b(f_a(h))`, the forget-gate projection `kda_gate` consumes. Kept
    /// separate from `out_gate`: same shape, same kind of low-rank pair, aliasing is silent.
    pub g_raw: DevicePtr,
    /// `[T, qkv]` BF16 — `g_b(g_a(h))`, the low-rank output gate.
    pub out_gate: DevicePtr,
    /// `[T, qkv]` BF16 — after the sigmoid-gated RMSNorm.
    pub o_norm_out: DevicePtr,
    /// `[T, hidden]` BF16 — the block output.
    pub final_out: DevicePtr,
    lowrank: DevicePtr,
    beta_bf16: DevicePtr,
    chunk_gc: DevicePtr,
    chunk_u: DevicePtr,
    chunk_w: DevicePtr,
    max_tokens: usize,
    t_pad: usize,
}

impl Glm5NextKdaWorkspace {
    pub fn new(gpu: &dyn GpuBackend, cfg: &Glm5NextKdaConfig, max_tokens: usize) -> Result<Self> {
        cfg.validate()?;
        if max_tokens == 0 {
            bail!("workspace needs max_tokens >= 1");
        }
        let (qkv, cd, hd, h) = (cfg.qkv_dim(), cfg.conv_dim(), cfg.head_dim, cfg.heads);
        let t = max_tokens;
        let t_pad = t.div_ceil(cfg.chunk) * cfg.chunk;
        let n = t_pad * qkv;
        Ok(Self {
            qkv_parts: gpu.alloc(3 * t * qkv * 2)?,
            qkv_proj: gpu.alloc(t * cd * 2)?,
            conv_out: gpu.alloc(t * cd * 2)?,
            q_f32: gpu.alloc(n * 4)?,
            k_f32: gpu.alloc(n * 4)?,
            v_f32: gpu.alloc(n * 4)?,
            gate: gpu.alloc(n * 4)?,
            beta: gpu.alloc(t_pad * h * 4)?,
            core: gpu.alloc(n * 4)?,
            g_raw: gpu.alloc(t * qkv * 2)?,
            out_gate: gpu.alloc(t * qkv * 2)?,
            o_norm_out: gpu.alloc(t * qkv * 2)?,
            final_out: gpu.alloc(t * cfg.hidden * 2)?,
            lowrank: gpu.alloc(t * hd * 2)?,
            beta_bf16: gpu.alloc(t * h * 2)?,
            chunk_gc: gpu.alloc(n * 4)?,
            chunk_u: gpu.alloc(n * 4)?,
            chunk_w: gpu.alloc(n * 4)?,
            max_tokens: t,
            t_pad,
        })
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }
    pub fn t_pad(&self) -> usize {
        self.t_pad
    }
}

/// One bound KDA attention block.
pub struct Glm5NextKdaLayer {
    /// Index in the checkpoint's 45-layer text stack, for diagnostics.
    pub layer_idx: usize,
    pub cfg: Glm5NextKdaConfig,
    pub weights: Glm5NextKdaWeights,
    pub kernels: Glm5NextKdaKernels,
}

impl Glm5NextKdaLayer {
    pub fn new(
        layer_idx: usize,
        cfg: Glm5NextKdaConfig,
        weights: Glm5NextKdaWeights,
        kernels: Glm5NextKdaKernels,
    ) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            layer_idx,
            cfg,
            weights,
            kernels,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &DenseWeight,
        out: DevicePtr,
        m: usize,
        n: usize,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        // 🔴 PREFILL WIDTHS GO TO cuBLASLt. `dense_gemm_bf16` is, by its own doc, a
        // "scalar strict-order BF16 GEMM, no reassociation" — correctness-first, not a
        // tensor-core kernel. With the KDA bucket split three ways it is the single
        // largest leaf in prefill: `kda_front` 642.4 ms/tok = 24.4% (and `kda_back`
        // another 7.7%), against `kda_recur` — the per-token recurrence everyone assumes
        // is the problem — at just 4.6%. At 256 rows that is 19.8 ms per layer-call to
        // move ~268 MB of weights, ~20x off a 273 GB/s part.
        //
        // 🪤 The threshold is not a tuning knob, it is the numerics boundary.
        // `dense_mm_bf16` routes M=1 to the GEMV and M=2..=16 to the batched single-sweep
        // kernel, both of which accumulate per row in FP32 in a fixed K order and are
        // BIT-IDENTICAL. Only above `DENSE_GEMV_BATCHM_MAX_M` does it fall to the tile
        // GEMM, which already reassociates. So this swaps one non-bit-exact path for
        // another and leaves every bit-exact tier — decode and the speculative verify —
        // untouched. A verify must still match what decode produced; it is M<=16 and
        // never reaches here.
        // 🪤 Same BF16-out invariant the DSA and MLP helpers ENFORCE with a
        // `batchm.0 != 0` guard. Stated, not enforced, here on purpose: those two take the
        // batchm handle as a per-call argument, so it genuinely marks the call site's
        // output dtype, whereas `self.kernels.gemv_batchm` is one optional `try_kernel`
        // handle shared by every KDA projection. Gating on it would disable cuBLASLt
        // wholesale on any target where that kernel is simply absent — losing the win for
        // no safety, since every caller in this file writes BF16.
        if m > ops::DENSE_GEMV_BATCHM_MAX_M as usize
            && crate::layers::glm5next_layer::cublas_wide_proj()
        {
            return ops::cublas_bf16_proj_dense(
                input,
                weight.weight,
                out,
                m as u32,
                n as u32,
                k as u32,
                stream,
            );
        }
        // M=1 decode -> GEMV, M=2..8 verify/short-chunk -> ONE weight sweep, wider -> tile GEMM.
        ops::dense_mm_bf16(
            gpu,
            &ops::DenseMmKernels {
                gemm: self.kernels.gemm,
                gemv: self.kernels.gemv,
                batchm: self.kernels.gemv_batchm,
            },
            input,
            weight.weight,
            out,
            m,
            n,
            k,
            stream,
        )
    }

    /// Projections, forget gate, beta and output gate — identical on both paths, and all driven
    /// by the RAW hidden state, never by the post-conv activations.
    fn front_end(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        t: usize,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (hid, qkv, hd) = (c.hidden, c.qkv_dim(), c.head_dim);

        // Three separate [T, qkv] GEMMs, then one pack. See the `dense_gemm_bf16` note above:
        // aiming them at offsets in one [T, 3*qkv] buffer is silently correct at T = 1 only.
        for (i, w) in [
            &self.weights.q_proj,
            &self.weights.k_proj,
            &self.weights.v_proj,
        ]
        .into_iter()
        .enumerate()
        {
            self.gemm(
                gpu,
                hidden,
                w,
                ws.qkv_parts.offset(i * t * qkv * 2),
                t,
                qkv,
                hid,
                stream,
            )?;
        }
        KernelLaunch::new(gpu, self.kernels.pack)
            .grid([div_ceil(qkv as u32, 256), t as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(ws.qkv_parts)
            .arg_ptr(ws.qkv_parts.offset(t * qkv * 2))
            .arg_ptr(ws.qkv_parts.offset(2 * t * qkv * 2))
            .arg_ptr(ws.qkv_proj)
            .arg_u32(t as u32)
            .arg_u32(qkv as u32)
            .launch(stream)?;

        // Low-rank forget gate: hidden -> head_dim -> heads*head_dim, then the BOUNDED law
        // `lower_bound * sigmoid(exp(A_log[h]) * (g[c] + dt_bias[c]))`. `A_log` is per HEAD,
        // `dt_bias` per CHANNEL; the asymmetry is why they are separate kernel arguments.
        self.gemm(
            gpu,
            hidden,
            &self.weights.f_a,
            ws.lowrank,
            t,
            hd,
            hid,
            stream,
        )?;
        self.gemm(
            gpu,
            ws.lowrank,
            &self.weights.f_b,
            ws.g_raw,
            t,
            qkv,
            hd,
            stream,
        )?;
        KernelLaunch::new(gpu, self.kernels.gate)
            .grid([(t * c.heads) as u32, 1, 1])
            .block([BLOCK, 1, 1])
            .arg_ptr(ws.g_raw)
            .arg_ptr(self.weights.dt_bias)
            .arg_ptr(self.weights.a_log)
            .arg_ptr(ws.gate)
            .arg_u32(t as u32)
            .arg_u32(c.heads as u32)
            .arg_u32(hd as u32)
            .arg_f32(c.gate_lower_bound)
            .launch(stream)?;

        // beta = sigmoid(b_proj(hidden)); the KDA kernels take it ALREADY sigmoided.
        self.gemm(
            gpu,
            hidden,
            &self.weights.b_proj,
            ws.beta_bf16,
            t,
            c.heads,
            hid,
            stream,
        )?;
        let n = t * c.heads;
        KernelLaunch::new(gpu, self.kernels.sigmoid)
            .grid([div_ceil(n as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(ws.beta_bf16)
            .arg_ptr(ws.beta)
            .arg_u32(n as u32)
            .launch(stream)?;

        // Low-rank OUTPUT gate — a KDA checkpoint has no `Z` tensor.
        self.gemm(
            gpu,
            hidden,
            &self.weights.g_a,
            ws.lowrank,
            t,
            hd,
            hid,
            stream,
        )?;
        self.gemm(
            gpu,
            ws.lowrank,
            &self.weights.g_b,
            ws.out_gate,
            t,
            qkv,
            hd,
            stream,
        )
    }

    /// Sigmoid-gated RMSNorm then `o_proj`.
    fn back_end(
        &self,
        gpu: &dyn GpuBackend,
        t: usize,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        KernelLaunch::new(gpu, self.kernels.o_norm)
            .grid([(t * c.heads) as u32, 1, 1])
            .block([c.head_dim as u32, 1, 1])
            .arg_ptr(ws.core)
            .arg_ptr(ws.out_gate)
            .arg_ptr(self.weights.o_norm.weight)
            .arg_ptr(ws.o_norm_out)
            .arg_u32(c.head_dim as u32)
            .arg_f32(c.rms_norm_eps)
            .launch(stream)?;
        self.gemm(
            gpu,
            ws.o_norm_out,
            &self.weights.o_proj,
            ws.final_out,
            t,
            c.hidden,
            c.qkv_dim(),
            stream,
        )
    }

    /// The stateful half of one KDA token: conv window update (with SiLU + L2 fused) then the
    /// recurrent scan, both reading row `row` of the workspace and updating `state` IN PLACE.
    ///
    /// 🔴 This is the part that CANNOT be batched. The recurrent state after token `t + 1` is a
    /// function of the state after `t`, so K verify rows walk it K times in order — which is why
    /// [`Self::decode_k`] batches only the projections around it. The chunked [`Self::prefill`]
    /// scan computes the same mathematics by a different association and is NOT bit-identical to
    /// this; using it for a verify would move the output of an ACCEPTED token.
    ///
    /// 🪤 The conv fuses SiLU **and** L2, so `q`/`k` reach `kda_recurrent` already normalised —
    /// exactly the pre-normalised contract that kernel takes. Re-normalising would silently
    /// restore the bf16 rounding the fused write destroyed and look like a kernel bug.
    fn stateful_row(
        &self,
        gpu: &dyn GpuBackend,
        row: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let qkv = c.qkv_dim();
        let cd = c.conv_dim();

        ops::conv1d_update_l2norm(
            gpu,
            self.kernels.conv_decode,
            state.conv,
            ws.qkv_proj.offset(row * cd * 2),
            &self.weights.conv,
            ws.conv_out.offset(row * cd * 2),
            c.conv_dim() as u32,
            c.conv_kernel as u32,
            1,
            c.qk_channels() as u32,
            c.head_dim as u32,
            c.l2_eps,
            stream,
        )?;

        let d = c.head_dim;
        // 1R+1W when the target carries the shared-memory sibling. The V axis has no
        // cross-thread dependency, so a block owns a SLICE of it: `vpb` columns need
        // `vpb * (d + 1)` floats of scratch (the `+1` is the bank-conflict pad the kernel
        // documents) and grid.y covers the rest. One warp per block keeps the request
        // inside the 48 KiB default at the production `head_dim = 128`.
        let vpb = KDA_V_PER_BLOCK.min(d);
        let smem_smem = (3 * d + vpb * (d + 1)) * 4;
        if self.kernels.recurrent_smem.0 != 0
            && d.is_multiple_of(vpb)
            && smem_smem <= KDA_SMEM_BUDGET
            && !kda_no_smem()
        {
            KernelLaunch::new(gpu, self.kernels.recurrent_smem)
                .grid([c.heads as u32, (d / vpb) as u32, 1])
                .block([vpb as u32, 1, 1])
                .shared_mem(smem_smem as u32)
                .arg_ptr(ws.conv_out.offset(row * cd * 2))
                .arg_ptr(ws.conv_out.offset(row * cd * 2 + qkv * 2))
                .arg_ptr(ws.conv_out.offset(row * cd * 2 + qkv * 4))
                .arg_ptr(ws.gate.offset(row * qkv * 4))
                .arg_ptr(ws.beta.offset(row * c.heads * 4))
                .arg_ptr(state.recurrent)
                .arg_ptr(ws.core.offset(row * qkv * 4))
                .arg_u32(c.heads as u32)
                .arg_u32(d as u32)
                .arg_f32(1.0 / (d as f32).sqrt())
                .arg_u32(vpb as u32)
                .launch(stream)?;
        } else {
            KernelLaunch::new(gpu, self.kernels.recurrent)
                .grid([c.heads as u32, 1, 1])
                .block([BLOCK.min(d as u32), 1, 1])
                .shared_mem((3 * d * 4) as u32)
                .arg_ptr(ws.conv_out.offset(row * cd * 2))
                .arg_ptr(ws.conv_out.offset(row * cd * 2 + qkv * 2))
                .arg_ptr(ws.conv_out.offset(row * cd * 2 + qkv * 4))
                .arg_ptr(ws.gate.offset(row * qkv * 4))
                .arg_ptr(ws.beta.offset(row * c.heads * 4))
                .arg_ptr(state.recurrent)
                .arg_ptr(ws.core.offset(row * qkv * 4))
                .arg_u32(c.heads as u32)
                .arg_u32(d as u32)
                .arg_f32(1.0 / (d as f32).sqrt())
                .launch(stream)?;
        }

        Ok(())
    }

    /// Single-token decode, carrying both states.
    ///
    /// The conv fuses SiLU **and** L2, so `q`/`k` reach `kda_recurrent` already normalised —
    /// exactly the pre-normalised contract that kernel takes. Re-normalising here would silently
    /// restore the bf16 rounding the fused write destroyed and look like a kernel bug.
    ///
    /// Result lands in `ws.final_out`; `state` is updated in place.
    pub fn decode(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        self.front_end(gpu, hidden, 1, ws, stream)?;
        self.stateful_row(gpu, 0, state, ws, stream)?;
        self.back_end(gpu, 1, ws, stream)
    }

    /// K tokens of ONE sequence: the projections batched, the recurrence NOT.
    ///
    /// This is the speculative-verify body. The weight-heavy halves — `Self::front_end`'s
    /// q/k/v, both low-rank gate pairs and `b_proj`, and `Self::back_end`'s `o_proj` — run
    /// once over all K rows, so a K-token verify reads KDA's 4.7 GB/rank/token ONCE instead of
    /// K times. That is the entire reason speculation can pay on this model.
    ///
    /// 🔴 **Bit-identical to K serial [`Self::decode`] calls**, which is not a nicety: an
    /// accepted draft token must be the token the unspeculated engine would have emitted, or
    /// speculation is silently lossy. It holds because `dense_gemv_bf16_batchm` reproduces each
    /// row's exact K-iteration order and reduction tree (`ops::dense_mm_bf16`), the pack / gate
    /// / sigmoid / `o_norm` kernels are grid-parallel over the token axis, and
    /// `Self::stateful_row` walks the state one token at a time exactly as `decode` does.
    ///
    /// `snapshots[t]` — `(h_dst, conv_dst)` — receives the state AFTER row `t`, which is what a
    /// partial accept rewinds to. Pass `k - 1` of them (a full accept never rewinds) or none.
    pub fn decode_k(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        snapshots: &[(DevicePtr, DevicePtr)],
        stream: u64,
    ) -> Result<()> {
        if k == 0 || k > ws.max_tokens {
            bail!(
                "KDA decode_k of {k} tokens does not fit a workspace built for {}",
                ws.max_tokens
            );
        }
        use crate::layers::glm5next_layer::profile;
        let c = &self.cfg;
        let (h_bytes, conv_bytes) = (c.recurrent_state_elems() * 4, c.conv_state_elems() * 4);
        // Three spans, not one: see the KDA_FRONT doc comment. Each is closed where it
        // ends — a bucket ended inside a loop against a start taken outside it is exactly
        // the double-count that made dsa_proj read 94.6%.
        let t_front = profile::start();
        self.front_end(gpu, hidden, k, ws, stream)?;
        profile::end(profile::KDA_FRONT, t_front, gpu, stream);
        let t_recur = profile::start();
        for row in 0..k {
            self.stateful_row(gpu, row, state, ws, stream)?;
            if let Some((h_dst, conv_dst)) = snapshots.get(row) {
                gpu.copy_d2d_async(state.recurrent, *h_dst, h_bytes, stream)?;
                gpu.copy_d2d_async(state.conv, *conv_dst, conv_bytes, stream)?;
            }
        }
        profile::end(profile::KDA_RECUR, t_recur, gpu, stream);
        let t_back = profile::start();
        let r = self.back_end(gpu, k, ws, stream);
        profile::end(profile::KDA_BACK, t_back, gpu, stream);
        r
    }

    /// Chunked prefill over `t` tokens from the carried state.
    pub fn prefill(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        t: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        self.prefill_with_pad_fill(gpu, hidden, t, state, ws, 0.0, stream)
    }

    /// [`Self::prefill`] with the padded q/k/v tails primed to an arbitrary value.
    ///
    /// Production passes zero. The numeric gate passes poison, because `kda_chunk_*` guard past
    /// `T` **in-kernel** and that guard needs a test with teeth: the Slice-5 bug wrote entirely
    /// correct outputs while leaving the carried recurrent state off by 1.623e13, so a pad tail
    /// the caller zeroes proves nothing.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_with_pad_fill(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        t: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        pad_fill: f32,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (qkv, cd, hd) = (c.qkv_dim(), c.conv_dim(), c.head_dim);
        if t == 0 || t > ws.max_tokens {
            bail!(
                "prefill of {t} tokens does not fit a workspace built for {}",
                ws.max_tokens
            );
        }
        let nchunks = t.div_ceil(c.chunk);
        let tp = nchunks * c.chunk;
        // Same three buckets as `decode_k`, so the recurrent and chunked arms can be
        // compared part by part rather than only on a total.
        let t_front = crate::layers::glm5next_layer::profile::start();
        self.front_end(gpu, hidden, t, ws, stream)?;
        crate::layers::glm5next_layer::profile::end(
            crate::layers::glm5next_layer::profile::KDA_FRONT,
            t_front,
            gpu,
            stream,
        );
        let t_recur = crate::layers::glm5next_layer::profile::start();

        // Prefill conv is conv + SiLU ONLY — L2 is a separate launch over q|k.
        KernelLaunch::new(gpu, self.kernels.conv_prefill)
            .grid([div_ceil(cd as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(state.conv)
            .arg_ptr(ws.qkv_proj)
            .arg_ptr(self.weights.conv.weight)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(ws.conv_out)
            .arg_u32(cd as u32)
            .arg_u32(c.conv_kernel as u32)
            .arg_u32(t as u32)
            .arg_u32(cd as u32)
            .arg_u32(cd as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.kernels.l2)
            .grid([(c.qk_channels() / hd) as u32, t as u32, 1])
            .block([hd as u32, 1, 1])
            .arg_ptr(ws.conv_out)
            .arg_u32(hd as u32)
            .arg_f32(c.l2_eps)
            .arg_u32(cd as u32)
            .launch(stream)?;

        // Gate and beta are read at padded positions by `kda_chunk_prepare`; zero them so a guard
        // failure surfaces in q/k/v, which the regression deliberately poisons.
        for (buf, real, padded) in [
            (ws.gate, t * qkv, tp * qkv),
            (ws.beta, t * c.heads, tp * c.heads),
        ] {
            if padded == real {
                continue;
            }
            KernelLaunch::new(gpu, self.kernels.fill)
                .grid([div_ceil((padded - real) as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(buf.offset(real * 4))
                .arg_u32((padded - real) as u32)
                .arg_f32(0.0)
                .launch(stream)?;
        }
        for p in [ws.q_f32, ws.k_f32, ws.v_f32] {
            KernelLaunch::new(gpu, self.kernels.fill)
                .grid([div_ceil((tp * qkv) as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(p)
                .arg_u32((tp * qkv) as u32)
                .arg_f32(pad_fill)
                .launch(stream)?;
        }
        KernelLaunch::new(gpu, self.kernels.split_widen)
            .grid([div_ceil(qkv as u32, 256), t as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(ws.conv_out)
            .arg_ptr(ws.q_f32)
            .arg_ptr(ws.k_f32)
            .arg_ptr(ws.v_f32)
            .arg_u32(t as u32)
            .arg_u32(qkv as u32)
            .launch(stream)?;

        KernelLaunch::new(gpu, self.kernels.chunk_prepare)
            .grid([nchunks as u32, c.heads as u32, 1])
            .block([BLOCK, 1, 1])
            .shared_mem(c.smem_prepare() as u32)
            .arg_ptr(ws.k_f32)
            .arg_ptr(ws.v_f32)
            .arg_ptr(ws.gate)
            .arg_ptr(ws.beta)
            .arg_ptr(ws.chunk_gc)
            .arg_ptr(ws.chunk_u)
            .arg_ptr(ws.chunk_w)
            .arg_u32(c.heads as u32)
            .arg_u32(hd as u32)
            .arg_u32(c.chunk as u32)
            .arg_u32(t as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.kernels.chunk_scan)
            .grid([c.heads as u32, 1, 1])
            .block([BLOCK, 1, 1])
            .shared_mem(c.smem_scan() as u32)
            .arg_ptr(ws.q_f32)
            .arg_ptr(ws.k_f32)
            .arg_ptr(ws.chunk_gc)
            .arg_ptr(ws.chunk_u)
            .arg_ptr(ws.chunk_w)
            .arg_ptr(state.recurrent)
            .arg_ptr(ws.core)
            .arg_u32(c.heads as u32)
            .arg_u32(hd as u32)
            .arg_u32(c.chunk as u32)
            .arg_u32(nchunks as u32)
            .arg_u32(t as u32)
            .arg_f32(1.0 / (hd as f32).sqrt())
            .launch(stream)?;
        crate::layers::glm5next_layer::profile::end(
            crate::layers::glm5next_layer::profile::KDA_RECUR,
            t_recur,
            gpu,
            stream,
        );

        let t_back = crate::layers::glm5next_layer::profile::start();
        let r = self.back_end(gpu, t, ws, stream);
        crate::layers::glm5next_layer::profile::end(
            crate::layers::glm5next_layer::profile::KDA_BACK,
            t_back,
            gpu,
            stream,
        );
        r
    }
}
