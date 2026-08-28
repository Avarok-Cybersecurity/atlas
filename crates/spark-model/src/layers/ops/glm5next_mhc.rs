// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3-Flash mHC kernel dispatch — the parts that are NOT DeepSeek-V4's.
//!
//! Separate file so `ops/hyper_connection.rs` (V4's proven dispatch) stays byte-untouched.
//!
//! `hc_pre` and `hc_post` need no wrapper here: the GLM kernels
//! `glm5next_mhc::{glm5next_hc_pre, glm5next_hc_post}` have signatures IDENTICAL to their
//! `hyper_connection` counterparts, so the GLM path calls `ops::hc_pre` / `ops::hc_post` with a
//! GLM `KernelHandle`. Only `hc_head` needs its own entry point, because GLM's takes no weights
//! at all.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Every kernel GLM-5.3's hyper-connection needs, all from the single module `glm5next_mhc`.
///
/// The point of this struct is target independence: a GLM kernel target must not have to carry
/// the DeepSeek-V4 `hyper_connection` module to resolve half of its own mHC.
/// `Copy` so all 45 layers can share one resolution — these are opaque handles, not state.
#[derive(Clone, Copy)]
pub struct Glm5NextMhcKernels {
    /// Broadcast the embedding into the `hc_mult` streams. First text layer only.
    ///
    /// 🪤 This is an architecture-neutral broadcast that Atlas already had — and GLM still
    /// needs its own, because `hyper_connection::hc_expand` lives in the **DeepSeek-V4
    /// target directory**. A kernel target merges `common/` plus its OWN model dir and
    /// cannot reach into another target's, so for the GLM target that module does not
    /// exist. Resolving it would fail at first construction, not fall back.
    pub hc_expand: KernelHandle,
    /// The FUSED single-block `hc_pre`. Still resolved, and still the oracle
    /// `examples/glm5next_hc_split_gate.rs` gates the split pair against — but the serve path
    /// goes through `hc_mix` + `hc_finish`, which are bit-identical and ~mix_hc times wider.
    pub hc_pre: KernelHandle,
    /// One block per mixing row: grid `(T, mix_hc)`.
    pub hc_mix: KernelHandle,
    /// Split + Sinkhorn + collapse, reading the mixes from global.
    pub hc_finish: KernelHandle,
    pub hc_post: KernelHandle,
    pub hc_head: KernelHandle,
}

/// The one module name GLM's mHC resolves from.
pub const GLM5NEXT_MHC_MODULE: &str = "glm5next_mhc";

impl Glm5NextMhcKernels {
    /// Resolve all six. `kernel()` (not `try_kernel`) — a missing mHC kernel is a hard error,
    /// never a silent fallback onto the DeepSeek variant.
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            hc_expand: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_expand")?,
            hc_pre: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_pre")?,
            hc_mix: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_mix")?,
            hc_finish: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_finish")?,
            hc_post: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_post")?,
            hc_head: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_head")?,
        })
    }
}

/// Expand one BF16 hidden state into the `hc_mult` FP32 highway streams. First layer only.
pub fn glm_hc_expand(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    streams: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(streams)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// Final collapse before the LM head: an **unweighted mean** over the `hc_mult` streams.
///
/// 🔴 Deliberately takes NO weight pointers. GLM's `Glm5NextTextHyperHead` has no parameters and
/// the checkpoint carries zero `hc_head` tensors; DeepSeek-V4's `ops::hc_head` reads
/// `hc_head.{fn,base,scale}`. The absent arguments are the guard against reaching for weights
/// that do not exist.
pub fn hc_head_mean(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    y_out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(y_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// Per-site mHC weights. One set for the attention site, one for the FFN site.
///
/// 🪤 `hc_fn` is **FP32 to the kernel** and BF16 on disk — the loader upcasts. Reading it at
/// the on-disk width is the #341/#347 dtype-mismatch class, and the shapes do not say so.
#[derive(Debug, Clone, Copy)]
pub struct Glm5NextMhcSiteWeights {
    /// `[mix_hc, hc_mult * hidden]` FP32, where `mix_hc = (2 + hc_mult) * hc_mult`.
    pub hc_fn: DevicePtr,
    /// `[3]` FP32 — the three logit scales (pre, post, comb), in that order.
    pub hc_scale: DevicePtr,
    /// `[mix_hc]` FP32.
    pub hc_base: DevicePtr,
    /// `[MHC_MIX_MAX_TOKENS, mix_hc]` FP32 scratch: `hc_mix` writes it, `hc_finish` reads it.
    /// Per-site, so the two sites of a layer cannot alias; both run on one stream in order.
    pub mix: DevicePtr,
}

/// Token bound on the `mix` scratch. The GLM stack drives mHC one token at a time (the highway
/// forces a serial prefill), so this is slack, not a shape — but `glm_hc_pre` REFUSES above it
/// rather than writing past the allocation.
pub const MHC_MIX_MAX_TOKENS: usize = 256;

/// `hc_pre`: collapse the `hc_mult` FP32 streams to one BF16 sequence and emit this site's
/// `post` / `comb` mixing coefficients.
///
/// 🪤 Named `glm_hc_pre`, not `hc_pre`: `ops` already exports DeepSeek-V4's `hc_pre`/`hc_post`
/// through a glob, and the two are NOT interchangeable — V4's reads `hc_head.{fn,base,scale}`
/// and uses a different mixing law. The rename is the compiler-enforced version of this
/// module's "never silently fall back onto the DeepSeek variant" rule; a glob collision here
/// resolved the wrong way would be a silent architecture swap.
///
/// The residual that `glm_hc_post` mixes is the stream tensor **as it entered here** — snapshot it
/// before calling, which is the skeleton's `ResidualStep::SaveResidual`. Overwriting the streams
/// in place before `hc_post` runs silently changes what the residual means.
#[allow(clippy::too_many_arguments)]
pub fn glm_hc_pre(
    gpu: &dyn GpuBackend,
    kernels: &Glm5NextMhcKernels,
    streams: DevicePtr,
    w: &Glm5NextMhcSiteWeights,
    y_out: DevicePtr,
    post_out: DevicePtr,
    comb_out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    sinkhorn_iters: u32,
    norm_eps: f32,
    hc_eps: f32,
    stream: u64,
) -> Result<()> {
    let mix_hc = (2 + hc_mult) * hc_mult;
    if num_tokens as usize > MHC_MIX_MAX_TOKENS {
        anyhow::bail!(
            "glm_hc_pre: {num_tokens} tokens exceeds the {MHC_MIX_MAX_TOKENS}-token `mix` \
             scratch. Raise MHC_MIX_MAX_TOKENS and rebind; do not launch past the allocation."
        );
    }
    KernelLaunch::new(gpu, kernels.hc_mix)
        .grid([num_tokens, mix_hc, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(w.hc_fn)
        .arg_ptr(w.mix)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_f32(norm_eps)
        .launch(stream)?;
    KernelLaunch::new(gpu, kernels.hc_finish)
        // `1 +` — `blockIdx.y == 0` runs the Sinkhorn and does NOT take a share of the collapse.
        .grid([num_tokens, 1 + collapse_blocks(hidden_size), 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(w.mix)
        .arg_ptr(w.hc_scale)
        .arg_ptr(w.hc_base)
        .arg_ptr(y_out)
        .arg_ptr(post_out)
        .arg_ptr(comb_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(sinkhorn_iters)
        .arg_f32(hc_eps)
        .launch(stream)
}

/// `glm_hc_post`: `out[j] = post[j] * block_out + Σ_i comb[i][j] * residual[i]`.
///
/// `residual` is the pre-`hc_pre` stream snapshot, `block_out` this site's sublayer output.
/// `out` may alias `streams` — each output stream is a fresh combination, so writing back over
/// the highway is the intended flow.
#[allow(clippy::too_many_arguments)]
pub fn glm_hc_post(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    block_out: DevicePtr,
    residual: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, collapse_blocks(hidden_size), 1])
        .block([256, 1, 1])
        .arg_ptr(block_out)
        .arg_ptr(residual)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_ptr(out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// Blocks to spread an `H`-wide, per-element-independent pass over — `hc_finish`'s collapse and
/// all of `hc_post`.
///
/// 🔴 Both used to run on grid `(T, 1, 1)`: one block, i.e. ONE of the GB10's 48 SMs, moving
/// `hc_mult * H` floats, 90 times per token each. Nothing in either is a reduction — every `d`
/// is an independent output element — so the block count is free parallelism and the result is
/// bit-identical at any value of it. 256 is the block width both kernels launch at.
const fn collapse_blocks(hidden_size: u32) -> u32 {
    // `max(1)` by hand: `Ord::max` is not const yet.
    if hidden_size < 256 { 1 } else { hidden_size.div_ceil(256) }
}

/// `mix_hc` — the row count of `hc_fn` and `hc_base`: `(2 + hc_mult) * hc_mult`.
///
/// `pre` and `post` contribute one row per stream each, `comb` contributes `hc_mult` rows per
/// stream. Sizing either tensor with a different formula still yields a well-formed 2-D weight.
pub fn mix_hc(hc_mult: usize) -> usize {
    (2 + hc_mult) * hc_mult
}

#[cfg(test)]
mod mhc_shape_tests {
    use super::*;

    /// The `[pre | post | comb]` split of `hc_fn`'s rows. Getting `mix_hc` wrong shifts every
    /// coefficient the kernel reads, with no shape error.
    #[test]
    fn mix_hc_splits_into_pre_post_and_comb() {
        for hc in 1usize..=8 {
            assert_eq!(mix_hc(hc), hc + hc + hc * hc, "pre + post + comb rows");
        }
        // GLM-5.3 carries hc_mult = 2.
        assert_eq!(mix_hc(2), 8);
    }

    /// The kernel caps `hc_mult` at 4 via `GLM_HC_MAX_MIX = 24 = (2 + 4) * 4`. A larger
    /// multiplicity would overrun its fixed-size register arrays.
    #[test]
    fn the_kernels_mix_bound_is_hc_mult_four() {
        assert_eq!(mix_hc(4), 24, "GLM_HC_MAX_MIX in glm5next_mhc.cu");
        assert!(mix_hc(5) > 24, "hc_mult 5 would exceed the kernel's bound");
    }
}
