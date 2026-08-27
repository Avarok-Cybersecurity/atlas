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
pub struct Glm5NextMhcKernels {
    pub hc_pre: KernelHandle,
    pub hc_post: KernelHandle,
    pub hc_head: KernelHandle,
}

/// The one module name GLM's mHC resolves from.
pub const GLM5NEXT_MHC_MODULE: &str = "glm5next_mhc";

impl Glm5NextMhcKernels {
    /// Resolve all three. `kernel()` (not `try_kernel`) — a missing mHC kernel is a hard error,
    /// never a silent fallback onto the DeepSeek variant.
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            hc_pre: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_pre")?,
            hc_post: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_post")?,
            hc_head: gpu.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_head")?,
        })
    }
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
