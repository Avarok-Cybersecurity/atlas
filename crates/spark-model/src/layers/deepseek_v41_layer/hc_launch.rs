// provenance-id: 526f6e616c6420522e205374657369616b
//! The hyper-connection launches of one DeepSeek-V4.1 block at decode.
//!
//! Per site (attention, FFN) the chain was four one-block launches:
//! `hc_v41_mixes_dot` (one block per mix, kept), `hc_v41_mixes_finish` (one
//! thread, the sinkhorn epilogue), `hc_v41_collapse` and `hc_post` (one block
//! of 256 threads over H = 5120). The finish and the collapse of a site are
//! independent (the collapse reads the previous site's `pre`, the finish
//! writes this site's), so they share one launch over ceil(H/256) blocks;
//! `hc_post` and the final collapse spread the same way, one column a thread.
//! Every column keeps its per-element expression and order, so the bytes are
//! the ones the one-block kernels produced.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::DeepSeekV41Layer;
use crate::layers::qwen3_attention::HcSiteWeights;

const HC_BLOCK: u32 = 256;

impl DeepSeekV41Layer {
    /// Site mixes (`pre_out`, `post_s`, `comb_s` from `streams`) and the
    /// block's collapse (`y = pre_in . streams`) in two launches: the dots
    /// over (m, mix_hc) blocks, then `hc_v41_finish_collapse`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mixes_collapse(
        &self,
        gpu: &dyn GpuBackend,
        site: &HcSiteWeights,
        streams: DevicePtr,
        pre_out: DevicePtr,
        pre_in: DevicePtr,
        y: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let mix_hc = (2 + rt.hc_mult) * rt.hc_mult;
        KernelLaunch::new(gpu, self.k_mixes_dot)
            .grid([m as u32, mix_hc as u32, 1])
            .block([HC_BLOCK, 1, 1])
            .arg_ptr(streams)
            .arg_ptr(site.hc_fn)
            .arg_ptr(rt.mixes_s)
            .arg_u32(rt.hidden as u32)
            .arg_u32(rt.hc_mult as u32)
            .arg_f32(rt.norm_eps)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.k_finish_collapse)
            .grid([(rt.hidden as u32).div_ceil(HC_BLOCK), m as u32, 1])
            .block([HC_BLOCK, 1, 1])
            .arg_ptr(rt.mixes_s)
            .arg_ptr(site.hc_scale)
            .arg_ptr(site.hc_base)
            .arg_ptr(pre_out)
            .arg_ptr(rt.post_s)
            .arg_ptr(rt.comb_s)
            .arg_ptr(streams)
            .arg_ptr(pre_in)
            .arg_ptr(y)
            .arg_u32(rt.hidden as u32)
            .arg_u32(rt.hc_mult as u32)
            .arg_u32(rt.sinkhorn_iters as u32)
            .arg_f32(rt.hc_eps)
            .launch(stream)
    }

    /// `y = pre . streams` alone (the final collapse of the last block).
    pub(super) fn collapse(
        &self,
        gpu: &dyn GpuBackend,
        streams: DevicePtr,
        pre: DevicePtr,
        y: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_collapse_wide)
            .grid([(self.rt.hidden as u32).div_ceil(HC_BLOCK), m as u32, 1])
            .block([HC_BLOCK, 1, 1])
            .arg_ptr(streams)
            .arg_ptr(pre)
            .arg_ptr(y)
            .arg_u32(self.rt.hidden as u32)
            .arg_u32(self.rt.hc_mult as u32)
            .launch(stream)
    }

    /// `streams[j] = post[j] * block_out + sum_i comb[i][j] * streams[i]`,
    /// in place, one column a thread.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn hc_post(
        &self,
        gpu: &dyn GpuBackend,
        block_out: DevicePtr,
        streams: DevicePtr,
        post: DevicePtr,
        comb: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_post_wide)
            .grid([(self.rt.hidden as u32).div_ceil(HC_BLOCK), m as u32, 1])
            .block([HC_BLOCK, 1, 1])
            .arg_ptr(block_out)
            .arg_ptr(streams)
            .arg_ptr(post)
            .arg_ptr(comb)
            .arg_ptr(streams)
            .arg_u32(self.rt.hidden as u32)
            .arg_u32(self.rt.hc_mult as u32)
            .launch(stream)
    }
}
