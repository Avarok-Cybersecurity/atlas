// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the geometry, scratch carving and BASELINE route launchers
//! for `examples/gdn_chunk_tc_ada_microtest.rs`: the production FLA route
//! (`gated_delta_rule_recompute_wu` -> `gated_delta_rule_chunk_delta_h_vfused`
//! -> `gated_delta_rule_chunk_fwd_o`, the ABI of `layers/ops/ssm_gdn_a3.rs`)
//! and the scalar token-sequential spine (`gated_delta_rule_prefill`), both at
//! batch 1 in Atlas's packed prefill layout. Included by `#[path]`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Qwen3.5-4B's GDN geometry: 16 key heads, 32 value heads, 128 x 128.
pub const NK: usize = 16;
pub const NV: usize = 32;
pub const D: usize = 128;
pub const C: u32 = 64;
pub const KEY_DIM: usize = NK * D;
/// q | k | v per token, the packed conv output row.
pub const CONV_DIM: usize = 2 * KEY_DIM + NV * D;
/// gate | beta per token, interleaved FP32.
pub const GB: usize = 2 * NV;

pub struct Kernels {
    pub tc: [KernelHandle; 3],
    pub fla: [KernelHandle; 3],
    pub scalar: KernelHandle,
    pub sm: u32,
}

/// The FLA scratch (W, U, S, uc, gc), which the TC route carves the same way.
pub fn scratch_bytes(batch: u32, nc: u32) -> usize {
    let n = (batch * nc) as usize * NV;
    let c = C as usize;
    n * c * D * 2 * 3 + n * D * D * 2 + n * c * 4
}

/// `[w, u, s, uc, gc]` of the carving above.
pub fn regions(base: DevicePtr, batch: u32, nc: u32) -> [DevicePtr; 5] {
    let n = (batch * nc) as usize * NV;
    let c = C as usize;
    let w = base;
    let u = w.offset(n * c * D * 2);
    let s = u.offset(n * c * D * 2);
    let uc = s.offset(n * D * D * 2);
    let gc = uc.offset(n * c * D * 2);
    [w, u, s, uc, gc]
}

/// The production FLA route at batch 1.
#[allow(clippy::too_many_arguments)]
pub fn run_fla(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    h: DevicePtr,
    qkv: DevicePtr,
    gb: DevicePtr,
    out: DevicePtr,
    scratch: DevicePtr,
    t: u32,
    stream: u64,
) -> Result<()> {
    let nc = t.div_ceil(C);
    let [w, u, s, uc, gc] = regions(scratch, 1, nc);
    let (q, kk, v) = (qkv, qkv.offset(KEY_DIM * 2), qkv.offset(2 * KEY_DIM * 2));
    let (gate, beta) = (gb, gb.offset(NV * 4));
    let (cd, nk, nv, d) = (CONV_DIM as u32, NK as u32, NV as u32, D as u32);
    let null = DevicePtr::NULL;
    KernelLaunch::new(gpu, k.fla[0])
        .grid([nc, nv, 1])
        .block([256, 1, 1])
        .shared_mem(C * d * 2 + C * C * 4 + C * 4)
        .arg_ptr(kk)
        .arg_ptr(v)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(w)
        .arg_ptr(u)
        .arg_ptr(gc)
        .arg_u32(1)
        .arg_u32(t)
        .arg_u32(nc)
        .arg_u32(nk)
        .arg_u32(nv)
        .arg_u32(d)
        .arg_u32(d)
        .arg_u32(cd)
        .arg_u32(cd)
        .arg_u32(GB as u32)
        .arg_ptr(null)
        .arg_ptr(null)
        .arg_u32(0)
        .launch(stream)?;
    KernelLaunch::new(gpu, k.fla[1])
        .grid([nv, 1, 1])
        .block([256, 1, 1])
        .shared_mem(C * d * 2 * 2 + C * d * 2 + (C + 1) * 4)
        .arg_ptr(h)
        .arg_ptr(w)
        .arg_ptr(u)
        .arg_ptr(kk)
        .arg_ptr(gate)
        .arg_ptr(gc)
        .arg_ptr(s)
        .arg_ptr(uc)
        .arg_u32(1)
        .arg_u32(t)
        .arg_u32(nc)
        .arg_u32(nk)
        .arg_u32(nv)
        .arg_u32(d)
        .arg_u32(d)
        .arg_u32(cd)
        .arg_u32(GB as u32)
        .arg_u32(0)
        .arg_ptr(null)
        .arg_ptr(null)
        .arg_u32(0)
        .launch(stream)?;
    KernelLaunch::new(gpu, k.fla[2])
        .grid([nc, nv, 1])
        .block([512, 1, 1])
        .shared_mem(C * d * 2 * 2 + C * C * 4 + C * d * 2 + d * d * 2 + 2 * C * 4)
        .arg_ptr(q)
        .arg_ptr(kk)
        .arg_ptr(gate)
        .arg_ptr(gc)
        .arg_ptr(s)
        .arg_ptr(uc)
        .arg_ptr(out)
        .arg_u32(1)
        .arg_u32(t)
        .arg_u32(nc)
        .arg_u32(nk)
        .arg_u32(nv)
        .arg_u32(d)
        .arg_u32(d)
        .arg_u32(cd)
        .arg_u32(GB as u32)
        .arg_ptr(null)
        .arg_ptr(null)
        .arg_u32(0)
        .launch(stream)
}

/// The scalar token-sequential spine (`gated_delta_rule_prefill`), batch 1:
/// its query offset carries no batch term, so it is never launched wider.
#[allow(clippy::too_many_arguments)]
pub fn run_scalar(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    h: DevicePtr,
    qkv: DevicePtr,
    gb: DevicePtr,
    out: DevicePtr,
    t: u32,
    stream: u64,
) -> Result<()> {
    let cd = CONV_DIM as u32;
    KernelLaunch::new(gpu, k.scalar)
        .grid([NV as u32, 1, 1])
        .block([128, 1, 1])
        .shared_mem(((D * D + 2 * D) * 4) as u32)
        .arg_ptr(h)
        .arg_ptr(qkv)
        .arg_ptr(qkv.offset(KEY_DIM * 2))
        .arg_ptr(qkv.offset(2 * KEY_DIM * 2))
        .arg_ptr(gb)
        .arg_ptr(gb.offset(NV * 4))
        .arg_ptr(out)
        .arg_u32(1)
        .arg_u32(t)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(D as u32)
        .arg_u32(D as u32)
        .arg_u32(cd)
        .arg_u32(cd)
        .arg_u32(GB as u32)
        .launch(stream)
}
