// SPDX-License-Identifier: AGPL-3.0-only

//! The GLM MLP decode forward — dense FFN and routed MoE, one token.
//!
//! Launch geometry is lifted verbatim from the two gated microtests
//! (`examples/glm5next_{ffn,moe}_microtest.rs`, Slice-10 gates 3/4/6/7), which measured this
//! exact sequence against HF 5.16.1 on real layer-0 and layer-3 weights. Nothing here
//! re-derives the equations.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::weights::{Glm5NextDenseMlpWeights, Glm5NextMoeWeights, Nvfp4Proj};
use super::{Glm5NextMlpConfig, Glm5NextMlpKernels};

const GEMM_TILE: u32 = 16;
const W4_TILE: u32 = 64;
const ACT_BLOCK: u32 = 256;

/// `C[M, N] = A[M, K] @ B[N, K]^T`, BF16 in and out.
#[allow(clippy::too_many_arguments)]
fn gemm(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([
            (n as u32).div_ceil(GEMM_TILE),
            (m as u32).div_ceil(GEMM_TILE),
            1,
        ])
        .block([GEMM_TILE, GEMM_TILE, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .launch(stream)?;
    Ok(())
}

/// `C[1, N] = A[1, K] @ dequant(B)[N, K]^T` — the M=1 decode kernel.
///
/// Same NVFP4 operand triple as [`w4a16`], one output row. Used for every routed-expert
/// projection because they are all M=1 and the tile GEMM measured 9.7 GB/s there.
fn w4a16_gemv(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    w: &Nvfp4Proj,
    c: DevicePtr,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([crate::layers::ops::w4a16_gemv_grid_x(n as u32), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w.packed)
        .arg_ptr(w.scale)
        // 🪤 by VALUE, as in `w4a16`.
        .arg_f32(w.scale_2)
        .arg_ptr(c)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .launch(stream)?;
    Ok(())
}

/// `C[M, N] = A[M, K] @ dequant(B)[N, K]^T` — NVFP4 weight, BF16 activation and output.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn w4a16(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    w: &Nvfp4Proj,
    c: DevicePtr,
    m: usize,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([
            (n as u32).div_ceil(W4_TILE),
            (m as u32).div_ceil(W4_TILE),
            1,
        ])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w.packed)
        .arg_ptr(w.scale)
        // 🪤 by VALUE. `weight_scale_2` is a scalar argument, not a pointer.
        .arg_f32(w.scale_2)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .launch(stream)?;
    Ok(())
}

/// `out = silu(min(gate, limit)) * clamp(up, -limit, limit)` over `n` elements.
fn swiglu(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    out: DevicePtr,
    n: usize,
    limit: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([(n as u32).div_ceil(ACT_BLOCK), 1, 1])
        .block([ACT_BLOCK, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(out)
        .arg_u32(n as u32)
        .arg_f32(limit)
        .launch(stream)?;
    Ok(())
}

/// Scratch for one MLP site, allocated once and reused every decode step.
///
/// Sized for the widest thing this site can run: a dense layer's `intermediate_size`, a routed
/// layer's `moe_intermediate_size`, and the shared expert's width.
pub struct Glm5NextMlpWorkspace {
    /// `[max_inter]` BF16 ×3 — gate, up, activated. Shared by every projection pair.
    a_gate: DevicePtr,
    a_up: DevicePtr,
    a_act: DevicePtr,
    /// `[num_experts]` F32 router logits.
    logits: DevicePtr,
    /// `[top_k]` I32 / F32 selection.
    ids: DevicePtr,
    wts: DevicePtr,
    /// `[top_k, hidden]` BF16. 🪤 Fully written every step — remote and invalid slots are
    /// memset to zero before the loop, never left stale.
    expert_out: DevicePtr,
    /// `[hidden]` BF16 shared-expert output.
    shared_out: DevicePtr,
    max_inter: usize,
}

impl Glm5NextMlpWorkspace {
    pub fn new(gpu: &dyn GpuBackend, cfg: &Glm5NextMlpConfig) -> Result<Self> {
        let max_inter = cfg
            .local_dense_intermediate
            .max(cfg.moe_intermediate)
            .max(cfg.local_shared_intermediate)
            .max(1);
        Ok(Self {
            a_gate: gpu.alloc(max_inter * 2)?,
            a_up: gpu.alloc(max_inter * 2)?,
            a_act: gpu.alloc(max_inter * 2)?,
            logits: gpu.alloc(cfg.num_experts * 4)?,
            ids: gpu.alloc(cfg.top_k * 4)?,
            wts: gpu.alloc(cfg.top_k * 4)?,
            expert_out: gpu.alloc(cfg.top_k * cfg.hidden * 2)?,
            shared_out: gpu.alloc(cfg.hidden * 2)?,
            max_inter,
        })
    }
}

/// A BF16 SwiGLU MLP of width `inter`: `down(clamped_swiglu(gate(x), up(x)))`.
///
/// Used for both the dense layers and the shared expert — identical math, different widths.
/// With `tp_world_size > 1` the result is a **partial sum**; the caller reduces.
#[allow(clippy::too_many_arguments)]
pub fn forward_dense(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextDenseMlpWeights,
    inter: usize,
    x: DevicePtr,
    out: DevicePtr,
    ws: &Glm5NextMlpWorkspace,
    stream: u64,
) -> Result<()> {
    if inter == 0 || inter > ws.max_inter {
        bail!(
            "GLM dense MLP: width {inter} does not fit a workspace built for {}",
            ws.max_inter
        );
    }
    gemm(
        gpu,
        k.gemm,
        x,
        w.gate_proj,
        ws.a_gate,
        1,
        inter,
        cfg.hidden,
        stream,
    )?;
    gemm(
        gpu, k.gemm, x, w.up_proj, ws.a_up, 1, inter, cfg.hidden, stream,
    )?;
    swiglu(
        gpu,
        k.swiglu,
        ws.a_gate,
        ws.a_up,
        ws.a_act,
        inter,
        cfg.swiglu_limit,
        stream,
    )?;
    gemm(
        gpu,
        k.gemm,
        ws.a_act,
        w.down_proj,
        out,
        1,
        cfg.hidden,
        inter,
        stream,
    )
}

/// One routed MoE site, one token. Leaves a **partial sum** in `out` whenever this rank shares
/// the experts (EP) or the shared expert (TP) with anyone else.
///
/// # 🪤 The device→host round trip
///
/// The expert loop reads the selected ids back to the host to decide which experts are local.
/// That is a synchronising `copy_d2h` on the decode critical path, once per routed layer.
/// It is deliberate for this slice: correctness first, and it is exactly what the gated
/// microtest does. The upgrade path is the pointer-table grouped GEMM
/// (`layers::moe::ptr_table_build`), which keeps the routing on device — not a change to
/// this math.
#[allow(clippy::too_many_arguments)]
pub fn forward_moe(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextMoeWeights,
    x: DevicePtr,
    out: DevicePtr,
    ws: &Glm5NextMlpWorkspace,
    stream: u64,
) -> Result<()> {
    if w.experts.len() != cfg.local_experts {
        bail!(
            "GLM MoE: {} bound experts but this rank owns {} of {}",
            w.experts.len(),
            cfg.local_experts,
            cfg.num_experts
        );
    }

    use crate::layers::glm5next_layer::profile;

    // ── router: FULL expert set, FP32 logits, replicated on every rank ──
    let t = profile::start();
    gemm(
        gpu,
        k.gemm_f32,
        x,
        w.router,
        ws.logits,
        1,
        cfg.num_experts,
        cfg.hidden,
        stream,
    )?;
    KernelLaunch::new(gpu, k.router)
        .grid([1, 1, 1])
        .block([ACT_BLOCK, 1, 1])
        .arg_ptr(ws.logits)
        .arg_ptr(w.router_bias)
        .arg_ptr(ws.ids)
        .arg_ptr(ws.wts)
        .arg_u32(cfg.num_experts as u32)
        .arg_u32(cfg.top_k as u32)
        // n_group: the parser already refuses anything but 1; the kernel refuses too.
        .arg_u32(1)
        .arg_f32(cfg.routed_scale)
        .arg_u32(u32::from(cfg.renormalize))
        .arg_u32(u32::from(cfg.router_bf16_ladder))
        .launch(stream)?;

    profile::end(profile::MOE_ROUTER, t, gpu, stream);

    // 🪤 Zero FIRST. A slot this rank does not own must contribute exactly zero to the
    // all-reduced sum; leaving the previous token's expert output there is a wrong answer
    // that only appears at EP > 1 and only for tokens whose routing moved.
    gpu.memset_async(ws.expert_out, 0, cfg.top_k * cfg.hidden * 2, stream)?;

    // 🚩 A FULL STREAM SYNC + D2H IN THE MIDDLE OF EVERY MoE LAYER. The routing decision
    // is read back to the host so the expert GEMMs can be launched by id. Timed on its own
    // because it is the one span here that is pure latency and scales with layer count,
    // not with weight bytes.
    let t = profile::start();
    let mut ids = vec![0u8; cfg.top_k * 4];
    gpu.synchronize(stream)?;
    gpu.copy_d2h(ws.ids, &mut ids)?;
    profile::end(profile::MOE_HOSTSYNC, t, gpu, stream);

    let t = profile::start();
    for slot in 0..cfg.top_k {
        let id = i32::from_le_bytes([
            ids[slot * 4],
            ids[slot * 4 + 1],
            ids[slot * 4 + 2],
            ids[slot * 4 + 3],
        ]);
        // -1 is the kernel's "slot unfilled" sentinel; it is reachable only if top_k exceeded
        // the expert count, which `validate` refuses. Treat it as zero rather than as an index.
        if id < 0 {
            continue;
        }
        let id = id as usize;
        if id >= cfg.num_experts {
            bail!(
                "GLM MoE: router selected expert {id} of {}",
                cfg.num_experts
            );
        }
        let Some(local) = cfg.local_slot(id) else {
            continue; // another rank owns it; its zero row is already in place.
        };
        let e = &w.experts[local];
        let dst = ws.expert_out.offset(slot * cfg.hidden * 2);
        let mi = cfg.moe_intermediate;
        w4a16_gemv(
            gpu,
            k.w4a16_gemv,
            x,
            &e.gate_proj,
            ws.a_gate,
            mi,
            cfg.hidden,
            stream,
        )?;
        w4a16_gemv(
            gpu,
            k.w4a16_gemv,
            x,
            &e.up_proj,
            ws.a_up,
            mi,
            cfg.hidden,
            stream,
        )?;
        swiglu(
            gpu,
            k.swiglu,
            ws.a_gate,
            ws.a_up,
            ws.a_act,
            mi,
            cfg.swiglu_limit,
            stream,
        )?;
        w4a16_gemv(
            gpu,
            k.w4a16_gemv,
            ws.a_act,
            &e.down_proj,
            dst,
            cfg.hidden,
            mi,
            stream,
        )?;
    }

    profile::end(profile::MOE_EXPERTS, t, gpu, stream);

    // ── shared expert: BF16, TP-sharded, NOT routed-scaled ──
    let t = profile::start();
    forward_dense(
        gpu,
        k,
        cfg,
        &w.shared,
        cfg.local_shared_intermediate,
        x,
        ws.shared_out,
        ws,
        stream,
    )?;

    // 🔴 The combine runs BEFORE the all-reduce, so the TP-partial shared expert and the
    // EP-partial routed sum reduce together in one collective. Adding the shared output after
    // a reduce — the `layers::moe` pattern, written for a replicated shared expert — would
    // keep only this rank's half of it.
    profile::end(profile::MOE_SHARED, t, gpu, stream);
    let t = profile::start();
    KernelLaunch::new(gpu, k.combine)
        .grid([1, 1, 1])
        .block([ACT_BLOCK, 1, 1])
        .arg_ptr(ws.expert_out)
        .arg_ptr(ws.wts)
        .arg_ptr(ws.shared_out)
        .arg_ptr(out)
        .arg_u32(cfg.hidden as u32)
        .arg_u32(cfg.top_k as u32)
        .launch(stream)?;
    profile::end(profile::MOE_COMBINE, t, gpu, stream);
    Ok(())
}
