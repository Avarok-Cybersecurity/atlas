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

const W4_TILE: u32 = 64;
const ACT_BLOCK: u32 = 256;

/// `C[M, N] = A[M, K] @ B[N, K]^T`, BF16 in and out.
#[allow(clippy::too_many_arguments)]
fn gemm(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    gemv: KernelHandle,
    batchm: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    // M=1 decode -> GEMV; M=2..8 (a K-token verify sweep) -> ONE weight read for all rows;
    // wider -> the tile GEMM. `ops::dense_mm_bf16` owns the policy and the grid coupling.
    //
    // 🔴 The router is the worst tile-GEMM case in the whole stack: N = 288 tiles to **18
    // blocks**, measured 6.8 GB/s. It has no FP32-out batchm twin, so it stays on gemv/tile.
    crate::layers::ops::dense_mm_bf16(
        gpu,
        &crate::layers::ops::DenseMmKernels {
            gemm: k,
            gemv,
            batchm,
        },
        a,
        b,
        c,
        m,
        n,
        kk,
        stream,
    )
}

/// `C[1, N] = A[1, K] @ dequant(B)[N, K]^T` — the M=1 decode kernel.
///
/// Same NVFP4 operand triple as [`w4a16`], one output row. Used for every routed-expert
/// projection because they are all M=1 and the tile GEMM measured 9.7 GB/s there.
fn w4a16_gemv(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    k_sw: KernelHandle,
    a: DevicePtr,
    w: &Nvfp4Proj,
    c: DevicePtr,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    // Prefer the single-warp sibling when the target carries it. BIT-IDENTICAL — the
    // per-orig-lane partial is the same function, the shuffle tree is the same tree, and
    // the final combine is the same two-term FP32 add; only the block packing differs.
    // This site launched the base kernel directly and so had never picked up the SW win
    // that `ops::w4a16_decode_gemv` has been handing every other decode GEMV.
    if k_sw.0 != 0 {
        return crate::layers::ops::w4a16_gemv_sw_raw(
            gpu, k_sw, a, w.packed, w.scale, w.scale_2, c, n as u32, kk as u32, stream,
        );
    }
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
    /// `[rows, max_inter]` BF16 ×3 — gate, up, activated. Shared by every projection pair.
    a_gate: DevicePtr,
    a_up: DevicePtr,
    a_act: DevicePtr,
    /// `[rows, num_experts]` F32 router logits.
    logits: DevicePtr,
    /// `[rows, top_k]` I32 / F32 selection.
    ids: DevicePtr,
    wts: DevicePtr,
    /// `[rows, top_k, hidden]` BF16. 🪤 Fully written every step — remote and invalid slots are
    /// memset to zero before the loop, never left stale.
    expert_out: DevicePtr,
    /// `[rows, hidden]` BF16 shared-expert output.
    shared_out: DevicePtr,
    max_inter: usize,
    /// Widest verify this scratch can serve. `1` on the serial decode path.
    max_rows: usize,
}

impl Glm5NextMlpWorkspace {
    pub fn new(gpu: &dyn GpuBackend, cfg: &Glm5NextMlpConfig, max_rows: usize) -> Result<Self> {
        let rows = max_rows.max(1);
        let max_inter = cfg
            .local_dense_intermediate
            .max(cfg.moe_intermediate)
            .max(cfg.local_shared_intermediate)
            .max(1);
        // The grouped MoE path activates all `top_k` slots in one launch, so the three
        // activation buffers are slot-major and `top_k` times a routed expert's width. The
        // dense path still writes only the first `inter` elements — `max_inter` stays the
        // guard for it.
        // The dense/shared arm is now `[rows, inter]`; the grouped MoE arm is still one row's
        // `top_k` slots at a time. Both share these buffers, so take the wider.
        let act_elems = (rows * max_inter).max(cfg.top_k * cfg.moe_intermediate).max(1);
        Ok(Self {
            a_gate: gpu.alloc(act_elems * 2)?,
            a_up: gpu.alloc(act_elems * 2)?,
            a_act: gpu.alloc(act_elems * 2)?,
            logits: gpu.alloc(rows * cfg.num_experts * 4)?,
            ids: gpu.alloc(rows * cfg.top_k * 4)?,
            wts: gpu.alloc(rows * cfg.top_k * 4)?,
            expert_out: gpu.alloc(rows * cfg.top_k * cfg.hidden * 2)?,
            shared_out: gpu.alloc(rows * cfg.hidden * 2)?,
            max_inter,
            max_rows: rows,
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
    m: usize,
    ws: &Glm5NextMlpWorkspace,
    stream: u64,
) -> Result<()> {
    if inter == 0 || inter > ws.max_inter {
        bail!(
            "GLM dense MLP: width {inter} does not fit a workspace built for {}",
            ws.max_inter
        );
    }
    if m == 0 || m > ws.max_rows {
        bail!(
            "GLM dense MLP: {m} rows do not fit a workspace built for {}",
            ws.max_rows
        );
    }
    gemm(
        gpu,
        k.gemm,
        k.gemv,
        k.gemv_batchm,
        x,
        w.gate_proj,
        ws.a_gate,
        m,
        inter,
        cfg.hidden,
        stream,
    )?;
    gemm(
        gpu, k.gemm, k.gemv, k.gemv_batchm, x, w.up_proj, ws.a_up, m, inter, cfg.hidden, stream,
    )?;
    swiglu(
        gpu,
        k.swiglu,
        ws.a_gate,
        ws.a_up,
        ws.a_act,
        // Elementwise over the whole `[m, inter]` block — bit-identical to m launches of
        // `inter`, because every output element depends only on its own gate/up pair.
        m * inter,
        cfg.swiglu_limit,
        stream,
    )?;
    gemm(
        gpu,
        k.gemm,
        k.gemv,
        k.gemv_batchm,
        ws.a_act,
        w.down_proj,
        out,
        m,
        cfg.hidden,
        inter,
        stream,
    )
}

/// `C[top_k, N] = A @ dequant(expert[ids[slot]])^T` — every routed slot in ONE launch.
///
/// Bit-identical per slot to the [`w4a16_gemv`] loop it replaces; see the kernel's header.
/// Slots whose expert this rank does not own are skipped, so `c` must already hold whatever
/// those rows should contribute (zero, for the routed sum).
#[allow(clippy::too_many_arguments)]
fn w4a16_gemv_moe(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    t: &super::weights::Glm5NextExpertPtrTable,
    c: DevicePtr,
    ids: DevicePtr,
    n: usize,
    kk: usize,
    top_k: usize,
    num_experts: usize,
    input_stride: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        // 🪤 COUPLED to the kernel's `N_PER_BLOCK_SW` = 8, and grid.y IS the slot.
        .grid([
            crate::layers::ops::w4a16_gemv_sw_grid_x(n as u32),
            top_k as u32,
            1,
        ])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(t.packed_ptrs)
        .arg_ptr(t.scale_ptrs)
        .arg_ptr(t.scale2_vals)
        .arg_ptr(c)
        .arg_ptr(ids)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .arg_u32(num_experts as u32)
        .arg_u32(input_stride as u32)
        .launch(stream)
}

/// Kill switch for the grouped path: `ATLAS_GLM_MOE_HOST_DISPATCH=1` restores the
/// read-ids-to-host expert loop. Read once — this sits on the per-layer decode path.
fn host_dispatch_forced() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("ATLAS_GLM_MOE_HOST_DISPATCH").as_deref() == Ok("1"))
}

/// Say once which expert-dispatch path this process took. A missing `w4a16_gemv_sw_moe`
/// entry point falls back SILENTLY otherwise, and the fallback is the slow one.
fn announce_dispatch(grouped: bool) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if grouped {
            tracing::info!("GLM MoE: grouped device dispatch (no per-layer D2H)");
        } else {
            tracing::info!("GLM MoE: host dispatch (per-layer stream sync + D2H)");
        }
    });
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
/// Routed MoE over `rows` rows.
///
/// 🔴 The routed experts are the one span that genuinely does NOT amortize over a verify's
/// rows. Measured on the live routing trace (42 series x 406 steps), the union of selected
/// experts over K consecutive tokens is 8.00 / 13.74 / 18.76 / 23.35 at K = 1..4 — so their
/// weight traffic scales with K however the loop is written, and they stay one row at a time.
/// (Deduplicating the union would recover the difference between 8K and that curve; it needs a
/// device-side sort the dispatch kernel does not have yet.)
///
/// The SHARED expert is a different animal: it is the same weights for every row, so it runs
/// once over all of them.
#[allow(clippy::too_many_arguments)]
pub fn forward_moe(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextMoeWeights,
    x: DevicePtr,
    out: DevicePtr,
    rows: usize,
    ws: &Glm5NextMlpWorkspace,
    stream: u64,
) -> Result<()> {
    if rows == 0 || rows > ws.max_rows {
        bail!(
            "GLM MoE: {rows} rows do not fit a workspace built for {}",
            ws.max_rows
        );
    }
    if w.experts.len() != cfg.local_experts {
        bail!(
            "GLM MoE: {} bound experts but this rank owns {} of {}",
            w.experts.len(),
            cfg.local_experts,
            cfg.num_experts
        );
    }

    use crate::layers::glm5next_layer::profile;

    for r in 0..rows {
        let xr = x.offset(r * cfg.hidden * 2);
        let logits_r = ws.logits.offset(r * cfg.num_experts * 4);
        let ids_r = ws.ids.offset(r * cfg.top_k * 4);
        let wts_r = ws.wts.offset(r * cfg.top_k * 4);
        let expert_out_r = ws.expert_out.offset(r * cfg.top_k * cfg.hidden * 2);
        // ── router: FULL expert set, FP32 logits, replicated on every rank ──
        let t = profile::start();
        gemm(
            gpu,
            k.gemm_f32,
            k.gemv_f32,
            // No FP32-out batchm twin exists; the router stays on gemv/tile.
            KernelHandle(0),
            xr,
            w.router,
            logits_r,
            1,
            cfg.num_experts,
            cfg.hidden,
            stream,
        )?;
        KernelLaunch::new(gpu, k.router)
            .grid([1, 1, 1])
            .block([ACT_BLOCK, 1, 1])
            .arg_ptr(logits_r)
            .arg_ptr(w.router_bias)
            .arg_ptr(ids_r)
            .arg_ptr(wts_r)
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
        gpu.memset_async(expert_out_r, 0, cfg.top_k * cfg.hidden * 2, stream)?;

        let grouped = !host_dispatch_forced() && k.w4a16_gemv_sw_moe.0 != 0;
        announce_dispatch(grouped);
        if grouped {
            // ── grouped, device-dispatched: routing never leaves the GPU ──
            //
            // 🔴 The host loop this replaces did `synchronize` + `copy_d2h(ids)` once per routed
            // layer — 42 full stream drains per decode token on GLM-5.3, and the reason the
            // decode step could not be graph-captured. It also issued one launch per LOCAL
            // expert per projection (~16/layer); this is four, whatever the routing picks.
            let t = profile::start();
            let mi = cfg.moe_intermediate;
            // gate and up: every slot reads the SAME x, so input_stride = 0.
            w4a16_gemv_moe(
                gpu,
                k.w4a16_gemv_sw_moe,
                xr,
                &w.ptrs.gate,
                ws.a_gate,
                ids_r,
                mi,
                cfg.hidden,
                cfg.top_k,
                cfg.num_experts,
                0,
                stream,
            )?;
            w4a16_gemv_moe(
                gpu,
                k.w4a16_gemv_sw_moe,
                xr,
                &w.ptrs.up,
                ws.a_up,
                ids_r,
                mi,
                cfg.hidden,
                cfg.top_k,
                cfg.num_experts,
                0,
                stream,
            )?;
            // Elementwise over all slots at once. Remote slots activate uninitialised rows; the
            // down projection skips them, so those rows are never read.
            swiglu(
                gpu,
                k.swiglu,
                ws.a_gate,
                ws.a_up,
                ws.a_act,
                cfg.top_k * mi,
                cfg.swiglu_limit,
                stream,
            )?;
            // down: slot-major activations, so input_stride = one expert's width.
            w4a16_gemv_moe(
                gpu,
                k.w4a16_gemv_sw_moe,
                ws.a_act,
                &w.ptrs.down,
                expert_out_r,
                ids_r,
                cfg.hidden,
                mi,
                cfg.top_k,
                cfg.num_experts,
                mi,
                stream,
            )?;
            profile::end(profile::MOE_EXPERTS, t, gpu, stream);

            if profile::trace_on() {
                let mut ids = vec![0u8; cfg.top_k * 4];
                gpu.synchronize(stream)?;
                gpu.copy_d2h(ids_r, &mut ids)?;
                let decoded: Vec<i32> = (0..cfg.top_k)
                    .map(|k| {
                        i32::from_le_bytes([ids[k * 4], ids[k * 4 + 1], ids[k * 4 + 2], ids[k * 4 + 3]])
                    })
                    .collect();
                profile::stash_route(&decoded);
            }
        } else {
            // 🚩 A FULL STREAM SYNC + D2H IN THE MIDDLE OF EVERY MoE LAYER. The routing decision
            // is read back to the host so the expert GEMMs can be launched by id. Timed on its own
            // because it is the one span here that is pure latency and scales with layer count,
            // not with weight bytes.
            let t = profile::start();
            let mut ids = vec![0u8; cfg.top_k * 4];
            gpu.synchronize(stream)?;
            gpu.copy_d2h(ids_r, &mut ids)?;
            profile::end(profile::MOE_HOSTSYNC, t, gpu, stream);

            if profile::trace_on() {
                let decoded: Vec<i32> = (0..cfg.top_k)
                    .map(|k| {
                        i32::from_le_bytes([ids[k * 4], ids[k * 4 + 1], ids[k * 4 + 2], ids[k * 4 + 3]])
                    })
                    .collect();
                profile::stash_route(&decoded);
            }

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
                let dst = expert_out_r.offset(slot * cfg.hidden * 2);
                let mi = cfg.moe_intermediate;
                w4a16_gemv(
                    gpu,
                    k.w4a16_gemv,
                    k.w4a16_gemv_sw,
                    xr,
                    &e.gate_proj,
                    ws.a_gate,
                    mi,
                    cfg.hidden,
                    stream,
                )?;
                w4a16_gemv(
                    gpu,
                    k.w4a16_gemv,
                    k.w4a16_gemv_sw,
                    xr,
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
                    k.w4a16_gemv_sw,
                    ws.a_act,
                    &e.down_proj,
                    dst,
                    cfg.hidden,
                    mi,
                    stream,
                )?;
            }

            profile::end(profile::MOE_EXPERTS, t, gpu, stream);
        }

    }

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
        rows,
        ws,
        stream,
    )?;

    // 🔴 The combine runs BEFORE the all-reduce, so the TP-partial shared expert and the
    // EP-partial routed sum reduce together in one collective. Adding the shared output after
    // a reduce — the `layers::moe` pattern, written for a replicated shared expert — would
    // keep only this rank's half of it.
    profile::end(profile::MOE_SHARED, t, gpu, stream);
    let t = profile::start();
    for r in 0..rows {
        KernelLaunch::new(gpu, k.combine)
            .grid([1, 1, 1])
            .block([ACT_BLOCK, 1, 1])
            .arg_ptr(ws.expert_out.offset(r * cfg.top_k * cfg.hidden * 2))
            .arg_ptr(ws.wts.offset(r * cfg.top_k * 4))
            .arg_ptr(ws.shared_out.offset(r * cfg.hidden * 2))
            .arg_ptr(out.offset(r * cfg.hidden * 2))
            .arg_u32(cfg.hidden as u32)
            .arg_u32(cfg.top_k as u32)
            .launch(stream)?;
    }
    profile::end(profile::MOE_COMBINE, t, gpu, stream);
    Ok(())
}
