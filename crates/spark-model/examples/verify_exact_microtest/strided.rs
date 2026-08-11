// SPDX-License-Identifier: AGPL-3.0-only
//! Strided (batch = n) leg of the exact-verify bitwise gate — split from
//! main.rs for the 500-LoC cap. See main.rs for the gate contract.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use crate::{
    CONV_DIM, CONV_ELEMS, D_CONV, EPS, H_ELEMS, Inputs, K, KD, KEY_DIM, Kernels, LegOut, NK, NV,
    QK_CH, QKVZ, VALUE_DIM, VD, dn, up,
};

/// Strided leg: N sequences, per-token one strided conv + one strided snap
/// launch. States sit on contiguous fake "pool slots".
pub(crate) fn run_strided(
    g: &dyn GpuBackend,
    ks: &Kernels,
    inps: &[Inputs],
) -> Result<Vec<LegOut>> {
    let n = inps.len();
    let state = up(
        g,
        &inps
            .iter()
            .flat_map(|i| i.conv0.clone())
            .collect::<Vec<u8>>(),
    )?;
    let h = up(
        g,
        &inps.iter().flat_map(|i| i.h0.clone()).collect::<Vec<u8>>(),
    )?;
    // Seq-major rows, exactly the production Multi layout.
    let mut deint_all = Vec::new();
    let mut gates_all = Vec::new();
    for i in inps {
        deint_all.extend_from_slice(&i.deint);
        gates_all.extend_from_slice(&i.gates);
    }
    let deint = up(g, &deint_all)?;
    let gates = up(g, &gates_all)?;
    let (w, nw) = (up(g, &inps[0].weight)?, up(g, &inps[0].norm_w)?);
    let conv_scratch = g.alloc(n * QKVZ * 4)?;
    let normed = g.alloc(n * K * VALUE_DIM * 2)?;
    // h intermediates: per-seq slab of (K-1) dense snapshots, contiguous.
    let inter_seq_stride = (K - 1) * H_ELEMS * 4;
    let h_inters = g.alloc(n * inter_seq_stride)?;
    for t in 0..K {
        KernelLaunch::new(g, ks.conv_f32_strided)
            .grid([CONV_DIM.div_ceil(256) as u32, n as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(state)
            .arg_ptr(deint.offset(t * QKVZ * 2))
            .arg_ptr(w)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(conv_scratch)
            .arg_u32(n as u32)
            .arg_u32(CONV_DIM as u32)
            .arg_u32(D_CONV as u32)
            .arg_u32(QK_CH as u32)
            .arg_u32(KD as u32)
            .arg_f32(EPS)
            .arg_u32((K * QKVZ) as u32)
            .arg_u32(QKVZ as u32)
            .launch(0)?;
        let snapshot = t + 1 < K;
        let (hi, stride) = if snapshot {
            (
                h_inters.offset(t * H_ELEMS * 4),
                (inter_seq_stride / 4) as u64,
            )
        } else {
            (DevicePtr::NULL, 0)
        };
        KernelLaunch::new(g, ks.snap_strided)
            .grid([NV as u32, n as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(h)
            .arg_ptr(conv_scratch)
            .arg_ptr(conv_scratch.offset(KEY_DIM * 4))
            .arg_ptr(conv_scratch.offset(KEY_DIM * 2 * 4))
            .arg_ptr(gates.offset(t * 2 * NV * 4))
            .arg_ptr(gates.offset((t * 2 * NV + NV) * 4))
            .arg_ptr(deint.offset((t * QKVZ + CONV_DIM) * 2))
            .arg_ptr(nw)
            .arg_ptr(normed.offset(t * VALUE_DIM * 2))
            .arg_u32(n as u32)
            .arg_u32(NK as u32)
            .arg_u32(NV as u32)
            .arg_u32(KD as u32)
            .arg_u32(VD as u32)
            .arg_u32(QKVZ as u32)
            .arg_u32(QKVZ as u32)
            .arg_u32((K * NV * 2) as u32)
            .arg_u32((K * QKVZ) as u32)
            .arg_u32((K * VALUE_DIM) as u32)
            .arg_f32(EPS)
            .arg_ptr(hi)
            .arg_u64(stride)
            .launch(0)?;
    }
    g.synchronize(0)?;
    let mut out = Vec::new();
    for i in 0..n {
        let mut his = Vec::new();
        for t in 0..K - 1 {
            his.push(dn(
                g,
                h_inters.offset(i * inter_seq_stride + t * H_ELEMS * 4),
                H_ELEMS * 4,
            )?);
        }
        out.push((
            dn(g, h.offset(i * H_ELEMS * 4), H_ELEMS * 4)?,
            dn(g, state.offset(i * CONV_ELEMS * 4), CONV_ELEMS * 4)?,
            dn(g, normed.offset(i * K * VALUE_DIM * 2), K * VALUE_DIM * 2)?,
            his,
            Vec::new(), // conv snapshots are d2d in production; not re-proven here
        ));
    }
    Ok(out)
}
