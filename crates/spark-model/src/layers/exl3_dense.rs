// SPDX-License-Identifier: AGPL-3.0-only

//! Layer-side carriers for natively-served EXL3 dense projections
//! (`ATLAS_EXL3_NATIVE_DENSE=1`): the GDN family on [`super::Qwen3SsmLayer`]
//! and the attention family on [`super::Qwen3AttentionLayer`].
//!
//! A carrier holds the projections as the kernels address them — the
//! ops-level [`Exl3DenseWeight`] (`trellis`/`suh`/`svh`/dims/K/kernel
//! codebook index), converted from the runtime's [`Exl3Weight`] at
//! construction — plus the model-shared [`Exl3DenseStage`] (f16 staging slabs
//! over the ONE per-model [`Exl3LaunchState`]: locks + fence + section mutex,
//! shared with the MoE arm). The layer dispatches through the carrier's
//! `*_linear` helpers, which are the SAME functions the parity example
//! exercises, so a layer site cannot drift from the proven arm.
//!
//! Construction validates each projection against the kernel envelope
//! ([`crate::weight_map::exl3_native_supported`]) and the layer geometry, and
//! probes every kernel instance the arm can select, so a layer only ever
//! carries a COMPLETE, servable, load-probed set.
//!
//! Milestone scope: the GDN carrier holds the WHOLE family — `in_proj_qkv`,
//! `in_proj_z` and `out_proj` — and the attention carrier the WHOLE
//! `q/k/v/o_proj` family (`weight_map::Exl3DenseFamily::leaves`); the
//! attention dispatch funnels live in `exl3_dense/attn_dispatch.rs`.
//!
//! GDN in-projection layout (the arena decision, design-map step 2): Atlas's
//! BF16 arm concatenates `in_proj_qkv [10240, 2560]` and `in_proj_z [6144,
//! 2560]` into ONE fused `[16384, 2560]` weight so a single GEMV/GEMM writes
//! `[M, 16384]` rows = `[Q|K|V (10240) | Z (6144)]`, and every consumer
//! (conv1d in-stride, the Z pitched copy, the per-token deinterleave
//! offsets) is parameterized by that row stride. Packed trellis weights
//! CANNOT be concatenated (each carries its own `suh`/`svh`, and `n`
//! differs), and `exl3_gemm` has no C row stride — so the native arm runs
//! TWO projections over ONE ingress ([`exl3_dense_linear_shared_a`]) and
//! lands each result in its column block of the SAME `[M, 16384]` arena row
//! through the STRIDED egress converters (`Exl3DenseOut::strided`, ld =
//! 16384, Z at column 10240). Chosen over re-laying the arena as a qkv block
//! followed by a z block because it needs ZERO consumer changes: the fused
//! row layout the conv1d / Z-copy / deinterleave sites rely on is preserved
//! byte-for-byte, only the producer differs. At M=1 (decode) the stride is
//! moot and the same call writes qkv at the arena base and z at `+10240`.

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;
use spark_runtime::weights::exl3::Exl3Weight;

// Attention-family dispatch funnels (`AttnProj`, `Exl3AttnWeights::{proj,
// qkv, kv, o_proj}_linear`) — sibling file, 500-LoC cap.
#[path = "exl3_dense/attn_dispatch.rs"]
mod attn_dispatch;
pub use attn_dispatch::AttnProj;

use super::ops::{
    Exl3DenseOut, Exl3DenseStage, Exl3DenseWeight, exl3_dense_linear, exl3_dense_linear_shared_a,
};

/// Resolve one projection from the store, check it against the dense
/// kernel envelope and the expected `[in -> out]` geometry, and convert it
/// to the kernel-facing descriptor.
fn resolve_checked(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    prefix: &str,
    in_dim: usize,
    out_dim: usize,
) -> Result<Exl3DenseWeight> {
    let w = Exl3Weight::from_store(gpu, store, prefix)
        .with_context(|| format!("EXL3 native dense: resolving {prefix}"))?;
    ensure!(
        crate::weight_map::exl3_native_supported(&w),
        "EXL3 native dense: {prefix} K={} cb={:?} [{}x{}] is outside the dense \
         kernel envelope (K in {{2,4}}, cb MCG/MUL1, dims %128) — the materialize \
         pass should have fallen this layer's family back to BF16",
        w.k_bits,
        w.cb,
        w.in_dim,
        w.out_dim,
    );
    ensure!(
        w.in_dim == in_dim && w.out_dim == out_dim,
        "EXL3 native dense: {prefix} trellis geometry [{} -> {}] does not match the \
         model config's [{in_dim} -> {out_dim}]",
        w.in_dim,
        w.out_dim,
    );
    Exl3DenseWeight::from_exl3(&w).with_context(|| format!("EXL3 native dense: {prefix}"))
}

/// Resolve the `exl3_matmul` instances a projection can dispatch to — the
/// small-row GEMV set (m=1 and 2..=8 modes x narrow/wide cfg x f16/f32 C,
/// the exact name rule of `ops::exl3_gemv`) and the universal shape-2 GEMM
/// (f16 and f32 C) — so a missing module or a JIT-compile failure is paid
/// at load, never on the first request. Mirrors the lm_head / MoE probes.
fn probe_kernels(gpu: &dyn GpuBackend, w: &Exl3DenseWeight, what: &str) -> Result<()> {
    let (k, cb) = (w.k_bits, w.cb);
    // The shape the Blackwell heuristic actually selects for this geometry
    // (shape 3 for n in {6144, 10240, 12288}) plus the universal shape-2
    // fallback — probing only sh2 left the first prefill to discover sh3.
    let picked = crate::layers::ops::select_exl3_gemm_shape(w.in_dim, w.out_dim, k, false, 1, 1);
    let mut names = vec![
        format!("exl3_gemm_k{k}_cb{cb}_sh2_f16"),
        format!("exl3_gemm_k{k}_cb{cb}_sh2_f32"),
    ];
    if picked != 2 && crate::layers::ops::exl3_gemm_shape_compat(picked, w.in_dim, w.out_dim) {
        names.push(format!("exl3_gemm_k{k}_cb{cb}_sh{picked}_f16"));
        names.push(format!("exl3_gemm_k{k}_cb{cb}_sh{picked}_f32"));
    }
    for mmode in [0, 1] {
        for cfg in [0, 1] {
            for suf in ["f16", "f32"] {
                names.push(format!("exl3_gemv_k{k}_cb{cb}_m{mmode}_cfg{cfg}_{suf}"));
            }
        }
    }
    for name in names {
        gpu.kernel("exl3_matmul", &name).with_context(|| {
            format!(
                "EXL3 native dense ({what}) needs exl3_matmul::{name} (gb10 targets \
                 only) — unset ATLAS_EXL3_NATIVE_DENSE on this target"
            )
        })?;
    }
    Ok(())
}

/// The stage geometry must cover a projection or the arm refuses at the
/// first call — check at install instead.
fn check_stage_fits(stage: &Exl3DenseStage, w: &Exl3DenseWeight, what: &str) -> Result<()> {
    ensure!(
        w.in_dim <= stage.max_in && w.out_dim <= stage.max_out,
        "EXL3 native dense ({what}): [{} -> {}] exceeds the model's dense stage \
         (max_in {} / max_out {}) — the loader must size the stage from model-wide maxima",
        w.in_dim,
        w.out_dim,
        stage.max_in,
        stage.max_out,
    );
    Ok(())
}

/// The natively-served GDN projections of one `Qwen3SsmLayer`: the whole
/// family, kept atomically.
///
/// * `in_proj_qkv` `[hidden -> conv_dim]` + `in_proj_z` `[hidden ->
///   value_dim]` — served as a shared-A pair into the fused `[m, conv_dim +
///   value_dim]` `[Q|K|V|Z]` arena row (see the module docs for why the row
///   layout is preserved rather than the weights concatenated).
/// * `out_proj` `[value_dim -> hidden]` — every GDN site writes its output
///   as a contiguous `[m, hidden]` BF16 row block.
#[derive(Debug, Clone)]
pub struct Exl3GdnWeights {
    /// `[hidden -> 2*key_dim + value_dim]` (the conv / QKV width).
    pub in_proj_qkv: Exl3DenseWeight,
    /// `[hidden -> value_dim]` (the Z gate).
    pub in_proj_z: Exl3DenseWeight,
    /// `[value_dim -> hidden]`.
    pub out_proj: Exl3DenseWeight,
    /// Model-shared staging + launch state (locks/fence/section).
    pub stage: Arc<Exl3DenseStage>,
}

impl Exl3GdnWeights {
    /// Resolve `{lp}.linear_attn.{in_proj_qkv, in_proj_z, out_proj}`,
    /// validate each against the (TP-full) GDN geometry (`conv_dim` =
    /// `2*key_dim + value_dim`), probe their kernels, and bind the
    /// model-shared stage.
    pub fn from_store(
        gpu: &dyn GpuBackend,
        store: &WeightStore,
        lp: &str,
        hidden: usize,
        conv_dim: usize,
        value_dim: usize,
        stage: Arc<Exl3DenseStage>,
    ) -> Result<Self> {
        let p = format!("{lp}.linear_attn");
        let in_proj_qkv =
            resolve_checked(gpu, store, &format!("{p}.in_proj_qkv"), hidden, conv_dim)?;
        let in_proj_z = resolve_checked(gpu, store, &format!("{p}.in_proj_z"), hidden, value_dim)?;
        let out_proj = resolve_checked(gpu, store, &format!("{p}.out_proj"), value_dim, hidden)?;
        for (w, what) in [
            (&in_proj_qkv, "GDN in_proj_qkv"),
            (&in_proj_z, "GDN in_proj_z"),
            (&out_proj, "GDN out_proj"),
        ] {
            probe_kernels(gpu, w, what)?;
            check_stage_fits(&stage, w, what)?;
        }
        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            out_proj,
            stage,
        })
    }

    /// Row width (elements) of the fused `[Q|K|V|Z]` arena row the in-proj
    /// pair writes — must equal the model's `ssm_qkvz_size()`.
    pub fn qkvz_row_elems(&self) -> usize {
        self.in_proj_qkv.out_dim + self.in_proj_z.out_dim
    }

    /// Resident packed bytes of the natively-served projections.
    pub fn packed_bytes(&self) -> usize {
        self.in_proj_qkv.packed_bytes()
            + self.in_proj_z.packed_bytes()
            + self.out_proj.packed_bytes()
    }

    /// What the same projections cost materialized as BF16.
    pub fn bf16_bytes(&self) -> usize {
        self.in_proj_qkv.bf16_bytes() + self.in_proj_z.bf16_bytes() + self.out_proj.bf16_bytes()
    }

    /// `arena[m, conv_dim + value_dim] = [a @ in_proj_qkv | a @ in_proj_z]`
    /// — BF16 `a[m, hidden]` in, the fused sequential `[Q|K|V|Z]` BF16 rows
    /// out (row stride = `qkvz_row_elems()`, Z block at column `conv_dim`),
    /// any `m`. ONE ingress, two matmuls, strided egress per block, ONE
    /// launch section; stream-ordered, no host sync, no allocation. This is
    /// the funnel for the M=1 decode, batched-decode, multi-seq and prefill
    /// QKVZ sites, so the arena layout every downstream consumer (conv1d
    /// in-stride, Z pitched copy) relies on is produced in exactly one place.
    ///
    /// Cooperative launches are not graph-capturable: the calling site must
    /// have refused `graph_capture` (see `out_proj_linear`).
    pub fn in_proj_linear(
        &self,
        gpu: &dyn GpuBackend,
        a_bf16: DevicePtr,
        arena_bf16: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        let ld = self.qkvz_row_elems();
        let z_col = self.in_proj_qkv.out_dim;
        exl3_dense_linear_shared_a(
            gpu,
            &[
                (self.in_proj_qkv, Exl3DenseOut::strided(arena_bf16, ld)),
                (
                    self.in_proj_z,
                    Exl3DenseOut::strided(arena_bf16.offset(z_col * 2), ld),
                ),
            ],
            a_bf16,
            m,
            &self.stage,
            stream,
        )
        .context("EXL3 native GDN in_proj_qkv + in_proj_z")
    }

    /// `dst[m, hidden] = a[m, value_dim] @ out_proj` — BF16 in, contiguous
    /// BF16 out, any `m` (GEMV tier at m <= 8, row-batched GEMM above). ONE
    /// launch section; stream-ordered, no host sync, no allocation.
    ///
    /// Cooperative launches are not graph-capturable: the calling site must
    /// have refused `graph_capture` (the layer's `exl3_graph_veto` keeps the
    /// capturing decode paths away from this layer in the first place).
    pub fn out_proj_linear(
        &self,
        gpu: &dyn GpuBackend,
        a_bf16: DevicePtr,
        dst_bf16: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        exl3_dense_linear(
            gpu,
            &self.out_proj,
            a_bf16,
            // Residual-bound: fp32 C on the GEMM tier (upstream out_dtype).
            Exl3DenseOut::contiguous(dst_bf16).with_fp32(),
            m,
            &self.stage,
            stream,
        )
        .context("EXL3 native GDN out_proj")
    }
}

/// The attention family of one `Qwen3AttentionLayer`, kept packed. `q_proj`
/// is the gated `[Q|gate]`-interleaved projection exactly as the checkpoint
/// packs it (the existing `deinterleave_qg` step follows the launch); `k`,
/// `v`, `o` are the plain projections. Every attention site — decode
/// (single- and multi-seq) and prefill (paged and cache-skip) — dispatches
/// through the funnels in `exl3_dense/attn_dispatch.rs` (`proj_linear`,
/// `qkv_linear`, `kv_linear`, `o_proj_linear`), the same functions the
/// parity example's leg J proves.
#[derive(Debug, Clone)]
pub struct Exl3AttnWeights {
    /// `[hidden -> num_heads * head_dim * (2 if gated else 1)]`.
    pub q_proj: Exl3DenseWeight,
    /// `[hidden -> num_kv_heads * head_dim]`.
    pub k_proj: Exl3DenseWeight,
    /// `[hidden -> num_kv_heads * head_dim]`.
    pub v_proj: Exl3DenseWeight,
    /// `[num_heads * head_dim -> hidden]`.
    pub o_proj: Exl3DenseWeight,
    /// Model-shared staging + launch state (locks/fence/section).
    pub stage: Arc<Exl3DenseStage>,
}

impl Exl3AttnWeights {
    /// Resolve `{lp}.self_attn.{q,k,v,o}_proj` and validate them against the
    /// (TP-full) attention geometry. Probes the kernels each will use.
    #[allow(clippy::too_many_arguments)]
    pub fn from_store(
        gpu: &dyn GpuBackend,
        store: &WeightStore,
        lp: &str,
        hidden: usize,
        q_proj_n: usize,
        kv_n: usize,
        o_in: usize,
        stage: Arc<Exl3DenseStage>,
    ) -> Result<Self> {
        let p = format!("{lp}.self_attn");
        let q_proj = resolve_checked(gpu, store, &format!("{p}.q_proj"), hidden, q_proj_n)?;
        let k_proj = resolve_checked(gpu, store, &format!("{p}.k_proj"), hidden, kv_n)?;
        let v_proj = resolve_checked(gpu, store, &format!("{p}.v_proj"), hidden, kv_n)?;
        let o_proj = resolve_checked(gpu, store, &format!("{p}.o_proj"), o_in, hidden)?;
        for (w, what) in [
            (&q_proj, "attention q_proj"),
            (&k_proj, "attention k_proj"),
            (&v_proj, "attention v_proj"),
            (&o_proj, "attention o_proj"),
        ] {
            probe_kernels(gpu, w, what)?;
            check_stage_fits(&stage, w, what)?;
        }
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            stage,
        })
    }

    /// Resident packed bytes of the four projections.
    pub fn packed_bytes(&self) -> usize {
        self.q_proj.packed_bytes()
            + self.k_proj.packed_bytes()
            + self.v_proj.packed_bytes()
            + self.o_proj.packed_bytes()
    }

    /// What the same projections cost materialized as BF16.
    pub fn bf16_bytes(&self) -> usize {
        self.q_proj.bf16_bytes()
            + self.k_proj.bf16_bytes()
            + self.v_proj.bf16_bytes()
            + self.o_proj.bf16_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::ops::Exl3LaunchState;
    use spark_runtime::gpu::mock::MockGpuBackend;

    fn mock_weight(gpu: &MockGpuBackend, k: usize, n: usize) -> Exl3DenseWeight {
        Exl3DenseWeight {
            trellis: gpu.alloc((k / 16) * (n / 16) * 16 * 4 * 2).unwrap(),
            suh: gpu.alloc(k * 2).unwrap(),
            svh: gpu.alloc(n * 2).unwrap(),
            in_dim: k,
            out_dim: n,
            k_bits: 4,
            cb: 2,
        }
    }

    #[test]
    fn gdn_out_proj_linear_takes_the_dense_arm() {
        // The carrier's helper IS `exl3_dense_linear` over a contiguous
        // destination: the mock's kernel-resolution log shows the GEMV tier
        // plan at m=1 and the f16-C GEMM plan at m=64 (numerics are the GPU
        // parity example's job, leg I).
        let gpu = MockGpuBackend::new();
        let launch = Arc::new(Exl3LaunchState::new(&gpu).unwrap());
        let stage =
            Arc::new(Exl3DenseStage::new_with_fp32(&gpu, launch, 256, 6144, 10240, 2560).unwrap());
        let out_proj = mock_weight(&gpu, 6144, 2560);
        assert_eq!(out_proj.bf16_bytes(), 6144 * 2560 * 2);
        assert_eq!(
            out_proj.packed_bytes(),
            6144 * 2560 * 4 / 8 + (6144 + 2560) * 2 + 4
        );
        let g = Exl3GdnWeights {
            in_proj_qkv: mock_weight(&gpu, 2560, 10240),
            in_proj_z: mock_weight(&gpu, 2560, 6144),
            out_proj,
            stage,
        };
        assert_eq!(g.qkvz_row_elems(), 16384);
        assert_eq!(
            g.bf16_bytes(),
            (6144 * 2560 + 2560 * 10240 + 2560 * 6144) * 2
        );
        let a = gpu.alloc(64 * 6144 * 2).unwrap();
        let dst = gpu.alloc(64 * 2560 * 2).unwrap();
        let names = |from: usize| -> Vec<String> {
            gpu.kernel_lookups_snapshot()[from..]
                .iter()
                .map(|(_, f)| f.clone())
                .collect()
        };
        let n0 = gpu.kernel_lookups_snapshot().len();
        g.out_proj_linear(&gpu, a, dst, 1, 0).unwrap();
        let plan = names(n0);
        assert_eq!(plan[0], "exl3_bf16_to_f16");
        // At [6144 -> 2560] K=4 the GEMV heuristic declines (n < 8192 with
        // k > 2048), so the small-m tier is the f32-C split-K GEMM.
        assert!(
            plan[1].starts_with("exl3_gemm_k4_cb2_sh") && plan[1].ends_with("_f32"),
            "{plan:?}"
        );
        assert_eq!(plan[2], "exl3_f32_to_bf16");
        let n1 = gpu.kernel_lookups_snapshot().len();
        g.out_proj_linear(&gpu, a, dst, 64, 0).unwrap();
        let plan = names(n1);
        assert_eq!(plan[0], "exl3_bf16_to_f16");
        // Residual-bound: the GEMM tier keeps fp32 C (upstream out_dtype).
        assert!(
            plan[1].starts_with("exl3_gemm_k4_cb2_sh") && plan[1].ends_with("_f32"),
            "{plan:?}"
        );
        assert_eq!(plan[2], "exl3_f32_to_bf16");
        // A stage too narrow for the projection is refused at install.
        let narrow = Exl3DenseStage::new(&gpu, g.stage.launch.clone(), 256, 2560, 2560).unwrap();
        assert!(check_stage_fits(&narrow, &g.out_proj, "x").is_err());
        assert!(check_stage_fits(&narrow, &g.in_proj_qkv, "x").is_err());

        // The in-proj pair: ONE ingress, two matmuls, and a STRIDED egress
        // per block (never the in-place 1-D convert — the arena row is wider
        // than either block) at both tiers.
        let a2 = gpu.alloc(64 * 2560 * 2).unwrap();
        let arena = gpu.alloc(64 * 16384 * 2).unwrap();
        let n2 = gpu.kernel_lookups_snapshot().len();
        g.in_proj_linear(&gpu, a2, arena, 1, 0).unwrap();
        let plan = names(n2);
        assert_eq!(plan[0], "exl3_bf16_to_f16");
        assert_eq!(plan.iter().filter(|n| *n == "exl3_bf16_to_f16").count(), 1);
        // [2560 -> 10240] takes the GEMV proper; [2560 -> 6144] declines to
        // the f32-C GEMM at K=4 — both end in the 2-D f32 egress.
        assert!(plan[1].starts_with("exl3_gemv_k4_cb2_m0_"), "{plan:?}");
        assert_eq!(plan[2], "exl3_f32_to_bf16_2d");
        assert_eq!(plan[4], "exl3_f32_to_bf16_2d");
        assert!(!plan.iter().any(|n| n == "exl3_f32_to_bf16"), "{plan:?}");
        let n3 = gpu.kernel_lookups_snapshot().len();
        g.in_proj_linear(&gpu, a2, arena, 64, 0).unwrap();
        let plan = names(n3);
        assert_eq!(plan.len(), 5, "{plan:?}");
        assert_eq!(plan[0], "exl3_bf16_to_f16");
        for i in [1, 3] {
            assert!(
                plan[i].starts_with("exl3_gemm_k4_cb2_sh") && plan[i].ends_with("_f16"),
                "{plan:?}"
            );
            assert_eq!(plan[i + 1], "exl3_f16_to_bf16_2d", "{plan:?}");
        }
    }
}
