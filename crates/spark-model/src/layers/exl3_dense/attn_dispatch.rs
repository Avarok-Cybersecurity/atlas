// SPDX-License-Identifier: AGPL-3.0-only

//! Dispatch funnels of the natively-served attention family
//! ([`Exl3AttnWeights`]): the ONE set of functions every
//! `Qwen3AttentionLayer` q/k/v/o site calls — decode (single-seq and
//! multi-seq) and prefill (paged and cache-skip) — and the set the parity
//! example (leg J) exercises, so a layer site cannot drift from the proven
//! arm. Sibling of `exl3_dense.rs` (500-LoC cap).
//!
//! Output contracts (the sites' existing buffer layouts, unchanged):
//!
//!  * `q_proj` writes the RAW `[Q|gate]`-interleaved row exactly as the
//!    checkpoint packs it (HF column order); the site's existing
//!    `deinterleave_qg` follows, as upstream does (`py_attn.py:552-566`).
//!  * `k_proj` / `v_proj` / `o_proj` write plain `[m, out_dim]` blocks.
//!  * Every destination may be pitched ([`Exl3DenseOut::strided`]) — the
//!    multi-seq decode `qkv_buf` is `[n, Q|K|V]` rows `per_seq_qkv` apart.
//!
//! Every funnel is ONE launch section (host mutex + device fence, shared
//! with the MoE and GDN arms) and never allocates or syncs. Cooperative
//! launches are not graph-capturable: the calling site refuses
//! `graph_capture` (`Qwen3AttentionLayer::exl3_attn_arm`) and the layer's
//! `exl3_graph_veto` keeps the capturing paths away in the first place.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::Exl3AttnWeights;
use crate::layers::ops::{
    Exl3DenseOut, Exl3DenseWeight, exl3_dense_linear, exl3_dense_linear_shared_a,
};

/// One of the four attention projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnProj {
    Q,
    K,
    V,
    O,
}

impl AttnProj {
    pub fn name(self) -> &'static str {
        match self {
            Self::Q => "q_proj",
            Self::K => "k_proj",
            Self::V => "v_proj",
            Self::O => "o_proj",
        }
    }
}

impl Exl3AttnWeights {
    /// The kernel-facing descriptor of one projection.
    pub fn weight(&self, p: AttnProj) -> &Exl3DenseWeight {
        match p {
            AttnProj::Q => &self.q_proj,
            AttnProj::K => &self.k_proj,
            AttnProj::V => &self.v_proj,
            AttnProj::O => &self.o_proj,
        }
    }

    /// `out[m, out_dim] = a[m, in_dim] @ p` — BF16 in (contiguous rows),
    /// BF16 out (contiguous or pitched), any `m` (GEMV tier at m <= 8,
    /// row-batched GEMM above). The generic single-projection arm the
    /// prefill sites use per projection (so their LoRA fold still follows
    /// each projection unchanged).
    pub fn proj_linear(
        &self,
        gpu: &dyn GpuBackend,
        p: AttnProj,
        a_bf16: DevicePtr,
        out: Exl3DenseOut,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        exl3_dense_linear(gpu, self.weight(p), a_bf16, out, m, &self.stage, stream)
            .with_context(|| format!("EXL3 native attention {}", p.name()))
    }

    /// Q, K and V over the SAME `normed` activation in ONE section: one
    /// ingress, three matmuls. The multi-seq decode arm (`[n, Q|K|V]` rows,
    /// pitched destinations) and any site that owns all three destinations.
    pub fn qkv_linear(
        &self,
        gpu: &dyn GpuBackend,
        a_bf16: DevicePtr,
        q_out: Exl3DenseOut,
        k_out: Exl3DenseOut,
        v_out: Exl3DenseOut,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        exl3_dense_linear_shared_a(
            gpu,
            &[
                (self.q_proj, q_out),
                (self.k_proj, k_out),
                (self.v_proj, v_out),
            ],
            a_bf16,
            m,
            &self.stage,
            stream,
        )
        .context("EXL3 native attention q/k/v")
    }

    /// K and V (the decode K/V site — Q was projected by its own arm):
    /// contiguous `k_out[m, kv_dim]` / `v_out[m, kv_dim]`, one ingress.
    pub fn kv_linear(
        &self,
        gpu: &dyn GpuBackend,
        a_bf16: DevicePtr,
        k_out: DevicePtr,
        v_out: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        exl3_dense_linear_shared_a(
            gpu,
            &[
                (self.k_proj, Exl3DenseOut::contiguous(k_out)),
                (self.v_proj, Exl3DenseOut::contiguous(v_out)),
            ],
            a_bf16,
            m,
            &self.stage,
            stream,
        )
        .context("EXL3 native attention k/v")
    }

    /// `dst[m, hidden] = attn_out[m, num_heads * head_dim] @ o_proj`,
    /// contiguous both sides (every o_proj site's layout).
    pub fn o_proj_linear(
        &self,
        gpu: &dyn GpuBackend,
        a_bf16: DevicePtr,
        dst_bf16: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        self.proj_linear(
            gpu,
            AttnProj::O,
            a_bf16,
            // Residual-bound: fp32 C on the GEMM tier (upstream out_dtype).
            Exl3DenseOut::contiguous(dst_bf16).with_fp32(),
            m,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::layers::ops::{Exl3DenseStage, Exl3LaunchState};
    use spark_runtime::gpu::mock::MockGpuBackend;

    fn weight(gpu: &MockGpuBackend, k: usize, n: usize) -> Exl3DenseWeight {
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

    fn names(gpu: &MockGpuBackend, from: usize) -> Vec<String> {
        gpu.kernel_lookups_snapshot()[from..]
            .iter()
            .map(|(_, f)| f.clone())
            .collect()
    }

    #[test]
    fn attention_funnels_take_the_dense_arm() {
        // The funnels ARE exl3_dense_linear(_shared_a) over the sites'
        // destinations: the mock's kernel-resolution log shows the launch
        // plan (numerics are the GPU parity example's job, leg J).
        let gpu = MockGpuBackend::new();
        let launch = Arc::new(Exl3LaunchState::new(&gpu).unwrap());
        let stage =
            Arc::new(Exl3DenseStage::new_with_fp32(&gpu, launch, 256, 6144, 12288, 2560).unwrap());
        let w = Exl3AttnWeights {
            q_proj: weight(&gpu, 2560, 12288),
            k_proj: weight(&gpu, 2560, 512),
            v_proj: weight(&gpu, 2560, 512),
            o_proj: weight(&gpu, 6144, 2560),
            stage,
        };
        assert_eq!(w.weight(AttnProj::Q).out_dim, 12288);
        assert_eq!(w.weight(AttnProj::O).in_dim, 6144);
        assert_eq!(AttnProj::V.name(), "v_proj");

        let normed = gpu.alloc(64 * 2560 * 2).unwrap();
        let qkv = gpu.alloc(64 * (12288 + 1024) * 2).unwrap();
        let attn_out = gpu.alloc(64 * 6144 * 2).unwrap();
        let o_out = gpu.alloc(64 * 2560 * 2).unwrap();

        // Decode Q (m=1, contiguous): ingress, the f32-C GEMV (n=12288 is
        // inside the heuristic), contiguous egress.
        let n0 = gpu.kernel_lookups_snapshot().len();
        w.proj_linear(
            &gpu,
            AttnProj::Q,
            normed,
            Exl3DenseOut::contiguous(qkv),
            1,
            0,
        )
        .unwrap();
        let plan = names(&gpu, n0);
        assert_eq!(plan[0], "exl3_bf16_to_f16");
        assert!(plan[1].starts_with("exl3_gemv_k4_cb2_m0"), "{plan:?}");
        assert_eq!(plan[2], "exl3_f32_to_bf16");
        assert_eq!(plan.len(), 3);

        // Decode K/V: ONE ingress, two small-row matmuls, two egresses.
        let n1 = gpu.kernel_lookups_snapshot().len();
        w.kv_linear(
            &gpu,
            normed,
            qkv.offset(12288 * 2),
            qkv.offset((12288 + 512) * 2),
            1,
            0,
        )
        .unwrap();
        let plan = names(&gpu, n1);
        assert_eq!(
            plan.iter().filter(|n| *n == "exl3_bf16_to_f16").count(),
            1,
            "{plan:?}"
        );
        assert_eq!(
            plan.iter().filter(|n| *n == "exl3_f32_to_bf16").count(),
            2,
            "{plan:?}"
        );

        // Multi-seq q/k/v at n=4 into pitched [n, Q|K|V] rows: strided
        // egress for all three, one ingress.
        let ld = 12288 + 1024;
        let n2 = gpu.kernel_lookups_snapshot().len();
        w.qkv_linear(
            &gpu,
            normed,
            Exl3DenseOut::strided(qkv, ld),
            Exl3DenseOut::strided(qkv.offset(12288 * 2), ld),
            Exl3DenseOut::strided(qkv.offset((12288 + 512) * 2), ld),
            4,
            0,
        )
        .unwrap();
        let plan = names(&gpu, n2);
        assert_eq!(
            plan.iter().filter(|n| *n == "exl3_bf16_to_f16").count(),
            1,
            "{plan:?}"
        );
        assert_eq!(
            plan.iter().filter(|n| *n == "exl3_f32_to_bf16_2d").count(),
            3,
            "{plan:?}"
        );

        // Prefill o_proj (m=64, contiguous): residual-bound, so fp32-C GEMM
        // (upstream out_dtype) + f32 egress, not the in-place f16 convert.
        let n3 = gpu.kernel_lookups_snapshot().len();
        w.o_proj_linear(&gpu, attn_out, o_out, 64, 0).unwrap();
        let plan = names(&gpu, n3);
        assert_eq!(plan[0], "exl3_bf16_to_f16");
        assert!(plan[1].starts_with("exl3_gemm_k4_cb2_sh"), "{plan:?}");
        assert!(plan[1].ends_with("_f32"), "{plan:?}");
        assert_eq!(plan[2], "exl3_f32_to_bf16");
        assert_eq!(plan.len(), 3);
    }
}
