// SPDX-License-Identifier: AGPL-3.0-only

//! Q2_0 keep-packed qkvz init + transient-dequant prefill GEMM.

use super::*;

impl Qwen3SsmLayer {
    /// Install the Tier-1c keep-packed ternary Q2_0 fused `in_proj_qkvz`
    /// (`ATLAS_GGUF_NATIVE_Q2`). Decode dispatches `q2_0_gemv_vec`; prefill
    /// transient-dequants via `Self::qkvz_q2_prefill_gemm`. `out_proj` is
    /// unaffected (stays NVFP4). Requires `sequential_qkvz` (Bonsai concats
    /// [Q|K|V|Z] at load).
    pub fn set_packed_q2_qkvz(&mut self, qkvz: crate::weight_map::PackedQ2Weight) {
        self.qkvz_q2 = Some(qkvz);
    }

    /// Transient-dequant prefill GEMM for the packed qkvz: dequant the 2-bit
    /// `[qkvz_size, h]` weight into the caller-provided PERSISTENT BF16 `scratch`
    /// (the arena `q2_dequant_scratch`, sized to the largest packed projection),
    /// run `dense_gemm` (`out[m, qkvz_size] = in[m, h] @ w^T`). Mirrors the
    /// FFN/attention packed prefill — no per-matmul alloc/sync/free; the dequant
    /// orders before the GEMM on the same `stream`. Errors if the dequant kernel
    /// is absent.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qkvz_q2_prefill_gemm(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        out: DevicePtr,
        scratch: DevicePtr,
        act_q8: DevicePtr,
        m: u32,
        stream: u64,
    ) -> Result<()> {
        let w = self
            .qkvz_q2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("qkvz_q2_prefill_gemm: no packed qkvz installed"))?;
        let (n, k) = (w.n, w.k);

        // Tier-2 native MMQ (ATLAS_GGUF_NATIVE_Q2_MMQ=1): quantize `input` to q8_1
        // then run the packed 2-bit MMQ GEMM for the fused qkvz — no BF16 weight
        // dequant, no shared `q2_dequant_scratch` race. Group-128 only.
        if self.q2_0_mmq_nc_k.0 != 0
            && self.q4k_quant_act_k.0 != 0
            && crate::layers::ops::native_q2_mmq_enabled()
            && w.group == 128
        {
            crate::layers::ops::quantize_act_q8_1(
                gpu,
                self.q4k_quant_act_k,
                input,
                act_q8,
                m,
                k,
                stream,
            )?;
            return crate::layers::ops::q2_0_mmq_gemm(
                gpu,
                self.q2_0_mmq_nc_k,
                self.q2_0_mmq_wc_k,
                act_q8,
                w.weight,
                out,
                m,
                n,
                k,
                stream,
            );
        }

        if self.dequant_q2_0_gn_k.0 == 0 {
            anyhow::bail!(
                "dequant_q2_0_gn_to_bf16 kernel missing — packed-Q2 GDN prefill unavailable"
            );
        }
        crate::layers::ops::dequant_q2_0_gn_to_bf16(
            gpu,
            self.dequant_q2_0_gn_k,
            w.weight,
            scratch,
            n,
            k,
            w.group as u32,
            stream,
        )?;
        let dw = DenseWeight { weight: scratch };
        if self.dense_gemm_pipelined_k.0 != 0 {
            crate::layers::ops::dense_gemm_bf16_pipelined(
                gpu,
                self.dense_gemm_pipelined_k,
                input,
                &dw,
                out,
                m,
                n,
                k,
                stream,
            )?;
        } else {
            crate::layers::ops::dense_gemm(
                gpu,
                self.dense_gemm_k,
                input,
                &dw,
                out,
                m,
                n,
                k,
                stream,
            )?;
        }
        Ok(())
    }
}
