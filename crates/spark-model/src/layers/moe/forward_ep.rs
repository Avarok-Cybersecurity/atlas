// SPDX-License-Identifier: AGPL-3.0-only

//! MoeLayer::forward_ep_dispatch.

use super::*;

impl MoeLayer {
    /// 4. Computes local experts on local + received tokens
    /// 5. Sends results back (combine)
    /// 6. Weighted sum into output
    ///
    /// Currently scaffolding only — builds routing table and logs statistics.
    /// Expert compute and actual dispatch use the existing per-token path.
    /// The all-reduce fallback is used for the actual output until dispatch
    /// kernels are implemented.
    pub fn forward_ep_dispatch(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        use super::super::ep_dispatch::build_ep_routing_table;

        let h = ctx.config.hidden_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let (local_start, local_end) = ctx.config.local_expert_range();

        // Gemma-4 router pre-norm (no-op for other models). EP dispatch is
        // per-token so uses num_tokens=1.
        let router_in = self.router_input(input, 1, h, ctx, stream)?;
        // Step 1: Gate projection (same as forward())
        let gate_logits = ctx.buffers.gate_logits();
        if let Some(ref nvfp4) = self.gate_nvfp4 {
            ops::w4a16_gemv(
                ctx.gpu,
                self.w4a16_gemv,
                router_in,
                nvfp4,
                gate_logits,
                num_experts,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv,
                router_in,
                &self.weights.gate,
                gate_logits,
                num_experts,
                h,
                stream,
            )?;
        }

        // Step 2: Top-K routing
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(top_k as usize * 4);

        ops::moe_topk_softmax(
            ctx.gpu,
            self.moe_topk,
            gate_logits,
            indices_dev,
            weights_dev,
            num_experts,
            top_k,
            ctx.config.norm_topk_prob,
            stream,
        )?;

        // Step 3: Build routing table (requires D2H copy of indices/weights)
        // This is CPU-side work — acceptable for scaffolding, will move to
        // GPU-side routing table construction in the optimized path.
        ctx.gpu.synchronize(stream)?;
        let k = top_k as usize;
        let mut idx_buf = vec![0u8; k * 4];
        let mut wt_buf = vec![0u8; k * 4];
        ctx.gpu.copy_d2h(indices_dev, &mut idx_buf)?;
        ctx.gpu.copy_d2h(weights_dev, &mut wt_buf)?;

        let gate_indices: Vec<u32> = (0..k)
            .map(|i| {
                u32::from_le_bytes([
                    idx_buf[i * 4],
                    idx_buf[i * 4 + 1],
                    idx_buf[i * 4 + 2],
                    idx_buf[i * 4 + 3],
                ])
            })
            .collect();
        let gate_weights: Vec<f32> = (0..k)
            .map(|i| {
                f32::from_le_bytes([
                    wt_buf[i * 4],
                    wt_buf[i * 4 + 1],
                    wt_buf[i * 4 + 2],
                    wt_buf[i * 4 + 3],
                ])
            })
            .collect();

        let routing =
            build_ep_routing_table(&gate_indices, &gate_weights, 1, k, local_start, local_end);

        tracing::debug!(
            "EP dispatch: local={} remote={} (rank {}, experts {}..{})",
            routing.local_count(),
            routing.remote_count(),
            ctx.config.ep_rank,
            local_start,
            local_end,
        );

        // Steps 4-6: For now, fall back to the existing forward() path
        // which uses all-reduce. The routing table is built but not yet
        // used for actual dispatch. This will be replaced with:
        //   - comm.group_start()
        //   - comm.send_to() for remote tokens
        //   - comm.recv_from() for incoming tokens
        //   - comm.group_end()
        //   - Local expert compute on local + received tokens
        //   - comm.group_start()
        //   - comm.send_to() results back
        //   - comm.recv_from() results from partner
        //   - comm.group_end()
        //   - Weighted sum into output
        let _ = routing; // suppress unused warning until dispatch is wired
        self.forward(input, ctx, stream)
    }
}
