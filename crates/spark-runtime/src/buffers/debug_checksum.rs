// SPDX-License-Identifier: AGPL-3.0-only

//! `BufferArena::debug_buffer_checksum`. Split from `accessors.rs` (500-LoC
//! cap) by exact piecewise copy.

use super::BufferArena;
use crate::gpu::{DevicePtr, GpuBackend};

impl BufferArena {
    pub fn debug_buffer_checksum(&self, gpu: &dyn GpuBackend, stream: u64, tag: &str) {
        gpu.synchronize(stream).ok();
        let probe = |name: &str, ptr: DevicePtr, bytes: usize| {
            let mut hb = vec![0u8; bytes];
            if gpu.copy_d2h(ptr, &mut hb).is_err() {
                return;
            }
            let (mut sum, mut ssq, mut sabs) = (0f64, 0f64, 0f64);
            for c in hb.chunks_exact(4) {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64;
                if v.is_finite() {
                    sum += v;
                    ssq += v * v;
                    sabs += v.abs();
                }
            }
            tracing::warn!(
                "AVAROK_BUF_CKSUM[{tag}] {name} bytes={bytes} sum={sum:.6} ssq={ssq:.6} sabs={sabs:.6}"
            );
        };
        probe(
            "hidden_states",
            self.hidden_states,
            self.sizes.hidden_states,
        );
        probe("residual", self.residual, self.sizes.residual);
        probe("norm_output", self.norm_output, self.sizes.norm_output);
        probe("qkv_output", self.qkv_output, self.sizes.qkv_output);
        probe("attn_output", self.attn_output, self.sizes.attn_output);
        probe("gate_logits", self.gate_logits, self.sizes.gate_logits);
        probe("moe_output", self.moe_output, self.sizes.moe_output);
        probe("ssm_qkvz", self.ssm_qkvz, self.sizes.ssm_qkvz);
        probe("ssm_ba", self.ssm_ba, self.sizes.ssm_ba);
        probe(
            "ssm_deinterleaved",
            self.ssm_deinterleaved,
            self.sizes.ssm_deinterleaved,
        );
        probe("ssm_gates", self.ssm_gates, self.sizes.ssm_gates);
        probe(
            "ssm_conv_out_f32",
            self.ssm_conv_out_f32,
            self.sizes.ssm_conv_out_f32,
        );
        probe(
            "expert_gate_out",
            self.expert_gate_out,
            self.sizes.expert_gate_out,
        );
        probe(
            "expert_up_out",
            self.expert_up_out,
            self.sizes.expert_up_out,
        );
        probe(
            "expert_down_out",
            self.expert_down_out,
            self.sizes.expert_down_out,
        );
        probe(
            "splitk_workspace",
            self.splitk_workspace,
            self.sizes.splitk_workspace,
        );
    }
}
