// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the bounded view's query capture at the sparse seam (port lane L31): the
//! post-norm, pre-RoPE Q rows of the last `query_tokens` tokens of a request's final user
//! message, per attention layer, copied out of the prefill by [`capture_rows`] (called
//! from the view selector's `on_query` hook) into the fixed slots of [`QueryCapture`]
//! (`spark_runtime::kv_select::capture`, which owns the device side and its tests), and
//! handed to the scorer through [`QueryCapture::rows_view`].
//!
//! Lane L32 (`port/kv_view/mod.rs`, which declares this module) owns the wiring: it builds
//! a [`QueryCapture`] when `kv_view_selection` is on, arms a slot with the message span
//! once the request's final user message is known, forwards its selector's `on_query`
//! here, scores after the message's last prefill chunk, and disarms. With the row off
//! nothing here is constructed: no byte is allocated and no copy issued.
//!
//! Only eager prefill calls are captured. Decode and verify (whose Q belongs to generated
//! tokens) and anything inside a graph capture return at once. A prefill site that only
//! has the post-RoPE Q (`q_pre_rope = None`, a chunk-0 prefill whose Q RoPE is fused into
//! the deinterleave) is captured from `q` and rotated back on the card.

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;
pub use spark_runtime::kv_select::capture::{
    CapturedRows, DEFAULT_QUERY_TOKENS, QueryCapture, QueryCaptureCfg, StreamRows,
};

use crate::layers::qwen3_attention::sparse_seam::{SeamCtx, SeamPhase, SeqBatch};

/// The streams of a prefill call as the capture reads them: one per logical sequence
/// with a published id, in the order their Q rows are stacked.
pub fn stream_rows(s: &SeqBatch) -> Vec<StreamRows> {
    let n = s
        .first_pos
        .len()
        .min(s.new_tokens.len())
        .min(s.seq_ids.len());
    (0..n)
        .map(|i| StreamRows {
            seq_id: s.seq_ids[i],
            first_pos: s.first_pos[i],
            n: s.new_tokens[i],
        })
        .collect()
}

/// The seam's `on_query` forwarding: copy the rows of armed sequences that fall in their
/// span into their slots, for layer `c.attn_layer`. Returns the rows captured (0 for
/// decode, verify, a capturing stream, or a call without published sequence ids).
pub fn capture_rows(
    cap: &QueryCapture,
    c: &SeamCtx,
    s: &SeqBatch,
    q_pre_rope: Option<DevicePtr>,
    q: DevicePtr,
) -> Result<usize> {
    if c.phase != SeamPhase::Prefill || c.in_graph || s.seq_ids.is_empty() {
        return Ok(0);
    }
    let cfg = cap.config();
    if c.num_q_heads as usize != cfg.q_heads || c.kv.head_dim as usize != cfg.head_dim {
        bail!(
            "query capture: layer {} has {} query heads of {} dims; the capture was built for \
             {} of {}",
            c.attn_layer,
            c.num_q_heads,
            c.kv.head_dim,
            cfg.q_heads,
            cfg.head_dim
        );
    }
    cap.capture(
        c.gpu,
        c.attn_layer,
        &stream_rows(s),
        q_pre_rope.unwrap_or(q),
        s.q_row_stride as usize,
        s.q_rows,
        q_pre_rope.is_some(),
        c.stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::qwen3_attention::sparse_seam::{KvDtypeTag, KvLayerView};
    use spark_runtime::gpu::GpuBackend;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::kv_select::QueryView;

    fn cfg() -> QueryCaptureCfg {
        QueryCaptureCfg {
            attn_layers: 2,
            q_heads: 2,
            head_dim: 4,
            query_tokens: 6,
            slots: 1,
            rotary_dim: 2,
            rope_theta: 1.0e7,
        }
    }

    fn ctx(gpu: &MockGpuBackend, phase: SeamPhase) -> SeamCtx<'_> {
        SeamCtx {
            gpu,
            attn_layer: 1,
            kv: KvLayerView {
                k_pool: DevicePtr::NULL,
                v_pool: DevicePtr::NULL,
                dtype: KvDtypeTag::Bf16,
                block_size: 16,
                kv_heads: 1,
                head_dim: 4,
                kv_scale: 1.0,
                v_scale: 1.0,
                block_stride_bytes: 0,
                cache_stride: 0,
            },
            stream: 5,
            in_graph: false,
            phase,
            num_q_heads: 2,
            inv_sqrt_d: 0.5,
        }
    }

    fn batch<'a>(ids: &'a [u64], first: &'a [u32], new: &'a [u32], q_rows: usize) -> SeqBatch<'a> {
        SeqBatch {
            n: ids.len(),
            block_table: DevicePtr::NULL,
            seq_lens: DevicePtr::NULL,
            max_blocks_per_seq: 0,
            host_seq_lens: new,
            seq_ids: ids,
            rows_per_seq: 1,
            q_row_stride: 8,
            q_rows,
            first_pos: first,
            new_tokens: new,
            host_block_tables: &[],
            block_table_ptrs: DevicePtr::NULL,
        }
    }

    #[test]
    fn prefill_rows_of_the_armed_stream_are_captured_from_the_stacked_q() {
        let gpu = MockGpuBackend::new();
        let cap = QueryCapture::new(&gpu, cfg()).unwrap();
        let host: Vec<u8> = (0..6 * 8u16)
            .flat_map(|i| ((i / 8) * 100 + i % 8).to_le_bytes())
            .collect();
        let q = gpu.alloc(host.len()).unwrap();
        gpu.copy_h2d(&host, q).unwrap();
        cap.arm(9, 20, 24);
        let (ids, first, new) = ([4u64, 9], [0u32, 20], [2u32, 4]);
        let s = batch(&ids, &first, &new, 6);
        assert_eq!(
            stream_rows(&s)[1],
            StreamRows {
                seq_id: 9,
                first_pos: 20,
                n: 4
            }
        );
        let pre = Some(q);
        assert_eq!(
            capture_rows(&cap, &ctx(&gpu, SeamPhase::Prefill), &s, pre, q).unwrap(),
            4
        );
        assert!(
            gpu.launches_snapshot().is_empty(),
            "pre-RoPE rows are copied as is"
        );
        // The post-RoPE fallback copies the same rows and rotates them back.
        assert_eq!(
            capture_rows(&cap, &ctx(&gpu, SeamPhase::Prefill), &s, None, q).unwrap(),
            4
        );
        assert_eq!(gpu.launches_snapshot().len(), 1);
        assert_eq!(cap.captured_rows(9), Some(vec![0, 4]));
        let slot = gpu.read_alloc(cap.buffer()).unwrap();
        let at = 6 * 16; // (layer 1 x 6 rows + row 0) x 16 bytes
        assert_eq!(
            u16::from_le_bytes([slot[at], slot[at + 1]]),
            200,
            "stream 9 starts at q row 2"
        );
        assert!(cap.rows_view(9).is_none(), "layer 0 still empty");
    }

    #[test]
    fn decode_verify_graph_and_unpublished_calls_capture_nothing() {
        let gpu = MockGpuBackend::new();
        let cap = QueryCapture::new(&gpu, cfg()).unwrap();
        cap.arm(7, 0, 5);
        let q = gpu.alloc(64).unwrap();
        let (ids, first, new) = ([7u64], [4u32], [1u32]);
        let s = batch(&ids, &first, &new, 1);
        for phase in [SeamPhase::Decode, SeamPhase::Verify] {
            assert_eq!(
                capture_rows(&cap, &ctx(&gpu, phase), &s, None, q).unwrap(),
                0
            );
        }
        let mut c = ctx(&gpu, SeamPhase::Prefill);
        c.in_graph = true;
        assert_eq!(capture_rows(&cap, &c, &s, Some(q), q).unwrap(), 0);
        let unpublished = batch(&[], &first, &new, 1);
        assert_eq!(
            capture_rows(
                &cap,
                &ctx(&gpu, SeamPhase::Prefill),
                &unpublished,
                Some(q),
                q
            )
            .unwrap(),
            0
        );
        let wrong = SeamCtx {
            num_q_heads: 3,
            ..ctx(&gpu, SeamPhase::Prefill)
        };
        assert!(capture_rows(&cap, &wrong, &s, Some(q), q).is_err());
        assert!(gpu.launches_snapshot().is_empty());
        let full = capture_rows(&cap, &ctx(&gpu, SeamPhase::Prefill), &s, Some(q), q).unwrap();
        assert_eq!(full, 1);
        assert_eq!(
            cap.rows_view(7).map(|v| v.rows()),
            None,
            "4 of 5 rows still missing"
        );
    }
}
