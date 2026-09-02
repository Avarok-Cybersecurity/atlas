// SPDX-License-Identifier: AGPL-3.0-only

//! Stage 0 route predicate: a layer that declines the batched multi-sequence
//! path must never reach it.
//!
//! WHY THIS IS A GATE AND NOT A COMMENT
//! ------------------------------------
//! `decode_multi_seq`'s default loop shares ONE `ForwardContext` across every
//! sequence in the batch, so a layer that indexes per-sequence state by a
//! fixed row (GLM-5.3: mHC highway slot 0, DSA `attn_metadata` row 0) does not
//! run slowly on that path — it returns the WRONG sequence's activations.
//!
//! The escape already exists: #753 item B routes mHC-highway models onto a
//! per-sequence loop (`decode_a2`'s `hc_perseq`, `decode_b`'s
//! `hc_qsa_perseq`), which suppresses graphs and runs one sequence at a time.
//! Stage 0 only adds a layer-declared term to those two disjunctions, plus the
//! matching term in `can_batch_verify_dispatch`.
//!
//! 🪤 THE TERM MUST BE HOISTED OUT OF THE `hc_mult > 0` / `index_topk > 0`
//! CONJUNCTIONS. Both routes otherwise key on
//! `bound = index_topk + index_compress_ratio - 1`; GLM sets `index_topk =
//! 2048` and no compress ratio, so `bound == 2047` and every sequence SHORTER
//! than 2047 tokens falls through to the batched path. A veto folded inside
//! the conjunction would therefore protect long contexts and silently miss
//! short ones — the failure shape that survives casual testing. These tests
//! pin the hoist, not merely the presence of the term.

#[cfg(test)]
mod tests {
    use crate::layer::{ForwardContext, LayerState, TransformerLayer};
    use anyhow::Result;
    use spark_runtime::gpu::{DevicePtr, GpuBackend};
    use spark_runtime::kv_cache::PagedKvCache;

    /// The two required `TransformerLayer` methods, stubbed. This test only
    /// ever calls the capability predicates, never a forward.
    macro_rules! stub_forward {
        () => {
            #[allow(clippy::too_many_arguments)]
            fn decode(
                &self,
                _hidden: DevicePtr,
                _residual: DevicePtr,
                _state: &mut dyn LayerState,
                _kv_cache: &mut PagedKvCache,
                _seq_len: usize,
                _block_table: &mut Vec<u32>,
                _disk_block_ids: &mut Vec<u32>,
                _disk_last_offloaded_per_layer: &mut Vec<u32>,
                _ctx: &ForwardContext,
                _stream: u64,
            ) -> Result<()> {
                unreachable!("capability-predicate test never runs a forward")
            }
            fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
                unreachable!("capability-predicate test never allocates state")
            }
        };
    }

    fn src(rel: &str) -> String {
        std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel))
            .unwrap_or_else(|e| panic!("read {rel}: {e}"))
    }

    /// Slice `src` from `from` to the next line that starts `needle_end`.
    fn block<'a>(s: &'a str, from: &str, to: &str) -> &'a str {
        let start = s.find(from).unwrap_or_else(|| panic!("missing anchor {from:?}"));
        let rest = &s[start..];
        let end = rest.find(to).unwrap_or(rest.len());
        &rest[..end]
    }

    /// The trait defaults must be permissive: adding these predicates may not
    /// change the route for any layer that has not opted in.
    #[test]
    fn defaults_are_false_so_no_existing_layer_changes_route() {
        struct Plain;
        impl TransformerLayer for Plain {
            stub_forward!();
        }
        assert!(
            !Plain.decode_multi_seq_unsupported(),
            "default must be false — a new predicate may not re-route existing models"
        );
        assert!(
            !Plain.decode_verify_multi_unsupported(),
            "default must be false"
        );
    }

    /// A layer that declines is honoured, and the two answers are independent.
    #[test]
    fn a_declining_layer_is_honoured_on_both_axes_independently() {
        struct DeclinesDecode;
        impl TransformerLayer for DeclinesDecode {
            stub_forward!();
            fn decode_multi_seq_unsupported(&self) -> bool {
                true
            }
        }
        struct DeclinesVerify;
        impl TransformerLayer for DeclinesVerify {
            stub_forward!();
            fn decode_verify_multi_unsupported(&self) -> bool {
                true
            }
        }
        assert!(DeclinesDecode.decode_multi_seq_unsupported());
        assert!(
            !DeclinesDecode.decode_verify_multi_unsupported(),
            "decode and verify answers must be independent"
        );
        assert!(DeclinesVerify.decode_verify_multi_unsupported());
        assert!(!DeclinesVerify.decode_multi_seq_unsupported());
    }

    /// `decode_a2` — the EP and single-GPU batched decode dispatcher.
    ///
    /// PROVEN BY: deleting the veto term, or folding it back inside the
    /// `hc_mult > 0` conjunction, turns this red.
    #[test]
    fn decode_a2_routes_a_declining_layer_per_sequence_at_every_length() {
        let s = src("src/model/trait_impl/decode_a2.rs");
        let b = block(&s, "let ms_layer_veto", "if self.comm.is_some()");
        assert!(
            b.contains("decode_multi_seq_unsupported()"),
            "decode_a2 must consult the layer predicate"
        );
        // The veto is the FIRST disjunct of hc_perseq, i.e. outside `hc_mult > 0`.
        assert!(
            b.contains("let hc_perseq = ms_layer_veto\n            || ("),
            "the veto must be hoisted OUT of the hc_mult/qsa_active conjunction; \
             folded inside, it would only fire at seq_len >= index_topk - 1"
        );
    }

    /// `decode_b` — the single-GPU fused decode+prefill caller. Skipping this
    /// site is the exact hole a serve-time `max_batch_size` clamp would leave.
    #[test]
    fn decode_b_routes_a_declining_layer_per_sequence_at_every_length() {
        let s = src("src/model/trait_impl/decode_b.rs");
        let b = block(&s, "let ms_layer_veto", "if self.comm.is_some()");
        assert!(
            b.contains("decode_multi_seq_unsupported()"),
            "decode_b must consult the layer predicate — it is the single-GPU path"
        );
        assert!(
            b.contains("let hc_qsa_perseq = ms_layer_veto\n            || ("),
            "the veto must be hoisted OUT of the hc_mult/index_topk conjunction"
        );
    }

    /// `can_batch_verify_dispatch` must refuse the batched verify sweep for a
    /// declining layer as a ROUTING decision, not a mid-request `bail!`.
    #[test]
    fn can_batch_verify_dispatch_consults_the_verify_predicate() {
        let s = src("src/model/trait_impl/verify_e.rs");
        let b = block(&s, "fn can_batch_verify_dispatch", "\n    pub(super) fn ");
        assert!(
            b.contains("decode_verify_multi_unsupported()"),
            "can_batch_verify_dispatch must consult the layer predicate"
        );
        assert!(
            b.contains("&& !self"),
            "the term must be a NEGATED conjunct of the existing self-gate"
        );
    }

    /// ⛔ Stage 0 is a ROUTE predicate, never a capacity clamp. Re-clamping
    /// `max_batch_size` in `serve_load.rs` would revert #753 item B, cost the
    /// operator concurrency, and still miss `decode_b`.
    #[test]
    fn stage0_did_not_reintroduce_a_serve_time_clamp() {
        let s = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../spark-server/src/main_modules/serve_load.rs"),
        )
        .expect("read serve_load.rs");
        assert!(
            !s.contains("decode_multi_seq_unsupported"),
            "the concurrency capability must be consumed at the DISPATCH site, \
             never as a serve-time max_batch_size clamp"
        );
    }
}
