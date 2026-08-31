// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// lm_head tile-GEMM decode path: **ON by default**, disabled by
/// `ATLAS_NO_LMHEAD_TGEMM=1`. Evaluated ONCE at construction — the switch
/// decides whether the transposed twin is built at all, so setting it later has
/// no effect. Presence-style check (`ATLAS_*=0` is NOT "off").
///
/// Measured C=16: 113.10 -> 119.32 tok/s (+5.50%, disjoint ranges, 4 reps).
/// `padded_n <= 4` is untouched and stays byte-identical, so C=1 is unaffected.
/// The twin costs ~681 MB and leaves the KV pool at 4759 blocks vs 4757 without
/// it — no measurable KV impact.
fn lmhead_tgemm_enabled() -> bool {
    std::env::var("ATLAS_NO_LMHEAD_TGEMM").ok().as_deref() != Some("1")
}

impl TransformerModel {
    pub fn new(
        config: ModelConfig,
        embed_tokens: DenseWeight,
        final_norm: DenseWeight,
        lm_head_weight: DenseWeight,
        lm_head_nvfp4: Option<QuantizedWeight>,
        // Runtime FP8 LM head (`--lm-head-dtype fp8`). Mutually exclusive with
        // `lm_head_nvfp4`; `None` for the NVFP4/BF16/default paths (byte-identical).
        lm_head_fp8: Option<crate::weight_map::Fp8DenseWeight>,
        // Separate NVFP4 head used ONLY by the MTP draft proposer when the
        // main head is kept BF16 (`skip_lm_head_quantization()`). `None` for
        // the NVFP4-main default, in which case the proposer falls back to
        // `lm_head_nvfp4`. Drafts are always verified by the main BF16 head,
        // so this approximate head never affects an accepted token.
        mtp_lm_head_nvfp4: Option<QuantizedWeight>,
        layers: Vec<Box<dyn TransformerLayer>>,
        buffers: BufferArena,
        kv_cache: PagedKvCache,
        mtp_weights: Vec<MtpWeights>,
        gpu: Box<dyn GpuBackend>,
        max_seq_len: usize,
        max_batch_size: usize,
        mtp_quant: crate::layers::MtpQuantization,
        use_speculative: bool,
        prefix_cache: Box<dyn spark_runtime::prefix_cache::PrefixCache>,
        mtp_vocab_size: u32,
        comm: Option<std::sync::Arc<dyn spark_comm::CommBackend>>,
        self_speculative: bool,
        num_drafts: usize,
        vision_encoder: Option<crate::layers::VisionEncoder>,
        ssm_cache_slots: usize,
        ssm_checkpoint_interval: usize,
    ) -> Result<Self> {
        // `rms_norm_kernel` normalizes exactly one weight: `final_norm` (a
        // checkpoint tensor). Models that ship HF-vanilla norm weights load it
        // exactly and must use the vanilla kernel.
        let rms_norm_kernel = if crate::ships_vanilla_norm_weights(&config) {
            gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?
        } else {
            gpu.kernel("norm", "rms_norm")?
        };
        let dense_gemv_kernel = gpu.kernel("gemv", "dense_gemv_bf16")?;
        // FP32-output dense GEMV — the FP32 logits path required an FP32
        // residual stream, which no longer exists, so this stays
        // KernelHandle(0) and the BF16 path is always taken.
        let dense_gemv_fp32out_kernel = KernelHandle(0);
        let w4a16_gemv_kernel = gpu.kernel("w4a16_gemv", "w4a16_gemv")?;
        let w4a16_gemv_logits_kernel = gpu.kernel("w4a16_gemv", "w4a16_gemv_logits")?;
        // lm_head shares the tile GEMM, so route it through the same resolver as
        // the SSM/attention sites — it picks the 3-deep pipeline variant when
        // present. lm_head launches 1938 CTAs and already sits at ~83% of
        // achievable, so the expected gain here is small; measured, not assumed.
        let w4a16_gemm_t_kernel = crate::layers::tgemm_kernel(gpu.as_ref());
        // Lossless BF16-MMA sibling for lm_head, OPT-IN via ATLAS_LMHEAD_LOSSLESS=1.
        // Measured cost 1.81% at C=16 (129.68 -> 127.33). Default is the faster
        // FP8-activation path because the accuracy question it addresses CANNOT
        // BE MEASURED until vLLM parity lifts the BFCL embargo — and the 1.81%
        // is throughput needed to REACH parity. The risk is real but indirect:
        // the bf16-floor finding was superseded on the WEIGHT axis, and this is
        // the ACTIVATION axis, which was never examined. Re-decide at parity.
        let w4a16_gemm_t_bf16_kernel = if std::env::var("ATLAS_LMHEAD_LOSSLESS").is_ok() {
            crate::layers::try_kernel(gpu.as_ref(), "w4a16", "w4a16_gemm_t_m128_bf16_v2")
        } else {
            spark_runtime::gpu::KernelHandle(0)
        };
        let w4a16_gemm_kernel = gpu.kernel("w4a16", "w4a16_gemm")?;
        let w4a16_gemv_batch2_kernel = gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?;
        // Narrow batched-GEMV family (M=4..8) for the K=3..8 verify lm_head
        // (try_kernel per tier: 0-handle on targets that predate a tier;
        // dispatch widens, then falls back to the GEMM).
        let w4a16_batchm = crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers::resolve(gpu.as_ref());
        // M<=16 batched GEMV for the wide BATCHED-DECODE lm_head. The SSM mixer
        // already carries this handle (qwen3_ssm/mod.rs); the model level did
        // not, so the decode head had no arm above 8 and fell to the M64-tile
        // GEMM. Same try_kernel contract: 0-handle -> dispatch falls back.
        let w4a16_gemv_batch16_kernel =
            crate::layers::try_kernel(gpu.as_ref(), "w4a16_gemv", "w4a16_gemv_batch16");
        // FP8 E4M3 LUT GEMV for the `--lm-head-dtype fp8` head. Loaded
        // unconditionally (a handle is cheap); only invoked when `lm_head_fp8`
        // is set, so the NVFP4/BF16 paths never touch it.
        let dense_gemv_fp8w_kernel = gpu.kernel("gemv_fp8w", "dense_gemv_fp8w")?;
        // FP8 dual-GEMV (batch=2): present on images that ship the kernel;
        // try_kernel keeps the handle 0 on older sets so dispatch falls back
        // to the per-token loop.
        let dense_gemv_fp8w_batch2_kernel = crate::layers::try_kernel(
            gpu.as_ref(),
            "dense_gemv_fp8w_batch2",
            "dense_gemv_fp8w_batch2",
        );
        let dense_gemm_kernel = gpu.kernel("gemm", "dense_gemm_bf16")?;
        let dense_gemv_batchm_kernel = gpu
            .kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")
            .unwrap_or(spark_runtime::gpu::KernelHandle(0));
        let argmax_kernel = gpu.kernel("argmax", "argmax_bf16")?;
        let argmax_batch_kernel = gpu
            .kernel("argmax", "argmax_bf16_batch")
            .unwrap_or(spark_runtime::gpu::KernelHandle(0));
        let argmax_logits_kernel = gpu.kernel("argmax", "argmax_fp32")?;
        let batched_embed_kernel = gpu.kernel("embed_from_argmax", "batched_embed")?;
        let fill_slots_kernel = gpu.kernel("metadata_fill", "fill_slots_from_block_table")?;
        let profile = config.profile;
        let profile_first = std::env::var("ATLAS_PROFILE_FIRST").is_ok();

        // Pin the split-K attention split count to the configured max batch so
        // a sequence's attention reduction is invariant to how many other
        // sequences are co-batched (concurrent-decode determinism — see
        // tasks/determinism_investigation.md).
        let mut levers = ops::ModelLevers::from_env();
        levers.max_decode_seqs = (max_batch_size as u32).max(1);

        tracing::info!(
            "TransformerModel: {} layers, vocab={}, hidden={}{}{}",
            layers.len(),
            config.vocab_size,
            config.hidden_size,
            if profile { " [PROFILE MODE]" } else { "" },
            if profile_first {
                " [PROFILE_FIRST]"
            } else {
                ""
            },
        );

        // Build SSM state pool (with MTP intermediate/checkpoint pools only if speculative decoding enabled)
        // num_intermediates = K, the verify-width ceiling. The CONV pools
        // allocate K snapshots per slot; the H pools allocate K-1 (index
        // K-1 is never written or read — see ssm_reserve) and tier by slot.
        // For MTP K=2/3/4 verify: K = num_drafts + 1.
        // For DFlash K=γ verify: K = γ + 1 (drafter's γ drafts + 1 verified bonus slot).
        // Pool size = max of both so DFlash and MTP can coexist on the same model.
        let dflash_kgamma = if !config.dflash_capture_layers.is_empty() {
            // The +1 is the prefix bonus position in the verify input
            // `[last_token, draft_0, ..., draft_{γ-1}]`. Sized from the
            // RESOLVED drafter γ (factory sets `config.dflash_gamma` from the
            // drafter checkpoint / --dflash-gamma): the legacy 17-wide
            // ceiling (γ=16-era) cost ~1.5 GB of intermediates per slot per
            // GB at γ=8 — ~12 GB across 8 slots on qwen3.8-27B — for slots
            // the verify never touches (2026-08-19 256K/C8 boot ledger).
            // Unknown γ keeps the 17-wide fallback.
            config.dflash_gamma.map(|g| g + 1).unwrap_or(17)
        } else {
            0
        };
        // DFlash needs the SSM verify pools regardless of MTP weight presence
        // or lm_head quantization — its K=γ verify path checkpoints SSM state
        // for partial-accept rollback. Force `has_mtp` on whenever DFlash is
        // active so the checkpoint pools exist.
        // The MTP proposer needs an NVFP4 vocab head for drafting: either the
        // main head (NVFP4 default) or the draft-only head built when the main
        // head is BF16. `draft_lm_head_nvfp4` resolves to whichever is present.
        let draft_lm_head_nvfp4 = mtp_lm_head_nvfp4.or(lm_head_nvfp4);
        let has_mtp = self_speculative
            || (use_speculative && !mtp_weights.is_empty() && draft_lm_head_nvfp4.is_some())
            || dflash_kgamma > 0;
        let num_intermediates = if !has_mtp {
            0
        } else if dflash_kgamma > 0 {
            // DFlash serve: the block drafter OWNS the verify path, so the
            // widest verify is K = gamma + 1 and `num_drafts` never reaches
            // the pool. Taking max(num_drafts+1, kgamma) sized these pools for
            // the MTP K=2/3/4 ladder that a DFlash serve never runs: at
            // num_drafts=15 that is 16 wide against a real ceiling of 9, and
            // these pools are the single largest non-weight allocation on the
            // box — 21.9 GB at 8 slots x 48 layers, as large as the model
            // itself. Sizing to the real ceiling reclaims ~9.6 GB
            // (2026-08-19 128K/C8 boot ledger; observed verify widths were
            // K=8 and ks=[8;n], never above).
            //
            // The K2/K3/K4 arms still reachable on a DFlash serve (when the
            // drafter returns <4 drafts) verify at K<=4, comfortably inside
            // this. `require_verify_rollback_supported` remains the backstop
            // if a future path asks for more.
            dflash_kgamma
        } else {
            num_drafts + 1
        };
        let ssm_pool = std::sync::Arc::new(SsmStatePool::new(
            &config,
            max_batch_size,
            has_mtp,
            num_intermediates,
            num_drafts,
            // Stage-3 f16-SIZED h pools. No CLI surface publishes this and
            // preflight refuses it until prefill narrowing lands, so it is
            // false on every serveable config today.
            crate::layers::qwen3_ssm::ssm_h_f16_pool_enabled(),
            // `--ssm-rollback-mode` (EXPERIMENTAL replay scaffold; default
            // snapshot, published by spark-server's serve_flags).
            crate::ssm_reserve::ssm_rollback_mode(),
            gpu.as_ref(),
        )?);

        // Fail fast if an SSM tier was requested (`ATLAS_SSM_TIER`) on a model
        // with no recurrent state — a tier request there was previously a
        // silent no-op. No-op when the tier is unset (default path).
        super::ssm_tier::ensure_ssm_tier_capability(&config)?;

        // SSM snapshot pool: Marconi prefix-cache slots + Phase-C
        // decode-rollback ring. The decode-rollback region is only sized
        // for SSM models — `num_ssm_layers == 0` makes both regions
        // collapse to empty. The ring retains DECODE_ROLLBACK_RING_SLOTS
        // boundary snapshots per sequence (DECOUPLED from ROLLBACK_RESTEER_CAP:
        // the cap bounds re-steer attempts, the ring must retain enough
        // boundaries that a clean PRE-loop one survives — `CAP+1=3` was too
        // small and forced NoSsmSnapshot declines). Sized for every
        // active-sequence pool slot (`max_batch_size`).
        // The ring's ONLY writer (scheduler snapshot_boundary_if_ssm) and
        // reader (content-loop rollback_to_boundary) live on the PLAIN decode
        // path — the speculative path does its rejection rollback through the
        // verify snapshot, never this ring. Under `--speculative` the ring is
        // therefore unreachable, and on this model it is NOT cheap: 8 slots x
        // max_batch x the full SSM blob (27B: 158.9 MB) = ~19.9 GB at batch 16,
        // allocated up front. Skip it when speculative decode is on.
        // The ring-depth decision (env overrides + speculative/watchdog
        // skip) is SSOT'd in `crate::ssm_reserve::decode_rollback_ring_slots`
        // — spark-server's `preflight_reserve` calls the SAME helper, so the
        // GPU reservation and this allocation cannot drift. The scheduler
        // keys off `decode_rollback_ring_slots()`, so a 0 here disables save
        // AND rollback coherently (rollback declines, the documented
        // fail-open).
        let ring = crate::ssm_reserve::decode_rollback_ring_slots(
            ssm_pool.num_ssm_layers,
            use_speculative,
        );
        if let Some(reason) = ring.skip_reason {
            let per_seq = (ssm_pool.h_bytes + ssm_pool.conv_bytes)
                * ssm_pool.num_ssm_layers
                * atlas_kernels::DECODE_ROLLBACK_RING_SLOTS;
            tracing::info!(
                "SSM decode-rollback ring: SKIPPED ({}) — the ring's save/rollback \
                 path only runs on plain decode with watchdogs enabled. Saves {:.1} GB \
                 ({} seqs x {} slots x full SSM blob). If plain-decode loop re-steer is \
                 ever reached it fail-opens to decline; ATLAS_SSM_DECODE_RING=1 \
                 force-restores the ring.",
                reason,
                (per_seq * max_batch_size) as f64 / 1e9,
                max_batch_size,
                atlas_kernels::DECODE_ROLLBACK_RING_SLOTS,
            );
        }
        let decode_ring_slots = ring.slots;
        let ssm_snapshots = SsmSnapshotPool::new(
            ssm_cache_slots,
            ssm_pool.h_bytes,
            ssm_pool.conv_bytes,
            ssm_pool.num_ssm_layers,
            decode_ring_slots,
            max_batch_size,
            // Last-token hidden snapshot: post-final-norm `norm_output` is
            // BF16 (`hidden_size` elements). Used to emit exact-hit logits
            // without re-running the last token through the SSM layers.
            config.hidden_size * 2,
            gpu.as_ref(),
        )?;
        // Optional SSM snapshot spill tier. `None` (default) keeps the reclaim
        // drop path byte-identical; blob sizing tracks the pool's spill layout.
        let ssm_tier_store = super::impl_a1_init::build_ssm_tier_store(
            &config,
            ssm_snapshots.spill_blob_bytes(),
            ssm_pool.num_ssm_layers,
        )?;
        if ssm_checkpoint_interval > 0 && ssm_cache_slots > 0 {
            tracing::info!(
                "Marconi intermediate checkpoints: every {} blocks ({} tokens at block_size={})",
                ssm_checkpoint_interval,
                ssm_checkpoint_interval * kv_cache.block_size(),
                kv_cache.block_size(),
            );
        }

        // Fixed metadata stride for CUDA graph compatibility
        let max_blocks_per_seq = (max_seq_len / kv_cache.block_size() + 1) as u32;

        // Permanent dummy KV block for padding sequences. Must be explicitly
        // zeroed: `gpu.alloc()` returns uninitialized memory, and any kernel
        // OOB-read (now routed here via the sentinel block_table_flat default
        // fill in upload_batch_metadata_*) would otherwise dequant random
        // bytes and inject garbage into attention scores.
        let mut kv_cache = kv_cache;
        let dummy_kv_block = kv_cache.alloc_block()?;
        kv_cache.zero_block(dummy_kv_block, gpu.as_ref(), gpu.default_stream())?;
        gpu.synchronize(gpu.default_stream())?;

        // Transposed lm_head twin, PADDED so the tile GEMM's 16-byte cp.async B
        // loads are aligned. The stride must be a multiple of 16; 128 also keeps
        // whole N-tiles. Without the pad, N = vocab = 248077 (ODD) misaligns 15 of
        // every 16 k-rows => the campaign's long-standing sticky CUDA 716.
        // Default ON (kill: ATLAS_NO_LMHEAD_TGEMM=1). KV impact is nil: the pool
        // reads 4759 blocks with the twin vs 4757 without. See STATE.md.
        let lm_head_nvfp4_t = match (&lm_head_nvfp4, lmhead_tgemm_enabled()) {
            (Some(w), true) => {
                let (t, stride) =
                    crate::weight_map::QuantizedWeight::transpose_concat_for_gemm_padded(
                        gpu.as_ref(),
                        &[(w, config.vocab_size)],
                        config.hidden_size,
                        16,
                        128,
                    )?;
                // A padded stride is only safe on targets whose `w4a16_gemm_t`
                // actually takes `ldb`. Every served vocab except this one is a
                // multiple of 128 (stride == vocab, so `ldb` is a no-op and any
                // kernel is fine); when it is NOT, a kernel missing the parameter
                // strides by N and shears every row past the first — silently, on
                // architectures that tolerate the misalignment. Say so loudly.
                if stride != config.vocab_size {
                    tracing::warn!(
                        "lm_head twin uses a PADDED stride ({} != vocab {}): this target's \
                         w4a16_gemm_t MUST accept the `ldb` argument, or decode at padded_n>=5 \
                         will read sheared rows. Disable with ATLAS_NO_LMHEAD_TGEMM=1.",
                        stride,
                        config.vocab_size
                    );
                }
                tracing::info!(
                    "lm_head transposed twin: vocab={} -> padded stride={} (vocab%16={}), tile GEMM active",
                    config.vocab_size,
                    stride,
                    config.vocab_size % 16
                );
                Some((t, stride as u32))
            }
            _ => None,
        };
        // Drafter-side view of the twin: valid ONLY when the drafter head IS
        // the shared main head (`mtp_lm_head_nvfp4` absent) — the twin is a
        // transpose of `lm_head_nvfp4` specifically, so handing it to a
        // DEDICATED draft head would silently score drafts against the wrong
        // weight. Zero extra memory in the shared case (aliases the twin).
        let draft_lm_head_nvfp4_t = if mtp_lm_head_nvfp4.is_none() {
            lm_head_nvfp4_t
        } else {
            None
        };
        // Build MTP proposer (extracted to keep `new` under the file cap).
        let proposer: Option<Arc<dyn DraftProposer>> = super::impl_a1_init::build_mtp_proposer(
            use_speculative,
            mtp_weights,
            embed_tokens,
            draft_lm_head_nvfp4,
            draft_lm_head_nvfp4_t,
            &config,
            gpu.as_ref(),
            mtp_quant,
            mtp_vocab_size,
            max_seq_len,
            kv_cache.num_blocks(),
            &levers,
        );

        if self_speculative {
            let num_ssm = config.num_ssm_layers();
            let num_attn = config.num_attention_layers();
            tracing::info!(
                "Self-speculative decoding: ENABLED (skipping {} SSM layers, keeping {} attention layers)",
                num_ssm,
                num_attn,
            );
        }

        // MTP hidden state save buffer (1 × hidden_size FP32)
        let mtp_hidden_save = gpu.alloc(config.hidden_size * 4)?;
        // Batched-verify hidden stash: [VERIFY_WY_TABLE_SEQS, hidden] BF16 —
        // one slot per sequence of the widest batched verify chunk (n ≤ 32,
        // the K-vs-batch ladder envelope, SSOT in `crate::layer`). Only
        // meaningful with an MTP proposer — NULL otherwise (the batched
        // verify path self-gates on it via can_batch_verify).
        let verify_hidden_stash = if proposer.is_some() {
            gpu.alloc(crate::layer::VERIFY_WY_TABLE_SEQS * config.hidden_size * 2)?
        } else {
            DevicePtr::NULL
        };
        // Batched-verify WY pointer-table staging (fixed address for CUDA
        // graph stability; contents refreshed pre-graph every batched verify
        // step). One [h|Hi0|Hi1|Hi2] x 4-entry slice per GDN layer — ~6 KB.
        // NULL without an MTP proposer or on non-SSM models (path self-gates).
        let verify_wy_tables = if proposer.is_some() && config.num_ssm_layers() > 0 {
            let bytes = config.num_ssm_layers() * crate::layer::VERIFY_WY_LAYER_STRIDE_BYTES;
            let buf = gpu.alloc(bytes)?;
            gpu.memset(buf, 0, bytes)?;
            buf
        } else {
            DevicePtr::NULL
        };
        // Catch-up ring: 512 rows covers the gate's serial re-probe interval
        // (256 tokens) with 2x margin; ~4 MB at hidden 4096. Only allocated
        // when the staged feature is enabled.
        let mtp_catchup_ring = if crate::speculative::mtp_catchup_enabled() {
            gpu.alloc(super::types::MTP_CATCHUP_RING_ROWS * config.hidden_size * 2)?
        } else {
            DevicePtr::NULL
        };

        // Whole-prompt hidden capture buffer, [max_seq_len, hidden_size] BF16 —
        // 335 MB at 32k/h=5120. Backs BOTH halves of the drafter-context
        // feature (see `crate::model::drafter_context`); NULL here disables
        // prefill AND carry, since the carry path reads this buffer.
        //
        // Three conditions, all necessary: MTP must be active, the feature must
        // not be killed, and the head must be a precision the batched prefill
        // can actually run at — an NVFP4/FP8 MTP head would allocate this and
        // never write it.
        let mtp_prefill_hidden = if has_mtp
            && mtp_quant.supports_drafter_prefill()
            && crate::layers::mtp_drafter_prefill_enabled(&levers)
        {
            // Bound the capture to what the drafter can actually CONSUME.
            // `prefill_drafter` writes into the drafter's own KV, which is
            // capped at the DFlash ctx window, so a capture longer than that
            // window is memory nothing can read: 1342 MB at --max-seq-len
            // 131072 against a 16K window that can hold 168 MB of it.
            // A prompt past the window simply does not get the whole-prompt
            // drafter prefill (the coverage check at the consume site already
            // handles that — blind beats poisoned); it costs acceptance on
            // very long cold turns, not correctness.
            //
            // Pure-MTP serves keep the full ceiling: this narrowing is only
            // sound because the DFlash drafter's own capacity is the binding
            // constraint, and that reasoning does not transfer.
            let capture_rows = if dflash_kgamma > 0 {
                let cap = crate::layers::dflash_ctx_cap();
                if cap == 0 {
                    max_seq_len
                } else {
                    max_seq_len.min(cap)
                }
            } else {
                max_seq_len
            };
            let bytes = capture_rows * config.hidden_size * 2;
            tracing::info!(
                "MTP drafter context: allocating {:.0} MB prompt-hidden capture \
                 ({} x {} BF16)",
                bytes as f64 / 1e6,
                capture_rows,
                config.hidden_size,
            );
            gpu.alloc(bytes)?
        } else {
            if has_mtp
                && !mtp_quant.supports_drafter_prefill()
                && crate::layers::mtp_drafter_prefill_enabled(&levers)
            {
                tracing::info!(
                    "MTP drafter context: INACTIVE — the batched drafter prefill \
                     needs a BF16 MTP head (--mtp-quantization bf16); this head is \
                     {mtp_quant:?}. No prompt-hidden capture allocated.",
                );
            }
            DevicePtr::NULL
        };

        // DFlash 5-layer hidden-state stack. Allocated only when a
        // BlockDiffusionDraftHead is the active proposer (`config.dflash_capture_layers`
        // populated by the loader from the drafter's `dflash_config.target_layer_ids`).
        // Size: N_capture × hidden_size × bf16 (typically 5 × 2048 × 2 = 20 KB).
        let dflash_capture_layers: Vec<usize> = config.dflash_capture_layers.clone();
        // Row capacity of the K-row capture buffer. KMAX = dflash_kgamma (=17 >=
        // max verify K = gamma) so the K=gamma EAGLE path can capture every verify row;
        // pre-fix paths use only rows 0-1. Stored on the model as the single
        // source of truth so `try_dflash_capture_all` can bound its writes.
        //
        // Widened to `max_batch_size` K-row BANDS: a cross-sequence batched
        // K=gamma verify (and batched decode) captures every (sequence, row)
        // pair, sequence i writing band i at row `i * dflash_kgamma`. The
        // scheduler reads a band back through `commit_ctx`'s `scratch_row`.
        // Cost is trivial (8 seqs x 9 rows x 5 layers x 5120 x 2 B ~ 3.7 MB)
        // and the single-sequence paths keep using band 0 unchanged.
        let dflash_hidden_save_rows = if dflash_capture_layers.is_empty() {
            0
        } else {
            dflash_kgamma.max(2) * max_batch_size.max(1)
        };
        let dflash_hidden_save = if dflash_capture_layers.is_empty() {
            None
        } else {
            let n = dflash_capture_layers.len();
            // Row-major K-row buffer: [row0 | row1 | ... | row_{KMAX-1}], each row =
            // n_capture * hidden_size * bf16. Rows 0/1 keep their legacy offsets
            // (0 and ctx_slot_bytes) so all K=2 readers (propose row 0,
            // dflash_accept_append row 1) are unaffected.
            Some(gpu.alloc(dflash_hidden_save_rows * n * config.hidden_size * 2)?)
        };

        // EP command buffer for token broadcast (4 bytes, u32)
        let ep_cmd_buf = gpu.alloc(4)?;

        // SOLID Incr-4: dedicated fixed-address buffer for the batched-decode MoE
        // per-row fold map. max_batch_size i32 rows (e.g. 32·4 = 128 B). Allocated
        // unconditionally like ep_cmd_buf/mtp_hidden_save — self.lora is populated
        // post-construction (set_lora_weights), so we can't gate on it here, and
        // the cost is negligible. Fixed address → graph-safe; contents copied per
        // decode step. Moving the map off the +160 metadata gap frees seq_slot to
        // reclaim +128..+256, lifting the concurrent-LoRA decode cap from 8 to 32.
        let moe_row_adapter_buf = gpu.alloc(max_batch_size.max(1) * 4)?;

        // Secondary stream + event for pipelining checkpoint D2D with MTP propose.
        let secondary_stream = gpu.create_stream()?;
        let secondary_event = gpu.create_event()?;
        // Event ordering SSM-snapshot saves (default stream) before a warm
        // Marconi restore (prefill stream). See `snapshot_event` doc in types.rs.
        let snapshot_event = gpu.create_event()?;

        // EP/TP: register the all-reduce target buffers with NCCL (caches the
        // IB/RoCE memory registration, enabling zero-copy user-buffer
        // collectives) and provide the bf16_add kernel for the 2-rank
        // send/recv fast path.
        //   - moe_output: EP MoE reduce + ALL GDN HeadParallel SSM out_proj
        //     reduces (decode `ssm_forward`, batched decode, multi-seq
        //     batched, prefill, prefill phase-3 all write out_proj into
        //     `buffers.moe_output()`).
        //   - norm_output: attention o_proj decode output
        //     (`attention_forward_oproj` writes o_out = `buffers.norm_output()`),
        //     reduced per attention layer under TP.
        if let Some(ref comm) = comm
            && comm.world_size() == 2
        {
            let moe_ptr = buffers.moe_output().0;
            let moe_bytes = buffers.sizes().moe_output;
            match comm.register_buffer(moe_ptr, moe_bytes) {
                Ok(_) => tracing::info!("Registered moe_output ({moe_bytes} B) with NCCL"),
                Err(e) => tracing::warn!("ncclCommRegister moe_output failed (non-fatal): {e}"),
            }
            let norm_ptr = buffers.norm_output().0;
            let norm_bytes = buffers.sizes().norm_output;
            match comm.register_buffer(norm_ptr, norm_bytes) {
                Ok(_) => tracing::info!("Registered norm_output ({norm_bytes} B) with NCCL"),
                Err(e) => tracing::warn!("ncclCommRegister norm_output failed (non-fatal): {e}"),
            }
            match gpu.kernel("bf16_add", "bf16_add_inplace") {
                Ok(k) => comm.set_add_kernel(k.0),
                Err(e) => {
                    tracing::warn!("bf16_add_inplace kernel not found (send/recv disabled): {e}")
                }
            }
        }

        // Allocate pinned host staging buffer for batched metadata H2D.
        let pinned_bytes = buffers.sizes().scratch.max(64 * 1024);
        let pinned_ptr = gpu.alloc_host_pinned(pinned_bytes)?;
        tracing::info!("Pinned metadata staging: {} KB", pinned_bytes / 1024);
        let max_batch_tokens = buffers.max_batch_tokens();
        let pinned_staging = std::cell::UnsafeCell::new(PinnedMetaStaging {
            ptr: pinned_ptr,
            bytes: pinned_bytes,
            positions: Vec::with_capacity(max_batch_tokens),
            positions_h: Vec::with_capacity(max_batch_tokens),
            positions_w: Vec::with_capacity(max_batch_tokens),
            slots: Vec::with_capacity(max_batch_tokens),
        });

        // SSM state normalization kernel + pointer buffer (for chunked prefill).
        let ssm_norm_k = gpu
            .kernel("ssm_state_norm", "ssm_state_clamp_norm_fused")
            .unwrap_or(KernelHandle(0));
        let ssm_norm_f16_k = gpu
            .kernel("ssm_state_norm", "ssm_state_clamp_norm_fused_f16")
            .unwrap_or(KernelHandle(0));
        let ssm_h_f32_to_f16_k =
            crate::layers::try_kernel(gpu.as_ref(), "ssm_h_dtype", "ssm_h_state_f32_to_f16");
        let ssm_h_f16_to_f32_k =
            crate::layers::try_kernel(gpu.as_ref(), "ssm_h_dtype", "ssm_h_state_f16_to_f32");

        // Logit softcapping (Gemma-4: cap=30.0). Only load if model uses it.
        let logit_softcap_kernel = if config.final_logit_softcapping > 0.0 {
            gpu.kernel("logit_softcap", "logit_softcap_bf16")
                .unwrap_or_else(|e| {
                    tracing::warn!("logit_softcap kernel not found: {e}");
                    KernelHandle(0)
                })
        } else {
            KernelHandle(0)
        };
        // FP32 softcap variant — only loaded when both softcap and FP32
        // residual are active (i.e. Gemma-4 dense). Other models keep the
        // BF16 softcap (or no softcap at all).
        // The FP32 logit softcap variant required an FP32 residual stream,
        // which no longer exists, so the BF16 softcap path is always taken.
        let logit_softcap_fp32_kernel = KernelHandle(0);
        // FP32 logits gate. The LM head produces FP32 (rather than BF16)
        // logits when the residual stream is FP32 AND the LM head is a
        // dense BF16 weight (no NVFP4 quant). NVFP4 LM heads keep their
        // existing path because that quantization is a much larger
        // precision floor than the BF16 store; FP32 wouldn't help there.
        // Today this only affects Gemma-4 dense (model_type=="gemma4",
        // num_experts==0, tied BF16 embed→lm_head).
        // Gemma-4-31B FP32 lm_head experiment. Disabled by default —
        // session 2026-05-01 verified the BF16 lm_head store is NOT the
        // source of Gemma-4's haiku argmax flip: FP32 view of step-1
        // logits keeps top1=` a` (21.85), top2=` waves` (21.706) — same
        // 0.14-margin tiebreak as BF16. The drift is upstream in attention
        // or MLP, not in the lm_head precision boundary. Code paths kept
        // wired so a future bisection (Phase 2 of the plan) can re-enable
        // via `ATLAS_GEMMA4_FP32_LMHEAD=1`. Keep `use_fp32_logits=false`
        // by default so the rest of the model behaves identically to the
        // pre-fix BF16 path on every model family.
        // FP32 lm_head + softcap. Default OFF — empirically the gain on
        // Gemma-4-31B is marginal (Creative occasionally cleaner; fib still
        // fails the same broken-indentation pattern) but the cost is huge:
        // FP32 forces host-side sampling (vocab=262144 × 4 bytes per
        // decode step → ~1 MB D2H per token) which crushes decode TPS
        // from ~35 tok/s to ~6 tok/s on Gemma-4-31B. Not worth it without
        // a GPU-side FP32 argmax kernel. `ATLAS_GEMMA4_FP32_LMHEAD=1`
        // re-enables for bisection / future work.
        //
        // The earlier "FP32 doesn't fix haiku" comment in this file was
        // arrived at via incomplete bisection (the scheduler readback
        // always assumed BF16 — see commit 16b2f3a's commit body). The
        // 2026-05-01 evening run with the dispatch wired confirmed the
        // bisection's *qualitative* conclusion: FP32 lm_head + softcap
        // doesn't materially fix Gemma-4's structural NVFP4 attention
        // drift on greedy code generation. Fix is upstream of lm_head.
        // FP32 logits (ATLAS_GEMMA4_FP32_LMHEAD) required an FP32 residual
        // stream as a precondition. With the residual stream now always BF16,
        // the FP32 logits path can never activate, so it is permanently off.
        let use_fp32_logits = false;
        // Dedicated FP32 logits scratch — only the single-token decode path
        // uses it. Prefill and batched-decode lm_head still write BF16 to the
        // shared `buffers.logits()`. Sized for one row of `vocab_size` FP32.
        let logits_fp32_buf = if use_fp32_logits {
            let bytes = config.vocab_size * 4;
            let p = gpu.alloc(bytes)?;
            tracing::info!(
                "FP32 LM head + softcap active (model_type={}, vocab={}). \
                 Decode logits scratch: {} bytes.",
                config.model_type,
                config.vocab_size,
                bytes,
            );
            p
        } else {
            DevicePtr::NULL
        };

        // Embedding scale (Gemma-4: sqrt(hidden_size)). Only load if model uses it.
        let embed_scale_kernel = if config.embed_scale > 0.0 {
            gpu.kernel("embed_scale", "bf16_scale_inplace")
                .unwrap_or_else(|e| {
                    tracing::warn!("embed_scale kernel not found: {e}");
                    KernelHandle(0)
                })
        } else {
            KernelHandle(0)
        };
        if config.embed_scale > 0.0 {
            tracing::info!(
                "Embedding scale: {:.4} (sqrt({}))",
                config.embed_scale,
                config.hidden_size
            );
        }
        let ssm_norm_ptrs = if ssm_pool.num_ssm_layers > 0 {
            gpu.alloc(ssm_pool.num_ssm_layers * 8)
                .unwrap_or(DevicePtr::NULL)
        } else {
            DevicePtr::NULL
        };

        // GDN prefill buffers: sized for max_batch_tokens (the prefill chunk size),
        // NOT max_seq_len. For prompts longer than this, prefill_twophase falls back
        // to standard chunked prefill which carries h_state/conv_state between chunks.
        // The GDN recurrence is sequential anyway, so chunking is mathematically identical.
        let (gdn_qkv, gdn_gate_beta, gdn_out, gdn_z, gdn_buf_len) =
            super::impl_a1_init::build_gdn_prefill_buffers(
                &config,
                max_batch_tokens,
                max_seq_len,
                gpu.as_ref(),
            )?;

        // FP8 calibration only runs when the cache is actually FP8 — the
        // observe() call in decode.rs sits inside the FP8 cache branch. For
        // BF16 or NVFP4 caches the MODEL.toml fp8_kv_calibration_tokens
        // value is dead code and must not suppress CUDA graphs.
        let has_fp8_calibration = config.fp8_kv_calibration_tokens > 0
            && kv_cache.dtype() == spark_runtime::kv_cache::KvCacheDtype::Fp8;
        // Feature-2 overlay kernels: resolve before `gpu` is moved into Self.
        let overlay_kernels = crate::layers::ops::token_overlay::OverlayKernels::new(gpu.as_ref());
        Ok(Self {
            // Installed by the factory after construction: the layers read
            // from the store during `new`, so it cannot be moved in here.
            weight_store: None,
            config,
            dispatch: crate::layers::ops::GemmDispatch::from_env(),
            derived: crate::layers::ops::DerivedWeights::new(),
            levers,
            stats: ops::ModelStats::new(),
            #[cfg(feature = "cuda")]
            innerq: gpu.kernel_registry().and_then(|reg| {
                let driver = crate::layers::qwen3_attention::InnerQDriver::from_env(reg)?;
                match driver.start() {
                    Ok(()) => Some(driver),
                    Err(e) => {
                        tracing::warn!("InnerQ calibration disabled: start() failed: {e:#}");
                        None
                    }
                }
            }),
            embed_tokens,
            ngram_embed: None,
            final_norm,
            lm_head_weight,
            lm_head_nvfp4,
            lm_head_nvfp4_t,
            lm_head_fp8,
            layers,
            buffers,
            lora: None,
            lora_rotatable: false,
            kv_cache: Mutex::new(kv_cache),
            gpu,
            rms_norm_kernel,
            dense_gemv_kernel,
            dense_gemv_fp32out_kernel,
            w4a16_gemv_kernel,
            w4a16_gemv_logits_kernel,
            w4a16_gemm_t_kernel,
            w4a16_gemm_t_bf16_kernel,
            w4a16_gemm_kernel,
            w4a16_gemv_batch2_kernel,
            w4a16_batchm,
            w4a16_gemv_batch16_kernel,
            dense_gemv_fp8w_kernel,
            dense_gemv_fp8w_batch2_kernel,
            dense_gemm_kernel,
            dense_gemv_batchm_kernel,
            argmax_kernel,
            argmax_batch_kernel,
            argmax_logits_kernel,
            batched_embed_kernel,
            fill_slots_kernel,
            decode_graph: Mutex::new(std::collections::HashMap::new()),
            batch_decode_graphs: Mutex::new((HashMap::new(), 0)),
            // Suppress graphs during FP8 calibration only. MLA used to be
            // suppressed because an internal sync was placed inside the graph
            // capture region — that sync is now conditional on eager mode
            // (see line ~3881), so graphs work for MLA too. The zero_all call
            // at line ~3751 runs in Phase 1 BEFORE begin_capture, so it is
            // naturally outside the captured region.
            suppress_graphs: std::sync::atomic::AtomicBool::new(
                has_fp8_calibration
                    || std::env::var("ATLAS_DIAG_GEMMA4").is_ok_and(|v| v == "1" || v == "true")
                    // PCND diagnostic: force eager decode (no CUDA-graph capture)
                    // so ATLAS_DEBUG_SYNC_KERNELS can synchronize per launch and
                    // surface async faults at the culprit kernel. Default-off.
                    || std::env::var("ATLAS_DEBUG_NO_GRAPH").as_deref() == Ok("1"),
            ),
            ssm_pool,
            ssm_snapshots,
            ssm_tier_store,
            max_blocks_per_seq,
            dummy_kv_block,
            profile,
            profile_first_pending: std::sync::atomic::AtomicBool::new(profile_first),
            proposer,
            mtp_hidden_save,
            verify_hidden_stash,
            mtp_catchup_ring,
            mtp_catchup_meta: parking_lot::Mutex::new((0, 0)),
            mtp_prefill_hidden,
            mtp_prefill_capacity: if mtp_prefill_hidden.is_null() {
                0
            } else {
                max_seq_len
            },
            mtp_prefill_capture_len: std::sync::atomic::AtomicUsize::new(0),
            mtp_prefill_capture_gen: std::sync::atomic::AtomicU64::new(0),
            mtp_carry: parking_lot::Mutex::new(None),
            mtp_store_range: parking_lot::Mutex::new((0, 0)),
            dflash_hidden_save,
            dflash_hidden_save_rows,
            dflash_kgamma,
            dflash_capture_layers,
            verify2_graph: Mutex::new(std::collections::HashMap::new()),
            verify3_graph: Mutex::new(std::collections::HashMap::new()),
            verify4_graph: Mutex::new(std::collections::HashMap::new()),
            verify_batched_graphs: Mutex::new((std::collections::HashMap::new(), 0)),
            verify_wy_tables,
            // Nothing staged yet: the buffer was memset to zero above, and no
            // key describes zero, so the first verify step always uploads.
            verify_wy_cache: Mutex::new(None),
            verify_kgamma_graph: Mutex::new(std::collections::HashMap::new()),
            fused_graph: Mutex::new(std::collections::HashMap::new()),
            prefix_cache,
            secondary_stream,
            secondary_event,
            snapshot_event,
            comm,
            ep_cmd_buf,
            ep_protocol_v2: matches!(std::env::var("ATLAS_EP_PROTOCOL").as_deref(), Ok("v2")),
            self_speculative,
            last_mtp_hidden_idx: std::sync::atomic::AtomicUsize::new(0),
            vision_encoder,
            vision_embed_patches: Mutex::new(0),
            vision_image_grids: Mutex::new(Vec::new()),
            vision_row_base: Mutex::new(0),
            vision_grid_base: Mutex::new(0),
            vision_owned_images: Mutex::new(0),
            pinned_staging,
            ssm_checkpoint_interval,
            ssm_state_norm_kernel: ssm_norm_k,
            ssm_state_norm_f16_kernel: ssm_norm_f16_k,
            ssm_h_f32_to_f16_kernel: ssm_h_f32_to_f16_k,
            ssm_h_f16_to_f32_kernel: ssm_h_f16_to_f32_k,
            ssm_h_f16_scratch: std::sync::OnceLock::new(),
            ssm_norm_ptrs_buf: ssm_norm_ptrs,
            moe_row_adapter_buf,
            gdn_buf_qkv: gdn_qkv,
            gdn_buf_gate_beta: gdn_gate_beta,
            gdn_buf_out: gdn_out,
            gdn_buf_z: gdn_z,
            gdn_buf_max_len: gdn_buf_len,
            logit_softcap_kernel,
            logit_softcap_fp32_kernel,
            use_fp32_logits,
            logits_fp32_buf,
            embed_scale_kernel,
            overlays: None,
            overlay_kernels,
            overlay_route_slot: std::sync::atomic::AtomicI32::new(-1),
            decode_moe_route: std::sync::atomic::AtomicI32::new(1), // Fold (safe default)
        })
    }
}
