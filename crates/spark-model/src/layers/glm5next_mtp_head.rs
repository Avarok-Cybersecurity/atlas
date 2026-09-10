// SPDX-License-Identifier: AGPL-3.0-only

//! `Glm5NextMtpHead` — GLM-5.3's MTP block as a [`DraftProposer`].
//!
//! One draft token per `forward_one`:
//!
//! ```text
//! x = eh_proj( concat( enorm(embed[token]), hnorm(target_hidden) ) )   [1, 2H] -> [1, H]
//! x = layers.45(x)                       DSA + routed MoE, PLAIN residual (no mHC)
//! logits = lm_head( shared_head.norm(x) )              the target's own BF16 head
//! draft  = argmax(logits)
//! ```
//!
//! 🔴 **This block is SHARDED — EP-sharded routed MoE (144 of 288 experts per rank) and a
//! row-parallel DSA `o_proj`** — unlike the Qwen and DeepSeek-V4 MTP modules, which load every
//! expert on every rank. So it needs the communicator exactly as a text layer does.
//!
//! 🪤 Historically it ran WITHOUT one and on RANK 0 ONLY (`run_mtp_propose_multi_dispatch`:
//! *"Rank 1 does not participate in MTP propose"*), which is correct for V4 and wrong here: the
//! drafter proposed from half the routed sum and half the attention output. Lossless — the
//! target verifies every draft — so the only symptom was acceptance. `ATLAS_MTP_EP_PROPOSE=1`
//! turns on BOTH halves of the fix: the worker executes propose on `EP_CMD_MTP_PROPOSE`, and
//! `needs_comm()` then hands the block a comm. Turning on only the second half is `t58`, which
//! deadlocked.
//!
//! 🪤 The embedding is read as a POINTER into the shared table, not a gather: the row for token
//! `t` is `embed_tokens + t * hidden * 2`. No kernel, no copy.

use anyhow::{Result, bail};
use parking_lot::Mutex;
use std::any::Any;

use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

use crate::layer::{ForwardContext, LayerState};
use crate::layers::glm5next_dsa::state::Glm5NextDsaState;

use crate::layers::ops;
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_loader::Glm5NextMtpModule;
use crate::weight_map::DenseWeight;

/// Per-sequence drafter state: the block's own indexer cache and KV blocks.
pub struct Glm5NextMtpProposerState {
    dsa: Glm5NextDsaState,
    /// Tokens the drafter has written. Rolled back by `after_verify` on a rejected draft.
    seq_len: usize,
    block_table: Vec<u32>,
    /// How many drafts the last `propose` wrote, so `after_verify` knows what to trim.
    last_drafted: usize,
    /// Scratch: `[2, hidden]` BF16 concat, `[hidden]` BF16 block input, `[vocab]` BF16 logits,
    /// `[1]` u32 argmax.
    concat: DevicePtr,
    x: DevicePtr,
    logits: DevicePtr,
    arg: DevicePtr,
    /// `[max_r0, max_r1, idx_r0, idx_r1]` f32, for the vocab-sharded head's cross-rank pick.
    head_xchg: DevicePtr,
    /// Once-only guard for `free_state`. `DevicePtr` has no `Drop`, so the
    /// release is explicit; this makes a second call a no-op, preserving the
    /// property the consuming `Glm5NextDsaState::free(self)` used to give.
    released: bool,
}

impl ProposerState for Glm5NextMtpProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Widest cross-sequence batch the GLM drafter can be asked for in ONE forward.
///
/// 🔴 SSOT, and it must be: the head sizes its own row-contiguous scratch from this, and the
/// MTP module's MoE WORKSPACE is sized from it in `weight_loader::glm5_next_mtp`. Those two
/// are allocated in different crates' worth of code and only meet at runtime, where a
/// disagreement is not a compile error — it is `forward_moe` bailing "N rows do not fit a
/// workspace built for 1" on every batched propose, which the caller then reports as a
/// declined group. That failure is INVISIBLE in the throughput number it produces: every
/// sequence in the group silently loses its drafts for that step, the verify batch narrows
/// to whatever is left, and the arm measures SLOWER than the serial one it was meant to
/// beat. Measured 2026-09-10 at exactly this bug: C=4 aggregate 16.5 tok/s batched vs 20.5
/// serial, accept p1 0.31 vs 0.49 — all of it the fallback, none of it the batching.
///
/// 🪤 8, the DECODE band (`DENSE_GEMV_BATCHM_DECODE_MAX_M`), not the GEMV kernel's own 16:
/// the batched verify this propose feeds is capped at 8 rows on the same reasoning, so a
/// wider drafter batch could never be filled. Raising it means raising both, and re-measuring
/// the band — see the constant's own doc.
pub const MTP_BATCH_PROPOSE_MAX_SEQS: usize =
    crate::layers::ops::DENSE_GEMV_BATCHM_DECODE_MAX_M as usize;

/// Row-contiguous scratch for the cross-sequence batched propose.
///
/// The per-sequence [`Glm5NextMtpProposerState`] buffers are one row each and
/// live in different allocations, so they cannot be the input to a batched
/// GEMV — the kernels read `[M, K]` as one packed block. This is that block,
/// allocated once on the head and sized to `max_rows`.
///
/// 🪤 It is scratch, NOT state: everything in here is written and consumed
/// inside a single `propose_batch`. The drafter's real per-sequence state —
/// the indexer cache, the KV blocks, `seq_len` — stays where it was.
struct BatchScratch {
    /// `[max_rows, 2 * hidden]` BF16: `enorm(embed)` ‖ `hnorm(target_hidden)`.
    concat: DevicePtr,
    /// `[max_rows, hidden]` BF16: the block input, then the block output, then
    /// (normed in place) the head input — exactly the three lives `st.x` has
    /// on the per-sequence path.
    x: DevicePtr,
    /// `[max_rows, head_n]` BF16.
    logits: DevicePtr,
    /// `[max_rows]` u32 argmax indices, drained in ONE D2H.
    arg: DevicePtr,
    /// `[max_rows, 8]` BF16 lanes for the vocab-sharded cross-rank pick — the
    /// per-sequence 8-lane packing of `forward_one`, laid end to end so all
    /// `n` sequences settle in ONE all-reduce instead of `n`.
    xchg: DevicePtr,
    max_rows: usize,
}

pub struct Glm5NextMtpHead {
    module: Glm5NextMtpModule,
    embed_tokens: DenseWeight,
    lm_head: DenseWeight,
    /// One-layer pool of its own: the drafter's entries must be trimmable independently of the
    /// target's, and the target's pool is sized to its own 11 KV-consuming layers.
    kv_cache: Mutex<PagedKvCache>,
    rms_norm_k: KernelHandle,
    gemv_k: KernelHandle,
    argmax_k: KernelHandle,
    hidden: usize,
    vocab: usize,
    max_seq_len: usize,
    /// Vocab shard of the shared `lm_head` this rank sweeps: `[head_v0, head_v0 + head_n)`.
    /// `head_n == vocab` when the head is not sharded (single rank, or a vocab that does not
    /// divide, or EP propose off).
    head_rank: usize,
    head_v0: usize,
    head_n: usize,
    /// FP8 E4M3 copy of THIS RANK'S vocab shard of `lm_head`, drafting only.
    ///
    /// 🔴 Correctness-safe by construction and not a precision compromise: the target
    /// verifies every drafted token with its own BF16 `lm_head_batched`, so an approximate
    /// draft head can only move the ACCEPTANCE rate, never an emitted token. Same argument
    /// the NVFP4 `mtp_lm_head` decouple already makes in `factory::lm_head_setup`.
    ///
    /// Worth 2.66 -> ~1.33 ms per draft sweep, twice a K=3 step, against the 8.14 ms/step
    /// the whole drafter costs (nsys 2026-08-29). Kill switch `ATLAS_GLM_MTP_HEAD_FP8=0`.
    head_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    gemv_fp8w_k: KernelHandle,
    /// Batched (`M <= 8`) BF16 GEMV — `eh_proj` for every sequence at once, and
    /// the unsharded `lm_head`. Zero when the target has no such kernel, which
    /// is one of the conditions that keeps `propose_batch_max` at 1.
    gemv_batchm_k: KernelHandle,
    /// One CTA per row instead of one CTA for the whole call.
    argmax_batch_k: KernelHandle,
    /// Register-tiled batched row-scaled FP8 GEMV, the DFlash drafter's own
    /// propose kernel — reused verbatim rather than reinvented. It is what
    /// makes the batched SHARDED head one sweep instead of `n`.
    ///
    /// 🪤 NOT bit-identical to `n` calls of `dense_gemv_fp8w`: it is a
    /// different reduction shape, so the batched drafter's logits differ from
    /// the serial drafter's in the last bits. That is correctness-free — the
    /// target verifies every drafted token with its own BF16 `lm_head`, the
    /// same argument the FP8 draft head itself rests on — but it does mean a
    /// batched-vs-serial A/B compares ACCEPTANCE, not bytes.
    /// `ATLAS_GLM_MTP_BATCH_PROPOSE=1` is the bisect lever.
    fp8_batchm_k: KernelHandle,
    /// `None` when the batched propose cannot run at all (a kernel is missing
    /// or an allocation failed). Never a silent half-capability.
    batch: Option<BatchScratch>,
}

/// Rows the GLM drafter can ever be asked for: the served context, clamped to what its own
/// DSA indexer cache can hold. ANOMALIES A59 — see the note in [`Glm5NextMtpHead::new`].
///
/// Derived from `max_dsa_context`, never a literal: the ceiling is a function of the top-k
/// kernel's shared-memory budget and `index_kpool`, so a kernel or config change must move
/// this sizing with it.
fn drafter_context_rows(
    max_seq_len: usize,
    cfg: &crate::layers::glm5next_dsa::Glm5NextDsaConfig,
) -> usize {
    max_seq_len.min(crate::layers::glm5next_dsa::state::max_dsa_context(cfg))
}

impl Glm5NextMtpHead {
    pub fn new(
        module: Glm5NextMtpModule,
        embed_tokens: DenseWeight,
        lm_head: DenseWeight,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
        max_seq_len: usize,
    ) -> Result<Self> {
        let dsa = match &module.layer.mixer {
            crate::layers::glm5next_layer::Glm5NextMixer::Dsa(l) => l,
            _ => bail!("GLM MTP block is not a DSA layer"),
        };
        // 🔴 ANOMALIES A59. The drafter block is a DSA layer, so it can never reach a position
        // past `max_dsa_context` — the indexer cache `Glm5NextDsaState::alloc` reserves and
        // `advance` refuses to grow beyond (`glm5next_dsa/state.rs`). The TARGET's DSA layers
        // cap the servable context at the same number, so a sequence that would need row
        // `max_dsa_context` fails in the target before this block ever sees it. Everything
        // sized off `max_seq_len` here — the private KV pool below, the pre-claimed block
        // table in `alloc_state`, both bounds checks, and (through `prefill_hidden_rows`) the
        // model's `mtp_prefill_hidden` capture — is therefore dead weight above the ceiling.
        //
        // At `--max-seq-len 524288` that dead weight was 4.0 GiB of capture buffer plus
        // 0.5 GiB of drafter pool against the flat 4 GiB `cuda_headroom` that is the ONLY
        // reserve covering them (`serve_phases/preflight.rs`, `inference_reserve`) — both are
        // allocated AFTER the KV pool is sized, so nothing else accounts for them. The serve
        // ran ~0.8 GiB past its own `--gpu-memory-utilization` ceiling: measured 2026-08-30,
        // open128 -18.6 %, counting -10.6 %, TTFT 1.0 s -> 4.9 s, with acceptance, output and
        // error count unchanged. Handing 1.5 GB back (GMU 0.89) restored every number.
        //
        // Deriving the cap from the same function that sets the ceiling keeps it honest: the
        // day a segmented/radix select lifts `max_dsa_context`, this lifts with it.
        let max_seq_len = drafter_context_rows(max_seq_len, &dsa.cfg);
        // Matches the target's absorbed-MLA cache shape so the block's own `latent_write` and
        // paged gather land at the strides they already assume.
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_dim: dsa.cfg.kv_lora_rank,
            num_layers: 1,
            dtype: KvCacheDtype::Fp8,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let blocks = max_seq_len / kv_config.block_size + 2;
        let kv_cache = PagedKvCache::new(kv_config, blocks, gpu)?;
        // 🔴 THE DRAFTER'S OWN `lm_head` IS 7.3 OF ITS 8.84 ms (measured 2026-08-29,
        // `ATLAS_GLM_MTP_SKIP=head`: propose 8.84 -> 1.52 ms). It is a 1.27 GB BF16 sweep
        // (154,880 x 4,096) and the block around it is only 1.5 ms.
        //
        // Both ranks now run propose in lockstep, so split the sweep by VOCAB: each reads its
        // half of the rows and they exchange (max, argmax) through one 16-byte all-reduce. The
        // drafted token is EXACTLY the unsharded argmax — each rank computes exact logits over
        // full K for its own rows, so there are no partial sums to reassociate.
        //
        // 🪤 Vocab, not hidden. Rows of `[vocab, hidden]` are contiguous, so a vocab shard is a
        // base-pointer offset and a smaller `n`. A hidden shard would need a row STRIDE the
        // gemv kernel does not take — it assumes rows are packed at K.
        let head_world = config.tp_world_size.max(1);
        let head_rank = config.tp_rank;
        let head_n = if head_world > 1 && config.vocab_size.is_multiple_of(head_world) {
            config.vocab_size / head_world
        } else {
            config.vocab_size
        };
        // Quantise ONLY the rows this rank sweeps: `head_n * hidden` bytes, not the whole
        // vocab. A failure here is not fatal — fall back to the BF16 sweep.
        let gemv_fp8w_k = crate::layers::try_kernel(gpu, "gemv_fp8w", "dense_gemv_fp8w");
        let head_fp8 = if std::env::var("ATLAS_GLM_MTP_HEAD_FP8").as_deref() == Ok("0")
            || gemv_fp8w_k.0 == 0
        {
            None
        } else {
            let shard = DenseWeight {
                weight: lm_head
                    .weight
                    .offset(head_rank * head_n * config.hidden_size * 2),
            };
            match gpu
                .kernel("gemv_fp8w", "quantize_bf16_to_fp8")
                .and_then(|qk| {
                    crate::weight_map::quantize_to_fp8(
                        &shard,
                        head_n,
                        config.hidden_size,
                        gpu,
                        qk,
                        gpu.default_stream(),
                    )
                }) {
                Ok(q) => {
                    tracing::info!(
                        "GLM MTP: draft lm_head shard quantised to FP8 ({} rows x {}, {} MB)",
                        head_n,
                        config.hidden_size,
                        head_n * config.hidden_size / (1024 * 1024),
                    );
                    Some(q)
                }
                Err(e) => {
                    tracing::warn!("GLM MTP: FP8 draft head unavailable ({e:#}); staying BF16");
                    None
                }
            }
        };

        // ── Cross-sequence batched propose ──────────────────────────────────
        //
        // Every batched site needs a kernel the target may not carry, so resolve them all
        // OPTIONALLY and treat a single miss as "no batched propose". A half-batched propose
        // — batched block, per-sequence head — would be slower than the serial one and would
        // hide the miss; `propose_batch_max() == 1` says it plainly instead.
        let gemv_batchm_k =
            crate::layers::try_kernel(gpu, "dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm");
        let argmax_batch_k = crate::layers::try_kernel(gpu, "argmax", "argmax_bf16_batch");
        let fp8_batchm_k =
            crate::layers::try_kernel(gpu, "fp8_gemv_rt", "fp8_gemv_rowscale_batch8_rt2");
        // The SAME constant the MTP module's MoE workspace is built from — see its doc for
        // what a disagreement costs and how it hides.
        let max_rows = MTP_BATCH_PROPOSE_MAX_SEQS;
        let h = config.hidden_size;
        let batch = if gemv_batchm_k.0 == 0 || argmax_batch_k.0 == 0 {
            tracing::info!(
                "GLM MTP: batched propose unavailable (batchm={:#x} argmax_batch={:#x}); \
                 staying on the per-sequence drafter",
                gemv_batchm_k.0,
                argmax_batch_k.0,
            );
            None
        } else {
            // `head_n` rows, not `vocab`: the logits block only ever holds this rank's shard,
            // and unsharded `head_n == vocab` already.
            match (|| -> Result<BatchScratch> {
                Ok(BatchScratch {
                    concat: gpu.alloc(max_rows * 2 * h * 2)?,
                    x: gpu.alloc(max_rows * h * 2)?,
                    logits: gpu.alloc(max_rows * head_n * 2)?,
                    arg: gpu.alloc(max_rows * 4)?,
                    xchg: gpu.alloc(max_rows * 16)?,
                    max_rows,
                })
            })() {
                Ok(b) => {
                    tracing::info!(
                        "GLM MTP: batched propose armed (up to {max_rows} sequences per drafter \
                         forward, fp8_batchm={:#x})",
                        fp8_batchm_k.0,
                    );
                    Some(b)
                }
                Err(e) => {
                    tracing::warn!("GLM MTP: batched propose scratch alloc failed ({e:#}); \
                                    staying on the per-sequence drafter");
                    None
                }
            }
        };

        Ok(Self {
            module,
            embed_tokens,
            lm_head,
            kv_cache: Mutex::new(kv_cache),
            rms_norm_k: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
            hidden: config.hidden_size,
            vocab: config.vocab_size,
            max_seq_len,
            head_rank,
            head_v0: head_rank * head_n,
            head_n,
            head_fp8,
            gemv_fp8w_k,
            gemv_batchm_k,
            argmax_batch_k,
            fp8_batchm_k,
            batch,
        })
    }

    fn norm(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        n: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.rms_norm_k)
            .grid([1, 1, 1])
            .block([(n.min(1024)) as u32, 1, 1])
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(out)
            .arg_u32(n as u32)
            .arg_f32(self.module.layer.rms_eps)
            .launch(stream)
    }

    /// One draft token. Advances the drafter's KV and indexer state by exactly one row.
    fn forward_one(
        &self,
        token: u32,
        hidden_in: DevicePtr,
        position: usize,
        st: &mut Glm5NextMtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<u32> {
        let gpu = ctx.gpu;
        let h = self.hidden;
        if position >= self.max_seq_len {
            bail!(
                "GLM MTP drafter: position {position} is past the {} it was sized for",
                self.max_seq_len
            );
        }
        // 🪤 Embedding row by POINTER — `embed_tokens` is `[vocab, hidden]` BF16 and the row is
        // contiguous, so there is nothing to gather.
        let embed_row = self.embed_tokens.weight.offset(token as usize * h * 2);
        self.norm(gpu, embed_row, self.module.enorm, st.concat, h, stream)?;
        self.norm(
            gpu,
            hidden_in,
            self.module.hnorm,
            st.concat.offset(h * 2),
            h,
            stream,
        )?;
        ops::dense_gemv(
            gpu,
            self.gemv_k,
            st.concat,
            &self.module.eh_proj,
            st.x,
            h as u32,
            (2 * h) as u32,
            stream,
        )?;

        // The block writes its output back over `st.x` (plain residual, in place).
        if skip_block() {
            st.seq_len += 1;
        } else {
            let mut kv = self.kv_cache.lock();
            let dsa_state: &mut dyn LayerState = &mut st.dsa;
            self.module.layer.decode_one_for_drafter(
                st.x,
                dsa_state,
                &mut kv,
                st.seq_len,
                &mut st.block_table,
                ctx,
                stream,
            )?;
            drop(kv);
            st.seq_len += 1;
        }

        // TIMING ARM `ATLAS_GLM_MTP_SKIP=head`: everything from `shared_head.norm` on is
        // skipped and the draft is a constant. Drafts become garbage (p1 -> ~0) — the point is
        // the `propose` ms, which then reads as "the block alone". `=block` is the mirror arm.
        // Neither is a deployment; both are byte-safe because the target verifies every draft.
        if skip_head() {
            return Ok(0);
        }
        // 🪤 `shared_head.norm`, then the TARGET's own `lm_head`. The drafter ships no head of
        // its own — sharing it is what keeps a draft comparable to what the target would emit.
        self.norm(gpu, st.x, self.module.final_norm, st.x, h, stream)?;
        // Sharded only when this rank has a partner in the propose (`ctx.comm`); otherwise
        // `head_n == vocab` and this is the original full sweep.
        let sharded = ctx.comm.is_some() && self.head_n != self.vocab;
        let (w, n, v0) = if sharded {
            (
                DenseWeight {
                    weight: self.lm_head.weight.offset(self.head_v0 * h * 2),
                },
                self.head_n,
                self.head_v0,
            )
        } else {
            (self.lm_head, self.vocab, 0)
        };
        // 🪤 The FP8 copy covers `[head_v0, head_v0 + head_n)` ONLY, so it serves the sharded
        // sweep and nothing else. Unsharded (no partner in the propose) falls back to BF16
        // rather than reading rows that were never quantised.
        match self.head_fp8.filter(|_| sharded && n == self.head_n) {
            Some(q) => ops::dense_gemv_fp8w(
                gpu,
                self.gemv_fp8w_k,
                st.x,
                &q,
                st.logits,
                n as u32,
                h as u32,
                stream,
            )?,
            None => ops::dense_gemv(
                gpu,
                self.gemv_k,
                st.x,
                &w,
                st.logits,
                n as u32,
                h as u32,
                stream,
            )?,
        }
        ops::argmax_bf16(gpu, self.argmax_k, st.logits, st.arg, n as u32, stream)?;
        let mut out = [0u8; 4];
        gpu.synchronize(stream)?;
        gpu.copy_d2h(st.arg, &mut out)?;
        let local = u32::from_le_bytes(out) as usize;
        let Some(comm) = ctx.comm.filter(|_| sharded) else {
            return Ok((v0 + local) as u32);
        };
        // Exchange (max, argmax) in 8 BF16 lanes: `[val_r0, val_r1, then 3 base-256 digits of
        // each rank's global index]`. Each rank writes only its own lanes and leaves the
        // others zero, so a SUM all-reduce delivers both ranks' values untouched (`x + 0.0`
        // is exact).
        //
        // 🪤 `CommBackend::all_reduce` IS BF16-TYPED on this backend (`NcclDataType::Bfloat16`,
        // and at 2 ranks a paired Send/Recv plus a local BF16 add) — the byte count is a BF16
        // element count, not an opaque buffer. Packing f32s here instead reduced them as 8
        // BF16 lanes and silently corrupted both the value and the index: p1 0.875 -> 0.636,
        // measured. A token id needs 18 bits and BF16 carries 8, hence the digits; integers
        // through 256 are exact in BF16, and the logit lane is already BF16 so it round-trips
        // bit for bit.
        let mut lb = [0u8; 2];
        gpu.copy_d2h(st.logits.offset(local * 2), &mut lb)?;
        let g = v0 + local;
        let bf = |x: f32| ((x.to_bits() >> 16) as u16).to_le_bytes();
        let mut pack = [0u8; 16];
        pack[self.head_rank * 2..][..2].copy_from_slice(&lb);
        for d in 0..3 {
            let digit = ((g >> (8 * d)) & 0xFF) as f32;
            pack[4 + (self.head_rank * 3 + d) * 2..][..2].copy_from_slice(&bf(digit));
        }
        gpu.copy_h2d(&pack, st.head_xchg)?;
        comm.all_reduce_async(st.head_xchg.0, 16, stream)?;
        gpu.synchronize(stream)?;
        gpu.copy_d2h(st.head_xchg, &mut pack)?;
        let lane = |i: usize| {
            f32::from_bits(
                (u16::from_le_bytes(pack[i * 2..][..2].try_into().unwrap()) as u32) << 16,
            )
        };
        // `>=` makes the LOWER rank win a tie, identically on both ranks — the two drafter KV
        // streams must not diverge on a coin flip.
        let win = if lane(0) >= lane(1) { 0 } else { 1 };
        let idx = (0..3).fold(0usize, |a, d| {
            a + ((lane(2 + win * 3 + d) as usize) << (8 * d))
        });
        Ok(idx as u32)
    }

    /// ONE draft token for each of `n` sequences, in a single sweep of the drafter's weights.
    ///
    /// The cross-sequence sibling of [`Self::forward_one`], site for site:
    ///
    /// ```text
    ///   per-seq  enorm(embed[tok_i]) ‖ hnorm(hidden_i)   ->  concat[i]      [n, 2H]
    ///   BATCHED  eh_proj                                 ->  x              [n, H]
    ///   BATCHED  layer 45 (per-seq DSA, ONE MoE, ONE reduce, in place)      [n, H]
    ///   BATCHED  shared_head.norm (in place)                                [n, H]
    ///   BATCHED  lm_head shard                            ->  logits        [n, V']
    ///   BATCHED  argmax                                   ->  arg           [n]
    ///   ONE      cross-rank (max, argmax) all-reduce over n * 8 lanes
    /// ```
    ///
    /// Only the two input norms stay per-sequence, because an embedding row is a POINTER into
    /// a scattered table rather than a row of a packed block — and they are two 4096-element
    /// reductions, the cheapest thing in the drafter.
    ///
    /// 🔴 WHAT THIS SAVES, in the order it matters. (1) The `lm_head` shard: 7.3 of the
    /// drafter's 8.84 ms is that sweep, and it is the SAME weights for every sequence, so it
    /// collapses from `n` reads to one. (2) The routed MoE, which costs the same at one row as
    /// at eight. (3) The collectives: one attention reduce, one MLP reduce and one head
    /// exchange for the whole batch, each of which is a network round trip whose cost is
    /// latency, not bytes. (4) The D2H drain: one copy of `n` indices behind one sync.
    ///
    /// 🪤 Row `i` is sequence `i` end to end and the rows must not overlap — the DSA writes
    /// its `o_proj` back over the row it was handed, and `shared_head.norm` writes over `x` in
    /// place exactly as the per-sequence path does. That in-place norm is also why the caller
    /// feeds the NEXT draft from `x`: the per-sequence path's `hidden = st.x` is the normed
    /// block output, not the raw one, and this must match it or acceptance moves for a reason
    /// that has nothing to do with batching.
    fn forward_n(
        &self,
        tokens: &[u32],
        hiddens: &[DevicePtr],
        sts: &mut [&mut Glm5NextMtpProposerState],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Vec<u32>> {
        let gpu = ctx.gpu;
        let h = self.hidden;
        let n = tokens.len();
        let b = self
            .batch
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GLM MTP forward_n without batch scratch"))?;
        if n == 0 || n > b.max_rows || hiddens.len() != n || sts.len() != n {
            bail!(
                "GLM MTP forward_n: n={n} against {} hiddens / {} states / {} scratch rows",
                hiddens.len(),
                sts.len(),
                b.max_rows,
            );
        }
        for st in sts.iter() {
            if st.seq_len >= self.max_seq_len {
                bail!(
                    "GLM MTP drafter: row {} is past the {} it was sized for",
                    st.seq_len,
                    self.max_seq_len
                );
            }
        }

        // ── inputs: enorm(embed) ‖ hnorm(target hidden), one row pair per sequence ──
        for (i, st) in sts.iter().enumerate() {
            let _ = st;
            let embed_row = self.embed_tokens.weight.offset(tokens[i] as usize * h * 2);
            let row = b.concat.offset(i * 2 * h * 2);
            self.norm(gpu, embed_row, self.module.enorm, row, h, stream)?;
            self.norm(gpu, hiddens[i], self.module.hnorm, row.offset(h * 2), h, stream)?;
        }
        ops::dense_gemv_batchm(
            gpu,
            self.gemv_batchm_k,
            b.concat,
            &self.module.eh_proj,
            b.x,
            n as u32,
            h as u32,
            (2 * h) as u32,
            h as u32,
            stream,
        )?;

        // ── the block: per-sequence DSA, everything else batched ──
        //
        // The block table is cloned per sequence because the layer takes `&mut Vec<u32>` and
        // the three per-sequence facts live behind one `&mut` state. `alloc_state` claims the
        // drafter's whole private pool up front, so this never grows; writing it back keeps
        // that an assumption the code does not RELY on.
        let mut seq_lens: Vec<usize> = Vec::with_capacity(n);
        let mut block_tables: Vec<Vec<u32>> = Vec::with_capacity(n);
        for st in sts.iter() {
            seq_lens.push(st.seq_len);
            block_tables.push(st.block_table.clone());
        }
        {
            let mut layer_states: Vec<&mut dyn LayerState> = Vec::with_capacity(n);
            for st in sts.iter_mut() {
                layer_states.push(&mut st.dsa);
            }
            let mut kv = self.kv_cache.lock();
            self.module.layer.decode_n_for_drafter(
                b.x,
                n,
                &mut layer_states,
                &mut kv,
                &seq_lens,
                &mut block_tables,
                ctx,
                stream,
            )?;
        }
        for (i, st) in sts.iter_mut().enumerate() {
            st.block_table = std::mem::take(&mut block_tables[i]);
            st.seq_len += 1;
        }

        // ── head ──
        self.norm(gpu, b.x, self.module.final_norm, b.x, n, stream)?;
        let sharded = ctx.comm.is_some() && self.head_n != self.vocab;
        let (w, cols, v0) = if sharded {
            (
                DenseWeight {
                    weight: self.lm_head.weight.offset(self.head_v0 * h * 2),
                },
                self.head_n,
                self.head_v0,
            )
        } else {
            (self.lm_head, self.vocab, 0)
        };
        match self
            .head_fp8
            .filter(|_| sharded && cols == self.head_n && self.fp8_batchm_k.0 != 0)
        {
            Some(q) => ops::fp8_gemv_rowscale_batch8_rt2(
                gpu,
                self.fp8_batchm_k,
                b.x,
                &q,
                b.logits,
                n as u32,
                cols as u32,
                h as u32,
                stream,
            )?,
            None => ops::dense_gemv_batchm(
                gpu,
                self.gemv_batchm_k,
                b.x,
                &w,
                b.logits,
                n as u32,
                cols as u32,
                h as u32,
                cols as u32,
                stream,
            )?,
        }
        ops::argmax_bf16_batch(
            gpu,
            self.argmax_batch_k,
            b.logits,
            b.arg,
            cols as u32,
            n as u32,
            cols as u32,
            stream,
        )?;

        // ONE sync, ONE drain, `n` indices.
        let mut argbuf = vec![0u8; n * 4];
        gpu.synchronize(stream)?;
        gpu.copy_d2h(b.arg, &mut argbuf)?;
        let locals: Vec<usize> = (0..n)
            .map(|i| u32::from_le_bytes(argbuf[i * 4..][..4].try_into().unwrap()) as usize)
            .collect();

        let Some(comm) = ctx.comm.filter(|_| sharded) else {
            return Ok(locals.iter().map(|&l| (v0 + l) as u32).collect());
        };

        // ── ONE cross-rank pick for the whole batch ──
        //
        // The 8-lane packing is `forward_one`'s, verbatim and for its reasons (BF16-typed
        // collective, a token id needs 18 bits and BF16 carries 8, so the index travels as
        // three base-256 digits). The only change is that `n` of those groups are laid end to
        // end and settled in ONE all-reduce: SUM is elementwise, so groups do not interact,
        // and each rank still writes only its own lanes and leaves the others zero.
        let mut pack = vec![0u8; n * 16];
        let bf = |x: f32| ((x.to_bits() >> 16) as u16).to_le_bytes();
        for (i, &local) in locals.iter().enumerate() {
            let mut lb = [0u8; 2];
            gpu.copy_d2h(b.logits.offset((i * cols + local) * 2), &mut lb)?;
            let g = v0 + local;
            let base = i * 16;
            pack[base + self.head_rank * 2..][..2].copy_from_slice(&lb);
            for d in 0..3 {
                let digit = ((g >> (8 * d)) & 0xFF) as f32;
                pack[base + 4 + (self.head_rank * 3 + d) * 2..][..2].copy_from_slice(&bf(digit));
            }
        }
        gpu.copy_h2d(&pack, b.xchg)?;
        comm.all_reduce_async(b.xchg.0, n * 16, stream)?;
        gpu.synchronize(stream)?;
        gpu.copy_d2h(b.xchg, &mut pack)?;

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * 16;
            let lane = |j: usize| {
                f32::from_bits(
                    (u16::from_le_bytes(pack[base + j * 2..][..2].try_into().unwrap()) as u32)
                        << 16,
                )
            };
            // `>=` makes the LOWER rank win a tie, identically on both ranks.
            let win = if lane(0) >= lane(1) { 0 } else { 1 };
            let idx = (0..3).fold(0usize, |a, d| {
                a + ((lane(2 + win * 3 + d) as usize) << (8 * d))
            });
            out.push(idx as u32);
        }
        Ok(out)
    }

    /// Append `tokens.len() - 1` drafter CONTEXT rows: row `r` is pair key `row_base + r` =
    /// `(embed(tokens[r + 1]), hiddens row r)`. Used for both the whole-prompt prefill and the
    /// catch-up feed — the only difference between them is `row_base`.
    ///
    /// 🔴 THE ROW SPACE IS DENSE HERE, unlike the Qwen head's. This block's KV slot, indexer
    /// row and RoPE position are all `seq_len` (see `Glm5NextDsaLayer::write_kv_row`), so
    /// decoupling slot from position would mean plumbing a second scalar through the DSA
    /// layer. Instead every pair key from 0 up is written, which makes slot == key == RoPE and
    /// the drafter's geometry a copy of the target's — one uniform −1 RoPE shift against the
    /// convention (key `k` sits at RoPE `k`, not `k + 1`), which is invisible to a relative
    /// attention. Density is what `after_verify`'s no-trim and the catch-up feed maintain.
    ///
    /// Cost: NO MoE, NO attention, NO `lm_head` — a context row's block output is discarded,
    /// and both caches are pure functions of the row's input.
    #[allow(clippy::too_many_arguments)]
    fn rows_impl(
        &self,
        tokens: &[u32],
        hiddens: DevicePtr,
        row_base: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let st = match state
            .as_any_mut()
            .downcast_mut::<Glm5NextMtpProposerState>()
        {
            Some(s) => s,
            None => return Ok(0),
        };
        // Rows must append exactly at the drafter's current length, or the dense row space
        // grows a hole and every later RoPE position is wrong.
        if st.seq_len != row_base || tokens.len() < 2 {
            return Ok(0);
        }
        let h = self.hidden;
        let rows = tokens.len() - 1;
        if row_base + rows > self.max_seq_len {
            return Ok(0);
        }
        let gpu = ctx.gpu;
        let dbg = crate::speculative::mtp_refeed_debug();
        let prefill_full = std::env::var("ATLAS_GLM_MTP_PREFILL_FULL").ok().as_deref() == Some("1");
        let mut kv = self.kv_cache.lock();
        let Glm5NextMtpProposerState {
            dsa,
            seq_len,
            block_table,
            concat,
            x,
            ..
        } = st;
        for r in 0..rows {
            let embed_row = self
                .embed_tokens
                .weight
                .offset(tokens[r + 1] as usize * h * 2);
            self.norm(gpu, embed_row, self.module.enorm, *concat, h, stream)?;
            self.norm(
                gpu,
                hiddens.offset(r * h * 2),
                self.module.hnorm,
                concat.offset(h * 2),
                h,
                stream,
            )?;
            ops::dense_gemv(
                gpu,
                self.gemv_k,
                *concat,
                &self.module.eh_proj,
                *x,
                h as u32,
                (2 * h) as u32,
                stream,
            )?;
            let dsa_state: &mut dyn LayerState = dsa;
            // DIAGNOSTIC ARM `ATLAS_GLM_MTP_PREFILL_FULL=1`: build the row through the SAME
            // full-block path a propose uses, so "the KV-only shortcut is wrong" and "the
            // drafter's attention over real context is wrong" become separable. The shortcut
            // is the shipping path; this arm exists to convict or clear it.
            if prefill_full {
                self.module.layer.decode_one_for_drafter(
                    *x,
                    dsa_state,
                    &mut kv,
                    *seq_len,
                    block_table,
                    ctx,
                    stream,
                )?;
            } else {
                self.module.layer.drafter_write_kv_row(
                    *x,
                    dsa_state,
                    &mut kv,
                    *seq_len,
                    block_table,
                    ctx,
                    stream,
                )?;
            }
            if dbg {
                let fp = crate::speculative::hidden_fingerprint(gpu, hiddens.offset(r * h * 2), h);
                tracing::info!(
                    "GLM_MTP_DBG ctx row slot={} key={} tok={} fp_hidden={fp:016x}",
                    *seq_len,
                    row_base + r,
                    tokens[r + 1],
                );
            }
            *seq_len += 1;
        }
        Ok(rows)
    }
}

impl DraftProposer for Glm5NextMtpHead {
    /// `self.max_seq_len` is ALREADY capped at `max_dsa_context` by `new`, so this both
    /// rightsizes the model's capture buffer and keeps it in lockstep with the drafter's own
    /// bounds checks — a capture longer than the drafter's row space could never be read.
    /// ANOMALIES A59.
    fn prefill_hidden_rows(&self, max_seq_len: usize) -> usize {
        max_seq_len.min(self.max_seq_len)
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        let dsa = match &self.module.layer.mixer {
            crate::layers::glm5next_layer::Glm5NextMixer::Dsa(l) => {
                Glm5NextDsaState::alloc(gpu, &l.cfg)?
            }
            _ => bail!("GLM MTP block is not a DSA layer"),
        };
        let h = self.hidden;
        // Every block of the drafter's private pool, claimed up front: it serves one sequence
        // and a mid-decode allocation inside a captured region is not an option.
        let blocks = (self.max_seq_len / 16 + 2) as u32;
        Ok(Box::new(Glm5NextMtpProposerState {
            dsa,
            seq_len: 0,
            block_table: (0..blocks).collect(),
            last_drafted: 0,
            concat: gpu.alloc(2 * h * 2)?,
            x: gpu.alloc(h * 2)?,
            logits: gpu.alloc(self.vocab * 2)?,
            arg: gpu.alloc(4)?,
            head_xchg: gpu.alloc(16)?,
            released: false,
        }))
    }

    /// Release everything `alloc_state` allocated.
    ///
    /// Without this the head inherits `DraftProposer::free_state`'s no-op
    /// default, whose own doc says: *"`DevicePtr` has no `Drop`, so anything
    /// `alloc_state` allocated leaks unless it is explicitly freed here."* That
    /// is exactly what happened — every finished sequence leaked its indexer
    /// cache. The cache is sized from `serve_max_seq_len`, so the leak scales
    /// with `--max-seq-len`: ~806 MB per sequence at `--max-seq-len 131072`,
    /// which walks a unified-memory host into the ground in a handful of
    /// requests (ANOMALIES A75). `DeepseekV4MtpHead` and `MultiModuleMtp`
    /// already override this; the GLM port did not.
    ///
    /// 🔴 Invariant L2 (slot reuse), not a line order: when this slot is re-occupied its
    /// `decode_graph` and `verify2/3/4_graph` — which bake these exact pointers — must already
    /// be destroyed AND these pointers freed and nulled. `free_sequence` satisfies both.
    /// ANOMALIES A56 is the history; the invariant is slot reuse, not the order of the two
    /// blocks. (The `released` flag below is what makes a second call safe.)
    fn free_state(&self, gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        let st = state
            .as_any_mut()
            .downcast_mut::<Glm5NextMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid GLM MTP proposer state"))?;
        if st.released {
            return Ok(());
        }
        st.released = true;
        st.dsa.free(gpu)?;
        for p in [st.concat, st.x, st.logits, st.arg, st.head_xchg] {
            gpu.free(p)?;
        }
        // The drafter's private pool is claimed whole by `alloc_state`
        // (`(0..blocks).collect()`), not drawn from an allocator, so there is
        // nothing to hand back — clearing it just stops a freed state from
        // looking live.
        st.block_table.clear();
        st.seq_len = 0;
        Ok(())
    }

    /// 🔴 EP-sharded MoE (144 of 288 experts) + row-parallel DSA `o_proj`. Without the
    /// communicator this block drafts from HALF of both. See the trait doc for why that is
    /// only safe once the WORKER rank runs propose too.
    fn needs_comm(&self) -> bool {
        crate::speculative::mtp_ep_propose_enabled()
    }

    /// 🔴 The GLM context prefill runs the block through `ctx.buffers`. See the trait doc —
    /// running it from the end-of-prefill hook corrupts the TARGET's output.
    fn prefill_uses_shared_buffers(&self) -> bool {
        true
    }

    fn drafter_rows(&self, state: &mut dyn ProposerState) -> usize {
        state
            .as_any_mut()
            .downcast_mut::<Glm5NextMtpProposerState>()
            .map_or(0, |st| st.seq_len)
    }

    /// Dense row space: the newest row's slot IS its pair key.
    fn last_pair_key(&self, state: &mut dyn ProposerState) -> Option<usize> {
        state
            .as_any_mut()
            .downcast_mut::<Glm5NextMtpProposerState>()
            .and_then(|st| st.seq_len.checked_sub(1))
    }

    fn prefill_drafter(
        &self,
        prompt_tokens: &[u32],
        hiddens: DevicePtr,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let t0 = std::time::Instant::now();
        let rows = self.rows_impl(prompt_tokens, hiddens, 0, state, ctx, stream)?;
        // Every later propose calls this and `rows_impl` fast-returns 0; only the real one logs.
        if rows > 0 {
            tracing::info!(
                "GLM MTP drafter prefill: {rows} rows ({} prompt tokens) in {:.1} ms",
                prompt_tokens.len(),
                t0.elapsed().as_secs_f64() * 1e3,
            );
        }
        Ok(rows)
    }

    /// 🪤 `pos_base` is ignored: this drafter's RoPE position is its slot (see `rows_impl`), so
    /// the caller's sequence-space position is already `row_base` up to the uniform shift. A
    /// feed that does not start exactly at `drafter_rows()` is refused by `rows_impl`.
    fn catchup_drafter(
        &self,
        tokens: &[u32],
        hiddens: DevicePtr,
        row_base: usize,
        _pos_base: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        self.rows_impl(tokens, hiddens, row_base, state, ctx, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        _grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let st = state
            .as_any_mut()
            .downcast_mut::<Glm5NextMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("not a GLM MTP proposer state"))?;
        // The drafter's own sequence must sit where the target's does, or its indexer selects
        // over the wrong context. A gap means serial decode steps ran without a propose.
        if st.seq_len > position {
            st.dsa.rewind_to(position)?;
            st.seq_len = position;
        }
        if crate::speculative::mtp_refeed_debug() {
            let fp = crate::speculative::hidden_fingerprint(ctx.gpu, target_hidden, self.hidden);
            tracing::info!(
                "GLM_MTP_DBG propose position={position} drafter_rows={} tok={last_token} \
                 fp_target={fp:016x}",
                st.seq_len,
            );
        }
        let mut drafts = Vec::with_capacity(num_drafts);
        let mut token = last_token;
        let mut hidden = target_hidden;
        for i in 0..num_drafts {
            let d = self.forward_one(token, hidden, position + i, st, ctx, stream)?;
            drafts.push(d);
            token = d;
            // 🪤 Draft 1 consumes the TARGET's verified hidden; every later draft consumes the
            // drafter's OWN block output. That handoff is where acceptance falls off, and it is
            // inherent to running one module autoregressively.
            hidden = st.x;
        }
        st.last_drafted = drafts.len();
        Ok(drafts)
    }

    /// Widest batch one drafter forward can carry.
    ///
    /// `1` means the batched path cannot run and the caller stays on the per-sequence
    /// [`Self::propose`] — every reason it can return 1 is a genuine missing capability, never
    /// a judgement call the caller has to second-guess.
    ///
    /// `ATLAS_GLM_MTP_BATCH_PROPOSE=<width>` overrides: `1` (or `0`) restores the
    /// per-sequence loop, `N` caps the batch at N sequences. Numeric rather than boolean for
    /// the reason the DFlash head gives — bisecting the WIDTH against acceptance is what
    /// localises a banding bug, and an on/off flag cannot ask that question. It is also the
    /// lever for the one numerics difference this path has: the batched sharded head runs
    /// `fp8_gemv_rowscale_batch8_rt2`, not `n` calls of `dense_gemv_fp8w`.
    fn propose_batch_max(
        &self,
        _buffers: &spark_runtime::buffers::BufferArena,
        _config: &atlas_core::config::ModelConfig,
    ) -> usize {
        let Some(b) = self.batch.as_ref() else {
            return 1;
        };
        let want: usize = std::env::var("ATLAS_GLM_MTP_BATCH_PROPOSE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(usize::MAX);
        if want < 2 {
            return 1;
        }
        want.min(b.max_rows)
    }

    /// Cross-sequence batched propose: ONE drafter forward per draft depth over all `n`
    /// sequences, instead of `n` forwards.
    ///
    /// Site for site the same drafter as [`Self::propose`] — see [`Self::forward_n`] for the
    /// mapping and for what it saves. The autoregressive shape is unchanged: draft `d + 1`
    /// consumes draft `d`'s token and the drafter's own normed block output, per sequence,
    /// which is why the batch is over SEQUENCES and the depth loop stays serial.
    ///
    /// Returns `Ok(None)` to decline, and the caller falls back to the per-sequence loop —
    /// never a wrong answer.
    ///
    /// 🪤 The timing arms (`ATLAS_GLM_MTP_SKIP`) decline rather than being reimplemented here.
    /// They exist to attribute the SERIAL drafter's milliseconds; a batched arm would measure
    /// a different thing under the same name.
    #[allow(clippy::too_many_arguments)]
    fn propose_batch(
        &self,
        last_tokens: &[u32],
        target_hiddens: &[spark_runtime::gpu::DevicePtr],
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut dyn ProposerState],
        ctx: &crate::layer::ForwardContext,
        stream: u64,
        _out_conf: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        let n = last_tokens.len();
        let Some(b) = self.batch.as_ref() else {
            return Ok(None);
        };
        if n < 2
            || n > b.max_rows
            || target_hiddens.len() != n
            || positions.len() != n
            || states.len() != n
            || num_drafts == 0
            || skip_block()
            || skip_head()
        {
            return Ok(None);
        }
        // Width the caller is allowed to ask for, including the kill switch.
        if n > DraftProposer::propose_batch_max(self, ctx.buffers, ctx.config) {
            return Ok(None);
        }

        let mut sts: Vec<&mut Glm5NextMtpProposerState> = Vec::with_capacity(n);
        for state in states.iter_mut() {
            match state.as_any_mut().downcast_mut::<Glm5NextMtpProposerState>() {
                Some(st) => sts.push(st),
                // A foreign state in the batch means the caller mixed proposers; decline the
                // whole batch rather than draft for the ones that happen to match.
                None => return Ok(None),
            }
        }

        // The drafter's own sequence must sit where the target's does, or its indexer selects
        // over the wrong context — `propose`'s rewind, per sequence.
        for (i, st) in sts.iter_mut().enumerate() {
            if st.seq_len > positions[i] {
                st.dsa.rewind_to(positions[i])?;
                st.seq_len = positions[i];
            }
        }
        if crate::speculative::mtp_refeed_debug() {
            for (i, st) in sts.iter().enumerate() {
                let fp =
                    crate::speculative::hidden_fingerprint(ctx.gpu, target_hiddens[i], self.hidden);
                tracing::info!(
                    "GLM_MTP_DBG propose_batch[{i}/{n}] position={} drafter_rows={} tok={} \
                     fp_target={fp:016x}",
                    positions[i],
                    st.seq_len,
                    last_tokens[i],
                );
            }
        }

        let h = self.hidden;
        let mut drafts: Vec<Vec<u32>> = vec![Vec::with_capacity(num_drafts); n];
        let mut tokens: Vec<u32> = last_tokens.to_vec();
        let mut hiddens: Vec<spark_runtime::gpu::DevicePtr> = target_hiddens.to_vec();
        for _ in 0..num_drafts {
            let picked = self.forward_n(&tokens, &hiddens, &mut sts, ctx, stream)?;
            for (i, &t) in picked.iter().enumerate() {
                drafts[i].push(t);
            }
            tokens = picked;
            // 🪤 Draft 1 consumes the TARGET's verified hidden; every later draft consumes the
            // drafter's OWN block output — row `i` of the batched `x`, which is exactly the
            // per-sequence path's `st.x`. Same handoff, same acceptance falloff.
            hiddens = (0..n).map(|i| b.x.offset(i * h * 2)).collect();
        }
        for (i, st) in sts.iter_mut().enumerate() {
            st.last_drafted = drafts[i].len();
        }
        Ok(Some(drafts))
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let st = state
            .as_any_mut()
            .downcast_mut::<Glm5NextMtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("not a GLM MTP proposer state"))?;
        // Rejected rows are simply unreachable: the indexer reads `[0, len)` and the next
        // propose writes from `seq_len`, so rolling the counters back is the whole rollback.
        //
        // 🔴 ROW 0 IS ALWAYS VALID and must NOT be trimmed. It pairs the last COMMITTED token
        // with the target's own hidden — both facts at propose time — so a rejected DRAFT does
        // not make its row wrong, only its output unused. Only rows 1.. depend on a draft
        // having been accepted. Trimming row 0 (the Qwen head's `drafted - accepted` rule,
        // written for a COMPACTED row space) drops a real row from this DENSE one, and every
        // later RoPE position shifts. At `num_drafts = 1` that means: never trim.
        //
        // 🪤 At `num_drafts >= 2` a partial accept still trims, which DOES leave the dense row
        // space one key short of the sequence — the catch-up feed refills it from the ring, so
        // K>=3 must run with `ATLAS_MTP_CATCHUP=1`.
        let keep = st.last_drafted.min(num_accepted + 1);
        let trim = st.last_drafted - keep;
        if trim > 0 {
            st.seq_len = st.seq_len.saturating_sub(trim);
            st.dsa.rewind_to(st.seq_len)?;
        }
        Ok(())
    }
}

/// `ATLAS_GLM_MTP_SKIP=head`: stop the drafter after the block, before `shared_head.norm`,
/// the `lm_head` gemv, the argmax and the D2H. Timing arm only — see `forward_one`.
fn skip_head() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_MTP_SKIP").ok().as_deref() == Some("head"))
}

/// `ATLAS_GLM_MTP_SKIP=block`: skip `layers.45` itself and run only the head. Timing arm.
fn skip_block() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_MTP_SKIP").ok().as_deref() == Some("block"))
}

#[cfg(test)]
mod a59_sizing_tests {
    use super::drafter_context_rows;
    use crate::layers::glm5next_dsa::Glm5NextDsaConfig;

    /// GLM-5.3's shape. Mirrors `glm5next_dsa::state::tests::cfg`.
    fn cfg() -> Glm5NextDsaConfig {
        Glm5NextDsaConfig {
            hidden: 4096,
            index_heads: 32,
            index_head_dim: 128,
            index_kpool: 4,
            index_topk: 2048,
            always_select_tail: true,
            local_heads: 64,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_nope_head_dim: 256,
            qk_rope_head_dim: 0,
            v_head_dim: 256,
            max_context: 16_384,
        }
    }

    /// 🔴 ANOMALIES A59. A declared context the drafter can never reach must not size its
    /// buffers. At 524,288 the uncapped sizing cost 4.0 GiB of `mtp_prefill_hidden` plus a
    /// 0.5 GiB private KV pool, neither of them in `inference_reserve`.
    #[test]
    fn a_declared_context_past_the_dsa_reservation_does_not_size_the_drafter() {
        let c = cfg();
        assert_eq!(drafter_context_rows(524_288, &c), 16_384);
        assert_eq!(drafter_context_rows(262_144, &c), 16_384);
    }

    /// Below the ceiling nothing changes — the pre-A59 sizing is preserved exactly, which is
    /// what keeps every served context up to the cap byte-identical.
    #[test]
    fn a_context_under_the_ceiling_is_untouched() {
        let c = cfg();
        assert_eq!(drafter_context_rows(8_192, &c), 8_192);
        assert_eq!(drafter_context_rows(16_384, &c), 16_384);
    }

    /// The cap is DERIVED, not a literal: it is the DSA indexer cache's own reservation,
    /// so raising `--max-seq-len` raises the drafter's sizing in lockstep — and rounding to
    /// whole pools follows too. A hardcoded 16,384 passes the two tests above, fails this.
    #[test]
    fn the_cap_tracks_the_indexer_reservation_not_a_constant() {
        let mut c = cfg();
        c.max_context = 65_536;
        assert_eq!(drafter_context_rows(524_288, &c), 65_536);
        assert_eq!(drafter_context_rows(32_768, &c), 32_768);
        c.max_context = 65_538;
        assert_eq!(
            drafter_context_rows(524_288, &c),
            65_536,
            "whole pools only"
        );
    }
}
