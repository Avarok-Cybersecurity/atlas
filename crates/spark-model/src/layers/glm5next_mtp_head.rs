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
    /// 🔴 Ordering is load-bearing and already correct: `free_sequence`
    /// destroys this slot's `decode_graph` and `verify2/3/4_graph` — which bake
    /// these exact pointers — roughly 110 lines BEFORE it calls `free_state`.
    /// That teardown is the ANOMALIES A56 fix. Do not reorder either half.
    ///
    /// Kill switch: `ATLAS_GLM_MTP_STATE_LEAK` (presence — `=0` is NOT "off")
    /// restores the leaking no-op, for A/B measurement only.
    fn free_state(&self, gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        if std::env::var("ATLAS_GLM_MTP_STATE_LEAK").is_ok() {
            return Ok(());
        }
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
