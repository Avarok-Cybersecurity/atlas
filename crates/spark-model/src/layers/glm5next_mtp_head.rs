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

        // 🪤 `shared_head.norm`, then the TARGET's own `lm_head`. The drafter ships no head of
        // its own — sharing it is what keeps a draft comparable to what the target would emit.
        self.norm(gpu, st.x, self.module.final_norm, st.x, h, stream)?;
        ops::dense_gemv(
            gpu,
            self.gemv_k,
            st.x,
            &self.lm_head,
            st.logits,
            self.vocab as u32,
            h as u32,
            stream,
        )?;
        ops::argmax_bf16(gpu, self.argmax_k, st.logits, st.arg, self.vocab as u32, stream)?;
        let mut out = [0u8; 4];
        gpu.synchronize(stream)?;
        gpu.copy_d2h(st.arg, &mut out)?;
        Ok(u32::from_le_bytes(out))
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
        let st = match state.as_any_mut().downcast_mut::<Glm5NextMtpProposerState>() {
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
            let embed_row = self.embed_tokens.weight.offset(tokens[r + 1] as usize * h * 2);
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
                let fp = crate::speculative::hidden_fingerprint(
                    gpu,
                    hiddens.offset(r * h * 2),
                    h,
                );
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
        }))
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
