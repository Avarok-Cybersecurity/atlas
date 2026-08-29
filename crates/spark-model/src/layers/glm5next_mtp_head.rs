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
//! 🔴 **Runs on every rank, and must agree bit for bit across them.** The routed MoE is
//! EP-sharded and the DSA `o_proj` is row-parallel, so the block all-reduces exactly as a text
//! layer does; both ranks then hold the same `x`, the same logits and the same argmax. If they
//! ever disagreed the two ranks would verify different drafts, which is not a quality
//! regression but a hang or garbage. (This is why it is NOT the rank-0-local drafter DeepSeek-V4
//! uses — V4's MTP module has all its experts local, GLM's does not.)
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
        let trim = st.last_drafted.saturating_sub(num_accepted);
        if trim > 0 {
            st.seq_len = st.seq_len.saturating_sub(trim);
            st.dsa.rewind_to(st.seq_len)?;
        }
        Ok(())
    }
}
