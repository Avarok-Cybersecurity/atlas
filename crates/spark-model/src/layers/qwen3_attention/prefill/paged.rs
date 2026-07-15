// SPDX-License-Identifier: AGPL-3.0-only

//! `prefill_attention_paged` — full N-token prefill (MLA + standard).

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::{BatchedAttnMetadata, ForwardContext};
use crate::layers::ops;

impl Qwen3AttentionLayer {
    pub(in crate::layers::qwen3_attention) fn prefill_attention_paged(
        &self,
        normed: DevicePtr,
        num_tokens: usize,
        seq_len_start: usize,
        kv_cache: &mut PagedKvCache,
        block_table: &Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        // Q12 Path B: when Some, attention compute uses the batched kernel
        // with `block_table_ptrs` from the supplied BatchedAttnMetadata;
        // positions / slot reads switch to the stacked variants from the
        // same struct. When None, the function behaves single-stream
        // (reading from ctx.attn_metadata as before).
        batched_meta: Option<&BatchedAttnMetadata>,
        // First `kv_write_floor` processed tokens skip the paged-cache K/V
        // write (Marconi warm-hit replay over already-cached positions —
        // see the section-7 comment). 0 on cold prefills.
        kv_write_floor: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let nq = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads) as u32;
        let nkv = self
            .num_kv_heads_override
            .unwrap_or(ctx.config.num_key_value_heads) as u32;
        let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let bs = kv_cache.block_size();
        let n = num_tokens as u32;
        let bf16 = 2usize;

        let q_dim = (nq * hd) as usize;
        let q_proj_dim = if self.gated { q_dim * 2 } else { q_dim };
        let kv_dim = (nkv * hd) as usize;

        // Q12 Path B: batched mode does not support MLA layers (separate
        // KV layout). Caller must gate this out at the outer dispatch.
        if batched_meta.is_some() && self.mla.is_some() {
            anyhow::bail!(
                "prefill_attention_paged: batched_meta with MLA layer is not supported \
                 (layer {}). Caller must route MLA layers to per-stream.",
                self.attn_layer_idx
            );
        }

        // ── MLA 2-step prefill (reference: HuggingFace modeling_mistral4.py) ──
        if let Some(ref mla) = self.mla {
            let args = super::paged_mla::MlaPrefillArgs {
                normed,
                num_tokens,
                n,
                h,
                nq,
                nkv,
                hd,
                kv_dim,
                eps,
                bf16,
                bs: bs as u32,
                stream,
            };
            // DeepSeek-V4-Flash: o_lora_rank > 0 selects the V4 prefill path
            // (wo_a→wo_b output LoRA, GQA FlashAttention). Non-V4 MLA models
            // (Mistral, DeepSeek-V3) keep o_lora_rank == 0 and fall through.
            if mla.o_lora_rank > 0 {
                return self.prefill_attention_paged_v4(kv_cache, ctx, &args, seq_len_start);
            }
            return self.prefill_attention_paged_mla(kv_cache, ctx, &args);
        }

        // ── 4+5. Deinterleave Q/Gate + per-head Q/K RMS norms ──
        // v_contiguous must point at where the V GEMM actually wrote
        // (k_contiguous + kv_dim*n). The previous binding to attn_output()
        // was a stale-buffer bug that corrupted V on chunk-1+ prefill for
        // every model that took this path (root cause of long-context
        // gibberish at 8 k+ contexts).
        let qg_out = ctx.buffers.qkv_output();
        let k_contiguous = ctx.buffers.ssm_qkvz();
        let v_contiguous = k_contiguous.offset(num_tokens * kv_dim * bf16);
        let q_contiguous = ctx.buffers.ssm_deinterleaved();
        // Defensive memset (audit fix #B2): clear the V region BEFORE the
        // V-projection GEMM runs. Historical comment on the aliasing math
        // notes a prior "stale V on chunk-1+" regression when the V GEMM
        // under-wrote the region. Zeroing first is ~5 µs at 9k tokens and
        // rules that class of bug out forever.
        //
        // Prefix-cache-recompute fix (iteration 4): this memset MUST precede
        // `prefill_attention_paged_qkv` — that function runs the V GEMM and
        // writes V to exactly this `v_contiguous` region (paged_qkv.rs).
        // When the memset ran AFTER the projection it clobbered the
        // freshly-projected V with zeros, so every chunk-1+/recompute block
        // got V≡0 written into the paged cache. Validated via the BF16 KV
        // checksum probe: recompute-suffix `v_ssq=0` vs cache-OFF `v_ssq≠0`
        // at the first full-attention layer (L3). On the cache-hit recompute
        // path this produced nondeterministic turn-3 output (zero-V positions
        // contribute nothing to attention; combined with the partial boundary
        // block the result diverged run-to-run).
        ctx.gpu
            .memset_async(v_contiguous, 0, num_tokens * kv_dim * bf16, stream)?;

        // ── Standard Q/K/V projection (non-MLA models) ──
        if self.mla.is_none() {
            self.prefill_attention_paged_qkv(
                normed, n, h, nkv, hd, q_proj_dim, kv_dim, num_tokens, bf16, ctx, stream,
            )?;
        } // end if self.mla.is_none() (standard projection path)
        // B1 fused K-path: save the post-GEMM RAW K to scratch BEFORE k_norm
        // and RoPE mutate `k_contiguous`. After the existing chain writes
        // (incorrectly-triple-rounded K + correct V) to the paged cache, we
        // re-do the K-path in a single fused kernel reading from this
        // scratch buffer and OVERWRITE the K side of the cache with
        // single-rounded values. ctx.buffers.attn_output() is free here
        // (it's written by FA later) and sized for `num_tokens * num_q_heads
        // * head_dim * 2`, plenty for our `num_tokens * num_kv_heads *
        // head_dim * 2` raw-K save. Gated on:
        //   - ATLAS_FUSED_KV=1  (opt-in during dev; expected default later)
        //   - mrope_interleaved kernel handle loaded
        //   - BF16 KV cache (FP8 path has its own quantization noise that
        //     masks the cliff; not the workload that needs this fix)
        let fused_kv_enabled = self.mrope_interleaved
            && self.fused_k_norm_rope_mrope_cache_write_bf16_k.0 != 0
            && self.reshape_and_cache_flash_v_only_k.0 != 0
            && std::env::var("ATLAS_FUSED_KV").ok().as_deref() == Some("1");
        let raw_k_scratch = if fused_kv_enabled {
            let scratch = ctx.buffers.attn_output();
            ctx.gpu
                .copy_d2d_async(k_contiguous, scratch, num_tokens * kv_dim * bf16, stream)?;
            Some(scratch)
        } else {
            None
        };
        if self.gated && !self.attn.q_norm.weight.is_null() {
            // Fused deinterleave + Q norm: eliminates Q global memory round-trip
            ops::deinterleave_qg_split_qnorm(
                ctx.gpu,
                self.deinterleave_qg_split_qnorm_k,
                qg_out,
                q_contiguous,
                self.attn.q_norm.weight,
                n,
                nq,
                hd,
                q_proj_dim as u32,
                eps,
                stream,
            )?;
        } else if self.gated {
            ops::deinterleave_qg_split(
                ctx.gpu,
                self.deinterleave_qg_split_k,
                qg_out,
                q_contiguous,
                n,
                nq,
                hd,
                q_proj_dim as u32,
                stream,
            )?;
        } else if let Some(mla_ref) = self.mla.as_ref() {
            // MLA: swap Q from [nope|rope] to [rope|nope] per head so RoPE rotates correct dims
            let mla_nope_sz = mla_ref.nope;
            let mla_rope_sz = mla_ref.rope;
            for t in 0..num_tokens {
                for head_idx in 0..nq as usize {
                    let src = qg_out.offset((t * q_dim + head_idx * hd as usize) * bf16);
                    let dst = q_contiguous.offset((t * q_dim + head_idx * hd as usize) * bf16);
                    ctx.gpu.copy_d2d_async(
                        src.offset(mla_nope_sz * bf16),
                        dst,
                        mla_rope_sz * bf16,
                        stream,
                    )?;
                    ctx.gpu.copy_d2d_async(
                        src,
                        dst.offset(mla_rope_sz * bf16),
                        mla_nope_sz * bf16,
                        stream,
                    )?;
                }
            }
        } else {
            ctx.gpu
                .copy_d2d_async(qg_out, q_contiguous, num_tokens * q_dim * bf16, stream)?;
            if let Some(ref q_norm_full) = self.attn.q_norm_full {
                // MiniMax: single RMS over full `[nq*hd]` per token
                // (rows=n, cols=nq*hd). Mistral/DeepSeek MLA models
                // never reach this branch — they early-return above.
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_w_k,
                    q_contiguous,
                    q_norm_full,
                    q_contiguous,
                    n,
                    nq * hd,
                    eps,
                    stream,
                )?;
            } else if !self.attn.q_norm.weight.is_null() {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_w_k,
                    q_contiguous,
                    &self.attn.q_norm,
                    q_contiguous,
                    nq * n,
                    hd,
                    eps,
                    stream,
                )?;
            }
        }
        if let Some(ref k_norm_full) = self.attn.k_norm_full {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                k_contiguous,
                k_norm_full,
                k_contiguous,
                n,
                nkv * hd,
                eps,
                stream,
            )?;
        } else if !self.attn.k_norm.weight.is_null() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                k_contiguous,
                &self.attn.k_norm,
                k_contiguous,
                nkv * n,
                hd,
                eps,
                stream,
            )?;
        }

        // Gemma-4 v_norm — applied at EVERY layer in HF reference
        // (modeling_gemma4.py:1220 `value_states = self.v_norm(value_states)`
        // with `Gemma4RMSNorm(with_scale=False)`). For full-attention K=V
        // layers, v_contiguous holds raw K (aliased v_proj). For sliding
        // layers, v_contiguous holds V_proj output. Either way normalize
        // with pure RMSNorm via the ones-buffer (Gemma-4's absolute-
        // formula rms_norm kernel: `x * rms * 1.0 = x * rms`). V does NOT
        // receive RoPE.
        if let Some(v_norm_w) = self.v_norm_weight.as_ref() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                v_contiguous,
                v_norm_w,
                v_contiguous,
                nkv * n,
                hd,
                eps,
                stream,
            )?;
        }

        // ── 6. RoPE for chunk tokens ──
        // Q12 Path B: in batched mode, ctx.attn_metadata is None — the
        // model dispatcher (prefill_batch_chunk_kernel_batched) sets
        // attn_metadata=None when calling the layer because per-stream
        // metadata is split across the batched_meta device arrays. So
        // we only require ctx.attn_metadata to be Some in single-stream
        // mode (batched_meta = None).
        let meta_for_single = match (batched_meta, ctx.attn_metadata) {
            (Some(_), _) => None,
            (None, Some(m)) => Some(m),
            (None, None) => anyhow::bail!(
                "prefill_attention_paged: single-stream mode requires ctx.attn_metadata"
            ),
        };

        // Resolve positions / slot pointers. When `batched_meta` is set,
        // RoPE / KV-write read from the stacked arrays. When in
        // single-stream mode, fall back to meta.{positions,slot}.
        let bmeta_positions = batched_meta
            .map(|m| m.positions_stacked)
            .or(meta_for_single.map(|m| m.positions))
            .unwrap();
        let bmeta_positions_h = batched_meta
            .map(|m| m.positions_h_stacked)
            .or(meta_for_single.map(|m| m.positions_h))
            .unwrap();
        let bmeta_positions_w = batched_meta
            .map(|m| m.positions_w_stacked)
            .or(meta_for_single.map(|m| m.positions_w))
            .unwrap();
        let bmeta_slot = batched_meta
            .map(|m| m.slot_stacked)
            .or(meta_for_single.map(|m| m.slot))
            .unwrap();
        if self.mla.is_some() {
            // MLA: RoPE already applied inside the MLA block to rope portions only.
        } else if let Some(ref mla) = self.mla {
            // unreachable but keeps the else chain valid
            if !mla.yarn_inv_freq.is_null() {
                ops::rope_yarn(
                    ctx.gpu,
                    self.rope_yarn_k,
                    q_contiguous,
                    k_contiguous,
                    bmeta_positions,
                    n,
                    nq,
                    nkv,
                    hd,
                    ctx.config.rotary_dim() as u32,
                    mla.yarn_inv_freq,
                    ctx.config.rope_theta as f32,
                    stream,
                )?;
            } else {
                ops::rope(
                    ctx.gpu,
                    self.rope_k,
                    q_contiguous,
                    k_contiguous,
                    bmeta_positions,
                    n,
                    nq,
                    nkv,
                    hd,
                    self.rotary_dim_override
                        .unwrap_or(ctx.config.rotary_dim() as u32),
                    self.rope_theta_override
                        .unwrap_or(ctx.config.rope_theta as f32),
                    stream,
                )?;
            }
        } else if self.rope_proportional && self.rope_proportional_k.0 != 0 {
            let rope_angles = self
                .rotary_dim_override
                .unwrap_or(ctx.config.rotary_dim() as u32);
            ops::rope_proportional(
                ctx.gpu,
                self.rope_proportional_k,
                q_contiguous,
                k_contiguous,
                bmeta_positions,
                n,
                nq,
                nkv,
                hd,
                rope_angles,
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )?;
        } else if self.mrope_interleaved && self.rope_mrope_interleaved_k.0 != 0 {
            ops::rope_mrope_interleaved(
                ctx.gpu,
                self.rope_mrope_interleaved_k,
                q_contiguous,
                k_contiguous,
                bmeta_positions,
                bmeta_positions_h,
                bmeta_positions_w,
                n,
                nq,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(ctx.config.rotary_dim() as u32),
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )?;
        } else {
            ops::rope(
                ctx.gpu,
                self.rope_k,
                q_contiguous,
                k_contiguous,
                bmeta_positions,
                n,
                nq,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(ctx.config.rotary_dim() as u32),
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )?;
        }

        // ── 7. Write all K/V to paged cache ──
        // MLA models write compressed cache inside the MLA block above (1 head × 320 dims).
        // Standard models write expanded cache here (nkv heads × hd dims).
        //
        // Warm-hit replay write floor: the first `kv_write_floor` processed
        // tokens are a Marconi SSM-replay over positions whose K/V already
        // sit in SHARED refcounted prefix-cache blocks (written by the pass
        // that originally produced them). The recompute is NOT bit-exact to
        // those originals (FLA-vs-WY4 chunk grids, batched-GEMM-vs-decode-
        // GEMV accumulation order), so rewriting would replace good cached
        // values with drifted ones and the drift ratchets across agentic
        // turns (2026-06-10 warm-hit token-stutter). Skip the write for the
        // floor region — section 8's paged attention reads the original
        // cached K/V at those positions, which is exactly what we want.
        let wf = kv_write_floor.min(num_tokens);
        if self.mla.is_none() && wf < num_tokens {
            self.write_kv_cache(
                ctx.gpu,
                k_contiguous.offset(wf * kv_dim * bf16),
                v_contiguous.offset(wf * kv_dim * bf16),
                kv_cache,
                // slot_mapping entries are int64 (8 bytes each)
                bmeta_slot.offset(wf * 8),
                n - wf as u32,
                nkv,
                hd,
                bs as u32,
                nkv * hd,
                nkv * hd,
                stream,
                ctx.graph_capture,
            )?;
            // B1 fused K-path: re-do the K side of the cache with the
            // single-rounded fused kernel (k_norm → RoPE → BF16 write all
            // in FP32 internally). Overwrites the triple-rounded K values
            // the chained path just wrote. V side is left as the chained
            // path wrote it (V passes through BF16 just once, no double-
            // rounding to fix).
            if let Some(raw_k) = raw_k_scratch
                && !self.attn.k_norm.weight.is_null()
            {
                use spark_runtime::kv_cache::KvCacheDtype;
                if kv_cache.dtype() == KvCacheDtype::Bf16 {
                    ops::fused_k_norm_rope_cache_write_bf16_mrope(
                        ctx.gpu,
                        self.fused_k_norm_rope_mrope_cache_write_bf16_k,
                        raw_k.offset(wf * kv_dim * bf16),
                        self.attn.k_norm.weight,
                        // positions are u32 (4 bytes), slots int64 (8 bytes)
                        bmeta_positions.offset(wf * 4),
                        bmeta_positions_h.offset(wf * 4),
                        bmeta_positions_w.offset(wf * 4),
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        bmeta_slot.offset(wf * 8),
                        n - wf as u32,
                        nkv,
                        hd,
                        self.rotary_dim_override
                            .unwrap_or(ctx.config.rotary_dim() as u32),
                        bs as u32,
                        ctx.config.rms_norm_eps as f32,
                        self.rope_theta_override
                            .unwrap_or(ctx.config.rope_theta as f32),
                        stream,
                    )?;
                }
            }
        }

        // ── 8. Paged Flash Attention for chunk 1+ ── (extracted to paged_attn.rs)
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);
        // In Q12 batched mode `num_tokens` is the stacked token count
        // (`batch_size * chunk_len`), while paged-attention kernels consume
        // per-stream q_len/kv_len. Single-stream mode keeps the historical
        // `seq_len_start + num_tokens` calculation.
        let kv_len = batched_meta
            .map(|m| seq_len_start + m.chunk_len as usize)
            .unwrap_or(seq_len_start + num_tokens) as u32;

        // TurboQuant WHT bookends (mirrors decode/attention_forward.rs).
        // write_kv_cache (section 7) WHT-rotated K/V in place before caching,
        // so both the contiguous buffers and the paged pools hold WHT(K)/
        // WHT(V). By Parseval, <WHT(Q), WHT(K)> = <Q, K>: rotate Q before
        // attention when the K side is turbo, and rotate the output back
        // (WHT is self-inverse) when the V side is turbo. Without these the
        // chunk≥2 history read scores raw Q against rotated K and leaves the
        // output in the rotated-V basis — the multi-chunk agentic collapse.
        let (wht_k_dtype, wht_v_dtype) = self.kv_dtype.kv_pair();
        let k_is_turbo = wht_k_dtype.is_wht_rotated();
        let v_is_turbo = wht_v_dtype.is_wht_rotated();
        let weight_pre_rotated = std::env::var("TQ_PLUS_WEIGHT_ROTATION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let wht_runtime_active = !weight_pre_rotated && (hd == 128 || hd == 256 || hd == 512);
        if k_is_turbo && wht_runtime_active && self.wht_bf16_k.0 != 0 {
            use spark_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(ctx.gpu, self.wht_bf16_k)
                .grid([n * nq, 1, 1]) // one warp per (token, q_head)
                .block([32, 1, 1])
                .arg_ptr(q_contiguous)
                .arg_u32(hd)
                .launch(stream)?;
        }
        if let Some(bmeta) = batched_meta {
            // Cross-request prefill via FlashInfer ragged/varlen attention
            // (ATLAS_FLASHINFER_PREFILL=1): the fresh contiguous, post-RoPE
            // Q/K/V already match FlashInfer's [rows, heads, 256] layout, so ONE
            // varlen launch runs all N co-dispatched requests' causal
            // self-attention — replacing the slow paged batched kernel (the
            // reason co-dispatch flatlined). Chunk-0 only (seq_len_start==0):
            // later chunks need the cached prefix, which isn't in the fresh
            // buffers. BF16 / no-WHT only; head_dim 256.
            let use_flashinfer = seq_len_start == 0
                && hd == 256
                && !k_is_turbo
                && !v_is_turbo
                && spark_runtime::flashinfer::available()
                && std::env::var("ATLAS_FLASHINFER_PREFILL").ok().as_deref() == Some("1");
            if use_flashinfer {
                // VARLEN: real per-request cu_seqlens from the staged metadata
                // (host + device copies) — works for both uniform and varied
                // request lengths. For chunk-0 self-attention qo_indptr==kv_indptr.
                let batch = bmeta.batch_size as usize;
                let total = bmeta.total_tokens;
                let indptr_h = &bmeta.cu_seqlens_host;
                let indptr_d = bmeta.cu_seqlens.0;
                {
                    use std::sync::atomic::{AtomicBool, Ordering};
                    static LOGGED: AtomicBool = AtomicBool::new(false);
                    if !LOGGED.swap(true, Ordering::Relaxed) {
                        tracing::warn!(
                            "FLASHINFER_PREFILL(varlen) batch={batch} total={total} \
                             num_tokens={num_tokens} cu_seqlens={indptr_h:?} \
                             nq={nq} nkv={nkv} hd={hd} sm_scale={inv_sqrt_d}"
                        );
                    }
                }
                spark_runtime::flashinfer::ragged_prefill_bf16_hd256(
                    q_contiguous.0,
                    k_contiguous.0,
                    v_contiguous.0,
                    attn_out.0,
                    indptr_h,
                    indptr_h,
                    indptr_d,
                    indptr_d,
                    batch as u32,
                    total,
                    total,
                    nq,
                    nkv,
                    hd,
                    inv_sqrt_d,
                    true,
                    stream,
                )?;
            } else {
                // Q12 Path B: batched paged-prefill attention. The kernel reads
                // Q/O at per-batch offsets internally via blockIdx.z and uses
                // block_table_ptrs[b] for each stream's paged KV pages.
                let args = super::paged_attn_batched::PagedAttnBatchedArgs {
                    q_contiguous,
                    attn_out,
                    seq_len_start,
                    nq,
                    nkv,
                    hd,
                    bs,
                    inv_sqrt_d,
                    kv_len,
                    batched_meta: bmeta,
                    stream,
                };
                self.prefill_attention_paged_attn_batched(kv_cache, ctx, &args)?;
            }
        } else {
            // Single-stream path requires meta_for_single (validated above).
            let meta = meta_for_single
                .expect("single-stream mode: meta_for_single guaranteed by validation above");
            let mut args = super::paged_attn::PagedAttnArgs {
                q_contiguous,
                k_contiguous,
                v_contiguous,
                attn_out,
                n,
                seq_len_start,
                num_tokens,
                nq,
                nkv,
                hd,
                bs,
                bf16,
                inv_sqrt_d,
                kv_len,
                meta: &meta,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                stream,
            };
            match self.prefill_attention_paged_attn(kv_cache, ctx, &mut args)? {
                super::paged_attn::PagedAttnOutcome::EarlyReturn(out) => return Ok(out),
                super::paged_attn::PagedAttnOutcome::Continue => {}
            }
        }

        // TurboQuant WHT bookend (output side): attention output is
        // sum(softmax * WHT(V)) — rotate back to the real basis.
        if v_is_turbo && wht_runtime_active && self.wht_bf16_k_inv.0 != 0 {
            use spark_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(ctx.gpu, self.wht_bf16_k_inv)
                .grid([n * nq, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(attn_out)
                .arg_u32(hd)
                .launch(stream)?;
        }

        // ATLAS_OP_DUMP: attn_out BEFORE sigmoid gate (raw attention-kernel output).
        // Compares 1:1 against vLLM's "attn_out" dump in qwen3_next.py:_dump_op.
        // Use last-token slice n_elements = num_heads * head_dim.
        if num_tokens > 0 {
            let nq_hd = (nq * hd) as usize;
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                attn_out,
                (num_tokens - 1) * nq_hd * bf16,
                nq_hd,
                self.attn_layer_idx,
                "attn_out_pre_gate",
                stream,
            )?;
        }

        // ── 9. Sigmoid gate × attn_out (gated only) — single batched kernel ──
        if self.gated {
            // Gate data is in qg_out at offset q_dim (after deinterleave_qg_split),
            // with stride q_proj_dim between tokens.
            let gate_base = qg_out.offset(q_dim * bf16);
            ops::sigmoid_gate_mul_batched(
                ctx.gpu,
                self.sigmoid_gate_mul_batched_k,
                attn_out,
                gate_base,
                attn_out,
                nq * hd,
                q_proj_dim as u32,
                n,
                stream,
            )?;
        }

        // ── 9b. Per-head attention gate (Step 3.7 g_proj) ──
        if let Some(ref g_proj) = self.head_gate_weight {
            let gate_buf = q_contiguous; // Q buffer free after attention
            ops::dense_gemm_tc(
                ctx.gpu,
                self.dense_gemm_tc_k,
                normed,
                g_proj,
                gate_buf,
                n,
                nq,
                h,
                stream,
            )?;
            ops::sigmoid_gate_mul_head_broadcast(
                ctx.gpu,
                self.sigmoid_gate_head_broadcast_k,
                attn_out,
                gate_buf,
                attn_out,
                nq,
                hd,
                n,
                stream,
            )?;
        }

        // ATLAS_OP_DUMP: attn_out AFTER sigmoid gate (input to o_proj linear).
        if num_tokens > 0 {
            let nq_hd = (nq * hd) as usize;
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                attn_out,
                (num_tokens - 1) * nq_hd * bf16,
                nq_hd,
                self.attn_layer_idx,
                "attn_out_post_gate",
                stream,
            )?;
        }

        // ── 10. O projection GEMM ── (extracted to paged_oproj.rs)
        let o_out = self.prefill_attention_paged_oproj(attn_out, n, h, nq, hd, ctx, stream)?;

        Ok(o_out)
    }
}
