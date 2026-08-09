// SPDX-License-Identifier: AGPL-3.0-only

//! GPU compute for the served NLLB model: kernel-launch primitives, the encoder
//! pass with cross-KV precompute, and the single-token decoder step. Promoted
//! from `examples/nllb_cuda_bf16/{ctx,decode}.rs`, reparented onto
//! [`NllbGpuModel`] (weights from the standard store, KV from the `kv` module).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::NllbGpuModel;
use super::kv::NllbSeqKv;
use super::util::{bf16_bytes, encoder_pos_bf16, u32_bytes};

/// Persistent single-token decode scratch (all bf16, sized `d` except `ff`/
/// `logits`). Reused every decode step; decode is driven sequentially so a
/// single shared instance is race-free at batch-1.
pub(super) struct DecScratch {
    dh: DevicePtr,
    residual: DevicePtr,
    normed: DevicePtr,
    q: DevicePtr,
    attn: DevicePtr,
    proj: DevicePtr,
    ff: DevicePtr,
    id_dev: DevicePtr,
    argmax_dev: DevicePtr,
}

impl DecScratch {
    pub(super) fn new(gpu: &dyn GpuBackend, d: usize, ffn: usize, _vocab: usize) -> Result<Self> {
        Ok(Self {
            dh: gpu.alloc(d * 2)?,
            residual: gpu.alloc(d * 2)?,
            normed: gpu.alloc(d * 2)?,
            q: gpu.alloc(d * 2)?,
            attn: gpu.alloc(d * 2)?,
            proj: gpu.alloc(d * 2)?,
            ff: gpu.alloc(ffn * 2)?,
            id_dev: gpu.alloc(4)?,
            argmax_dev: gpu.alloc(4)?,
        })
    }
}

/// Per-prefill encoder scratch (bf16, `enc_len` rows). Allocated once per
/// request (not per token) and freed when the encoder pass completes.
struct EncScratch {
    residual: DevicePtr,
    normed: DevicePtr,
    q: DevicePtr,
    kk: DevicePtr,
    v: DevicePtr,
    attn: DevicePtr,
    proj: DevicePtr,
    ff: DevicePtr,
}

impl EncScratch {
    fn new(gpu: &dyn GpuBackend, rows: usize, d: usize, ffn: usize) -> Result<Self> {
        Ok(Self {
            residual: gpu.alloc(rows * d * 2)?,
            normed: gpu.alloc(rows * d * 2)?,
            q: gpu.alloc(rows * d * 2)?,
            kk: gpu.alloc(rows * d * 2)?,
            v: gpu.alloc(rows * d * 2)?,
            attn: gpu.alloc(rows * d * 2)?,
            proj: gpu.alloc(rows * d * 2)?,
            ff: gpu.alloc(rows * ffn * 2)?,
        })
    }
    fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        for p in [
            self.residual,
            self.normed,
            self.q,
            self.kk,
            self.v,
            self.attn,
            self.proj,
            self.ff,
        ] {
            gpu.free(p)?;
        }
        Ok(())
    }
}

impl NllbGpuModel {
    #[inline]
    pub(super) fn stream(&self) -> u64 {
        self.gpu.default_stream()
    }

    // ── kernel-launch primitives (bf16 storage, f32 accumulation) ──

    pub(super) fn layer_norm(&self, prefix: &str, x: DevicePtr, rows: usize) -> Result<()> {
        KernelLaunch::new(self.gpu.as_ref(), self.kernels.ln)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .shared_mem(256 * 4)
            .arg_ptr(x)
            .arg_ptr(self.w(&format!("{prefix}.weight")))
            .arg_ptr(self.w(&format!("{prefix}.bias")))
            .arg_u32(rows as u32)
            .arg_u32(self.d as u32)
            .arg_f32(1e-5)
            .launch(self.stream())
    }

    /// Out-of-place layer norm: normalize `src` into `dst` (must be distinct
    /// buffers). Bit-identical arithmetic to `layer_norm`; used by the beam
    /// decode to fold the pre-LN `dh->normed` copy into the LN launch itself.
    pub(super) fn layer_norm_to(
        &self,
        prefix: &str,
        src: DevicePtr,
        dst: DevicePtr,
        rows: usize,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu.as_ref(), self.kernels.ln_oop)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .shared_mem(256 * 4)
            .arg_ptr(src)
            .arg_ptr(dst)
            .arg_ptr(self.w(&format!("{prefix}.weight")))
            .arg_ptr(self.w(&format!("{prefix}.bias")))
            .arg_u32(rows as u32)
            .arg_u32(self.d as u32)
            .arg_f32(1e-5)
            .launch(self.stream())
    }

    /// Tensor-core GEMM `C[m,n] = A[m,k] @ W[n,k]^T` (bf16, no bias).
    pub(super) fn gemm(
        &self,
        a: DevicePtr,
        wt: DevicePtr,
        c: DevicePtr,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu.as_ref(), self.kernels.gemm)
            .grid([div_ceil(n as u32, 128), div_ceil(m as u32, 128), 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(wt)
            .arg_ptr(c)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(self.stream())
    }

    /// Multi-row biased linear (GEMM + row-broadcast bias). `prefix.{weight,bias}`.
    pub(super) fn linear(
        &self,
        prefix: &str,
        a: DevicePtr,
        c: DevicePtr,
        rows: usize,
        n_out: usize,
        k_in: usize,
    ) -> Result<()> {
        self.gemm(a, self.w(&format!("{prefix}.weight")), c, rows, n_out, k_in)?;
        KernelLaunch::new(self.gpu.as_ref(), self.kernels.bias)
            .grid([div_ceil((rows * n_out) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(c)
            .arg_ptr(self.w(&format!("{prefix}.bias")))
            .arg_u32(rows as u32)
            .arg_u32(n_out as u32)
            .launch(self.stream())?;
        self.apply_lora(prefix, a, c, rows as u32)
    }

    /// Apply the LoRA residual for `prefix` onto `base_out` in place, when an
    /// adapter overrides that module. No-op for the base model.
    fn apply_lora(&self, prefix: &str, x: DevicePtr, base_out: DevicePtr, m: u32) -> Result<()> {
        if self.lora_is_active()
            && let Some(lora) = &self.lora
        {
            lora.apply(self.gpu.as_ref(), prefix, x, base_out, m, self.stream())?;
        }
        Ok(())
    }

    /// Single-token GEMV `y[n] = W[n,k] @ x[k] + bias` (bias may be NULL).
    fn gemv(
        &self,
        x: DevicePtr,
        wt: DevicePtr,
        bias: DevicePtr,
        y: DevicePtr,
        n: usize,
        k: usize,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu.as_ref(), self.kernels.gemv)
            .grid([div_ceil(n as u32, 8), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(wt)
            .arg_ptr(bias)
            .arg_ptr(y)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(self.stream())
    }

    /// M=1 biased linear via GEMV. `prefix.{weight,bias}`.
    fn linear1(
        &self,
        prefix: &str,
        x: DevicePtr,
        y: DevicePtr,
        n_out: usize,
        k_in: usize,
    ) -> Result<()> {
        self.gemv(
            x,
            self.w(&format!("{prefix}.weight")),
            self.w(&format!("{prefix}.bias")),
            y,
            n_out,
            k_in,
        )?;
        self.apply_lora(prefix, x, y, 1)
    }

    fn attention(
        &self,
        q: DevicePtr,
        kk: DevicePtr,
        v: DevicePtr,
        out: DevicePtr,
        tq: usize,
        tk: usize,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu.as_ref(), self.kernels.attn)
            .grid([(tq * self.heads) as u32, 1, 1])
            .block([self.head_dim as u32, 1, 1])
            .shared_mem(((tk + self.head_dim) * 4) as u32)
            .arg_ptr(q)
            .arg_ptr(kk)
            .arg_ptr(v)
            .arg_ptr(out)
            .arg_u32(tq as u32)
            .arg_u32(tk as u32)
            .arg_u32(self.heads as u32)
            .arg_u32(self.head_dim as u32)
            .arg_f32(self.attn_scale)
            .arg_u32(0) // non-causal: NLLB attention masks nothing (padding-free)
            .launch(self.stream())
    }

    pub(super) fn add(&self, dst: DevicePtr, src: DevicePtr, n: usize) -> Result<()> {
        KernelLaunch::new(self.gpu.as_ref(), self.kernels.add)
            .grid([div_ceil(n as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(dst)
            .arg_ptr(src)
            .arg_u32(n as u32)
            .launch(self.stream())
    }

    /// Embed `rows` token ids (device `u32[rows]`) into `out[rows,d]` bf16.
    pub(super) fn embed_rows(&self, ids_dev: DevicePtr, out: DevicePtr, rows: usize) -> Result<()> {
        KernelLaunch::new(self.gpu.as_ref(), self.kernels.embed)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(ids_dev)
            .arg_ptr(self.embed_table)
            .arg_ptr(out)
            .arg_u32(self.d as u32)
            .launch(self.stream())
    }

    pub(super) fn scale(&self, x: DevicePtr, n: usize) -> Result<()> {
        KernelLaunch::new(self.gpu.as_ref(), self.kernels.scale)
            .grid([div_ceil(n as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_u32(n as u32)
            .arg_f32(self.embed_scale)
            .launch(self.stream())
    }

    pub(super) fn relu(&self, x: DevicePtr, n: usize) -> Result<()> {
        KernelLaunch::new(self.gpu.as_ref(), self.kernels.relu)
            .grid([div_ceil(n as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_u32(n as u32)
            .launch(self.stream())
    }

    // ── encoder + cross-KV precompute ──

    /// Run the full encoder over the formatted source ids and precompute the
    /// per-decoder-layer cross-attention K/V into `kv`. Allocates + frees its
    /// own scratch and the encoder output (once per request).
    pub(super) fn run_encoder(&self, src_ids: &[u32], kv: &mut NllbSeqKv) -> Result<()> {
        let (d, gpu) = (self.d, self.gpu.as_ref());
        let seq = src_ids.len();
        kv.alloc_cross(gpu, self.dec_layers, seq, d)?;
        let enc_out = gpu.alloc(seq * d * 2)?;
        let s = EncScratch::new(gpu, seq, d, self.ffn)?;

        // embed + scale + encoder positions
        let ids_dev = gpu.alloc(seq * 4)?;
        gpu.copy_h2d(u32_bytes(src_ids), ids_dev)?;
        KernelLaunch::new(gpu, self.kernels.embed)
            .grid([seq as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(ids_dev)
            .arg_ptr(self.embed_table)
            .arg_ptr(enc_out)
            .arg_u32(d as u32)
            .launch(self.stream())?;
        self.scale(enc_out, seq * d)?;
        let pos = encoder_pos_bf16(src_ids, d, self.lang.pad_id);
        let pos_dev = gpu.alloc(seq * d * 2)?;
        gpu.copy_h2d(bf16_bytes(&pos), pos_dev)?;
        self.add(enc_out, pos_dev, seq * d)?;
        gpu.free(ids_dev)?;
        gpu.free(pos_dev)?;

        for l in 0..self.enc_layers {
            let p = format!("model.encoder.layers.{l}");
            self.enc_self_attn(&p, enc_out, seq, &s)?;
            self.enc_ffn(&p, enc_out, seq, &s)?;
        }
        self.layer_norm("model.encoder.layer_norm", enc_out, seq)?;

        for l in 0..self.dec_layers {
            let p = format!("model.decoder.layers.{l}.encoder_attn");
            self.linear(&format!("{p}.k_proj"), enc_out, kv.cross_k[l], seq, d, d)?;
            self.linear(&format!("{p}.v_proj"), enc_out, kv.cross_v[l], seq, d, d)?;
        }
        gpu.free(enc_out)?;
        s.free(gpu)?;
        Ok(())
    }

    fn enc_self_attn(&self, layer: &str, x: DevicePtr, seq: usize, s: &EncScratch) -> Result<()> {
        let (d, bytes) = (self.d, seq * self.d * 2);
        let p = format!("{layer}.self_attn");
        self.gpu.copy_d2d(x, s.residual, bytes)?;
        self.gpu.copy_d2d(x, s.normed, bytes)?;
        self.layer_norm(&format!("{layer}.self_attn_layer_norm"), s.normed, seq)?;
        self.linear(&format!("{p}.q_proj"), s.normed, s.q, seq, d, d)?;
        self.linear(&format!("{p}.k_proj"), s.normed, s.kk, seq, d, d)?;
        self.linear(&format!("{p}.v_proj"), s.normed, s.v, seq, d, d)?;
        self.attention(s.q, s.kk, s.v, s.attn, seq, seq)?;
        self.linear(&format!("{p}.out_proj"), s.attn, s.proj, seq, d, d)?;
        self.add(s.proj, s.residual, seq * d)?;
        self.gpu.copy_d2d(s.proj, x, bytes)
    }

    fn enc_ffn(&self, layer: &str, x: DevicePtr, rows: usize, s: &EncScratch) -> Result<()> {
        let (d, ffn, bytes) = (self.d, self.ffn, rows * self.d * 2);
        self.gpu.copy_d2d(x, s.residual, bytes)?;
        self.gpu.copy_d2d(x, s.normed, bytes)?;
        self.layer_norm(&format!("{layer}.final_layer_norm"), s.normed, rows)?;
        self.linear(&format!("{layer}.fc1"), s.normed, s.ff, rows, ffn, d)?;
        self.relu(s.ff, rows * ffn)?;
        self.linear(&format!("{layer}.fc2"), s.ff, s.proj, rows, d, ffn)?;
        self.add(s.proj, s.residual, rows * d)?;
        self.gpu.copy_d2d(s.proj, x, bytes)
    }

    // ── single-token decoder step ──

    /// Forward one decoder token at position `kv.dec_pos`: embed + position,
    /// self-attn (append K/V at that row), cross-attn over the encoder KV, FFN,
    /// tied lm_head → bf16 `logits_out` (`[vocab]`). Advances `kv.dec_pos`.
    pub(super) fn forward_one(
        &self,
        tok: u32,
        kv: &mut NllbSeqKv,
        logits_out: DevicePtr,
    ) -> Result<()> {
        kv.ensure_room()?;
        let (c, d) = (self.gpu.as_ref(), self.d);
        let s = &self.dec;
        let pos = kv.dec_pos;
        c.copy_h2d(u32_bytes(&[tok]), s.id_dev)?;
        KernelLaunch::new(c, self.kernels.embed)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.id_dev)
            .arg_ptr(self.embed_table)
            .arg_ptr(s.dh)
            .arg_u32(d as u32)
            .launch(self.stream())?;
        self.scale(s.dh, d)?;
        self.add(s.dh, self.pos_table.offset(pos * d * 2), d)?;

        let off = pos * d * 2;
        let tk = pos + 1;
        for l in 0..self.dec_layers {
            let p = format!("model.decoder.layers.{l}");
            // causal self-attention (write K/V into cache row `pos`)
            c.copy_d2d(s.dh, s.residual, d * 2)?;
            c.copy_d2d(s.dh, s.normed, d * 2)?;
            self.layer_norm(&format!("{p}.self_attn_layer_norm"), s.normed, 1)?;
            self.linear1(&format!("{p}.self_attn.q_proj"), s.normed, s.q, d, d)?;
            self.linear1(
                &format!("{p}.self_attn.k_proj"),
                s.normed,
                kv.self_k[l].offset(off),
                d,
                d,
            )?;
            self.linear1(
                &format!("{p}.self_attn.v_proj"),
                s.normed,
                kv.self_v[l].offset(off),
                d,
                d,
            )?;
            self.attention(s.q, kv.self_k[l], kv.self_v[l], s.attn, 1, tk)?;
            self.linear1(&format!("{p}.self_attn.out_proj"), s.attn, s.proj, d, d)?;
            self.add(s.proj, s.residual, d)?;
            c.copy_d2d(s.proj, s.dh, d * 2)?;
            // cross-attention over encoder K/V
            c.copy_d2d(s.dh, s.residual, d * 2)?;
            c.copy_d2d(s.dh, s.normed, d * 2)?;
            self.layer_norm(&format!("{p}.encoder_attn_layer_norm"), s.normed, 1)?;
            self.linear1(&format!("{p}.encoder_attn.q_proj"), s.normed, s.q, d, d)?;
            self.attention(s.q, kv.cross_k[l], kv.cross_v[l], s.attn, 1, kv.enc_len)?;
            self.linear1(&format!("{p}.encoder_attn.out_proj"), s.attn, s.proj, d, d)?;
            self.add(s.proj, s.residual, d)?;
            c.copy_d2d(s.proj, s.dh, d * 2)?;
            // FFN
            c.copy_d2d(s.dh, s.residual, d * 2)?;
            c.copy_d2d(s.dh, s.normed, d * 2)?;
            self.layer_norm(&format!("{p}.final_layer_norm"), s.normed, 1)?;
            self.linear1(&format!("{p}.fc1"), s.normed, s.ff, self.ffn, d)?;
            self.relu(s.ff, self.ffn)?;
            self.linear1(&format!("{p}.fc2"), s.ff, s.proj, d, self.ffn)?;
            self.add(s.proj, s.residual, d)?;
            c.copy_d2d(s.proj, s.dh, d * 2)?;
        }
        self.layer_norm("model.decoder.layer_norm", s.dh, 1)?;
        // tied lm_head (no bias) → bf16 logits row
        self.gemv(
            s.dh,
            self.embed_table,
            DevicePtr(0),
            logits_out,
            self.vocab,
            d,
        )?;
        kv.dec_pos += 1;
        Ok(())
    }

    /// `decode_logits` row `i` (batch position); `prefill_logits` row `slot`.
    #[inline]
    pub(super) fn decode_logits_row(&self, i: usize) -> DevicePtr {
        self.decode_logits.offset(i * self.vocab * 2)
    }
    #[inline]
    pub(super) fn prefill_logits_row(&self, slot: usize) -> DevicePtr {
        self.prefill_logits
            .offset((slot % self.max_batch) * self.vocab * 2)
    }

    /// On-device argmax over a bf16 `[vocab]` logits buffer → token id.
    pub(super) fn argmax_of(&self, logits_ptr: DevicePtr) -> Result<u32> {
        let c = self.gpu.as_ref();
        KernelLaunch::new(c, self.kernels.argmax)
            .grid([1, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(logits_ptr)
            .arg_ptr(self.dec.argmax_dev)
            .arg_u32(self.vocab as u32)
            .launch(self.stream())?;
        c.synchronize(self.stream())?;
        let mut idx = [0u8; 4];
        c.copy_d2h(self.dec.argmax_dev, &mut idx)?;
        Ok(u32::from_le_bytes(idx))
    }

    #[inline]
    pub(super) fn sync(&self) -> Result<()> {
        self.gpu.synchronize(self.stream())
    }
}
