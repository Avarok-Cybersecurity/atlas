// SPDX-License-Identifier: AGPL-3.0-only

//! The fused GPU embedding: base word row + the K*(N-1) hashed table
//! lookups, composed from existing kernels.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::ids::ngram_ids;
use super::{NgramDims, NgramTable};
use crate::weight_map::DenseWeight;

/// GPU n-gram embedding: base word embedding fused with the K*(N-1)
/// hashed-table lookups, composed entirely from existing kernels
/// (`batched_embed` gathers + `dense_gemm_bf16_pipelined` projections +
/// `bf16_scaled_add` accumulation). A fused single-kernel version is a
/// later optimization; per-token cost here is 1+T gathers, T tiny GEMMs
/// and T+1 scaled adds (T = 12 for LongCat-Lite) — negligible next to a
pub struct NgramEmbedding {
    pub dims: NgramDims,
    /// Base word embedding `[vocab, hidden]` BF16.
    pub word: DenseWeight,
    /// The K*(N-1) lookup tables, index order `(ngram-2)*K + split`,
    /// each `[table_rows(i), table_dim]` — BF16 or FP8-quantized.
    pub tables: Vec<NgramTable>,
    /// Per-table projections `[hidden, table_dim]` BF16 (nn.Linear layout).
    pub projs: Vec<DenseWeight>,

    pub(super) batched_embed_k: KernelHandle,
    pub(super) batched_embed_fp8_k: KernelHandle,
    pub(super) gemm_k: KernelHandle,
    pub(super) scaled_add_k: KernelHandle,

    /// Device staging: ids `[max_tokens]` u32 (reused per table),
    /// gathered rows `[max_tokens, table_dim]` BF16, projected rows
    /// `[max_tokens, hidden]` BF16.
    pub(super) ids_dev: DevicePtr,
    pub(super) gather_buf: DevicePtr,
    pub(super) proj_buf: DevicePtr,
    pub(super) max_tokens: usize,
}

impl NgramEmbedding {
    pub fn new(
        dims: NgramDims,
        word: DenseWeight,
        tables: Vec<NgramTable>,
        projs: Vec<DenseWeight>,
        max_tokens: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        anyhow::ensure!(tables.len() == dims.num_tables(), "table count");
        anyhow::ensure!(projs.len() == dims.num_tables(), "proj count");
        let td = dims.table_dim();
        Ok(Self {
            dims,
            word,
            tables,
            projs,
            batched_embed_k: gpu.kernel("embed_from_argmax", "batched_embed")?,
            batched_embed_fp8_k: gpu.kernel("embed_from_argmax", "batched_embed_fp8")?,
            gemm_k: gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?,
            scaled_add_k: gpu.kernel("residual_add", "bf16_scaled_add")?,
            ids_dev: gpu.alloc(max_tokens * 4)?,
            gather_buf: gpu.alloc(max_tokens * td * 2)?,
            proj_buf: gpu.alloc(max_tokens * dims.hidden_size * 2)?,
            max_tokens,
        })
    }

    /// Fused embedding for the LAST `seq_len` tokens of `ctx` (`ctx` =
    /// up to n-1 cached context tokens followed by the new tokens —
    /// exactly the reference NgramCache contract). Writes
    /// `[seq_len, hidden]` BF16 to `out`.
    pub fn embed(
        &mut self,
        ctx_tokens: &[u32],
        seq_len: usize,
        out: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::ops;
        anyhow::ensure!(
            seq_len <= self.max_tokens,
            "ngram embed: seq_len over staging"
        );
        anyhow::ensure!(seq_len <= ctx_tokens.len(), "ngram embed: seq_len over ctx");
        let h = self.dims.hidden_size;
        let td = self.dims.table_dim();
        let inv_scale = 1.0f32 / (1 + self.dims.num_tables()) as f32;

        // out = 0; each contribution lands via scaled_add(out += src/13),
        // matching the reference's final 1/(1+T) over base + all tables.
        gpu.memset(out, 0, seq_len * h * 2)?;

        // Base word rows for the NEW tokens only.
        let new_tokens = &ctx_tokens[ctx_tokens.len() - seq_len..];
        let ids_bytes: Vec<u8> = new_tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        gpu.copy_h2d_async(&ids_bytes, self.ids_dev, stream)?;
        ops::batched_embed(
            gpu,
            self.batched_embed_k,
            self.ids_dev,
            self.word.weight,
            self.proj_buf,
            seq_len as u32,
            h as u32,
            stream,
        )?;
        ops::scaled_add(
            gpu,
            self.scaled_add_k,
            out,
            self.proj_buf,
            inv_scale,
            (seq_len * h) as u32,
            stream,
        )?;

        // Host-side hash ids over the full ctx, one table at a time.
        let all_ids = ngram_ids(&self.dims, ctx_tokens);
        for (index, ids) in all_ids.iter().enumerate() {
            let tail = &ids[ids.len() - seq_len..];
            let id_bytes: Vec<u8> = tail
                .iter()
                .map(|&v| u32::try_from(v).context("ngram id exceeds u32"))
                .collect::<Result<Vec<u32>>>()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            gpu.copy_h2d_async(&id_bytes, self.ids_dev, stream)?;
            // NVMe-backed table: fault the rows in and REPLACE the ids with
            // their slot indices, then gather from the arena exactly as if it
            // were a small resident table.
            #[cfg(feature = "cuda")]
            if let NgramTable::Cached(cache) = &mut self.tables[index] {
                let mut slots: Vec<u32> = Vec::with_capacity(seq_len);
                cache.resolve(tail, &mut slots)?;
                let slot_bytes: Vec<u8> =
                    slots.iter().flat_map(|v: &u32| v.to_le_bytes()).collect();
                gpu.copy_h2d_async(&slot_bytes, self.ids_dev, stream)?;
                let table = DevicePtr(cache.table_dev_va()?);
                match cache.scale_dev_va()? {
                    Some(sc) => ops::batched_embed_fp8(
                        gpu,
                        self.batched_embed_fp8_k,
                        self.ids_dev,
                        table,
                        DevicePtr(sc),
                        self.gather_buf,
                        seq_len as u32,
                        td as u32,
                        stream,
                    )?,
                    None => ops::batched_embed(
                        gpu,
                        self.batched_embed_k,
                        self.ids_dev,
                        table,
                        self.gather_buf,
                        seq_len as u32,
                        td as u32,
                        stream,
                    )?,
                }
                cache.end_batch();
                ops::dense_gemm_bf16_pipelined(
                    gpu,
                    self.gemm_k,
                    self.gather_buf,
                    &self.projs[index],
                    self.proj_buf,
                    seq_len as u32,
                    h as u32,
                    td as u32,
                    stream,
                )?;
                ops::scaled_add(
                    gpu,
                    self.scaled_add_k,
                    out,
                    self.proj_buf,
                    inv_scale,
                    (seq_len * h) as u32,
                    stream,
                )?;
                continue;
            }
            match &self.tables[index] {
                NgramTable::Bf16(w) => ops::batched_embed(
                    gpu,
                    self.batched_embed_k,
                    self.ids_dev,
                    w.weight,
                    self.gather_buf,
                    seq_len as u32,
                    td as u32,
                    stream,
                )?,
                NgramTable::Fp8(w) => ops::batched_embed_fp8(
                    gpu,
                    self.batched_embed_fp8_k,
                    self.ids_dev,
                    w.weight,
                    w.row_scale,
                    self.gather_buf,
                    seq_len as u32,
                    td as u32,
                    stream,
                )?,
                #[cfg(feature = "cuda")]
                NgramTable::Cached(_) => unreachable!("resolved above"),
            }
            ops::dense_gemm_bf16_pipelined(
                gpu,
                self.gemm_k,
                self.gather_buf,
                &self.projs[index],
                self.proj_buf,
                seq_len as u32,
                h as u32,
                td as u32,
                stream,
            )?;
            ops::scaled_add(
                gpu,
                self.scaled_add_k,
                out,
                self.proj_buf,
                inv_scale,
                (seq_len * h) as u32,
                stream,
            )?;
        }
        Ok(())
    }
}
