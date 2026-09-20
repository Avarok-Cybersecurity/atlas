// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! DeepSeek-V4.1 Flash engram on the GPU.
//!
//! Engram sits before layers 1 and 14 (`engram_layer_ids`). Per token it hashes
//! the n-gram window of compressed token ids into `n_hash_cols` row ids of a
//! per-layer table, gathers those rows, projects their concatenation through
//! `wkv` into one key per hyper-connection stream plus one shared value, and
//! adds `gate * value` to each stream on the mHC highway. Four stages, each
//! held to the CPU reference [`crate::layers::deepseek_v41_ref::engram`]:
//!
//! 1. **Hash** ([`EngramHasher`]): integer-exact on the CPU; 48 ids per token
//!    is not a GPU problem. The tables (`token_map`, multipliers, primes,
//!    offsets) come from the GGUF metadata, proven identical to `engram.py`.
//! 2. **Rows**: the checkpoint's tables are Q2_K, one 84-byte block per
//!    256-wide row, read by id through
//!    `spark_runtime::weights::expert_stream::EngramRowReader` and dequantised
//!    on the device by the Q2_K arm of `dequant_gguf_bf16` straight into the
//!    `[tokens * cols, head_dim]` bf16 input ([`EngramV41::rows_from_q2k`]).
//! 3. **Projection**: the dense bf16 GEMM, `[tokens, cols * head_dim]` x
//!    `wkv^T` -> `[tokens, dim * (hc + 1)]`.
//! 4. **Gate** (`engram_v41_gate`, `kernels/gb10/deepseek-v4-flash/nvfp4/engram_v41.cu`):
//!    normalised dot of stream against key, signed-sqrt sigmoid, `h += gate * value`,
//!    in place on the FP32 highway `[T, hc, H]` the mHC kernels keep.
//!
//! Oracle: `engram_v41_tests.rs` (the hash exact against the reference; the
//! GPU path against the golden's `engram_out` captures; real-table rows
//! bit-identical to the CPU decoder).

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// Bytes of one Q2_K super-block = one engram row.
pub const ENGRAM_ROW_BYTES: usize = 84;
const DEQUANT_MODULE: &str = "dequant_gguf_bf16";
const GATE_MODULE: &str = "engram_v41";
const GEMM_MODULE: &str = "gemm";

/// The hash tables, shaped as the reference indexes them.
pub struct EngramHashTables {
    pub layer_ids: Vec<usize>,
    pub max_ngram: usize,
    pub n_heads: usize,
    /// `token_map[engram_pad_token_id]`: the pad in the compressed space.
    pub pad_id: i64,
    /// `[vocab]`, -1 = unmapped.
    pub token_map: Vec<i64>,
    /// `[layer][lookback]`
    pub multipliers: Vec<Vec<i64>>,
    /// `[layer][ngram_size - 2][head]`
    pub primes: Vec<Vec<Vec<i64>>>,
    /// `[layer][(ngram_size - 2) * n_heads + head]`
    pub offsets: Vec<Vec<i64>>,
}

impl EngramHashTables {
    /// From the flat GGUF metadata arrays (`deepseek41.engram.*`, as
    /// `ModelConfig` carries them): multipliers `[layers * max_ngram]`, primes
    /// `[layers * (max_ngram - 1) * n_heads]`, offsets the same length.
    pub fn from_flat(
        layer_ids: Vec<usize>,
        max_ngram: usize,
        n_heads: usize,
        pad_token_id: u32,
        token_map: Vec<i64>,
        multipliers: &[u64],
        primes: &[u64],
        offsets: &[u64],
    ) -> Result<Self> {
        let nl = layer_ids.len();
        ensure!(
            nl > 0 && max_ngram >= 2 && n_heads > 0,
            "engram geometry: {nl} layers, ngram {max_ngram}, heads {n_heads}"
        );
        let cols = (max_ngram - 1) * n_heads;
        ensure!(
            multipliers.len() == nl * max_ngram,
            "engram.multipliers: {} entries, expected {}",
            multipliers.len(),
            nl * max_ngram
        );
        ensure!(
            primes.len() == nl * cols,
            "engram.primes: {} entries, expected {}",
            primes.len(),
            nl * cols
        );
        ensure!(
            offsets.len() == nl * cols,
            "engram.offsets: {} entries, expected {}",
            offsets.len(),
            nl * cols
        );
        let pad_id = *token_map
            .get(pad_token_id as usize)
            .with_context(|| format!("engram pad token {pad_token_id} outside token_map"))?;
        let to_i64 = |v: &u64| i64::try_from(*v).context("engram table value exceeds i64");
        let multipliers = multipliers
            .chunks(max_ngram)
            .map(|c| c.iter().map(to_i64).collect())
            .collect::<Result<_>>()?;
        let primes = primes
            .chunks(cols)
            .map(|layer| {
                layer
                    .chunks(n_heads)
                    .map(|g| g.iter().map(to_i64).collect())
                    .collect()
            })
            .collect::<Result<_>>()?;
        let offsets = offsets
            .chunks(cols)
            .map(|c| c.iter().map(to_i64).collect())
            .collect::<Result<_>>()?;
        Ok(EngramHashTables {
            layer_ids,
            max_ngram,
            n_heads,
            pad_id,
            token_map,
            multipliers,
            primes,
            offsets,
        })
    }

    pub fn n_hash_cols(&self) -> usize {
        (self.max_ngram - 1) * self.n_heads
    }

    pub fn n_layers(&self) -> usize {
        self.layer_ids.len()
    }

    /// Position of model layer `layer` in `layer_ids`.
    pub fn hash_index(&self, layer: usize) -> Option<usize> {
        self.layer_ids.iter().position(|&l| l == layer)
    }
}

/// The n-gram hash for one sequence: keeps the compressed ids of every
/// position so decode steps see the prefill's look-back, exactly as the
/// reference's cache does.
pub struct EngramHasher {
    t: Arc<EngramHashTables>,
    cache: Vec<i64>,
}

impl EngramHasher {
    pub fn new(t: Arc<EngramHashTables>, max_seq: usize) -> Self {
        EngramHasher {
            t,
            cache: vec![0; max_seq],
        }
    }

    pub fn tables(&self) -> &Arc<EngramHashTables> {
        &self.t
    }

    pub fn reset(&mut self) {
        self.cache.iter_mut().for_each(|v| *v = 0);
    }

    /// Hash `input_ids` placed at `start_pos..`. Returns
    /// `[seqlen, n_layers, n_hash_cols]` row ids.
    pub fn hash(&mut self, input_ids: &[u32], start_pos: usize) -> Result<Vec<i64>> {
        let t = &self.t;
        let seqlen = input_ids.len();
        ensure!(
            start_pos + seqlen <= self.cache.len(),
            "engram hasher: {} positions, {} needed",
            self.cache.len(),
            start_pos + seqlen
        );
        for (s, &id) in input_ids.iter().enumerate() {
            let m = *t
                .token_map
                .get(id as usize)
                .with_context(|| format!("token {id} outside engram token_map"))?;
            self.cache[start_pos + s] = m;
        }
        let (ng, nl, cols) = (t.max_ngram, t.n_layers(), t.n_hash_cols());
        let mut out = vec![0i64; seqlen * nl * cols];
        let mut tokens = vec![0i64; ng];
        for s in 0..seqlen {
            let pos = start_pos + s;
            let mut blocked = false;
            for (shift, tok) in tokens.iter_mut().enumerate() {
                let src = self.cache[pos.saturating_sub(shift)];
                blocked = blocked || pos < shift || src == -1;
                *tok = if blocked { t.pad_id } else { src };
            }
            for (li, mults) in t.multipliers.iter().enumerate() {
                let mut rolling = tokens[0].wrapping_mul(mults[0]);
                for i in 1..ng {
                    rolling ^= tokens[i].wrapping_mul(mults[i]);
                    for h in 0..t.n_heads {
                        let col = (i - 1) * t.n_heads + h;
                        out[(s * nl + li) * cols + col] =
                            rolling.rem_euclid(t.primes[li][i - 1][h]) + t.offsets[li][col];
                    }
                }
            }
        }
        Ok(out)
    }

    /// The row ids one engram layer needs, in `[token][col]` order, from a
    /// `hash()` result of `tokens` positions.
    pub fn layer_row_ids(&self, hashes: &[i64], tokens: usize, hash_index: usize) -> Vec<u64> {
        let (nl, cols) = (self.t.n_layers(), self.t.n_hash_cols());
        (0..tokens)
            .flat_map(|s| (0..cols).map(move |c| hashes[(s * nl + hash_index) * cols + c] as u64))
            .collect()
    }
}

/// One engram layer's resident weights on the device.
pub struct EngramLayerWeights {
    pub layer: usize,
    /// `[dim * (hc + 1), cols * head_dim]` bf16.
    pub wkv: DevicePtr,
    /// `[hc, dim]` f32, the elementwise product `q_weight * k_weight`.
    pub qk: DevicePtr,
    /// This layer's row workspace (`add_layer` allocates): the raw Q2_K rows
    /// and their bf16 expansion, so every engram layer's rows for a step can
    /// be resident at once (the whole-step graph uploads them all up front).
    pub raw: DevicePtr,
    pub rows: DevicePtr,
}

/// The GPU side: kernels, per-layer weights, and workspaces for up to
/// `max_tokens` positions per call.
pub struct EngramV41 {
    pub dim: usize,
    pub hc: usize,
    pub head_dim: usize,
    pub cols: usize,
    pub eps: f32,
    pub max_tokens: usize,
    gemm_k: KernelHandle,
    /// the projection at one token: bandwidth-bound GEMV, not the 16x16 tile
    gemv_k: KernelHandle,
    gate_k: KernelHandle,
    dequant_k: KernelHandle,
    layers: Vec<EngramLayerWeights>,
    /// `[max_tokens * cols * head_dim]` bf16.
    rows: DevicePtr,
    /// `[max_tokens * cols * ENGRAM_ROW_BYTES]` raw Q2_K blocks.
    raw: DevicePtr,
    /// `[max_tokens * dim * (hc + 1)]` bf16.
    kv: DevicePtr,
}

impl EngramV41 {
    pub fn new(
        gpu: &dyn GpuBackend,
        dim: usize,
        hc: usize,
        head_dim: usize,
        cols: usize,
        eps: f32,
        max_tokens: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=4).contains(&hc),
            "engram gate supports hc_mult 1..=4, got {hc}"
        );
        let n_rows = max_tokens * cols;
        Ok(EngramV41 {
            dim,
            hc,
            head_dim,
            cols,
            eps,
            max_tokens,
            gemm_k: gpu.kernel(GEMM_MODULE, "dense_gemm_bf16")?,
            gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            gate_k: gpu.kernel(GATE_MODULE, "engram_v41_gate")?,
            dequant_k: gpu.kernel(DEQUANT_MODULE, "dequant_q2_k_to_bf16")?,
            layers: Vec::new(),
            rows: gpu.alloc(n_rows * head_dim * 2)?,
            raw: gpu.alloc(n_rows * ENGRAM_ROW_BYTES)?,
            kv: gpu.alloc(max_tokens * dim * (hc + 1) * 2)?,
        })
    }

    pub fn in_features(&self) -> usize {
        self.cols * self.head_dim
    }

    pub fn out_features(&self) -> usize {
        self.dim * (self.hc + 1)
    }

    /// Register a layer's weights; its row workspace is allocated here.
    pub fn add_layer(&mut self, gpu: &dyn GpuBackend, mut w: EngramLayerWeights) -> Result<()> {
        let n_rows = self.max_tokens * self.cols;
        w.raw = gpu.alloc(n_rows * ENGRAM_ROW_BYTES)?;
        w.rows = gpu.alloc(n_rows * self.head_dim * 2)?;
        self.layers.push(w);
        Ok(())
    }

    pub fn layer(&self, layer: usize) -> Option<&EngramLayerWeights> {
        self.layers.iter().find(|l| l.layer == layer)
    }

    /// `q_weight * k_weight` (both `[hc, dim]`, given as f32 values of the
    /// stored bf16) uploaded as the gate's `qk`.
    pub fn upload_qk(gpu: &dyn GpuBackend, q: &[f32], k: &[f32]) -> Result<DevicePtr> {
        ensure!(q.len() == k.len(), "engram q/k weights differ in length");
        let prod: Vec<u8> = q
            .iter()
            .zip(k)
            .flat_map(|(a, b)| (a * b).to_le_bytes())
            .collect();
        let p = gpu.alloc(prod.len())?;
        gpu.copy_h2d(&prod, p)?;
        Ok(p)
    }

    /// Fill the row input from raw Q2_K blocks (`n_rows * 84` bytes, one block
    /// per row in `[token][col]` order): upload, dequantise on the device.
    pub fn rows_from_q2k(
        &self,
        gpu: &dyn GpuBackend,
        layer: Option<usize>,
        raw_blocks: &[u8],
        n_rows: usize,
        stream: u64,
    ) -> Result<()> {
        let (raw, rows) = self.row_bufs(layer)?;
        ensure!(
            n_rows <= self.max_tokens * self.cols,
            "engram: {n_rows} rows exceeds the {} workspace",
            self.max_tokens * self.cols
        );
        ensure!(
            self.head_dim == 256,
            "engram rows are one Q2_K block: head_dim must be 256, got {}",
            self.head_dim
        );
        ensure!(
            raw_blocks.len() == n_rows * ENGRAM_ROW_BYTES,
            "engram: {} raw bytes for {n_rows} rows",
            raw_blocks.len()
        );
        gpu.copy_h2d_async(raw_blocks, raw, stream)?;
        KernelLaunch::new(gpu, self.dequant_k)
            .grid([n_rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(raw)
            .arg_ptr(rows)
            .arg_u32(n_rows as u32)
            .arg_u32(ENGRAM_ROW_BYTES as u32)
            .launch(stream)
    }

    /// Fill the row input from already-dequantised bf16 rows (bits), for
    /// oracles and for tables held resident in bf16.
    pub fn rows_from_bf16(
        &self,
        gpu: &dyn GpuBackend,
        layer: Option<usize>,
        rows_bf16: &[u16],
        n_rows: usize,
    ) -> Result<()> {
        ensure!(
            n_rows <= self.max_tokens * self.cols,
            "engram: {n_rows} rows exceeds the workspace"
        );
        ensure!(
            rows_bf16.len() == n_rows * self.head_dim,
            "engram: {} values for {n_rows} rows",
            rows_bf16.len()
        );
        let bytes: Vec<u8> = rows_bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
        gpu.copy_h2d(&bytes, self.row_bufs(layer)?.1)
    }

    /// The dequantised rows as the GEMM sees them (`[n_rows, head_dim]` bf16):
    /// the shared workspace (`layer` = `None`).
    pub fn rows_ptr(&self) -> DevicePtr {
        self.rows
    }

    /// `(raw, rows)` of layer `layer`'s own workspace, or the shared one.
    fn row_bufs(&self, layer: Option<usize>) -> Result<(DevicePtr, DevicePtr)> {
        match layer {
            None => Ok((self.raw, self.rows)),
            Some(l) => {
                let w = self
                    .layer(l)
                    .with_context(|| format!("engram: layer {l} has no weights"))?;
                Ok((w.raw, w.rows))
            }
        }
    }

    /// The rows `apply` reads for `w`: its own workspace (a null `rows` is a
    /// layer registered without one: the shared workspace).
    fn rows_of(&self, w: &EngramLayerWeights) -> DevicePtr {
        if w.rows.is_null() { self.rows } else { w.rows }
    }

    /// Project the rows already in the workspace and gate `tokens` positions
    /// of `streams` (`[T, hc, dim]` f32) in place, for model layer `layer`.
    pub fn apply(
        &self,
        gpu: &dyn GpuBackend,
        layer: usize,
        streams: DevicePtr,
        tokens: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            tokens <= self.max_tokens,
            "engram: {tokens} tokens exceeds the {} workspace",
            self.max_tokens
        );
        let w = self
            .layer(layer)
            .with_context(|| format!("engram: layer {layer} has no weights"))?;
        let wkv = DenseWeight { weight: w.wkv };
        if tokens == 1 {
            ops::dense_gemv(
                gpu,
                self.gemv_k,
                self.rows_of(w),
                &wkv,
                self.kv,
                self.out_features() as u32,
                self.in_features() as u32,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                gpu,
                self.gemm_k,
                self.rows_of(w),
                &wkv,
                self.kv,
                tokens as u32,
                self.out_features() as u32,
                self.in_features() as u32,
                stream,
            )?;
        }
        KernelLaunch::new(gpu, self.gate_k)
            .grid([tokens as u32, self.hc as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(streams)
            .arg_ptr(self.kv)
            .arg_ptr(w.qk)
            .arg_u32(self.dim as u32)
            .arg_u32(self.hc as u32)
            .arg_f32(self.eps)
            .launch(stream)
    }

    /// The projected `[tokens, dim * (hc + 1)]` bf16 buffer after `apply`.
    pub fn kv_ptr(&self) -> DevicePtr {
        self.kv
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        for l in &self.layers {
            gpu.free(l.wkv)?;
            gpu.free(l.qk)?;
            gpu.free(l.raw)?;
            gpu.free(l.rows)?;
        }
        gpu.free(self.rows)?;
        gpu.free(self.raw)?;
        gpu.free(self.kv)
    }
}

#[cfg(test)]
#[path = "engram_v41_tests.rs"]
mod tests;
