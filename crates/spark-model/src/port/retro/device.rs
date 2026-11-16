// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm: RetroInfer, arXiv 2505.02922 (VLDB 2026).

//! Responsibility: the retro device state of one backend: the arena
//! allocation, the per-step buffers (zone lists, the exact buffer, the build
//! block table), index regions, and the host-driven device operations around
//! the L34 kernels: running a build or an update through
//! `build::build_layer_index`, writing an index header, copying an index into
//! a larger or private region (copy-on-write), and reading a content
//! fingerprint. Every operation is stream-ordered on the layer's stream and
//! runs only on eager steps (never while a graph is captured).

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::arena::{Arena, Region};
use super::config::{self, MAX_VERIFY_ROWS, RetroCfg};
use super::lifecycle::Fingerprint;
use super::state::{BF16, ExactBuf, IndexLayout, LayerIndex, ZoneLayout, ZoneLists, align};
use super::update::BuildStep;
use crate::layers::qwen3_attention::sparse_seam::{KvLayerView, SeqBatch};

/// Tokens a private index is sized beyond the current context.
pub const GROWTH_TOKENS: u32 = 32768;

/// Bytes of K a fingerprint reads.
pub const FINGERPRINT_BYTES: usize = 64;

/// The address of a backend, which keys every device state.
pub fn backend_key(gpu: &dyn GpuBackend) -> usize {
    std::ptr::from_ref(gpu).cast::<()>() as usize
}

/// The per-step buffers.
pub struct StepBufs {
    pub region: Region,
    /// `MAX_VERIFY_ROWS` per-row lists, then the shared (union) list.
    pub zones: Vec<ZoneLists>,
    pub exact: ExactBuf,
    /// `[max_blocks]` i32 block table the builds read.
    pub table: DevicePtr,
    /// `[1]` i32 length for the build and gather `SeqBatch`es.
    pub seq_len: DevicePtr,
    pub max_blocks: u32,
}

/// What a build needed that the arena could not give.
#[derive(Debug, PartialEq, Eq)]
pub struct NoRoom(pub usize);

/// One backend's retro device state.
pub struct Device {
    owner: usize,
    pub base: DevicePtr,
    pub arena: Arena,
    pub cfg: RetroCfg,
    /// The serving model instance (backend and geometry), part of cache keys.
    pub model: u64,
    pub bufs: Option<StepBufs>,
}

impl Device {
    /// Allocate the arena on `gpu` (`--retro-index-cache-mb`).
    pub fn new(gpu: &dyn GpuBackend, cfg: RetroCfg) -> Result<Self> {
        let bytes = (cfg.params.index_cache_mb as usize) << 20;
        let base = gpu.alloc(bytes).with_context(|| {
            format!(
                "retro: allocating the {} MiB arena",
                cfg.params.index_cache_mb
            )
        })?;
        let owner = backend_key(gpu);
        let mut h = config::Fnv::new();
        for v in [
            owner as u64,
            u64::from(cfg.kv_heads),
            u64::from(cfg.head_dim),
            u64::from(cfg.block_size),
            u64::from(cfg.kv_dtype.bytes_per_elem() as u32),
        ] {
            h.write(&v.to_le_bytes());
        }
        Ok(Self {
            owner,
            base,
            arena: Arena::new(bytes),
            model: h.finish(),
            cfg,
            bufs: None,
        })
    }

    /// Whether this state belongs to `gpu`.
    pub fn owned_by(&self, gpu: &dyn GpuBackend) -> bool {
        self.owner == backend_key(gpu)
    }

    pub fn ptr(&self, r: Region) -> DevicePtr {
        self.base.offset(r.offset)
    }

    /// Allocate the per-step buffers on first use.
    pub fn ensure_bufs(&mut self, max_blocks: u32) -> Result<(), NoRoom> {
        if self
            .bufs
            .as_ref()
            .is_some_and(|b| b.max_blocks >= max_blocks)
        {
            return Ok(());
        }
        if let Some(old) = self.bufs.take() {
            self.arena.free(old.region);
        }
        let c = &self.cfg;
        let zl = ZoneLayout {
            kv_heads: c.kv_heads,
            retrieval_cap: c.retrieval_cap,
            es_cap: c.es_cap,
        };
        let lists = (MAX_VERIFY_ROWS as usize + 1) * zl.bytes();
        let scores = align(c.kv_heads as usize * c.cap_centroids as usize * 4);
        let scratch = align(c.select_scratch_bytes());
        let exact = align(ExactBuf::bytes(c.kv_heads, c.head_dim, c.exact_cap));
        let table = align(max_blocks as usize * 4);
        let total = lists + scores + scratch + exact + table + align(16);
        let region = self.arena.alloc(total).ok_or(NoRoom(total))?;
        let base = self.ptr(region);
        let scores_p = base.offset(lists);
        let scratch_p = scores_p.offset(scores);
        let zones = (0..=MAX_VERIFY_ROWS as usize)
            .map(|i| {
                ZoneLists::at(
                    base.offset(i * zl.bytes()),
                    zl,
                    scores_p,
                    scratch_p,
                    c.select_scratch_bytes(),
                )
            })
            .collect();
        let exact_p = scratch_p.offset(scratch);
        let table_p = exact_p.offset(exact);
        let mut exact_buf = ExactBuf::at(exact_p, c.kv_heads, c.head_dim, c.exact_cap);
        exact_buf.num_q_heads = c.num_q_heads;
        exact_buf.inv_sqrt_d = c.inv_sqrt_d;
        self.bufs = Some(StepBufs {
            region,
            zones,
            exact: exact_buf,
            table: table_p,
            seq_len: table_p.offset(table),
            max_blocks,
        });
        Ok(())
    }

    /// The layout of an index sized for `context` tokens.
    pub fn layout_for(&self, context: u32) -> IndexLayout {
        let ctx = context.min(self.cfg.max_context).max(1);
        IndexLayout {
            kv_heads: self.cfg.kv_heads,
            head_dim: self.cfg.head_dim,
            cap_centroids: config::centroid_capacity(&self.cfg.params, ctx),
            cap_tokens: ctx,
        }
    }

    /// The layout of a private index for a sequence now at `len` tokens.
    pub fn private_layout(&self, len: u32) -> IndexLayout {
        self.layout_for(len.saturating_add(GROWTH_TOKENS.max(len / 4)))
    }

    /// An index region of `layout`, or `None` when the arena has no room.
    pub fn alloc_index(&mut self, layout: IndexLayout) -> Option<(LayerIndex, Region)> {
        let r = self.arena.alloc(layout.total_bytes())?;
        let mut index = LayerIndex::at(self.ptr(r), layout);
        index.num_q_heads = self.cfg.num_q_heads;
        Some((index, r))
    }

    pub fn free(&mut self, r: Region) {
        self.arena.free(r);
    }

    /// Upload a block table and a length for a one-row `SeqBatch`.
    pub fn upload_row(
        &self,
        gpu: &dyn GpuBackend,
        table: &[u32],
        len: u32,
        stream: u64,
    ) -> Result<()> {
        let b = self
            .bufs
            .as_ref()
            .context("retro: step buffers not allocated")?;
        anyhow::ensure!(
            table.len() <= b.max_blocks as usize,
            "retro: a {}-block table exceeds the {}-block buffer",
            table.len(),
            b.max_blocks
        );
        let bytes: Vec<u8> = table
            .iter()
            .flat_map(|&v| (v as i32).to_le_bytes())
            .collect();
        // Transient sources: `copy_h2d_async` (never the `_retained` form,
        // whose source must outlive the next synchronization).
        gpu.copy_h2d_async(&bytes, b.table, stream)?;
        gpu.copy_h2d_async(&(len as i32).to_le_bytes(), b.seq_len, stream)
    }

    /// Write `index`'s host counts to its device header.
    pub fn write_header(
        &self,
        gpu: &dyn GpuBackend,
        index: &LayerIndex,
        stream: u64,
    ) -> Result<()> {
        gpu.copy_h2d_async(&index.counts.to_bytes(), index.header, stream)
    }

    /// Run one build (initial, update or append) into `index` over the
    /// sequence with host `table` and `len` tokens; the counts and header
    /// follow `step`. Synchronizes the stream so the build time is measured.
    #[allow(clippy::too_many_arguments)]
    pub fn run_build(
        &mut self,
        gpu: &dyn GpuBackend,
        kv: &KvLayerView,
        table: &[u32],
        len: u32,
        index: &mut LayerIndex,
        step: &BuildStep,
        stream: u64,
    ) -> Result<std::result::Result<f64, NoRoom>> {
        if !super::update::fits(
            &step.after,
            index.layout.cap_centroids,
            index.layout.cap_tokens,
        ) {
            return Ok(Err(NoRoom(0)));
        }
        let need = step.scratch_bytes(&self.cfg);
        let Some(scratch) = self.arena.alloc(need) else {
            return Ok(Err(NoRoom(need)));
        };
        let t0 = std::time::Instant::now();
        let res = self.build_with(
            gpu,
            kv,
            table,
            len,
            index,
            step,
            self.ptr(scratch),
            scratch.bytes,
            stream,
        );
        self.arena.free(scratch);
        res?;
        Ok(Ok(t0.elapsed().as_secs_f64()))
    }

    #[allow(clippy::too_many_arguments)]
    fn build_with(
        &self,
        gpu: &dyn GpuBackend,
        kv: &KvLayerView,
        table: &[u32],
        len: u32,
        index: &mut LayerIndex,
        step: &BuildStep,
        scratch: DevicePtr,
        scratch_bytes: usize,
        stream: u64,
    ) -> Result<()> {
        self.upload_row(gpu, table, len, stream)?;
        let b = self
            .bufs
            .as_ref()
            .context("retro: step buffers not allocated")?;
        index.pending = Some(step.pending(&index.counts, scratch, scratch_bytes));
        let lens = [len];
        let first = [step.first_pos];
        let new = [step.tokens];
        let tables = [table];
        let s = SeqBatch {
            n: 1,
            block_table: b.table,
            seq_lens: b.seq_len,
            max_blocks_per_seq: b.max_blocks,
            host_seq_lens: &lens,
            seq_ids: &[],
            rows_per_seq: 1,
            q_row_stride: 0,
            q_rows: 0,
            first_pos: &first,
            new_tokens: &new,
            host_block_tables: &tables,
            block_table_ptrs: DevicePtr::NULL,
        };
        let res = super::build::build_layer_index(gpu, kv, &s, &self.cfg, index, stream);
        index.pending = None;
        res?;
        index.counts = step.after;
        self.write_header(gpu, index, stream)?;
        gpu.synchronize(stream)
    }

    /// Copy `src`'s used entries into `dst` (equal or larger capacities) and
    /// give `dst` the same counts.
    pub fn copy_index(
        &self,
        gpu: &dyn GpuBackend,
        src: &LayerIndex,
        dst: &mut LayerIndex,
        stream: u64,
    ) -> Result<()> {
        let (sc, so, st) = src.layout.head_strides();
        let (dc, d_o, dt) = dst.layout.head_strides();
        let c = src.counts;
        anyhow::ensure!(
            dst.layout.cap_centroids >= c.n_centroids && dst.layout.cap_tokens >= c.listed,
            "retro: index copy into a smaller region ({} centroids, {} tokens)",
            dst.layout.cap_centroids,
            dst.layout.cap_tokens
        );
        let hd = src.layout.head_dim as usize;
        let nc = c.n_centroids as usize;
        for h in 0..src.layout.kv_heads as usize {
            let row = hd * BF16;
            gpu.copy_d2d_async(
                src.centroids.offset(h * sc * row),
                dst.centroids.offset(h * dc * row),
                nc * row,
                stream,
            )?;
            gpu.copy_d2d_async(
                src.value_sum.offset(h * sc * row),
                dst.value_sum.offset(h * dc * row),
                nc * row,
                stream,
            )?;
            gpu.copy_d2d_async(
                src.cluster_size.offset(h * sc * 4),
                dst.cluster_size.offset(h * dc * 4),
                nc * 4,
                stream,
            )?;
            gpu.copy_d2d_async(
                src.list_offsets.offset(h * so * 4),
                dst.list_offsets.offset(h * d_o * 4),
                (nc + 1) * 4,
                stream,
            )?;
            gpu.copy_d2d_async(
                src.list_tokens.offset(h * st * 4),
                dst.list_tokens.offset(h * dt * 4),
                c.listed as usize * 4,
                stream,
            )?;
        }
        dst.counts = c;
        self.write_header(gpu, dst, stream)
    }

    /// The first [`FINGERPRINT_BYTES`] of head 0's K at `pos` (stream-ordered read).
    pub fn read_fingerprint(
        &self,
        gpu: &dyn GpuBackend,
        kv: &KvLayerView,
        table: &[u32],
        pos: u32,
        stream: u64,
    ) -> Result<Fingerprint> {
        let bs = kv.block_size.max(1);
        let blk = *table
            .get((pos / bs) as usize)
            .context("retro: fingerprint position beyond the block table")?;
        let elem = kv.dtype.bytes_per_elem();
        let row = kv.kv_heads as usize * kv.head_dim as usize * elem;
        let addr = kv
            .k_pool
            .offset(blk as usize * kv.block_stride_bytes as usize + (pos % bs) as usize * row);
        let mut bytes = vec![0u8; FINGERPRINT_BYTES.min(kv.head_dim as usize * elem)];
        gpu.copy_d2h_on_stream(addr, &mut bytes, stream)?;
        Ok(Fingerprint { pos, bytes })
    }

    /// Bytes in use, for `atlas_retro_index_bytes`.
    pub fn used(&self) -> usize {
        self.arena.used()
    }
}
