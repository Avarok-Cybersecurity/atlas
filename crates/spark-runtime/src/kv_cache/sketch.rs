// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).
// Block key-mean statistic per arXiv:2609.04852; independent implementation.

//! Responsibility: the per-page pre-RoPE key sketch of the bounded KV view (port lane L31):
//! the device buffers ([`SketchBuffers`]), the host job planning ([`plan_jobs`]) and the two
//! launch paths of `kernels/ada/common/kv_page_sketch_ada.cu`.
//!
//! # What is stored
//! For every physical page (KV block) and attention layer and KV head, the mean over the
//! page's stored tokens of the key AFTER k-norm and BEFORE RoPE, in BF16, plus one u8 token
//! count per page: `means [num_blocks, attn_layers, kv_heads, head_dim]`, `counts
//! [num_blocks]`. Per page, not per sequence, so a prefix-cache page shared by several
//! sequences carries one sketch that serves all of them (a shared page holds the same
//! tokens at the same positions for every sharer). 16 KiB per 16-token page on the 4B
//! (8 layers x 4 heads x 256 x 2 B): 6.25% of FP8 KV, 3.1% of BF16 KV.
//!
//! # How it is computed
//! The fused K writes are not touched. After a K write the sketch kernel re-reads the
//! page's stored keys (BF16, or FP8 E4M3 times the layer's K scale), undoes RoPE with the
//! write's own convention (NeoX half pairing over the first `rotary_dim` dims, frequency
//! `theta^(-2i/rotary_dim)` evaluated in f64 like `fused_k_norm_rope_cache.cu`, position =
//! logical page x block_size + offset, text positions only) and RECOMPUTES the page mean
//! from scratch over the page's `fill` tokens, in FP32, storing BF16. Recomputing instead
//! of accumulating makes every update idempotent: a verify step's rejected draft rows that
//! the next step rewrites, a page reused by a new owner, or a replayed chunk can never be
//! counted twice. A prefill chunk of 4,096 tokens re-reads 8 MiB per layer of BF16 K on the
//! 4B, which stays in the 32 MB L2 right after the write.
//!
//! * [`SketchBuffers::update_jobs`] (prefill, eager): one CTA per (page job, KV head) from
//!   a host-planned job list ([`plan_jobs`]) uploaded into a fixed device buffer.
//! * [`SketchBuffers::update_tail`] (decode and verify, graph-safe): one CTA per (sequence,
//!   page of its last `n_new` tokens, KV head), everything read from the step's device
//!   block table and lengths, so a captured launch stays right on every replay.
//!
//! Nothing here reads a row: lane L32 builds the buffers only when `kv_view_selection` is
//! on, so with the row off no byte is allocated and no kernel launched.

use anyhow::{Context, Result, bail};

use super::KvCacheDtype;
use crate::gpu::{DevicePtr, GpuBackend, KernelHandle};
use crate::kernel_args::KernelLaunch;
pub use crate::kv_select::SketchView;

/// Page jobs and the tail-update shape (host planning only).
mod jobs;
pub use jobs::{SketchJob, TailBatch, WrittenRange, plan_jobs, tail_pages_span};

/// The sketch kernels' module (the file stem of its `.cu`).
pub const KERNEL_MODULE: &str = "kv_page_sketch_ada";
/// In-place inverse RoPE of captured BF16 query rows (lane L31's query capture).
pub const UNROPE_KERNEL: &str = "kv_query_unrope_bf16";

const K_JOBS_BF16: &str = "kv_page_sketch_jobs_bf16";
const K_JOBS_FP8: &str = "kv_page_sketch_jobs_fp8";
const K_TAIL_BF16: &str = "kv_page_sketch_tail_bf16";
const K_TAIL_FP8: &str = "kv_page_sketch_tail_fp8";

/// i32 words per job: block (or -1), stream, logical page, fill.
pub const JOB_WORDS: usize = 4;
/// Default job capacity: a 4,096-token chunk on 16 streams is 4,112 page jobs.
pub const DEFAULT_MAX_JOBS: usize = 8192;

/// Shapes of the sketch (fixed for the serve).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SketchConfig {
    /// Physical pages in the KV pool.
    pub num_blocks: usize,
    pub attn_layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Tokens per page (the KV block size; at most 255, the count is a u8).
    pub block_size: usize,
    /// Leading dims RoPE rotates (64 on Qwen3.5: `partial_rotary_factor` 0.25 of 256).
    pub rotary_dim: usize,
    pub rope_theta: f32,
    /// Page jobs one `update_jobs` launch covers (larger lists take several launches).
    pub max_jobs: usize,
}

impl SketchConfig {
    pub fn validate(&self) -> Result<()> {
        let c = self;
        if c.num_blocks == 0 || c.attn_layers == 0 || c.kv_heads == 0 || c.head_dim == 0 {
            bail!("kv sketch: empty shape {c:?}");
        }
        if c.block_size == 0 || c.block_size > 255 {
            bail!("kv sketch: block_size {} outside 1..=255", c.block_size);
        }
        if c.rotary_dim > c.head_dim || !c.rotary_dim.is_multiple_of(2) {
            bail!(
                "kv sketch: rotary_dim {} must be even and at most head_dim {}",
                c.rotary_dim,
                c.head_dim
            );
        }
        if c.max_jobs == 0 || !c.rope_theta.is_finite() || c.rope_theta <= 0.0 {
            bail!(
                "kv sketch: max_jobs {} / rope_theta {} invalid",
                c.max_jobs,
                c.rope_theta
            );
        }
        if u32::try_from(c.num_blocks).is_err() {
            bail!("kv sketch: {} pages exceed u32", c.num_blocks);
        }
        Ok(())
    }

    /// Sketch bytes per page: the BF16 means of every layer and head plus the count.
    pub fn bytes_per_block(attn_layers: usize, kv_heads: usize, head_dim: usize) -> usize {
        attn_layers * kv_heads * head_dim * 2 + 1
    }

    pub fn means_bytes(&self) -> usize {
        self.num_blocks * self.attn_layers * self.kv_heads * self.head_dim * 2
    }

    pub fn counts_bytes(&self) -> usize {
        self.num_blocks
    }

    pub fn jobs_bytes(&self) -> usize {
        self.max_jobs * JOB_WORDS * 4
    }

    /// Every device byte [`SketchBuffers::new`] allocates.
    pub fn total_bytes(&self) -> usize {
        self.means_bytes() + self.counts_bytes() + self.jobs_bytes()
    }
}

/// Pages a KV budget holds when every page also carries a sketch: the budget split so
/// that `pages * (kv_block_bytes + sketch_block_bytes) <= budget_bytes`.
pub fn blocks_for_budget(
    budget_bytes: usize,
    kv_block_bytes: usize,
    sketch_block_bytes: usize,
) -> usize {
    budget_bytes / (kv_block_bytes + sketch_block_bytes).max(1)
}

/// One attention layer's K pool as the sketch kernels read it.
#[derive(Clone, Copy, Debug)]
pub struct SketchLayer {
    pub attn_layer: usize,
    pub k_pool: DevicePtr,
    /// BF16 or FP8 (E4M3); anything else is refused.
    pub dtype: KvCacheDtype,
    /// Bytes from one page of this layer's K pool to the next.
    pub block_stride_bytes: u64,
    /// FP8 dequant scale (`value = fp8 * kv_scale`); ignored for BF16.
    pub kv_scale: f32,
}

struct Kernels {
    jobs_bf16: KernelHandle,
    jobs_fp8: KernelHandle,
    tail_bf16: KernelHandle,
    tail_fp8: KernelHandle,
}

/// The sketch's device buffers and kernel handles.
pub struct SketchBuffers {
    cfg: SketchConfig,
    means: DevicePtr,
    counts: DevicePtr,
    jobs: DevicePtr,
    k: Kernels,
}

/// Threads of one sketch CTA (loops over `head_dim`).
fn sketch_threads(head_dim: usize) -> u32 {
    (head_dim.div_ceil(32) * 32).clamp(32, 256) as u32
}

impl SketchBuffers {
    /// Allocate the sketch (counts zeroed, so nothing is eligible until written) and
    /// resolve the kernels. Fails, allocating nothing, when the module is not in the image.
    pub fn new(gpu: &dyn GpuBackend, cfg: SketchConfig) -> Result<Self> {
        cfg.validate()?;
        if !gpu.has_module(KERNEL_MODULE) {
            bail!(
                "kv sketch: module {KERNEL_MODULE} is not in this image (the bounded KV view \
                 needs the Ada kernels; build with AVAROK_TARGET_HW=ada)"
            );
        }
        let k = |name: &str| {
            gpu.kernel(KERNEL_MODULE, name)
                .with_context(|| format!("kv sketch: {KERNEL_MODULE}::{name}"))
        };
        let k = Kernels {
            jobs_bf16: k(K_JOBS_BF16)?,
            jobs_fp8: k(K_JOBS_FP8)?,
            tail_bf16: k(K_TAIL_BF16)?,
            tail_fp8: k(K_TAIL_FP8)?,
        };
        let means = gpu.alloc(cfg.means_bytes())?;
        let counts = match gpu.alloc(cfg.counts_bytes()) {
            Ok(p) => p,
            Err(e) => {
                let _ = gpu.free(means);
                return Err(e);
            }
        };
        let jobs = match gpu.alloc(cfg.jobs_bytes()) {
            Ok(p) => p,
            Err(e) => {
                let _ = gpu.free(means);
                let _ = gpu.free(counts);
                return Err(e);
            }
        };
        let me = Self {
            cfg,
            means,
            counts,
            jobs,
            k,
        };
        if let Err(e) = gpu.memset(counts, 0, cfg.counts_bytes()) {
            let _ = me.free(gpu);
            return Err(e);
        }
        Ok(me)
    }

    pub fn config(&self) -> &SketchConfig {
        &self.cfg
    }

    pub fn means_ptr(&self) -> DevicePtr {
        self.means
    }

    pub fn counts_ptr(&self) -> DevicePtr {
        self.counts
    }

    /// The scorer's view of one sequence whose pages are `pages` (chronological).
    pub fn view<'a>(&'a self, pages: &'a [u32]) -> DeviceSketch<'a> {
        DeviceSketch { buf: self, pages }
    }

    fn kernel_for(&self, layer: &SketchLayer, tail: bool) -> Result<KernelHandle> {
        Ok(match (layer.dtype, tail) {
            (KvCacheDtype::Bf16, false) => self.k.jobs_bf16,
            (KvCacheDtype::Fp8, false) => self.k.jobs_fp8,
            (KvCacheDtype::Bf16, true) => self.k.tail_bf16,
            (KvCacheDtype::Fp8, true) => self.k.tail_fp8,
            (other, _) => bail!("kv sketch: KV dtype {other} is not served (BF16 or FP8 only)"),
        })
    }

    /// Recompute the pages in `jobs` for one layer (eager: uploads the list). Jobs with
    /// `block = -1` resolve through `block_table_ptrs` (`[streams]` device pointers to i32
    /// page lists of `max_blocks_per_seq` entries); pass NULL when every job names its page.
    /// Returns the launches issued.
    pub fn update_jobs(
        &self,
        gpu: &dyn GpuBackend,
        layer: &SketchLayer,
        jobs: &[SketchJob],
        block_table_ptrs: DevicePtr,
        max_blocks_per_seq: u32,
        stream: u64,
    ) -> Result<usize> {
        if jobs.is_empty() {
            return Ok(0);
        }
        if gpu.stream_is_capturing(stream) {
            bail!(
                "kv sketch: update_jobs is eager-only (it uploads the job list); use update_tail under a graph"
            );
        }
        if layer.attn_layer >= self.cfg.attn_layers {
            bail!(
                "kv sketch: attention layer {} of {}",
                layer.attn_layer,
                self.cfg.attn_layers
            );
        }
        let kernel = self.kernel_for(layer, false)?;
        let mut launches = 0;
        for chunk in jobs.chunks(self.cfg.max_jobs) {
            let words: Vec<u8> = chunk
                .iter()
                .flat_map(|j| {
                    [
                        j.block,
                        j.stream as i32,
                        j.logical_page as i32,
                        j.fill as i32,
                    ]
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                })
                .collect();
            gpu.copy_h2d_async(&words, self.jobs, stream)?;
            self.launch_common(
                KernelLaunch::new(gpu, kernel)
                    .grid([chunk.len() as u32, self.cfg.kv_heads as u32, 1])
                    .block([sketch_threads(self.cfg.head_dim), 1, 1])
                    .arg_ptr(layer.k_pool)
                    .arg_u64(layer.block_stride_bytes)
                    .arg_f32(layer.kv_scale)
                    .arg_ptr(self.jobs)
                    .arg_u32(chunk.len() as u32)
                    .arg_ptr(block_table_ptrs)
                    .arg_u32(max_blocks_per_seq),
                layer,
            )
            .launch(stream)?;
            launches += 1;
        }
        Ok(launches)
    }

    /// Recompute the pages the last `n_new` tokens of every sequence landed in, reading
    /// pages and extents from the step's device arrays (graph-safe, one launch).
    pub fn update_tail(
        &self,
        gpu: &dyn GpuBackend,
        layer: &SketchLayer,
        t: &TailBatch,
        stream: u64,
    ) -> Result<()> {
        if t.n_new == 0 || t.n_rows == 0 {
            return Ok(());
        }
        if t.rows_per_seq == 0 || !t.n_rows.is_multiple_of(t.rows_per_seq) {
            bail!(
                "kv sketch: {} rows are not whole groups of {}",
                t.n_rows,
                t.rows_per_seq
            );
        }
        if layer.attn_layer >= self.cfg.attn_layers {
            bail!(
                "kv sketch: attention layer {} of {}",
                layer.attn_layer,
                self.cfg.attn_layers
            );
        }
        let kernel = self.kernel_for(layer, true)?;
        let n_seqs = t.n_rows / t.rows_per_seq;
        let span = tail_pages_span(t.n_new, self.cfg.block_size as u32);
        self.launch_common(
            KernelLaunch::new(gpu, kernel)
                .grid([n_seqs * span, self.cfg.kv_heads as u32, 1])
                .block([sketch_threads(self.cfg.head_dim), 1, 1])
                .arg_ptr(layer.k_pool)
                .arg_u64(layer.block_stride_bytes)
                .arg_f32(layer.kv_scale)
                .arg_ptr(t.block_table)
                .arg_ptr(t.seq_lens)
                .arg_u32(t.max_blocks_per_seq)
                .arg_u32(n_seqs)
                .arg_u32(t.rows_per_seq)
                .arg_u32(t.n_new)
                .arg_u32(span),
            layer,
        )
        .launch(stream)
    }

    /// Mark pages unsketched (count 0) so the scorer ignores them until their next write,
    /// e.g. pages a caller could not sketch. Eager.
    pub fn invalidate(&self, gpu: &dyn GpuBackend, blocks: &[u32], stream: u64) -> Result<usize> {
        let jobs: Vec<SketchJob> = blocks
            .iter()
            .filter_map(|&b| i32::try_from(b).ok())
            .map(|block| SketchJob {
                block,
                stream: 0,
                logical_page: 0,
                fill: 0,
            })
            .collect();
        let layer = SketchLayer {
            attn_layer: 0,
            k_pool: DevicePtr::NULL,
            dtype: KvCacheDtype::Bf16,
            block_stride_bytes: 0,
            kv_scale: 1.0,
        };
        self.update_jobs(gpu, &layer, &jobs, DevicePtr::NULL, 0, stream)
    }

    /// The trailing arguments both launch paths share.
    fn launch_common<'a>(&self, l: KernelLaunch<'a>, layer: &SketchLayer) -> KernelLaunch<'a> {
        let c = &self.cfg;
        l.arg_ptr(self.means)
            .arg_ptr(self.counts)
            .arg_u32(c.num_blocks as u32)
            .arg_u32(layer.attn_layer as u32)
            .arg_u32(c.attn_layers as u32)
            .arg_u32(c.kv_heads as u32)
            .arg_u32(c.head_dim as u32)
            .arg_u32(c.block_size as u32)
            .arg_u32(c.rotary_dim as u32)
            .arg_f32(c.rope_theta)
    }

    /// Return every buffer to the backend.
    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        let mut first = None;
        for p in [self.means, self.counts, self.jobs] {
            if let Err(e) = gpu.free(p)
                && first.is_none()
            {
                first = Some(e);
            }
        }
        first.map_or(Ok(()), Err)
    }
}

/// [`SketchView`] over the shared buffers for one sequence's page list.
pub struct DeviceSketch<'a> {
    buf: &'a SketchBuffers,
    pages: &'a [u32],
}

impl SketchView for DeviceSketch<'_> {
    fn pages(&self) -> &[u32] {
        self.pages
    }
    fn num_blocks(&self) -> usize {
        self.buf.cfg.num_blocks
    }
    fn layers(&self) -> usize {
        self.buf.cfg.attn_layers
    }
    fn kv_heads(&self) -> usize {
        self.buf.cfg.kv_heads
    }
    fn head_dim(&self) -> usize {
        self.buf.cfg.head_dim
    }
    fn page_tokens(&self) -> usize {
        self.buf.cfg.block_size
    }
    fn means(&self) -> DevicePtr {
        self.buf.means
    }
    fn counts(&self) -> DevicePtr {
        self.buf.counts
    }
}
