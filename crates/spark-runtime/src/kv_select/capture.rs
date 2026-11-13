// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the device side of the bounded view's query capture (lane L31): fixed
//! BF16 slots holding the post-norm, pre-RoPE Q rows of the last `query_tokens` tokens of
//! a request's final user message, per attention layer ([`QueryCapture`]), the copies that
//! fill them from a prefill call's Q ([`QueryCapture::capture`]), and the scorer's view of
//! a full slot ([`CapturedRows`]). The sparse seam adapter that calls `capture` from the
//! selector's `on_query` hook is `spark-model/src/port/kv_view/query_capture.rs`; this half
//! knows nothing about the seam's types, so it is tested here on the mock backend.
//!
//! Slot layout: `[slots, attn_layers, query_tokens, q_heads, head_dim]` BF16; row `i` of a
//! slot is absolute position `span_start + i`. 32 MiB per slot on the 4B (8 x 512 x 16 x
//! 256 x 2 B). Rows that arrive post-RoPE (a prefill site whose Q RoPE is fused) are copied
//! anyway and rotated back in place by `kv_query_unrope_bf16`, so a slot always holds
//! pre-RoPE rows (exact up to the BF16 rounding of the RoPE output). Nothing is allocated
//! until a caller constructs a capture, which only happens with `kv_view_selection` on.

use std::sync::Mutex;

use anyhow::{Context, Result, bail};

use super::QueryView;
use crate::gpu::{DevicePtr, GpuBackend, KernelHandle};
use crate::kernel_args::KernelLaunch;
use crate::kv_cache::sketch::{KERNEL_MODULE, UNROPE_KERNEL};

/// Query rows captured per request when the flag gives no other value.
pub const DEFAULT_QUERY_TOKENS: usize = 512;

/// Shapes of the capture (fixed for the serve).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueryCaptureCfg {
    pub attn_layers: usize,
    pub q_heads: usize,
    pub head_dim: usize,
    /// Rows kept per request: the last `query_tokens` tokens of the final user message.
    pub query_tokens: usize,
    /// Requests that can be captured at once (one slot each).
    pub slots: usize,
    pub rotary_dim: usize,
    pub rope_theta: f32,
}

impl QueryCaptureCfg {
    pub fn validate(&self) -> Result<()> {
        let c = self;
        if c.attn_layers == 0 || c.q_heads == 0 || c.head_dim == 0 {
            bail!("query capture: empty shape {c:?}");
        }
        if c.query_tokens == 0 || c.slots == 0 || u32::try_from(c.query_tokens).is_err() {
            bail!(
                "query capture: {} query tokens / {} slots invalid",
                c.query_tokens,
                c.slots
            );
        }
        if c.rotary_dim > c.head_dim || !c.rotary_dim.is_multiple_of(2) {
            bail!(
                "query capture: rotary_dim {} must be even and at most head_dim {}",
                c.rotary_dim,
                c.head_dim
            );
        }
        Ok(())
    }

    /// Bytes of one captured row of one layer.
    pub fn row_bytes(&self) -> usize {
        self.q_heads * self.head_dim * 2
    }

    pub fn slot_bytes(&self) -> usize {
        self.attn_layers * self.query_tokens * self.row_bytes()
    }

    /// Every device byte [`QueryCapture::new`] allocates.
    pub fn total_bytes(&self) -> usize {
        self.slots * self.slot_bytes()
    }
}

/// The Q rows one stream of a prefill call carries: positions `first_pos .. first_pos + n`,
/// stacked after the previous streams' rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamRows {
    pub seq_id: u64,
    pub first_pos: u32,
    pub n: u32,
}

#[derive(Clone, Debug, Default)]
struct Slot {
    seq_id: Option<u64>,
    span_start: u32,
    span_end: u32,
    /// Per layer, one bit per span row already captured.
    covered: Vec<Vec<u64>>,
}

impl Slot {
    fn rows(&self) -> usize {
        (self.span_end - self.span_start) as usize
    }

    fn mark(&mut self, layer: usize, lo: usize, hi: usize) {
        for i in lo..hi {
            self.covered[layer][i / 64] |= 1u64 << (i % 64);
        }
    }

    fn captured(&self, layer: usize) -> usize {
        self.covered[layer]
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum()
    }
}

/// The capture's device slots and their owners.
pub struct QueryCapture {
    cfg: QueryCaptureCfg,
    buf: DevicePtr,
    unrope: KernelHandle,
    slots: Mutex<Vec<Slot>>,
}

impl QueryCapture {
    /// Allocate every slot and resolve the unrope kernel. Fails, allocating nothing, when
    /// the module is not in this image.
    pub fn new(gpu: &dyn GpuBackend, cfg: QueryCaptureCfg) -> Result<Self> {
        cfg.validate()?;
        if !gpu.has_module(KERNEL_MODULE) {
            bail!(
                "query capture: module {KERNEL_MODULE} is not in this image (the bounded KV \
                 view needs the Ada kernels)"
            );
        }
        let unrope = gpu
            .kernel(KERNEL_MODULE, UNROPE_KERNEL)
            .with_context(|| format!("query capture: {KERNEL_MODULE}::{UNROPE_KERNEL}"))?;
        let buf = gpu.alloc(cfg.total_bytes())?;
        let slots = Mutex::new(vec![Slot::default(); cfg.slots]);
        Ok(Self {
            cfg,
            buf,
            unrope,
            slots,
        })
    }

    pub fn config(&self) -> &QueryCaptureCfg {
        &self.cfg
    }

    /// Base of the slot buffer (slot `i` starts `i * slot_bytes()` later).
    pub fn buffer(&self) -> DevicePtr {
        self.buf
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Slot>> {
        self.slots.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Claim a slot for `seq_id`, whose final user message spans positions `[msg_start,
    /// msg_end)`; the slot keeps its last `query_tokens` positions. Returns the kept span,
    /// or `None` when the span is empty or every slot is taken (the caller skips the
    /// selection). Re-arming a sequence resets its slot.
    pub fn arm(&self, seq_id: u64, msg_start: u32, msg_end: u32) -> Option<(u32, u32)> {
        if msg_end <= msg_start {
            return None;
        }
        let start = msg_start.max(msg_end.saturating_sub(self.cfg.query_tokens as u32));
        let mut slots = self.lock();
        let i = slots
            .iter()
            .position(|s| s.seq_id == Some(seq_id))
            .or_else(|| slots.iter().position(|s| s.seq_id.is_none()))?;
        let words = (msg_end - start).div_ceil(64) as usize;
        slots[i] = Slot {
            seq_id: Some(seq_id),
            span_start: start,
            span_end: msg_end,
            covered: vec![vec![0; words]; self.cfg.attn_layers],
        };
        Some((start, msg_end))
    }

    /// Release `seq_id`'s slot (finished, cancelled, edited or selected).
    pub fn disarm(&self, seq_id: u64) {
        for s in self.lock().iter_mut().filter(|s| s.seq_id == Some(seq_id)) {
            *s = Slot::default();
        }
    }

    /// The span `seq_id` is armed for.
    pub fn armed(&self, seq_id: u64) -> Option<(u32, u32)> {
        let slots = self.lock();
        let s = slots.iter().find(|s| s.seq_id == Some(seq_id))?;
        Some((s.span_start, s.span_end))
    }

    /// Rows captured so far, per layer.
    pub fn captured_rows(&self, seq_id: u64) -> Option<Vec<usize>> {
        let slots = self.lock();
        let s = slots.iter().find(|s| s.seq_id == Some(seq_id))?;
        Some((0..self.cfg.attn_layers).map(|l| s.captured(l)).collect())
    }

    /// The captured rows as the scorer reads them, once every layer holds every row of the
    /// span; `None` before that (or when `seq_id` is not armed).
    pub fn rows_view(&self, seq_id: u64) -> Option<CapturedRows> {
        let slots = self.lock();
        let i = slots.iter().position(|s| s.seq_id == Some(seq_id))?;
        let s = &slots[i];
        if (0..self.cfg.attn_layers).any(|l| s.captured(l) < s.rows()) {
            return None;
        }
        Some(CapturedRows {
            data: self.buf.offset(i * self.cfg.slot_bytes()),
            rows: s.rows(),
            rows_cap: self.cfg.query_tokens,
            layers: self.cfg.attn_layers,
            q_heads: self.cfg.q_heads,
            head_dim: self.cfg.head_dim,
        })
    }

    /// Copy the rows of armed sequences that fall in their span into their slots for
    /// attention layer `layer`, on `stream` (eager: the caller never captures prefill).
    /// `src` is `[q_rows, src_row_stride]` BF16 elements, streams stacked in order;
    /// `pre_rope` false rotates the copied rows back. Returns the rows captured.
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        &self,
        gpu: &dyn GpuBackend,
        layer: usize,
        streams: &[StreamRows],
        src: DevicePtr,
        src_row_stride: usize,
        q_rows: usize,
        pre_rope: bool,
        stream: u64,
    ) -> Result<usize> {
        let cfg = self.cfg;
        if layer >= cfg.attn_layers {
            return Ok(0);
        }
        let row_bytes = cfg.row_bytes();
        let src_pitch = src_row_stride * 2;
        if src_pitch < row_bytes {
            bail!(
                "query capture: Q row stride {src_row_stride} below {} heads x {} dims",
                cfg.q_heads,
                cfg.head_dim
            );
        }
        let mut slots = self.lock();
        let (mut row_base, mut captured) = (0usize, 0usize);
        for st in streams {
            let base = row_base;
            row_base += st.n as usize;
            let Some(k) = slots.iter().position(|sl| sl.seq_id == Some(st.seq_id)) else {
                continue;
            };
            let slot = &mut slots[k];
            let lo = st.first_pos.max(slot.span_start);
            let hi = (st.first_pos + st.n).min(slot.span_end);
            if lo >= hi || base + (hi - st.first_pos) as usize > q_rows {
                continue;
            }
            let rows = (hi - lo) as usize;
            let from = src.offset((base + (lo - st.first_pos) as usize) * src_pitch);
            let first = (lo - slot.span_start) as usize;
            let dst = self
                .buf
                .offset(k * cfg.slot_bytes() + (layer * cfg.query_tokens + first) * row_bytes);
            if src_pitch == row_bytes {
                gpu.copy_d2d_async(from, dst, rows * row_bytes, stream)?;
            } else {
                gpu.copy_d2d_2d_async(from, src_pitch, dst, row_bytes, row_bytes, rows, stream)?;
            }
            if !pre_rope && cfg.rotary_dim > 0 {
                KernelLaunch::new(gpu, self.unrope)
                    .grid([rows as u32, cfg.q_heads as u32, 1])
                    .block([32, 1, 1])
                    .arg_ptr(dst)
                    .arg_u32(rows as u32)
                    .arg_u32((cfg.q_heads * cfg.head_dim) as u32)
                    .arg_u32(cfg.q_heads as u32)
                    .arg_u32(cfg.head_dim as u32)
                    .arg_u32(cfg.rotary_dim as u32)
                    .arg_u32(lo)
                    .arg_f32(cfg.rope_theta)
                    .launch(stream)?;
            }
            slot.mark(layer, first, first + rows);
            captured += rows;
        }
        Ok(captured)
    }

    /// Return the slots to the backend.
    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.buf)
    }
}

/// One sequence's captured rows ([`QueryView`] for [`super::score_blocks`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapturedRows {
    pub data: DevicePtr,
    pub rows: usize,
    pub rows_cap: usize,
    pub layers: usize,
    pub q_heads: usize,
    pub head_dim: usize,
}

impl QueryView for CapturedRows {
    fn rows(&self) -> usize {
        self.rows
    }
    fn layers(&self) -> usize {
        self.layers
    }
    fn q_heads(&self) -> usize {
        self.q_heads
    }
    fn head_dim(&self) -> usize {
        self.head_dim
    }
    fn data(&self) -> DevicePtr {
        self.data
    }
    fn layer_stride_rows(&self) -> usize {
        self.rows_cap
    }
}
