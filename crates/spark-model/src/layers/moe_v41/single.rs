// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! The single-token expert path: the plan that groups a token's selected
//! experts, the pointer table for them, the routed m=1 compute that sums
//! the expert rows in plan order, and the host stage / device compute halves
//! a captured decode step runs. Split from `moe_v41/forward.rs` (500-LoC cap)
//! when the line was re-cut onto main.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::expert_stream::{ExpertLru, ExpertSource};

use super::{MoeV41, MoeV41LayerWeights, MoeV41Stage, MoeV41Timing, RouterWeights, prefetch_k};
use crate::layers::ops::{
    kquant_mmvq_experts_wn, kquant_q8_1_rows, kquant_q8_1_rows_bytes, kquant_swiglu_q8_1_rows,
};

impl MoeV41 {
    /// Group the `(token, k)` assignments by expert (ascending expert id):
    /// the token rows and routing weights group-major, and the plan
    /// `(first assignment index, row offset, rows)` per group.
    #[allow(clippy::type_complexity)]
    pub(super) fn plan(
        &self,
        indices: &[usize],
        weights: &[f32],
        m: usize,
    ) -> (Vec<u8>, Vec<u8>, Vec<(usize, usize, usize)>) {
        let c = &self.cfg;
        let mut groups: std::collections::BTreeMap<usize, Vec<(i32, f32, usize)>> =
            std::collections::BTreeMap::new();
        for t in 0..m {
            for kk in 0..c.topk {
                let a = t * c.topk + kk;
                groups
                    .entry(indices[a])
                    .or_default()
                    .push((t as i32, weights[a], a));
            }
        }
        let mut rows_host: Vec<u8> = Vec::with_capacity(m * c.topk * 4);
        let mut w_host: Vec<u8> = Vec::with_capacity(m * c.topk * 4);
        let mut plan: Vec<(usize, usize, usize)> = Vec::with_capacity(groups.len());
        for members in groups.values() {
            let off = rows_host.len() / 4;
            for &(t, rw, _a) in members {
                rows_host.extend_from_slice(&t.to_le_bytes());
                w_host.extend_from_slice(&rw.to_le_bytes());
            }
            plan.push((members[0].2, off, members.len()));
        }
        (rows_host, w_host, plan)
    }

    /// The single-token arm's pointer table: gate, up and down block pointers
    /// of the plan's experts, `3 * ne` entries into `ptrs_dev`. Returns `ne`.
    pub(super) fn upload_expert_table(
        &self,
        gpu: &dyn GpuBackend,
        plan: &[(usize, usize, usize)],
        slots: &[spark_runtime::weights::expert_stream::ExpertSlot],
        stream: u64,
    ) -> Result<usize> {
        let ne = plan.len();
        let mut ptrs: Vec<u8> = Vec::with_capacity(3 * ne * 8);
        for which in 0..3 {
            for &(a0, _, _) in plan {
                let slot = slots[a0];
                let p = [slot.gate, slot.up, slot.down][which];
                ptrs.extend_from_slice(&p.0.to_le_bytes());
            }
        }
        gpu.copy_h2d_async(&ptrs, self.ptrs_dev, stream)?;
        Ok(ne)
    }

    /// The token's q8_1 rows into `a_q8`, once for the routed experts and the
    /// shared expert's two input projections (`x_pre` at the readers).
    pub fn ffn_input_q8_m1(&self, gpu: &dyn GpuBackend, x: DevicePtr, stream: u64) -> Result<()> {
        kquant_q8_1_rows(
            gpu,
            self.k.q8_rows,
            x,
            self.a_q8,
            1,
            self.cfg.dim as u32,
            stream,
        )
    }

    /// The single-token routed experts from the pointer table: the token
    /// quantised once (already in `a_q8` when `x_pre`), each projection one
    /// launch over the `ne` experts, the routing weight folded in at the
    /// SwiGLU (which writes `h` and its q8_1 rows in one launch), the rows
    /// summed into `acc` in plan order. Device work only.
    pub(super) fn routed_m1(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        ne: usize,
        x_pre: bool,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let table = |which: usize| DevicePtr(self.ptrs_dev.0 + (which * ne * 8) as u64);
        if !x_pre {
            self.ffn_input_q8_m1(gpu, x, stream)?;
        }
        // gate and up in ONE 2 * ne expert batch: the pointer table holds the
        // ne gate pointers then the ne up pointers contiguously, every entry
        // reads the same q8_1 token row, and expert e writes row e of
        // gate_out, so up's rows start at gate_out + ne * inter. Same bytes
        // per row as the two launches.
        let up_view = DevicePtr(self.gate_out.0 + (ne * c.inter * 2) as u64);
        kquant_mmvq_experts_wn(
            gpu,
            self.k.mmvq_q2k_experts,
            table(0),
            self.a_q8,
            self.gate_out,
            c.inter as u32,
            c.dim as u32,
            1,
            2 * ne as u32,
            0,
            self.k.experts_warps,
            stream,
        )?;
        kquant_swiglu_q8_1_rows(
            gpu,
            self.k.swiglu_q8,
            self.gate_out,
            up_view,
            self.weight_dev,
            self.h,
            self.h_q8,
            ne as u32,
            c.inter as u32,
            c.swiglu_limit,
            stream,
        )?;
        kquant_mmvq_experts_wn(
            gpu,
            self.k.mmvq_q3k_experts,
            table(2),
            self.h_q8,
            self.down_out,
            c.dim as u32,
            c.inter as u32,
            1,
            ne as u32,
            kquant_q8_1_rows_bytes(1, c.inter as u32) as u32,
            self.k.experts_warps,
            stream,
        )?;
        // All `ne` rows land in the one token row: summed in plan order by a
        // single kernel. `scatter_add` here (one block per expert row, every
        // block on the same `acc` row) was an inter-block data race.
        self.launch_n(gpu, self.k.sum_rows, c.dim, stream, |l| {
            l.arg_ptr(self.acc)
                .arg_ptr(self.down_out)
                .arg_u32(ne as u32)
                .arg_u32(c.dim as u32)
        })
    }

    /// The HOST span of a single-token step between the router GEMV and the
    /// expert compute: the next layer's router on `x` (prediction, if
    /// `next`), the selection (drains the stream), the cache fetch with the
    /// predicted experts started in the background, and the rows / weights /
    /// pointer-table uploads into fixed buffers. `route_launch` must have
    /// been issued on `stream` first.
    pub fn stage_m1<S: ExpertSource + ?Sized>(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        lru: &mut ExpertLru,
        src: &S,
        reader_threads: usize,
        next: Option<&RouterWeights>,
        x: DevicePtr,
        stream: u64,
    ) -> Result<MoeV41Stage> {
        let t0 = std::time::Instant::now();
        let predict = next.filter(|_| prefetch_k() > 0 && lru.has_pool());
        if let Some(nw) = predict {
            self.predict_launch(gpu, nw, x, stream)?;
        }
        let (weights, indices) = self.route_select(gpu, w, 1, stream)?;
        let predicted = match predict {
            Some(nw) => self.predict_select(gpu, nw, prefetch_k(), stream)?,
            None => Vec::new(),
        };
        let t1 = std::time::Instant::now();
        lru.begin_token();
        let before = lru.stats();
        let keys: Vec<(u32, u32)> = indices.iter().map(|&e| (w.layer, e as u32)).collect();
        let slots = lru.fetch_many_on(gpu, stream, src, &keys, &predicted, reader_threads)?;
        let after = lru.stats();
        let t2 = std::time::Instant::now();
        let (rows_host, w_host, plan) = self.plan(&indices, &weights, 1);
        gpu.copy_h2d_async(&rows_host, self.rows_dev, stream)?;
        gpu.copy_h2d_async(&w_host, self.weight_dev, stream)?;
        let ne = self.upload_expert_table(gpu, &plan, &slots, stream)?;
        Ok(MoeV41Stage {
            ne,
            weights,
            indices,
            timing: MoeV41Timing {
                route_ms: (t1 - t0).as_secs_f64() * 1e3,
                fetch_ms: (t2 - t1).as_secs_f64() * 1e3,
                compute_ms: 0.0,
                hits: after.hits - before.hits,
                misses: after.misses - before.misses,
                bytes_read: after.bytes_read - before.bytes_read,
            },
        })
    }

    /// The single-token expert compute after `stage_m1`: the accumulator
    /// cleared, the routed experts from the pointer table, the shared expert,
    /// the finish into `out`. Device work only, no per-token scalar
    /// arguments (`ne` is the top-k: one token's picks are distinct experts),
    /// so a CUDA graph of it replays for any token once `stage_m1` has
    /// refilled `rows_dev` / `weight_dev` / `ptrs_dev`.
    pub fn compute_m1(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        x: DevicePtr,
        ne: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let c = &self.cfg;
        ensure!(
            ne == c.topk,
            "moe_v41: {ne} distinct experts for one token, expected the top-{}",
            c.topk
        );
        gpu.memset_async(self.acc, 0, c.dim * 4, stream)?;
        self.routed_m1(gpu, x, ne, false, stream)?;
        self.shared_expert(gpu, w, x, 1, true, stream)?;
        Ok(self.out)
    }

    /// The single-token shared expert up to `sd` on `side`, reading the
    /// token's q8_1 rows that [`Self::ffn_input_q8_m1`] wrote on the forking
    /// stream (its own `h` scratch, as the eager step's side stream uses):
    /// the fork half of [`Self::compute_m1_joined`]. Device work only.
    pub fn shared_expert_on(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        x: DevicePtr,
        side: u64,
    ) -> Result<()> {
        self.shared_expert_body(gpu, w, x, 1, self.a_q8, self.sh_q8, true, side)
    }

    /// [`Self::compute_m1`] with the shared expert forked onto a side stream
    /// by [`Self::shared_expert_on`]: the routed experts, then the wait on
    /// `join` (recorded on the side after `sd`), then the same accumulate
    /// and finish. The eager step's order and bits.
    pub fn compute_m1_joined(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        ne: usize,
        join: u64,
        stream: u64,
    ) -> Result<DevicePtr> {
        let c = &self.cfg;
        ensure!(
            ne == c.topk,
            "moe_v41: {ne} distinct experts for one token, expected the top-{}",
            c.topk
        );
        gpu.memset_async(self.acc, 0, c.dim * 4, stream)?;
        self.routed_m1(gpu, x, ne, true, stream)?;
        gpu.stream_wait_event(stream, join)?;
        self.shared_expert_tail(gpu, 1, stream)?;
        Ok(self.out)
    }
}
