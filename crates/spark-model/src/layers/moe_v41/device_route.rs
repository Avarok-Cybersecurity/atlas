// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! The single-token routing on the device (`ATLAS_DS41_DEVICE_ROUTE=1`):
//! `moe_v41_route_select` computes the reference's selection bit for bit
//! (glibc-exact softplus, the stable ranking, the weights in pick order),
//! writes `weight_dev` / `rows_dev` and the gate / up / down pointer table in
//! plan order from a device slot table, and leaves one header for the host:
//! the miss flag, the picks, the weight bits and the plan's slots. The host
//! reads that header back once (the layer's one stream drain), touches the
//! picks in the cache, fetches the misses exactly as before, uploads the
//! cache's slot changes to the device table, and on a miss re-uploads the
//! pointer table from the slots the fetch returned. On a hit token nothing
//! else crosses the bus. Built on the Spark for the B200, where the step is
//! launch- and wait-bound; on the Spark it is expected neutral.
//!
//! `ATLAS_DS41_DEVICE_ROUTE_DIAG=1` also runs the host chain on the same
//! logits and asserts the picks, the weight bits, the plan weights and the
//! pointer table equal on every token (a mismatch fails the step).

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::weights::expert_stream::{ExpertLru, ExpertSource};

use super::{MoeV41, MoeV41LayerWeights, SLOT_TABLE_LAYERS, route_from_logits};

/// The two constants the kernel is written for.
const ROUTE_NR: usize = 384;
const ROUTE_TOPK: usize = 6;

fn device_route() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("ATLAS_DS41_DEVICE_ROUTE").is_ok_and(|v| v == "1"))
}

fn device_route_diag() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("ATLAS_DS41_DEVICE_ROUTE_DIAG").is_ok_and(|v| v == "1"))
}

static DIAG_CHECKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl MoeV41 {
    /// The device bias for a layer's router (`MoeV41LayerWeights::gate_bias_dev`).
    pub fn upload_bias(gpu: &dyn GpuBackend, bias: &[f32]) -> Result<DevicePtr> {
        let bytes: Vec<u8> = bias.iter().flat_map(|v| v.to_le_bytes()).collect();
        let p = gpu.alloc(bytes.len().max(16))?;
        gpu.copy_h2d(&bytes, p)?;
        Ok(p)
    }

    /// Is the device selection on, and does this layer fit it?
    pub(super) fn device_route_ok(&self, w: &MoeV41LayerWeights) -> bool {
        if !device_route() {
            return false;
        }
        let c = &self.cfg;
        let fits = c.n_routed == ROUTE_NR
            && c.topk == ROUTE_TOPK
            && (w.layer as usize) < SLOT_TABLE_LAYERS
            && !w.gate_bias_dev.is_null();
        if !fits {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    "moe_v41: ATLAS_DS41_DEVICE_ROUTE=1 ignored (n_routed {} topk {} layer {}): the kernel is written for {ROUTE_NR} x {ROUTE_TOPK}",
                    c.n_routed,
                    c.topk,
                    w.layer
                );
            });
        }
        fits
    }

    /// The selection kernel on `self.logits` (row 0) for layer `layer`, the
    /// pointer table from `arena` = (base, slot bytes, gate / up / down
    /// offsets). Device work only; the header lands in `self.route_hdr`.
    pub(super) fn route_select_launch(
        &self,
        gpu: &dyn GpuBackend,
        layer: u32,
        bias_dev: DevicePtr,
        arena: (u64, u64, u64, u64, u64),
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let row = DevicePtr(self.slot_table.0 + (layer as usize * c.n_routed * 4) as u64);
        KernelLaunch::new(gpu, self.k.route_select)
            .grid([1, 1, 1])
            .block([ROUTE_NR as u32, 1, 1])
            .arg_ptr(self.logits)
            .arg_ptr(bias_dev)
            .arg_f32(c.gate_temp)
            .arg_f32(c.route_scale)
            .arg_u32(c.norm_topk_prob as u32)
            .arg_ptr(row)
            .arg_u64(arena.0)
            .arg_u64(arena.1)
            .arg_u64(arena.2)
            .arg_u64(arena.3)
            .arg_u64(arena.4)
            .arg_ptr(self.route_hdr)
            .arg_ptr(self.weight_dev)
            .arg_ptr(self.rows_dev)
            .arg_ptr(self.ptrs_dev)
            .launch(stream)
    }

    /// Bring the device slot table up to date with the cache's changes.
    pub(super) fn slot_table_update(
        &self,
        gpu: &dyn GpuBackend,
        changes: &[(u32, u32, i32)],
        stream: u64,
    ) -> Result<()> {
        if changes.is_empty() {
            return Ok(());
        }
        let c = &self.cfg;
        // the triples are staged through `a_rows` (free until the experts
        // run), in chunks that fit it
        let per = (c.max_tokens * c.dim * 2 / 12).max(1);
        for chunk in changes.chunks(per) {
            let mut bytes: Vec<u8> = Vec::with_capacity(chunk.len() * 12);
            for &(l, e, s) in chunk {
                ensure!(
                    (l as usize) < SLOT_TABLE_LAYERS && (e as usize) < c.n_routed,
                    "slot table: layer {l} expert {e} outside the table"
                );
                bytes.extend_from_slice(&(l as i32).to_le_bytes());
                bytes.extend_from_slice(&(e as i32).to_le_bytes());
                bytes.extend_from_slice(&s.to_le_bytes());
            }
            gpu.copy_h2d_async(&bytes, self.a_rows, stream)?;
            let n = chunk.len() as u32;
            KernelLaunch::new(gpu, self.k.slot_table_set)
                .grid([n.div_ceil(256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.slot_table)
                .arg_u32(c.n_routed as u32)
                .arg_ptr(self.a_rows)
                .arg_u32(n)
                .launch(stream)?;
            // the next chunk rewrites the scratch: the pageable H2D is
            // synchronous with respect to the host buffer, and the kernel
            // reads it in stream order before the next copy lands
            gpu.synchronize(stream)?;
        }
        Ok(())
    }

    /// The device selection for one token, the header read back, the cache
    /// touched and the misses fetched, the slot table and (on a miss) the
    /// pointer table brought up to date. `route_launch` must have run on
    /// `stream`. Returns the routing as `route_select` does.
    pub(super) fn route_device_m1<S: ExpertSource + ?Sized>(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        lru: &mut ExpertLru,
        src: &S,
        reader_threads: usize,
        stream: u64,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        let c = &self.cfg;
        let lay = lru.layout();
        let arena = (
            lru.arena_dev().0,
            lay.bytes as u64,
            lay.gate_off as u64,
            lay.up_off as u64,
            lay.down_off as u64,
        );
        // the table must know every slot the cache filled so far (prefill,
        // the host-routed layers): drain before the kernel reads it
        let pending = lru.drain_slot_changes();
        self.slot_table_update(gpu, &pending, stream)?;
        self.route_select_launch(gpu, w.layer, w.gate_bias_dev, arena, stream)?;
        let k = c.topk;
        let mut hdr = vec![0u8; (1 + 3 * k) * 4];
        gpu.copy_d2h_on_stream(self.route_hdr, &mut hdr, stream)?;
        let word = |i: usize| {
            i32::from_le_bytes([hdr[4 * i], hdr[4 * i + 1], hdr[4 * i + 2], hdr[4 * i + 3]])
        };
        let miss_flag = word(0) != 0;
        let indices: Vec<usize> = (0..k).map(|i| word(1 + i) as usize).collect();
        let weights: Vec<f32> = (0..k)
            .map(|i| f32::from_bits(word(1 + k + i) as u32))
            .collect();
        ensure!(
            indices.iter().all(|&e| e < c.n_routed),
            "device route: pick outside 0..{}",
            c.n_routed
        );
        // the cache: hits touched, misses read (and, on the device arena,
        // their copies enqueued on `stream` ahead of the experts)
        lru.begin_token();
        let keys: Vec<(u32, u32)> = indices.iter().map(|&e| (w.layer, e as u32)).collect();
        let slots = lru.fetch_many_on(gpu, stream, src, &keys, &[], reader_threads)?;
        let changes = lru.drain_slot_changes();
        let changed = !changes.is_empty();
        self.slot_table_update(gpu, &changes, stream)?;
        if miss_flag || changed {
            // a pick was not resident when the kernel looked, or the fetch
            // moved a slot: the pointer table from the slots the fetch
            // returned, in plan order, as the host path builds it
            let (_, _, plan) = self.plan(&indices, &weights, 1);
            ensure!(
                plan.len() == k,
                "device route: {} distinct picks, expected {k}",
                plan.len()
            );
            self.upload_expert_table(gpu, &plan, &slots, stream)?;
        }
        if device_route_diag() {
            self.device_route_check(gpu, w, &indices, &weights, &slots, stream)?;
        }
        Ok((weights, indices))
    }

    /// The host chain on the same logits against the device's header, the
    /// plan weights and the pointer table; a mismatch fails the step.
    fn device_route_check(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        indices: &[usize],
        weights: &[f32],
        slots: &[spark_runtime::weights::expert_stream::ExpertSlot],
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let k = c.topk;
        let mut bytes = vec![0u8; c.n_routed * 4];
        gpu.copy_d2h_on_stream(self.logits, &mut bytes, stream)?;
        let logits: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let (hw, hi) = route_from_logits(&logits, 1, &w.gate_bias, c);
        let l = w.layer;
        if hi != indices {
            bail!("device route L{l}: picks {indices:?} != host {hi:?}");
        }
        let (wb, hwb): (Vec<u32>, Vec<u32>) = (
            weights.iter().map(|v| v.to_bits()).collect(),
            hw.iter().map(|v| v.to_bits()).collect(),
        );
        if wb != hwb {
            bail!("device route L{l}: weight bits {wb:08x?} != host {hwb:08x?}");
        }
        // the plan the device wrote: weight_dev in ascending id, ptrs_dev
        let (_, w_host, plan) = self.plan(&hi, &hw, 1);
        let mut w_dev = vec![0u8; k * 4];
        gpu.copy_d2h_on_stream(self.weight_dev, &mut w_dev, stream)?;
        if w_dev != w_host {
            bail!("device route L{l}: plan weights differ from the host plan");
        }
        let mut ptrs = vec![0u8; 3 * k * 8];
        gpu.copy_d2h_on_stream(self.ptrs_dev, &mut ptrs, stream)?;
        let mut want: Vec<u8> = Vec::with_capacity(3 * k * 8);
        for which in 0..3 {
            for &(a0, _, _) in &plan {
                let s = slots[a0];
                want.extend_from_slice(&[s.gate, s.up, s.down][which].0.to_le_bytes());
            }
        }
        if ptrs != want {
            bail!("device route L{l}: pointer table differs from the fetched slots");
        }
        let n = DIAG_CHECKED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n.is_multiple_of(4000) {
            tracing::info!("device route diag: {n} layer-tokens checked, 0 mismatches");
        }
        Ok(())
    }
}
