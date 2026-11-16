// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the bounded view's live views (port lane L32), the host
//! mirror of the device descriptors: binding a view (least recently set entry
//! replaced when all are live), dropping a sequence's view by owner slot or by
//! its blocks, and publishing the live count and attended tokens. Every write
//! goes to the decode stream from an eager call, between steps.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::device::backend_key;
use super::metrics::Counters;
use super::state::{Inner, Live};
use super::view::Desc;

impl Inner {
    /// Bind a view on the device.
    pub(crate) fn activate(
        &mut self,
        gpu: &dyn GpuBackend,
        owner: u64,
        desc: Desc,
        view_tokens: u32,
        counters: &Counters,
    ) -> Result<()> {
        let Some(dev) = self.dev.as_ref() else {
            return Ok(());
        };
        let e = self
            .live
            .iter()
            .position(Option::is_none)
            .or_else(|| {
                self.live
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, l)| l.as_ref().map_or(0, |l| l.stamp))
                    .map(|(i, _)| i)
            })
            .unwrap_or(0);
        dev.write_entry(gpu, e, Some(&desc), gpu.default_stream())?;
        self.live[e] = Some(Live {
            owner,
            desc,
            view_tokens,
            stamp: self.clock,
        });
        self.publish(counters);
        Ok(())
    }

    /// Clear every live view `pred` selects (on the backend that holds them).
    pub(crate) fn clear_where(
        &mut self,
        gpu: &dyn GpuBackend,
        counters: &Counters,
        pred: impl Fn(&Live) -> bool,
    ) -> Result<()> {
        let Some(dev) = self.dev.as_ref() else {
            return Ok(());
        };
        if dev.owner != backend_key(gpu) {
            return Ok(());
        }
        let mut changed = false;
        for (e, slot) in self.live.iter_mut().enumerate() {
            if slot.as_ref().is_some_and(&pred) {
                dev.write_entry(gpu, e, None, gpu.default_stream())?;
                *slot = None;
                changed = true;
            }
        }
        if changed {
            self.publish(counters);
        }
        Ok(())
    }

    /// Drop the view a sequence in `owner` was given.
    pub(crate) fn deactivate_owner(
        &mut self,
        gpu: &dyn GpuBackend,
        owner: u64,
        counters: &Counters,
    ) -> Result<()> {
        self.clear_where(gpu, counters, |l| l.owner == owner)
    }

    /// Drop every view bound to a freed sequence's blocks.
    pub(crate) fn release_table(
        &mut self,
        gpu: &dyn GpuBackend,
        table: &[u32],
        len: u32,
        counters: &Counters,
    ) -> Result<()> {
        let Some(bs) = self.geom.map(|g| g.block_size) else {
            return Ok(());
        };
        self.clear_where(gpu, counters, |l| l.desc.matches(table, len, bs))
    }

    /// Publish the live count and the tokens the live views attend.
    pub(crate) fn publish(&self, counters: &Counters) {
        let (n, tokens) = self.live.iter().flatten().fold((0u64, 0u64), |(n, t), l| {
            (n + 1, t + u64::from(l.view_tokens))
        });
        counters.set_live(n, tokens);
    }
}
