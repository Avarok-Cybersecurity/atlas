// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: `impl ModelPortExt for TransformerModel`, the port sprint's
//! (`port/consumer-opt`) model-level entry points.
//!
//! One delegating line per method, each calling the owning lane's own
//! function. `Model::port_ext` returns `Some(self)` for `TransformerModel`
//! (the override sits in the one `impl Model` block in `mod.rs`, because a
//! trait method override cannot live in a second file). Lines are inserted
//! immediately BEFORE the marker below by the coordinator's fragment apply
//! pass.

use super::super::types::TransformerModel;
use crate::traits::port_ext::ModelPortExt;

impl ModelPortExt for TransformerModel {
    fn timing_event_create(&self) -> anyhow::Result<u64> {
        self.gpu.create_timing_event()
    }

    fn timing_event_record(&self, event: u64, stream: u64) -> anyhow::Result<()> {
        self.gpu.record_event(event, stream)
    }

    fn timing_event_elapsed_ms(&self, start: u64, end: u64) -> anyhow::Result<Option<f32>> {
        self.gpu.event_elapsed_ms(start, end)
    }

    fn kv_tier_active(&self) -> bool {
        self.tier_active()
    }

    fn kv_tier_prefetch(
        &self,
        tokens: &[u32],
        adapter: u32,
        session_hash: u64,
    ) -> anyhow::Result<crate::model::kv_tier_glue::TierPrefetch> {
        self.tier_prefetch(tokens, adapter, session_hash)
    }

    fn kv_tier_restore_begin(
        &self,
        tokens: &[u32],
        adapter: u32,
        session_hash: u64,
    ) -> anyhow::Result<Option<spark_runtime::kv_host_tier::RestoreTicket>> {
        self.tier_restore_begin(tokens, adapter, session_hash)
    }

    fn kv_tier_settle(&self, ticket: spark_runtime::kv_host_tier::RestoreTicket) {
        self.tier_settle(ticket)
    }

    fn kv_tier_tick(&self) {
        self.tier_tick()
    }

    fn kv_tier_observe_prefill(&self, tokens: usize, secs: f64) {
        self.tier_observe_prefill(tokens, secs)
    }

    fn kv_restore_begin(
        &self,
        tokens: &[u32],
        adapter: u32,
    ) -> anyhow::Result<Option<spark_runtime::kv_host_tier::RestoreTicket>> {
        self.restore_begin(tokens, adapter)
    }

    fn kv_restore_poll(
        &self,
        t: &spark_runtime::kv_host_tier::RestoreTicket,
    ) -> anyhow::Result<spark_runtime::kv_host_tier::RestoreState> {
        self.restore_poll(t)
    }
    // @port:impls
}
