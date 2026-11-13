// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the port sprint's (`port/consumer-opt`) model-level entry
//! points, as one extension trait reached through `Model::port_ext`.
//!
//! A separate trait rather than more `Model` methods, so the lanes' methods
//! never touch `traits/model.rs` (allow-listed over the cap) and every other
//! `Model` implementor (the NLLB model, the test mocks) compiles unchanged:
//! `Model::port_ext` returns `None` by default, and only `TransformerModel`
//! overrides it (`model/trait_impl/mod.rs`) with its implementation in
//! `model/trait_impl/port_ext.rs`.
//!
//! Every method has a default body, so a lane's method is inert for any model
//! that does not implement it. Lines are inserted immediately BEFORE the marker
//! below by the coordinator's fragment apply pass.

/// The port lanes' model-level entry points. Every method has a default body.
pub trait ModelPortExt {

    /// Host KV tier (lane L26b): whether the tier serves this model (the `kv_host_tier` row
    /// on, a `--kv-host-tier-gb` budget, the prefix cache on). Default: no.
    fn kv_tier_active(&self) -> bool {
        false
    }

    /// Host KV tier (L26b): restore the prompt's host-held prefix blocks into the prefix cache
    /// before the normal lookup, waiting for the copy. Default: nothing restored.
    fn kv_tier_prefetch(
        &self,
        _tokens: &[u32],
        _adapter: u32,
        _session_hash: u64,
    ) -> anyhow::Result<crate::model::kv_tier_glue::TierPrefetch> {
        Ok(Default::default())
    }

    /// Host KV tier (L26b): start the same restore without waiting (the `kv_restore_async`
    /// path). Poll the ticket with `kv_restore_poll` (L27) and end it with `kv_tier_settle`.
    /// Default: nothing to restore.
    fn kv_tier_restore_begin(
        &self,
        _tokens: &[u32],
        _adapter: u32,
        _session_hash: u64,
    ) -> anyhow::Result<Option<spark_runtime::kv_host_tier::RestoreTicket>> {
        Ok(None)
    }

    /// Host KV tier (L26b): finish a restore (waiting for its copy if it still runs) and hand
    /// its ticket back. Every ticket ends here, including a cancelled parked request's.
    fn kv_tier_settle(&self, _ticket: spark_runtime::kv_host_tier::RestoreTicket) {}

    /// Host KV tier (L26b): once per scheduler tick (write-back budget, finished copies, the
    /// lazy store's drain). Default: nothing.
    fn kv_tier_tick(&self) {}

    /// Host KV tier (L26b): one measured prefill (`tokens` computed in `secs`) for the restore
    /// cost model. Default: ignored.
    fn kv_tier_observe_prefill(&self, _tokens: usize, _secs: f64) {}
    // @port:methods
}
