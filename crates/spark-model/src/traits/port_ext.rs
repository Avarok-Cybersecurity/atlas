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
    /// A CUDA event with timing enabled, for `AVAROK_MTP_TIMING=2` (L06).
    /// Default: unsupported, which disables the device phase timer.
    fn timing_event_create(&self) -> anyhow::Result<u64> {
        anyhow::bail!("timing events are not supported by this model")
    }

    /// Record a timing event on `stream` (L06).
    fn timing_event_record(&self, _event: u64, _stream: u64) -> anyhow::Result<()> {
        anyhow::bail!("timing events are not supported by this model")
    }

    /// Milliseconds from `start` to `end`; `Ok(None)` while `end` has not
    /// completed (L06). Never blocks.
    fn timing_event_elapsed_ms(&self, _start: u64, _end: u64) -> anyhow::Result<Option<f32>> {
        anyhow::bail!("timing events are not supported by this model")
    }

    /// Drop every slot still recorded as folded by the last GDN
    /// write-on-accept fold (L10). The MTP batched verify calls this after
    /// its verdicts: a sequence that finished while emitting returned before
    /// the commit that consumes its entry, and slot indices are reused.
    /// Default: nothing to release.
    fn gdn_woa_release_folded(&self) {}

    /// Lane L12 (`mtp_device_accept`): pick one sequence's verify positions on the device
    /// (`crate::port::spec_accept`). Results are one per (regime, position) job, regime-major.
    /// `Ok(None)`: this model has no device path, so the caller keeps the host pipeline.
    fn spec_accept(
        &self,
        _req: &crate::port::spec_accept::AcceptRequest<'_>,
    ) -> anyhow::Result<Option<Vec<crate::port::spec_accept::JobResult>>> {
        Ok(None)
    }

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

    /// Start an asynchronous host-tier KV restore of the prefix of `tokens` for
    /// `adapter` (lane L27, row `kv_restore_async`; `TransformerModel` delegates
    /// to lane L26b's `model/kv_tier_glue.rs`). `Ok(None)` = nothing to restore.
    /// The caller parks the request (`scheduler/restore_queue.rs`) and polls
    /// [`Self::kv_restore_poll`]. Default: no host tier, nothing to restore.
    fn kv_restore_begin(
        &self,
        _tokens: &[u32],
        _adapter: u32,
    ) -> anyhow::Result<Option<spark_runtime::kv_host_tier::RestoreTicket>> {
        Ok(None)
    }

    /// Non-blocking state of a restore from [`Self::kv_restore_begin`] (lane
    /// L27). `Ready`: the restored blocks are in the prefix cache, so the
    /// request's admission finds them as an HBM hit. `Failed`: nothing was
    /// restored and every block it allocated is free again. Default: `Failed`
    /// (no tier issued the ticket), so the caller recomputes.
    fn kv_restore_poll(
        &self,
        _t: &spark_runtime::kv_host_tier::RestoreTicket,
    ) -> anyhow::Result<spark_runtime::kv_host_tier::RestoreState> {
        Ok(spark_runtime::kv_host_tier::RestoreState::Failed)
    }

    /// Bounded KV view (lane L32): before a prefill call, arm each stream's
    /// final user message for query capture and record the call's streams.
    /// Default: nothing.
    fn kv_view_begin_call(
        &self,
        _streams: &[crate::port::kv_view::CallStream<'_>],
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Bounded KV view (L32): after the prefill call. Default: nothing.
    fn kv_view_end_call(&self) {}

    /// Bounded KV view (L32): after a sequence's final prefill chunk, select
    /// (or keep, or skip) its per-turn view (`Some`), or drop its arm and view
    /// (`None`). `seq_id` is the sequence's pool slot. Default: off.
    fn set_kv_view(
        &self,
        _seq_id: u64,
        _spec: Option<crate::port::kv_view::ViewSpec<'_>>,
    ) -> anyhow::Result<crate::port::kv_view::ViewOutcome> {
        Ok(crate::port::kv_view::ViewOutcome::Off)
    }
    // @port:methods
}
