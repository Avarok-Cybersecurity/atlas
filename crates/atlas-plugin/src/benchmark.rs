// SPDX-License-Identifier: AGPL-3.0-only

//! The [`Benchmark`] trait — a [`Plugin`] that is a drivable state machine.
//!
//! A benchmark owns its own phase state and does one step of work per
//! [`Benchmark::next`], returning the frame the TUI renders. It is driven, not
//! in control: it must never block the runtime and never loop internally to
//! completion, or the pane freezes and cancellation stops working.
//!
//! [`Benchmark::run`] is implemented here — it drives `next()` in a loop and
//! streams the frames. The loop itself lives in [`crate::executor::drive`] so
//! that direct and registry-dispatched runs cannot diverge.

use std::future::Future;

use anyhow::Result;
use futures::Stream;

use crate::dynamic::DynBenchmark;
use crate::params::{ParamSpec, ParamValues};
use crate::plugin::Plugin;
use crate::result::BenchmarkResult;

/// Static identity of a benchmark, and how to construct one.
///
/// This is the SSOT the registry, the list pane and the run-history filenames
/// all read — an id that appears here and nowhere else cannot drift.
pub struct BenchmarkDescriptor {
    /// Stable, filename-safe. Used for `~/.atlas/runs/<id>/`.
    pub id: &'static str,
    pub name: &'static str,
    /// One line for the suite list.
    pub summary: &'static str,
    /// A paragraph for the detail pane: what it measures and what it costs.
    pub detail: &'static str,
    /// Rough wall time at default parameters, e.g. `"~15 min"`.
    pub duration_hint: &'static str,
    /// True when starting has a side effect beyond load on the endpoint. The
    /// pane requires an explicit confirmation for these — currently only the
    /// agentic test, which executes model-authored shell in a sandbox.
    pub needs_confirmation: bool,
    pub ctor: fn() -> Box<dyn DynBenchmark>,
}

impl BenchmarkDescriptor {
    pub fn build(&self) -> Box<dyn DynBenchmark> {
        (self.ctor)()
    }
}

pub trait Benchmark: Plugin {
    fn descriptor(&self) -> &'static BenchmarkDescriptor;

    /// The parameters the terminal renders BEFORE the run starts, so the user
    /// can change them. Defaults live in the returned specs and nowhere else.
    fn parameters(&self) -> Vec<ParamSpec>;

    /// Receive the edited values. Validate here and return a message naming the
    /// offending field — a bad value must never reach `next()`.
    fn configure(&mut self, values: &ParamValues) -> Result<()>;

    /// Drive `next()` to completion, streaming every frame. The stream ends
    /// after the first terminal [`crate::RunStatus`], or after an error.
    fn run(&mut self) -> impl Stream<Item = Result<BenchmarkResult>> + '_
    where
        Self: Sized + Send,
    {
        crate::executor::drive(self)
    }

    /// One step of work. Implemented by the benchmark; called repeatedly.
    fn next(&mut self) -> impl Future<Output = Result<BenchmarkResult>> + Send;

    /// Release whatever the run acquired. Runs on every exit path — completion,
    /// failure and cancellation alike.
    fn cleanup(&mut self) -> impl Future<Output = Result<()>> + Send {
        async { Ok(()) }
    }
}
