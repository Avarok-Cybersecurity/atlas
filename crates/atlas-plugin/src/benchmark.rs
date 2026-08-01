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
    /// When this benchmark's definition last changed, as `YYYY-MM-DD`.
    ///
    /// Not when the code was edited — when the MEASUREMENT changed: new
    /// thresholds, a different prompt set, a changed scoring rule. That is the
    /// date that decides whether two runs are comparable, which is the only
    /// question a reader has when they look at it.
    ///
    /// A compiled-in literal rather than a lookup: a benchmark ships with the
    /// binary, so unlike a recipe there is no upstream to ask.
    pub updated: &'static str,
    /// True when starting has a side effect beyond load on the endpoint. The
    /// pane requires an explicit confirmation for these — currently only the
    /// agentic test, which executes model-authored shell in a sandbox.
    pub needs_confirmation: bool,
    /// The checkpoints this benchmark is DEFINED on, if it is defined on any.
    ///
    /// Several gates are only meaningful against a particular model — Gate A's
    /// webserver_ok thresholds were measured on the 35B MoE flagship, and
    /// running it against the dense 27B produces numbers that look like a
    /// result but compare to nothing. The endpoint check reports a mismatch;
    /// it never refuses, because measuring a new checkpoint is how a gate gets
    /// extended.
    ///
    /// `None` means the benchmark measures whatever it is pointed at — true of
    /// the latency sweeps, which have no baseline tied to a checkpoint.
    pub intended_for: Option<ModelExpectation>,
    pub ctor: fn() -> Box<dyn DynBenchmark>,
}

/// Which checkpoints a benchmark's numbers mean something for.
#[derive(Clone, Copy, Debug)]
pub struct ModelExpectation {
    /// Lower-case substrings identifying an acceptable checkpoint FAMILY, not
    /// an exact id: the same model ships as `Qwen/…-FP8`, `nvidia/…-NVFP4` and
    /// `unsloth/…`, and a gate defined on the family accepts all of them.
    pub families: &'static [&'static str],
    /// What the reader needs to know when it does not match — which model the
    /// gate is defined on, and what running it elsewhere means.
    pub note: &'static str,
}

impl ModelExpectation {
    /// Does `model` belong to a family this benchmark is defined on?
    pub fn accepts(&self, model: &str) -> bool {
        let lowered = model.to_lowercase();
        self.families.iter().any(|f| lowered.contains(f))
    }
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

#[cfg(test)]
#[path = "benchmark_desc_tests.rs"]
mod desc_tests;
