// SPDX-License-Identifier: AGPL-3.0-only
#![deny(warnings)]
#![deny(clippy::all)]

//! Atlas plugins — and the benchmark suite the `spark serve` TUI drives.
//!
//! Two layers:
//!
//! * [`Plugin`] is the general abstraction. It gets a [`PluginHandle`] on
//!   [`Plugin::load`]: the seam onto the host terminal (status, log, progress,
//!   the run-glow), the `~/.atlas` artifact store, the endpoint it is pointed
//!   at, and a cancellation flag. A future plugin registry lives at this layer.
//!
//! * [`Benchmark`] specialises it into a **drivable state machine**. The
//!   implementor owns its phase state and does one step per [`Benchmark::next`];
//!   [`Benchmark::run`] (implemented here) drives that in a loop and streams
//!   [`BenchmarkResult`] frames. A benchmark must never block the runtime and
//!   never loop internally to completion — the pane would freeze and cancel
//!   would stop working.
//!
//! Because `impl Stream` and `async fn` in a trait are not dyn-compatible,
//! [`DynBenchmark`] mirrors the contract with boxed futures and is blanket-
//! implemented for every `Benchmark`. That is what [`registry`] stores and what
//! [`executor::drive`] — the single driver loop — consumes.
//!
//! Module map (CI caps every `.rs` at 500 LoC):
//!   params      ParamSpec/ParamValues — the schema the pane renders pre-run
//!   result      BenchmarkResult and the style-free presentation types
//!   plugin      Plugin, PluginHandle, PluginEvent, TargetEndpoint
//!   metadata    PluginMetadata — authorship/provenance, shown before a run
//!   benchmark   Benchmark + BenchmarkDescriptor
//!   dynamic     the dyn-safe bridge
//!   executor    drive() + BenchmarkExecutor (tokio task -> render thread)
//!   registry    the suite, in list order
//!   artifacts   ~/.atlas layout, asset writing, provisioning stamps
//!   python      python/venv/pip preflight for the one benchmark that needs it
//!   http        minimal OpenAI chat client (SSE) — no TLS, no client stack
//!   benchmarks/ the implementations

pub mod artifacts;
pub mod benchmark;
pub mod benchmarks;
pub mod coherence;
pub mod dynamic;
pub mod executor;
pub mod gate;
pub mod hardware;
pub mod headless;
pub mod history;
pub mod http;
pub mod metadata;
pub mod param_text;
pub mod params;
pub mod plugin;
pub mod python;
pub mod registry;
pub mod result;

pub use artifacts::ArtifactStore;
pub use benchmark::{Benchmark, BenchmarkDescriptor};
pub use coherence::CoherencePolicy;
pub use dynamic::DynBenchmark;
pub use executor::{BenchmarkExecutor, ExecutorMessage, RunHandle};
pub use history::{RunRecord, RunSource};
pub use metadata::PluginMetadata;
pub use params::{ParamKind, ParamSpec, ParamValue, ParamValues};
pub use plugin::{Plugin, PluginEvent, PluginHandle, TargetEndpoint};
pub use result::{
    Align, BenchmarkResult, Cell, CellStyle, Column, LogLevel, LogLine, ResultTable, RunStatus,
    Stat, Verdict, VerdictKind,
};
