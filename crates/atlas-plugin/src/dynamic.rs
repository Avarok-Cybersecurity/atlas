// SPDX-License-Identifier: AGPL-3.0-only

//! The dyn-safe bridge.
//!
//! [`Benchmark`] is written for implementors: `async fn` and an `impl Stream`
//! return. Both are return-position-impl-trait, which is **not dyn-compatible**
//! — so a registry cannot hold `Box<dyn Benchmark>` and there is no way to
//! store heterogeneous benchmarks behind one constructor table.
//!
//! [`DynBenchmark`] is the same contract with boxed futures, and every
//! `Benchmark` gets it for free via the blanket impl below. Implementors keep
//! the ergonomic trait; the registry and the executor talk to this one.
//!
//! Adding a method to `Benchmark` means adding it here too — that duplication
//! is the price of the ergonomic trait, and the compiler enforces it, since the
//! blanket impl stops compiling until both agree.

use anyhow::Result;
use futures::future::BoxFuture;

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::metadata::PluginMetadata;
use crate::params::{ParamSpec, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::BenchmarkResult;

/// Object-safe form of [`Benchmark`] + [`Plugin`].
pub trait DynBenchmark: Send {
    fn descriptor(&self) -> &'static BenchmarkDescriptor;
    fn metadata(&self) -> &'static PluginMetadata;
    fn parameters(&self) -> Vec<ParamSpec>;
    fn configure(&mut self, values: &ParamValues) -> Result<()>;
    fn load<'a>(&'a mut self, handle: PluginHandle) -> BoxFuture<'a, Result<()>>;
    fn next<'a>(&'a mut self) -> BoxFuture<'a, Result<BenchmarkResult>>;
    fn cleanup<'a>(&'a mut self) -> BoxFuture<'a, Result<()>>;
}

impl<T> DynBenchmark for T
where
    T: Benchmark + Send,
{
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        Benchmark::descriptor(self)
    }
    fn metadata(&self) -> &'static PluginMetadata {
        Plugin::metadata(self)
    }
    fn parameters(&self) -> Vec<ParamSpec> {
        Benchmark::parameters(self)
    }
    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        Benchmark::configure(self, values)
    }
    fn load<'a>(&'a mut self, handle: PluginHandle) -> BoxFuture<'a, Result<()>> {
        Box::pin(Plugin::load(self, handle))
    }
    fn next<'a>(&'a mut self) -> BoxFuture<'a, Result<BenchmarkResult>> {
        Box::pin(Benchmark::next(self))
    }
    fn cleanup<'a>(&'a mut self) -> BoxFuture<'a, Result<()>> {
        Box::pin(Benchmark::cleanup(self))
    }
}
