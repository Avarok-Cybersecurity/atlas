// SPDX-License-Identifier: AGPL-3.0-only

//! The seam the serve matrix boots models through.
//!
//! This crate must stay GPU-free and server-free (see `Cargo.toml`), and
//! booting a checkpoint is the one thing the serve matrix cannot do without a
//! server. So it does not: the host supplies a [`ServeHost`], the benchmark
//! drives it, and every unit test drives a fake one.
//!
//! Two things the implementation owes the benchmark, and both are load-bearing:
//!
//! * [`ServeHost::roster`] is **derived, never listed**. The caller reads what
//!   the box actually has — cached checkpoints intersected with the kernels
//!   this build compiled — so the matrix cannot drift from the box the way a
//!   second hardcoded roster does.
//! * [`ServeHost::serve`] must not return until the endpoint **answers**. A log
//!   line saying a server started is not the same claim, and the Python
//!   orchestrator's readiness check was exactly that log-line grep — one that
//!   matched `Listening on 127.0.0.1:8888` from a server bound inside a network
//!   namespace nothing could reach, turning "unreachable" into a green READY
//!   that read like a model regression.

use std::sync::Arc;

use anyhow::Result;
use futures::future::BoxFuture;
use parking_lot::RwLock;

use crate::plugin::TargetEndpoint;

/// Why a cached checkpoint cannot take part in the matrix.
///
/// Kept distinct from a boot failure on purpose: this is "the box never had
/// what it takes to try", which is a legitimate skip, while a boot failure is
/// a FAIL. Collapsing the two is how a crashed model becomes invisible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Absence {
    /// A cache entry exists but the weights are not all on disk.
    NoWeights,
    /// Downloaded, but its `config.json` could not be read or parsed — so the
    /// architecture is UNKNOWN. Distinct from [`Absence::NoKernels`] on
    /// purpose: "we have no kernels for this architecture" is a claim about
    /// the architecture, and making it from a config nothing could read is a
    /// guess dressed as a finding.
    NoConfig,
    /// Downloaded and parsed, and this build compiled no kernels for it.
    NoKernels,
}

impl Absence {
    pub fn reason(self) -> &'static str {
        match self {
            Absence::NoWeights => "weights not fully downloaded",
            Absence::NoConfig => "config.json unreadable — architecture unknown",
            Absence::NoKernels => "no compiled kernels for this architecture",
        }
    }
}

/// One checkpoint the box knows about. A quant is a separate checkpoint, so
/// this IS the model×quant axis — derived from the cache, not enumerated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServeCandidate {
    /// HF id, `org/name`.
    pub model: String,
    /// Quantization as the checkpoint's own `config.json` declares it.
    pub quant: String,
    /// `None` means runnable right now.
    pub absent: Option<Absence>,
}

impl ServeCandidate {
    pub fn ready(model: impl Into<String>, quant: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            quant: quant.into(),
            absent: None,
        }
    }

    pub fn absent(model: impl Into<String>, quant: impl Into<String>, why: Absence) -> Self {
        Self {
            model: model.into(),
            quant: quant.into(),
            absent: Some(why),
        }
    }
}

/// How one round is served. Every field is explicit — the matrix never relies
/// on a flag it did not state, because the model's own defaults are what the
/// round is supposed to be measuring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServeOptions {
    pub max_seq_len: usize,
    /// Add the MTP arm. Off by default: a checkpoint with no MTP head silently
    /// decodes single-token and reports the baseline's numbers under a
    /// "+MTP" label, which is a fabricated row.
    pub speculative: bool,
}

/// Bringing checkpoints up and down, supplied by whoever owns the server.
pub trait ServeHost: Send + Sync {
    /// What the box can serve, derived from the box.
    fn roster(&self) -> Result<Vec<ServeCandidate>>;

    /// Bring `model` up and return the endpoint **once it answers**.
    fn serve(&self, model: &str, opts: ServeOptions) -> BoxFuture<'_, Result<TargetEndpoint>>;

    /// Put back whatever was serving before the matrix started. Runs on every
    /// exit path, so a cancelled run does not leave the box on round seven.
    fn restore(&self) -> BoxFuture<'_, Result<()>>;
}

/// The installed host.
///
/// STATIC, DELIBERATELY — process lifecycle, exactly like `tui::THREAD`. There
/// is one server per process and the benchmark registry builds plugins from a
/// `fn()` pointer with no context to thread a handle through. Nothing
/// model-derived is stored: it holds the seam, and the seam asks the server
/// fresh every time.
static HOST: RwLock<Option<Arc<dyn ServeHost>>> = RwLock::new(None);

/// Called once by the host as it starts. A second call replaces the first,
/// which is what makes a restart of the dashboard sound.
pub fn install(host: Arc<dyn ServeHost>) {
    *HOST.write() = Some(host);
}

pub fn installed() -> Option<Arc<dyn ServeHost>> {
    HOST.read().clone()
}

/// What to tell the operator when nothing is installed.
///
/// `Plugin::load` returns this, so it lands where the Start button would be —
/// which is why it names the thing that is missing rather than saying no.
pub const NO_HOST: &str = "the serve matrix needs the Atlas server that hosts this dashboard: it \
                           boots each checkpoint in-process. Run it from `spark serve`'s \
                           dashboard rather than a standalone harness.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_absence_reads_differently_because_each_needs_a_different_fix() {
        assert_eq!(
            [
                (Absence::NoWeights, Absence::NoWeights.reason()),
                (Absence::NoConfig, Absence::NoConfig.reason()),
                (Absence::NoKernels, Absence::NoKernels.reason()),
            ],
            [
                (Absence::NoWeights, "weights not fully downloaded"),
                (
                    Absence::NoConfig,
                    "config.json unreadable — architecture unknown",
                ),
                (
                    Absence::NoKernels,
                    "no compiled kernels for this architecture",
                ),
            ]
        );
    }

    #[test]
    fn a_ready_candidate_is_distinguishable_from_an_absent_one() {
        assert_eq!(
            ServeCandidate::ready("org/ready", "nvfp4"),
            ServeCandidate {
                model: "org/ready".into(),
                quant: "nvfp4".into(),
                absent: None,
            }
        );
        assert_eq!(
            ServeCandidate::absent("org/absent", "fp8", Absence::NoWeights),
            ServeCandidate {
                model: "org/absent".into(),
                quant: "fp8".into(),
                absent: Some(Absence::NoWeights),
            }
        );
    }
}
