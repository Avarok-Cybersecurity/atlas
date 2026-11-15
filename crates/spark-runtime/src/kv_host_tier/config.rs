// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache lmcache/v1/distributed/storage_controllers/eviction_controller.py and lmcache/integration/vllm/lazy_offload_policy/eviction_aware.py (commit 3aefae1, Apache-2.0); no code copied.

//! Responsibility: the host tier's configuration: the contract's [`TierConfig`] (what the serve
//! flags set), the [`TierOptions`] tunables (watermarks, in-flight cap, lease TTL, lazy-store and
//! cost-model knobs, all with their documented defaults) and the process-wide setters that
//! `port_apply` feeds (crates below spark-model never read rows; the glue resolves
//! the `kv_host_tier` row and passes plain values down through here).

use std::fmt;
use std::str::FromStr;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Result, bail};

/// Host-tier eviction policy (`--kv-tier-policy`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TierPolicy {
    /// Least recently used; a request's later blocks leave first (prefix order).
    #[default]
    Lru,
    /// Adaptive replacement (T1/T2 with ghost lists), scan resistant.
    Arc,
    /// Access frequency halved every N accesses, recency as the tie-break (KVMem's heat idea).
    Heat,
}

impl TierPolicy {
    pub const ALL: [TierPolicy; 3] = [TierPolicy::Lru, TierPolicy::Arc, TierPolicy::Heat];

    pub fn as_str(self) -> &'static str {
        match self {
            TierPolicy::Lru => "lru",
            TierPolicy::Arc => "arc",
            TierPolicy::Heat => "heat",
        }
    }
}

impl FromStr for TierPolicy {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "lru" => Ok(TierPolicy::Lru),
            "arc" => Ok(TierPolicy::Arc),
            "heat" => Ok(TierPolicy::Heat),
            other => bail!("unknown KV tier policy {other:?} (expected lru, arc or heat)"),
        }
    }
}

impl fmt::Display for TierPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// When newly cached blocks are copied to host RAM (`--kv-tier-store`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum StoreMode {
    /// Copy every newly cached full block right away (capped per step).
    #[default]
    Eager,
    /// Buffer the store and copy only when the blocks approach the GPU eviction head.
    Lazy,
}

impl StoreMode {
    pub fn as_str(self) -> &'static str {
        match self {
            StoreMode::Eager => "eager",
            StoreMode::Lazy => "lazy",
        }
    }
}

impl FromStr for StoreMode {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "eager" => Ok(StoreMode::Eager),
            "lazy" => Ok(StoreMode::Lazy),
            other => bail!("unknown KV tier store mode {other:?} (expected eager or lazy)"),
        }
    }
}

impl fmt::Display for StoreMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// `--kv-tier-writeback-mb-per-step` default.
pub const DEFAULT_WRITEBACK_BYTES_PER_STEP: u64 = 32 << 20;
/// `--kv-tier-min-tokens` default (the SSM fault gate's value).
pub const DEFAULT_MIN_TOKENS: u32 = 256;
/// Restore only when the copy is estimated under this fraction of the recompute.
pub const DEFAULT_RESTORE_GATE_RATIO: f32 = 0.5;

/// The tier's serve-time configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TierConfig {
    /// Host bytes the tier may hold (`--kv-host-tier-gb`; 0 = off).
    pub budget_bytes: u64,
    /// D2H bytes one scheduler step may start.
    pub writeback_bytes_per_step: u64,
    /// The gate declines restores shorter than this many tokens.
    pub min_tokens: u32,
    pub policy: TierPolicy,
    /// Restore when estimated copy time < ratio x estimated prefill time.
    pub restore_gate_ratio: f32,
}

impl Default for TierConfig {
    fn default() -> Self {
        Self {
            budget_bytes: 0,
            writeback_bytes_per_step: DEFAULT_WRITEBACK_BYTES_PER_STEP,
            min_tokens: DEFAULT_MIN_TOKENS,
            policy: TierPolicy::Lru,
            restore_gate_ratio: DEFAULT_RESTORE_GATE_RATIO,
        }
    }
}

impl TierConfig {
    /// The defaults with a budget of `gib` GiB.
    pub fn with_budget_gib(gib: u64) -> Self {
        Self {
            budget_bytes: gib << 30,
            ..Self::default()
        }
    }

    /// A budget was given.
    pub fn has_budget(&self) -> bool {
        self.budget_bytes > 0
    }
}

/// Intrinsic checks of a configuration (the glue adds the host-RAM and prefix-cache checks).
pub fn validate(cfg: &TierConfig) -> Result<()> {
    if !cfg.restore_gate_ratio.is_finite() || cfg.restore_gate_ratio <= 0.0 {
        bail!(
            "KV tier restore gate ratio must be a positive number, got {}",
            cfg.restore_gate_ratio
        );
    }
    if cfg.restore_gate_ratio > 100.0 {
        bail!(
            "KV tier restore gate ratio {} is above 100",
            cfg.restore_gate_ratio
        );
    }
    if cfg.budget_bytes > 0 && cfg.budget_bytes < (1 << 20) {
        bail!(
            "KV host tier budget of {} bytes is below 1 MiB; give 0 to turn the tier off",
            cfg.budget_bytes
        );
    }
    if cfg.budget_bytes > 0 && cfg.writeback_bytes_per_step == 0 {
        bail!("KV tier write-back budget per step must be above 0 when the tier is on");
    }
    Ok(())
}

/// Tunables that are not serve flags.
#[derive(Clone, Debug, PartialEq)]
pub struct TierOptions {
    pub store_mode: StoreMode,
    /// Batch eviction starts when used bytes would pass this fraction of the budget (0.8).
    pub high_watermark: f64,
    /// ... and evicts until used bytes are at (high_watermark - eviction_ratio) (0.2).
    pub eviction_ratio: f64,
    /// Staging (D2H) and leased (H2D) bytes are each capped at this fraction (0.5).
    pub inflight_fraction: f64,
    /// A restore record whose copy completed is reclaimed after this long (30 s).
    pub lease_ttl: Duration,
    /// Lazy store: scheduler steps of block consumption treated as imminent eviction (2.5).
    pub horizon_steps: f64,
    /// Lazy store: at most this many blocks start per drain (256).
    pub max_drain_per_step: usize,
    /// Lazy store: aim to start this multiple of the per-step block demand per drain (2.0).
    pub lazy_target_factor: f64,
    /// Lazy store: a buffered store older than this is started anyway (None = never).
    pub max_deferral: Option<Duration>,
    /// Heat policy: all heat halves every this many accesses (1024).
    pub heat_halve_every: u64,
    /// Cost model: observations kept for the held-out check (32).
    pub cost_window: usize,
    /// Cost model: observations needed before it may gate (16).
    pub cost_min_observations: usize,
    /// Cost model: held-out p95 relative error it must stay under (0.25).
    pub cost_p95_limit: f64,
    /// Cost model: two observations are "materially different" past this ratio (1.5).
    pub cost_material_ratio: f64,
    /// Cost model: EWMA weight of a new rate observation (0.2).
    pub cost_ewma_alpha: f64,
    /// Model fingerprint written into cold-store records (also `KvHostTier::set_namespace`).
    pub namespace: u64,
    /// Register this tier's stats as the process's `/metrics` source.
    pub register_metrics: bool,
}

impl Default for TierOptions {
    fn default() -> Self {
        Self {
            store_mode: StoreMode::Eager,
            high_watermark: 0.8,
            eviction_ratio: 0.2,
            inflight_fraction: 0.5,
            lease_ttl: Duration::from_secs(30),
            horizon_steps: 2.5,
            max_drain_per_step: 256,
            lazy_target_factor: 2.0,
            max_deferral: None,
            heat_halve_every: 1024,
            cost_window: 32,
            cost_min_observations: 16,
            cost_p95_limit: 0.25,
            cost_material_ratio: 1.5,
            cost_ewma_alpha: 0.2,
            namespace: 0,
            register_metrics: true,
        }
    }
}

impl TierOptions {
    /// Defaults plus what `port_apply` published (the store mode).
    pub fn from_globals() -> Self {
        Self {
            store_mode: store_mode(),
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<()> {
        let unit = |name: &str, v: f64| -> Result<()> {
            if !(v.is_finite() && v > 0.0 && v <= 1.0) {
                bail!("KV tier option {name} must be in (0, 1], got {v}");
            }
            Ok(())
        };
        unit("high_watermark", self.high_watermark)?;
        unit("inflight_fraction", self.inflight_fraction)?;
        if !(self.eviction_ratio.is_finite() && self.eviction_ratio >= 0.0) {
            bail!(
                "KV tier option eviction_ratio must be >= 0, got {}",
                self.eviction_ratio
            );
        }
        if !(self.horizon_steps.is_finite() && self.horizon_steps > 0.0) {
            bail!(
                "KV tier option horizon_steps must be > 0, got {}",
                self.horizon_steps
            );
        }
        if self.max_drain_per_step == 0 {
            bail!("KV tier option max_drain_per_step must be >= 1");
        }
        if self.heat_halve_every == 0 {
            bail!("KV tier option heat_halve_every must be >= 1");
        }
        if self.cost_window < 2 || self.cost_min_observations < 2 {
            bail!("KV tier cost model needs a window and a minimum of at least 2 observations");
        }
        Ok(())
    }
}

static CONFIG: OnceLock<TierConfig> = OnceLock::new();
static ENABLED: AtomicBool = AtomicBool::new(false);
static STORE_MODE: OnceLock<StoreMode> = OnceLock::new();

/// Publish the serve-time configuration (called once by `port_apply`). The first call wins; a
/// later different value is ignored with a warning (a hot swap re-runs nothing here).
pub fn set(cfg: TierConfig) {
    if CONFIG.set(cfg).is_err() && CONFIG.get() != Some(&cfg) {
        tracing::warn!(
            "kv_host_tier::config::set called twice with different values; keeping the first"
        );
    }
}

/// The published configuration, if any.
pub fn get() -> Option<TierConfig> {
    CONFIG.get().copied()
}

/// Publish the resolved `kv_host_tier` row (default off).
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Release);
}

/// The resolved row value as published (false until `set_enabled(true)`).
pub fn row_enabled() -> bool {
    ENABLED.load(Ordering::Acquire)
}

/// The configuration to build a tier from: the row is on AND a budget was given.
pub fn active() -> Option<TierConfig> {
    if !row_enabled() {
        return None;
    }
    get().filter(TierConfig::has_budget)
}

/// Publish `--kv-tier-store` (default eager). The first call wins.
pub fn set_store_mode(mode: StoreMode) {
    let _ = STORE_MODE.set(mode);
}

pub fn store_mode() -> StoreMode {
    STORE_MODE.get().copied().unwrap_or_default()
}
