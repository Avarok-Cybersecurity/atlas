// SPDX-License-Identifier: AGPL-3.0-only

//! Per-identity token-bucket rate limiter.
//!
//! Two independent buckets per key (identity): one metered in **requests**,
//! one in **tokens**. Each bucket refills linearly toward its cap based on
//! elapsed wall time.
//!
//! Identity resolution order (first match wins):
//! 1. `Authorization: Bearer <token>` — the authenticated client's key.
//! 2. First entry of `X-Forwarded-For` — when Atlas sits behind a reverse
//!    proxy or load balancer. Trusted because Atlas is typically deployed
//!    behind a tenant-operated proxy.
//! 3. Socket peer address — fallback for unauthenticated direct calls.
//!
//! Env configuration (all default 0 = disabled, pure passthrough):
//!   ATLAS_RATE_LIMIT_RPM       — requests per minute cap
//!   ATLAS_RATE_LIMIT_TPM       — tokens per minute cap
//!   ATLAS_RATE_LIMIT_BURST_RPM — max request burst (default = RPM)
//!   ATLAS_RATE_LIMIT_BURST_TPM — max token burst   (default = TPM)
//!
//! The limiter keeps the static "effectively unlimited" headers
//! byte-for-byte when both RPM and TPM are 0 so existing deployments see no
//! behavior change.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// Configuration for the limiter. Zero means "disabled" for that bucket.
#[derive(Clone, Copy, Debug)]
pub struct RateLimitConfig {
    pub rpm: u64,
    pub tpm: u64,
    pub burst_rpm: u64,
    pub burst_tpm: u64,
}

impl RateLimitConfig {
    pub fn from_env() -> Self {
        let rpm = read_env_u64("ATLAS_RATE_LIMIT_RPM", 0);
        let tpm = read_env_u64("ATLAS_RATE_LIMIT_TPM", 0);
        let burst_rpm = read_env_u64("ATLAS_RATE_LIMIT_BURST_RPM", rpm);
        let burst_tpm = read_env_u64("ATLAS_RATE_LIMIT_BURST_TPM", tpm);
        Self {
            rpm,
            tpm,
            burst_rpm: burst_rpm.max(1),
            burst_tpm: burst_tpm.max(1),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.rpm > 0 || self.tpm > 0
    }
}

fn read_env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Snapshot of a bucket's remaining budget — used to populate the
/// `x-ratelimit-*-remaining` / `-reset` response headers.
#[derive(Clone, Copy, Debug)]
pub struct BucketSnapshot {
    pub limit: u64,
    pub remaining: u64,
    /// Seconds until this bucket fully refills.
    pub reset_secs: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct RateDecision {
    pub allowed: bool,
    pub requests: BucketSnapshot,
    pub tokens: BucketSnapshot,
    /// Seconds client should wait before retrying (Retry-After header).
    /// Zero when `allowed`.
    pub retry_after_secs: u64,
    /// Which bucket triggered the denial — used for the error message.
    pub denied_by: Option<DenialReason>,
}

#[derive(Clone, Copy, Debug)]
pub enum DenialReason {
    Requests,
    Tokens,
}

/// Per-request context carried from the rate-limit middleware into the
/// handler so the streaming true-up can refund over-estimated tokens
/// once the actual usage is known. Injected into
/// `Request::extensions_mut()` by the middleware when the limiter is
/// enabled; extracted by handlers that want to refund.
#[derive(Clone, Debug)]
pub struct RequestContext {
    pub identity: String,
    /// Tokens reserved at admission time (conservative upper bound).
    pub reserved_tokens: u64,
}

struct Bucket {
    /// Tokens/requests currently available. f64 for fractional refill.
    available: f64,
    /// Last refill tick.
    last_refill: Instant,
}

impl Bucket {
    fn new(burst: u64) -> Self {
        Self {
            available: burst as f64,
            last_refill: Instant::now(),
        }
    }

    /// Refill based on `rate_per_sec`, then try to debit `cost`. Returns
    /// true if the debit succeeded. Caps at `burst`.
    fn try_consume(&mut self, cost: f64, rate_per_sec: f64, burst: f64, now: Instant) -> bool {
        let dt = now
            .saturating_duration_since(self.last_refill)
            .as_secs_f64();
        if dt > 0.0 {
            self.available = (self.available + dt * rate_per_sec).min(burst);
            self.last_refill = now;
        }
        if self.available >= cost {
            self.available -= cost;
            true
        } else {
            false
        }
    }

    fn snapshot(&self, rate_per_sec: f64, burst: f64, now: Instant) -> (f64, u64) {
        let dt = now
            .saturating_duration_since(self.last_refill)
            .as_secs_f64();
        let available = (self.available + dt * rate_per_sec).min(burst);
        // Seconds until fully refilled.
        let deficit = (burst - available).max(0.0);
        let reset = if rate_per_sec > 0.0 {
            (deficit / rate_per_sec).ceil() as u64
        } else {
            0
        };
        (available, reset)
    }

    /// Add tokens back (used when a streaming request admits with a
    /// reservation and actual consumption was lower).
    fn refund(&mut self, amount: f64, burst: f64) {
        self.available = (self.available + amount).min(burst);
    }
}

struct KeyState {
    requests: Bucket,
    tokens: Bucket,
}

/// Shared concurrent rate-limiter state.
pub struct RateLimiter {
    cfg: RateLimitConfig,
    inner: Mutex<HashMap<String, KeyState>>,
    /// Last-scrubbed timestamp for idle-entry cleanup. We sweep once per
    /// `SCRUB_INTERVAL` on the admission hot path.
    last_scrub: Mutex<Instant>,
}

const SCRUB_INTERVAL: Duration = Duration::from_secs(120);
/// Keys with no requests for this long are dropped from the map.
const IDLE_EVICT: Duration = Duration::from_secs(600);
/// Cap the per-key map at this size. Prevents OOM under a DoS where a
/// malicious client rotates the Bearer header every request to balloon
/// the map past `SCRUB_INTERVAL`. When the cap is hit, force an
/// out-of-band scrub regardless of the timer.
const MAX_KEYS: usize = 100_000;

impl RateLimiter {
    pub fn from_env() -> Arc<Self> {
        Arc::new(Self {
            cfg: RateLimitConfig::from_env(),
            inner: Mutex::new(HashMap::new()),
            last_scrub: Mutex::new(Instant::now()),
        })
    }

    #[cfg(test)]
    pub fn with_config(cfg: RateLimitConfig) -> Arc<Self> {
        Arc::new(Self {
            cfg,
            inner: Mutex::new(HashMap::new()),
            last_scrub: Mutex::new(Instant::now()),
        })
    }

    pub fn config(&self) -> RateLimitConfig {
        self.cfg
    }

    /// Admission check for a new request. `estimated_tokens` is the
    /// caller's best guess at how many tokens this call will burn
    /// (prompt + max completion). Streaming callers pass their worst-case
    /// so the reservation is conservative; on completion they call
    /// `refund_tokens` with the true-up.
    pub fn admit(&self, key: &str, estimated_tokens: u64) -> RateDecision {
        let now = Instant::now();
        self.scrub_if_due(now);

        let rpm = self.cfg.rpm;
        let tpm = self.cfg.tpm;

        // When disabled, return effectively-unlimited snapshots.
        if !self.cfg.is_enabled() {
            return RateDecision {
                allowed: true,
                requests: BucketSnapshot {
                    limit: 1_000_000,
                    remaining: 999_999,
                    reset_secs: 0,
                },
                tokens: BucketSnapshot {
                    limit: 1_000_000_000,
                    remaining: 999_999_999,
                    reset_secs: 0,
                },
                retry_after_secs: 0,
                denied_by: None,
            };
        }

        let req_rate = rpm as f64 / 60.0;
        let tok_rate = tpm as f64 / 60.0;
        let req_burst = self.cfg.burst_rpm as f64;
        let tok_burst = self.cfg.burst_tpm as f64;

        let mut map = self.inner.lock();
        // DoS guard: if the map has grown past MAX_KEYS without natural
        // scrubbing kicking in, force-evict every idle entry now. The
        // periodic scrub still runs every SCRUB_INTERVAL on its own.
        if map.len() >= MAX_KEYS && !map.contains_key(key) {
            map.retain(|_, state| {
                state.requests.last_refill.elapsed() < IDLE_EVICT
                    || state.tokens.last_refill.elapsed() < IDLE_EVICT
            });
            // If the cap is still hit (every key is genuinely active),
            // fail-open for the new key — better to admit than to OOM.
            // Real production with this many distinct identities should
            // have an upstream gateway shaping traffic.
        }
        let state = map.entry(key.to_string()).or_insert_with(|| KeyState {
            requests: Bucket::new(self.cfg.burst_rpm),
            tokens: Bucket::new(self.cfg.burst_tpm),
        });

        // Request bucket.
        let req_allowed = if rpm > 0 {
            state.requests.try_consume(1.0, req_rate, req_burst, now)
        } else {
            true
        };
        if !req_allowed {
            let (req_avail, req_reset) = state.requests.snapshot(req_rate, req_burst, now);
            let (tok_avail, tok_reset) = state.tokens.snapshot(tok_rate, tok_burst, now);
            return RateDecision {
                allowed: false,
                requests: BucketSnapshot {
                    limit: self.cfg.burst_rpm,
                    remaining: req_avail.max(0.0) as u64,
                    reset_secs: req_reset,
                },
                tokens: BucketSnapshot {
                    limit: self.cfg.burst_tpm,
                    remaining: tok_avail.max(0.0) as u64,
                    reset_secs: tok_reset,
                },
                retry_after_secs: req_reset.max(1),
                denied_by: Some(DenialReason::Requests),
            };
        }

        // Token bucket. Zero-cost debit when TPM is disabled.
        let tok_allowed = if tpm > 0 {
            state
                .tokens
                .try_consume(estimated_tokens as f64, tok_rate, tok_burst, now)
        } else {
            true
        };
        if !tok_allowed {
            // Refund the request we just consumed — we're going to deny.
            if rpm > 0 {
                state.requests.refund(1.0, req_burst);
            }
            let (req_avail, req_reset) = state.requests.snapshot(req_rate, req_burst, now);
            let (tok_avail, tok_reset) = state.tokens.snapshot(tok_rate, tok_burst, now);
            return RateDecision {
                allowed: false,
                requests: BucketSnapshot {
                    limit: self.cfg.burst_rpm,
                    remaining: req_avail.max(0.0) as u64,
                    reset_secs: req_reset,
                },
                tokens: BucketSnapshot {
                    limit: self.cfg.burst_tpm,
                    remaining: tok_avail.max(0.0) as u64,
                    reset_secs: tok_reset,
                },
                retry_after_secs: tok_reset.max(1),
                denied_by: Some(DenialReason::Tokens),
            };
        }

        let (req_avail, req_reset) = state.requests.snapshot(req_rate, req_burst, now);
        let (tok_avail, tok_reset) = state.tokens.snapshot(tok_rate, tok_burst, now);
        RateDecision {
            allowed: true,
            requests: BucketSnapshot {
                limit: self.cfg.burst_rpm,
                remaining: req_avail.max(0.0) as u64,
                reset_secs: req_reset,
            },
            tokens: BucketSnapshot {
                limit: self.cfg.burst_tpm,
                remaining: tok_avail.max(0.0) as u64,
                reset_secs: tok_reset,
            },
            retry_after_secs: 0,
            denied_by: None,
        }
    }

    /// Called after a streaming request completes with `(reserved - actual)`
    /// so over-estimated reservations don't burn the token bucket forever.
    pub fn refund_tokens(&self, key: &str, amount: u64) {
        if amount == 0 || !self.cfg.is_enabled() || self.cfg.tpm == 0 {
            return;
        }
        let mut map = self.inner.lock();
        if let Some(state) = map.get_mut(key) {
            state
                .tokens
                .refund(amount as f64, self.cfg.burst_tpm as f64);
        }
    }

    fn scrub_if_due(&self, now: Instant) {
        let mut last = self.last_scrub.lock();
        if now.saturating_duration_since(*last) < SCRUB_INTERVAL {
            return;
        }
        *last = now;
        drop(last);
        let mut map = self.inner.lock();
        map.retain(|_, state| {
            state.requests.last_refill.elapsed() < IDLE_EVICT
                || state.tokens.last_refill.elapsed() < IDLE_EVICT
        });
    }
}

/// Resolve a stable identity from the request headers + peer addr. Used by
/// the axum middleware; exposed here so tests can reuse it.
pub fn extract_identity(
    headers: &axum::http::HeaderMap,
    peer: Option<std::net::SocketAddr>,
) -> String {
    use axum::http::header;
    // 1. Bearer token.
    if let Some(v) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        && let Some(tok) = v.strip_prefix("Bearer ")
    {
        let tok = tok.trim();
        if !tok.is_empty() {
            // Hash the token so we don't retain sensitive data in the map
            // keys or Prometheus labels (if ever exposed).
            return format!("bearer:{}", hash_token(tok));
        }
    }
    // 2. X-Forwarded-For.
    if let Some(xff) = headers.get("x-forwarded-for").and_then(|v| v.to_str().ok())
        && let Some(first) = xff.split(',').next()
    {
        let first = first.trim();
        if !first.is_empty() {
            return format!("xff:{first}");
        }
    }
    // 3. Peer socket.
    match peer {
        Some(addr) => format!("peer:{}", addr.ip()),
        None => "peer:unknown".to_string(),
    }
}

/// FNV-1a 64-bit hash. Avoids pulling a crypto dep; unnecessary here since
/// we only need a stable opaque label, not collision resistance.
fn hash_token(tok: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in tok.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

#[cfg(test)]
#[path = "rate_limiter/tests.rs"]
mod tests;
