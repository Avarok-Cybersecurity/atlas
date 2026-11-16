// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the host KV tier's serve flags (lane L26b): `--kv-host-tier-gb` (0 = off),
//! `--kv-tier-min-tokens` (256), `--kv-tier-writeback-mb-per-step` (32), `--kv-tier-policy
//! {lru,arc,heat}` and `--kv-tier-store {eager,lazy}`. `port_apply` calls [`validate_flags`]
//! and then [`apply_flags`], which publishes the [`TierConfig`], the store mode and the
//! resolved `kv_host_tier` row to `spark_runtime::kv_host_tier::config` (spark-runtime never
//! reads a row). `serve_flags` then calls [`check_prefix_cache`] with the RESOLVED prefix-cache
//! value (`--hermetic` closes the prefix cache through the serve resolvers), because a tier
//! without a prefix cache can neither store nor restore anything.
//!
//! Also here: the byte bound of the SSM spill tier's host-RAM store while the KV tier serves
//! ([`ssm_store_cap_bytes`], read by `ssm_tier/selectors.rs`).

use anyhow::{Result, bail};
use spark_runtime::kv_host_tier::config::{self as tier_config, DEFAULT_MIN_TOKENS};
use spark_runtime::kv_host_tier::{StoreMode, TierConfig, TierPolicy};

/// Host RAM the tier must leave to the rest of the system.
pub const HOST_RESERVE_BYTES: u64 = 8 << 30;
/// While the KV tier serves, the SSM spill tier's host-RAM store is bounded to this fraction
/// (1 / N) of the KV tier budget, on top of it.
pub const SSM_STORE_SHARE_DIV: u64 = 4;

/// The tier configuration and store mode the flags ask for.
pub fn config_from_flags(
    host_tier_gb: Option<u32>,
    min_tokens: Option<u32>,
    writeback_mb_per_step: Option<u32>,
    policy: Option<&str>,
    store: Option<&str>,
) -> Result<(TierConfig, StoreMode)> {
    let mut cfg = TierConfig::with_budget_gib(host_tier_gb.unwrap_or(0) as u64);
    cfg.min_tokens = min_tokens.unwrap_or(DEFAULT_MIN_TOKENS);
    if let Some(mb) = writeback_mb_per_step {
        if mb == 0 {
            bail!("--kv-tier-writeback-mb-per-step must be at least 1");
        }
        cfg.writeback_bytes_per_step = (mb as u64) << 20;
    }
    if let Some(p) = policy {
        cfg.policy = p
            .parse::<TierPolicy>()
            .map_err(|e| anyhow::anyhow!("--kv-tier-policy: {e:#}"))?;
    }
    let mode = match store {
        Some(s) => s
            .parse::<StoreMode>()
            .map_err(|e| anyhow::anyhow!("--kv-tier-store: {e:#}"))?,
        None => StoreMode::Eager,
    };
    Ok((cfg, mode))
}

/// Startup check of the flags: parseable values, the tier's own limits, and a budget that
/// leaves [`HOST_RESERVE_BYTES`] of host RAM (plus the SSM store share when `AVAROK_SSM_TIER`
/// spills snapshots to host RAM too).
pub fn validate_flags(
    host_tier_gb: Option<u32>,
    min_tokens: Option<u32>,
    writeback_mb_per_step: Option<u32>,
    policy: Option<&str>,
    store: Option<&str>,
) -> Result<()> {
    let (cfg, _) = config_from_flags(
        host_tier_gb,
        min_tokens,
        writeback_mb_per_step,
        policy,
        store,
    )?;
    tier_config::validate(&cfg)?;
    if cfg.has_budget() {
        let ssm = crate::model::ssm_tier::ssm_tier_enabled();
        check_host_ram(cfg.budget_bytes, ssm, host_ram_bytes())?;
    }
    Ok(())
}

/// Publish the flags and the resolved `kv_host_tier` row.
pub fn apply_flags(
    host_tier_gb: Option<u32>,
    min_tokens: Option<u32>,
    writeback_mb_per_step: Option<u32>,
    policy: Option<&str>,
    store: Option<&str>,
) -> Result<()> {
    let (cfg, mode) = config_from_flags(
        host_tier_gb,
        min_tokens,
        writeback_mb_per_step,
        policy,
        store,
    )?;
    let row = crate::layers::ops::port_rows::kv_host_tier::row();
    tier_config::set(cfg);
    tier_config::set_store_mode(mode);
    tier_config::set_enabled(row);
    match (row, cfg.has_budget()) {
        (true, true) => tracing::info!(
            "kv_host_tier: on ({} GiB, policy {}, store {mode}, min {} tokens, write-back {} MiB/step)",
            cfg.budget_bytes >> 30,
            cfg.policy,
            cfg.min_tokens,
            cfg.writeback_bytes_per_step >> 20,
        ),
        (true, false) => tracing::info!(
            "kv_host_tier: the row is on but --kv-host-tier-gb is 0, so the host KV tier is off"
        ),
        (false, true) => tracing::warn!(
            "kv_host_tier: --kv-host-tier-gb is set but the kv_host_tier row is off \
             (AVAROK_KV_HOST_TIER=1 turns it on); the host KV tier is off"
        ),
        (false, false) => {}
    }
    Ok(())
}

/// Refuse a serving tier with the prefix cache off (call with the RESOLVED value, after
/// [`apply_flags`]).
pub fn check_prefix_cache(prefix_cache_on: bool) -> Result<()> {
    refuse_without_prefix_cache(tier_config::active().is_some(), prefix_cache_on)
}

fn refuse_without_prefix_cache(tier_on: bool, prefix_cache_on: bool) -> Result<()> {
    if tier_on && !prefix_cache_on {
        bail!(
            "the host KV tier (kv_host_tier row on, --kv-host-tier-gb > 0) needs the prefix \
             cache, which is off for this serve (--enable-prefix-caching not given, or closed by \
             --hermetic); turn the prefix cache on or drop --kv-host-tier-gb"
        );
    }
    Ok(())
}

/// The budget (plus the SSM store share when `ssm_spill` is on) must fit `host` minus the
/// reserve. An unknown host size (`None`, non-Linux) is not refused.
pub(crate) fn check_host_ram(budget: u64, ssm_spill: bool, host: Option<u64>) -> Result<()> {
    let Some(host) = host else {
        tracing::warn!("kv_host_tier: host RAM size unknown; the budget is not checked");
        return Ok(());
    };
    let need = if ssm_spill {
        budget + budget / SSM_STORE_SHARE_DIV
    } else {
        budget
    };
    let room = host.saturating_sub(HOST_RESERVE_BYTES);
    if need > room {
        bail!(
            "--kv-host-tier-gb {} GiB{} does not fit this host: {:.1} GiB of RAM minus the \
             {} GiB reserve leaves {:.1} GiB",
            budget >> 30,
            if ssm_spill {
                " (plus the SSM spill store's quarter, AVAROK_SSM_TIER)"
            } else {
                ""
            },
            host as f64 / (1u64 << 30) as f64,
            HOST_RESERVE_BYTES >> 30,
            room as f64 / (1u64 << 30) as f64,
        );
    }
    Ok(())
}

/// Total host RAM (`MemTotal` of `/proc/meminfo`), or `None` where it cannot be read.
pub(crate) fn host_ram_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    parse_mem_total(&text)
}

/// `MemTotal:  65536000 kB` -> bytes.
pub(crate) fn parse_mem_total(meminfo: &str) -> Option<u64> {
    let line = meminfo.lines().find(|l| l.starts_with("MemTotal:"))?;
    let mut it = line["MemTotal:".len()..].split_whitespace();
    let n: u64 = it.next()?.parse().ok()?;
    match it.next() {
        Some("kB") | Some("KB") => n.checked_mul(1024),
        None => Some(n),
        Some(_) => None,
    }
}

/// The byte bound of the SSM spill tier's default host-RAM store while the KV tier serves
/// (row on and a budget), or `None` (the store stays as it was: unbounded).
pub fn ssm_store_cap_bytes() -> Option<usize> {
    ssm_store_cap_for(tier_config::active())
}

pub(crate) fn ssm_store_cap_for(cfg: Option<TierConfig>) -> Option<usize> {
    let cfg = cfg?;
    let cap = cfg.budget_bytes / SSM_STORE_SHARE_DIV;
    usize::try_from(cap).ok().filter(|&c| c > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_plan_values() {
        let (cfg, mode) = config_from_flags(None, None, None, None, None).unwrap();
        assert_eq!(cfg.budget_bytes, 0, "no budget = off");
        assert_eq!(cfg.min_tokens, 256);
        assert_eq!(cfg.writeback_bytes_per_step, 32 << 20);
        assert_eq!(cfg.policy, TierPolicy::Lru);
        assert_eq!(mode, StoreMode::Eager);
    }

    #[test]
    fn flags_reach_the_config() {
        let (cfg, mode) =
            config_from_flags(Some(24), Some(512), Some(64), Some("arc"), Some("lazy")).unwrap();
        assert_eq!(cfg.budget_bytes, 24 << 30);
        assert_eq!(cfg.min_tokens, 512);
        assert_eq!(cfg.writeback_bytes_per_step, 64 << 20);
        assert_eq!(cfg.policy, TierPolicy::Arc);
        assert_eq!(mode, StoreMode::Lazy);
        let (cfg, _) = config_from_flags(None, None, None, Some(" HEAT "), None).unwrap();
        assert_eq!(cfg.policy, TierPolicy::Heat);
    }

    #[test]
    fn bad_values_are_startup_errors() {
        assert!(config_from_flags(None, None, None, Some("mru"), None).is_err());
        assert!(config_from_flags(None, None, None, None, Some("sometimes")).is_err());
        assert!(config_from_flags(Some(1), None, Some(0), None, None).is_err());
    }

    #[test]
    fn the_budget_must_leave_the_reserve() {
        let gib = 1u64 << 30;
        // 64 GiB host: 56 GiB of room.
        assert!(check_host_ram(56 * gib, false, Some(64 * gib)).is_ok());
        assert!(check_host_ram(57 * gib, false, Some(64 * gib)).is_err());
        // With the SSM spill store: 44 GiB + 11 GiB = 55 GiB fits, 48 + 12 = 60 does not.
        assert!(check_host_ram(44 * gib, true, Some(64 * gib)).is_ok());
        assert!(check_host_ram(48 * gib, true, Some(64 * gib)).is_err());
        // Unknown host size: not refused.
        assert!(check_host_ram(1 << 50, true, None).is_ok());
        // A host smaller than the reserve leaves no room at all.
        assert!(check_host_ram(gib, false, Some(4 * gib)).is_err());
    }

    #[test]
    fn meminfo_is_parsed_in_bytes() {
        let text = "MemFree:  100 kB\nMemTotal:       65536000 kB\nSwapTotal: 0 kB\n";
        assert_eq!(parse_mem_total(text), Some(65_536_000 * 1024));
        assert_eq!(parse_mem_total("MemTotal: 4096\n"), Some(4096));
        assert_eq!(parse_mem_total("MemTotal: x kB\n"), None);
        assert_eq!(parse_mem_total("MemTotal: 1 MB\n"), None);
        assert_eq!(parse_mem_total("nothing here"), None);
    }

    #[test]
    fn the_tier_refuses_a_closed_prefix_cache() {
        assert!(refuse_without_prefix_cache(true, false).is_err());
        assert!(refuse_without_prefix_cache(true, true).is_ok());
        assert!(refuse_without_prefix_cache(false, false).is_ok());
    }

    #[test]
    fn the_ssm_store_cap_is_a_quarter_of_a_serving_budget() {
        assert_eq!(ssm_store_cap_for(None), None);
        let cfg = TierConfig::with_budget_gib(8);
        assert_eq!(ssm_store_cap_for(Some(cfg)), Some(2 << 30));
        assert_eq!(ssm_store_cap_for(Some(TierConfig::default())), None);
    }
}
