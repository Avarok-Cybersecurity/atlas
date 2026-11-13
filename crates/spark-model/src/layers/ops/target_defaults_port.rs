// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: resolve the port sprint's (`port/consumer-opt`) serving
//! rows, **baked first, environment second**, and print them as one boot line.
//!
//! The rows are the `port` sub-table of [`avarok_kernels::TargetDefaults`]
//! ([`avarok_kernels::target_defaults_port::PortDefaults`]). Each row's
//! resolver lives in its own file, `layers/ops/port_rows/<row>.rs`, owned by
//! the lane that adds the row; this file only collects them. The override
//! grammar and the [`Resolved`] / [`Source`] types are
//! [`super::target_defaults`]'s, re-exported here so a row file needs one
//! import path.
//!
//! [`resolved_port`] is a separate `OnceLock` from
//! [`super::target_defaults::resolved`], with the same environment reader, so
//! the pre-sprint table and its boot line are untouched by the sprint. A lane
//! with a CLI flag publishes it from `spark-server`'s `port_apply::apply`,
//! which runs before any model is built and before this table is first read.
//!
//! Every row is OFF (or its baseline) on every target when it lands, so a
//! binary with an empty environment serves exactly as it did before the row.
//!
//! Rows arrive through the coordinator's fragment apply pass: each `@port:`
//! marker below is an insertion point, and text is inserted immediately BEFORE
//! the marker line.

pub use super::target_defaults::{Resolved, Source};

/// Every port row this target declares, resolved against the environment.
///
/// Field order is the order the boot line prints them in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortLevers {
    /// `kernels/<hw>` this binary was compiled from, for the log line only.
    pub hw: &'static str,
    /// Batched sequence swap: gather a swapped-out sequence's image with batched async copies into one reused page-locked staging buffer.
    pub kv_batched_swap: Resolved<bool>,
    /// SSM tier blob store eviction order: a capped host-RAM store gives up the least recently used blob instead of the oldest insert.
    pub ssm_store_lru: Resolved<bool>,
    /// Radix prefix-cache eviction takes its victim from an ordered (last_access, node id) index of evictable leaves instead of scanning every node.
    pub radix_lru_index: Resolved<bool>,
    /// A prefix-cache hit may end inside a KV block: the chunked prefill copies that block into a private block instead of sharing the donor's writable tail.
    pub prefix_cow_tail: Resolved<bool>,
    // @port:fields
}

/// The port table, as a pure function of the baked declaration and a variable
/// lookup, so a CPU test can resolve it for any target without touching the
/// process environment.
#[allow(unused_variables, unused_mut)]
pub fn resolve(
    defaults: &avarok_kernels::TargetDefaults,
    mut var: impl FnMut(&str) -> Option<String>,
) -> PortLevers {
    let d = &defaults.port;
    PortLevers {
        hw: defaults.hw,
        kv_batched_swap: super::port_rows::kv_batched_swap::resolve(
            d.kv_batched_swap,
            var(super::port_rows::kv_batched_swap::ENV).as_deref(),
        ),
        ssm_store_lru: super::port_rows::ssm_store_lru::resolve(
            d.ssm_store_lru,
            var(super::port_rows::ssm_store_lru::ENV).as_deref(),
        ),
        radix_lru_index: super::port_rows::radix_lru_index::resolve(
            d.radix_lru_index,
            var(super::port_rows::radix_lru_index::ENV).as_deref(),
        ),
        prefix_cow_tail: super::port_rows::prefix_cow_tail::resolve(
            d.prefix_cow_tail,
            var(super::port_rows::prefix_cow_tail::ENV).as_deref(),
        ),
        // @port:resolve
    }
}

/// The process-wide resolution of the port rows.
///
/// `OnceLock`-cached for the reason [`super::target_defaults::resolved`] is:
/// rows are read on hot paths, and a route must stay constant across CUDA-graph
/// replays.
pub fn resolved_port() -> &'static PortLevers {
    static LEVERS: std::sync::OnceLock<PortLevers> = std::sync::OnceLock::new();
    LEVERS.get_or_init(|| {
        resolve(&avarok_kernels::TARGET_DEFAULTS, |name| {
            std::env::var(name).ok()
        })
    })
}

/// `port defaults (<hw>): …`, one line naming every resolved port row and
/// which came from the environment. Logged by `spark-server` right after the
/// `target defaults` line.
pub fn port_boot_line() -> String {
    format_port(resolved_port())
}

/// [`port_boot_line`] over a table the caller already has (pure, so a CPU test
/// can grade the line without sealing the `OnceLock`).
pub fn format_port(l: &PortLevers) -> String {
    let hw = if l.hw.is_empty() { "unknown" } else { l.hw };
    let mut s = format!("port defaults ({hw}):");
    let header = s.len();
    push(&mut s, "kv_batched_swap", &l.kv_batched_swap);
    push(&mut s, "ssm_store_lru", &l.ssm_store_lru);
    push(&mut s, "radix_lru_index", &l.radix_lru_index);
    push(&mut s, "prefix_cow_tail", &l.prefix_cow_tail);
    // @port:boot
    if s.len() == header {
        s.push_str(" none");
    }
    s
}

/// A value a port row can hold, as the boot line prints it.
pub trait PortValue: Copy {
    /// The printed form: `on`/`off`, a number (`max` for `u32::MAX`), or the
    /// string itself.
    fn show(self) -> String;
}

impl PortValue for bool {
    fn show(self) -> String {
        (if self { "on" } else { "off" }).to_string()
    }
}

impl PortValue for u32 {
    fn show(self) -> String {
        // `u32::MAX` is a no-cap baseline, not a chosen bound.
        if self == u32::MAX {
            "max".to_string()
        } else {
            self.to_string()
        }
    }
}

impl PortValue for &'static str {
    fn show(self) -> String {
        self.to_string()
    }
}

/// Append ` <name>=<value>` to the boot line, with ` (env)` when the
/// environment overrode the target's declaration.
pub fn push<T: PortValue>(s: &mut String, name: &str, r: &Resolved<T>) {
    s.push(' ');
    s.push_str(name);
    s.push('=');
    s.push_str(&r.value.show());
    s.push_str(r.source.tag());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_prints_each_value_type_and_its_source() {
        let mut s = String::new();
        let on_env = Resolved {
            value: true,
            source: Source::Env,
        };
        push(&mut s, "a", &on_env);
        let cap = Resolved {
            value: u32::MAX,
            source: Source::Target,
        };
        push(&mut s, "b", &cap);
        let word = Resolved {
            value: "off",
            source: Source::Target,
        };
        push(&mut s, "c", &word);
        assert_eq!(s, " a=on (env) b=max c=off");
    }

    #[test]
    fn the_line_names_its_target() {
        let l = resolve(&avarok_kernels::TARGET_DEFAULTS, |_| None);
        let line = format_port(&l);
        let hw = if l.hw.is_empty() { "unknown" } else { l.hw };
        assert!(
            line.starts_with(&format!("port defaults ({hw}):")),
            "{line}"
        );
    }

    #[test]
    fn an_empty_environment_resolves_every_row_from_the_target() {
        let l = resolve(&avarok_kernels::TARGET_DEFAULTS, |_| None);
        assert!(!format_port(&l).contains(" (env)"));
    }
}
