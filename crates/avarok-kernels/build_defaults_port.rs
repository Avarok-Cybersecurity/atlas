// SPDX-License-Identifier: AGPL-3.0-only
//
// Responsibility: the port sprint's (`port/consumer-opt`) rows of
// `kernels/<hw>/HARDWARE.toml` `[defaults]`, parsed at build time and baked
// into the `port` field of `avarok_kernels::TARGET_DEFAULTS`.
//
// Declared INSIDE `build_defaults.rs` (`#[path = "build_defaults_port.rs"]`),
// which resolves to this file in both crates that compile it: the build script
// (`build.rs`) and the integration tests (`tests/target_defaults*.rs`). So
// there is still exactly one parser, and the tests grade it against the real
// `kernels/` tree.
//
// Its own file so the sprint's rows never touch `build_defaults.rs`: rows
// arrive through the coordinator's fragment apply pass, which inserts text
// immediately BEFORE each `@port:` marker line. A key that is not a port row
// returns `false` from `parse`, and `parse_defaults`' unknown-key panic fires
// exactly as before.

/// The port rows of one target's `[defaults]` table, owned (build-time shape).
///
/// Mirrors `avarok_kernels::target_defaults_port::PortDefaults` field for
/// field, with `String` where the runtime type has `&'static str`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortFields {
    pub kv_batched_swap: bool,
    pub ssm_store_lru: bool,
    pub radix_lru_index: bool,
    pub prefix_cow_tail: bool,
    // @port:struct
}

/// What a target that declares no port row gets: every row's baseline.
///
/// NOT derived: a str row's baseline is a word (`"off"`, `"id"`) and a u32
/// row's is its declared number, neither of which `#[derive(Default)]` would
/// produce. Agrees with `PORT_BASELINE` row for row.
#[allow(clippy::derivable_impls)]
impl Default for PortFields {
    fn default() -> Self {
        Self {
            kv_batched_swap: false,
            ssm_store_lru: false,
            radix_lru_index: false,
            prefix_cow_tail: false,
            // @port:default
        }
    }
}

/// Parse one `[defaults]` key if it is a port row: write it into `out` and
/// return `true`. `false` hands the key back to `parse_defaults`, so an unknown
/// key still panics there.
#[allow(unused_variables, clippy::match_single_binding)]
pub fn parse(out: &mut PortFields, key: &str, value: &toml::Value) -> bool {
    match key {
        "kv_batched_swap" => {
            out.kv_batched_swap = boolean(key, value);
            true
        }
        "ssm_store_lru" => {
            out.ssm_store_lru = boolean(key, value);
            true
        }
        "radix_lru_index" => {
            out.radix_lru_index = boolean(key, value);
            true
        }
        "prefix_cow_tail" => {
            out.prefix_cow_tail = boolean(key, value);
            true
        }
        // @port:parse
        _ => false,
    }
}

/// The `crate::target_defaults_port::PortDefaults { .. }` initializer that
/// `build_defaults::literal` places in the generated constant's `port` field.
///
/// The `@port:literal-fmt` marker sits INSIDE the format string on purpose:
/// row lines are inserted before it, and the marker itself is emitted as a
/// comment line into the generated code, where it is inert.
#[allow(clippy::useless_format, unused_variables)]
pub fn literal(d: &PortFields) -> String {
    format!(
        "crate::target_defaults_port::PortDefaults {{\n\
         \x20   kv_batched_swap: {kv_batched_swap},\n\
         \x20   ssm_store_lru: {ssm_store_lru},\n\
         \x20   radix_lru_index: {radix_lru_index},\n\
         \x20   prefix_cow_tail: {prefix_cow_tail},\n\
         // @port:literal-fmt\n\
         \x20   }}",
        kv_batched_swap = d.kv_batched_swap,
        ssm_store_lru = d.ssm_store_lru,
        radix_lru_index = d.radix_lru_index,
        prefix_cow_tail = d.prefix_cow_tail,
        // @port:literal-args
    )
}

/// A bool row's value. Panics on any other TOML type, for the reason
/// `parse_defaults` panics on an unknown key: a wrong type must not read as
/// agreement with the baseline.
#[allow(dead_code)]
fn boolean(key: &str, v: &toml::Value) -> bool {
    v.as_bool().unwrap_or_else(|| {
        panic!("HARDWARE.toml: [defaults] {key} must be a bool (a port sprint row)")
    })
}

/// A u32 row's value. Panics on a non-integer or out-of-range value.
#[allow(dead_code)]
fn integer(key: &str, v: &toml::Value) -> u32 {
    let n = v.as_integer().unwrap_or_else(|| {
        panic!("HARDWARE.toml: [defaults] {key} must be an integer (a port sprint row)")
    });
    u32::try_from(n).unwrap_or_else(|_| {
        panic!("HARDWARE.toml: [defaults] {key} = {n} is not a u32 (a port sprint row)")
    })
}

/// A str row's value. Panics on any other TOML type.
#[allow(dead_code)]
fn string(key: &str, v: &toml::Value) -> String {
    v.as_str()
        .unwrap_or_else(|| {
            panic!("HARDWARE.toml: [defaults] {key} must be a string (a port sprint row)")
        })
        .to_string()
}
