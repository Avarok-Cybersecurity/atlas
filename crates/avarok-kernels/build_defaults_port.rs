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
    pub marlin_w4a16: bool,
    pub gguf_native_iq4: bool,
    pub weights_exl3: bool,
    pub dense_affine4_from_bf16: bool,
    pub gdn_affine4_from_bf16: bool,
    pub attn_affine4_from_bf16: bool,
    pub affine4_prefill_int8: bool,
    pub dense_nvfp4_from_bf16: bool,
    pub embed_host_offload: bool,
    pub kv_batched_swap: bool,
    pub ssm_store_lru: bool,
    pub kv_host_tier: bool,
    pub kv_restore_async: bool,
    pub kv_tier_disk: bool,
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
            marlin_w4a16: false,
            gguf_native_iq4: false,
            weights_exl3: false,
            dense_affine4_from_bf16: false,
            gdn_affine4_from_bf16: false,
            attn_affine4_from_bf16: false,
            affine4_prefill_int8: false,
            dense_nvfp4_from_bf16: false,
            embed_host_offload: false,
            kv_batched_swap: false,
            ssm_store_lru: false,
            kv_host_tier: false,
            kv_restore_async: false,
            kv_tier_disk: false,
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
        "marlin_w4a16" => {
            out.marlin_w4a16 = boolean(key, value);
            true
        }
        "gguf_native_iq4" => {
            out.gguf_native_iq4 = boolean(key, value);
            true
        }
        "weights_exl3" => {
            out.weights_exl3 = boolean(key, value);
            true
        }
        "dense_affine4_from_bf16" => {
            out.dense_affine4_from_bf16 = boolean(key, value);
            true
        }
        "gdn_affine4_from_bf16" => {
            out.gdn_affine4_from_bf16 = boolean(key, value);
            true
        }
        "attn_affine4_from_bf16" => {
            out.attn_affine4_from_bf16 = boolean(key, value);
            true
        }
        "affine4_prefill_int8" => {
            out.affine4_prefill_int8 = boolean(key, value);
            true
        }
        "dense_nvfp4_from_bf16" => {
            out.dense_nvfp4_from_bf16 = boolean(key, value);
            true
        }
        "embed_host_offload" => {
            out.embed_host_offload = boolean(key, value);
            true
        }
        "kv_batched_swap" => {
            out.kv_batched_swap = boolean(key, value);
            true
        }
        "ssm_store_lru" => {
            out.ssm_store_lru = boolean(key, value);
            true
        }
        "kv_host_tier" => {
            out.kv_host_tier = boolean(key, value);
            true
        }
        "kv_restore_async" => {
            out.kv_restore_async = boolean(key, value);
            true
        }
        "kv_tier_disk" => {
            out.kv_tier_disk = boolean(key, value);
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
         \x20   marlin_w4a16: {marlin_w4a16},\n\
         \x20   gguf_native_iq4: {gguf_native_iq4},\n\
         \x20   weights_exl3: {weights_exl3},\n\
         \x20   dense_affine4_from_bf16: {dense_affine4_from_bf16},\n\
         \x20   gdn_affine4_from_bf16: {gdn_affine4_from_bf16},\n\
         \x20   attn_affine4_from_bf16: {attn_affine4_from_bf16},\n\
         \x20   affine4_prefill_int8: {affine4_prefill_int8},\n\
         \x20   dense_nvfp4_from_bf16: {dense_nvfp4_from_bf16},\n\
         \x20   embed_host_offload: {embed_host_offload},\n\
         \x20   kv_batched_swap: {kv_batched_swap},\n\
         \x20   ssm_store_lru: {ssm_store_lru},\n\
         \x20   kv_host_tier: {kv_host_tier},\n\
         \x20   kv_restore_async: {kv_restore_async},\n\
         \x20   kv_tier_disk: {kv_tier_disk},\n\
         \x20   radix_lru_index: {radix_lru_index},\n\
         \x20   prefix_cow_tail: {prefix_cow_tail},\n\
         // @port:literal-fmt\n\
         \x20   }}",
        marlin_w4a16 = d.marlin_w4a16,
        gguf_native_iq4 = d.gguf_native_iq4,
        weights_exl3 = d.weights_exl3,
        dense_affine4_from_bf16 = d.dense_affine4_from_bf16,
        gdn_affine4_from_bf16 = d.gdn_affine4_from_bf16,
        attn_affine4_from_bf16 = d.attn_affine4_from_bf16,
        affine4_prefill_int8 = d.affine4_prefill_int8,
        dense_nvfp4_from_bf16 = d.dense_nvfp4_from_bf16,
        embed_host_offload = d.embed_host_offload,
        kv_batched_swap = d.kv_batched_swap,
        ssm_store_lru = d.ssm_store_lru,
        kv_host_tier = d.kv_host_tier,
        kv_restore_async = d.kv_restore_async,
        kv_tier_disk = d.kv_tier_disk,
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
