// SPDX-License-Identifier: AGPL-3.0-only

//! Backwards compatibility shim for the `AVAROK_*` to `AVAROK_*` env rename.
//!
//! The repo-wide rebrand moved every variable the server reads from `AVAROK_*`
//! to `AVAROK_*`, but the CLI that launches the server lives in a different
//! repository and still exports the old names (recipes set things like
//! `AVAROK_MTP_DCUT_RATIO`, `AVAROK_FP8_ROWWISE`, `AVAROK_GDN_FLASHINFER`,
//! `AVAROK_GDN_LIB` and `AVAROK_TARGET_MODEL`; the agent sets `AVAROK_HOME`).
//! Without this shim a server built from the renamed tree ignores all of them
//! silently, which looks exactly like a recipe that does nothing.
//!
//! [`mirror_legacy_env`] copies each legacy name onto its new name at process
//! startup, so the ~440 `std::env::var("AVAROK_...")` reads scattered across
//! the crates need no per-site fallback.
//!
//! REMOVAL POINT: delete this module, its call sites in `main`, and the
//! `AVAROK_SKIP_BUILD` fallback in `avarok-kernels/build.rs` once the CLI ships
//! `AVAROK_*` names and no supported release still emits `AVAROK_*`.

use std::ffi::OsString;

/// The name every legacy variable starts with. Matched as an exact ASCII
/// prefix, so sibling namespaces such as `ATLASCTL_*` are left alone.
const LEGACY_PREFIX: &str = "AVAROK_";

/// The name the renamed tree reads.
const CURRENT_PREFIX: &str = "AVAROK_";

/// Mirror every `AVAROK_*` variable onto its `AVAROK_*` name when the new
/// name is unset. Returns the mirrored keys (new names). Must run before any
/// thread is spawned; call it first thing in `main`.
///
/// An `AVAROK_*` value already present always wins: an operator who sets the
/// new name explicitly is never overridden by a stale legacy export.
///
/// # Example
///
/// ```no_run
/// // First statement of `main`, before any runtime, logging or thread init.
/// let mirrored = avarok_core::env_compat::mirror_legacy_env();
/// if !mirrored.is_empty() {
///     eprintln!("mirrored {} legacy variables", mirrored.len());
/// }
/// ```
pub fn mirror_legacy_env() -> Vec<String> {
    // Snapshot the environment before touching it. `vars_os` walks the live
    // environ block, and `set_var` may reallocate that block, so collecting
    // first keeps the iterator and the mutation strictly separated.
    //
    // Keys must be UTF-8 to be rewritten (every variable in play is ASCII);
    // values stay `OsString` so a non-UTF-8 path is mirrored byte for byte.
    let candidates: Vec<(String, OsString)> = std::env::vars_os()
        .filter_map(|(key, value)| {
            let key = key.into_string().ok()?;
            let rest = key.strip_prefix(LEGACY_PREFIX)?;
            Some((format!("{CURRENT_PREFIX}{rest}"), value))
        })
        .collect();

    let mut mirrored = Vec::new();
    for (current_key, value) in candidates {
        if std::env::var_os(&current_key).is_some() {
            // The new name is already set. Never clobber it.
            continue;
        }
        // SAFETY: `std::env::set_var` is unsafe in edition 2024 because the
        // underlying `setenv` is not thread safe: it can reallocate the
        // environ block while another thread is inside `getenv`, and the C
        // library gives no lock to share. The contract that makes this call
        // sound is that it runs as the very first statement of `main`, before
        // any runtime, logging subscriber, GPU context or `std::thread::spawn`
        // exists, so this thread is the only one that can observe the
        // environment. Callers must not call it from anywhere else.
        unsafe { std::env::set_var(&current_key, &value) };
        mirrored.push(current_key);
    }

    mirrored.sort();
    mirrored
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// These tests mutate the process environment, which is global to the test
    /// binary, and `mirror_legacy_env` reads all of it. Serialize them so one
    /// test's half-built key pair is never visible to another's mirror pass.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// A failed assertion poisons the lock; the data is `()`, so recovering the
    /// guard keeps one failure from cascading into unrelated test failures.
    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Key names are unique per test and carry the pid so a stray variable
    /// inherited from the surrounding shell cannot collide with them.
    fn unique(suffix: &str) -> String {
        format!("ZZ_TEST_{}_{suffix}", std::process::id())
    }

    /// SAFETY: same contract as [`mirror_legacy_env`]. Every test that calls
    /// this holds `ENV_LOCK`, and the crate spawns no threads in its tests, so
    /// no other thread can read the environment concurrently.
    fn set(key: &str, value: &str) {
        unsafe { std::env::set_var(key, value) };
    }

    /// SAFETY: as [`set`] above.
    fn unset(key: &str) {
        unsafe { std::env::remove_var(key) };
    }

    #[test]
    fn legacy_key_is_mirrored_onto_the_new_name() {
        let _guard = env_guard();
        let legacy = format!("AVAROK_{}", unique("MIRRORED"));
        let current = format!("AVAROK_{}", unique("MIRRORED"));
        unset(&current);
        set(&legacy, "recipe-value");

        let mirrored = mirror_legacy_env();

        assert_eq!(
            std::env::var(&current).ok().as_deref(),
            Some("recipe-value"),
            "{legacy} should have been copied onto {current}"
        );
        assert!(
            mirrored.contains(&current),
            "{current} should be reported as mirrored, got {mirrored:?}"
        );

        unset(&legacy);
        unset(&current);
    }

    #[test]
    fn existing_new_name_is_not_overwritten() {
        let _guard = env_guard();
        let legacy = format!("AVAROK_{}", unique("KEPT"));
        let current = format!("AVAROK_{}", unique("KEPT"));
        // Set the winner first so no interleaving can mirror the legacy name
        // before the new name exists.
        set(&current, "explicit-new-value");
        set(&legacy, "stale-legacy-value");

        let mirrored = mirror_legacy_env();

        assert_eq!(
            std::env::var(&current).ok().as_deref(),
            Some("explicit-new-value"),
            "an explicitly set {current} must win over {legacy}"
        );
        assert!(
            !mirrored.contains(&current),
            "{current} was already set and must not be reported, got {mirrored:?}"
        );

        unset(&legacy);
        unset(&current);
    }

    #[test]
    fn atlasctl_prefix_is_left_alone() {
        let _guard = env_guard();
        // `ATLASCTL_` shares the first five letters but is a different
        // namespace: the prefix match is on `AVAROK_`, underscore included.
        let legacy = format!("ATLASCTL_{}", unique("UNTOUCHED"));
        let would_be = format!("ATLASCTL_{}", unique("UNTOUCHED"));
        set(&legacy, "control-plane-value");

        let mirrored = mirror_legacy_env();

        assert!(
            std::env::var_os(&would_be).is_none(),
            "{legacy} must not produce {would_be}"
        );
        assert!(
            !mirrored.iter().any(|key| key.contains("UNTOUCHED")),
            "no ATLASCTL_ key should be mirrored, got {mirrored:?}"
        );

        unset(&legacy);
    }
}
