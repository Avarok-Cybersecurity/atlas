// SPDX-License-Identifier: AGPL-3.0-only

//! What `--hermetic` must do, and — the part that keeps it true — the guard
//! that no production code reads the raw fields it is supposed to override.

use super::{mtp_gate_force, prefix_caching_enabled};

#[test]
fn hermetic_closes_the_prefix_cache_even_when_it_was_asked_for() {
    assert!(
        !prefix_caching_enabled(true, true),
        "--hermetic must close the prefix cache; it is the M2 channel"
    );
}

#[test]
fn without_hermetic_the_prefix_cache_is_exactly_what_was_asked_for() {
    assert!(prefix_caching_enabled(true, false));
    assert!(!prefix_caching_enabled(false, false));
}

#[test]
fn hermetic_forces_the_mtp_gate_rather_than_deferring_to_the_environment() {
    // The subtle one. `None` is not "off" here — it means "no flag was given,
    // so ATLAS_MTP_GATE_FORCE decides". Returning `None` under --hermetic
    // would let the environment reopen the M1 probe channel from outside the
    // recorded regime, while the record still read `hermetic=true`.
    assert_eq!(
        mtp_gate_force(None, true),
        Some(true),
        "--hermetic must PIN the gate, not leave it to the environment"
    );
}

#[test]
fn hermetic_wins_over_a_contradicting_gate_value() {
    // `validate_serve_args` refuses this pair, so it should not reach the
    // resolver. Asserted anyway: a resolver that depends on a validator having
    // run is a resolver that is wrong the first time someone calls it directly.
    assert_eq!(mtp_gate_force(Some("auto"), true), Some(true));
}

#[test]
fn without_hermetic_the_gate_is_exactly_what_was_asked_for() {
    assert_eq!(
        mtp_gate_force(None, false),
        None,
        "absent means env decides"
    );
    assert_eq!(mtp_gate_force(Some("force"), false), Some(true));
    assert_eq!(mtp_gate_force(Some("auto"), false), Some(false));
}

/// THE GUARD THIS MODULE EXISTS FOR.
///
/// `enable_prefix_caching` had FOUR independent production readers before
/// `--hermetic` — and two of them (`logo`, `preflight`) do not act on the
/// value, they ANNOUNCE it. A `--hermetic` wired into `build` but not into
/// `logo` yields a server that runs a KAT while its own banner says the
/// prefix cache is on, and the banner is what an operator reads when a score
/// moves. The fix is not "remember to update all four"; it is this test.
///
/// So: outside this module, the declaration, and the validator, no production
/// source may read the raw fields. Everything reads the resolvers.
#[test]
fn no_production_code_reads_the_raw_fields_behind_hermetic() {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    // hermetic.rs IS the resolver; serve_args.rs DECLARES the fields;
    // validate.rs must compare the raw request to spot the contradiction —
    // resolving there would make every contradiction resolve itself away.
    const ALLOWED: &[&str] = &["cli/hermetic.rs", "cli/serve_args.rs", "cli/validate.rs"];
    let mut offenders = Vec::new();
    let mut scanned = 0usize;
    for entry in walk(&src) {
        let rel = entry
            .strip_prefix(&src)
            .unwrap_or(&entry)
            .to_string_lossy()
            .replace('\\', "/");
        // Tests may read whatever they like; they assert, they do not serve.
        if rel.ends_with("_tests.rs") || ALLOWED.contains(&rel.as_str()) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&entry) else {
            continue;
        };
        scanned += 1;
        for (n, line) in text.lines().enumerate() {
            for field in ["enable_prefix_caching", "mtp_gate"] {
                if reads_field(line, field) {
                    offenders.push(format!("{rel}:{} reads .{field}", n + 1));
                }
            }
        }
    }
    // A scan that visited nothing reports "no offenders" and means nothing.
    // The floor is far below the real count (hundreds) and exists only to
    // make a broken walk fail LOUDLY instead of silently certifying.
    assert!(
        scanned > 50,
        "the scan visited {scanned} files — it is not scanning the tree, so its \
         green means nothing"
    );
    assert!(
        offenders.is_empty(),
        "these read a raw --hermetic-controlled field instead of its resolver \
         (`args.prefix_caching_enabled()` / `args.mtp_gate_force()`), so --hermetic \
         would not reach them:\n  {}",
        offenders.join("\n  ")
    );
}

/// Does `line` contain a field access `.<field>` as a WHOLE token?
///
/// Exact-token, never `contains`. `.mtp_gate` is a prefix of the unrelated
/// `.mtp_gate_force` (the resolved lever the scheduler reads, which is
/// exactly what this rule wants code to use) — a `contains` check would flag
/// the correct call site and, worse, would pass while a typo'd needle matched
/// nothing.
fn reads_field(line: &str, field: &str) -> bool {
    let needle = format!(".{field}");
    let bytes = line.as_bytes();
    let mut from = 0;
    while let Some(i) = line[from..].find(&needle) {
        let start = from + i;
        let end = start + needle.len();
        let next_is_ident = bytes
            .get(end)
            .is_some_and(|c| c.is_ascii_alphanumeric() || *c == b'_');
        // `self.enable_prefix_caching` inside the resolver is the one legal
        // read, and that file is allow-listed; anything else is an offender.
        if !next_is_ident {
            return true;
        }
        from = end;
    }
    false
}

fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(walk(&p));
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
    out
}

/// `reads_field` must match a WHOLE token. This is not hypothetical: the
/// scheduler's correct, resolved read is `sched.levers.mtp_gate_force`, and a
/// `contains(".mtp_gate")` check would flag it as an offender — which would
/// have been "fixed" by loosening the guard until it measured nothing.
#[test]
fn the_scan_matches_whole_tokens_not_prefixes() {
    assert!(reads_field("if args.mtp_gate.is_some() {", "mtp_gate"));
    assert!(!reads_field("if sched.levers.mtp_gate_force {", "mtp_gate"));
    assert!(reads_field(
        "a.enable_prefix_caching,",
        "enable_prefix_caching"
    ));
    assert!(!reads_field(
        "args.prefix_caching_enabled()",
        "enable_prefix_caching"
    ));
}
