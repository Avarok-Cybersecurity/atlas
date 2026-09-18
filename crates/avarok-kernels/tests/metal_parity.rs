// SPDX-License-Identifier: AGPL-3.0-only

//! The 1:1 map between gb10's kernels and Metal's, as a checked artefact.
//!
//! The owner's requirement is that EVERY kernel be mapped 1:1, with any
//! exception written as a comment saying why. That is a claim about two sets,
//! so it needs three things a README cannot provide: both sets resolved by the
//! build's own resolver, a row per pair, and a verdict that goes red when the
//! rows and the sets disagree.
//!
//! Keyed by `(source path, entry point)` and NEVER by the bare name. Measured
//! reason: `kernels/gb10/common/rms_norm.cu:106` computes
//! `xv0*rms*(1.0f + wv0)` for Qwen3-Next's offset-from-1 weights, while
//! `kernels/gb10/gemma-4-26b-a4b/nvfp4/rms_norm.cu:100` computes `xv0*rms*wv0`
//! for Gemma-4's standard ones. Same entry-point name, two different formulas,
//! selected by target. The census found **121 gb10 names declared in more than
//! one source**, so a name-keyed manifest would let one Metal kernel claim to
//! map six different implementations.
//!
//! Regenerate the skeleton with `AVAROK_METAL_PARITY_EMIT=1 cargo test -p
//! avarok-kernels --test metal_parity`. The emitted file is the starting point
//! for hand-editing, not a build artefact: rows carry judgement (tier, test
//! name, mutations, the exception rationale) that no generator can invent.
//!
//! Two sets means two directions. `[[kernel]]` rows constrain the gb10 set;
//! `[[metal]]` rows constrain the Metal set: every resolved Metal entry point
//! is a `mapped` target or is declared `metal_only` (nothing on the CUDA side
//! does its job), `counterpart` (the gb10 kernel in `gb10 =` does, but the
//! contract differs or nothing measures it) or `stub` (empty body, read from
//! source, never mappable). Without it, 73 Metal kernels were invisible.

#[allow(dead_code)]
#[path = "../build_shadow.rs"]
mod build_shadow;

use build_shadow::entry_points;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/avarok-kernels -> repo root")
        .to_path_buf()
}

/// Sources under `dir` with extension `ext`, symlinks skipped.
///
/// gb10 carries 91 symlinks (hopper and b200 inherit its kernel set that way).
/// Following them would attribute one kernel to two paths, and the manifest is
/// keyed by path, so that is a duplicated row rather than a duplicated file.
fn sources(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            let Ok(md) = std::fs::symlink_metadata(&p) else {
                continue;
            };
            if md.file_type().is_symlink() {
                continue;
            }
            if md.is_dir() {
                stack.push(p);
            } else if p.extension().and_then(|s| s.to_str()) == Some(ext) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// `(relative source path, entry point)` for every kernel under `dir`.
fn resolved(root: &Path, dir: &str, ext: &str) -> BTreeSet<(String, String)> {
    let mut out = BTreeSet::new();
    for src in sources(&root.join(dir), ext) {
        let rel = src
            .strip_prefix(root)
            .unwrap_or(&src)
            .to_string_lossy()
            .into_owned();
        for name in entry_points(&src) {
            out.insert((rel.clone(), name));
        }
    }
    out
}

/// One `[[kernel]]` row, parsed. Deliberately a tiny hand parser: pulling a
/// TOML crate into avarok-kernels' dev-dependencies for this would put a
/// parser in the dependency graph of the build's own resolver.
#[derive(Debug, Default)]
struct Row {
    table: String,
    source: String,
    entry: String,
    state: String,
    metal: Option<String>,
    gb10: Option<String>,
    test: Option<String>,
    why: Option<String>,
}

/// The tables the manifest may carry. Anything else is refused: the parser
/// attributes a key to the row above it, so an unknown header folds silently.
const TABLES: &[&str] = &["kernel", "metal"];

fn parse_manifest(text: &str) -> Vec<Row> {
    let mut rows = Vec::new();
    let mut cur: Option<Row> = None;
    for line in text.lines() {
        let t = line.trim();
        if let Some(name) = t.strip_prefix("[[").and_then(|s| s.strip_suffix("]]")) {
            assert!(
                TABLES.contains(&name),
                "{MANIFEST}: unknown table [[{name}]]; its keys would land in the row above"
            );
            if let Some(r) = cur.take() {
                rows.push(r);
            }
            cur = Some(Row {
                table: name.to_string(),
                ..Row::default()
            });
            continue;
        }
        let Some(r) = cur.as_mut() else { continue };
        let Some((k, v)) = t.split_once('=') else {
            continue;
        };
        let k = k.trim();
        let v = v.trim().trim_matches('"').to_string();
        match k {
            "source" => r.source = v,
            "entry" => r.entry = v,
            "state" => r.state = v,
            "metal" => r.metal = Some(v),
            "gb10" => r.gb10 = Some(v),
            "test" => r.test = Some(v),
            "why" => r.why = Some(v),
            _ => {}
        }
    }
    if let Some(r) = cur {
        rows.push(r);
    }
    rows
}

/// Body of the top-level `fn name(` in `src`: from its signature to the first
/// column-0 `}` after it. The test sources are rustfmt'd, so that brace closes
/// the function. `None` when no such function is declared.
fn fn_body<'a>(src: &'a str, name: &str) -> Option<&'a str> {
    let start = src.find(&format!("fn {name}("))?;
    let rest = &src[start..];
    let end = rest.find("\n}\n").map_or(rest.len(), |i| i + 1);
    Some(&rest[..end])
}

const MANIFEST: &str = "kernels/metal/parity/manifest.toml";

/// Placeholders that are not a reason. An exception's `why` has to say what is
/// different about the kernel, not that somebody meant to look later.
const PLACEHOLDERS: &[&str] = &[
    "todo",
    "n/a",
    "na",
    "see above",
    "not needed",
    "later",
    "tbd",
];

/// The one reason standard: gb10-side exceptions and every `[[metal]]` row go
/// through here, so no row kind gets a second, weaker floor.
fn assert_reason(label: &str, kind: &str, why: Option<&str>) {
    let why = why.unwrap_or("");
    assert!(
        why.len() >= 80,
        "{label} is {kind} with a {}-char reason; 80 is the floor, because a \
         one-line reason is how 'no Metal analogue' comes to mean 'nobody tried'",
        why.len()
    );
    let lower = why.to_lowercase();
    for p in PLACEHOLDERS {
        assert!(
            !lower.starts_with(p) && lower != *p,
            "{label} {kind} reason is the placeholder {p:?}"
        );
    }
}

/// True when `entry` in the Metal source `src` is declared with an empty body
/// (`kernel void name(...) {}`), which is how lora_bgmv.metal spells a stub.
/// Read from source, not the manifest, so editing a row cannot promote a stub.
fn empty_body(root: &Path, src: &str, entry: &str) -> bool {
    let text = std::fs::read_to_string(root.join(src)).unwrap_or_default();
    let Some(start) = text.find(&format!("kernel void {entry}(")) else {
        return false;
    };
    let mut depth = 0i32;
    let close = text[start..].char_indices().find(|&(_, c)| {
        depth += (c == '(') as i32 - (c == ')') as i32;
        c == ')' && depth == 0
    });
    close.is_some_and(|(i, _)| {
        let after = text[start + i + 1..].chars().filter(|c| !c.is_whitespace());
        after.take(2).collect::<String>() == "{}"
    })
}

#[test]
fn the_map_covers_both_sets_and_refuses_when_it_cannot_read_them() {
    let root = repo_root();
    let gb10 = resolved(&root, "kernels/gb10", "cu");
    let metal = resolved(&root, "kernels/metal", "metal");

    // ── Refusals. An empty read must never present as agreement. This is the
    // whole reason the check is a test and not a report: a resolver that
    // silently returns nothing once made every shadow comparison in this repo
    // answer "drops nothing".
    assert!(
        !gb10.is_empty(),
        "resolved ZERO gb10 kernels. Either kernels/gb10 is unreadable or the \
         entry-point resolver broke. This is 'could not look', not '0 kernels', \
         and a 1:1 claim built on it would be a claim about nothing."
    );
    assert!(
        !metal.is_empty(),
        "resolved ZERO Metal kernels from kernels/metal/**/*.metal."
    );

    let path = root.join(MANIFEST);
    if std::env::var("AVAROK_METAL_PARITY_EMIT").as_deref() == Ok("1") {
        emit_skeleton(&path, &gb10, &metal);
        return;
    }

    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "cannot read {MANIFEST}: {e}\n\
             The absence of the manifest is NOT a pass. Regenerate the skeleton \
             with AVAROK_METAL_PARITY_EMIT=1."
        )
    });
    let all = parse_manifest(&text);
    let metal_rows: Vec<&Row> = all.iter().filter(|r| r.table == "metal").collect();
    let rows: Vec<&Row> = all.iter().filter(|r| r.table == "kernel").collect();
    assert!(
        !rows.is_empty(),
        "{MANIFEST} parsed to ZERO rows. A manifest that says nothing cannot \
         certify a 1:1 map."
    );

    // 1. every resolved gb10 (path, entry) appears exactly once
    let keyed: BTreeMap<(String, String), &Row> = rows
        .iter()
        .map(|r| ((r.source.clone(), r.entry.clone()), *r))
        .collect();
    let missing: Vec<_> = gb10
        .difference(&keyed.keys().cloned().collect())
        .cloned()
        .collect();
    assert!(
        missing.is_empty(),
        "{} resolved gb10 kernel(s) have no row in {MANIFEST}; first few: {:?}",
        missing.len(),
        missing.iter().take(5).collect::<Vec<_>>()
    );
    assert_eq!(
        rows.len(),
        keyed.len(),
        "{MANIFEST} has duplicate (source, entry) rows"
    );

    // 2. an exception's reason must be a reason
    for r in rows.iter().filter(|r| r.state == "exception") {
        let label = format!("{}::{}", r.source, r.entry);
        assert_reason(&label, "an exception", r.why.as_deref());
    }

    // 3. a mapped row must name a Metal entry point that actually resolves
    let metal_names: BTreeSet<&String> = metal.iter().map(|(_, n)| n).collect();
    for r in &rows {
        if r.state != "mapped" {
            continue;
        }
        let m = r.metal.as_deref().unwrap_or("");
        assert!(
            !m.is_empty(),
            "{}::{} is mapped but names no Metal entry point",
            r.source,
            r.entry
        );
        assert!(
            metal_names.contains(&m.to_string()),
            "{}::{} maps to Metal kernel {m:?}, which does not resolve in \
             kernels/metal. A row cannot claim a kernel that is not there.",
            r.source,
            r.entry
        );
    }

    // ── A `mapped` row must name a test that EXISTS.
    //
    // Without this the checker accepts any row whose `metal` value resolves,
    // which is satisfied by a NAME MATCH. 28 of the 83 Metal kernels share a
    // name with exactly one gb10 entry point, so the checker would have
    // certified 2.8% of a 1:1 map on the strength of spelling. "Mapped" has to
    // mean something would go red if the kernel were wrong.
    let mut test_src = String::new();
    let tdir = root.join("crates/spark-runtime/src/metal_backend/tests");
    if let Ok(rd) = std::fs::read_dir(&tdir) {
        for e in rd.flatten() {
            if e.path().extension().and_then(|s| s.to_str()) == Some("rs") {
                test_src.push_str(&std::fs::read_to_string(e.path()).unwrap_or_default());
            }
        }
    }
    assert!(
        !test_src.is_empty(),
        "read no Metal test sources from {}. That is 'could not look', and          every mapped row's test would then pass unchecked.",
        tdir.display()
    );
    for r in rows.iter().filter(|r| r.state == "mapped") {
        let t = r.test.as_deref().unwrap_or("");
        assert!(
            !t.is_empty(),
            "{}::{} is mapped but names no test. A row without one claims a              mapping nothing would notice breaking.",
            r.source,
            r.entry
        );
        assert!(
            test_src.contains(&format!("fn {t}(")),
            "{}::{} names test {t:?}, which does not exist under              crates/spark-runtime/src/metal_backend/tests/",
            r.source,
            r.entry
        );

        // ── ...and that test must LAUNCH the row's Metal kernel.
        //
        // "Exists" is satisfied by naming ANY test in the directory: nothing
        // above stops a gelu row from citing `metal_silu_gate_matches_reference`,
        // which exists, passes, and never touches gelu (measured: the checker
        // accepted exactly that). Every parity test looks its kernel up as
        // `backend.kernel("<module>", "<entry>")`, so the entry point must
        // appear as a string literal inside the named test's body. This is
        // the floor beneath the campaign's mutation registry, not a substitute
        // for it: it proves the test ran this kernel, not that it would
        // notice the kernel being wrong.
        let m = r.metal.as_deref().unwrap_or("");
        let body = fn_body(&test_src, t).expect("existence asserted just above");
        assert!(
            body.contains(&format!("\"{m}\"")),
            "{}::{} names test {t:?}, whose body never launches Metal kernel \
             {m:?} (no \"{m}\" literal in it). A test that exists but drives a \
             different kernel is evidence about that kernel, not about this row.",
            r.source,
            r.entry
        );
    }

    // ── The other direction: every resolved Metal entry point is accounted for.
    //
    // Everything above constrains the gb10 set only; the mapped rows name 10
    // of 83 Metal kernels and nothing said the other 73 were looked at. A map
    // that constrains one of two sets is not 1:1, so a Metal entry point that
    // is neither a mapped target nor declared in a `[[metal]]` row fails by name.
    let mut by_name: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (src, name) in &metal {
        by_name.entry(name).or_default().push(src);
    }
    for (name, srcs) in &by_name {
        let n = srcs.len();
        assert!(
            n == 1,
            "Metal entry {name:?} resolves in {n} sources {srcs:?}; `metal =` is ambiguous"
        );
    }
    let mapped_targets: BTreeSet<&str> = rows
        .iter()
        .filter(|r| r.state == "mapped")
        .filter_map(|r| r.metal.as_deref())
        .collect();
    for m in &mapped_targets {
        let src = by_name[m][0];
        assert!(
            !empty_body(&root, src, m),
            "mapped target {m:?} has an EMPTY body in {src}; a stub is not a port"
        );
    }
    let mut declared: BTreeSet<(String, String)> = BTreeSet::new();
    for r in &metal_rows {
        let label = format!("[[metal]] {}::{}", r.source, r.entry);
        let key = (r.source.clone(), r.entry.clone());
        assert!(
            metal.contains(&key),
            "{label} does not resolve in kernels/metal: the row outlived the kernel, \
             and a stale row is how a deleted kernel stays 'accounted for'"
        );
        assert!(declared.insert(key), "{label} is declared twice");
        assert!(
            !mapped_targets.contains(r.entry.as_str()),
            "{label} is a mapped target AND declared {}; it is one or the other",
            r.state
        );
        assert!(
            ["metal_only", "counterpart", "stub"].contains(&r.state.as_str()),
            "{label} has state {:?}; [[metal]] rows are metal_only, counterpart or stub",
            r.state
        );
        assert_eq!(
            r.gb10.is_some(),
            r.state != "metal_only",
            "{label}: metal_only names no gb10 kernel; counterpart and stub name theirs in `gb10 =`"
        );
        assert_eq!(
            r.state == "stub",
            empty_body(&root, &r.source, &r.entry),
            "{label}: stub and an empty body go together; a body that does work must be measured"
        );
        if let Some(g) = r.gb10.as_deref() {
            let (src, entry) = g.split_once("::").unwrap_or_else(|| {
                panic!("{label}: `gb10 =` must be \"<source>::<entry>\", got {g:?}")
            });
            assert!(
                gb10.contains(&(src.to_string(), entry.to_string())),
                "{label} names gb10 counterpart {g:?}, which does not resolve in kernels/gb10"
            );
        }
        assert_reason(&label, &r.state, r.why.as_deref());
    }
    let unaccounted: Vec<_> = metal
        .iter()
        .filter(|k| !mapped_targets.contains(k.1.as_str()) && !declared.contains(k))
        .collect();
    assert!(
        unaccounted.is_empty(),
        "{} resolved Metal kernel(s) are neither a mapped target nor declared in a \
         [[metal]] row: {:?}. The map constrains both sets or it is not 1:1.",
        unaccounted.len(),
        unaccounted.iter().take(5).collect::<Vec<_>>()
    );

    let mapped = rows.iter().filter(|r| r.state == "mapped").count();
    let exc = rows.iter().filter(|r| r.state == "exception").count();
    let unported = rows.iter().filter(|r| r.state == "unported").count();
    let count = |s: &str| metal_rows.iter().filter(|r| r.state == s).count();
    let (mo, cp, st) = (count("metal_only"), count("counterpart"), count("stub"));
    let (mt, total) = (
        mapped_targets.len(),
        mapped_targets.len() + metal_rows.len(),
    );
    println!("gb10 resolved: {}", gb10.len());
    println!("metal resolved: {}", metal.len());
    println!(
        "rows: {} (mapped {mapped}, exception {exc}, unported {unported})",
        rows.len()
    );
    println!(
        "metal: {mt} mapped targets + {mo} metal_only + {cp} counterpart + {st} stub = {total}"
    );
}

fn emit_skeleton(
    path: &Path,
    gb10: &BTreeSet<(String, String)>,
    metal: &BTreeSet<(String, String)>,
) {
    let mut s = String::new();
    s.push_str("# SPDX-License-Identifier: AGPL-3.0-only\n#\n");
    s.push_str("# THE 1:1 MAP. Generated skeleton -- every row starts `unported` and is\n");
    s.push_str("# promoted BY HAND, because a row carries judgement (which Metal kernel, which\n");
    s.push_str("# parity tier, which test, which mutations, or why 1:1 is impossible) that no\n");
    s.push_str("# generator can invent. Regenerate with AVAROK_METAL_PARITY_EMIT=1; the\n");
    s.push_str("# checker in crates/avarok-kernels/tests/metal_parity.rs is the verdict.\n");
    s.push_str("#\n# Keyed by (source, entry), never by name: 121 gb10 entry-point names are\n");
    s.push_str("# declared in more than one source file, and rms_norm alone has two different\n");
    s.push_str("# formulas under one name.\n\n");
    s.push_str(&format!(
        "# resolved at generation time: {} gb10 kernels, {} metal kernels\n\n",
        gb10.len(),
        metal.len()
    ));
    for (src, entry) in gb10 {
        s.push_str("[[kernel]]\n");
        s.push_str(&format!("source = \"{src}\"\n"));
        s.push_str(&format!("entry  = \"{entry}\"\n"));
        s.push_str("state  = \"unported\"\n\n");
    }
    std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    std::fs::write(path, s).expect("write manifest");
    println!("emitted {} rows to {}", gb10.len(), path.display());
}
