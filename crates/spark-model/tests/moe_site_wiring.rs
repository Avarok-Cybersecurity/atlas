// SPDX-License-Identifier: AGPL-3.0-only

//! Every `MoeLayer` must be told where it sits — and told correctly.
//!
//! `MoeSite` exists because two features key off it: per-request expert
//! telemetry attributes routing to a layer index, and boot-time expert
//! loading (`--expert-category`) selects that layer's row of the MODEL.toml
//! category table. Both fail SILENTLY on a wrong index — the wrong experts
//! get masked, or a category's measured set is attributed to the wrong
//! layer — so the mistake never surfaces as a crash, only as worse output.
//!
//! The compiler already guarantees every construction site passes *a* site
//! (it is a required parameter, no `Default`). What it cannot check is that
//! the value is the loader's live layer variable rather than a constant
//! someone pasted in. This scans the loaders for that.
//!
//! Source-scanning is the same tactic `tests/kernel_shadow_detector.rs`
//! uses: the property lives across a dozen loader files that no unit test
//! can reach without a GPU and a checkpoint.

use std::path::{Path, PathBuf};

fn src_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).expect("readable source dir") {
        let p = e.unwrap().path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().and_then(|x| x.to_str()) == Some("rs") {
            out.push(p);
        }
    }
}

/// The first argument of every `MoeLayer::new(...)` / `new_with_hash(...)`
/// call in the tree, as written, with its file and line.
fn construction_sites() -> Vec<(PathBuf, usize, String)> {
    let mut files = Vec::new();
    rust_files(&src_root(), &mut files);
    files.sort();

    let mut sites = Vec::new();
    for path in files {
        // `moe/init.rs` DEFINES the constructors; it has no call to check.
        if path.ends_with("layers/moe/init.rs") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            // `NemotronMoeLayer::new(` ends in the same text: require the
            // match to start a path segment, or the scan claims a different
            // type's constructor (NemotronMoeLayer is its own layer type and
            // carries no MoeSite).
            let starts_segment = |c: usize| {
                c == 0
                    || !line[..c]
                        .chars()
                        .next_back()
                        .is_some_and(|ch| ch.is_alphanumeric() || ch == '_')
            };
            let Some(col) = ["MoeLayer::new(", "MoeLayer::new_with_hash("]
                .iter()
                .filter_map(|pat| {
                    line.match_indices(pat)
                        .find(|(c, _)| starts_segment(*c))
                        .map(|(c, _)| c + pat.len())
                })
                .max()
            else {
                continue;
            };
            // Skip doc comments and prose that merely name the constructor.
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") || trimmed.starts_with("///") || trimmed.starts_with("*") {
                continue;
            }
            // The first argument is either on this line or the next
            // non-empty one (rustfmt breaks long call lists).
            let rest = line[col..].trim();
            let first_arg = if rest.is_empty() {
                lines
                    .get(i + 1)
                    .map(|l| l.trim())
                    .unwrap_or_default()
                    .to_string()
            } else {
                rest.to_string()
            };
            let first_arg = first_arg
                .split(',')
                .next()
                .unwrap_or_default()
                .trim()
                .to_string();
            sites.push((path.clone(), i + 1, first_arg));
        }
    }
    sites
}

#[test]
fn every_moe_construction_names_its_site() {
    let sites = construction_sites();
    // If this trips, the scan stopped seeing call sites (a rename, a new
    // formatting shape) — the test would then pass vacuously forever.
    assert!(
        sites.len() >= 12,
        "expected the known MoeLayer construction sites, found {}: {sites:#?}",
        sites.len()
    );

    for (path, line, arg) in &sites {
        let ok = arg == "MoeSite::MtpHead"
            || arg.starts_with("MoeSite::Layer(")
            || arg.starts_with("crate::layers::MoeSite::Layer(")
            || arg == "site";
        assert!(
            ok,
            "{}:{line}: first argument to MoeLayer::new must be a MoeSite, got `{arg}`",
            path.display()
        );
    }
}

#[test]
fn no_loader_hardcodes_a_layer_index() {
    // The failure this exists for: a new loader copy-pasted from another
    // one, keeping `MoeSite::Layer(0)` instead of its own loop variable.
    // Every layer would then claim to be layer 0 — BEL would mask every
    // layer with layer 0's expert set, and telemetry would pile all
    // routing onto one layer. Both are silent.
    for (path, line, arg) in construction_sites() {
        if let Some(inner) = arg
            .strip_prefix("MoeSite::Layer(")
            .or_else(|| arg.strip_prefix("crate::layers::MoeSite::Layer("))
        {
            let inner = inner.trim_end_matches(')').trim();
            assert!(
                inner.parse::<usize>().is_err(),
                "{}:{line}: MoeSite::Layer({inner}) hardcodes a layer index — pass the \
                 loader's layer variable, or every layer will claim to be layer {inner}",
                path.display()
            );
        }
    }
}

#[test]
fn mtp_head_is_not_a_model_layer() {
    // BEL and telemetry both branch on this: the drafter's internal MoE
    // routes on its own weights, so folding it into a category's expert set
    // would record experts the target model never used. `layer_idx()`
    // returning `None` is what keeps it out.
    use spark_model::layers::MoeSite;
    assert_eq!(MoeSite::MtpHead.layer_idx(), None);
    assert_eq!(MoeSite::Layer(7).layer_idx(), Some(7));
}
