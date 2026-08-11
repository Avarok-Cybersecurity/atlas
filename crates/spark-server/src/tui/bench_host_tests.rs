// SPDX-License-Identifier: AGPL-3.0-only

//! The roster derivation and the round argv. No GPU: every case here is about
//! how a cached checkpoint is CLASSIFIED, which is the half of the coverage
//! guarantee that lives on this side of the seam.

use super::*;
use crate::tui::data::library::LibraryEntry;

fn entry(id: &str, has_weights: bool, optimized: bool) -> LibraryEntry {
    LibraryEntry {
        id: id.into(),
        snapshot_dir: std::path::PathBuf::from("/nonexistent"),
        size_bytes: 0,
        has_weights,
        model_type: "qwen3_6_moe".into(),
        quant: "nvfp4".into(),
        layers: 40,
        hidden: 2048,
        heads: 16,
        experts: 256,
        context: 262_144,
        optimized,
    }
}

#[test]
fn a_complete_supported_checkpoint_is_runnable() {
    assert_eq!(classify(&entry("org/m", true, true)), None);
}

#[test]
fn a_half_downloaded_checkpoint_is_absent_for_weights_not_for_kernels() {
    // Order matters: weights that are not all there cannot be inspected for
    // architecture support, so reporting "no kernels" would be a guess.
    assert_eq!(
        classify(&entry("org/m", false, false)),
        Some(Absence::NoWeights)
    );
}

#[test]
fn a_downloaded_model_this_build_has_no_kernels_for_is_skipped_with_that_reason() {
    assert_eq!(
        classify(&entry("org/m", true, false)),
        Some(Absence::NoKernels)
    );
    assert!(Absence::NoKernels.reason().contains("kernels"));
}

#[test]
fn an_unreadable_config_is_not_reported_as_an_unsupported_architecture() {
    // `library::scan` fills `model_type` only when it parsed `config.json`, and
    // leaves `optimized` false either way. Answering "no kernels for this
    // architecture" from a config nothing could read states a finding about an
    // architecture nobody established.
    let mut e = entry("org/m", true, false);
    e.model_type = String::new();
    assert_eq!(classify(&e), Some(Absence::NoConfig));
    assert!(Absence::NoConfig.reason().contains("unreadable"));
}

fn bound_host() -> Arc<ModelHost> {
    let host = Arc::new(ModelHost::empty());
    host.set_bound("127.0.0.1".into(), 8899);
    host
}

#[test]
fn a_round_is_served_on_the_port_this_server_already_bound() {
    // The matrix never opens a listener of its own. That is what makes the
    // Python's `--host`/`--bind` mismatch — a server bound to loopback inside a
    // bridged namespace that no probe could reach — structurally impossible
    // here: there is one bind, and it happened before the run started.
    let h = TuiServeHost::new(bound_host(), None);
    let args = h
        .argv_for(
            "org/m",
            ServeOptions {
                max_seq_len: 32_768,
                speculative: false,
            },
        )
        .expect("a valid command line");
    assert_eq!(args.port, 8899);
    assert_eq!(args.model.as_deref(), Some("org/m"));
    assert_eq!(args.max_seq_len, 32_768);
    assert!(!args.speculative);
    assert_eq!(
        h.endpoint("org/m").expect("bound").base_url,
        "http://127.0.0.1:8899"
    );
}

#[test]
fn the_mtp_arm_is_the_only_thing_the_speculative_option_changes() {
    let h = TuiServeHost::new(bound_host(), None);
    let opts = |speculative| ServeOptions {
        max_seq_len: 16_384,
        speculative,
    };
    let off = h.argv_for("org/m", opts(false)).expect("valid");
    let on = h.argv_for("org/m", opts(true)).expect("valid");
    assert!(!off.speculative && on.speculative);
    // Everything else comes from the checkpoint's own MODEL.toml defaults —
    // which is what the matrix is measuring, so the harness must not override
    // more than it says it does.
    assert_eq!(off.max_seq_len, on.max_seq_len);
    assert_eq!(off.port, on.port);
}

#[test]
fn a_round_cannot_be_built_before_the_server_has_bound() {
    // Guessing a port here would send every probe somewhere else.
    let h = TuiServeHost::new(Arc::new(ModelHost::empty()), None);
    let err = h
        .argv_for(
            "org/m",
            ServeOptions {
                max_seq_len: 4096,
                speculative: false,
            },
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("binding its port"), "{err}");
}

#[test]
fn a_cache_directory_with_nothing_in_it_yields_an_empty_roster() {
    // Derived, not listed: the Python's hand-maintained `ROUNDS` named twelve
    // checkpoints that were not on the box, and nothing noticed.
    let dir = tempfile::tempdir().expect("scratch dir");
    let h = TuiServeHost::new(bound_host(), Some(dir.path().to_path_buf()));
    assert!(h.roster().expect("a scan of an empty cache").is_empty());
}

#[test]
fn a_half_downloaded_checkpoint_reaches_the_roster_as_a_named_absence() {
    // Skipped, but SAID: a round that silently vanished from the matrix reads
    // as a model that passed.
    let dir = tempfile::tempdir().expect("scratch dir");
    let snapshot = dir
        .path()
        .join("models--org--half")
        .join("snapshots")
        .join("deadbeef");
    std::fs::create_dir_all(&snapshot).expect("mock cache");
    std::fs::write(snapshot.join("config.json"), "{}").expect("config");

    let h = TuiServeHost::new(bound_host(), Some(dir.path().to_path_buf()));
    let roster = h.roster().expect("a scan");
    assert_eq!(roster.len(), 1, "{roster:?}");
    assert_eq!(roster[0].model, "org/half");
    assert_eq!(roster[0].absent, Some(Absence::NoWeights));
}

#[test]
fn the_cache_override_is_carried_into_every_rounds_command_line() {
    // A round served from a different cache than the one the roster was scanned
    // from would serve a checkpoint the matrix never saw.
    let dir = tempfile::tempdir().expect("scratch dir");
    let h = TuiServeHost::new(bound_host(), Some(dir.path().to_path_buf()));
    let args = h
        .argv_for(
            "org/m",
            ServeOptions {
                max_seq_len: 4096,
                speculative: false,
            },
        )
        .expect("valid");
    assert_eq!(args.cache_dir.as_deref(), Some(dir.path()));
}

#[test]
fn a_scanned_id_reaches_the_round_argv_exactly_as_the_cache_spells_it() {
    // The roster's ids go through a hand-assembled command line, so a round
    // that served a re-spelled id would be measuring a different checkpoint
    // than the one the matrix reports.
    let h = TuiServeHost::new(bound_host(), None);
    for id in [
        "unsloth/Qwen3.6-27B-NVFP4",
        "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4",
        "org/name.with.dots",
    ] {
        let args = h
            .argv_for(
                id,
                ServeOptions {
                    max_seq_len: 4096,
                    speculative: false,
                },
            )
            .expect("valid");
        assert_eq!(args.model.as_deref(), Some(id));
    }
}

#[test]
fn restoring_a_box_that_was_serving_nothing_is_a_no_op_not_a_teardown() {
    // Leaving the last round loaded is closer to where the box was found than
    // tearing it down to nothing, and either way there is no argv to restore.
    // No swap is reachable from here, so nothing touches the GPU.
    let h = TuiServeHost::new(bound_host(), None);
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    rt.block_on(h.restore()).expect("nothing to put back");
}
