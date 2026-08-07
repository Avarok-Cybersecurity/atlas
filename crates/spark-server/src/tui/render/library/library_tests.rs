// SPDX-License-Identifier: AGPL-3.0-only

//! The Library's three panes: the joined list, the recipe cards, the settings
//! form.
//!
//! The recurring defect class here is a claim the pane had not checked — a
//! badge shown because the network was down, a launch offered for a recipe the
//! dashboard cannot start, a progress line that looked identical to a dead
//! download. Each of those is a test.

use crate::recipe::Recipe;
use crate::recipe::fetch::Index;
use crate::tui::app::{App, Section};
use crate::tui::data::library::LibraryEntry;
use crate::tui::lib_state::View;
use crate::tui::render::harness::{has, screen};

fn recipe(stem: &str) -> Recipe {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recipes/qwen3.6")
        .join(format!("{stem}.yaml"));
    Recipe::parse(
        format!("qwen3.6/{stem}"),
        &std::fs::read_to_string(path).expect("fixture"),
    )
    .expect("parses")
}

fn local(id: &str, has_weights: bool) -> LibraryEntry {
    LibraryEntry {
        id: id.to_string(),
        snapshot_dir: Default::default(),
        size_bytes: 34_900_000_000,
        has_weights,
        model_type: "qwen3_6_moe".into(),
        quant: "fp8".into(),
        layers: 40,
        hidden: 4096,
        heads: 32,
        experts: 128,
        context: 65536,
        optimized: true,
    }
}

/// A Library holding one recipe whose weights are on disk.
fn lib(recipes: Vec<Recipe>, locals: Vec<LibraryEntry>) -> App {
    let mut a = crate::tui::render::tests::app();
    a.section = Section::Library;
    a.library = locals;
    a.lib.index = Index {
        recipes,
        ..Default::default()
    };
    a.lib.rebuild(&a.library);
    a
}

fn flagship() -> App {
    let r = recipe("qwen3.6-35b-a3b-fp8-mtp");
    let model = r.model.clone();
    lib(vec![r], vec![local(&model, true)])
}

/// A library of one model, mid-download, so the per-row progress line is
/// reached.
///
/// The repo id is deliberately not a real one: `start` spawns a worker that
/// would otherwise begin pulling tens of gigabytes off the Hub for the length
/// of the test binary's life.
fn downloading(done: u64, total: u64) -> App {
    const REPO: &str = "atlas-render-tests/no-such-model";
    let mut a = lib(Vec::new(), vec![local(REPO, false)]);
    let root = std::env::temp_dir().join("atlas-library-render-tests");
    std::fs::create_dir_all(&root).ok();
    a.download.start(REPO, root);
    let job = a.download.job.as_mut().expect("a job");
    job.done = done;
    job.total = total;
    job.rate_bps = 96_000_000.0;
    job.file = Some((3, 85, "model-00003-of-00085.safetensors".into()));
    a
}

#[test]
fn the_progress_line_sheds_fields_from_the_right_as_the_pane_narrows() {
    // The bar and the percentage answer "is this still moving", so they are
    // the last things to go; the rate and the file counter are the first.
    let a = downloading(4_000_000_000, 34_900_000_000);

    let cramped = screen(&a, 120, 40);
    assert!(
        has(&cramped, " 11%"),
        "the percentage survives:\n{cramped:#?}"
    );
    assert!(has(&cramped, "▓"), "and so does the bar");
    assert!(!has(&cramped, "MB/s"), "{cramped:#?}");
    assert!(!has(&cramped, "file 3/85"), "{cramped:#?}");

    let roomy = screen(&a, 200, 40);
    // ★ This asserted "4.0 GB / 34.9 GB" — the decimal divisor this line used
    // to apply, which is the same bytes the Library card next to it rendered
    // as 32.5 GB. The test pinned the disagreement rather than the
    // requirement, which is that one file has one size wherever it is shown.
    assert!(has(&roomy, "3.7 GB / 32.5 GB"), "{roomy:#?}");
    // ★ And this asserted "96 MB/s" — the SAME decimal divisor, on the same
    // line, left standing when the sizes beside it were fixed. A user dividing
    // the 28.8 GB still to go by the rate got a time 7% short.
    assert!(has(&roomy, "92 MB/s"), "{roomy:#?}");

    let wide = screen(&a, 280, 40);
    assert!(has(&wide, "file 3/85"), "{wide:#?}");
}

#[test]
fn a_download_the_hub_gave_no_size_for_says_fetching_rather_than_nought_percent() {
    let a = downloading(900_000_000, 0);
    let rows = screen(&a, 200, 40);
    assert!(has(&rows, "fetching"), "{rows:#?}");
    assert!(!has(&rows, "0.0%"), "no fraction was knowable:\n{rows:#?}");
}

#[test]
fn a_cancelling_download_says_stopping_instead_of_a_bar_that_keeps_moving() {
    let mut a = downloading(4_000_000_000, 34_900_000_000);
    a.download.job.as_mut().expect("a job").cancelling = true;
    let rows = screen(&a, 200, 40);
    assert!(has(&rows, "stopping…"), "{rows:#?}");
    assert!(
        !has(&rows, "MB/s"),
        "and drops the fields that imply progress:\n{rows:#?}"
    );
    assert!(has(&rows, "x stop the download"), "{rows:#?}");
}

#[test]
fn an_update_badge_appears_only_for_a_mismatch_that_was_actually_confirmed() {
    use crate::model_download::stale::Freshness;
    let mut a = flagship();
    let model = a.lib.visible()[0].model.clone();

    // Unreachable Hub: `Unknown`. A badge here would be a lie about the disk.
    a.download
        .freshness
        .insert(model.clone(), Freshness::Unknown);
    assert!(!has(&screen(&a, 200, 40), " update "));

    a.download.freshness.insert(
        model.clone(),
        Freshness::Stale {
            local: "abc".into(),
            remote: "def".into(),
        },
    );
    let stale = screen(&a, 200, 40);
    assert!(has(&stale, " update "), "{stale:#?}");
    assert!(has(&stale, "d update"), "and the key that fixes it");
}

#[test]
fn a_freshness_check_in_flight_holds_the_badges_place_so_the_row_cannot_reflow() {
    let mut a = flagship();
    let model = a.lib.visible()[0].model.clone();
    a.download.checking = Some(model);
    assert!(has(&screen(&a, 200, 40), "░░░░░░"));
}

#[test]
fn a_search_that_matches_nothing_says_so_rather_than_looking_empty() {
    let mut a = flagship();
    a.lib.filter = "llama".into();
    a.lib.rebuild(&a.library.clone());
    let rows = screen(&a, 200, 40);
    assert!(has(&rows, "nothing matches this search"), "{rows:#?}");
    assert!(
        !has(&rows, "press r to fetch recipes"),
        "an empty SEARCH is not an empty library:\n{rows:#?}"
    );
}

#[test]
fn the_search_box_echoes_what_is_being_typed_and_keeps_it_after() {
    let mut a = flagship();
    a.lib.filter = "qwen".into();
    a.lib.filter_editing = true;
    assert!(has(&screen(&a, 200, 40), "search: qwen▏"));

    a.lib.filter_editing = false;
    let rows = screen(&a, 200, 40);
    assert!(has(&rows, "search: qwen"), "{rows:#?}");
    assert!(!has(&rows, "search: qwen▏"));
}

#[test]
fn an_offline_recipe_index_says_why_and_what_to_do_about_it() {
    let mut a = flagship();
    a.lib.index.offline = Some("no route to host".into());
    let rows = screen(&a, 200, 40);
    assert!(has(&rows, "offline"), "the title admits it:\n{rows:#?}");
    assert!(
        has(&rows, "HTTPS_PROXY"),
        "and the body names the fix for THIS cause:\n{rows:#?}"
    );
}

#[test]
fn a_checkpoint_with_no_recipe_is_still_listed_and_says_the_flags_are_yours() {
    let a = lib(Vec::new(), vec![local("org/unknown-checkpoint", true)]);
    let rows = screen(&a, 200, 40);
    assert!(has(&rows, "org/unknown-checkpoint"), "{rows:#?}");
    assert!(has(&rows, "no recipe"), "{rows:#?}");
    assert!(has(&rows, "No recipe covers this checkpoint"), "{rows:#?}");
    assert!(has(&rows, "ON DISK"), "what IS known is still shown");
    assert!(has(&rows, "qwen3_6_moe"));
}

#[test]
fn the_row_glyph_distinguishes_present_partial_and_absent_weights() {
    let r = recipe("qwen3.6-35b-a3b-fp8-mtp");
    let model = r.model.clone();

    let present = lib(vec![r.clone()], vec![local(&model, true)]);
    assert!(has(&screen(&present, 200, 40), "✓ "));

    let partial = lib(vec![r.clone()], vec![local(&model, false)]);
    let partial_rows = screen(&partial, 200, 40);
    assert!(has(&partial_rows, "◐ "), "{partial_rows:#?}");
    assert!(
        has(&partial_rows, "d resume the download"),
        "a half-downloaded model is resumed, not restarted:\n{partial_rows:#?}"
    );

    let absent = lib(vec![r], Vec::new());
    let absent_rows = screen(&absent, 200, 40);
    assert!(has(&absent_rows, "· "), "{absent_rows:#?}");
    assert!(has(&absent_rows, "weights are not in the local cache"));
    assert!(has(&absent_rows, "d download the weights"));
}

#[test]
fn the_settings_form_marks_the_rows_that_were_changed_and_previews_the_command() {
    let mut a = flagship();
    a.lib.open_cards().expect("opens");
    a.lib.open_config().expect("opens");
    assert_eq!(a.lib.view, View::Config);
    let (key, _, _) = a.lib.config_rows()[0].clone();
    a.lib.overrides.insert(key.clone(), "4242".into());

    let rows = screen(&a, 200, 50);
    assert!(has(&rows, "SETTINGS ─ 1 changed"), "{rows:#?}");
    assert!(has(&rows, "•"), "the gutter marks the changed row");
    assert!(has(&rows, "4242"), "{rows:#?}");
    assert!(has(&rows, "COMMAND"), "{rows:#?}");
    assert!(
        has(&rows, "spark serve"),
        "a form that hides its output makes the user guess:\n{rows:#?}"
    );
}

#[test]
fn a_field_being_edited_shows_its_buffer_and_a_cursor_not_the_stored_value() {
    let mut a = flagship();
    a.lib.open_cards().expect("opens");
    a.lib.open_config().expect("opens");
    a.lib.editing = true;
    a.lib.edit_buffer = "99991".into();
    let rows = screen(&a, 200, 50);
    assert!(has(&rows, "99991▏"), "{rows:#?}");
}

#[test]
fn a_validation_error_attaches_to_the_row_that_caused_it() {
    let mut a = flagship();
    a.lib.open_cards().expect("opens");
    a.lib.open_config().expect("opens");
    a.lib.error = Some("must be a positive integer".into());
    assert!(has(&screen(&a, 200, 50), "must be a positive integer"));
}

#[test]
fn a_card_says_on_its_face_when_enter_will_refuse_it() {
    // A multi-node recipe reaching the form only to fail a world-size check
    // sends the reader after the wrong fix.
    let mut r = recipe("qwen3.6-35b-a3b-fp8-mtp");
    r.min_nodes = 2;
    let model = r.model.clone();
    let mut a = lib(vec![r], vec![local(&model, true)]);
    a.lib.open_cards().expect("opens");
    let rows = screen(&a, 200, 50);
    assert!(
        has(
            &rows,
            "needs 2 nodes — the dashboard starts single-node runs only"
        ),
        "{rows:#?}"
    );
    assert!(!has(&rows, "⏎ configure and start"), "{rows:#?}");
    assert!(has(&rows, "EP=2"), "the topology chip says why:\n{rows:#?}");
}

#[test]
fn a_recipe_for_another_runtime_is_listed_but_never_offered_as_launchable() {
    let mut r = recipe("qwen3.6-35b-a3b-fp8-mtp");
    r.runtime = Some("vllm".into());
    let model = r.model.clone();
    let mut a = lib(vec![r], vec![local(&model, true)]);
    let list = screen(&a, 200, 50);
    assert!(
        has(&list, " vllm "),
        "it is listed, never hidden:\n{list:#?}"
    );

    a.lib.open_cards().expect("opens");
    let cards = screen(&a, 200, 50);
    assert!(
        has(&cards, "this runtime cannot be launched from here"),
        "{cards:#?}"
    );
    assert!(has(&cards, "⊘ vllm"), "and the card is marked:\n{cards:#?}");
}

#[test]
fn every_library_pane_survives_narrow_and_short_terminals() {
    let mut a = flagship();
    for (w, h) in [(20u16, 4u16), (20, 8), (20, 40), (40, 12), (200, 3)] {
        assert_eq!(screen(&a, w, h).len(), h as usize, "list at {w}x{h}");
    }
    a.lib.open_cards().expect("opens");
    for (w, h) in [(20u16, 4u16), (20, 8), (40, 12), (200, 3)] {
        assert_eq!(screen(&a, w, h).len(), h as usize, "cards at {w}x{h}");
    }
    a.lib.open_config().expect("opens");
    for (w, h) in [(20u16, 4u16), (20, 8), (40, 12), (200, 3)] {
        assert_eq!(screen(&a, w, h).len(), h as usize, "config at {w}x{h}");
    }
}
