// SPDX-License-Identifier: AGPL-3.0-only

//! What the header claims about the server.
//!
//! Every test here is a case where the header asserted something it had not
//! checked: a status pill reading SERVING because the *listener* was up, a chip
//! strip describing clap defaults as a running configuration, a mini-strip
//! naming a KV dtype for a process that had loaded no model. They are grouped
//! because they share one fix — `app.awaiting_model`, read rather than
//! re-derived, so the three cannot disagree.

#[test]
fn the_status_pill_does_not_claim_to_be_serving_with_no_model() {
    // `progress.ready` means the LISTENER is up. Since the no-model boot binds
    // a socket, that is true before any model exists, and checking it first
    // reported "SERVING" on a server serving nothing.
    use clap::Parser as _;
    let mut args = crate::cli::ServeArgs::parse_from(["spark", "m"]);
    args.model = None;
    let mut app = crate::tui::app::App::new(args);
    assert!(app.awaiting_model, "no model was named");

    app.progress.ready = true; // the listener came up
    let pill = super::status_pill(&app);
    assert!(
        pill.content.contains("NO MODEL"),
        "a bound socket is not a loaded model: {:?}",
        pill.content
    );

    // And once a model is actually loaded it says so.
    app.awaiting_model = false;
    assert!(super::status_pill(&app).content.contains("SERVING"));
}

#[test]
fn the_chip_strip_describes_the_running_config_not_the_boot_argv() {
    // Every chip is read from ServeArgs, and a swap replaces the whole argv.
    // Boot with no model and the strip showed bare defaults and a literal
    // "<model>"; launch a recipe and it kept showing them.
    use clap::Parser as _;
    let boot = crate::cli::ServeArgs::parse_from(["spark", "m"]);
    let boot_chips = crate::tui::logo::badges(&boot, false);
    let boot_text: String = boot_chips.iter().map(|b| b.text.clone()).collect();

    let mut live = crate::cli::ServeArgs::parse_from(["spark", "org/loaded"]);
    live.max_batch_size = boot.max_batch_size + 7;
    let live_chips = crate::tui::logo::badges(&live, false);
    let live_text: String = live_chips.iter().map(|b| b.text.clone()).collect();

    assert!(live_text.contains("org/loaded"), "names the live model");
    assert_ne!(
        boot_text, live_text,
        "the strip must be able to differ from the boot argv"
    );
    assert!(
        live_text.contains(&(boot.max_batch_size + 7).to_string()),
        "and it reports the live batch size: {live_text}"
    );
}

#[test]
fn the_chip_strip_asserts_nothing_about_a_model_that_is_not_loaded() {
    // With nothing serving, every chip except the address is a clap default
    // dressed up as a running configuration — the strip read "kv fp8" for a
    // process that had loaded no KV cache at all.
    use clap::Parser as _;
    let args = crate::cli::ServeArgs::parse_from(["spark"]);
    let text: String = crate::tui::logo::badges(&args, true)
        .iter()
        .map(|b| b.text.clone())
        .collect::<Vec<_>>()
        .join(" ");

    for claim in ["kv ", "lm ", "mtp ", "batch ", "ctx ", "sched ", "<model>"] {
        assert!(
            !text.contains(claim),
            "awaiting strip must not claim {claim:?}: {text}"
        );
    }
    // The listener really is up — it binds before any model loads — so the
    // address is the one thing it may still assert, and it must stay: it is
    // what a user needs to point a client at.
    assert!(
        text.contains(&format!(":{}", args.port)),
        "the bound port is true with no model: {text}"
    );
    assert!(
        text.contains("Library"),
        "and it says how to fix it: {text}"
    );
}

#[test]
fn the_header_mini_strip_does_not_claim_a_kv_dtype_with_no_model() {
    // This is the line the report was actually about — the top-right corner
    // read " · kv fp8 · :8123" on a server that had loaded nothing. It is a
    // second renderer of the same claim as the chip strip, and fixing only the
    // chips would have left the visible one wrong.
    use clap::Parser as _;
    let mut args = crate::cli::ServeArgs::parse_from(["spark", "m"]);
    args.model = None;
    let mut app = crate::tui::app::App::new(args);
    assert!(app.awaiting_model, "no model was named");
    let line = super::header_line(&app);
    assert!(!line.contains("kv "), "no dtype with no model: {line}");
    assert!(line.contains("Library"), "says the way out: {line}");
    assert!(
        line.contains(&format!(":{}", app.args.port)),
        "keeps the bound port: {line}"
    );

    // And once something is loaded it describes it again.
    app.awaiting_model = false;
    let line = super::header_line(&app);
    assert!(
        line.contains("kv "),
        "a loaded model has a KV dtype: {line}"
    );
}
