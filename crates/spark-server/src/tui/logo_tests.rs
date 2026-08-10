// SPDX-License-Identifier: AGPL-3.0-only

//! What the header chips claim, and what the logo draws.
//!
//! The chips are derived entirely from `ServeArgs`, which clap fills in whether
//! or not anything is serving — so every branch here is a claim about a running
//! model, and the `awaiting_model` arm is the one that must assert almost
//! nothing. `render/header_tests.rs` covers that arm against the real `App`;
//! this file covers the per-flag branches underneath it.

use super::*;
use clap::Parser as _;

fn args(extra: &[&str]) -> crate::cli::ServeArgs {
    let mut argv = vec!["spark", "org/m"];
    argv.extend_from_slice(extra);
    crate::cli::ServeArgs::parse_from(argv)
}

fn strip(a: &crate::cli::ServeArgs) -> String {
    badges(a, false)
        .iter()
        .map(|b| b.text.clone())
        .collect::<Vec<_>>()
        .join(" | ")
}

#[test]
fn the_model_chip_prefers_the_served_name_over_the_path() {
    // `--model-name` is what clients ask for; the path is an implementation
    // detail that can be a local directory nobody would recognise.
    let mut a = args(&[]);
    a.model = Some("/scratch/checkpoints/run-17".into());
    a.model_name = Some("org/pretty".into());
    let chips = badges(&a, false);
    assert_eq!(chips[0].text, "org/pretty");
    assert_eq!(chips[0].tint, BadgeTint::Model);

    a.model_name = None;
    assert_eq!(badges(&a, false)[0].text, "/scratch/checkpoints/run-17");

    a.model = None;
    assert_eq!(
        badges(&a, false)[0].text,
        "<model>",
        "a placeholder, not a panic"
    );
}

#[test]
fn speculation_is_reported_as_off_rather_than_omitted() {
    // A missing chip reads as "not shown"; "spec off" is a fact worth stating,
    // because it is the single biggest difference between two runs of the same
    // checkpoint.
    let a = args(&[]);
    assert!(!a.speculative);
    let chips = badges(&a, false);
    let spec = chips
        .iter()
        .find(|b| b.text.starts_with("spec"))
        .expect("a chip");
    assert_eq!(spec.text, "spec off");
    assert_eq!(spec.tint, BadgeTint::Neutral);
}

#[test]
fn a_speculating_run_reports_the_depth_the_scheduler_will_use() {
    // `--num-drafts N` means K = N+1 verified tokens per step. Reporting N
    // would understate the depth by one against every log line that says K.
    let a = args(&["--speculative", "--num-drafts", "3"]);
    let text = strip(&a);
    assert!(text.contains("MTP k=4"), "K is drafts + 1: {text}");
    assert!(!text.contains("spec off"), "{text}");

    for flag in ["--self-speculative", "--ngram-speculative"] {
        let a = args(&[flag]);
        assert!(
            strip(&a).contains("MTP k="),
            "{flag} is speculation too: {}",
            strip(&a)
        );
    }
}

#[test]
fn dflash_displaces_the_mtp_chip_rather_than_sitting_beside_it() {
    // They are alternative drafting strategies — clap refuses both at once —
    // so showing two drafter chips would describe a run that cannot exist.
    let a = args(&["--dflash"]);
    let text = strip(&a);
    assert!(text.contains("DFlash"), "{text}");
    assert!(!text.contains("MTP k="), "one drafter, one chip: {text}");
    assert!(!text.contains("spec off"), "{text}");
}

#[test]
fn the_role_chip_distinguishes_a_head_from_a_worker() {
    // On a multi-node run the two processes are otherwise indistinguishable in
    // the header, and only one of them answers requests.
    let head = args(&["--world-size", "2"]);
    assert!(strip(&head).contains("head 0/2"), "{}", strip(&head));

    let worker = args(&["--world-size", "2", "--rank", "1"]);
    assert!(strip(&worker).contains("worker 1/2"), "{}", strip(&worker));
}

#[test]
fn the_prefix_cache_chip_appears_only_when_it_is_on() {
    // It names the SSM slot pool, which is meaningless without caching — and a
    // pinned slot count is the difference between a warm turn and a 15 s TTFT.
    let off = args(&[]);
    assert!(!strip(&off).contains("prefix-cache"), "{}", strip(&off));

    let on = args(&["--enable-prefix-caching", "--ssm-cache-slots", "256"]);
    let text = strip(&on);
    assert!(text.contains("prefix-cache"), "{text}");
    assert!(text.contains("256"), "and the pinned slot count: {text}");
}

#[test]
fn the_context_chip_rounds_only_when_the_rounding_is_exact() {
    // "60000" is not "58k". A chip that rounds silently makes two runs with
    // different KV budgets look identical.
    for (tokens, want) in [
        (65536usize, "ctx 64k"),
        (4096, "ctx 4k"),
        (1024, "ctx 1k"),
        (60000, "ctx 60000"),
        (1023, "ctx 1023"),
    ] {
        let mut a = args(&[]);
        a.max_seq_len = tokens;
        assert!(
            strip(&a).contains(want),
            "{tokens} should read {want}: {}",
            strip(&a)
        );
    }
}

#[test]
fn the_quant_chip_names_all_three_dtypes_that_can_differ() {
    // KV, LM head and MTP are quantized independently, and a run is only
    // comparable to another that matches on all three.
    let mut a = args(&[]);
    a.kv_cache_dtype = "fp8".into();
    a.lm_head_dtype = "bf16".into();
    a.mtp_quantization = "nvfp4".into();
    let chip = badges(&a, false)
        .into_iter()
        .find(|b| b.text.starts_with("kv "))
        .expect("a chip");
    assert_eq!(chip.text, "kv fp8 · lm bf16 · mtp nvfp4");
    assert_eq!(chip.tint, BadgeTint::Quant);
}

#[test]
fn the_awaiting_strip_says_the_way_out_and_the_bound_port() {
    // Everything else on the strip describes a model that is not loaded; the
    // listener is genuinely up, so the address is the one thing still true.
    let a = args(&["--port", "9123"]);
    let chips = badges(&a, true);
    assert_eq!(chips.len(), 2, "nothing else may be asserted: {chips:?}");
    assert!(chips[0].text.contains("Library"), "{:?}", chips[0].text);
    assert_eq!(chips[1].text, ":9123");
    assert!(chips.iter().all(|b| b.tint == BadgeTint::Neutral));
}

#[test]
fn the_address_is_the_last_chip_so_it_survives_a_narrow_terminal() {
    // Chips wrap in order; the port is what a reader needs to point a client
    // at, so it must not be the first thing pushed off the line.
    let a = args(&["--port", "8871"]);
    let chips = badges(&a, false);
    assert_eq!(chips.last().expect("chips").text, ":8871");
}

#[test]
fn the_three_row_logo_is_exactly_three_rows_of_equal_chevron_width() {
    // It shares a header row budget with the status pill; a fourth line or a
    // ragged chevron shears the whole header.
    let lines = three_line(None);
    assert_eq!(lines.len(), 3);
    for (i, line) in lines.iter().enumerate() {
        let chevrons: Vec<&str> = line
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .filter(|c| CHEVRON_ROWS.contains(c))
            .collect();
        assert_eq!(chevrons.len(), 3, "row {i} draws three chevrons");
        assert!(chevrons.iter().all(|c| *c == CHEVRON_ROWS[i]));
    }
    let wordmark: String = lines[1].spans.iter().map(|s| s.content.as_ref()).collect();
    assert!(wordmark.contains("A T L A S"), "{wordmark}");
}

#[test]
fn the_one_line_logo_is_three_chevrons_and_the_name() {
    let line = one_line(None);
    let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
    assert_eq!(text, "❯❯❯ Atlas");
}

/// Pin the color mode before reading any color.
///
/// `theme::C::color` reads `COLORTERM` on EVERY call, and a sibling test sets
/// it mid-run — so two colors read either side of that set compare unequal for
/// a reason that has nothing to do with the logo. Setting it to the value the
/// sibling sets makes the two agree whichever runs first.
fn pin_truecolor() {
    unsafe { std::env::set_var("COLORTERM", "truecolor") };
}

#[test]
fn a_steady_logo_uses_the_brand_colors_unmodified() {
    // The wave stops permanently once SERVING; `None` is that state, and it
    // must not leave a chevron dimmed.
    pin_truecolor();
    let want = [
        theme::PURPLE.color(),
        theme::CYAN.color(),
        theme::GREEN.color(),
    ];
    let got: Vec<Color> = three_line(None)[0]
        .spans
        .iter()
        .filter(|s| CHEVRON_ROWS.contains(&s.content.as_ref()))
        .map(|s| s.style.fg.expect("a chevron is colored"))
        .collect();
    assert_eq!(got, want, "purple, cyan, green, left to right");
}

#[test]
fn the_wave_returns_to_where_it_started_every_three_steps() {
    // Three chevrons, so step 3 is step 0 — otherwise the animation drifts and
    // never repeats cleanly.
    pin_truecolor();
    for step in 0..3usize {
        assert_eq!(
            format!("{:?}", one_line(Some(step)).spans),
            format!("{:?}", one_line(Some(step + 3)).spans),
            "step {step}"
        );
    }
}
