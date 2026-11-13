// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: unit tests of the copy planner: the plan selection table
//! pinned clause by clause, the lookup round length, the planner end to end
//! on synthetic committed streams (copy chosen on copied text, MTP kept on
//! fresh text, the source cursor kept across accepted rounds and cleared on a
//! rejection, no draft or rejected token ever entering the index), and the
//! configuration grammar.

use super::config::{self, Cli, Config, Mode};
use super::estimator::GateVerdict;
use super::index::CopyIndex;
use super::policy::*;

fn base() -> Inputs {
    Inputs {
        mode: Mode::Auto,
        no_drafts: false,
        grammar: false,
        mtp_drafts: 3,
        lookup_ok: true,
        lookup_len: 15,
        match_len: Some(40),
        confident: true,
        min_match: 8,
    }
}

fn take() -> GateVerdict {
    GateVerdict::Take
}

#[test]
fn clause_1_no_drafts_means_no_speculative_step() {
    let i = Inputs {
        no_drafts: true,
        ..base()
    };
    assert_eq!(select(&i, take), Choice::NoSpec);
    // Even with the drafter off and a grammar.
    let i = Inputs {
        no_drafts: true,
        mode: Mode::Off,
        grammar: true,
        ..base()
    };
    assert_eq!(select(&i, take), Choice::NoSpec);
}

#[test]
fn clause_2_and_3_off_or_grammar_keep_mtp() {
    let mut gate_ran = false;
    let off = Inputs {
        mode: Mode::Off,
        ..base()
    };
    assert_eq!(
        select(&off, || {
            gate_ran = true;
            GateVerdict::Take
        }),
        Choice::Mtp
    );
    let grammar = Inputs {
        grammar: true,
        ..base()
    };
    assert_eq!(
        select(&grammar, || {
            gate_ran = true;
            GateVerdict::Take
        }),
        Choice::Mtp
    );
    assert!(!gate_ran, "the gate never runs for an ineligible row");
}

#[test]
fn clause_4_no_source_keeps_mtp() {
    let i = Inputs {
        match_len: None,
        ..base()
    };
    assert_eq!(select(&i, take), Choice::Mtp);
}

#[test]
fn clause_5_lookup_needs_rounds_eligibility_confidence_and_length() {
    assert_eq!(select(&base(), take), Choice::Lookup);
    let rounds = Inputs {
        mode: Mode::Rounds,
        ..base()
    };
    assert_eq!(select(&rounds, take), Choice::Lookup);
    // Each missing condition falls through to the chain clause.
    for i in [
        Inputs {
            mode: Mode::Chain,
            ..base()
        },
        Inputs {
            lookup_ok: false,
            ..base()
        },
        Inputs {
            confident: false,
            ..base()
        },
        Inputs {
            lookup_len: LOOKUP_MIN - 1,
            ..base()
        },
    ] {
        assert_eq!(select(&i, take), Choice::Chain, "{i:?}");
    }
    // Rounds only: no chain fallback.
    let rounds_unsure = Inputs {
        mode: Mode::Rounds,
        confident: false,
        ..base()
    };
    assert_eq!(select(&rounds_unsure, take), Choice::Mtp);
}

#[test]
fn clause_6_chain_needs_agreement_drafts_and_the_gate() {
    let chain = Inputs {
        confident: false,
        ..base()
    };
    assert_eq!(select(&chain, take), Choice::Chain);
    assert_eq!(select(&chain, || GateVerdict::Probe), Choice::Chain);
    assert_eq!(select(&chain, || GateVerdict::Skip), Choice::Mtp);
    let short = Inputs {
        match_len: Some(7),
        ..chain
    };
    assert_eq!(select(&short, take), Choice::Mtp);
    let at_min = Inputs {
        match_len: Some(8),
        ..chain
    };
    assert_eq!(select(&at_min, take), Choice::Chain);
    let no_mtp = Inputs {
        mtp_drafts: 0,
        ..chain
    };
    assert_eq!(select(&no_mtp, take), Choice::Mtp);
}

#[test]
fn lookup_len_takes_the_largest_bucket_the_slot_and_budget_allow() {
    assert_eq!(lookup_len(15, usize::MAX, 1000), 15);
    assert_eq!(lookup_len(15, 13, 1000), 12);
    assert_eq!(lookup_len(15, 9, 1000), 8);
    assert_eq!(lookup_len(15, 7, 1000), 6);
    assert_eq!(lookup_len(15, 5, 1000), 0);
    // remaining - 1 leaves room for the bonus token.
    assert_eq!(lookup_len(15, usize::MAX, 10), 8);
    assert_eq!(lookup_len(15, usize::MAX, 6), 0);
    // A max_len between buckets is itself a length.
    assert_eq!(lookup_len(10, usize::MAX, 1000), 10);
    assert_eq!(lookup_len(4, usize::MAX, 1000), 0);
}

/// A copy-heavy committed stream: a 60-token "file", some prose, then the
/// file reprinted. Returns (stream, where the reprint starts).
fn reprint_stream() -> (Vec<u32>, usize) {
    let file: Vec<u32> = (0..60).map(|i| 1000 + (i * 37 % 97) as u32).collect();
    let mut s = file.clone();
    s.extend(500..530u32);
    let at = s.len();
    s.extend_from_slice(&file);
    (s, at)
}

fn facts<'a>(committed: &'a [u32], drafts: &'a [u32]) -> RowFacts<'a> {
    let (last, head) = committed.split_last().expect("non-empty");
    RowFacts {
        head,
        last: *last,
        no_drafts: false,
        grammar: false,
        mtp_drafts: drafts,
        lookup_ok: false,
        capacity: usize::MAX,
        remaining: 1000,
    }
}

fn chain_cfg() -> Config {
    Config {
        mode: Mode::Chain,
        ..Config::default()
    }
}

#[test]
fn a_reprint_is_copied_and_fresh_text_is_left_to_mtp() {
    let (s, at) = reprint_stream();
    let cfg = chain_cfg();
    let mut st = CopyState::default();
    // Fresh text: no source of 8 agreeing tokens.
    assert_eq!(plan(&mut st, &cfg, &facts(&s[..40], &[1, 2, 3])), Plan::Mtp);
    // Ten tokens into the reprint: the source agrees on 10 >= 8, and the
    // chain is the file's next three tokens.
    let k = at + 10;
    match plan(&mut st, &cfg, &facts(&s[..k], &[1, 2, 3])) {
        Plan::Chain(t) => assert_eq!(t, s[k..k + 3]),
        other => panic!("expected a chain, got {other:?}"),
    }
}

#[test]
fn the_cursor_survives_accepted_rounds_and_dies_on_a_rejection() {
    let (s, at) = reprint_stream();
    let cfg = chain_cfg();
    let mut st = CopyState::default();
    let mut k = at + 10;
    // Accepted rounds: three copy tokens + the bonus commit each round.
    for _ in 0..4 {
        match plan(&mut st, &cfg, &facts(&s[..k], &[1, 2, 3])) {
            Plan::Chain(t) => assert_eq!(t, s[k..k + 3]),
            other => panic!("expected a chain, got {other:?}"),
        }
        assert!(st.has_cursor());
        k += 4;
    }
    assert!(st.gate_state().copy_avg().unwrap() > 0.99);
    // The target diverges from the file: the next sync clears the cursor.
    let mut diverged = s[..k].to_vec();
    diverged.push(9_999);
    let _ = plan(&mut st, &cfg, &facts(&diverged, &[1, 2, 3]));
    assert_eq!(st.index().tokens(), &diverged[..]);
    // A fresh lookup ran: the new suffix ends in an unseen token, so there is
    // no source and no cursor.
    assert!(!st.has_cursor());
}

#[test]
fn no_draft_or_rejected_token_ever_enters_the_index() {
    let (s, at) = reprint_stream();
    let cfg = Config {
        mode: Mode::Auto,
        ..Config::default()
    };
    let mut st = CopyState::default();
    let mut committed = s[..at + 12].to_vec();
    for round in 0..30u32 {
        // Drafts that are never committed (distinct sentinel values).
        let drafts = [77_000 + round, 78_000 + round, 79_000 + round];
        let _ = plan(&mut st, &cfg, &facts(&committed, &drafts));
        assert_eq!(st.index().tokens(), &committed[..], "round {round}");
        // The target commits its own token (a rejection of every draft).
        committed.push(60_000 + round);
    }
    for round in 0..30u32 {
        assert!(!st.index().tokens().contains(&(77_000 + round)));
        assert!(!st.index().tokens().contains(&(79_000 + round)));
    }
}

#[test]
fn a_lookup_round_needs_confidence_and_a_single_sequence() {
    // A long verbatim match (> 80 tokens) is confident from its first round.
    let file: Vec<u32> = (0..200).map(|i| 2000 + i).collect();
    let mut s = file.clone();
    s.extend(10..20u32);
    s.extend_from_slice(&file[..100]);
    let cfg = Config {
        mode: Mode::Auto,
        ..Config::default()
    };
    let mut st = CopyState::default();
    let f = RowFacts {
        lookup_ok: true,
        ..facts(&s, &[1, 2, 3])
    };
    match plan(&mut st, &cfg, &f) {
        Plan::Lookup(t) => assert_eq!(t, file[100..115]),
        other => panic!("expected a lookup round, got {other:?}"),
    }
    // The same stream at C > 1 (lookup_ok false) takes the chain instead.
    let mut st = CopyState::default();
    match plan(&mut st, &cfg, &facts(&s, &[1, 2, 3])) {
        Plan::Chain(t) => assert_eq!(t, file[100..103]),
        other => panic!("expected a chain, got {other:?}"),
    }
}

#[test]
fn grammar_and_no_drafts_rows_never_touch_the_state() {
    let (s, at) = reprint_stream();
    let cfg = chain_cfg();
    let mut st = CopyState::default();
    let g = RowFacts {
        grammar: true,
        ..facts(&s[..at + 10], &[1, 2, 3])
    };
    assert_eq!(plan(&mut st, &cfg, &g), Plan::Mtp);
    let n = RowFacts {
        no_drafts: true,
        ..facts(&s[..at + 10], &[1, 2, 3])
    };
    assert_eq!(plan(&mut st, &cfg, &n), Plan::NoSpec);
    assert!(st.index().is_empty());
}

#[test]
fn either_per_request_switch_takes_a_sequence_out_of_speculation() {
    // Lane L05's request: `seq.disable_mtp || seq.port.no_drafts`. The first
    // survives a swap-in today, the second once lane L27's hunk lands.
    let (mut a, _rx) = crate::scheduler::test_support::test_seq(vec![1], 8, None, 4);
    assert!(!super::no_drafts(&a));
    a.disable_mtp = true;
    assert!(super::no_drafts(&a));
    a.disable_mtp = false;
    a.port.no_drafts = true;
    assert!(super::no_drafts(&a));
    a.disable_mtp = true;
    assert!(super::no_drafts(&a));
}

#[test]
fn the_shadow_proposal_warms_the_estimate_while_mtp_runs() {
    let (s, at) = reprint_stream();
    let cfg = Config {
        mode: Mode::Chain,
        // An agreement no source here reaches: every round stays MTP.
        min_match: 64,
        ..Config::default()
    };
    let mut st = CopyState::with_index(CopyIndex::new());
    let before = st.estimates().clone();
    let mut k = at + 10;
    for _ in 0..5 {
        assert_eq!(plan(&mut st, &cfg, &facts(&s[..k], &[1, 2, 3])), Plan::Mtp);
        k += 3;
    }
    assert_ne!(st.estimates(), &before, "copy proposals were scored");
}

#[test]
fn config_resolves_cli_then_row_and_refuses_bad_values() {
    let none = |_: &str| None::<String>;
    let c = config::resolve(&Cli::default(), "off", false, none).unwrap();
    assert_eq!(c, Config::default());
    let c = config::resolve(&Cli::default(), "auto", true, none).unwrap();
    assert_eq!((c.mode, c.mode_source), (Mode::Auto, "env"));
    let cli = Cli {
        mode: Some("chain".into()),
        min_match: Some(12),
        max_len: Some(10),
        advantage: Some(2.0),
    };
    let c = config::resolve(&cli, "auto", true, none).unwrap();
    assert_eq!(
        (c.mode, c.mode_source, c.min_match, c.max_len, c.advantage),
        (Mode::Chain, "cli", 12, 10, 2.0)
    );
    let bad = |cli: Cli| config::resolve(&cli, "off", false, none).is_err();
    assert!(bad(Cli {
        mode: Some("tree".into()),
        ..Cli::default()
    }));
    assert!(bad(Cli {
        min_match: Some(1),
        ..Cli::default()
    }));
    assert!(bad(Cli {
        min_match: Some(65),
        ..Cli::default()
    }));
    assert!(bad(Cli {
        max_len: Some(16),
        ..Cli::default()
    }));
    assert!(bad(Cli {
        advantage: Some(0.0),
        ..Cli::default()
    }));
    assert!(bad(Cli {
        advantage: Some(f32::NAN),
        ..Cli::default()
    }));
    // Environment knobs apply when the flag is absent; junk is refused.
    let env = |name: &str| match name {
        config::MIN_MATCH_ENV => Some("12".to_string()),
        config::ADVANTAGE_ENV => Some("1.5".to_string()),
        _ => None,
    };
    let c = config::resolve(&Cli::default(), "chain", true, env).unwrap();
    assert_eq!((c.min_match, c.advantage), (12, 1.5));
    let junk = |name: &str| (name == config::MODE_ENV).then(|| "sometimes".to_string());
    assert!(config::resolve(&Cli::default(), "off", false, junk).is_err());
    let junk_int = |name: &str| (name == config::MAX_LEN_ENV).then(|| "lots".to_string());
    assert!(config::resolve(&Cli::default(), "off", false, junk_int).is_err());
}

#[test]
fn modes_parse_case_insensitively_and_round_trip() {
    for m in config::MODES {
        let mode = Mode::parse(m).unwrap();
        assert_eq!(mode.as_str(), m);
        assert_eq!(Mode::parse(&m.to_uppercase()), Some(mode));
    }
    assert_eq!(Mode::parse(" Chain "), Some(Mode::Chain));
    assert_eq!(Mode::parse("tree"), None);
    assert!(Mode::Auto.chains() && Mode::Auto.rounds());
    assert!(Mode::Chain.chains() && !Mode::Chain.rounds());
    assert!(!Mode::Rounds.chains() && Mode::Rounds.rounds());
    assert!(!Mode::Off.chains() && !Mode::Off.rounds());
}
