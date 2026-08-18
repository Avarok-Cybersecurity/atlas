// SPDX-License-Identifier: AGPL-3.0-only

//! `agentic-webserver`'s tunables.
//!
//! Split out of `mod.rs` for the repo's 500-line ceiling — the same reason
//! `score.rs` was split out before it. The bodies are unchanged; only their
//! home moved.
//!
//! ★ Read the two budget specs together. `wall_budget_s` is a BLOWUP bound and
//! `s_per_turn_budget` is the SPEED bound, and neither schema default is what a
//! gated run enforces: `apply_threshold_params` substitutes the selected
//! variant's committed `[benchmarks.metrics.*]` bound over both, and
//! `check_record` scores the recorded metric against that same committed
//! number. Editing a default here changes only a bare `spark benchmark run`
//! with no checkpoint.

use crate::{ParamKind, ParamSpec, ParamValue};

pub(super) fn parameters() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new(
            "iterations",
            "Iterations",
            "How many independent agent runs. The gate tier is 10; use 1 for a smoke test.",
            ParamKind::Int { min: 1, max: 50 },
            ParamValue::Int(10),
        ),
        ParamSpec::new(
            "wall_budget_s",
            "Σ wall budget",
            "Total agent seconds across all iterations, scorer INCLUDED, \
             before the gate fails. This is a BLOWUP/DEGENERACY bound, not a \
             speed one — `s_per_turn_budget` is the speed bound. It stays a \
             schema default of 1000 s because a bare run has no variant to \
             read from; every MEASURED variant overrides it from its own \
             BENCH.toml (see threshold_params), and that committed number, \
             not this one, is what --pull-request-gate and the TUI enforce.",
            ParamKind::Float {
                min: 1.0,
                max: 100_000.0,
            },
            ParamValue::Float(1000.0),
        ),
        ParamSpec::new(
            "s_per_turn_budget",
            "Seconds per turn",
            "The agent's own seconds ÷ agent turns before the gate fails — \
             the SPEED bound. **0.0 means NON-GATING**, and that is the \
             schema default ON PURPOSE: per-turn cost is the most \
             model-dependent number this benchmark produces (6.8 s/turn on \
             the 35B MoE against 18-40 s/turn on the dense 27B), so a single \
             schema figure cannot be right for two variants and a wrong one \
             would fail every run of the model it was not measured on. A \
             variant is gated only once it commits a measured \
             `[benchmarks.metrics.s_per_turn]` bound to its BENCH.toml. \
             \
             Why per-turn rather than Σwall: turn count is drawn by the \
             agent, not the engine. Across five 10/10 tiers Σwall spanned \
             774-1039 s (34%) while s/turn spanned 6.83-7.22 (5.7%) on one \
             box, and the two tiers the old 1000 s wall bound REJECTED were \
             respectively the slowest and the FASTEST per turn — it ranked \
             them backwards. \
             \
             Why not tokens, which would be better still: `decode_tps` is \
             now recorded for exactly that reason, but no variant has a \
             measured token-rate bound yet and inventing one is the mistake \
             this change undoes. Ratchet to it once tiers exist.",
            ParamKind::Float {
                min: 0.0,
                max: 10_000.0,
            },
            ParamValue::Float(0.0),
        ),
        ParamSpec::new(
            "max_turns",
            "Max turns",
            "Tool-calling rounds per iteration before the agent is stopped.",
            ParamKind::Int { min: 1, max: 200 },
            ParamValue::Int(40),
        ),
        ParamSpec::new(
            "command_timeout_s",
            "Command timeout",
            "Seconds a single agent shell command may run before it is killed.",
            ParamKind::Int { min: 5, max: 3600 },
            ParamValue::Int(180),
        ),
        ParamSpec::new(
            "build_timeout_s",
            "Scorer build timeout",
            "Seconds the scorer's cargo build may take. A cold dependency tree is slow.",
            ParamKind::Int { min: 30, max: 3600 },
            ParamValue::Int(600),
        ),
        ParamSpec::new(
            "serve_timeout_s",
            "Ping timeout",
            "Seconds to wait for /ping to answer 'pong' after the server is started. \
             The harness's own budget is 15s (score_run.py::webserver_test), so this is \
             already the looser of the two — lowering it would not match anything.",
            ParamKind::Int { min: 5, max: 300 },
            ParamValue::Int(30),
        ),
        ParamSpec::new(
            "max_tokens",
            "Max tokens per turn",
            "Output budget for one model turn.",
            ParamKind::Int {
                min: 256,
                max: 32_768,
            },
            ParamValue::Int(8192),
        ),
        ParamSpec::new(
            "request_timeout_s",
            "Request timeout",
            "Seconds before a single model call is abandoned.",
            ParamKind::Int { min: 10, max: 3600 },
            ParamValue::Int(900),
        ),
    ]
}
