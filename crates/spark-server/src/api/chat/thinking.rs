// SPDX-License-Identifier: AGPL-3.0-only
//
// Resolve `(enable_thinking, thinking_budget)` for a single request
// from the neutral thinking directive. Precedence (highest wins):
//   1. `--disable-thinking` CLI flag (forces OFF for every request)
//   2. The request directive (client channels resolved at the API edge,
//      or the server-level default directive when the client is silent)
//   3. MODEL.toml `[behavior].thinking_default`
//
// Lifted out of `chat::chat_completions_inner` (wave 4g); flipped from
// the OpenAI wire request to `ir::ThinkingDirective` (IR migration).

use std::sync::Arc;

use crate::AppState;
use crate::ir::ThinkingDirective;

pub(super) fn resolve_thinking(
    state: &Arc<AppState>,
    directive: ThinkingDirective,
    max_tokens: u32,
    tools_active: bool,
) -> (bool, Option<u32>) {
    resolve(
        directive,
        Policy {
            disable_thinking: state.disable_thinking,
            model_default: state.behavior.thinking_default,
            thinking_in_tools: state.behavior.thinking_in_tools,
            max_thinking_budget: state.behavior.max_thinking_budget,
            effort_capped_at_ceiling: state.behavior.effort_capped_at_ceiling,
            cap_at_max_tokens: state.behavior.cap_thinking_at_max_tokens,
        },
        max_tokens,
        tools_active,
    )
}

/// Server/model policy inputs, split from `AppState` so the resolution
/// core is a pure function.
#[derive(Clone, Copy)]
struct Policy {
    disable_thinking: bool,
    model_default: bool,
    thinking_in_tools: bool,
    max_thinking_budget: u32,
    /// MODEL.toml `[behavior].effort_capped_at_ceiling`: clamp the effort
    /// ladder at E (high/xhigh -> E instead of 2E/4E). Never touches an
    /// explicit client token budget. See `behavior_defaults.rs`.
    effort_capped_at_ceiling: bool,
    /// When false, `max_thinking_budget` is the SOLE thinking cap (vLLM
    /// single-budget); the 90%-of-max_tokens clamp is skipped.
    cap_at_max_tokens: bool,
}

fn resolve(
    directive: ThinkingDirective,
    policy: Policy,
    max_tokens: u32,
    tools_active: bool,
) -> (bool, Option<u32>) {
    if policy.disable_thinking {
        return (false, None);
    }
    let (et, tb) = match directive {
        // No client/server directive → MODEL.toml decides. `None` budget
        // defers to the per-model `max_thinking_budget` below rather than
        // a conservative hardcoded default.
        ThinkingDirective::Unspecified => (policy.model_default, None),
        ThinkingDirective::Off => (false, None),
        ThinkingDirective::On { budget } => (true, budget),
        // A qualitative effort level scales against the model's effective
        // ceiling HERE — the one place that knows it — so MODEL.toml and
        // `--max-thinking-budget` govern what "medium" means. Resolved at
        // the wire edge it was a hardcoded absolute (medium = 256) that no
        // server knob could reach (#328 family).
        ThinkingDirective::OnEffort(level) => (
            true,
            Some(effort_budget(
                level,
                policy.max_thinking_budget,
                policy.effort_capped_at_ceiling,
            )),
        ),
    };
    // `thinking_in_tools=false` is the MODEL.toml DEFAULT for tool-
    // active turns: it suppresses thinking when the client is silent.
    // An explicit directive (enabled OR disabled — including the
    // server-level default directive) still wins.
    let et = if tools_active && !policy.thinking_in_tools && !directive.is_explicit() {
        false
    } else {
        et
    };
    let budget = if et {
        let b = tb.unwrap_or(policy.max_thinking_budget);
        if !policy.cap_at_max_tokens {
            // vLLM single-budget: `max_thinking_budget` (or an explicit
            // client budget) is the sole cap. Reasoning may use the full
            // generation budget; there is no second, max_tokens-derived cap.
            Some(b)
        } else {
            // 2026-05-23 sweep: dropped the 70% special case for
            // `tools_active && thinking_in_tools` (previously 7/10, now
            // 9/10 uniformly). With `thinking_in_tools=true` as the
            // project-wide default the 70% branch fired on every tool turn
            // and silently undermined the MODEL.toml `max_thinking_budget`
            // bump (opencode-style requests at max_tokens=2048 capped to
            // 1433 instead of 2048). 90% leaves headroom for content +
            // tool args without crippling reasoning chains that now run
            // naturally after the F1 reflection-penalty removal.
            let safety_cap_pct = 9;
            let max = ((max_tokens * safety_cap_pct) / 10).max(1);
            Some(b.min(max))
        }
    } else {
        None
    };
    (et, budget)
}

/// Token budget for a qualitative effort level, as a ratio of the model's
/// effective `max_thinking_budget` E: minimal=E/4, low=E/2, medium=E,
/// high=2E, xhigh=4E.
///
/// At the built-in default E=256 these are exactly the historical absolute
/// ladder (64/128/256/512/1024), so a server with no MODEL.toml value and
/// no `--max-thinking-budget` resolves byte-identically to the pre-symbolic
/// code — including high/xhigh EXCEEDING the ceiling, which the old ladder
/// also did (512 > 256). The `cap_at_max_tokens` clamp below still applies
/// on top, unchanged.
///
/// `capped_at_ceiling` (MODEL.toml `[behavior].effort_capped_at_ceiling`,
/// default false) clamps the ladder at E for models with MEASURED
/// degradation above their ceiling (budget non-monotonicity — Qwen3.5-397B,
/// 2026-05-07 sweep: 256 thinking tokens scores worse than 128). It binds
/// only this server-policy ladder; explicit client budgets never pass
/// through here.
fn effort_budget(
    level: crate::ir::EffortLevel,
    max_thinking_budget: u32,
    capped_at_ceiling: bool,
) -> u32 {
    use crate::ir::EffortLevel;
    let e = max_thinking_budget.max(1);
    let b = match level {
        EffortLevel::Minimal => (e / 4).max(1),
        EffortLevel::Low => (e / 2).max(1),
        EffortLevel::Medium => e,
        EffortLevel::High => e.saturating_mul(2),
        EffortLevel::XHigh => e.saturating_mul(4),
    };
    if capped_at_ceiling { b.min(e) } else { b }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> Policy {
        Policy {
            disable_thinking: false,
            model_default: false,
            thinking_in_tools: true,
            max_thinking_budget: 2048,
            effort_capped_at_ceiling: false,
            cap_at_max_tokens: true,
        }
    }

    #[test]
    fn kill_switch_overrides_everything() {
        let (et, tb) = resolve(
            ThinkingDirective::On { budget: Some(512) },
            Policy {
                disable_thinking: true,
                ..policy()
            },
            4096,
            false,
        );
        assert!(!et);
        assert!(tb.is_none());
    }

    #[test]
    fn unspecified_falls_to_model_default() {
        let (et, tb) = resolve(
            ThinkingDirective::Unspecified,
            Policy {
                model_default: true,
                ..policy()
            },
            4096,
            false,
        );
        assert!(et);
        // Defers to max_thinking_budget, capped at 90% of max_tokens.
        assert_eq!(tb, Some(2048));

        let (et, tb) = resolve(ThinkingDirective::Unspecified, policy(), 4096, false);
        assert!(!et);
        assert!(tb.is_none());
    }

    #[test]
    fn explicit_budget_capped_at_90_pct_of_max_tokens() {
        let (et, tb) = resolve(
            ThinkingDirective::On { budget: Some(4096) },
            policy(),
            1000,
            false,
        );
        assert!(et);
        assert_eq!(tb, Some(900));
    }

    #[test]
    fn budgetless_on_defers_to_model_cap() {
        let (et, tb) = resolve(
            ThinkingDirective::On { budget: None },
            policy(),
            4096,
            false,
        );
        assert!(et);
        assert_eq!(tb, Some(2048));
    }

    #[test]
    fn effort_ladder_at_default_ceiling_matches_the_historical_absolutes() {
        // Regression pin for the symbolic-effort switch: with the built-in
        // default ceiling (256, `ModelBehavior::default()`), every effort
        // level resolves to EXACTLY the number the old wire-edge ladder
        // hardcoded. Fails if anyone re-anchors the ratios.
        use crate::ir::EffortLevel::*;
        let default_ceiling = Policy {
            max_thinking_budget: 256,
            cap_at_max_tokens: false,
            ..policy()
        };
        for (level, historical) in [
            (Minimal, 64),
            (Low, 128),
            (Medium, 256),
            (High, 512),
            (XHigh, 1024),
        ] {
            let (et, tb) = resolve(
                ThinkingDirective::OnEffort(level),
                default_ceiling,
                4096,
                false,
            );
            assert!(et);
            assert_eq!(tb, Some(historical), "level={level:?}");
        }
    }

    #[test]
    fn effort_ladder_scales_with_the_operator_ceiling() {
        // The #328-family fix: `--max-thinking-budget` (folded into
        // `Policy::max_thinking_budget` by serve) now GOVERNS what a client's
        // qualitative effort means, instead of being silently outranked by a
        // hardcoded 256. Daniel's repro: flag 16256 + harness "medium".
        use crate::ir::EffortLevel::*;
        let operator = Policy {
            max_thinking_budget: 16256,
            cap_at_max_tokens: false,
            ..policy()
        };
        let (et, tb) = resolve(ThinkingDirective::OnEffort(Medium), operator, 32768, false);
        assert!(et);
        assert_eq!(tb, Some(16256));
        let (_, tb) = resolve(ThinkingDirective::OnEffort(Minimal), operator, 32768, false);
        assert_eq!(tb, Some(4064));
        // The 90%-of-max_tokens safety clamp still applies on top when the
        // model keeps `cap_thinking_at_max_tokens`.
        let capped = Policy {
            max_thinking_budget: 16256,
            ..policy()
        };
        let (_, tb) = resolve(ThinkingDirective::OnEffort(Medium), capped, 1000, false);
        assert_eq!(tb, Some(900));
    }

    #[test]
    fn effort_cap_defaults_off_so_shipping_it_changes_nothing() {
        // The built-in default MUST stay `false`: with it, the parity test
        // above proves every non-opted-in model resolves byte-identically
        // to the pre-clamp code. Reads the same `ModelBehavior::default()`
        // the server boots with (SSOT constant in behavior_defaults.rs),
        // not a copy of the literal.
        assert!(!atlas_kernels::ModelBehavior::default().effort_capped_at_ceiling);
    }

    #[test]
    fn effort_cap_clamps_the_ladder_at_the_ceiling() {
        // Opt-in shape for measured budget non-monotonicity (Qwen3.5-397B,
        // 2026-05-07 sweep: 256 thinking tokens scores WORSE than 128): a
        // client's boilerplate high/xhigh must not exceed the model's E.
        use crate::ir::EffortLevel::*;
        let p = Policy {
            max_thinking_budget: 128,
            effort_capped_at_ceiling: true,
            cap_at_max_tokens: false,
            ..policy()
        };
        for (level, expected) in [
            (Minimal, 32),
            (Low, 64),
            (Medium, 128),
            (High, 128),
            (XHigh, 128),
        ] {
            let (et, tb) = resolve(ThinkingDirective::OnEffort(level), p, 4096, false);
            assert!(et);
            assert_eq!(tb, Some(expected), "level={level:?}");
        }
    }

    #[test]
    fn effort_cap_never_touches_an_explicit_client_budget() {
        // Precedence stays: explicit client token budget > effort mapping >
        // server flag > MODEL.toml > default. The clamp binds ONLY the
        // effort ladder — a client that states a number gets that number.
        let p = Policy {
            max_thinking_budget: 128,
            effort_capped_at_ceiling: true,
            cap_at_max_tokens: false,
            ..policy()
        };
        let (et, tb) = resolve(ThinkingDirective::On { budget: Some(4096) }, p, 8192, false);
        assert!(et);
        assert_eq!(tb, Some(4096));
    }

    #[test]
    fn effort_is_explicit_for_thinking_in_tools_purposes() {
        // An effort-carrying request stated thinking intent, so the
        // `thinking_in_tools=false` tools-suppression must NOT apply —
        // parity with the old `On { budget }` encoding of the ladder.
        let p = Policy {
            thinking_in_tools: false,
            ..policy()
        };
        let (et, _) = resolve(
            ThinkingDirective::OnEffort(crate::ir::EffortLevel::Medium),
            p,
            4096,
            true,
        );
        assert!(et);
    }

    #[test]
    fn cap_at_max_tokens_false_skips_the_90pct_clamp() {
        // vLLM single-budget: max_thinking_budget is the SOLE cap; a small
        // max_tokens no longer clamps thinking down to 90% of it.
        let no_cap = Policy {
            cap_at_max_tokens: false,
            ..policy()
        };
        // Explicit client budget passes through untouched (was 900 under cap).
        let (et, tb) = resolve(
            ThinkingDirective::On { budget: Some(4096) },
            no_cap,
            1000,
            false,
        );
        assert!(et);
        assert_eq!(tb, Some(4096));
        // Budgetless On → max_thinking_budget, not clamped to 0.9*max_tokens.
        let (_, tb) = resolve(ThinkingDirective::On { budget: None }, no_cap, 1000, false);
        assert_eq!(tb, Some(2048));
    }

    #[test]
    fn tools_suppression_only_when_client_silent() {
        let no_tools_thinking = Policy {
            model_default: true,
            thinking_in_tools: false,
            ..policy()
        };
        // Silent client on a tool turn → suppressed.
        let (et, _) = resolve(
            ThinkingDirective::Unspecified,
            Policy {
                ..no_tools_thinking
            },
            4096,
            true,
        );
        assert!(!et);
        // Explicit enable survives the suppression.
        let (et, _) = resolve(
            ThinkingDirective::On { budget: None },
            Policy {
                model_default: true,
                thinking_in_tools: false,
                ..policy()
            },
            4096,
            true,
        );
        assert!(et);
        // Explicit disable is likewise respected (no double negation).
        let (et, _) = resolve(
            ThinkingDirective::Off,
            Policy {
                model_default: true,
                thinking_in_tools: false,
                ..policy()
            },
            4096,
            true,
        );
        assert!(!et);
    }

    #[test]
    fn explicit_off_wins_over_model_default() {
        let (et, tb) = resolve(
            ThinkingDirective::Off,
            Policy {
                model_default: true,
                ..policy()
            },
            4096,
            false,
        );
        assert!(!et);
        assert!(tb.is_none());
    }
}
