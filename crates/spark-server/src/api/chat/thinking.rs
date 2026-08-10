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
        },
        max_tokens,
        tools_active,
    )
}

/// Server/model policy inputs, split from `AppState` so the resolution
/// core is a pure function.
struct Policy {
    disable_thinking: bool,
    model_default: bool,
    thinking_in_tools: bool,
    max_thinking_budget: u32,
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
    } else {
        None
    };
    (et, budget)
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
