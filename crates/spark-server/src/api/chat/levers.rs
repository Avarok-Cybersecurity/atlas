// SPDX-License-Identifier: AGPL-3.0-only

//! [`ChatLevers`] — the chat request path's model- and deployment-scoped
//! switches, resolved once when the server is built and carried on
//! [`crate::AppState`].
//!
//! Each of these was a private `OnceLock<bool>` beside its one or two call
//! sites. Individually harmless; collectively they are the reason the request
//! path could not be exercised in a test without mutating the process
//! environment, and the reason a second model loaded into a live process would
//! inherit the first one's configuration.
//!
//! Grouped rather than added to `AppState` one field at a time so the request
//! path has a single, greppable answer to "what can vary here", and so the
//! renderer can be handed [`PromptLevers`] alone — the narrow sub-struct it
//! actually needs — instead of the whole server state.

use crate::tool_parser::PromptLevers;

/// Chat-path levers for one server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ChatLevers {
    /// Prompt-rendering decisions, handed to `ToolCallParser::system_prompt`
    /// and the chat template.
    pub prompt: PromptLevers,
    /// `ATLAS_BASH_WANDER_WATCHDOG=1` — append the BW1 steering hint when the
    /// agent has made many tool calls with no productive file output.
    /// Default-off (PCND).
    pub bash_wander: bool,
    /// `ATLAS_CHAT_PHASE_TIMING=1` — emit per-phase `CHAT_PHASE` timing lines
    /// for the handler and the prepare stage. Diagnostic; default-off, so the
    /// production path is unchanged.
    pub phase_timing: bool,
    /// MODEL.toml `[behavior] disable_cwd_hint_injection` — suppress the
    /// `<environment>working_directory: …</environment>` hint appended to the
    /// system message when tools are active. Model-scoped: some checkpoints
    /// treat the injected block as conversation content and degrade.
    pub disable_cwd_hint_injection: bool,
}

impl ChatLevers {
    /// Every lever off — the default production shape, and what the request
    /// path tests build against so none of them has to touch the environment.
    pub const OFF: Self = Self {
        prompt: PromptLevers::OFF,
        bash_wander: false,
        phase_timing: false,
        disable_cwd_hint_injection: false,
    };

    /// Resolve from the environment plus this model's `[behavior]` table.
    /// Called once, when the server's `AppState` is built.
    pub fn resolve(tscg: bool, disable_cwd_hint_injection: bool) -> Self {
        Self {
            prompt: PromptLevers::new(tscg),
            bash_wander: std::env::var("ATLAS_BASH_WANDER_WATCHDOG").as_deref() == Ok("1"),
            phase_timing: std::env::var("ATLAS_CHAT_PHASE_TIMING").as_deref() == Ok("1"),
            disable_cwd_hint_injection,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_is_the_default() {
        assert_eq!(ChatLevers::default(), ChatLevers::OFF);
    }

    #[test]
    fn the_model_behavior_reaches_the_renderer() {
        // `resolve` reads the env for the two diagnostics but takes `tscg`
        // from the caller, because it is MODEL.toml state and not a
        // process-wide setting.
        assert!(ChatLevers::resolve(true, false).prompt.tscg);
        assert!(!ChatLevers::resolve(false, false).prompt.tscg);
        assert!(ChatLevers::resolve(false, true).disable_cwd_hint_injection);
    }
}
