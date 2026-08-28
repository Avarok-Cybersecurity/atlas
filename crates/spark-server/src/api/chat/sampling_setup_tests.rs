// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for `sampling_setup`: tool-choice/parser coupling and the
//! grammar escape hatch. A sibling file (the `msg_entry_tests.rs`
//! pattern) so the parent stays under the 500-LoC cap.

#[cfg(test)]
mod tests {
    // `super::super::` — this file is a sibling module of `sampling_setup`,
    // so one `super` only reaches `sampling_setup_tests` itself.
    use super::super::{tool_choice_required_for_parser, tool_grammar_escape_applies};

    use crate::tool_parser::{ToolChoice, ToolChoiceFunction};

    /// The escape hatch's own help text scopes it to `tool_choice="auto"`.
    #[test]
    fn the_escape_hatch_applies_in_auto_mode() {
        let required = tool_choice_required_for_parser(true, None, Some("qwen3_coder"));

        assert!(!required);
        assert!(tool_grammar_escape_applies(true, required));
    }

    /// `required` is *implemented by* the grammar, so the hatch must not
    /// remove it — the escape hatch is a model property, `required` is a
    /// request property, and the request wins.
    #[test]
    fn required_mode_keeps_the_grammar_despite_the_escape_hatch() {
        let choice = ToolChoice::Mode("required".to_string());
        let required = tool_choice_required_for_parser(true, Some(&choice), Some("qwen3_coder"));

        assert!(required);
        assert!(!tool_grammar_escape_applies(true, required));
    }

    #[test]
    fn specific_function_keeps_the_grammar_despite_the_escape_hatch() {
        let choice = ToolChoice::Specific {
            function: ToolChoiceFunction {
                name: "memory".to_string(),
            },
        };
        let required = tool_choice_required_for_parser(true, Some(&choice), Some("qwen3_coder"));

        assert!(required);
        assert!(!tool_grammar_escape_applies(true, required));
    }

    /// minimax_xml's grammar is the anti-corruption frame, not just a
    /// tool-choice enforcer, so the hatch must not remove it either.
    #[test]
    fn minimax_xml_keeps_the_grammar_despite_the_escape_hatch() {
        let required = tool_choice_required_for_parser(true, None, Some("minimax_xml"));

        assert!(required);
        assert!(!tool_grammar_escape_applies(true, required));
    }

    /// The fix must not switch the grammar ON for models that never asked
    /// for the hatch — with the hatch off, nothing about this changes.
    #[test]
    fn the_hatch_being_off_is_unaffected_by_tool_choice() {
        assert!(!tool_grammar_escape_applies(false, false));
        assert!(!tool_grammar_escape_applies(false, true));
    }

    #[test]
    fn bare_json_auto_uses_triggered_grammar() {
        assert!(!tool_choice_required_for_parser(
            true,
            None,
            Some("bare_json")
        ));
    }

    #[test]
    fn bare_json_required_mode_enforces_from_first_token() {
        let choice = ToolChoice::Mode("required".to_string());

        assert!(tool_choice_required_for_parser(
            true,
            Some(&choice),
            Some("bare_json")
        ));
    }

    #[test]
    fn specific_function_enforces_from_first_token() {
        let choice = ToolChoice::Specific {
            function: ToolChoiceFunction {
                name: "memory".to_string(),
            },
        };

        assert!(tool_choice_required_for_parser(
            true,
            Some(&choice),
            Some("bare_json")
        ));
    }

    #[test]
    fn minimax_xml_remains_parser_required() {
        assert!(tool_choice_required_for_parser(
            true,
            None,
            Some("minimax_xml")
        ));
    }

    #[test]
    fn inactive_tools_are_not_required() {
        assert!(!tool_choice_required_for_parser(
            false,
            None,
            Some("bare_json")
        ));
    }
}
