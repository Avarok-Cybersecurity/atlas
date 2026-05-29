// SPDX-License-Identifier: AGPL-3.0-only

use crate::tool_parser;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ToolMode {
    pub(super) required: bool,
    pub(super) suppressed_for_turn: bool,
}

pub(super) fn resolve_tool_mode(
    tools_active: bool,
    tool_choice: Option<&tool_parser::ToolChoice>,
    parser_name: Option<&str>,
    requested_suppression: bool,
) -> ToolMode {
    if !tools_active {
        return ToolMode {
            required: false,
            suppressed_for_turn: false,
        };
    }

    let explicit_required = tool_choice.is_some_and(|tc| {
        matches!(tc, tool_parser::ToolChoice::Mode(m) if m == "required")
            || matches!(tc, tool_parser::ToolChoice::Specific { .. })
    });
    // Bare JSON has no envelope token that can serve as a safe auto trigger;
    // MiniMax XML does, so it can follow normal `tool_choice=auto` semantics.
    let parser_required = matches!(parser_name, Some("bare_json"));
    let suppressed_for_turn = requested_suppression && !explicit_required;

    ToolMode {
        required: !suppressed_for_turn && (explicit_required || parser_required),
        suppressed_for_turn,
    }
}

#[cfg(test)]
mod tests {
    use super::{ToolMode, resolve_tool_mode};
    use crate::tool_parser::{ToolChoice, ToolChoiceFunction};

    #[test]
    fn loop_suppression_disables_minimax_auto_for_one_turn() {
        let mode = resolve_tool_mode(true, None, Some("minimax_xml"), true);

        assert_eq!(
            mode,
            ToolMode {
                required: false,
                suppressed_for_turn: true,
            }
        );
    }

    #[test]
    fn minimax_parser_uses_auto_without_loop_suppression() {
        let mode = resolve_tool_mode(true, None, Some("minimax_xml"), false);

        assert_eq!(
            mode,
            ToolMode {
                required: false,
                suppressed_for_turn: false,
            }
        );
    }

    #[test]
    fn bare_json_parser_still_requires_without_loop_suppression() {
        let mode = resolve_tool_mode(true, None, Some("bare_json"), false);

        assert_eq!(
            mode,
            ToolMode {
                required: true,
                suppressed_for_turn: false,
            }
        );
    }

    #[test]
    fn explicit_required_ignores_loop_suppression() {
        let choice = ToolChoice::Mode("required".to_string());
        let mode = resolve_tool_mode(true, Some(&choice), Some("minimax_xml"), true);

        assert_eq!(
            mode,
            ToolMode {
                required: true,
                suppressed_for_turn: false,
            }
        );
    }

    #[test]
    fn specific_function_ignores_loop_suppression() {
        let choice = ToolChoice::Specific {
            function: ToolChoiceFunction {
                name: "session_search".to_string(),
            },
        };
        let mode = resolve_tool_mode(true, Some(&choice), Some("minimax_xml"), true);

        assert_eq!(
            mode,
            ToolMode {
                required: true,
                suppressed_for_turn: false,
            }
        );
    }

    #[test]
    fn inactive_tools_ignore_requested_suppression() {
        let mode = resolve_tool_mode(false, None, Some("minimax_xml"), true);

        assert_eq!(
            mode,
            ToolMode {
                required: false,
                suppressed_for_turn: false,
            }
        );
    }
}
