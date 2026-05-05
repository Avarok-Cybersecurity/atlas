// SPDX-License-Identifier: AGPL-3.0-only

//! Tool error recovery hints.
//!
//! When a tool call returns an error, hint injectors append short guidance
//! to the tool response to help the model recover. Each injector targets
//! a specific error class (file path errors, permission errors, etc.) and
//! provides escalating hints based on how many consecutive errors have occurred.

/// Context passed to each hint injector for a single tool response.
pub struct HintContext<'a> {
    /// The raw tool response text (before `<tool_response>` wrapping).
    pub text: &'a str,
    /// Number of consecutive tool errors seen so far (1-indexed).
    /// Reset to 0 on any successful tool response.
    pub consecutive_errors: u32,
}

/// Trait for injecting recovery hints into failed tool responses.
///
/// Implementations detect specific error patterns and generate targeted
/// guidance to help the model break out of retry loops.
pub trait HintInjector: Send + Sync {
    /// Check if this injector is relevant to the given tool response.
    /// Called for every tool response — should be cheap (substring checks).
    fn is_relevant(&self, ctx: &HintContext) -> bool;

    /// Generate a hint string to append to the tool response.
    /// Only called when `is_relevant` returned true.
    /// Return empty string to skip injection.
    fn hint(&self, ctx: &HintContext) -> String;
}

// ─── Concrete Implementations ────────────────────────────────────────

/// Detects file-as-directory errors (EISDIR) and guides toward correct paths.
pub struct WritePathHint;

impl HintInjector for WritePathHint {
    fn is_relevant(&self, ctx: &HintContext) -> bool {
        ctx.text.contains("EISDIR") || ctx.text.contains("illegal operation on a directory")
    }

    fn hint(&self, ctx: &HintContext) -> String {
        if ctx.consecutive_errors >= 3 {
            "\n\n<CRITICAL>\n\
             STOP using the Write tool — it has failed 3+ times with a directory path.\n\
             You MUST use Bash instead: cat > ./path/to/file.ext << 'EOF'\n\
             ...content...\nEOF\n\
             Write/Edit file_path MUST be a FILE (e.g. ./dir/file.txt), NEVER a directory (./dir).\n\
             </CRITICAL>"
                .to_string()
        } else {
            "\n\nHint: file_path is a directory, not a file. \
             Use a full path like ./dir/file.txt (not ./dir). \
             If Write keeps failing, use Bash: cat > file << 'EOF'"
                .to_string()
        }
    }
}

/// Detects file-not-found errors (ENOENT) and suggests creating the directory.
pub struct FileNotFoundHint;

impl HintInjector for FileNotFoundHint {
    fn is_relevant(&self, ctx: &HintContext) -> bool {
        ctx.text.contains("ENOENT") || ctx.text.contains("No such file or directory")
    }

    fn hint(&self, ctx: &HintContext) -> String {
        if ctx.consecutive_errors >= 3 {
            "\n\n<CRITICAL>\n\
             The file or directory does not exist. Create the parent directory first:\n\
             mkdir -p ./parent/dir && cat > ./parent/dir/file.ext << 'EOF'\n\
             ...content...\nEOF\n\
             </CRITICAL>"
                .to_string()
        } else {
            "\n\nHint: File or directory not found. \
             Create the parent directory first with: mkdir -p ./parent/dir"
                .to_string()
        }
    }
}

/// Detects Read tool failures (file not found, permission issues).
pub struct ReadErrorHint;

impl HintInjector for ReadErrorHint {
    fn is_relevant(&self, ctx: &HintContext) -> bool {
        // Read-specific patterns: file doesn't exist, can't open, etc.
        (ctx.text.contains("ENOENT") || ctx.text.contains("No such file"))
            && (ctx.text.contains("read") || ctx.text.contains("open"))
    }

    fn hint(&self, ctx: &HintContext) -> String {
        if ctx.consecutive_errors >= 3 {
            "\n\nHint: File does not exist. Check the path with: ls ./dir/ \
             or find . -name 'filename'. Do NOT keep reading non-existent files."
                .to_string()
        } else {
            "\n\nHint: File not found. Verify the path exists before reading.".to_string()
        }
    }
}

/// Detects Task/agent delegation failures (unknown agent type).
pub struct TaskDelegationHint;

impl HintInjector for TaskDelegationHint {
    fn is_relevant(&self, ctx: &HintContext) -> bool {
        ctx.text.contains("Unknown agent type")
            || ctx.text.contains("not a valid agent type")
            || ctx.text.contains("agent type")
    }

    fn hint(&self, ctx: &HintContext) -> String {
        if ctx.consecutive_errors >= 2 {
            "\n\nHint: STOP using the Task tool — use Bash, Write, Read, and Glob directly instead. \
             Do NOT delegate to sub-agents."
                .to_string()
        } else {
            "\n\nHint: Task delegation failed. Use direct tools (Bash, Write, Read) instead."
                .to_string()
        }
    }
}

/// Detects Edit tool failures (old_string not found).
pub struct EditMismatchHint;

impl HintInjector for EditMismatchHint {
    fn is_relevant(&self, ctx: &HintContext) -> bool {
        ctx.text.contains("not found in file")
            || ctx.text.contains("old_string")
            || ctx.text.contains("not unique")
            || ctx.text.contains("No match found")
    }

    fn hint(&self, ctx: &HintContext) -> String {
        if ctx.consecutive_errors >= 3 {
            "\n\nHint: Edit keeps failing. Read the file first to see exact content, \
             then retry with the correct old_string. Or use Write to replace the entire file."
                .to_string()
        } else {
            "\n\nHint: Edit failed — old_string doesn't match file content. \
             Read the file first to see the exact text."
                .to_string()
        }
    }
}

/// F24 (2026-04-26): non-retryable command-not-found / Exit 127 errors.
/// Per A4 research: Anthropic's documented retry budget is "2-3 times
/// with corrections then apologise" — RL-trained, not server-enforced.
/// `cargo: command not found` / `Exit code 127` is structurally a
/// PERMANENT failure (the binary isn't installed). Atlas's previous
/// behaviour waited for `consecutive_errors >= 3` (the GenericErrorHint
/// path) before escalating; this injector fires at N=1 with a soft
/// "do not retry" hint and at N>=2 with a CRITICAL stop directive.
/// Front-loads the failure-pressure response that fix30 cc-session-20
/// never reached because the retry-bucket variations evaded F7.
pub struct NotInstalledHint;

impl HintInjector for NotInstalledHint {
    fn is_relevant(&self, ctx: &HintContext) -> bool {
        ctx.text.contains("command not found")
            || ctx.text.contains("Exit code 127")
            || ctx.text.contains(": not found")
            || ctx.text.contains("[tool error]\nExit code 127")
    }

    fn hint(&self, ctx: &HintContext) -> String {
        if ctx.consecutive_errors >= 2 {
            "\n\n<CRITICAL>\n\
             STOP retrying. The command is NOT installed in this \
             environment. Cosmetic variations (different mkdir, cd, \
             flag order) will not change the outcome. Reply to the \
             user about the missing dependency and ask whether to \
             proceed differently. Do NOT call Bash with the same \
             command again.\n\
             </CRITICAL>"
                .to_string()
        } else {
            "\n\nHint: that command is not installed. Do NOT retry \
             with cosmetic variations (mkdir, cd, &&). Either tell \
             the user the tool is missing, or use a different \
             approach that doesn't require it."
                .to_string()
        }
    }
}

/// Catches generic tool errors that no specific injector matched.
/// Always runs last as a fallback.
pub struct GenericErrorHint;

impl HintInjector for GenericErrorHint {
    fn is_relevant(&self, ctx: &HintContext) -> bool {
        ctx.text.starts_with("Error")
            || ctx.text.starts_with("error")
            || ctx.text.contains("Error:")
            || ctx.text.contains("error:")
            || ctx.text.contains("failed")
            || ctx.text.contains("Permission denied")
    }

    fn hint(&self, ctx: &HintContext) -> String {
        if ctx.consecutive_errors >= 3 {
            "\n\n<CRITICAL>\n\
             This tool has failed 3+ times. STOP retrying the same approach.\n\
             Use Bash as a fallback for file operations.\n\
             Do NOT retry the exact same call — change your strategy.\n\
             </CRITICAL>"
                .to_string()
        } else {
            "\n\nHint: Tool call failed. If it keeps failing, \
             try Bash as a fallback. Do NOT retry the exact same call."
                .to_string()
        }
    }
}

// ─── Registry ────────────────────────────────────────────────────────

/// Run all registered hint injectors on a tool response.
/// Returns the first matching hint (specific injectors first, generic last).
pub fn inject_hints(text: &mut String, consecutive_errors: u32) {
    let ctx = HintContext {
        text,
        consecutive_errors,
    };

    // Ordered: specific injectors first, generic fallback last.
    let injectors: &[&dyn HintInjector] = &[
        &WritePathHint,
        &ReadErrorHint,
        &EditMismatchHint,
        &TaskDelegationHint,
        &FileNotFoundHint,
        &NotInstalledHint, // F24: must run BEFORE GenericErrorHint
        &GenericErrorHint,
    ];

    for injector in injectors {
        if injector.is_relevant(&ctx) {
            let hint = injector.hint(&ctx);
            if !hint.is_empty() {
                text.push_str(&hint);
                return; // First match wins — don't stack hints.
            }
        }
    }
}

/// Check if a tool response looks like an error (any injector matches).
pub fn looks_like_error(text: &str) -> bool {
    let ctx = HintContext {
        text,
        consecutive_errors: 0,
    };
    let injectors: &[&dyn HintInjector] = &[
        &WritePathHint,
        &ReadErrorHint,
        &EditMismatchHint,
        &TaskDelegationHint,
        &FileNotFoundHint,
        &NotInstalledHint, // F24: must run BEFORE GenericErrorHint
        &GenericErrorHint,
    ];
    injectors.iter().any(|i| i.is_relevant(&ctx))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eisdir_detection() {
        let mut text = "Error: EISDIR: illegal operation on a directory, read".to_string();
        inject_hints(&mut text, 1);
        assert!(text.contains("file_path is a directory"));
    }

    #[test]
    fn test_eisdir_escalation() {
        let mut text = "Error: EISDIR: illegal operation on a directory, read".to_string();
        inject_hints(&mut text, 3);
        assert!(text.contains("<CRITICAL>"));
        assert!(text.contains("STOP using the Write tool"));
    }

    #[test]
    fn test_enoent_detection() {
        let mut text = "Error: ENOENT: no such file or directory".to_string();
        inject_hints(&mut text, 1);
        assert!(text.contains("mkdir -p"));
    }

    #[test]
    fn test_generic_fallback() {
        let mut text = "Error: something went wrong".to_string();
        inject_hints(&mut text, 1);
        assert!(text.contains("Bash as a fallback"));
    }

    #[test]
    fn test_no_hint_on_success() {
        let mut text = "File written successfully".to_string();
        let original = text.clone();
        inject_hints(&mut text, 0);
        assert_eq!(text, original);
    }

    #[test]
    fn test_specific_wins_over_generic() {
        let mut text = "Error: EISDIR: illegal operation on a directory".to_string();
        inject_hints(&mut text, 1);
        assert!(text.contains("file_path is a directory"));
        assert!(!text.contains("Tool call failed"));
    }

    #[test]
    fn test_read_error() {
        let mut text =
            "Error: ENOENT: no such file or directory, open './test/foo.txt'".to_string();
        inject_hints(&mut text, 1);
        assert!(text.contains("Verify the path"));
    }

    #[test]
    fn test_task_delegation() {
        let mut text = "Error: Unknown agent type:  is not a valid agent type".to_string();
        inject_hints(&mut text, 2);
        assert!(text.contains("STOP using the Task tool"));
    }

    #[test]
    fn test_edit_mismatch() {
        let mut text = "Error: old_string not found in file".to_string();
        inject_hints(&mut text, 1);
        assert!(text.contains("Read the file first"));
    }
}
