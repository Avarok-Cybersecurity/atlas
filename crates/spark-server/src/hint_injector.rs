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

/// P0-4 (2026-07-09): structured client errors like opencode's
/// `BadResource: FileSystem.readFile (...)` matched NO hint pattern, so
/// `consecutive_tool_errors` reset to 0 on every retry and the escalating
/// `<CRITICAL>` rescue never fired across 6 identical failures. Recognize a
/// PascalCase/ALLCAPS error-code prefix followed by `:` (BadResource:,
/// NotFound:, EISDIR:), excluding single-hump prose words (Note:, Warning:
/// has its own patterns).
fn pascal_error_prefix(text: &str) -> bool {
    let t = text.trim_start();
    let Some(first) = t.split(':').next() else {
        return false;
    };
    // split(':') guarantees the byte after `first` is `:` when first < t.
    if first.len() < 4
        || first.len() >= t.len()
        || !first.chars().all(|c| c.is_ascii_alphanumeric())
    {
        return false;
    }
    let caps = first.chars().filter(|c| c.is_ascii_uppercase()).count();
    let has_lower = first.chars().any(|c| c.is_ascii_lowercase());
    (has_lower && caps >= 2) || (!has_lower && caps == first.chars().count())
}

impl HintInjector for GenericErrorHint {
    fn is_relevant(&self, ctx: &HintContext) -> bool {
        ctx.text.starts_with("Error")
            || ctx.text.starts_with("error")
            || ctx.text.contains("Error:")
            || ctx.text.contains("error:")
            || ctx.text.contains("failed")
            || ctx.text.contains("Permission denied")
            || ctx.text.starts_with("BadResource")
            || pascal_error_prefix(ctx.text)
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

// ─── BW1: bash-wandering / content-completeness watchdog ─────────────
//
// FP8 agentic failure mode (gap #9): the model explores (bash ls/cat/find,
// read, glob) or narrates across many turns but never writes the deliverable
// file(s) — the run finishes with a valid Cargo.toml but no real src/main.rs,
// so webserver_ok never fires. This steering nudge fires when the agent has
// made many tool calls with NO productive file output yet, redirecting it to
// write + verify. Env-gated (PCND): ATLAS_BASH_WANDER_WATCHDOG=1.

/// One-time read of the watchdog flag.
fn bash_wander_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("ATLAS_BASH_WANDER_WATCHDOG").as_deref() == Ok("1"))
}

/// Classify a tool call as PRODUCTIVE (produces/verifies a deliverable) vs
/// exploratory. `write`/`edit`/`create` tools are productive; a `bash` call is
/// productive only when its command writes a file or builds/runs
/// (`cat >`/`tee`/`>`/`cargo build`/`cargo run`/`rustc`/`go build`/`npm run`).
/// Everything else (ls/cat/find/grep/read/glob/pwd/echo) is exploration.
pub fn tool_call_is_productive(name: &str, args: &serde_json::Value) -> bool {
    let n = name.to_ascii_lowercase();
    if n.contains("write") || n.contains("edit") || n == "create" || n == "patch" {
        return true;
    }
    if n.contains("bash") || n == "shell" || n == "run" || n.contains("exec") {
        let cmd = args
            .get("command")
            .or_else(|| args.get("cmd"))
            .or_else(|| args.get("script"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        const WRITE_VERBS: &[&str] = &[
            "cat >",
            "cat>",
            "tee ",
            ">>",
            " > ",
            "cargo build",
            "cargo run",
            "cargo test",
            "rustc ",
            "go build",
            "go run",
            "npm run",
            "npm install",
            "make ",
            "python ",
            "node ",
            "touch ",
        ];
        return WRITE_VERBS.iter().any(|v| cmd.contains(v));
    }
    false
}

/// BW1 steering hint. Returns `Some(hint)` to append to the most recent tool
/// response when the agent appears to be wandering: ≥ `MIN_CALLS` tool calls
/// with zero productive (file-writing/building) calls so far. Escalates with
/// call count. `None` when disabled, below threshold, or progress was made.
pub fn bash_wander_hint(total_tool_calls: usize, productive_calls: usize) -> Option<String> {
    if !bash_wander_enabled() {
        return None;
    }
    bash_wander_hint_inner(total_tool_calls, productive_calls)
}

/// Pure threshold/escalation logic for [`bash_wander_hint`], split out so it
/// is testable without the env gate.
fn bash_wander_hint_inner(total_tool_calls: usize, productive_calls: usize) -> Option<String> {
    const MIN_CALLS: usize = 5;
    if productive_calls > 0 || total_tool_calls < MIN_CALLS {
        return None;
    }
    let n = total_tool_calls;
    let body = if n >= 9 {
        format!(
            "<CRITICAL PROGRESS WATCHDOG>\n\
             You have run {n} tool calls and have NOT written or edited a single file. \
             Exploration will not complete the task. In your next message, call the write \
             tool to create the required source file(s), then build and run to verify. \
             Do not run any more read-only commands.\n</CRITICAL PROGRESS WATCHDOG>"
        )
    } else {
        format!(
            "<PROGRESS WATCHDOG>\n\
             You have run {n} tool calls without writing or editing any file yet. If the \
             task asks you to create or modify files, do that now with the write/edit tool \
             instead of more exploration, then verify by building/running.\n</PROGRESS WATCHDOG>"
        )
    };
    Some(format!("\n\n{body}"))
}

#[cfg(test)]
#[path = "hint_injector_tests.rs"]
mod hint_injector_tests;
