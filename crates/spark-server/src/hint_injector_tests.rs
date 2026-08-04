// SPDX-License-Identifier: AGPL-3.0-only

#[cfg(test)]
mod tests {
    use super::super::*;

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

    #[test]
    fn bw1_classify_productive_vs_explore() {
        use serde_json::json;
        // write/edit tools → productive
        assert!(tool_call_is_productive(
            "write",
            &json!({"filePath":"src/main.rs","content":"fn main(){}"})
        ));
        assert!(tool_call_is_productive("Edit", &json!({})));
        // bash that writes/builds/runs → productive
        assert!(tool_call_is_productive(
            "bash",
            &json!({"command":"cargo run --release"})
        ));
        assert!(tool_call_is_productive(
            "bash",
            &json!({"command":"cat > src/main.rs << 'EOF'"})
        ));
        // bash exploration → NOT productive
        assert!(!tool_call_is_productive(
            "bash",
            &json!({"command":"ls -la /tmp"})
        ));
        assert!(!tool_call_is_productive(
            "bash",
            &json!({"command":"cat Cargo.toml"})
        ));
        // read/glob → NOT productive
        assert!(!tool_call_is_productive("read", &json!({"filePath":"x"})));
        assert!(!tool_call_is_productive("glob", &json!({"pattern":"**/*"})));
    }

    #[test]
    fn bw1_hint_threshold_and_escalation() {
        // Below threshold or any productive call → no hint.
        assert!(bash_wander_hint_inner(4, 0).is_none());
        assert!(bash_wander_hint_inner(20, 1).is_none());
        // At/over threshold with zero productive → standard nudge.
        let h = bash_wander_hint_inner(5, 0).expect("should fire");
        assert!(h.contains("PROGRESS WATCHDOG"));
        assert!(h.contains("write"));
        // High count → critical escalation.
        let c = bash_wander_hint_inner(10, 0).expect("should fire");
        assert!(c.contains("CRITICAL"));
    }
}
