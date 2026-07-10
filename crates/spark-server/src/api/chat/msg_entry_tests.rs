// SPDX-License-Identifier: AGPL-3.0-only

#[cfg(test)]
mod error_dedup_tests {
    use super::super::duplicate_error_masks;

    #[test]
    fn exact_duplicate_errors_mask_older_keep_newest() {
        // The 45k-collapse shape: identical error text repeated.
        let err = "Error: BadResource: FileSystem.readFile (/home/nologik)";
        let results = vec![
            (2, err.to_string()),
            (5, format!("  {err}\n")), // trim-equal counts as duplicate
            (9, err.to_string()),
        ];
        let masks = duplicate_error_masks(&results);
        assert_eq!(
            masks,
            vec![
                (2, "[same error as below, attempt 1 of 3]".to_string()),
                (5, "[same error as below, attempt 2 of 3]".to_string()),
            ],
            "older duplicates masked, newest (idx 9) untouched"
        );
    }

    #[test]
    fn non_error_duplicates_untouched() {
        let ok = "src\nCargo.toml\nREADME.md";
        let results = vec![
            (1, ok.to_string()),
            (3, ok.to_string()),
            (7, ok.to_string()),
        ];
        assert!(
            duplicate_error_masks(&results).is_empty(),
            "identical SUCCESS observations are legitimate — never masked"
        );
    }

    #[test]
    fn distinct_errors_untouched() {
        let results = vec![
            (1, "Error: ENOENT no such file /a/b".to_string()),
            (
                4,
                "Error: Permission denied writing /etc/passwd".to_string(),
            ),
        ];
        assert!(duplicate_error_masks(&results).is_empty());
    }

    #[test]
    fn near_duplicate_errors_group_via_jaccard() {
        // Long errors differing in one trailing token: Jaccard over
        // 4-gram shingles lands just above 0.9 at ~90 tokens.
        let body: Vec<String> = (0..90).map(|i| format!("frame{i}")).collect();
        let base = format!("Error: {}", body.join(" "));
        let mut variant_body = body.clone();
        *variant_body.last_mut().unwrap() = "different".to_string();
        let variant = format!("Error: {}", variant_body.join(" "));
        let results = vec![(0, base), (2, variant)];
        let masks = duplicate_error_masks(&results);
        assert_eq!(
            masks,
            vec![(0, "[same error as below, attempt 1 of 2]".to_string())]
        );
    }

    #[test]
    fn single_error_never_masked() {
        let results = vec![(0, "Error: something failed".to_string())];
        assert!(duplicate_error_masks(&results).is_empty());
    }
}

#[cfg(test)]
mod vacuous_system_tests {
    use super::super::is_vacuous_system_content;

    #[test]
    fn empty_or_whitespace_is_vacuous() {
        assert!(is_vacuous_system_content(""));
        assert!(is_vacuous_system_content("   \n\t  "));
    }

    #[test]
    fn open_webui_empty_context_residue_is_vacuous() {
        // The exact 2026-05-17 field artifact.
        assert!(is_vacuous_system_content("User Context:\n\n"));
        assert!(is_vacuous_system_content("Context:"));
        assert!(is_vacuous_system_content("  System:  "));
    }

    #[test]
    fn substantive_prompt_is_not_vacuous() {
        assert!(!is_vacuous_system_content(
            "User Context:\nThe user is a senior Rust engineer."
        ));
        assert!(!is_vacuous_system_content("You are a helpful assistant."));
        assert!(!is_vacuous_system_content("Always answer in French."));
        // Label-like but with a real payload after the colon.
        assert!(!is_vacuous_system_content("Role: expert chess coach"));
        // Long single line ending in ':' is unusual prose, not a bare
        // header — keep it (avoid false-strip).
        assert!(!is_vacuous_system_content(
            "Summarize the following transcript and then ask the user this:"
        ));
    }
}
