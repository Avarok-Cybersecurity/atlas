// SPDX-License-Identifier: AGPL-3.0-only

//! Presence + polarity pins for the dual-issue batch2 experiment.
//!
//! The kernel is compiled and microtested. Production dispatch stays on
//! template `w4a16_gemv_batch2` until the bandwidth oracle passes.

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    #[test]
    fn gb10_exports_dualissue_kernel() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../kernels/gb10/common/w4a16_gemv_batch2_dualissue.cu");
        let src = fs::read_to_string(&p).unwrap();
        assert!(
            src.contains("void w4a16_gemv_batch2_dualissue("),
            "{} missing export",
            p.display()
        );
        assert!(
            !src.contains("cp.async.ca.shared.global"),
            "{} must not issue the failed smem mailbox",
            p.display()
        );
        assert!(
            src.contains("load phase-0 AND phase-1"),
            "{} must document the dual-issue hoist",
            p.display()
        );
    }

    #[test]
    fn production_init_still_uses_template_batch2() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let needle = "w4a16_gemv_batch2_dualissue";
        let mut hits = Vec::new();
        fn visit(d: &Path, needle: &str, out: &mut Vec<String>) {
            let Ok(rd) = fs::read_dir(d) else { return };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    visit(&p, needle, out);
                } else if p.extension().is_some_and(|x| x == "rs")
                    && p.file_name()
                        .is_some_and(|n| n != "gemv_batch2_dualissue.rs")
                {
                    let src = fs::read_to_string(&p).unwrap();
                    if src.contains(needle) {
                        out.push(p.display().to_string());
                    }
                }
            }
        }
        visit(&root, needle, &mut hits);
        assert!(
            hits.is_empty(),
            "do not wire dualissue into production until the oracle passes: {hits:?}"
        );
    }
}
