// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in cp.async `w4a16_gemv_batch2_cpasync` dispatch.
//!
//! Same grid/block/signature as `w4a16_gemv_batch2`. GB10 cold-DRAM was
//! 5–16% slower than template batch2, so this stays off unless
//! `ATLAS_GEMV_BATCH2_CPASYNC=1`. Missing kernel handle falls back to
//! the template `w4a16_gemv_batch2`.

use anyhow::Result;
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use std::sync::OnceLock;

/// Opt-in env var. Presence of the name is not enough — value must be `"1"`.
pub const ENABLE_ENV: &str = "ATLAS_GEMV_BATCH2_CPASYNC";

const MODULE: &str = "w4a16_gemv";
const FALLBACK: &str = "w4a16_gemv_batch2";
const CPASYNC: &str = "w4a16_gemv_batch2_cpasync";

/// ON only when `ATLAS_GEMV_BATCH2_CPASYNC` is exactly `"1"`.
pub fn batch2_cpasync_from(val: Option<&str>) -> bool {
    val == Some("1")
}

/// Resolve the live K=2 NVFP4 GEMV handle. Default is template batch2.
pub fn resolve_w4a16_gemv_batch2(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static LOGGED: OnceLock<bool> = OnceLock::new();
    let on = batch2_cpasync_from(std::env::var(ENABLE_ENV).ok().as_deref());
    if on {
        let h = super::super::try_kernel(gpu, MODULE, CPASYNC);
        if h.0 != 0 {
            LOGGED.get_or_init(|| {
                tracing::info!("w4a16_gemv_batch2_cpasync ON ({ENABLE_ENV}=1)");
                true
            });
            return Ok(h);
        }
    }
    LOGGED.get_or_init(|| {
        tracing::info!("w4a16_gemv_batch2_cpasync OFF (fallback {FALLBACK})");
        false
    });
    gpu.kernel(MODULE, FALLBACK)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};

    #[test]
    fn cpasync_is_opt_in() {
        assert!(!batch2_cpasync_from(None), "unset → OFF");
        assert!(!batch2_cpasync_from(Some("0")), "`=0` does not enable");
        assert!(!batch2_cpasync_from(Some("")), "empty does not enable");
        assert!(batch2_cpasync_from(Some("1")), "`=1` is the opt-in");
    }

    fn kernel_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels")
    }

    #[test]
    fn gb10_gemv_cu_exports_cpasync_kernel_with_cp_async() {
        let p = kernel_root().join("gb10/common/w4a16_gemv.cu");
        let src = fs::read_to_string(&p).unwrap();
        assert!(
            src.contains("void w4a16_gemv_batch2_cpasync("),
            "{} missing w4a16_gemv_batch2_cpasync",
            p.display()
        );
        assert!(
            src.contains("cp.async.ca.shared.global"),
            "{}: batch2 cpasync path must issue cp.async",
            p.display()
        );
        assert!(
            src.contains("w4a16_gemv_batchm_fma_chunk<MAX_M>"),
            "{}: template must call the shared FMA helper",
            p.display()
        );
        assert!(
            src.contains("w4a16_gemv_batchm_fma_chunk<2>"),
            "{}: cpasync must share the template FMA helper",
            p.display()
        );
        assert!(
            src.contains("w4a16_gemv_batchm_impl<2>"),
            "{}: keep current batch2 as fallback template inst",
            p.display()
        );
        assert!(
            src.contains("src16") && src.contains("pair_live"),
            "{}: 16-byte cp.async must gate on gmem alignment (K-tail)",
            p.display()
        );
    }

    /// NEGATIVE: production init must go through the resolver so the kill
    /// switch actually selects a kernel. A new `gpu.kernel(..., batch2)` site
    /// ships the fallback on the default path.
    #[test]
    fn init_sites_use_the_resolver_not_raw_batch2() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let needle = format!(
            "gpu.kernel({q}w4a16_gemv{q}, {q}w4a16_gemv_batch2{q})",
            q = '"'
        );
        let mut offenders = Vec::new();
        fn visit(d: &Path, needle: &str, out: &mut Vec<String>) {
            let Ok(rd) = fs::read_dir(d) else { return };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    visit(&p, needle, out);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    let src = fs::read_to_string(&p).unwrap();
                    if src.contains(needle) {
                        out.push(p.display().to_string());
                    }
                }
            }
        }
        visit(&root, &needle, &mut offenders);
        assert!(
            offenders.is_empty(),
            "use ops::resolve_w4a16_gemv_batch2, not raw kernel lookup: {offenders:?}"
        );
    }
}
