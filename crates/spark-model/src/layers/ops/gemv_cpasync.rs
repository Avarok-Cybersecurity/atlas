// SPDX-License-Identifier: AGPL-3.0-only

//! Default-ON cp.async `w4a16_gemv_cpasync` dispatch for M=1 NVFP4 GEMV.
//!
//! Same grid/block/signature as `w4a16_gemv`. Kill with
//! `ATLAS_NO_GEMV_CPASYNC=1` (`=0` does **not** disable). Missing kernel
//! handle falls back to the 64-thread `w4a16_gemv`. Does not touch SW.

use anyhow::Result;
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use std::sync::OnceLock;

/// Kill-switch env var. Presence of the name is not enough — value must be `"1"`.
pub const DISABLE_ENV: &str = "ATLAS_NO_GEMV_CPASYNC";

const MODULE: &str = "w4a16_gemv";
const FALLBACK: &str = "w4a16_gemv";
const CPASYNC: &str = "w4a16_gemv_cpasync";

/// ON unless `ATLAS_NO_GEMV_CPASYNC` is exactly `"1"`.
pub fn gemv_cpasync_from(val: Option<&str>) -> bool {
    val != Some("1")
}

/// Resolve the live M=1 NVFP4 GEMV handle. Default is the cp.async kernel.
pub fn resolve_w4a16_gemv(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static LOGGED: OnceLock<bool> = OnceLock::new();
    let on = gemv_cpasync_from(std::env::var(DISABLE_ENV).ok().as_deref());
    if on {
        let h = super::super::try_kernel(gpu, MODULE, CPASYNC);
        if h.0 != 0 {
            LOGGED.get_or_init(|| {
                tracing::info!("w4a16_gemv_cpasync ON (kill {DISABLE_ENV}=1)");
                true
            });
            return Ok(h);
        }
    }
    LOGGED.get_or_init(|| {
        tracing::info!("w4a16_gemv_cpasync OFF (fallback {FALLBACK})");
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
    fn cpasync_ships_on_and_only_the_one_value_kills() {
        assert!(gemv_cpasync_from(None), "unset → ON");
        assert!(gemv_cpasync_from(Some("0")), "`=0` is NOT off");
        assert!(gemv_cpasync_from(Some("")), "empty is NOT off");
        assert!(!gemv_cpasync_from(Some("1")), "`=1` is the kill");
    }

    fn kernel_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels")
    }

    #[test]
    fn gb10_gemv_cu_exports_cpasync_and_shares_mac_helper() {
        let p = kernel_root().join("gb10/common/w4a16_gemv.cu");
        let src = fs::read_to_string(&p).unwrap();
        assert!(
            src.contains("void w4a16_gemv_cpasync("),
            "{} missing w4a16_gemv_cpasync",
            p.display()
        );
        assert!(
            src.contains("cp.async.ca.shared.global"),
            "{}: M=1 cpasync path must issue cp.async",
            p.display()
        );
        assert!(
            src.contains("w4a16_gemv_mac_chunk("),
            "{}: base partial and cpasync must share the MAC helper",
            p.display()
        );
        assert!(
            src.contains("w4a16_gemv_partial("),
            "{}: keep current w4a16_gemv on the shared partial",
            p.display()
        );
        assert!(
            src.contains("for (unsigned int super = 0; super < K16; super += 128u)"),
            "{}: cpasync K-loop must be uniform-trip (syncwarp deadlock)",
            p.display()
        );
    }

    #[test]
    fn init_sites_use_the_resolver_not_raw_m1() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let needle = format!("gpu.kernel({q}w4a16_gemv{q}, {q}w4a16_gemv{q})", q = '"');
        let mut offenders = Vec::new();
        fn visit(d: &Path, needle: &str, out: &mut Vec<String>) {
            let Ok(rd) = fs::read_dir(d) else { return };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    visit(&p, needle, out);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    // Unit/mock tests may look up the fallback kernel by name.
                    if name.contains("tests") {
                        continue;
                    }
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
            "use ops::resolve_w4a16_gemv, not raw kernel lookup: {offenders:?}"
        );
    }
}
