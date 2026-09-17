// SPDX-License-Identifier: AGPL-3.0-only

//! Where a FREE-device-memory figure comes from: the CUDA driver, or the
//! amdgpu kernel driver's own VRAM counters in sysfs.
//!
//! ## Why this module exists
//!
//! Every memory guard in Atlas keys off one number: the fast loader's
//! per-shard OOM guard, the pre-flight peak estimate, the KV sizer's headroom,
//! the OOM watchdog, the TUI gauge. On NVIDIA that number is
//! `cuMemGetInfo_v2`'s `free` and it is correct. On an AMD discrete board
//! driven through SCALE (scale-lang.com) it is not.
//!
//! **Measured 2026-09-17 on an AMD Radeon AI PRO R9700 (gfx1201), SCALE
//! 1.7.1, ROCm 7.2.0, Ubuntu 24.04, kernel 7.0.** Loading
//! `unsloth/Qwen3.8-27B-NVFP4` (1953 tensors) through the fast loader, Atlas
//! logged `Shard 1/2 done, GPU memory: 31.56 GB used, 0.05 GB free` while
//! `/sys/class/drm/card1/device/mem_info_vram_used` peaked at 22,926,888,960
//! bytes (22.9 GB) against a 31.86 GB total, with 22.57 GB of tensors on the
//! allocation ledger. The driver leg over-reported usage by roughly 9 GB.
//!
//! A two-loop repro in one program, same box and stack, isolates it as an
//! ALLOCATION-COUNT effect rather than a byte-count one:
//!
//! * 56 allocations of 512 MiB: SCALE's `cudaMemGetInfo` tracks sysfs within
//!   1 percent (free 3704 MiB at 28672 MiB allocated; sysfs used 29379 MiB).
//! * 2000 allocations of 11 MiB: SCALE reports free 60 MiB at only 16500 MiB
//!   allocated, while sysfs shows 17227 MiB used of 32624 MiB, i.e. about 15 GB
//!   genuinely free. It stays pinned at 64 MiB free through 22000 MiB
//!   allocated, and after freeing every allocation it reports only 22064 MiB
//!   free: the figure never recovers.
//! * Native HIP `hipMemGetInfo` in the same small-allocation loop reports
//!   8556 MiB free at 22000 MiB allocated with sysfs used 24375 MiB, i.e. honest.
//!
//! So this is a **SCALE runtime reporting defect**, not a hardware or ROCm
//! one, and not real VRAM pressure: SCALE charges roughly 16 MiB of phantom
//! usage per allocation to its own `MemGetInfo` accounting (a pool
//! granularity), the kernel's TTM counters track the real bytes, and HIP
//! underneath reports them correctly. Real VRAM use is fine; only the
//! reported number is wrong. That is why a 1953-tensor shard drove the
//! reading to 0.05 GB with 9 GB actually free. The clean repro above exists
//! for Spectral (SCALE's vendor) to reproduce it in one program.
//!
//! Note what is NOT wrong: `total` from the driver read 32624 MiB, the
//! board's true capacity. `total_memory_cu` therefore stays on the driver:
//! only the FREE leg moves here.
//!
//! ## What sysfs gives instead
//!
//! `/sys/class/drm/card<N>/device/mem_info_vram_{total,used}` is amdgpu's own
//! TTM accounting, maintained by the kernel across EVERY process on the
//! board. That second property matters independently of the defect above: a
//! desktop compositor holding 0.4 to 1.6 GB of VRAM is invisible to a
//! per-context driver query, and this target is explicitly a discrete board
//! that may also be driving a session.
//!
//! ## What this module does NOT change
//!
//! Nothing on NVIDIA. The default is [`MemInfoSource::Driver`] unless the
//! build is a SCALE build (`cfg!(avarok_scale)`, set by `build.rs` from
//! `kernels/<hw>/HARDWARE.toml` `[hardware].vendor`) AND a matching amdgpu
//! node is actually found, so an NVIDIA binary resolves to `Driver` without
//! so much as a `read_dir`. `AVAROK_MEMINFO_SOURCE` overrides in both
//! directions for a bisect.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result};

/// `driver` | `sysfs` | `sysfs:/sys/class/drm/cardN/device`.
pub(super) const ENV_VAR: &str = "AVAROK_MEMINFO_SOURCE";

/// Where the free-memory figure comes from for this process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum MemInfoSource {
    /// `cuMemGetInfo_v2`: every NVIDIA build, and the fallback everywhere.
    Driver,
    /// The amdgpu `mem_info_vram_*` pair in this device directory.
    Sysfs(PathBuf),
}

/// The resolved source for this process, decided once.
///
/// Takes the driver-reported TOTAL as a hint because auto-detection has to
/// tell the board apart from any other DRM node (an integrated display
/// adapter, a second card) and the total is the one figure the driver gets
/// right. Every caller has just read it from the same `cuMemGetInfo_v2` pair,
/// so the hint costs no extra driver call.
pub(super) fn resolve(total_hint: usize) -> &'static MemInfoSource {
    static RESOLVED: OnceLock<MemInfoSource> = OnceLock::new();
    RESOLVED.get_or_init(|| {
        let env = std::env::var(ENV_VAR).ok();
        let decision = decide(env.as_deref(), cfg!(avarok_scale), || {
            autodetect_amdgpu_sysfs(total_hint)
        });
        if let Some(warning) = &decision.warning {
            tracing::warn!("{warning}");
        }
        match &decision.source {
            MemInfoSource::Driver => tracing::info!(
                "free-memory source: CUDA driver (cuMemGetInfo_v2), {}",
                decision.why
            ),
            MemInfoSource::Sysfs(dir) => tracing::info!(
                "free-memory source: amdgpu sysfs {}, {}",
                dir.display(),
                decision.why
            ),
        }
        decision.source
    })
}

/// A resolution plus the words the one INFO line needs.
pub(super) struct Decision {
    pub(super) source: MemInfoSource,
    /// Why this source, for the log: forced, defaulted, or fallen back.
    pub(super) why: &'static str,
    /// A separate WARN when the environment asked for something that could
    /// not be honoured. `None` on every healthy path.
    pub(super) warning: Option<String>,
}

/// The resolution rule, pure: environment in, source out.
///
/// `autodetect` is a parameter rather than a direct call so the rule is
/// testable without a GPU, without SCALE, and without `/sys`.
pub(super) fn decide(
    env_value: Option<&str>,
    scale_target: bool,
    autodetect: impl FnOnce() -> Option<PathBuf>,
) -> Decision {
    let requested = env_value.map(str::trim).filter(|v| !v.is_empty());
    match requested {
        // Explicit driver: the escape hatch if sysfs ever misleads.
        Some(v) if v.eq_ignore_ascii_case("driver") => Decision {
            source: MemInfoSource::Driver,
            why: "forced by AVAROK_MEMINFO_SOURCE=driver",
            warning: None,
        },
        // Explicit sysfs with the device directory spelled out. Taken
        // verbatim: an operator who names a path has a reason, and a bad one
        // is reported by `sysfs_free_bytes` with the path in the message.
        Some(v) if v.len() > 6 && v[..6].eq_ignore_ascii_case("sysfs:") => Decision {
            source: MemInfoSource::Sysfs(PathBuf::from(v[6..].trim())),
            why: "path given by AVAROK_MEMINFO_SOURCE=sysfs:<dir>",
            warning: None,
        },
        Some(v) if v.eq_ignore_ascii_case("sysfs") => match autodetect() {
            Some(dir) => Decision {
                source: MemInfoSource::Sysfs(dir),
                why: "forced by AVAROK_MEMINFO_SOURCE=sysfs, device auto-detected",
                warning: None,
            },
            None => Decision {
                source: MemInfoSource::Driver,
                why: "AVAROK_MEMINFO_SOURCE=sysfs, but no amdgpu device matched",
                warning: Some(format!(
                    "{ENV_VAR}=sysfs but no /sys/class/drm/card*/device carries a \
                     mem_info_vram_total within 5% of the driver-reported total; \
                     falling back to the driver. Name the directory explicitly with \
                     {ENV_VAR}=sysfs:/sys/class/drm/cardN/device."
                )),
            },
        },
        // Anything else is a typo, and a typo must not silently select a
        // source. Driver is the safe answer: it is what every build did
        // before this module existed.
        Some(v) => Decision {
            source: MemInfoSource::Driver,
            why: "unrecognised AVAROK_MEMINFO_SOURCE value",
            warning: Some(format!(
                "{ENV_VAR}={v:?} is not one of driver | sysfs | sysfs:<dir>; \
                 using the CUDA driver"
            )),
        },
        None if scale_target => match autodetect() {
            Some(dir) => Decision {
                source: MemInfoSource::Sysfs(dir),
                why: "SCALE build, amdgpu device auto-detected",
                warning: None,
            },
            None => Decision {
                source: MemInfoSource::Driver,
                why: "SCALE build, but no amdgpu device matched the driver total",
                warning: None,
            },
        },
        None => Decision {
            source: MemInfoSource::Driver,
            why: "default (not a SCALE build)",
            warning: None,
        },
    }
}

/// The amdgpu device directory whose `mem_info_vram_total` matches the
/// driver-reported total, or `None` if no DRM node does.
///
/// Entries without the file (a display-only node, a non-amdgpu driver) are
/// skipped rather than treated as zero-sized candidates.
fn autodetect_amdgpu_sysfs(total_hint: usize) -> Option<PathBuf> {
    let mut candidates: Vec<(PathBuf, u64)> = Vec::new();
    for entry in std::fs::read_dir("/sys/class/drm").ok()?.flatten() {
        // `card0`, not `card0-DP-1` (a connector, which has no device
        // counters) and not `renderD128` (the same device by another name,
        // which would double-count a candidate).
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(index) = name.strip_prefix("card") else {
            continue;
        };
        if index.is_empty() || !index.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let dir = entry.path().join("device");
        if let Ok(total) = read_u64(&dir.join("mem_info_vram_total")) {
            candidates.push((dir, total));
        }
    }
    pick_device(&candidates, total_hint as u64)
}

/// The candidate whose VRAM total is within 5 percent of `total_hint`,
/// largest first.
///
/// Pure, and the whole matching rule: 5 percent absorbs the difference
/// between the driver's view of the board and the kernel's (carve-outs,
/// reserved pages) while still excluding an integrated adapter's half-gigabyte
/// aperture from a 32 GB board. Largest wins a tie so a two-board host with
/// one big card and one small one cannot pick the small one on a wide hint.
pub(super) fn pick_device(candidates: &[(PathBuf, u64)], total_hint: u64) -> Option<PathBuf> {
    if total_hint == 0 {
        // No hint means no basis to match on, and "any card" is exactly the
        // guess this function exists to avoid.
        return None;
    }
    candidates
        .iter()
        .filter(|(_, total)| total.abs_diff(total_hint).saturating_mul(20) <= total_hint)
        // `max_by_key` keeps the LAST maximum, so ties resolve to the last
        // candidate in directory order rather than arbitrarily.
        .max_by_key(|(_, total)| *total)
        .map(|(dir, _)| dir.clone())
}

/// Free VRAM from an amdgpu device directory: `mem_info_vram_total` minus
/// `mem_info_vram_used`, as the kernel accounts them across all processes.
pub(super) fn sysfs_free_bytes(dir: &Path) -> Result<usize> {
    let total = read_u64(&dir.join("mem_info_vram_total"))?;
    let used = read_u64(&dir.join("mem_info_vram_used"))?;
    // Saturating: the two files are sampled a few microseconds apart and a
    // concurrent allocation can make `used` momentarily exceed `total`. A
    // free figure of 0 is a correct answer to that race; a wrapped `usize` is
    // 16 exabytes of headroom handed to the KV sizer.
    Ok(total.saturating_sub(used) as usize)
}

/// One `/sys` counter, with the path in every error.
fn read_u64(path: &Path) -> Result<u64> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading amdgpu VRAM counter {}", path.display()))?;
    text.trim()
        .parse::<u64>()
        .with_context(|| format!("parsing amdgpu VRAM counter {}", path.display()))
}

/// Both legs of the free-device-memory question, for a caller that has just
/// made a `cuMemGetInfo_v2` call.
pub(super) struct DeviceFree {
    /// The figure callers must use: the sysfs reading where one is available,
    /// the driver's otherwise.
    pub(super) bytes: usize,
    /// The driver's own figure, always: the A68 comparison leg.
    pub(super) driver: usize,
    /// The sysfs reading, when the resolved source produced one.
    pub(super) sysfs: Option<usize>,
}

/// Resolve the source once, then answer with both legs.
///
/// A sysfs read that fails does NOT fail the query: the driver's figure is
/// still a figure, and losing the memory guards entirely is worse than losing
/// their accuracy. It warns once rather than per poll, because the watchdog calls
/// this every two seconds.
pub(super) fn device_free(driver_free: usize, driver_total: usize) -> DeviceFree {
    let sysfs = match resolve(driver_total) {
        MemInfoSource::Driver => None,
        MemInfoSource::Sysfs(dir) => match sysfs_free_bytes(dir) {
            Ok(free) => Some(free),
            Err(e) => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(
                        "amdgpu sysfs free-memory read failed, falling back to the CUDA \
                         driver's figure for the rest of this process: {e:#}"
                    );
                });
                None
            }
        },
    };
    DeviceFree {
        bytes: sysfs.unwrap_or(driver_free),
        driver: driver_free,
        sysfs,
    }
}

#[cfg(test)]
#[path = "meminfo_source_tests.rs"]
mod tests;
