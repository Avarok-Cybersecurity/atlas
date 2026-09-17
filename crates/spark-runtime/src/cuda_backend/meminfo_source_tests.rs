// SPDX-License-Identifier: AGPL-3.0-only

//! The free-memory source decision, on a host with no GPU and no `/sys`.
//!
//! Its own file rather than a `#[cfg(test)]` module inside
//! `meminfo_source.rs`, following `arch_preflight_tests.rs`: the module states
//! the rule, this file drives it. Every case here is pure except the three
//! `sysfs_free_bytes` ones, which build a directory of fake `mem_info_*` files
//! under the platform temp dir, never a real sysfs node, so they run on
//! macOS and in CI containers.

use std::path::{Path, PathBuf};

use super::{MemInfoSource, decide, pick_device, sysfs_free_bytes};

const MIB: u64 = 1024 * 1024;

/// The R9700's real `mem_info_vram_total`: 32624 MiB, as read on the board.
const R9700_TOTAL: u64 = 32624 * MIB;

fn dir(path: &str) -> PathBuf {
    PathBuf::from(path)
}

// ── pick_device ─────────────────────────────────────────────────────

#[test]
fn pick_device_takes_an_exact_total_match() {
    let candidates = [(dir("/sys/class/drm/card1/device"), R9700_TOTAL)];
    assert_eq!(
        pick_device(&candidates, R9700_TOTAL),
        Some(dir("/sys/class/drm/card1/device"))
    );
}

#[test]
fn pick_device_accepts_a_total_within_five_percent() {
    // The driver's total and the kernel's differ by carve-outs and reserved
    // pages; on the R9700 the two read 31.86 GB and 32624 MiB. 4 percent low
    // must still match.
    let kernel_total = R9700_TOTAL - (R9700_TOTAL * 4 / 100);
    let candidates = [(dir("/sys/class/drm/card1/device"), kernel_total)];
    assert_eq!(
        pick_device(&candidates, R9700_TOTAL),
        Some(dir("/sys/class/drm/card1/device")),
        "a 4% spread between the driver's total and the kernel's must still match"
    );
}

#[test]
fn pick_device_ignores_an_integrated_adapter() {
    // The bring-up box: a 512 MiB integrated display adapter on card0 and the
    // 32 GB R9700 on card1. Matching "any DRM node with VRAM counters" would
    // have sized the KV pool against half a gigabyte.
    let candidates = [
        (dir("/sys/class/drm/card0/device"), 512 * MIB),
        (dir("/sys/class/drm/card1/device"), R9700_TOTAL),
    ];
    assert_eq!(
        pick_device(&candidates, R9700_TOTAL),
        Some(dir("/sys/class/drm/card1/device"))
    );
}

#[test]
fn pick_device_matches_nothing_when_no_total_is_close() {
    // A 16 GB board and a 512 MiB adapter, hinted with 32 GB: neither is the
    // board the driver is talking about, so the answer is "no idea", not
    // "the closest one".
    let candidates = [
        (dir("/sys/class/drm/card0/device"), 512 * MIB),
        (dir("/sys/class/drm/card1/device"), 16 * 1024 * MIB),
    ];
    assert_eq!(pick_device(&candidates, R9700_TOTAL), None);
}

#[test]
fn pick_device_without_a_hint_matches_nothing() {
    // `cuMemGetInfo` returning total 0 is a broken driver, and "any card"
    // is exactly the guess this function exists to avoid.
    let candidates = [(dir("/sys/class/drm/card1/device"), R9700_TOTAL)];
    assert_eq!(pick_device(&candidates, 0), None);
}

#[test]
fn pick_device_prefers_the_largest_of_two_matches() {
    // Two boards inside the 5 percent window: the larger is the one whose
    // total the driver reported.
    let candidates = [
        (dir("/sys/class/drm/card0/device"), R9700_TOTAL - 300 * MIB),
        (dir("/sys/class/drm/card1/device"), R9700_TOTAL),
    ];
    assert_eq!(
        pick_device(&candidates, R9700_TOTAL),
        Some(dir("/sys/class/drm/card1/device"))
    );
}

// ── the environment rule ────────────────────────────────────────────

/// An auto-detection that succeeds, and records that it was consulted.
fn found() -> Option<PathBuf> {
    Some(dir("/sys/class/drm/card1/device"))
}

/// An auto-detection that finds no matching board.
fn missing() -> Option<PathBuf> {
    None
}

/// An auto-detection that must never run. Used to prove an NVIDIA build does
/// not go near `/sys`.
fn never() -> Option<PathBuf> {
    panic!("auto-detection must not be attempted on this path");
}

#[test]
fn env_driver_forces_the_driver_even_on_a_scale_build() {
    let d = decide(Some("driver"), true, never);
    assert_eq!(d.source, MemInfoSource::Driver);
    assert!(d.warning.is_none());
    assert!(d.why.contains("forced"), "why = {}", d.why);
}

#[test]
fn env_sysfs_autodetects() {
    let d = decide(Some("sysfs"), false, found);
    assert_eq!(
        d.source,
        MemInfoSource::Sysfs(dir("/sys/class/drm/card1/device")),
        "an explicit request must work on a non-SCALE build too"
    );
    assert!(d.warning.is_none());
}

#[test]
fn env_sysfs_with_no_match_warns_and_falls_back() {
    let d = decide(Some("sysfs"), true, missing);
    assert_eq!(d.source, MemInfoSource::Driver);
    let warning = d.warning.expect("an unhonoured request must warn");
    assert!(
        warning.contains("ATLAS_MEMINFO_SOURCE=sysfs:"),
        "the warning must name the explicit-path form: {warning}"
    );
}

#[test]
fn env_sysfs_with_a_path_takes_it_verbatim() {
    let d = decide(Some("sysfs:/sys/class/drm/card3/device"), false, never);
    assert_eq!(
        d.source,
        MemInfoSource::Sysfs(dir("/sys/class/drm/card3/device")),
        "a named path must not be re-detected"
    );
    assert!(d.warning.is_none());
}

#[test]
fn env_junk_falls_back_to_the_driver_with_a_warning() {
    let d = decide(Some("sysfs!"), true, never);
    assert_eq!(
        d.source,
        MemInfoSource::Driver,
        "a typo must not silently select a source"
    );
    let warning = d.warning.expect("a typo must warn");
    assert!(
        warning.contains("sysfs!"),
        "the warning must quote it: {warning}"
    );
}

#[test]
fn unset_on_a_scale_build_auto_detects() {
    let d = decide(None, true, found);
    assert_eq!(
        d.source,
        MemInfoSource::Sysfs(dir("/sys/class/drm/card1/device")),
        "the SCALE default is the whole point of this module"
    );
    assert!(d.warning.is_none());
}

#[test]
fn unset_on_a_scale_build_with_no_match_uses_the_driver_quietly() {
    // Not a warning: a SCALE build on a board with no amdgpu sysfs node (a
    // container without /sys, a future non-amdgpu SCALE target) is simply the
    // behaviour every build had before this module existed.
    let d = decide(None, true, missing);
    assert_eq!(d.source, MemInfoSource::Driver);
    assert!(d.warning.is_none());
}

#[test]
fn unset_without_scale_never_touches_sysfs() {
    // The NVIDIA path, byte-identical to what it always did: `never` panics
    // if auto-detection is attempted at all.
    let d = decide(None, false, never);
    assert_eq!(d.source, MemInfoSource::Driver);
    assert!(d.warning.is_none());
}

#[test]
fn an_empty_value_reads_as_unset() {
    // `ATLAS_MEMINFO_SOURCE=` in a serve script is an operator clearing the
    // knob, not asking for a source named "".
    let d = decide(Some(""), false, never);
    assert_eq!(d.source, MemInfoSource::Driver);
    assert!(d.warning.is_none(), "clearing the knob is not a mistake");
}

// ── reading the counters ────────────────────────────────────────────

/// A directory of fake `mem_info_*` files, removed when the test ends.
struct FakeSysfs(PathBuf);

impl FakeSysfs {
    fn new(tag: &str, files: &[(&str, &str)]) -> Self {
        let path =
            std::env::temp_dir().join(format!("atlas-meminfo-{}-{}", std::process::id(), tag));
        std::fs::create_dir_all(&path).expect("create the fake sysfs dir");
        for (name, contents) in files {
            std::fs::write(path.join(name), contents).expect("write a fake counter");
        }
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for FakeSysfs {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn sysfs_free_is_total_minus_used() {
    // The numbers the R9700 actually reported at the peak of the failing
    // load: 22,926,888,960 bytes used of 34,208,645,120.
    let fake = FakeSysfs::new(
        "free",
        &[
            ("mem_info_vram_total", "34208645120\n"),
            ("mem_info_vram_used", "22926888960\n"),
        ],
    );
    assert_eq!(
        sysfs_free_bytes(fake.path()).expect("both counters present"),
        34208645120usize - 22926888960usize
    );
}

#[test]
fn sysfs_free_saturates_when_used_exceeds_total() {
    // The two files are sampled microseconds apart; a concurrent allocation
    // can invert them. Zero free is a correct answer to that race, and a
    // wrapped usize is 16 exabytes of headroom handed to the KV sizer.
    let fake = FakeSysfs::new(
        "saturate",
        &[
            ("mem_info_vram_total", "1000\n"),
            ("mem_info_vram_used", "1001\n"),
        ],
    );
    assert_eq!(sysfs_free_bytes(fake.path()).expect("both present"), 0);
}

#[test]
fn sysfs_free_errors_name_the_path() {
    let fake = FakeSysfs::new("missing-used", &[("mem_info_vram_total", "1000\n")]);
    let err = sysfs_free_bytes(fake.path()).expect_err("mem_info_vram_used is absent");
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains("mem_info_vram_used"),
        "the error must name the file it could not read: {rendered}"
    );
}

#[test]
fn sysfs_free_errors_on_an_unparsable_counter() {
    let fake = FakeSysfs::new(
        "unparsable",
        &[
            ("mem_info_vram_total", "not a number\n"),
            ("mem_info_vram_used", "1\n"),
        ],
    );
    let err = sysfs_free_bytes(fake.path()).expect_err("the total is not a number");
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains("mem_info_vram_total"),
        "the error must name the file it could not parse: {rendered}"
    );
}
