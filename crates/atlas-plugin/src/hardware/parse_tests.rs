// SPDX-License-Identifier: AGPL-3.0-only

//! Parser tests against text captured from a real GB10 box (spark-256a,
//! 2026-08-17, driver 580.126.09), plus the degraded and unsupported shapes.
//!
//! The fixtures are verbatim tool output, not hand-written approximations:
//! every trap this module exists to survive — `[N/A]` cells, `N/A` cells, the
//! two sections that share key names, a `Cached` line sitting next to
//! `SwapCached` — is a property of the real text and would be edited away by a
//! hand-written sample.

use super::*;

const QUERY_GPU: &str = include_str!("fixtures/gb10_query_gpu.csv");
const QUERY_GPU_NA: &str = include_str!("fixtures/gb10_query_gpu_na.csv");
const COMPUTE_APPS: &str = include_str!("fixtures/gb10_compute_apps.csv");
const COMPUTE_APPS_LEFTOVER: &str = include_str!("fixtures/gb10_compute_apps_leftover.csv");
const PERFORMANCE: &str = include_str!("fixtures/gb10_performance.txt");
const PERFORMANCE_HOT: &str = include_str!("fixtures/gb10_performance_throttling.txt");
const PERFORMANCE_NA: &str = include_str!("fixtures/gb10_performance_unsupported.txt");
const MEMINFO: &str = include_str!("fixtures/gb10_meminfo.txt");

#[test]
fn query_gpu_reads_every_cell_of_a_healthy_gb10() {
    let q = gpu_query(QUERY_GPU);
    assert_eq!(q.name.as_deref(), Some("NVIDIA GB10"));
    assert_eq!(q.driver.as_deref(), Some("580.126.09"));
    assert_eq!(q.sm_clock_mhz, Some(208.0));
    assert_eq!(q.sm_clock_max_mhz, Some(3003.0));
    assert_eq!(q.gpu_temp_c, Some(55.0));
    assert_eq!(q.persistence_mode, Some(true));
}

/// ★ The GB10 trap. `nvidia-smi` answers `[N/A]` for the unified-memory
/// queries, and an idle-guard written against one of those never fires. Every
/// `[N/A]` cell must come back `None` — a parser that returned `0.0` would
/// report a stone-cold GPU at zero clock, which reads as perfectly healthy.
#[test]
fn na_cells_are_unknown_and_never_zero() {
    let q = gpu_query(QUERY_GPU_NA);
    assert_eq!(q.name.as_deref(), Some("NVIDIA GB10"));
    assert_eq!(q.driver.as_deref(), Some("580.126.09"));
    assert_eq!(q.sm_clock_mhz, None);
    assert_eq!(q.sm_clock_max_mhz, None);
    assert_eq!(q.gpu_temp_c, None);
    assert_eq!(q.persistence_mode, None);
}

#[test]
fn no_output_at_all_is_every_field_unknown() {
    assert_eq!(gpu_query(""), GpuQuery::default());
    assert_eq!(gpu_query("\n  \n"), GpuQuery::default());
}

/// A short row must leave the tail unknown rather than shifting the values it
/// does have into the wrong slots.
#[test]
fn a_short_row_does_not_shift_values_left() {
    assert_eq!(
        gpu_query("NVIDIA GB10, 580.126.09"),
        GpuQuery {
            name: Some("NVIDIA GB10".into()),
            driver: Some("580.126.09".into()),
            ..GpuQuery::default()
        }
    );
}

#[test]
fn persistence_mode_only_recognises_the_two_spellings() {
    let off = gpu_query("NVIDIA GB10, 580, 208, 3003, 55, Disabled");
    assert_eq!(off.persistence_mode, Some(false));
    let odd = gpu_query("NVIDIA GB10, 580, 208, 3003, 55, Sometimes");
    assert_eq!(
        odd.persistence_mode, None,
        "an unknown spelling is not 'off'"
    );
}

#[test]
fn compute_apps_reads_pid_name_and_memory() {
    let apps = compute_apps(COMPUTE_APPS);
    assert_eq!(apps.len(), 1);
    assert_eq!(apps[0].pid, 2_392_883);
    assert_eq!(apps[0].name, "./target/release/spark");
    assert_eq!(apps[0].used_mib, Some(109_281));
}

/// The 2026-08-15 shape: a killed `spark serve` left its driver process
/// holding 87 GB, so the box hosted TWO compute apps while looking free to
/// every `--query-gpu` reading.
#[test]
fn compute_apps_sees_the_leftover_allocation() {
    assert_eq!(
        compute_apps(COMPUTE_APPS_LEFTOVER),
        vec![
            GpuComputeApp {
                pid: 2_392_883,
                name: "./target/release/spark".into(),
                used_mib: Some(109_281),
            },
            GpuComputeApp {
                pid: 2_118_440,
                name: "./target/release/spark".into(),
                used_mib: Some(87_014),
            },
        ]
    );
}

#[test]
fn compute_apps_of_an_idle_gpu_is_empty_not_an_error() {
    assert!(compute_apps("").is_empty());
}

/// A row we cannot identify must be DROPPED, not invented: the count feeds a
/// refusal, and a synthetic pid would refuse a run for a parse bug.
#[test]
fn compute_apps_drops_a_row_with_no_usable_pid() {
    let apps = compute_apps("not-a-pid, thing, 10\n77, real, [N/A]\n");
    assert_eq!(apps.len(), 1);
    assert_eq!(apps[0].pid, 77);
    assert_eq!(apps[0].used_mib, None, "[N/A] memory keeps the row");
}

/// ★ The two sections share key names. "SW Thermal Slowdown" is `Not Active`
/// under `Clocks Event Reasons` and `502088297 us` under `Clocks Event Reasons
/// Counters`; a flat key scan reads whichever came last and reports the
/// healthy box as either throttling now or as having never throttled.
#[test]
fn performance_keeps_the_two_sections_apart() {
    let (counters, active) = performance(PERFORMANCE);
    assert_eq!(counters.sw_thermal_us, Some(502_088_297));
    assert_eq!(counters.hw_thermal_us, Some(704_014));
    assert_eq!(counters.hw_power_brake_us, Some(0));
    assert_eq!(counters.sw_power_cap_us, Some(16_130_111_127));
    assert_eq!(active.sw_thermal, Some(false));
    assert_eq!(active.hw_thermal, Some(false));
    assert_eq!(active.hw_power_brake, Some(false));
    assert_eq!(active.thermal(), Some(false));
}

/// dgx2's numbers: 2,914 s of SW thermal slowdown and 228 s of HW thermal
/// slowdown, against dgx1's 502 s and 0.7 s.
#[test]
fn performance_reads_the_degraded_box() {
    let (counters, active) = performance(PERFORMANCE_HOT);
    assert_eq!(
        counters,
        ThrottleCounters {
            sw_power_cap_us: Some(51_740_221_883),
            sw_thermal_us: Some(2_914_366_812),
            hw_thermal_us: Some(228_104_991),
            hw_power_brake_us: Some(0),
            sync_boost_us: Some(0),
        }
    );
    assert_eq!(
        active,
        ThrottleActive {
            sw_power_cap: Some(true),
            sw_thermal: Some(true),
            hw_thermal: Some(true),
            hw_power_brake: Some(false),
        }
    );
    assert_eq!(active.thermal(), Some(true));
}

/// A stack that reports `N/A` for every reason must read as UNKNOWN. `Some(0)`
/// here would say "this box has never once throttled", which is the most
/// healthy-looking answer the type can hold.
#[test]
fn performance_that_reports_na_is_unknown_not_zero() {
    let (counters, active) = performance(PERFORMANCE_NA);
    assert_eq!(counters, ThrottleCounters::default());
    assert_eq!(counters.sw_thermal_us, None);
    assert_eq!(active, ThrottleActive::default());
    assert_eq!(active.thermal(), None);
}

#[test]
fn performance_of_empty_output_is_unknown() {
    assert_eq!(
        performance(""),
        (ThrottleCounters::default(), ThrottleActive::default())
    );
}

/// SW power cap is asserted for 16,130 s of an 11.2-day uptime on the HEALTHY
/// box — it is the steady state of a power-limited part, not a fault, so it
/// must not make `thermal()` true on its own.
#[test]
fn sw_power_cap_alone_is_not_a_thermal_event() {
    let (_, active) = performance(
        "    Clocks Event Reasons\n\
         \x20       SW Power Cap                                   : Active\n\
         \x20       HW Thermal Slowdown                            : Not Active\n\
         \x20       HW Power Brake Slowdown                        : Not Active\n\
         \x20       SW Thermal Slowdown                            : Not Active\n",
    );
    assert_eq!(active.sw_power_cap, Some(true));
    assert_eq!(active.thermal(), Some(false));
}

#[test]
fn meminfo_reads_available_total_and_cache() {
    let m = meminfo(MEMINFO);
    assert_eq!(m.total_kb, Some(127_601_452));
    assert_eq!(m.available_kb, Some(5_272_528));
    // 60.7 GiB of page cache — the reading that once stopped a gate booting.
    assert_eq!(m.cached_kb, Some(63_661_056));
}

/// `SwapCached` sits directly under `Cached` in `/proc/meminfo` and measures
/// something else. A prefix match would overwrite the page-cache figure.
#[test]
fn meminfo_does_not_confuse_cached_with_swapcached() {
    let m = meminfo("Cached:         63661056 kB\nSwapCached:            8 kB\n");
    assert_eq!(m.cached_kb, Some(63_661_056));
}

#[test]
fn meminfo_of_nothing_is_unknown() {
    assert_eq!(meminfo(""), MemInfo::default());
}

#[test]
fn thermal_zone_temp_is_milli_celsius() {
    assert_eq!(milli_celsius("82700\n"), Some(82.7));
    assert_eq!(milli_celsius(""), None);
    assert_eq!(milli_celsius("warm"), None);
}

#[test]
fn non_finite_measurements_are_unknown() {
    assert_eq!(milli_celsius("NaN"), None);
    assert_eq!(milli_celsius("inf"), None);
    assert_eq!(milli_celsius("-inf"), None);
    assert_eq!(
        gpu_query("NVIDIA GB10, 580, NaN, inf, -inf, Enabled"),
        GpuQuery {
            name: Some("NVIDIA GB10".into()),
            driver: Some("580".into()),
            persistence_mode: Some(true),
            ..GpuQuery::default()
        }
    );
    assert_eq!(thermal_zone(Some("gpu"), "NaN"), None);
}

#[test]
fn an_unreadable_zone_is_dropped_rather_than_read_as_zero() {
    assert_eq!(
        thermal_zone(Some("acpitz\n"), "65000\n"),
        Some(ThermalZone {
            name: "acpitz".to_string(),
            temp_c: 65.0
        })
    );
    assert!(thermal_zone(Some("acpitz"), "").is_none());
}
