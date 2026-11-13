// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: event timing, device facts and the profiler capture range
//! shared by the L07 microbench examples (`hbm_bandwidth_probe`,
//! `linear_t_sweep`, `w4a16_ada_microtest`, `pinned_probe` and the retrofitted
//! FP8 and FP4 microtests).
//!
//! Method credit: the "wrap only the measured repetitions in a profiler range"
//! rule comes from Cinference's benchmark README (`--profile-measured`); the
//! implementation here is Atlas-native.
//!
//! * [`GpuTimer`]: two CUDA events with timing enabled (the backend's own
//!   `create_event` disables timing, so the examples declare the driver calls
//!   themselves, as `dense_gemm_microtest.rs` does).
//! * [`ProfilerRange`]: `cuProfilerStart`/`cuProfilerStop` around the measured
//!   repetitions only, so load, warm-up and the L2 flush stay out of an nsys
//!   capture made with `--capture-range=cudaProfilerApi`. With no profiler
//!   attached both calls are cheap no-ops.
//! * [`device_info`]: SM count, L2 size, memory clock and bus width from the
//!   driver, for the cold-L2 rule and for the spec-sheet number that every
//!   report prints next to the measured ceiling.
//!
//! An example that includes this file declares
//! `#[path = "common/profiler.rs"] mod profiler;` at its root.
#![allow(dead_code)]

use anyhow::{Result, bail};

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
    fn cuProfilerStart() -> i32;
    fn cuProfilerStop() -> i32;
    fn cuCtxGetDevice(dev: *mut i32) -> i32;
    fn cuDeviceGetAttribute(pi: *mut i32, attrib: i32, dev: i32) -> i32;
}

/// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`.
const ATTR_SM_COUNT: i32 = 16;
/// `CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE` (kHz; deprecated in CUDA 13, so a
/// failure here only removes the spec-sheet figure).
const ATTR_MEM_CLOCK_KHZ: i32 = 36;
/// `CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH` (bits).
const ATTR_BUS_BITS: i32 = 37;
/// `CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE` (bytes).
const ATTR_L2_BYTES: i32 = 38;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR` and `_MINOR`.
const ATTR_CC_MAJOR: i32 = 75;
const ATTR_CC_MINOR: i32 = 76;

/// Two timing-enabled events bracketing a run of launches on one stream.
pub struct GpuTimer {
    start: u64,
    end: u64,
}

impl GpuTimer {
    pub fn new() -> Result<Self> {
        let (mut start, mut end) = (0u64, 0u64);
        // SAFETY: both out-pointers are valid for the call; flags 0 keeps timing.
        let rc = unsafe { cuEventCreate(&mut start, 0) };
        if rc != 0 {
            bail!("cuEventCreate(start) failed: status {rc}");
        }
        let rc = unsafe { cuEventCreate(&mut end, 0) };
        if rc != 0 {
            unsafe { cuEventDestroy_v2(start) };
            bail!("cuEventCreate(end) failed: status {rc}");
        }
        Ok(Self { start, end })
    }

    /// Record the start event on `stream`.
    pub fn start(&self, stream: u64) -> Result<()> {
        let rc = unsafe { cuEventRecord(self.start, stream) };
        if rc != 0 {
            bail!("cuEventRecord(start) failed: status {rc}");
        }
        Ok(())
    }

    /// Record the end event on `stream`, wait for it, and return the elapsed
    /// GPU time in milliseconds.
    pub fn stop(&self, stream: u64) -> Result<f64> {
        let rc = unsafe { cuEventRecord(self.end, stream) };
        if rc != 0 {
            bail!("cuEventRecord(end) failed: status {rc}");
        }
        let rc = unsafe { cuEventSynchronize(self.end) };
        if rc != 0 {
            bail!("cuEventSynchronize failed: status {rc}");
        }
        let mut ms = 0f32;
        let rc = unsafe { cuEventElapsedTime(&mut ms, self.start, self.end) };
        if rc != 0 {
            bail!("cuEventElapsedTime failed: status {rc}");
        }
        Ok(f64::from(ms))
    }
}

impl Drop for GpuTimer {
    fn drop(&mut self) {
        // SAFETY: both handles came from cuEventCreate and are destroyed once.
        unsafe {
            cuEventDestroy_v2(self.start);
            cuEventDestroy_v2(self.end);
        }
    }
}

/// A profiler capture range. Dropping it ends the range, so an early return
/// cannot leave a capture open.
pub struct ProfilerRange {
    active: bool,
}

impl ProfilerRange {
    /// Start a range when `enabled`; otherwise a no-op guard.
    pub fn begin(enabled: bool) -> Result<Self> {
        if enabled {
            let rc = unsafe { cuProfilerStart() };
            if rc != 0 {
                bail!("cuProfilerStart failed: status {rc}");
            }
        }
        Ok(Self { active: enabled })
    }
}

impl Drop for ProfilerRange {
    fn drop(&mut self) {
        if self.active {
            unsafe { cuProfilerStop() };
        }
    }
}

/// What the driver says about the device the current context sits on. A field
/// the driver would not report stays zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceInfo {
    pub sm_count: u32,
    pub l2_bytes: usize,
    pub mem_clock_khz: u32,
    pub bus_bits: u32,
    pub cc_major: u32,
    pub cc_minor: u32,
}

impl DeviceInfo {
    /// The spec-sheet bandwidth in GB/s (DDR: two transfers per memory clock),
    /// or `None` when the driver reported no clock or bus width. It is printed
    /// for context only; every percent-of-peak figure uses the MEASURED ceiling.
    pub fn spec_gbs(&self) -> Option<f64> {
        spec_gbs_from(self.mem_clock_khz, self.bus_bits)
    }
}

/// `2 * clock * bus / 8` in GB/s; `None` for a zero input.
pub fn spec_gbs_from(mem_clock_khz: u32, bus_bits: u32) -> Option<f64> {
    if mem_clock_khz == 0 || bus_bits == 0 {
        return None;
    }
    Some(2.0 * f64::from(mem_clock_khz) * 1e3 * f64::from(bus_bits) / 8.0 / 1e9)
}

fn attribute(attrib: i32, dev: i32) -> Option<i32> {
    let mut v = 0i32;
    // SAFETY: `v` is a valid out-pointer for the call.
    let rc = unsafe { cuDeviceGetAttribute(&mut v, attrib, dev) };
    (rc == 0).then_some(v)
}

/// Query the device of the current context. Call after the backend exists.
pub fn device_info() -> DeviceInfo {
    let mut dev = 0i32;
    // SAFETY: valid out-pointer.
    if unsafe { cuCtxGetDevice(&mut dev) } != 0 {
        return DeviceInfo::default();
    }
    let get = |a: i32| attribute(a, dev).map_or(0, |v| u32::try_from(v).unwrap_or(0));
    DeviceInfo {
        sm_count: get(ATTR_SM_COUNT),
        l2_bytes: get(ATTR_L2_BYTES) as usize,
        mem_clock_khz: get(ATTR_MEM_CLOCK_KHZ),
        bus_bits: get(ATTR_BUS_BITS),
        cc_major: get(ATTR_CC_MAJOR),
        cc_minor: get(ATTR_CC_MINOR),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_gbs_matches_the_4060_ti_sheet() {
        // 9,001 MHz reported memory clock, 128-bit bus: 288 GB/s.
        let g = spec_gbs_from(9_001_000, 128).unwrap();
        assert!((g - 288.03).abs() < 0.1, "{g}");
    }

    #[test]
    fn spec_gbs_needs_both_inputs() {
        assert_eq!(spec_gbs_from(0, 128), None);
        assert_eq!(spec_gbs_from(9_001_000, 0), None);
        assert_eq!(DeviceInfo::default().spec_gbs(), None);
    }

    #[test]
    fn device_info_default_is_all_zero() {
        let d = DeviceInfo::default();
        assert_eq!((d.sm_count, d.l2_bytes, d.cc_major), (0, 0, 0));
    }
}
