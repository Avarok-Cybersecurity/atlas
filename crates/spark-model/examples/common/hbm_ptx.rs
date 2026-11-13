// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the three streaming probe kernels of `hbm_bandwidth_probe`,
//! written in PTX, with the driver calls that load and launch them.
//!
//! Method credit: the read, write and copy probe shape follows Cinference's
//! `tools/hbm_bandwidth_probe.cu`; the PTX below is written for Atlas.
//!
//! Why PTX in a string and not a `.cu` file: the probe is a measurement tool,
//! not a serving kernel, and a new `.cu` under `kernels/` would enter the
//! kernel build, the closure hash and the cross-hardware checks. The driver
//! JIT-compiles `.target sm_80` PTX for any newer part, so one text serves every
//! NVIDIA target.
//!
//! Each thread handles `UNROLL` (4) independent 16-byte vectors per trip, so a
//! launch of `sm_count * 8` blocks of 256 threads keeps over a megabyte in
//! flight, far more than the card's latency-bandwidth product (about 170 KB at
//! 288 GB/s and 600 ns). `n_vec` must be a multiple of four trips' vectors, and
//! the example rounds its buffers to a whole number of trips. `reps` repeats the
//! whole scan inside one launch (the hot-L2 contrast uses it). The read kernel
//! xors everything it loads and stores only on a value the data never has, so
//! the loads cannot be removed.
#![allow(dead_code)]

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;
use std::ffi::{CString, c_char, c_void};

unsafe extern "C" {
    fn cuModuleLoadData(module: *mut u64, image: *const c_void) -> i32;
    fn cuModuleGetFunction(func: *mut u64, module: u64, name: *const c_char) -> i32;
    fn cuModuleUnload(module: u64) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cuLaunchKernel(
        f: u64,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
        shmem: u32,
        stream: u64,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
}

/// Launch geometry. A buffer must be a whole number of trips so the unrolled
/// kernels need no tail code.
#[derive(Clone, Copy, Debug)]
pub struct Geom {
    pub grid: u32,
    pub block: u32,
}

/// Vectors (16 bytes) each thread moves per trip of the unrolled loop.
pub const UNROLL: usize = 4;

impl Geom {
    pub fn threads(&self) -> usize {
        self.grid as usize * self.block as usize
    }

    pub fn trip_bytes(&self) -> usize {
        self.threads() * UNROLL * 16
    }
}

pub const PTX_HEAD: &str = ".version 7.0\n.target sm_80\n.address_size 64\n\n";

pub const PTX_READ: &str = r"
.visible .entry hbm_read(.param .u64 p_src, .param .u64 p_nvec, .param .u32 p_reps, .param .u64 p_sink)
{
    .reg .pred %p<4>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<20>;
    ld.param.u64 %rd1, [p_src];
    ld.param.u64 %rd2, [p_nvec];
    ld.param.u32 %r20, [p_reps];
    ld.param.u64 %rd3, [p_sink];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %nctaid.x;
    mad.lo.u32 %r5, %r2, %r3, %r1;
    mul.lo.u32 %r6, %r3, %r4;
    cvt.u64.u32 %rd4, %r5;
    cvt.u64.u32 %rd5, %r6;
    shl.b64 %rd6, %rd5, 4;
    shl.b64 %rd7, %rd5, 2;
    shl.b64 %rd8, %rd5, 6;
    shl.b64 %rd10, %rd4, 4;
    add.u64 %rd11, %rd1, %rd10;
    mov.u32 %r7, 0;
$L_rep:
    setp.eq.u32 %p1, %r20, 0;
    @%p1 bra $L_done;
    mov.u64 %rd12, %rd4;
    mov.u64 %rd13, %rd11;
$L_loop:
    setp.ge.u64 %p2, %rd12, %rd2;
    @%p2 bra $L_next;
    ld.global.nc.v4.u32 {%r8, %r9, %r10, %r11}, [%rd13];
    add.u64 %rd14, %rd13, %rd6;
    ld.global.nc.v4.u32 {%r12, %r13, %r14, %r15}, [%rd14];
    add.u64 %rd15, %rd14, %rd6;
    ld.global.nc.v4.u32 {%r16, %r17, %r18, %r19}, [%rd15];
    add.u64 %rd16, %rd15, %rd6;
    ld.global.nc.v4.u32 {%r21, %r22, %r23, %r24}, [%rd16];
    xor.b32 %r7, %r7, %r8;
    xor.b32 %r7, %r7, %r9;
    xor.b32 %r7, %r7, %r10;
    xor.b32 %r7, %r7, %r11;
    xor.b32 %r7, %r7, %r12;
    xor.b32 %r7, %r7, %r13;
    xor.b32 %r7, %r7, %r14;
    xor.b32 %r7, %r7, %r15;
    xor.b32 %r7, %r7, %r16;
    xor.b32 %r7, %r7, %r17;
    xor.b32 %r7, %r7, %r18;
    xor.b32 %r7, %r7, %r19;
    xor.b32 %r7, %r7, %r21;
    xor.b32 %r7, %r7, %r22;
    xor.b32 %r7, %r7, %r23;
    xor.b32 %r7, %r7, %r24;
    add.u64 %rd12, %rd12, %rd7;
    add.u64 %rd13, %rd13, %rd8;
    bra $L_loop;
$L_next:
    sub.u32 %r20, %r20, 1;
    bra $L_rep;
$L_done:
    setp.eq.u32 %p3, %r7, 0x7f4a7c15;
    @%p3 st.global.u32 [%rd3], %r7;
    ret;
}
";

pub const PTX_WRITE: &str = r"
.visible .entry hbm_write(.param .u64 p_dst, .param .u64 p_nvec, .param .u32 p_reps)
{
    .reg .pred %p<4>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<20>;
    ld.param.u64 %rd1, [p_dst];
    ld.param.u64 %rd2, [p_nvec];
    ld.param.u32 %r20, [p_reps];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %nctaid.x;
    mad.lo.u32 %r5, %r2, %r3, %r1;
    mul.lo.u32 %r6, %r3, %r4;
    cvt.u64.u32 %rd4, %r5;
    cvt.u64.u32 %rd5, %r6;
    shl.b64 %rd6, %rd5, 4;
    shl.b64 %rd7, %rd5, 2;
    shl.b64 %rd8, %rd5, 6;
    shl.b64 %rd10, %rd4, 4;
    add.u64 %rd11, %rd1, %rd10;
    mov.u32 %r8, 0x01234567;
    mov.u32 %r9, 0x09abcdef;
    mov.u32 %r10, 0x0f1e2d3c;
    mov.u32 %r11, 0x4b5a6978;
$L_rep:
    setp.eq.u32 %p1, %r20, 0;
    @%p1 bra $L_done;
    mov.u64 %rd12, %rd4;
    mov.u64 %rd13, %rd11;
$L_loop:
    setp.ge.u64 %p2, %rd12, %rd2;
    @%p2 bra $L_next;
    st.global.cs.v4.u32 [%rd13], {%r8, %r9, %r10, %r11};
    add.u64 %rd14, %rd13, %rd6;
    st.global.cs.v4.u32 [%rd14], {%r8, %r9, %r10, %r11};
    add.u64 %rd15, %rd14, %rd6;
    st.global.cs.v4.u32 [%rd15], {%r8, %r9, %r10, %r11};
    add.u64 %rd16, %rd15, %rd6;
    st.global.cs.v4.u32 [%rd16], {%r8, %r9, %r10, %r11};
    add.u64 %rd12, %rd12, %rd7;
    add.u64 %rd13, %rd13, %rd8;
    bra $L_loop;
$L_next:
    sub.u32 %r20, %r20, 1;
    bra $L_rep;
$L_done:
    ret;
}
";

pub const PTX_COPY: &str = r"
.visible .entry hbm_copy(.param .u64 p_src, .param .u64 p_dst, .param .u64 p_nvec, .param .u32 p_reps)
{
    .reg .pred %p<4>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<28>;
    ld.param.u64 %rd1, [p_src];
    ld.param.u64 %rd17, [p_dst];
    ld.param.u64 %rd2, [p_nvec];
    ld.param.u32 %r20, [p_reps];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %nctaid.x;
    mad.lo.u32 %r5, %r2, %r3, %r1;
    mul.lo.u32 %r6, %r3, %r4;
    cvt.u64.u32 %rd4, %r5;
    cvt.u64.u32 %rd5, %r6;
    shl.b64 %rd6, %rd5, 4;
    shl.b64 %rd7, %rd5, 2;
    shl.b64 %rd8, %rd5, 6;
    shl.b64 %rd10, %rd4, 4;
    add.u64 %rd11, %rd1, %rd10;
    add.u64 %rd18, %rd17, %rd10;
$L_rep:
    setp.eq.u32 %p1, %r20, 0;
    @%p1 bra $L_done;
    mov.u64 %rd12, %rd4;
    mov.u64 %rd13, %rd11;
    mov.u64 %rd19, %rd18;
$L_loop:
    setp.ge.u64 %p2, %rd12, %rd2;
    @%p2 bra $L_next;
    ld.global.nc.v4.u32 {%r8, %r9, %r10, %r11}, [%rd13];
    add.u64 %rd14, %rd13, %rd6;
    ld.global.nc.v4.u32 {%r12, %r13, %r14, %r15}, [%rd14];
    add.u64 %rd15, %rd14, %rd6;
    ld.global.nc.v4.u32 {%r16, %r17, %r18, %r19}, [%rd15];
    add.u64 %rd16, %rd15, %rd6;
    ld.global.nc.v4.u32 {%r21, %r22, %r23, %r24}, [%rd16];
    st.global.cs.v4.u32 [%rd19], {%r8, %r9, %r10, %r11};
    add.u64 %rd20, %rd19, %rd6;
    st.global.cs.v4.u32 [%rd20], {%r12, %r13, %r14, %r15};
    add.u64 %rd21, %rd20, %rd6;
    st.global.cs.v4.u32 [%rd21], {%r16, %r17, %r18, %r19};
    add.u64 %rd22, %rd21, %rd6;
    st.global.cs.v4.u32 [%rd22], {%r21, %r22, %r23, %r24};
    add.u64 %rd12, %rd12, %rd7;
    add.u64 %rd13, %rd13, %rd8;
    add.u64 %rd19, %rd19, %rd8;
    bra $L_loop;
$L_next:
    sub.u32 %r20, %r20, 1;
    bra $L_rep;
$L_done:
    ret;
}
";

pub fn ptx_source() -> String {
    format!("{PTX_HEAD}{PTX_READ}{PTX_WRITE}{PTX_COPY}")
}

/// The loaded module and its three entry points.
pub struct Probe {
    module: u64,
    read: u64,
    write: u64,
    copy: u64,
}

pub fn driver_check(rc: i32, what: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let hint = match rc {
        200 => "CUDA_ERROR_INVALID_IMAGE",
        218 => "CUDA_ERROR_INVALID_PTX",
        222 => "CUDA_ERROR_UNSUPPORTED_PTX_VERSION",
        209 => "CUDA_ERROR_NO_BINARY_FOR_GPU",
        _ => "see the CUDA driver API error table",
    };
    bail!("{what} failed: status {rc} ({hint})")
}

impl Probe {
    pub fn load() -> Result<Self> {
        let src = CString::new(ptx_source())?;
        let mut module = 0u64;
        // SAFETY: `src` is NUL-terminated PTX text that outlives the call.
        driver_check(
            unsafe { cuModuleLoadData(&mut module, src.as_ptr().cast()) },
            "cuModuleLoadData (the embedded probe PTX)",
        )?;
        let get = |name: &str| -> Result<u64> {
            let c = CString::new(name)?;
            let mut f = 0u64;
            // SAFETY: valid module handle, NUL-terminated name, valid out-pointer.
            driver_check(
                unsafe { cuModuleGetFunction(&mut f, module, c.as_ptr()) },
                &format!("cuModuleGetFunction({name})"),
            )?;
            Ok(f)
        };
        Ok(Self {
            module,
            read: get("hbm_read")?,
            write: get("hbm_write")?,
            copy: get("hbm_copy")?,
        })
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        // SAFETY: the module came from cuModuleLoadData and is unloaded once.
        unsafe { cuModuleUnload(self.module) };
    }
}

fn launch(f: u64, g: Geom, stream: u64, params: &mut [*mut c_void]) -> Result<()> {
    // SAFETY: `params` points at argument values that match the entry's
    // .param list and live until the call returns (cuLaunchKernel copies them).
    let rc = unsafe {
        cuLaunchKernel(
            f,
            g.grid,
            1,
            1,
            g.block,
            1,
            1,
            0,
            stream,
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    driver_check(rc, "cuLaunchKernel")
}

impl Probe {
    /// Read `n_vec` 16-byte vectors from `src`, `reps` times.
    pub fn read(
        &self,
        g: Geom,
        stream: u64,
        src: DevicePtr,
        n_vec: u64,
        reps: u32,
        sink: DevicePtr,
    ) -> Result<()> {
        let (mut a, mut n, mut r, mut k) = (src.0, n_vec, reps, sink.0);
        let mut params = [
            (&mut a as *mut u64).cast::<c_void>(),
            (&mut n as *mut u64).cast::<c_void>(),
            (&mut r as *mut u32).cast::<c_void>(),
            (&mut k as *mut u64).cast::<c_void>(),
        ];
        launch(self.read, g, stream, &mut params)
    }

    /// Write `n_vec` 16-byte vectors to `dst`, `reps` times.
    pub fn write(&self, g: Geom, stream: u64, dst: DevicePtr, n_vec: u64, reps: u32) -> Result<()> {
        let (mut a, mut n, mut r) = (dst.0, n_vec, reps);
        let mut params = [
            (&mut a as *mut u64).cast::<c_void>(),
            (&mut n as *mut u64).cast::<c_void>(),
            (&mut r as *mut u32).cast::<c_void>(),
        ];
        launch(self.write, g, stream, &mut params)
    }

    /// Copy `n_vec` 16-byte vectors from `src` to `dst`, `reps` times.
    pub fn copy(
        &self,
        g: Geom,
        stream: u64,
        src: DevicePtr,
        dst: DevicePtr,
        n_vec: u64,
        reps: u32,
    ) -> Result<()> {
        let (mut a, mut b, mut n, mut r) = (src.0, dst.0, n_vec, reps);
        let mut params = [
            (&mut a as *mut u64).cast::<c_void>(),
            (&mut b as *mut u64).cast::<c_void>(),
            (&mut n as *mut u64).cast::<c_void>(),
            (&mut r as *mut u32).cast::<c_void>(),
        ];
        launch(self.copy, g, stream, &mut params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_has_three_entries_and_balanced_braces() {
        let p = ptx_source();
        for e in ["hbm_read", "hbm_write", "hbm_copy"] {
            assert!(p.contains(&format!(".visible .entry {e}(")), "{e}");
        }
        assert!(p.starts_with(".version 7.0\n.target sm_80"));
        // Vector operands open and close on one line, so the counts balance.
        assert_eq!(p.matches('{').count(), p.matches('}').count());
        assert!(p.is_ascii());
    }

    /// Highest register index with this prefix, e.g. `%rd` in `%rd17`.
    fn max_index(src: &str, prefix: &str) -> usize {
        src.match_indices(prefix)
            .filter_map(|(i, _)| {
                let d: String = src[i + prefix.len()..]
                    .chars()
                    .take_while(char::is_ascii_digit)
                    .collect();
                d.parse::<usize>().ok()
            })
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn every_ptx_register_is_declared() {
        // The highest register used in each kernel must sit inside its .reg
        // range: %r<32>, %rd<20> (read, write) or %rd<28> (copy), %p<4>.
        for (name, src, rd_decl) in [
            ("read", PTX_READ, 20),
            ("write", PTX_WRITE, 20),
            ("copy", PTX_COPY, 28),
        ] {
            assert!(max_index(src, "%rd") < rd_decl, "{name} %rd");
            assert!(max_index(src, "%r") < 32, "{name} %r");
            assert!(max_index(src, "%p") < 4, "{name} %p");
        }
    }

    #[test]
    fn a_trip_is_four_vectors_per_thread() {
        let g = Geom {
            grid: 272,
            block: 256,
        };
        assert_eq!(g.trip_bytes(), 272 * 256 * 4 * 16);
    }

    #[test]
    fn driver_errors_name_the_ptx_failures() {
        let e = driver_check(218, "load").unwrap_err().to_string();
        assert!(e.contains("INVALID_PTX"), "{e}");
        assert!(driver_check(0, "ok").is_ok());
    }
}
