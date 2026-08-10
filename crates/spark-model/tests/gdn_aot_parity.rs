// SPDX-License-Identifier: AGPL-3.0-only
//
//! Bit-parity gate for the LINKED FlashInfer GDN AOT kernel (Track B vendoring).
//!
//! The vendoring swap (spark-model/build.rs `build_gdn_aot`) changed HOW the
//! kernel is reached — static link + `gdn_cute_rt_stub.cpp` instead of
//! dlopen(libatlasgdn.so) + the proprietary `libcute_dsl_runtime.so` — but not
//! WHAT runs: the same committed `gdn_holo_0.o`, the same shim. This test
//! proves that by running both paths in one process on identical random inputs
//! at the validated Holo shape (nk=16, nv=32, kd=vd=128) and requiring
//! BIT-IDENTICAL outputs, full-sequence and chunked (state-carry) both.
//! Any drift here means the stub's forwarding semantics differ from the cute
//! runtime's — exactly the failure the stub's disassembly-derived signatures
//! must be checked against.
//!
//! `#[ignore]`d: needs a GB10 GPU, the `atlas_gdn_aot` cfg, and a reference
//! .so. Build the reference from the SAME committed artifacts the linked path
//! uses, with the ORIGINAL proprietary runtime (see STATUS.md "Track B"):
//!
//! ```text
//! cd 3rdparty_patches/gdn_aot
//! nvcc -arch=sm_121a -Xcompiler -fPIC -c gdn_transpose.cu -o /tmp/gdn_transpose.o
//! g++ -O2 -fPIC -shared gdn_shim.cpp /tmp/gdn_transpose.o gdn_holo_0.o \
//!   -o /tmp/libatlasgdn_ref.so -I. -I/usr/local/cuda/include \
//!   -L/usr/local/cuda/lib64 -lcudart \
//!   -L<cute_dsl_lib_dir> -lcute_dsl_runtime -Wl,-rpath,<cute_dsl_lib_dir>
//! ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL='*' ATLAS_TARGET_QUANT='*' \
//! ATLAS_GDN_PARITY_SO=/tmp/libatlasgdn_ref.so \
//! LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat:$LD_LIBRARY_PATH \
//!   cargo test -p spark-model --test gdn_aot_parity -- --ignored --nocapture
//! ```
//!
//! Without `ATLAS_GDN_PARITY_SO` the test SKIPS (prints why, passes): the
//! reference .so needs the proprietary runtime, which is exactly what the
//! production build no longer ships.
#![cfg(all(atlas_gdn_aot, unix))]

// Link the parent crate even though no item is used: the spark-model rlib is
// what carries the `libatlas_gdn_aot.a` + static-cudart native-link
// requirements (build_gdn_aot's rustc-link-lib directives); an unused extern
// crate is otherwise dropped and the `extern "C"` blocks below go unresolved.
use spark_model as _;

use std::os::raw::{c_char, c_float, c_int, c_void};

// Holo-3.1-35B GDN shape (the gate in trait_prefill_gdn.rs).
const T: usize = 2048;
const NK: usize = 16;
const NV: usize = 32;
const KD: usize = 128;
const VD: usize = 128;
const KEY_DIM: usize = NK * KD; // 2048
const VALUE_DIM: usize = NV * VD; // 4096
const CONV_DIM: usize = 2 * KEY_DIM + VALUE_DIM; // 8192
const GB_STRIDE: usize = 2 * NV; // 64
const SCALE: f32 = 0.088_388_35; // 1/sqrt(128)

// Linked-in AOT entry points (libatlas_gdn_aot.a — the code under test).
unsafe extern "C" {
    fn atlas_gdn_load();
    fn atlas_gdn_prefill_packed_managed(
        qkv: *mut c_void,
        gate_beta: *mut c_void,
        output: *mut c_void,
        h_state: *mut c_void,
        scale: c_float,
        total_seqlen: c_int,
        nk: c_int,
        nv: c_int,
        kd: c_int,
        vd: c_int,
        conv_dim: c_int,
        gb_stride: c_int,
        num_seqs: c_int,
        stream: *mut c_void,
    ) -> c_int;
}

// Raw cudart (statically linked by build_gdn_aot — no backend/PTX ceremony,
// this test exercises the AOT artifact, not Atlas kernels).
unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, n: usize, kind: c_int) -> c_int;
    fn cudaMemset(ptr: *mut c_void, v: c_int, n: usize) -> c_int;
    fn cudaDeviceSynchronize() -> c_int;
    fn cudaGetLastError() -> c_int;
}
const H2D: c_int = 1;
const D2H: c_int = 2;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
const RTLD_NOW: c_int = 2;

type PackedFn = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    c_float,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *mut c_void,
) -> c_int;

fn ck(what: &str, code: c_int) {
    assert_eq!(code, 0, "{what} failed: cuda error {code}");
}

fn dmalloc(n: usize) -> *mut c_void {
    let mut p: *mut c_void = std::ptr::null_mut();
    ck("cudaMalloc", unsafe { cudaMalloc(&mut p, n) });
    p
}

fn h2d(dst: *mut c_void, src: &[u8]) {
    ck("cudaMemcpy h2d", unsafe {
        cudaMemcpy(dst, src.as_ptr() as *const c_void, src.len(), H2D)
    });
}

fn d2h(dst: &mut [u8], src: *mut c_void) {
    ck("cudaMemcpy d2h", unsafe {
        cudaMemcpy(dst.as_mut_ptr() as *mut c_void, src, dst.len(), D2H)
    });
}

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn r(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
}

/// One prefill schedule (full or chunked with state carry) through `f`.
/// Returns (output bytes, h_state bytes) read back after a device sync.
fn run_path(
    f: PackedFn,
    qkv_d: *mut c_void,
    gb_d: *mut c_void,
    chunks: &[usize],
) -> (Vec<u8>, Vec<u8>) {
    let out_bytes = T * VALUE_DIM * 2;
    let state_bytes = NV * KD * VD * 4;
    let out_d = dmalloc(out_bytes);
    let state_d = dmalloc(state_bytes);
    ck("memset out", unsafe { cudaMemset(out_d, 0, out_bytes) });
    ck("memset state", unsafe {
        cudaMemset(state_d, 0, state_bytes)
    });
    let mut row = 0usize;
    for &chunk in chunks {
        // Chunked prefill hands the shim per-chunk base pointers and carries
        // h_state across calls — same contract as trait_prefill_gdn.rs.
        let qkv_c = (qkv_d as usize + row * CONV_DIM * 2) as *mut c_void;
        let gb_c = (gb_d as usize + row * GB_STRIDE * 4) as *mut c_void;
        let out_c = (out_d as usize + row * VALUE_DIM * 2) as *mut c_void;
        let ret = unsafe {
            f(
                qkv_c,
                gb_c,
                out_c,
                state_d,
                SCALE,
                chunk as c_int,
                NK as c_int,
                NV as c_int,
                KD as c_int,
                VD as c_int,
                CONV_DIM as c_int,
                GB_STRIDE as c_int,
                1,
                std::ptr::null_mut(), // legacy default stream
            )
        };
        assert_eq!(ret, 0, "atlas_gdn_prefill_packed_managed returned {ret}");
        row += chunk;
    }
    ck("sync", unsafe { cudaDeviceSynchronize() });
    ck("last error", unsafe { cudaGetLastError() });
    let mut out = vec![0u8; out_bytes];
    let mut state = vec![0u8; state_bytes];
    d2h(&mut out, out_d);
    d2h(&mut state, state_d);
    (out, state)
}

/// Max abs diff over bf16 buffers + mismatch count (diagnostics on failure).
fn bf16_diff(a: &[u8], b: &[u8]) -> (f32, usize) {
    let (mut max, mut n) = (0f32, 0usize);
    for (ca, cb) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
        if ca != cb {
            n += 1;
            let fa = half::bf16::from_bits(u16::from_le_bytes([ca[0], ca[1]])).to_f32();
            let fb = half::bf16::from_bits(u16::from_le_bytes([cb[0], cb[1]])).to_f32();
            max = max.max((fa - fb).abs());
        }
    }
    (max, n)
}

fn f32_diff(a: &[u8], b: &[u8]) -> (f32, usize) {
    let (mut max, mut n) = (0f32, 0usize);
    for (ca, cb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        if ca != cb {
            n += 1;
            let fa = f32::from_le_bytes([ca[0], ca[1], ca[2], ca[3]]);
            let fb = f32::from_le_bytes([cb[0], cb[1], cb[2], cb[3]]);
            max = max.max((fa - fb).abs());
        }
    }
    (max, n)
}

#[test]
#[ignore] // Requires a GB10 GPU + ATLAS_GDN_PARITY_SO (see module docs).
fn linked_aot_bit_matches_dlopen_so() {
    let Ok(so_path) = std::env::var("ATLAS_GDN_PARITY_SO") else {
        eprintln!(
            "SKIP: ATLAS_GDN_PARITY_SO not set — no reference libatlasgdn.so to compare \
             against (build recipe in the module docs)"
        );
        return;
    };

    // Reference path: dlopen the cute-runtime-linked .so.
    let cpath = std::ffi::CString::new(so_path.clone()).unwrap();
    let h = unsafe { dlopen(cpath.as_ptr(), RTLD_NOW) };
    assert!(!h.is_null(), "dlopen({so_path}) failed");
    let ref_load = unsafe { dlsym(h, c"atlas_gdn_load".as_ptr()) };
    let ref_prefill = unsafe { dlsym(h, c"atlas_gdn_prefill_packed_managed".as_ptr()) };
    assert!(
        !ref_load.is_null() && !ref_prefill.is_null(),
        "symbols missing in {so_path}"
    );
    // SAFETY: same shim source, same signature as the linked prototype above.
    let ref_load: unsafe extern "C" fn() = unsafe { std::mem::transmute(ref_load) };
    let ref_prefill: PackedFn = unsafe { std::mem::transmute(ref_prefill) };

    // Inputs: bf16 qkv in ±1, gate = linear alpha in (0.6, 1.0), beta in (0, 1)
    // — plausible post-activation ranges; both paths read the SAME device
    // buffers, so ranges only need to keep the scan finite.
    let mut rng = Lcg(0x5EED_6D4E_A07B_1230);
    let mut qkv = vec![0u8; T * CONV_DIM * 2];
    for c in qkv.chunks_exact_mut(2) {
        c.copy_from_slice(
            &half::bf16::from_f32(rng.r(-1.0, 1.0))
                .to_bits()
                .to_le_bytes(),
        );
    }
    let mut gb = vec![0u8; T * GB_STRIDE * 4];
    for t in 0..T {
        for hh in 0..NV {
            let g = rng.r(0.6, 1.0);
            let b = rng.r(0.05, 0.95);
            gb[(t * GB_STRIDE + hh) * 4..][..4].copy_from_slice(&g.to_le_bytes());
            gb[(t * GB_STRIDE + NV + hh) * 4..][..4].copy_from_slice(&b.to_le_bytes());
        }
    }
    let qkv_d = dmalloc(qkv.len());
    h2d(qkv_d, &qkv);
    let gb_d = dmalloc(gb.len());
    h2d(gb_d, &gb);

    unsafe { atlas_gdn_load() };
    unsafe { ref_load() };

    // Scenario A: one full-sequence call. Scenario B: chunked prefill with
    // state carry (the multi-chunk contract 5fe12ddf's cu_seqlens fix guards).
    for (name, chunks) in [
        ("full T=2048", &[T][..]),
        ("chunked 1536+512 (state carry)", &[1536usize, 512][..]),
    ] {
        let (out_l, st_l) = run_path(atlas_gdn_prefill_packed_managed, qkv_d, gb_d, chunks);
        let (out_r, st_r) = run_path(ref_prefill, qkv_d, gb_d, chunks);

        // Anti-triviality: a no-op parity pass (both all-zero) must not count.
        let nonzero = out_l.chunks_exact(2).filter(|c| *c != [0, 0]).count();
        assert!(
            nonzero > out_l.len() / 8,
            "{name}: linked output suspiciously sparse ({nonzero} nonzero bf16 of {})",
            out_l.len() / 2
        );

        let (omax, on) = bf16_diff(&out_l, &out_r);
        let (smax, sn) = f32_diff(&st_l, &st_r);
        println!(
            "{name}: output {} bytes, state {} bytes — mismatches out={on} (max abs {omax:e}) \
             state={sn} (max abs {smax:e})",
            out_l.len(),
            st_l.len()
        );
        assert!(
            on == 0 && sn == 0,
            "{name}: linked AOT kernel is NOT bit-identical to the dlopen reference: \
             out mismatches={on} max={omax:e}, state mismatches={sn} max={smax:e}"
        );
        println!("{name}: BIT-EXACT ✅");
    }
}
