// SPDX-License-Identifier: AGPL-3.0-only
//
// Build-time codegen gate for the frozen FlashQLA cubin. Included via
// `#[path = "build_flashqla.rs"] mod build_flashqla;` from build.rs.
//
// The `flashqla_gdn` device source is AOT-compiled to an ELF cubin with the
// exact TileLang-toolchain configuration (`--cubin -arch=sm_121a -O3
// --use_fast_math -std=c++17`).  Compiling it as PTX without `--use_fast_math`
// produced SASS that waited forever on a GB10 TMA barrier inside
// `cp_prepare_h`; the fast-math cubin is byte-identical to the validated
// TileLang kernel.  This module verifies that identity before the cubin is
// embedded so a silent codegen drift cannot reintroduce the hang.

use std::path::Path;
use std::process::Command;

/// Whether `source` is the frozen FlashQLA device source (the only Atlas
/// kernel AOT-compiled to an ELF cubin).  Used by `build_target.rs` for the
/// fixed compile configuration and per-source output extension.
pub(crate) fn is_flashqla_source(source: &Path) -> bool {
    source
        .file_stem()
        .and_then(|s| s.to_str())
        .is_some_and(|stem| stem == "flashqla_gdn")
}

/// The ten `__global__` symbols the native FlashQLA launcher resolves from
/// the embedded `flashqla_gdn` module.  `fused_nocp` remains as a diagnostic
/// baseline; production no-CP uses `flashqla_fused_nocp_packed_strided`.
const FLASHQLA_SYMBOLS: &[&str] = &[
    "flashqla_unpack_gate_beta",
    "flashqla_chunk_local_cumsum",
    "flashqla_kkt_solve",
    "flashqla_fused_nocp",
    "flashqla_fused_nocp_packed_strided",
    "flashqla_cp_warmup",
    "flashqla_prepare_h_packed_strided",
    "flashqla_cp_correct_h0",
    "flashqla_kkt_packed_strided",
    "flashqla_fused_cp_packed_strided",
    "flashqla_fused_cp_packed_strided_qkg_pair",
];

/// Normalized SASS fingerprint of `cp_prepare_h` under CUDA 13.0.88
/// (SHA-256 over the cuobjdump encoding words joined with `\n`, plus a
/// trailing newline).  Verified byte-identical to the runnable TileLang
/// cubin; the no-fast-math PTX build produced 8672 words and hung.
const VERIFIED_PREPARE_H_SASS_SHA256: &str =
    "456690dfd3b92653aa86dfcff5a10a8b13637eb938212d73f05e145d76a000a7";

/// Normalized SASS fingerprint of the packed-strided no-CP fused kernel under
/// CUDA 13.0.88.  Verified byte-identical to the TileLang-generated cubin; its
/// masked fallback uses the packed 8192-element token pitch for Q/K/V (the
/// contiguous `flashqla_fused_nocp` used 2048/2048/4096 and read the wrong
/// elements on Atlas's packed QKV buffer for T < chunk).
const VERIFIED_FUSED_NOCP_PACKED_SASS_SHA256: &str =
    "f7b5e9f93c05b72ca2610954b1b620a5c81db1d503b237b0bd659d3726513367";

/// Verify the AOT `flashqla_gdn` cubin before it is embedded into the
/// binary.  Panics on any gate failure so a codegen regression aborts the
/// build instead of shipping a kernel that hangs the GPU.
///
/// `cuda_bin` is the CUDA toolkit `bin/` dir (`{cuda_dir}/bin`) holding
/// `cuobjdump`; the hash gate is enforced only when the toolkit matches the
/// version the fingerprint was validated against (CUDA 13.0.88), because SASS
/// is not guaranteed to be byte-identical across toolkit releases.  ELF /
/// symbol / attribute gates run unconditionally.
pub fn verify_flashqla_cubin(cubin: &Path, cuda_bin: &Path) {
    // ── Gate 1: product must be an ELF cubin ──
    let bytes = std::fs::read(cubin).unwrap_or_else(|e| {
        panic!(
            "atlas-kernels: read flashqla cubin {}: {e}",
            cubin.display()
        )
    });
    assert!(
        bytes.starts_with(b"\x7fELF"),
        "atlas-kernels: flashqla_gdn output {} is not an ELF cubin (codegen gate)",
        cubin.display()
    );

    // ── Gate 2: all nine FlashQLA symbols present ──
    let cuobjdump = cuda_bin.join("cuobjdump");
    let sass = run(&cuobjdump, &["--dump-sass"], cubin, "cuobjdump --dump-sass");
    for symbol in FLASHQLA_SYMBOLS {
        assert!(
            sass.contains(&format!("Function : {symbol}")),
            "atlas-kernels: flashqla_gdn cubin missing symbol {symbol} (codegen gate)"
        );
    }

    // ── Gate 3: cp_prepare_h resource attributes pinned ──
    let res = run(
        &cuobjdump,
        &["--dump-resource-usage"],
        cubin,
        "cuobjdump --dump-resource-usage",
    );
    let prepare_block = res
        .split("Function flashqla_prepare_h_packed_strided:")
        .nth(1)
        .and_then(|b| b.split("Function ").next())
        .unwrap_or("");
    for (attr, want) in [
        ("REG", "REG:128"),
        ("STACK", "STACK:72"),
        ("SHARED", "SHARED:2048"),
        ("CONSTANT[0]", "CONSTANT[0]:1928"),
    ] {
        assert!(
            prepare_block.contains(want),
            "atlas-kernels: cp_prepare_h {attr} changed (want {want}, got {prepare_block:?}) (codegen gate)"
        );
    }

    // ── Gate 3b: packed-strided no-CP fused attributes pinned ──
    let fused_block = res
        .split("Function flashqla_fused_nocp_packed_strided:")
        .nth(1)
        .and_then(|b| b.split("Function ").next())
        .unwrap_or("");
    for (attr, want) in [
        ("REG", "REG:128"),
        ("STACK", "STACK:8"),
        ("SHARED", "SHARED:2048"),
        ("CONSTANT[0]", "CONSTANT[0]:2188"),
    ] {
        assert!(
            fused_block.contains(want),
            "atlas-kernels: flashqla_fused_nocp_packed_strided {attr} changed (want {want}, got {fused_block:?}) (codegen gate)"
        );
    }

    // ── Gate 3c: qkg_pair fused attributes pinned ──
    let qkg_block = res
        .split("Function flashqla_fused_cp_packed_strided_qkg_pair:")
        .nth(1)
        .and_then(|b| b.split("Function ").next())
        .unwrap_or("");
    for (attr, want) in [
        ("REG", "REG:128"),
        ("STACK", "STACK:8"),
        ("SHARED", "SHARED:2048"),
        ("CONSTANT[0]", "CONSTANT[0]:2444"),
    ] {
        assert!(
            qkg_block.contains(want),
            "atlas-kernels: qkg_pair fused {attr} changed (want {want}, got {qkg_block:?}) (codegen gate)"
        );
    }

    // ── Gate 4: normalized SASS fingerprint (CUDA 13.0.88 only) ──
    let nvcc_out = Command::new(cuda_bin.join("nvcc"))
        .arg("--version")
        .output()
        .expect("atlas-kernels: failed to run nvcc --version");
    let nvcc_ver = String::from_utf8_lossy(&nvcc_out.stdout);
    if nvcc_ver.contains("V13.0.88") {
        let words = prepare_h_sass_words(&sass);
        let normalized = format!("{}\n", words.join("\n"));
        let actual = sha256(normalized.as_bytes());
        assert_eq!(
            actual, VERIFIED_PREPARE_H_SASS_SHA256,
            "atlas-kernels: flashqla_gdn cp_prepare_h SASS drift (codegen gate). \
             Got {actual}, want {VERIFIED_PREPARE_H_SASS_SHA256} — re-run the TileLang \
             byte-identity comparison before landing."
        );

        // Gate 4b: packed-strided no-CP fused SASS fingerprint.
        let fused_words = kernel_sass_words(&sass, "flashqla_fused_nocp_packed_strided");
        let fused_norm = format!("{}\n", fused_words.join("\n"));
        let fused_actual = sha256(fused_norm.as_bytes());
        assert_eq!(
            fused_actual, VERIFIED_FUSED_NOCP_PACKED_SASS_SHA256,
            "atlas-kernels: flashqla_gdn flashqla_fused_nocp_packed_strided SASS drift \
             (codegen gate). Got {fused_actual}, want \
             {VERIFIED_FUSED_NOCP_PACKED_SASS_SHA256} — re-run the TileLang byte-identity \
             comparison before landing."
        );
        println!(
            "cargo:warning=atlas-kernels: flashqla_gdn cubin verified (11 symbols, prepare_h \
             REG:128/STACK:72/SHARED:2048/CONSTANT:1928, fused_nocp_packed_strided \
             REG:128/STACK:8/SHARED:2048/CONSTANT:2188, SASS {actual})"
        );
    } else {
        println!(
            "cargo:warning=atlas-kernels: flashqla_gdn SASS fingerprint gate skipped \
             (toolkit != CUDA 13.0.88); ELF/symbol/attribute gates enforced"
        );
    }
}

/// Run `cuobjdump --dump-sass` and extract the encoding words of
/// `cp_prepare_h` in the order printed.
fn prepare_h_sass_words(sass: &str) -> Vec<String> {
    kernel_sass_words(sass, "flashqla_prepare_h_packed_strided")
}

/// Extract the encoding words of the given kernel from a cuobjdump SASS dump.
fn kernel_sass_words(sass: &str, symbol: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut in_kernel = false;
    for line in sass.lines() {
        if line.contains(&format!("Function : {symbol}")) {
            in_kernel = true;
            continue;
        }
        if in_kernel && line.contains("Function : ") {
            break;
        }
        if in_kernel {
            // `/* 0x0000df00ff017b82 */` — the 64-bit instruction encoding.
            if let Some(start) = line.find("/* 0x") {
                let rest = &line[start + 2..];
                if let Some(end) = rest.find(" */") {
                    let token = rest[..end].trim();
                    if token.len() == 18 && token.starts_with("0x") {
                        words.push(token.to_string());
                    }
                }
            }
        }
    }
    words
}

/// Run `cmd` with `args` followed by `file` and return stdout as a lossy
/// string, panicking on a non-zero exit.
fn run(cmd: &Path, args: &[&str], file: &Path, what: &str) -> String {
    let out = Command::new(cmd)
        .args(args)
        .arg(file)
        .output()
        .unwrap_or_else(|e| panic!("atlas-kernels: failed to run {what}: {e}"));
    assert!(
        out.status.success(),
        "atlas-kernels: {what} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// FIPS 180-4 SHA-256.  The build script has no crate deps beyond `toml`;
/// a 100-line local implementation keeps the gate dependency-free.
pub(crate) fn sha256(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());

    let mut w = [0u32; 64];
    for chunk in msg.chunks_exact(64) {
        for (i, word) in w.iter_mut().take(16).enumerate() {
            *word = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    h.iter().map(|v| format!("{v:08x}")).collect()
}
