// SPDX-License-Identifier: AGPL-3.0-only
//! Real-file correctness oracle for the GGUF container + CPU ternary dequant.
//!
//! Gated `#[ignore]` so CI without the on-disk model is unaffected. Run with:
//!   ATLAS_SKIP_BUILD=1 cargo test -p spark-runtime -- --ignored gguf_real_file

use super::container::{GgmlType, GgufFile};
use super::dequant_cpu::{self, f16_to_f32};

const REAL_GGUF: &str = "/tank/hf/hub/models--prism-ml--Ternary-Bonsai-27B-gguf/snapshots/\
a4e1b9d50e8e0149ce84544f954006c8f0867f2c/Ternary-Bonsai-27B-dspark-Q4_1.gguf";

/// Parse the real Ternary-Bonsai GGUF, confirm the container facts, and prove
/// the CPU ternary dequant of `token_embd.weight` (ggml id 42, group-128)
/// produces only codes 0/1/2 — i.e. every reconstructed value ∈ {-d, 0, +d}
/// for that block's per-block scale `d`.
#[test]
#[ignore = "requires the on-disk Ternary-Bonsai-27B GGUF (see path)"]
fn gguf_real_file_ternary_bonsai() {
    let file = match std::fs::File::open(REAL_GGUF) {
        Ok(f) => f,
        Err(e) => panic!("cannot open real GGUF {REAL_GGUF}: {e}"),
    };
    // SAFETY: read-only mmap of an existing file for the duration of the test.
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file).expect("mmap") };

    let gguf = GgufFile::parse(&mmap).expect("parse container");

    // Container facts.
    assert_eq!(
        gguf.get_str("general.architecture"),
        Some("dspark"),
        "architecture"
    );
    assert_eq!(gguf.data_offset, 5792, "data-section start");

    let t = gguf
        .tensor("token_embd.weight")
        .expect("token_embd.weight present");
    assert_eq!(t.ggml_type, GgmlType::Q2_0, "token_embd is ggml id 42");
    assert_eq!(t.ggml_type.id(), 42);
    assert_eq!(t.dims, vec![5120, 248320], "token_embd dims (ggml order)");
    assert_eq!(t.offset, 3392, "token_embd rel offset");

    // Dequant the first ~20000 group-128 blocks and assert ternary values.
    const GROUP: usize = 128;
    const BLOCK_BYTES: usize = 2 + GROUP / 4; // 34
    const N_BLOCKS: usize = 20_000;
    let n_elems = N_BLOCKS * GROUP;

    let start = gguf.tensor_abs_offset(t);
    let raw = &mmap[start..start + N_BLOCKS * BLOCK_BYTES];

    let mut out = vec![0f32; n_elems];
    dequant_cpu::dequant_to_f32(
        super::dequant_cpu::GgmlType::Q2_0 { group: GROUP },
        raw,
        n_elems,
        &mut out,
    )
    .expect("cpu dequant");

    for b in 0..N_BLOCKS {
        // Per-block scale d (fp16 at the FRONT of the block).
        let d = f16_to_f32(u16::from_le_bytes([
            raw[b * BLOCK_BYTES],
            raw[b * BLOCK_BYTES + 1],
        ]));
        for j in 0..GROUP {
            let v = out[b * GROUP + j];
            assert!(
                v == -d || v == 0.0 || v == d,
                "block {b} elem {j}: value {v} not in {{-d,0,d}} for d={d} (code 3 / +2d unused)"
            );
        }
    }
}
