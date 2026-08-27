// SPDX-License-Identifier: AGPL-3.0-only

//! The one definition of an MTP proposer's attention-metadata slab.
//!
//! Three call sites pack the identical layout — `MtpHead::forward`,
//! `MtpHead::propose_batch`, and `DeepseekV4MtpHead::propose` — and until this
//! module existed each carried its own copy of the layout constants, its own
//! `unsafe { from_raw_parts }` over the block table, and its own idea of
//! whether the destination region was big enough. Only ONE of the three
//! (`propose_batch`) actually checked. The other two wrote `256 + 4 *
//! block_table.len()` bytes into the shared `scratch` arena at a fixed offset
//! with nothing bounding the block table, whose length grows with the
//! sequence's context — so a long enough context walked off the end of the
//! region into whatever the allocator had placed after it.
//!
//! That is the same shape as the pinned-staging overflow fixed in
//! `913f7c1f`, on the device side instead of the host side: a rule stated in
//! one of several copies is a rule that holds in one of several copies.
//!
//! `deepseek_v4_mtp.rs` even said so in prose — "Mirrors the Qwen `MtpHead`
//! choice of `49152`" — which is the tell that the constant wanted to be
//! shared rather than mirrored.

use anyhow::{Result, ensure};

/// Bytes of header before the block table. The three scalars live inside it at
/// fixed offsets and the rest is zero padding, so `AttnMetadataDev` can point
/// at `base`, `base+8` and `base+16` for positions, slot and seq_len.
pub(crate) const MTP_META_HEADER_BYTES: usize = 256;

/// Scratch-buffer byte offset of a single-sequence MTP metadata slab.
///
/// Must stay distinct from the target model's metadata at `32768` so a
/// `propose()` does not clobber the in-flight target `attn_metadata`. Shared by
/// `MtpHead::forward` and `DeepseekV4MtpHead::propose`, which write the same
/// layout to the same place; `propose_batch` uses its own `propose_meta`
/// allocation instead and passes that region's stride as `region_bytes`.
pub(crate) const MTP_META_OFFSET: usize = 49152;

fn mtp_meta_len(block_entries: usize) -> Result<usize> {
    block_entries
        .checked_mul(4)
        .and_then(|bt_bytes| MTP_META_HEADER_BYTES.checked_add(bt_bytes))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "MTP attention metadata exceeds meta stride: byte size overflows for a {block_entries}-entry block table"
            )
        })
}

/// Pack one MTP attention-metadata slab, refusing up front to build one that
/// would not fit `region_bytes`.
///
/// Layout: `position` u32 @0, `global_slot` i64 @8, `seq_len` i32 @16, then the
/// block table as i32 @[`MTP_META_HEADER_BYTES`..]. Block indices are always
/// well under `2^31`, so the `u32 -> i32` reinterpretation is lossless and the
/// table is written straight from `block_table` without an intermediate
/// `Vec<i32>`.
///
/// `region_bytes` is how many bytes the caller owns at the destination — the
/// per-sequence stride for `propose_batch`, or `scratch_bytes - MTP_META_OFFSET`
/// for the two single-sequence callers. Checking it HERE is the point of this
/// function: the caller cannot forget, because there is no other way to build
/// the slab.
pub(crate) fn pack_mtp_attn_meta(
    position: u32,
    global_slot: i64,
    seq_len: i32,
    block_table: &[u32],
    region_bytes: usize,
) -> Result<Vec<u8>> {
    let need = mtp_meta_len(block_table.len())?;
    // ★ The phrase "exceeds meta stride" is LOAD-BEARING, not decoration:
    // `scheduler/mtp_bootstrap_step.rs` matches it (`msg.contains`) to demote
    // this one overflow to debug while every other propose failure stays at
    // ERROR. Reword it and that demotion silently stops matching, which puts a
    // per-occurrence ERROR back into production logs for a condition that is
    // expected and recoverable. String-matching an error is fragile and worth
    // replacing with a typed error, but until then the coupling is real and
    // documented at both ends.
    ensure!(
        need <= region_bytes,
        "MTP attention metadata exceeds meta stride: needs {need} B for a {}-entry \
         block table, have {region_bytes} B",
        block_table.len()
    );

    let mut buf = vec![0u8; need];
    buf[0..4].copy_from_slice(&position.to_le_bytes());
    buf[8..16].copy_from_slice(&global_slot.to_le_bytes());
    buf[16..20].copy_from_slice(&seq_len.to_le_bytes());
    for (i, &block) in block_table.iter().enumerate() {
        let at = MTP_META_HEADER_BYTES + i * 4;
        buf[at..at + 4].copy_from_slice(&block.to_le_bytes());
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_the_documented_layout() {
        let buf = pack_mtp_attn_meta(7, 0x1234_5678_9abc, 42, &[3, 9], 4096).unwrap();
        assert_eq!(buf.len(), MTP_META_HEADER_BYTES + 8);
        assert_eq!(u32::from_le_bytes(buf[0..4].try_into().unwrap()), 7);
        assert_eq!(
            i64::from_le_bytes(buf[8..16].try_into().unwrap()),
            0x1234_5678_9abc
        );
        assert_eq!(i32::from_le_bytes(buf[16..20].try_into().unwrap()), 42);
        assert_eq!(u32::from_le_bytes(buf[256..260].try_into().unwrap()), 3);
        assert_eq!(u32::from_le_bytes(buf[260..264].try_into().unwrap()), 9);
        // Everything the layout does not define is zero, so a device read of
        // the gap sees a defined value rather than stale scratch.
        assert!(buf[20..256].iter().all(|&b| b == 0));
    }

    /// The check the two single-sequence callers did not have. A block table
    /// long enough to overrun the destination region must be refused BEFORE the
    /// bytes are handed to a copy, not asserted afterwards.
    #[test]
    fn refuses_a_block_table_that_would_overrun_the_region() {
        // 2048 B region, 256 B header => 448 entries fit, 449 do not.
        let fits = vec![0u32; 448];
        assert_eq!(
            pack_mtp_attn_meta(0, 0, 1, &fits, 2048).unwrap().len(),
            2048
        );
        let over = vec![0u32; 449];
        let e = pack_mtp_attn_meta(0, 0, 1, &over, 2048).unwrap_err();
        // ★ Pins the exact phrase `scheduler/mtp_bootstrap_step.rs` matches to
        // demote this overflow to debug. Asserting the wording is the point:
        // without it, a reword here silently restores a per-occurrence ERROR
        // in production for an expected, recoverable condition — and nothing
        // else in the tree would catch it.
        assert!(
            e.to_string().contains("exceeds meta stride"),
            "the demotion phrase mtp_bootstrap_step matches must survive: {e}"
        );
    }

    /// A region smaller than the header alone is rejected rather than producing
    /// a short buffer whose scalar writes would panic on the slice index.
    #[test]
    fn refuses_a_region_too_small_for_the_header() {
        assert!(pack_mtp_attn_meta(0, 0, 1, &[], MTP_META_HEADER_BYTES - 1).is_err());
        assert!(pack_mtp_attn_meta(0, 0, 1, &[], MTP_META_HEADER_BYTES).is_ok());
    }

    #[test]
    fn refuses_an_unrepresentable_block_table_length() {
        let err = mtp_meta_len(usize::MAX).unwrap_err();
        assert!(err.to_string().contains("exceeds meta stride"), "{err}");
    }
}
