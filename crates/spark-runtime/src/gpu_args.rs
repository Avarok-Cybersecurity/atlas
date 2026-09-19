// SPDX-License-Identifier: AGPL-3.0-only

//! Packing typed kernel arguments into the driver's u64 parameter slots.
//!
//! Split out of `gpu.rs` for the repo's 500-line cap. It earns its own file: the
//! packing is the one place that knows an argument may span MORE than one slot,
//! and the version this replaced silently truncated anything wider than 8 bytes.

use crate::gpu::KernelArg;

/// Pack typed args into u64 slots, returning the slots and each argument's
/// STARTING slot index.
///
/// Slots are u64-granular, but an argument is NOT limited to one slot: a
/// by-value struct parameter (`CUtensorMap` is 128 bytes) occupies
/// `ceil(len/8)` CONSECUTIVE slots and contributes exactly ONE entry to the
/// param array, pointing at the first.
///
/// ★ The packing this replaces was `let n = b.len().min(8)` — anything wider
/// was TRUNCATED TO ITS FIRST 8 BYTES AND LAUNCHED, with no error. Nothing
/// passed more than 8 bytes yet, so it never fired; the first caller that did
/// would have got a kernel reading garbage out of a struct parameter and no
/// diagnostic anywhere. Silent truncation is not an acceptable failure mode on
/// the launch path, so this is a free function with its own tests rather than
/// eight lines buried in a default trait method.
pub fn pack_kernel_args(args: &[KernelArg<'_>]) -> (Vec<u64>, Vec<usize>) {
    let total_slots: usize = args
        .iter()
        .map(|a| match a {
            KernelArg::Buffer(_) => 1,
            KernelArg::Bytes(b) => b.len().div_ceil(8).max(1),
        })
        .sum();
    let mut storage: Vec<u64> = Vec::with_capacity(total_slots);
    let mut starts: Vec<usize> = Vec::with_capacity(args.len());
    for arg in args {
        starts.push(storage.len());
        match arg {
            KernelArg::Buffer(p) => storage.push(p.0),
            KernelArg::Bytes(b) => {
                for c in b.chunks(8) {
                    let mut slot = [0u8; 8];
                    slot[..c.len()].copy_from_slice(c);
                    storage.push(u64::from_le_bytes(slot));
                }
                if b.is_empty() {
                    storage.push(0);
                }
            }
        }
    }
    (storage, starts)
}
