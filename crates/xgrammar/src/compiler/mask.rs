// SPDX-License-Identifier: AGPL-3.0-only
//
// AdaptiveTokenMask — port of `struct AdaptiveTokenMask` from
// `cpp/compiled_grammar_impl.h` + `cpp/compiled_grammar.cc`.
//
// Preprocessed per-`ParserState` partition of the token vocabulary into
// three categories:
//   * accepted  — tokens the state alone proves acceptable;
//   * rejected  — tokens the state alone proves unacceptable;
//   * uncertain — tokens whose acceptance needs the parent states.
//
// To save memory the accepted/rejected partition is stored in one of
// three forms (see [`StoreType`]); uncertain indices are always stored
// directly. All indices are positions into the tokenizer's
// `sorted_decoded_vocab`, not raw token ids — this is what makes the
// matcher's bitmask fill fast.

use bitvec::vec::BitVec;

/// Above this many accepted (and rejected) indices, the partition is
/// stored as a vocab-sized bitset instead of an index vector. Faithful
/// to the C++ `AdaptiveTokenMask::USE_BITSET_THRESHOLD`.
pub const USE_BITSET_THRESHOLD: usize = 1000;

/// How an [`AdaptiveTokenMask`] stores its accepted/rejected partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreType {
    /// Only the accepted indices are stored; rejected = all − accepted
    /// − uncertain. Used when `|accepted| < |rejected|`.
    Accepted,
    /// Only the rejected indices are stored; accepted = all − rejected
    /// − uncertain. Used when `|accepted| > |rejected|`.
    Rejected,
    /// Accepted token *ids* are stored in a vocab-sized bitset. Used
    /// when both partitions are large.
    AcceptedBitset,
}

/// Preprocessed accept/reject/uncertain partition for one parser state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveTokenMask {
    /// Which representation the partition uses.
    pub store_type: StoreType,
    /// Accepted sorted-vocab indices (only for [`StoreType::Accepted`]).
    pub accepted_indices: Vec<i32>,
    /// Rejected sorted-vocab indices (only for [`StoreType::Rejected`]).
    pub rejected_indices: Vec<i32>,
    /// Accepted token ids as a bitset (only for
    /// [`StoreType::AcceptedBitset`]); length == vocab size.
    pub accepted_bitset: BitVec,
    /// Uncertain sorted-vocab indices (always stored directly).
    pub uncertain_indices: Vec<i32>,
}

impl AdaptiveTokenMask {
    /// Build a mask from accepted + rejected + uncertain partitions.
    ///
    /// Port of the four-argument C++ constructor. `sorted_decoded_vocab`
    /// supplies the `index -> token_id` mapping for the bitset form.
    pub fn from_accepted_rejected(
        vocab_size: usize,
        sorted_decoded_vocab: &[(i32, Vec<u8>)],
        accepted_indices: &[i32],
        rejected_indices: &[i32],
        uncertain_indices: &[i32],
    ) -> Self {
        let size_acc = accepted_indices.len();
        let size_rej = rejected_indices.len();

        let store_type = if size_acc >= USE_BITSET_THRESHOLD && size_rej >= USE_BITSET_THRESHOLD {
            StoreType::AcceptedBitset
        } else if size_acc < size_rej {
            StoreType::Accepted
        } else {
            StoreType::Rejected
        };

        let mut mask = Self::empty(store_type, vocab_size);
        match store_type {
            StoreType::AcceptedBitset => {
                for &idx in accepted_indices {
                    let id = sorted_decoded_vocab[idx as usize].0 as usize;
                    mask.accepted_bitset.set(id, true);
                }
            }
            StoreType::Accepted => mask.accepted_indices = accepted_indices.to_vec(),
            StoreType::Rejected => mask.rejected_indices = rejected_indices.to_vec(),
        }
        mask.uncertain_indices = uncertain_indices.to_vec();
        mask
    }

    /// Build a mask from only accepted + uncertain partitions (the
    /// rejected partition was not materialized — too large to be worth
    /// storing). Port of the three-argument C++ constructor.
    pub fn from_accepted(
        vocab_size: usize,
        sorted_decoded_vocab: &[(i32, Vec<u8>)],
        accepted_indices: &[i32],
        uncertain_indices: &[i32],
    ) -> Self {
        let store_type = if accepted_indices.len() >= USE_BITSET_THRESHOLD {
            StoreType::AcceptedBitset
        } else {
            StoreType::Accepted
        };

        let mut mask = Self::empty(store_type, vocab_size);
        match store_type {
            StoreType::AcceptedBitset => {
                for &idx in accepted_indices {
                    let id = sorted_decoded_vocab[idx as usize].0 as usize;
                    mask.accepted_bitset.set(id, true);
                }
            }
            StoreType::Accepted => mask.accepted_indices = accepted_indices.to_vec(),
            StoreType::Rejected => unreachable!("from_accepted never produces Rejected"),
        }
        mask.uncertain_indices = uncertain_indices.to_vec();
        mask
    }

    /// An empty mask of the given store type — used internally and for
    /// the `vocab_size == 0` degenerate compile path.
    pub fn empty(store_type: StoreType, vocab_size: usize) -> Self {
        Self {
            store_type,
            accepted_indices: Vec::new(),
            rejected_indices: Vec::new(),
            accepted_bitset: BitVec::repeat(false, vocab_size),
            uncertain_indices: Vec::new(),
        }
    }

    /// Materialize the accepted / rejected partitions as explicit
    /// sorted-vocab index vectors, regardless of [`StoreType`].
    ///
    /// Port of the partition-reconstruction logic in
    /// `AdaptiveTokenMask::Print`. Returns `(accepted, rejected)`.
    pub fn materialize(&self, sorted_decoded_vocab: &[(i32, Vec<u8>)]) -> (Vec<i32>, Vec<i32>) {
        let n = sorted_decoded_vocab.len();
        let uncertain: std::collections::HashSet<i32> =
            self.uncertain_indices.iter().copied().collect();
        let mut accepted = Vec::new();
        let mut rejected = Vec::new();

        match self.store_type {
            StoreType::AcceptedBitset => {
                for i in 0..n {
                    if uncertain.contains(&(i as i32)) {
                        continue;
                    }
                    let id = sorted_decoded_vocab[i].0 as usize;
                    if self.accepted_bitset[id] {
                        accepted.push(i as i32);
                    } else {
                        rejected.push(i as i32);
                    }
                }
            }
            StoreType::Accepted => {
                accepted = self.accepted_indices.clone();
                let mut acc_ptr = 0usize;
                for i in 0..n as i32 {
                    while acc_ptr < accepted.len() && accepted[acc_ptr] < i {
                        acc_ptr += 1;
                    }
                    if acc_ptr < accepted.len() && accepted[acc_ptr] == i {
                        continue;
                    }
                    if uncertain.contains(&i) {
                        continue;
                    }
                    rejected.push(i);
                }
            }
            StoreType::Rejected => {
                rejected = self.rejected_indices.clone();
                let mut rej_ptr = 0usize;
                for i in 0..n as i32 {
                    while rej_ptr < rejected.len() && rejected[rej_ptr] < i {
                        rej_ptr += 1;
                    }
                    if rej_ptr < rejected.len() && rejected[rej_ptr] == i {
                        continue;
                    }
                    if uncertain.contains(&i) {
                        continue;
                    }
                    accepted.push(i);
                }
            }
        }
        (accepted, rejected)
    }

    /// Approximate heap memory used by this mask, in bytes. Port of the
    /// C++ `MemorySize(const AdaptiveTokenMask&)` friend.
    pub fn memory_size(&self) -> usize {
        self.accepted_indices.len() * 4
            + self.rejected_indices.len() * 4
            + self.uncertain_indices.len() * 4
            + self.accepted_bitset.len() / 8
    }
}
