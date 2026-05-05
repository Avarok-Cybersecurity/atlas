// SPDX-License-Identifier: AGPL-3.0-only

//! Per-request grammar matching state.

use xgrammar::{
    CompiledGrammar, DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLTensor, GrammarMatcher,
    allocate_token_bitmask, get_bitmask_shape, reset_token_bitmask,
};

use super::engine::GrammarError;

// ── GrammarState ───────────────────────────────────────────────────────

/// Per-request grammar matching state.
///
/// Wraps a [`GrammarMatcher`] with its own bitmask buffer. The bitmask
/// is reused across decode steps to avoid re-allocation.
pub struct GrammarState {
    matcher: GrammarMatcher,
    /// Bitmask buffer: `Box<[i32]>` of shape `(1, ceil(vocab_size / 32))`.
    bitmask_data: Box<[i32]>,
    /// Shape array kept alive for DLTensor pointer stability.
    bitmask_shape: [i64; 2],
    /// Stride array kept alive for DLTensor pointer stability.
    bitmask_strides: [i64; 2],
    vocab_size: usize,
}

impl GrammarState {
    /// Create a new per-request grammar state from a compiled grammar.
    ///
    /// `vocab_size` must match the tokenizer vocabulary used during compilation.
    pub fn new(compiled: &CompiledGrammar, vocab_size: usize) -> Result<Self, GrammarError> {
        let matcher = GrammarMatcher::new(
            compiled, None,  // use stop tokens from compiled grammar
            false, // require stop token for proper termination
            -1,    // unlimited rollback
        )
        .map_err(GrammarError::Compilation)?;

        let bitmask_data = allocate_token_bitmask(1, vocab_size);
        let (_, bitmask_cols) = get_bitmask_shape(1, vocab_size);
        let bitmask_shape = [1i64, bitmask_cols as i64];
        let bitmask_strides = [bitmask_cols as i64, 1i64];

        Ok(Self {
            matcher,
            bitmask_data,
            bitmask_shape,
            bitmask_strides,
            vocab_size,
        })
    }

    /// Fill the allowed-token bitmask for the next decode step.
    ///
    /// Returns `true` if the bitmask constrains at least one token (i.e., is not
    /// all-ones). When `false`, the caller can skip bitmask application.
    ///
    /// Optimized for structural-tag grammars: in preamble state (before trigger),
    /// fill_bitmask() is called every 4 tokens instead of every token, saving
    /// fill_bitmask MUST be called every token to keep the xgrammar NPDA
    /// stacks synchronized with accept_token(). Skipping calls desynchronizes
    /// the FSM and causes fill_next_token_bitmask to hang (~47 tokens in).
    pub fn fill_bitmask(&mut self) -> bool {
        // Guard: calling fill_next_token_bitmask after the matcher has accepted
        // its stop token throws xgrammar::LogFatalError, which std::terminate()s
        // the whole process. Return false so callers skip bitmask application —
        // the grammar is already satisfied and imposes no further constraint.
        if self.matcher.is_terminated() {
            return false;
        }
        reset_token_bitmask(&mut self.bitmask_data);
        let mut tensor = self.make_bitmask_dltensor();
        self.matcher.fill_next_token_bitmask(&mut tensor, 0, false)
    }

    /// Raw bitmask data: `ceil(vocab_size / 32)` i32 words.
    ///
    /// Bit `token_id` is at `data[token_id / 32] & (1 << (token_id % 32))`.
    /// A set bit means the token is allowed.
    pub fn bitmask_data(&self) -> &[i32] {
        &self.bitmask_data
    }

    /// Check if a specific token is allowed by the current bitmask.
    pub fn is_token_allowed(&self, token_id: u32) -> bool {
        let word = (token_id / 32) as usize;
        let bit = token_id % 32;
        if word >= self.bitmask_data.len() {
            return false;
        }
        (self.bitmask_data[word] & (1i32 << bit)) != 0
    }

    /// Accept a sampled token and advance the grammar state.
    ///
    /// Returns `true` if the token was accepted by the grammar.
    /// Returns `false` if the token violates the grammar (should not happen
    /// if the bitmask was applied correctly).
    pub fn accept_token(&mut self, token_id: u32) -> bool {
        self.matcher.accept_token(token_id as i32)
    }

    /// Whether the grammar has been fully matched (all required structure generated).
    pub fn is_terminated(&self) -> bool {
        self.matcher.is_terminated()
    }

    /// Rollback the grammar state by `n` tokens.
    ///
    /// Used for MTP speculative decode: when draft tokens are rejected,
    /// the grammar state must be rewound to match.
    pub fn rollback(&mut self, n: usize) {
        self.matcher.rollback(n as i32);
    }

    /// Reset the grammar state to the initial position.
    pub fn reset(&mut self) {
        self.matcher.reset();
    }

    /// Apply the current bitmask to a slice of f32 logits in-place.
    ///
    /// Masked tokens (disallowed by grammar) are set to `f32::NEG_INFINITY`.
    /// This is the CPU-side application; for GPU-side, a CUDA kernel would
    /// be needed (future optimization).
    pub fn apply_bitmask_to_logits(&self, logits: &mut [f32]) {
        let n = logits.len().min(self.vocab_size);
        for token_id in 0..n {
            let word = token_id / 32;
            let bit = token_id % 32;
            if word < self.bitmask_data.len() && (self.bitmask_data[word] & (1i32 << bit)) == 0 {
                logits[token_id] = f32::NEG_INFINITY;
            }
        }
    }

    /// Construct a [`DLTensor`] pointing at the internal bitmask buffer.
    ///
    /// The tensor is valid only while `self` is alive and the bitmask data
    /// is not reallocated (it never is — size is fixed at construction).
    fn make_bitmask_dltensor(&mut self) -> DLTensor {
        DLTensor {
            data: self.bitmask_data.as_mut_ptr() as *mut std::ffi::c_void,
            device: DLDevice {
                device_type: DLDeviceType::kDLCPU,
                device_id: 0,
            },
            ndim: 2,
            dtype: DLDataType {
                code: DLDataTypeCode::kDLInt as u8,
                bits: 32,
                lanes: 1,
            },
            shape: self.bitmask_shape.as_mut_ptr(),
            strides: self.bitmask_strides.as_mut_ptr(),
            byte_offset: 0,
        }
    }
}

// ── Vocabulary extraction helper ───────────────────────────────────────

// F72 helpers (`decoded_vocab_bytes`, `compute_trigger_breakers`)
// were removed in F73 / fix42. The byte-level partial-trigger anchor
// at the sampler level hung the server in production despite passing
// isolated tests. The xgrammar non-anchored TagDispatch limitation is
// now handled at the streaming-sanitizer + parser layer (envelope_open
// markers in `LeakMarkers`, plus `<minimax:_call>` → `<tool_call>`
// normalisation in `parse_tool_calls`).
