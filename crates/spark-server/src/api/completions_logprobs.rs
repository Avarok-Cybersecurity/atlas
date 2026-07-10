// SPDX-License-Identifier: AGPL-3.0-only

//! Legacy `/v1/completions` logprobs assembly: the OpenAI four-parallel-
//! array shape (`tokens` / `token_logprobs` / `top_logprobs` /
//! `text_offset`), echo merging, and the null-first-token convention.
//!
//! `text_offset` is the cumulative character length of the per-token
//! decoded strings — the same approximation vLLM uses. Byte-level BPE
//! per-token decodes may not concatenate byte-for-byte to the single
//! full-sequence decode shown in `text` (multibyte splits, leading-space
//! pieces), so offsets can drift a few chars near such boundaries;
//! reference implementations accept this.

use crate::api::inference_types::TokenLogprobs;
use crate::openai::CompletionLogprobs;

/// Assemble the legacy logprobs block.
///
/// * `decode` — per-token-id detokenizer (injected so unit tests need no
///   real tokenizer; production passes `state.tokenizer.decode(&[id])`).
/// * `prompt_tokens`/`prompt_lps` — when echoing: `prompt_lps[i]` scores
///   `prompt_tokens[i+1]` (the model-side collection excludes both the
///   first token, which has no context, and the position that scores the
///   first *generated* token). The first echoed token gets `null`.
/// * `gen_tokens`/`gen_lps` — generated tokens and their decode-time
///   logprobs. If the first generated token's logprobs were not
///   captured (`gen_lps.len() == gen_tokens.len() - 1`), it is padded
///   with `null` rather than misaligning the parallel arrays.
pub(super) fn build_completion_logprobs(
    decode: &dyn Fn(u32) -> String,
    echo: bool,
    prompt_tokens: &[u32],
    prompt_lps: &[TokenLogprobs],
    gen_tokens: &[u32],
    gen_lps: &[TokenLogprobs],
) -> CompletionLogprobs {
    let cap = if echo { prompt_tokens.len() } else { 0 } + gen_tokens.len();
    let mut tokens: Vec<String> = Vec::with_capacity(cap);
    let mut token_logprobs: Vec<Option<f32>> = Vec::with_capacity(cap);
    let mut top_logprobs = Vec::with_capacity(cap);
    let mut text_offset: Vec<usize> = Vec::with_capacity(cap);
    let mut offset = 0usize;

    let mut push = |tok_id: u32, lp: Option<&TokenLogprobs>| {
        let piece = decode(tok_id);
        text_offset.push(offset);
        offset += piece.len();
        tokens.push(piece);
        token_logprobs.push(lp.map(|l| l.logprob));
        top_logprobs.push(lp.map(|l| {
            l.top
                .iter()
                .map(|&(id, p)| (decode(id), p))
                .collect::<std::collections::HashMap<String, f32>>()
        }));
    };

    if echo {
        for (i, &tok) in prompt_tokens.iter().enumerate() {
            // Position i is scored by prompt_lps[i-1]; token 0 is null.
            push(tok, i.checked_sub(1).and_then(|j| prompt_lps.get(j)));
        }
    }
    // First-generated-token pad: see doc comment.
    let gen_pad = gen_tokens.len().saturating_sub(gen_lps.len());
    for (i, &tok) in gen_tokens.iter().enumerate() {
        push(tok, i.checked_sub(gen_pad).and_then(|j| gen_lps.get(j)));
    }

    CompletionLogprobs {
        tokens,
        token_logprobs,
        top_logprobs,
        text_offset,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lp(token_id: u32, logprob: f32) -> TokenLogprobs {
        TokenLogprobs {
            token_id,
            logprob,
            top: vec![(token_id, logprob)],
        }
    }

    // Test decoder: token id N → "tN " (3+ chars, deterministic offsets).
    fn dec(id: u32) -> String {
        format!("t{id} ")
    }

    #[test]
    fn echo_null_first_then_prompt_scores() {
        // prompt [10, 11, 12]; collection scored positions 0,1 → targets 11,12.
        let out = build_completion_logprobs(
            &dec,
            true,
            &[10, 11, 12],
            &[lp(11, -0.1), lp(12, -0.2)],
            &[],
            &[],
        );
        assert_eq!(out.tokens, vec!["t10 ", "t11 ", "t12 "]);
        assert_eq!(out.token_logprobs, vec![None, Some(-0.1), Some(-0.2)]);
        assert!(out.top_logprobs[0].is_none());
        assert!(out.top_logprobs[1].as_ref().unwrap().contains_key("t11 "));
    }

    #[test]
    fn text_offset_is_cumulative_decoded_length() {
        let out =
            build_completion_logprobs(&dec, true, &[7, 8], &[lp(8, -0.3)], &[9], &[lp(9, -0.4)]);
        // "t7 "(3) "t8 "(3) "t9 "(3)
        assert_eq!(out.text_offset, vec![0, 3, 6]);
    }

    #[test]
    fn echo_concatenates_prompt_then_generated() {
        let out = build_completion_logprobs(
            &dec,
            true,
            &[1, 2],
            &[lp(2, -0.1)],
            &[3, 4],
            &[lp(3, -0.2), lp(4, -0.3)],
        );
        assert_eq!(out.tokens.len(), 4);
        assert_eq!(
            out.token_logprobs,
            vec![None, Some(-0.1), Some(-0.2), Some(-0.3)]
        );
    }

    #[test]
    fn no_echo_generated_only_with_first_token_pad() {
        // 3 generated tokens but only 2 logprob entries (first-token
        // logprobs not captured at prefill) → null-padded front.
        let out = build_completion_logprobs(
            &dec,
            false,
            &[1, 2, 3],
            &[],
            &[5, 6, 7],
            &[lp(6, -0.1), lp(7, -0.2)],
        );
        assert_eq!(out.tokens.len(), 3);
        assert_eq!(out.token_logprobs, vec![None, Some(-0.1), Some(-0.2)]);
    }

    #[test]
    fn scoring_only_prompt_len_entries() {
        // max_tokens=0: entries == prompt_len, first null, none missing.
        let prompt = [10u32, 11, 12, 13];
        let lps = [lp(11, -1.0), lp(12, -2.0), lp(13, -3.0)];
        let out = build_completion_logprobs(&dec, true, &prompt, &lps, &[], &[]);
        assert_eq!(out.tokens.len(), prompt.len());
        assert_eq!(out.token_logprobs[0], None);
        assert!(out.token_logprobs[1..].iter().all(Option::is_some));
    }
}
