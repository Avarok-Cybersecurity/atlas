// SPDX-License-Identifier: AGPL-3.0-only

//! Pure log-softmax / top-k logprob extraction. SSOT for the math shared
//! by the scheduler's decode-time logprobs (`spark-server`) and the
//! model's prompt-logprob collection during prefill (legacy
//! `/v1/completions` `echo` + `logprobs`).

/// One scored prompt position: `log P(tokens[i+1] | tokens[..=i])` plus
/// the top-k alternative tokens under the same distribution.
#[derive(Clone, Debug)]
pub struct PromptTokenLogprob {
    /// The actual next prompt token (the one being scored).
    pub token_id: u32,
    /// Log-probability of `token_id` under this position's distribution.
    pub logprob: f32,
    /// Top-k `(token_id, logprob)` alternatives, sorted descending.
    /// Empty when k=0 (logprob-only mode).
    pub top: Vec<(u32, f32)>,
}

/// Log-softmax over `f32_logits`; returns the target token's logprob and
/// the top-k alternatives sorted descending by logprob. `k=0` returns an
/// empty top vector (chosen-token logprob only). A target outside the
/// vocab yields `-inf` (fail-visible, never panics).
pub fn logprob_of(f32_logits: &[f32], target: u32, k: usize) -> (f32, Vec<(u32, f32)>) {
    // Log-softmax: logprob = logit - log(sum(exp(logits)))
    let max_logit = f32_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp = max_logit
        + f32_logits
            .iter()
            .map(|&l| (l - max_logit).exp())
            .sum::<f32>()
            .ln();
    let target_logprob = if (target as usize) < f32_logits.len() {
        f32_logits[target as usize] - log_sum_exp
    } else {
        f32::NEG_INFINITY
    };
    if k == 0 {
        return (target_logprob, Vec::new());
    }
    // Top-K by partial sort.
    let mut indexed: Vec<(u32, f32)> = f32_logits
        .iter()
        .enumerate()
        .map(|(j, &l)| (j as u32, l - log_sum_exp))
        .collect();
    let nth = k.min(indexed.len().saturating_sub(1));
    indexed.select_nth_unstable_by(nth, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut top: Vec<(u32, f32)> = indexed[..k.min(indexed.len())].to_vec();
    top.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    (target_logprob, top)
}

/// BF16 → FP32 (upper 16 bits of the IEEE-754 f32 pattern).
#[inline]
pub fn bf16_to_f32(lo: u8, hi: u8) -> f32 {
    f32::from_bits(((lo as u32) | ((hi as u32) << 8)) << 16)
}

/// Extract one position's `PromptTokenLogprob` from a BF16 `[vocab]`
/// logits slice (the layout `lm_head_batched` writes per row).
pub fn extract_bf16(bf16: &[u8], target: u32, k: usize, vocab: usize) -> PromptTokenLogprob {
    let f32_logits: Vec<f32> = (0..vocab)
        .map(|j| bf16_to_f32(bf16[j * 2], bf16[j * 2 + 1]))
        .collect();
    let (logprob, top) = logprob_of(&f32_logits, target, k);
    PromptTokenLogprob {
        token_id: target,
        logprob,
        top,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logprob_matches_manual_log_softmax() {
        let logits = [1.0f32, 2.0, 3.0];
        let sum: f32 = logits.iter().map(|l| l.exp()).sum();
        let expect = 2.0 - sum.ln();
        let (lp, top) = logprob_of(&logits, 1, 2);
        assert!((lp - expect).abs() < 1e-6, "{lp} vs {expect}");
        // Top-2 sorted descending: token 2 first, then token 1.
        assert_eq!(top[0].0, 2);
        assert_eq!(top[1].0, 1);
        assert!(top[0].1 > top[1].1);
    }

    #[test]
    fn k_zero_returns_empty_top() {
        let (lp, top) = logprob_of(&[0.0, 1.0], 0, 0);
        assert!(top.is_empty());
        assert!(lp < 0.0);
    }

    #[test]
    fn out_of_vocab_target_is_neg_inf_not_panic() {
        let (lp, _) = logprob_of(&[0.0, 1.0], 99, 1);
        assert_eq!(lp, f32::NEG_INFINITY);
    }

    #[test]
    fn bf16_slice_roundtrip_extract() {
        // BF16 encodings of [0.0, 1.0, 2.0]: f32 bit patterns >> 16.
        let vals = [0.0f32, 1.0, 2.0];
        let mut bytes = Vec::new();
        for v in vals {
            let b = (v.to_bits() >> 16) as u16;
            bytes.push((b & 0xFF) as u8);
            bytes.push((b >> 8) as u8);
        }
        let r = extract_bf16(&bytes, 2, 1, 3);
        let sum: f32 = vals.iter().map(|l| l.exp()).sum();
        assert!((r.logprob - (2.0 - sum.ln())).abs() < 1e-3);
        assert_eq!(r.top[0].0, 2);
    }
}
