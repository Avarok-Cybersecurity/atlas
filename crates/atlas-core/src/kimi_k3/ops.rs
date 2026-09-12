// SPDX-License-Identifier: AGPL-3.0-only

//! Tiny host GEMV / gather helpers for the C1 CPU graph.

/// `y = W x` with `W` row-major `[out, inn]`.
pub fn matvec(w: &[f32], x: &[f32], out: usize, inn: usize) -> Vec<f32> {
    assert_eq!(
        w.len(),
        out * inn,
        "matvec weight {} vs {out}x{inn}",
        w.len()
    );
    assert_eq!(x.len(), inn);
    let mut y = vec![0.0f32; out];
    for o in 0..out {
        let row = &w[o * inn..(o + 1) * inn];
        let mut acc = 0.0f32;
        for i in 0..inn {
            acc += row[i] * x[i];
        }
        y[o] = acc;
    }
    y
}

/// Token embedding gather: `embed[token, :]`.
pub fn embed_token(table: &[f32], token: u32, hidden: usize, vocab: usize) -> Vec<f32> {
    let t = token as usize;
    assert!(t < vocab, "token {token} >= vocab {vocab}");
    table[t * hidden..(t + 1) * hidden].to_vec()
}

/// Greedy next-token: argmax, lower id on ties.
pub fn argmax(logits: &[f32]) -> u32 {
    assert!(!logits.is_empty());
    let mut best_i = 0usize;
    let mut best = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best {
            best = v;
            best_i = i;
        }
    }
    best_i as u32
}

/// Deterministic fill in `(-scale/2, scale/2)`.
pub fn fill(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = seed
                .wrapping_mul(1664525)
                .wrapping_add(1013904223)
                .wrapping_add((i as u32).wrapping_mul(2654435761));
            ((h % 1000) as f32 / 1000.0 - 0.5) * scale
        })
        .collect()
}

/// Ones vector (RMSNorm / identity-ish scales).
pub fn ones(n: usize) -> Vec<f32> {
    vec![1.0; n]
}

/// Row-major identity padded to `[out, inn]`.
pub fn ident(out: usize, inn: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; out * inn];
    for i in 0..out.min(inn) {
        w[i * inn + i] = 1.0;
    }
    w
}
