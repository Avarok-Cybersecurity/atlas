// SPDX-License-Identifier: AGPL-3.0-only

//! MRoPE (T, H, W) position streams for a prefill chunk.
//!
//! Pure arithmetic, extracted from `upload_meta` so it can be tested: the
//! caller needs a GPU, a KV cache and a pinned staging allocation, none of
//! which the position rule depends on. This decides where every vision token
//! sits in all three rotary streams, and a mistake here is invisible — the
//! model produces fluent, confidently wrong output rather than failing.

/// Append the (T, H, W) streams for `chunk_tokens` to the three output
/// vectors, starting the running position at `start_pos`.
///
/// Returns the running position AFTER the walk — the index the next token
/// would take. The caller needs it because a vision item consumes far more
/// TOKENS than it consumes POSITIONS, so the rotary stream and the token
/// index diverge permanently at the first image and everything afterwards
/// (later chunks, and every decode step) has to resume from this value
/// rather than from its own token index.
///
/// Matches HF Qwen3-VL's `get_rope_index` / `get_vision_position_ids`:
///
/// - a TEXT token takes `T = H = W = pos` and advances `pos` by one;
/// - a VISION item of `t_len` temporal groups over a post-merge `gh × gw`
///   grid occupies `t_len * gh * gw` consecutive pad tokens, where token `k`
///   of group `g` takes `T = base + g`, `H = base + row`, `W = base + col`,
///   and afterwards `pos` advances by `max(t_len, gh, gw)`.
///
/// An IMAGE is the `t_len = 1` case and reduces exactly to the image-only
/// rule that preceded this: `base + g` collapses to `base`, and the advance
/// to `max(gh, gw)`.
///
/// Both pad tokens are recognized. They are consumed identically — the item's
/// own `t_len` already says whether it is a still or a clip — but a video run
/// whose token went unrecognized would be walked one text token at a time,
/// handing each of its thousands of pad positions a distinct index and
/// shifting every token after it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build(
    chunk_tokens: &[u32],
    grids: &[(usize, usize, usize)],
    grid_base: usize,
    grid_hi: usize,
    start_pos: u32,
    image_pad: u32,
    video_pad: u32,
    t_out: &mut Vec<u32>,
    h_out: &mut Vec<u32>,
    w_out: &mut Vec<u32>,
) -> u32 {
    walk(
        chunk_tokens,
        grids,
        grid_base,
        grid_hi,
        start_pos,
        image_pad,
        video_pad,
        Some((t_out, h_out, w_out)),
    )
    .0
}

/// The same walk as [`build`], but emitting nothing — used to DERIVE the state
/// at an arbitrary point in the token stream.
///
/// Returns `(pos, item, pad_rows)`: the rotary position the next token would
/// take, the index of the next unconsumed vision item, and how many pad tokens
/// (equivalently, encoder rows) were consumed.
///
/// This exists because a prefix-cache hit narrows a prefill pass to a suffix,
/// and EVERYTHING that phase 1 derived by walking from the start of the prompt
/// then has to be re-derived for the narrowed range: the rotary anchor, which
/// vision item comes next, and which encoder row the splice should start from.
/// Deriving it here rather than trusting carried state matters — on a warm
/// chunk-0 vision prefill `seq.mrope_delta` is 0 because the SequenceState is
/// fresh, so the carried value silently degenerates to the token index and the
/// three streams describe different tokens than the rows being computed.
///
/// One function owns the rule so the anchor and the splice seed cannot drift
/// apart; hand-rolling a second pad counter anywhere is how that happens.
#[allow(clippy::too_many_arguments)]
pub(crate) fn advance(
    tokens: &[u32],
    grids: &[(usize, usize, usize)],
    grid_base: usize,
    grid_hi: usize,
    start_pos: u32,
    image_pad: u32,
    video_pad: u32,
) -> (u32, usize, usize) {
    walk(
        tokens, grids, grid_base, grid_hi, start_pos, image_pad, video_pad, None,
    )
}

/// Encoder rows consumed by `tokens` — one per vision pad token.
///
/// The splice indexes the encoder's packed output, which is ordered over the
/// WHOLE prompt, so a narrowed range must seed its row index with this rather
/// than restarting at 0.
pub(crate) fn pad_rows_before(tokens: &[u32], image_pad: u32, video_pad: u32) -> usize {
    tokens
        .iter()
        .filter(|&&t| t == image_pad || t == video_pad)
        .count()
}

/// Shared body. `out` present => append the three streams; absent => derive only.
#[allow(clippy::too_many_arguments)]
fn walk(
    chunk_tokens: &[u32],
    grids: &[(usize, usize, usize)],
    grid_base: usize,
    grid_hi: usize,
    start_pos: u32,
    image_pad: u32,
    video_pad: u32,
    mut out: Option<(&mut Vec<u32>, &mut Vec<u32>, &mut Vec<u32>)>,
) -> (u32, usize, usize) {
    let is_pad = |tok: u32| tok == image_pad || tok == video_pad;
    let mut pad_rows = 0usize;
    let mut pos = start_pos;
    let mut item = grid_base;
    let mut i = 0usize;
    while i < chunk_tokens.len() {
        if is_pad(chunk_tokens[i]) && item < grid_hi {
            let (t_len, gh, gw) = grids[item];
            let t_len = t_len.max(1);
            let plane = (gh * gw).max(1);
            let run_len = t_len * plane;
            let base = pos;
            for k in 0..run_len {
                // [group, row, col] order — the order the encoder emitted the
                // groups, and therefore the order the merged rows are spliced.
                let g = (k / plane) as u32;
                let within = k % plane;
                let row = (within / gw.max(1)) as u32;
                let col = (within % gw.max(1)) as u32;
                if let Some((t_out, h_out, w_out)) = out.as_mut() {
                    t_out.push(base + g);
                    h_out.push(base + row);
                    w_out.push(base + col);
                }
            }
            pad_rows += run_len;
            // The item's extent on EVERY axis, so the next text token starts
            // clear of all three streams. A long clip can exceed its own
            // spatial extent, which is why t_len joins the max rather than
            // the spatial pair being assumed to dominate.
            pos += t_len.max(gh).max(gw) as u32;
            i += run_len;
            item += 1;
        } else {
            if let Some((t_out, h_out, w_out)) = out.as_mut() {
                t_out.push(pos);
                h_out.push(pos);
                w_out.push(pos);
            }
            pos += 1;
            i += 1;
        }
    }
    (pos, item, pad_rows)
}

#[cfg(test)]
#[path = "mrope_pos_tests.rs"]
mod tests;
