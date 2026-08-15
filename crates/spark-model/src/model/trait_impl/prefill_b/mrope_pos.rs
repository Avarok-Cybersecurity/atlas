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
) {
    let is_pad = |tok: u32| tok == image_pad || tok == video_pad;
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
                t_out.push(base + g);
                h_out.push(base + row);
                w_out.push(base + col);
            }
            // The item's extent on EVERY axis, so the next text token starts
            // clear of all three streams. A long clip can exceed its own
            // spatial extent, which is why t_len joins the max rather than
            // the spatial pair being assumed to dominate.
            pos += t_len.max(gh).max(gw) as u32;
            i += run_len;
            item += 1;
        } else {
            t_out.push(pos);
            h_out.push(pos);
            w_out.push(pos);
            pos += 1;
            i += 1;
        }
    }
}

#[cfg(test)]
#[path = "mrope_pos_tests.rs"]
mod tests;
