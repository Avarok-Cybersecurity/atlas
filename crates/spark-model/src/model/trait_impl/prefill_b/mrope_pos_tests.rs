// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the MRoPE (T, H, W) position rule.
//!
//! These are the assertions that were impossible before the loop was pulled
//! out of `upload_meta`: it needed a GPU, a KV cache and a pinned staging
//! buffer to reach. A position bug is silent — the model stays fluent and is
//! simply wrong — so the arithmetic is pinned here directly.

use super::*;

const IMG: u32 = 248_056;
const VID: u32 = 248_057;
const TXT: u32 = 7;

fn run(
    tokens: &[u32],
    grids: &[(usize, usize, usize)],
    start: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let (mut t, mut h, mut w) = (Vec::new(), Vec::new(), Vec::new());
    build(
        tokens,
        grids,
        0,
        grids.len(),
        start,
        IMG,
        VID,
        &mut t,
        &mut h,
        &mut w,
    );
    (t, h, w)
}

// ── text ─────────────────────────────────────────────────────────────────

#[test]
fn text_only_is_the_identity_ramp_on_all_three_streams() {
    let (t, h, w) = run(&[TXT; 5], &[], 0);
    assert_eq!(t, vec![0, 1, 2, 3, 4]);
    assert_eq!(h, t);
    assert_eq!(w, t);
}

#[test]
fn the_streams_start_where_the_caller_says() {
    let (t, h, w) = run(&[TXT; 3], &[], 100);
    assert_eq!(t, vec![100, 101, 102]);
    assert_eq!(h, t);
    assert_eq!(w, t);
}

// ── images: must be byte-identical to the pre-video rule ─────────────────

/// A 2×3 image: T constant across the whole run, H by row, W by column, then
/// the running position advances by max(gh, gw) = 3.
#[test]
fn an_image_holds_t_constant_and_advances_by_its_long_side() {
    let tokens = [TXT, IMG, IMG, IMG, IMG, IMG, IMG, TXT];
    let (t, h, w) = run(&tokens, &[(1, 2, 3)], 0);

    // text(0) | image run based at 1 | text after
    assert_eq!(t, vec![0, 1, 1, 1, 1, 1, 1, 4]);
    assert_eq!(h, vec![0, 1, 1, 1, 2, 2, 2, 4]);
    assert_eq!(w, vec![0, 1, 2, 3, 1, 2, 3, 4]);
    // The trailing text token is at 1 + max(2, 3) = 4.
    assert_eq!(*t.last().unwrap(), 4);
}

/// The independent statement of the same guarantee: an image is the t_len = 1
/// case, so introducing the temporal axis must not have moved a single
/// position. Computed here from the OLD rule directly rather than from a
/// remembered constant.
#[test]
fn the_image_rule_is_unchanged_by_the_temporal_axis() {
    fn old_rule(tokens: &[u32], grids: &[(usize, usize)]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let (mut t, mut h, mut w) = (Vec::new(), Vec::new(), Vec::new());
        let (mut pos, mut idx, mut i) = (0u32, 0usize, 0usize);
        while i < tokens.len() {
            if tokens[i] == IMG && idx < grids.len() {
                let (gh, gw) = grids[idx];
                let run = gh * gw;
                let base = pos;
                for k in 0..run {
                    t.push(base);
                    h.push(base + (k / gw.max(1)) as u32);
                    w.push(base + (k % gw.max(1)) as u32);
                }
                pos += gh.max(gw) as u32;
                i += run;
                idx += 1;
            } else {
                t.push(pos);
                h.push(pos);
                w.push(pos);
                pos += 1;
                i += 1;
            }
        }
        (t, h, w)
    }

    for (gh, gw) in [(1usize, 1usize), (2, 3), (3, 2), (4, 4), (1, 7), (7, 1)] {
        let mut tokens = vec![TXT, TXT];
        tokens.extend(std::iter::repeat_n(IMG, gh * gw));
        tokens.extend([TXT, TXT]);
        assert_eq!(
            run(&tokens, &[(1, gh, gw)], 0),
            old_rule(&tokens, &[(gh, gw)]),
            "the {gh}x{gw} image moved when the temporal axis was added"
        );
    }
}

#[test]
fn two_images_each_advance_the_running_position() {
    let tokens = [IMG, IMG, IMG, IMG, IMG, TXT];
    // 1x2 at base 0 (advances 2), then 1x3 at base 2 (advances 3) -> text at 5.
    let (t, h, w) = run(&tokens, &[(1, 1, 2), (1, 1, 3)], 0);
    assert_eq!(t, vec![0, 0, 2, 2, 2, 5]);
    assert_eq!(h, vec![0, 0, 2, 2, 2, 5]);
    assert_eq!(w, vec![0, 1, 2, 3, 4, 5]);
}

// ── video ────────────────────────────────────────────────────────────────

/// ★ The rule video exists for. Three temporal groups over a 2×2 grid: T
/// advances once per GROUP while H and W restart each group, and the item
/// advances the running position by max(3, 2, 2) = 3.
#[test]
fn a_video_advances_t_once_per_temporal_group() {
    let tokens = [
        TXT, VID, VID, VID, VID, VID, VID, VID, VID, VID, VID, VID, VID, TXT,
    ];
    let (t, h, w) = run(&tokens, &[(3, 2, 2)], 0);

    // base = 1. Group g gets T = 1 + g, four tokens each.
    assert_eq!(
        t,
        vec![0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4],
        "T must step per group"
    );
    // H and W repeat their 2x2 pattern in every group.
    assert_eq!(h, vec![0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 4]);
    assert_eq!(w, vec![0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 4]);
    // Trailing text at 1 + max(3, 2, 2) = 4.
    assert_eq!(*t.last().unwrap(), 4);
}

/// A clip longer than it is wide must advance by its TEMPORAL extent. If the
/// advance assumed the spatial pair dominates, a long video would overlap the
/// text that follows it — every subsequent token sharing positions with the
/// tail of the clip.
#[test]
fn a_long_clip_advances_by_its_temporal_extent() {
    let t_len = 10usize;
    let mut tokens = vec![TXT];
    tokens.extend(std::iter::repeat_n(VID, t_len)); // 1x1 grid
    tokens.push(TXT);
    let (t, _, _) = run(&tokens, &[(t_len, 1, 1)], 0);
    assert_eq!(t[0], 0);
    assert_eq!(&t[1..=t_len], &(1..=t_len as u32).collect::<Vec<_>>()[..]);
    assert_eq!(
        *t.last().unwrap(),
        1 + t_len as u32,
        "the clip's temporal extent did not advance the running position"
    );
}

/// One video and one image in the same request, which is the case where a
/// mis-scan shows up as everything after the video being shifted.
#[test]
fn a_video_and_an_image_interleave_correctly() {
    let mut tokens = vec![TXT];
    tokens.extend(std::iter::repeat_n(VID, 2)); // 2 groups over a 1x1 grid
    tokens.push(TXT);
    tokens.extend(std::iter::repeat_n(IMG, 2 * 2)); // 2x2 still
    tokens.push(TXT);
    let (t, _, _) = run(&tokens, &[(2, 1, 1), (1, 2, 2)], 0);
    //  text 0
    //  video base 1, groups at 1 and 2, advance max(2,1,1)=2 -> 3
    //  text 3
    //  image base 4, T=4 x4, advance max(2,2)=2 -> 6
    //  text 6
    assert_eq!(t, vec![0, 1, 2, 3, 4, 4, 4, 4, 6]);
}

/// A single-group video is positionally identical to an image of the same
/// grid. It should be: one temporal group IS a still.
#[test]
fn a_one_group_video_matches_the_equivalent_image() {
    let img_tokens = [TXT, IMG, IMG, IMG, IMG, TXT];
    let vid_tokens = [TXT, VID, VID, VID, VID, TXT];
    assert_eq!(
        run(&img_tokens, &[(1, 2, 2)], 0),
        run(&vid_tokens, &[(1, 2, 2)], 0)
    );
}

// ── bounds and degenerate input ──────────────────────────────────────────

/// Co-dispatch: a request owns `grids[base..hi]` of a shared vector. Pad runs
/// past `hi` are not this request's to consume.
#[test]
fn the_grid_window_bounds_which_items_are_consumed() {
    let tokens = [IMG, IMG, IMG, IMG];
    let grids = [(1, 1, 4), (1, 2, 2), (1, 4, 1)];
    let (mut t, mut h, mut w) = (Vec::new(), Vec::new(), Vec::new());
    // Own only grids[1..2] — the second 2x2.
    build(&tokens, &grids, 1, 2, 0, IMG, VID, &mut t, &mut h, &mut w);
    assert_eq!(t, vec![0, 0, 0, 0], "consumed its one owned item");
    assert_eq!(h, vec![0, 0, 1, 1]);
    assert_eq!(w, vec![0, 1, 0, 1]);
}

/// Pad tokens with no grid left to describe them fall through to the text
/// rule rather than indexing past the end.
#[test]
fn pads_beyond_the_owned_grids_do_not_panic() {
    let tokens = [IMG, IMG, IMG, IMG, IMG];
    let (t, _, _) = run(&tokens, &[(1, 2, 2)], 0);
    assert_eq!(t.len(), 5, "one token in, one position out");
    assert_eq!(t, vec![0, 0, 0, 0, 2]);
}

/// Every input length must produce exactly that many positions on all three
/// streams. A stream shorter than the chunk is a buffer overrun waiting to
/// happen at the pack step.
#[test]
fn the_three_streams_always_match_the_token_count() {
    for grids in [
        vec![],
        vec![(1usize, 2usize, 2usize)],
        vec![(3, 2, 2)],
        vec![(2, 1, 1), (1, 3, 3)],
    ] {
        let total: usize = grids.iter().map(|(t, h, w)| t * h * w).sum();
        let mut tokens = vec![TXT, TXT];
        tokens.extend(std::iter::repeat_n(IMG, total));
        tokens.push(TXT);
        let (t, h, w) = run(&tokens, &grids, 0);
        assert_eq!(t.len(), tokens.len(), "T stream length, grids={grids:?}");
        assert_eq!(h.len(), tokens.len(), "H stream length, grids={grids:?}");
        assert_eq!(w.len(), tokens.len(), "W stream length, grids={grids:?}");
    }
}

/// A zero in a grid must not divide by zero or loop forever.
#[test]
fn a_degenerate_grid_is_survivable() {
    let (t, h, w) = run(&[IMG, TXT], &[(0, 0, 0)], 7);
    assert_eq!(t, vec![7, 8]);
    assert_eq!(h, t);
    assert_eq!(w, t);
}
