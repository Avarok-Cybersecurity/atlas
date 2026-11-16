// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: tests that `AVAROK_NO_FAST_GREEDY_CHAT=1` really serves the host pipeline, so the
//! `mtp_device_accept` identity screen has a host pipeline to compare against (lane L12).
//!
//! The screen's off arm was `AVAROK_DENSE_FP8_FROM_BF16=1 AVAROK_NO_FAST_GREEDY_CHAT=1`; the on arm
//! added `AVAROK_MTP_DEVICE_ACCEPT=1`. They differed on 2 of 8 prompts at C=1, and the K4 trace of
//! both arms showed why: the off arm logged "verify chat fast path ACTIVE", the masked-verify
//! shortcut (`fast_masked`), which ignored the kill switch and answered with the verify graph's raw
//! argmax. At an exact BF16 top tie that argmax (a tree reduction) and the host pipeline (first
//! index wins) pick different ids, and so does the device path, which is the host pipeline on the
//! device. These tests drive the helper on exactly that row (greedy, no grammar, neutral penalties)
//! with the raw argmax set to the OTHER tied id, which is what the graphed argmax answered there.
//!
//! The model is a stub holding the rows, so the `[K, vocab]` D2H tail has something to read.

use super::device_accept::picks_via;
use super::device_accept_tests::{VOCAB, bf16_rows, rows, seq, with_ctx};
use super::pick_positions::pick_positions_from_host;
use super::verify_pick_all_via;
use crate::scheduler::logit_processors::SamplingLevers;
use crate::scheduler::test_support::PreemptStubModel;
use spark_model::port::spec_accept::reference;
use spark_runtime::gpu::DevicePtr;

/// The lower id of the exact top tie in each row: the host pipeline's and the device path's pick.
const FIRST_WINS: [u32; 4] = [5, 6, 7, 8];

/// The higher id of the tie: what the verify graph's raw argmax answered for the row.
const RAW_ARGMAX: [u32; 4] = [100, 101, 102, 103];

/// The levers of a default serve (the masked-verify lever defaults on, with or without DFlash).
fn default_serve() -> SamplingLevers {
    SamplingLevers {
        dflash_masked_verify: true,
        fast_masked: true,
        fast_greedy_chat: true,
        ..Default::default()
    }
}

/// `AVAROK_NO_FAST_GREEDY_CHAT=1` on a default serve.
fn no_fast_greedy_chat() -> SamplingLevers {
    SamplingLevers {
        fast_greedy_chat: false,
        ..default_serve()
    }
}

/// Four rows whose top two logits are EXACTLY equal (3.0 in BF16), the lower id in `lo`, the higher
/// in `hi`; everything else is on the 0.25 grid below 0.75.
fn tie_rows(lo: &[u32; 4], hi: &[u32; 4]) -> Vec<u8> {
    let hot: Vec<(usize, u32, f32)> = (0..4)
        .flat_map(|r| [(r, lo[r], 3.0), (r, hi[r], 3.0)])
        .collect();
    bf16_rows(&rows(4, 17, &hot))
}

/// Four rows with one clear winner each.
fn gap_rows(winners: &[u32; 4]) -> Vec<u8> {
    let hot: Vec<(usize, u32, f32)> = winners
        .iter()
        .enumerate()
        .map(|(r, &id)| (r, id, 3.0))
        .collect();
    bf16_rows(&rows(4, 17, &hot))
}

fn stub(buf: &[u8]) -> PreemptStubModel {
    PreemptStubModel {
        vocab: VOCAB,
        host_logits: buf.to_vec(),
        ..Default::default()
    }
}

/// What `verify_pick_all_via` answers with the row off (the device declines), for `levers`.
fn off_picks(levers: SamplingLevers, buf: &[u8], raw: &[u32; 4]) -> Vec<u32> {
    let model = stub(buf);
    let mut a = seq();
    with_ctx(levers, None, |c| {
        verify_pick_all_via(&model, raw, &mut a, c, 0, |_| None)
    })
}

/// The host slow tail over the same rows: the pipeline the device path claims to equal.
fn host_picks(buf: &[u8]) -> Vec<u32> {
    let mut a = seq();
    with_ctx(SamplingLevers::default(), None, |c| {
        pick_positions_from_host(buf, VOCAB, 2, 4, &mut a, c)
    })
}

#[test]
fn the_kill_switch_serves_the_host_pipeline_on_an_exact_top_tie() {
    let buf = tie_rows(&FIRST_WINS, &RAW_ARGMAX);
    let picks = off_picks(no_fast_greedy_chat(), &buf, &RAW_ARGMAX);
    assert_eq!(
        picks, FIRST_WINS,
        "AVAROK_NO_FAST_GREEDY_CHAT=1 must pick the lower id of an exact tie, as the host pipeline"
    );
    assert_ne!(picks, RAW_ARGMAX, "the raw argmax shortcut answered");
    assert_eq!(picks, host_picks(&buf));
}

#[test]
fn the_kill_switch_changes_nothing_where_the_top_logit_is_unique() {
    let buf = gap_rows(&FIRST_WINS);
    // The graphed argmax agrees with the pipeline wherever there is no tie.
    assert_eq!(
        off_picks(no_fast_greedy_chat(), &buf, &FIRST_WINS),
        FIRST_WINS
    );
    assert_eq!(off_picks(default_serve(), &buf, &FIRST_WINS), FIRST_WINS);
    assert_eq!(host_picks(&buf), FIRST_WINS);
}

#[test]
fn a_default_serve_still_takes_the_raw_argmax_shortcut() {
    // Unchanged by the kill-switch fix: with the shortcuts on, the verify graph's argmax answers,
    // ties included. Pinned so a change to the default is a decision, not a side effect.
    let buf = tie_rows(&FIRST_WINS, &RAW_ARGMAX);
    assert_eq!(off_picks(default_serve(), &buf, &RAW_ARGMAX), RAW_ARGMAX);
    // The chat shortcut alone (masked verify off) answers the same way.
    let chat_only = SamplingLevers {
        dflash_masked_verify: false,
        fast_masked: false,
        ..default_serve()
    };
    assert_eq!(off_picks(chat_only, &buf, &RAW_ARGMAX), RAW_ARGMAX);
}

#[test]
fn no_lever_combination_beats_the_kill_switch() {
    let buf = tie_rows(&FIRST_WINS, &RAW_ARGMAX);
    for dflash_masked_verify in [false, true] {
        for fast_masked in [false, true] {
            let levers = SamplingLevers {
                dflash_masked_verify,
                fast_masked,
                fast_greedy_chat: false,
                ..Default::default()
            };
            assert_eq!(
                off_picks(levers, &buf, &RAW_ARGMAX),
                FIRST_WINS,
                "dflash_masked_verify={dflash_masked_verify} fast_masked={fast_masked}"
            );
        }
    }
}

#[test]
fn the_device_path_equals_the_host_pipeline_on_an_exact_top_tie() {
    // The row's claim, on the row that broke the screen: with the row on, the device path picks the
    // host pipeline's id (the lower one) where the raw argmax shortcut picked the higher.
    let buf = tie_rows(&FIRST_WINS, &RAW_ARGMAX);
    let model = stub(&buf);
    let mut a = seq();
    let device = with_ctx(default_serve(), None, |c| {
        verify_pick_all_via(&model, &RAW_ARGMAX, &mut a, c, 0, |a| {
            picks_via(4, VOCAB, DevicePtr::NULL, a, c, |req| {
                Ok(Some(reference::run(&buf, req)))
            })
        })
    });
    assert_eq!(device, FIRST_WINS);
    assert_eq!(device, host_picks(&buf));
    assert_eq!(
        device,
        off_picks(no_fast_greedy_chat(), &buf, &RAW_ARGMAX),
        "device on equals the kill-switch serve it is screened against"
    );
    assert_ne!(device, RAW_ARGMAX);
}
