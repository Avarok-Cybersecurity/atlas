// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::scheduler::logit_processors::SamplingLevers;
use crate::scheduler::test_support::test_seq;

fn context(sched: &SchedCtx) -> LogitsContext<'_> {
    LogitsContext {
        scratch: &sched.scratch,
        dumps: &sched.dumps,
        stats: sched.stats.clone(),
        watchdog: crate::scheduler::helpers::WatchdogParams::default(),
        boundary_mask: None,
        mid_word_mask: None,
        sampling: SamplingLevers {
            mtp_verify_sample: true,
            mtp_minp: true,
            ..SamplingLevers::default()
        },
        timing: sched.timing.clone(),
        think_end_token: None,
        think_start_token: None,
        tool_call_start_token: None,
        tool_call_end_token: None,
    }
}

fn bytes(rows: &[[f32; 4]]) -> Vec<u8> {
    rows.iter()
        .flatten()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

#[test]
fn history_penalties_and_logprobs_follow_each_accepted_row() {
    for nd in [2, 3] {
        let sched = SchedCtx::for_test();
        let ctx = context(&sched);
        let (mut seq, _rx) = test_seq(Vec::new(), 100, None, 10);
        seq.finished = false;
        seq.presence_penalty = 4.0;
        seq.top_logprobs = Some(2);
        let rows = bytes(&[
            [0.0, 8.0, 0.0, 0.0],
            [0.0, 5.0, 4.0, 0.0],
            [0.0, 0.0, 5.0, 4.0],
            [5.0, 0.0, 0.0, 6.0],
        ]);
        let picks = sample_and_emit(&rows, 4, &[1, 2, 3][..nd], &mut seq, &sched, &ctx);
        assert_eq!(picks, [1, 2, 3, 0][..=nd]);
        assert_eq!(seq.output_tokens, picks);
        assert_eq!(seq.logprobs_data.len(), nd + 1);
        let expected =
            crate::scheduler::logprobs::extract_logprobs_from_f32(&[0.0, 1.0, 4.0, 0.0], 2, 2);
        assert_eq!(seq.logprobs_data[1].logprob, expected.logprob);
    }
}

#[test]
fn reject_at_every_position_does_not_process_suffix() {
    for nd in [2, 3] {
        for rejected in 0..nd {
            let sched = SchedCtx::for_test();
            let ctx = context(&sched);
            let (mut seq, _rx) = test_seq(Vec::new(), 100, None, 10);
            seq.finished = false;
            seq.inside_thinking = true;
            seq.force_end_thinking = true;
            let rows = bytes(&[[0.0, 10.0, 0.0, 0.0]; 4]);
            let mut drafts = vec![1; nd];
            drafts[rejected] = 2;
            let picks = sample_and_emit(&rows, 4, &drafts, &mut seq, &sched, &ctx);
            assert_eq!(picks, vec![1; rejected + 1]);
            assert_eq!(seq.sentence_defer_count as usize, rejected + 1);
        }
    }
}

#[test]
fn finish_at_every_row_does_not_process_suffix() {
    for nd in [2, 3] {
        for count in 1..=nd + 1 {
            let sched = SchedCtx::for_test();
            let ctx = context(&sched);
            let (mut seq, _rx) = test_seq(Vec::new(), count, None, 10);
            seq.finished = false;
            seq.inside_thinking = true;
            seq.force_end_thinking = true;
            let rows = bytes(&[[0.0, 10.0, 0.0, 0.0]; 4]);
            let picks = sample_and_emit(&rows, 4, &vec![1; nd], &mut seq, &sched, &ctx);
            assert_eq!(picks.len(), count);
            assert!(seq.finished);
            assert_eq!(seq.sentence_defer_count as usize, count);
        }
    }
}

#[test]
fn thinking_close_updates_next_row_tool_pin() {
    let sched = SchedCtx::for_test();
    let mut ctx = context(&sched);
    ctx.think_end_token = Some(1);
    ctx.tool_call_start_token = Some(3);
    let (mut seq, _rx) = test_seq(Vec::new(), 100, None, 10);
    seq.finished = false;
    seq.inside_thinking = true;
    seq.think_end_token = Some(1);
    seq.tool_call_start_token = Some(3);
    seq.thinking_tokens = 20;
    seq.require_tool_call = true;
    let rows = bytes(&[
        [0.0, 20.0, 0.0, 0.0],
        [0.0, 0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0, 0.0],
    ]);
    assert_eq!(
        sample_and_emit(&rows, 4, &[1, 3], &mut seq, &sched, &ctx),
        [1, 3, 2]
    );
    assert!(!seq.inside_thinking);
    assert!(seq.tool_call_opened);
}

#[test]
fn seed_and_ties_match_serial_sampling() {
    for temp in [0.0, 0.8] {
        let sched = SchedCtx::for_test();
        let ctx = context(&sched);
        let (mut seq, _rx) = test_seq(Vec::new(), 100, None, 10);
        let (mut serial, _serial_rx) = test_seq(Vec::new(), 100, None, 10);
        for s in [&mut seq, &mut serial] {
            s.finished = false;
            s.temperature = temp;
            s.seed = Some(42);
        }
        let rows = bytes(&[[0.0, 8.0, 8.0, 8.0]; 4]);
        let mut expected = Vec::new();
        for row in 0..4 {
            let (token, lp) = process_seq_logits(&mut serial, &rows, row, 4, 2, false, &ctx, false);
            emit_token(&mut serial, token, lp, &sched);
            expected.push(token);
        }
        assert_eq!(
            sample_and_emit(&rows, 4, &expected[..3], &mut seq, &sched, &ctx),
            expected
        );
        if temp == 0.0 {
            assert_eq!(expected, [3; 4]);
        }
    }
}

#[test]
fn eos_at_every_position_stops_before_suffix_sampling() {
    for nd in [2, 3] {
        for stopped in 0..=nd {
            let sched = SchedCtx::for_test();
            let ctx = context(&sched);
            let (mut seq, _rx) = test_seq(Vec::new(), 100, None, 10);
            seq.finished = false;
            seq.eos_tokens = vec![2];
            seq.min_tokens = 0;
            let mut values = [[0.0, 10.0, 0.0, 0.0]; 4];
            values[stopped] = [0.0, 0.0, 10.0, 0.0];
            let picks = sample_and_emit(&bytes(&values), 4, &vec![1; nd], &mut seq, &sched, &ctx);
            assert!(seq.finished);
            assert_eq!(picks.len(), stopped + 1);
            assert_eq!(seq.output_tokens.last(), Some(&2));
        }
    }
}

#[test]
fn context_limit_uses_committed_row_and_restores_forward_length() {
    for nd in [2, 3] {
        let k = nd + 1;
        let base = 100;
        for until_limit in 1..=k {
            for rejection in 0..=nd {
                let mut sched = SchedCtx::for_test();
                sched.limits.max_seq_len = base + until_limit + 1;
                let ctx = context(&sched);
                let (mut seq, _rx) = test_seq(Vec::new(), 100, None, base + k);
                seq.finished = false;
                seq.inside_thinking = true;
                seq.force_end_thinking = true;
                let mut drafts = vec![1; nd];
                if rejection < nd {
                    drafts[rejection] = 2;
                }
                let picks = sample_and_emit(
                    &bytes(&[[0.0, 10.0, 0.0, 0.0]; 4]),
                    4,
                    &drafts,
                    &mut seq,
                    &sched,
                    &ctx,
                );
                let emitted = until_limit.min(rejection + 1);
                assert_eq!(
                    picks.len(),
                    emitted,
                    "K={k}, limit={until_limit}, reject={rejection}"
                );
                assert_eq!(seq.finished, emitted == until_limit);
                assert_eq!(seq.sentence_defer_count as usize, emitted);
                // The GPU forward length must survive until the verdict
                // broadcast and the common model-state rewind use it.
                assert_eq!(seq.seq.seq_len, base + k);
            }
        }
    }
}
