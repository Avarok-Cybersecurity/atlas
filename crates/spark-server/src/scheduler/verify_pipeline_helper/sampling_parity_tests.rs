// SPDX-License-Identifier: AGPL-3.0-only

use super::verify_pick_with_pipeline;
use crate::scheduler::logit_processors::{LogitsContext, SamplingLevers};
use crate::scheduler::sample_step::{PositionKind, penalty_params_for};
use crate::scheduler::test_support::test_seq;

#[test]
fn verify_preserves_request_bias_at_greedy_and_card_temperatures() {
    let scratch = crate::scheduler::sched_ctx::DecodeScratch::default();
    let dumps = crate::scheduler::dumps::RunDumps::default();
    let ctx = LogitsContext {
        scratch: &scratch,
        dumps: &dumps,
        stats: std::sync::Arc::new(crate::scheduler::spec_stats::SpecStats::new()),
        watchdog: crate::scheduler::helpers::WatchdogParams::default(),
        boundary_mask: None,
        mid_word_mask: None,
        sampling: SamplingLevers {
            mtp_verify_sample: true,
            mtp_minp: true,
            ..SamplingLevers::default()
        },
        timing: std::sync::Arc::default(),
        think_end_token: None,
        think_start_token: None,
        tool_call_start_token: None,
        tool_call_end_token: None,
    };
    // The highest raw logit belongs to a request-banned token. The request
    // leaves only token 1 available, so this also checks stochastic verify
    // without relying on any particular random draw.
    let bytes: Vec<u8> = [10.0f32, 1.0, 0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    for temperature in [0.0, 0.7, 1.0] {
        let (mut seq, _rx) = test_seq(Vec::new(), 100, None, 10);
        seq.temperature = temperature;
        seq.top_k = 20;
        seq.top_p = 0.95;
        seq.seed = Some(123);
        seq.logit_bias = vec![(0, f32::NEG_INFINITY), (2, f32::NEG_INFINITY)];
        assert_eq!(
            verify_pick_with_pipeline(&bytes, true, 3, &mut seq, &ctx, 0),
            1,
            "verify ignored request bias at temperature {temperature}"
        );
    }
}

#[test]
fn verify_and_serial_share_tool_opener_bias_rules() {
    let (mut seq, _rx) = test_seq(Vec::new(), 100, None, 10);
    seq.tool_call_start_token = Some(1);
    seq.logit_bias = vec![(1, 3.0), (2, -5.0)];
    for (inside_tool_body, inside_thinking) in [(false, false), (true, false), (true, true)] {
        seq.inside_tool_body = inside_tool_body;
        seq.inside_thinking = inside_thinking;
        let serial = penalty_params_for(
            &seq,
            PositionKind::FinalDecode,
            0.7,
            Some(123),
            seq.logit_bias.clone(),
        );
        let verify = penalty_params_for(&seq, PositionKind::Verify, 0.0, None, Vec::new());
        assert_eq!(verify.logit_bias, serial.logit_bias);
        assert_eq!(seq.logit_bias, vec![(1, 3.0), (2, -5.0)]);
    }
}
