// SPDX-License-Identifier: AGPL-3.0-only

//! Cooperative EXL3 slot scheduling, independent of device I/O.
//!
//! Two policies, selected once per process by `ATLAS_EXL3_MGEMM_GRID`:
//!
//! * `onewave` (default): `per_slot = clamp(sms / group, 1, tiles)` blocks per
//!   expert slot, `concurrency = min(sms / per_slot, slots)`. Sized so ONE
//!   token's `group = top_k` slots fill the device in a single wave (48 SMs /
//!   10 experts → 4 blocks per slot, 10 slots resident). A wider slot batch
//!   (MTP verify, `slots = rows * top_k`) keeps the SAME per-slot split-K and
//!   walks more slots per wave (12 resident at 48 SMs), so serial and verify
//!   reduce each slot identically — the `stable_token_grid` contract.
//! * `legacy`: upstream ExLlamaV3's heuristic as first ported — the same
//!   start, then "if tiles / per_slot > 48, double per_slot". Tuned for
//!   128+-SM parts; on GB10 it turns 4 x 10 into 8 x 6, i.e. two waves for
//!   ten slots (6 + 4, the second wave 2/3 empty) and five waves for the
//!   30-slot verify batch. The standalone microbench
//!   (`.research/exl3_decode_perf/exl3_decode_bench.cu grid`, idle GB10)
//!   measured the routed gate+up+down chain at 163 → 144 us per layer for
//!   one token (4 x 10 vs 8 x 6) and 416 → 405 us at the 30-slot verify
//!   width (4 x 12 vs 8 x 6 in five waves) — ~0.9 ms per decode token.

fn onewave_enabled() -> bool {
    static POLICY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *POLICY.get_or_init(|| std::env::var("ATLAS_EXL3_MGEMM_GRID").as_deref() != Ok("legacy"))
}

pub(super) fn grid(
    tiles: usize,
    slots: usize,
    sms: usize,
    replay_group: Option<usize>,
) -> Option<(usize, usize)> {
    grid_with_policy(tiles, slots, sms, replay_group, onewave_enabled())
}

pub(super) fn grid_with_policy(
    tiles: usize,
    slots: usize,
    sms: usize,
    replay_group: Option<usize>,
    onewave: bool,
) -> Option<(usize, usize)> {
    let group = replay_group.unwrap_or(slots);
    if tiles == 0 || slots == 0 || sms == 0 || group == 0 || !slots.is_multiple_of(group) {
        return None;
    }
    let mut per_slot = tiles;
    if per_slot > sms / group {
        per_slot = (sms / group).max(1);
    }
    if !onewave && per_slot <= sms && tiles / per_slot > 48 {
        per_slot = sms.min(per_slot * 2);
    }
    Some((per_slot, (sms / per_slot).min(slots).max(1)))
}

#[cfg(test)]
mod tests {
    use super::grid_with_policy;

    #[test]
    fn legacy_keeps_decode_split_and_schedules_extra_slot_waves() {
        // Eight experts per token on GB10 under the legacy heuristic.
        assert_eq!(grid_with_policy(64, 8, 48, None, false), Some((6, 8)));
        assert_eq!(grid_with_policy(64, 24, 48, None, false), Some((2, 24)));
        assert_eq!(grid_with_policy(64, 24, 48, Some(8), false), Some((6, 8)));
    }

    #[test]
    fn onewave_fills_the_device_with_one_token_and_widens_for_verify() {
        // qwen4_exp on GB10: 400 tiles (2560x640 at shape 2), top_k 10.
        assert_eq!(grid_with_policy(400, 10, 48, None, true), Some((4, 10)));
        assert_eq!(grid_with_policy(400, 10, 48, Some(10), true), Some((4, 10)));
        // MTP verify, 3 rows: same 4-block split, 12 slots per wave.
        assert_eq!(grid_with_policy(400, 30, 48, Some(10), true), Some((4, 12)));
        // Legacy on the same shapes: 8 x 6, two waves at one row, five at three.
        assert_eq!(grid_with_policy(400, 10, 48, Some(10), false), Some((8, 6)));
        assert_eq!(grid_with_policy(400, 30, 48, Some(10), false), Some((8, 6)));
    }

    #[test]
    fn every_replay_preserves_decode_split_and_cooperative_residency() {
        for onewave in [false, true] {
            for sms in [1, 16, 48, 80, 132] {
                for tiles in [1, 4, 16, 128, 1024, 16384] {
                    for experts in [1, 2, 8, 10, 32] {
                        let serial = grid_with_policy(tiles, experts, sms, None, onewave).unwrap();
                        for rows in 1..=4 {
                            let slots = rows * experts;
                            let replay =
                                grid_with_policy(tiles, slots, sms, Some(experts), onewave)
                                    .unwrap();
                            assert_eq!(replay.0, serial.0);
                            assert!(replay.0 * replay.1 <= sms);
                            assert!(replay.1 > 0 && replay.1 <= slots);
                            // Slot waves cover every matrix exactly once.
                            let visited: Vec<_> = (0..slots.div_ceil(replay.1))
                                .flat_map(|wave| (0..replay.1).map(move |z| wave * replay.1 + z))
                                .filter(|&slot| slot < slots)
                                .collect();
                            assert_eq!(visited, (0..slots).collect::<Vec<_>>());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn invalid_launch_dimensions_are_refused() {
        for args in [
            (0, 8, 48, None),
            (128, 0, 48, None),
            (128, 8, 0, None),
            (128, 8, 48, Some(0)),
            (128, 8, 48, Some(3)),
        ] {
            assert_eq!(grid_with_policy(args.0, args.1, args.2, args.3, true), None);
            assert_eq!(
                grid_with_policy(args.0, args.1, args.2, args.3, false),
                None
            );
        }
    }
}
