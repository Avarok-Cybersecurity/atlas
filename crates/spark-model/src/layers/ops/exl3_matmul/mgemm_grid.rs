// SPDX-License-Identifier: AGPL-3.0-only

//! Cooperative EXL3 slot scheduling, independent of device I/O.

pub(super) fn grid(
    tiles: usize,
    slots: usize,
    sms: usize,
    replay_group: Option<usize>,
) -> Option<(usize, usize)> {
    let group = replay_group.unwrap_or(slots);
    if tiles == 0 || slots == 0 || sms == 0 || group == 0 || !slots.is_multiple_of(group) {
        return None;
    }
    let mut per_slot = tiles;
    if per_slot > sms / group {
        per_slot = (sms / group).max(1);
    }
    if per_slot <= sms && tiles / per_slot > 48 {
        per_slot = sms.min(per_slot * 2);
    }
    Some((per_slot, (sms / per_slot).min(slots).max(1)))
}

#[cfg(test)]
mod tests {
    use super::grid;

    #[test]
    fn verify_keeps_decode_split_and_schedules_extra_slot_waves() {
        // Eight experts per token on GB10. All three tokens fit in one
        // cooperative launch, traversed in waves of eight resident slots.
        assert_eq!(grid(64, 8, 48, None), Some((6, 8)));
        assert_eq!(grid(64, 24, 48, None), Some((2, 24)));
        assert_eq!(grid(64, 24, 48, Some(8)), Some((6, 8)));
    }

    #[test]
    fn every_replay_preserves_decode_split_and_cooperative_residency() {
        for sms in [1, 16, 48, 80, 132] {
            for tiles in [1, 4, 16, 128, 1024, 16384] {
                for experts in [1, 2, 8, 10, 32] {
                    let serial = grid(tiles, experts, sms, None).unwrap();
                    for rows in 1..=4 {
                        let slots = rows * experts;
                        let replay = grid(tiles, slots, sms, Some(experts)).unwrap();
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

    #[test]
    fn invalid_launch_dimensions_are_refused() {
        for args in [
            (0, 8, 48, None),
            (128, 0, 48, None),
            (128, 8, 0, None),
            (128, 8, 48, Some(0)),
            (128, 8, 48, Some(3)),
        ] {
            assert_eq!(grid(args.0, args.1, args.2, args.3), None);
        }
    }
}
