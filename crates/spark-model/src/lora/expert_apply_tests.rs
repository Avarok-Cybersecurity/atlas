// SPDX-License-Identifier: AGPL-3.0-only

//! Feature-1 apply-planner tests: the correctness-critical mapping from the base
//! MoE `expert_offsets` prefix-sum + the adapted-expert set to the per-expert
//! (row_off, rows) delta work-items. GPU-free (the fold itself is
//! `apply_lora_delta`, exercised on hardware).

use crate::lora::*;

#[test]
fn workitems_map_offsets_to_row_ranges() {
    // 4 experts; expert_offsets is the [E+1] prefix sum of sorted rows.
    // expert 0: rows [0,3)=3 ; 1: [3,3)=0 (none routed) ; 2: [3,10)=7 ; 3: [10,12)=2.
    let offsets = [0u32, 3, 3, 10, 12];
    // Adapter adapts experts 0, 2, 3 (not 1).
    let work = expert_delta_workitems(&offsets, &[0, 2, 3]);
    assert_eq!(
        work,
        vec![
            ExpertWork {
                expert: 0,
                row_off: 0,
                rows: 3
            },
            ExpertWork {
                expert: 2,
                row_off: 3,
                rows: 7
            },
            ExpertWork {
                expert: 3,
                row_off: 10,
                rows: 2
            },
        ]
    );
}

#[test]
fn workitems_skip_empty_and_out_of_range() {
    let offsets = [0u32, 5, 5]; // 2 experts; expert 1 has zero rows.
    // Adapts expert 1 (empty → skipped) and expert 9 (out of range → skipped).
    assert!(expert_delta_workitems(&offsets, &[1, 9]).is_empty());
    // Adapts expert 0 (5 rows) → single work-item.
    assert_eq!(
        expert_delta_workitems(&offsets, &[0]),
        vec![ExpertWork {
            expert: 0,
            row_off: 0,
            rows: 5
        }]
    );
}

#[test]
fn workitems_only_adapted_experts_launch() {
    // Every expert has routed rows, but the adapter adapts only expert 1.
    let offsets = [0u32, 4, 8, 12];
    let work = expert_delta_workitems(&offsets, &[1]);
    assert_eq!(
        work,
        vec![ExpertWork {
            expert: 1,
            row_off: 4,
            rows: 4
        }]
    );
}
