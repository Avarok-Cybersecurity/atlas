// SPDX-License-Identifier: AGPL-3.0-only

//! What the Network pane says about the topology.
//!
//! The rule this pane is built on is that it never fakes a measurement it did
//! not take: a worker's state is not visible from the head, and the link
//! latency is deliberately not probed. Both of those are assertions here,
//! because "—" is easy to replace with a plausible number by accident.

use super::super::harness::{has, screen};
use crate::tui::app::{App, Section};

fn net(world: usize, tp: usize, ep: usize) -> App {
    let mut a = crate::tui::render::tests::app();
    a.section = Section::Network;
    a.args.world_size = world;
    a.args.tp_size = tp;
    a.args.ep_size = ep;
    a
}

#[test]
fn a_single_node_deployment_gets_a_centrepiece_rather_than_an_empty_screen() {
    let mut a = net(1, 1, 1);
    a.stats.atlas_used_gb = 57.0;
    a.stats.gpu_total_gb = 119.7;
    let rows = screen(&a, 160, 48);
    assert!(has(&rows, "rank 0 · HEAD"), "{rows:#?}");
    assert!(has(&rows, "NVIDIA GB10 · local"));
    assert!(
        has(&rows, "57/120 GB"),
        "the memory bar is labelled:\n{rows:#?}"
    );
    assert!(has(&rows, "tp 1 · ep 1 · world 1 — all local"));
    assert!(has(&rows, "❯❯❯"), "the stage chevrons flank the card");
}

#[test]
fn two_ranks_draw_two_cards_and_name_the_fabric_between_them() {
    let rows = screen(&net(2, 1, 2), 160, 48);
    assert!(has(&rows, "rank 0"), "{rows:#?}");
    assert!(has(&rows, "rank 1"));
    assert!(has(&rows, "══"), "the connector is drawn");
    assert!(
        has(&rows, "NCCL ═ ep 2 · experts sharded"),
        "EP is named ahead of TP because it is what shards the weights:\n{rows:#?}"
    );
}

#[test]
fn a_tensor_parallel_world_is_labelled_as_a_ring_not_as_expert_sharding() {
    let rows = screen(&net(2, 2, 1), 160, 48);
    assert!(has(&rows, "NCCL ═ tp 2 · ring"), "{rows:#?}");
    assert!(!has(&rows, "experts sharded"));
}

#[test]
fn a_world_wider_than_four_draws_the_four_that_fit() {
    let rows = screen(&net(8, 8, 1), 200, 50);
    assert!(has(&rows, "rank 3"), "{rows:#?}");
    assert!(
        !has(&rows, "rank 4"),
        "a fifth card would be two columns wide and unreadable:\n{rows:#?}"
    );
    // The detail pane still reports the REAL world size, so the cap cannot be
    // mistaken for the topology.
    assert!(has(&rows, "world 8 · tp 8 · ep 1"), "{rows:#?}");
}

#[test]
fn a_worker_card_admits_that_the_head_cannot_see_its_state() {
    let rows = screen(&net(2, 2, 1), 160, 48);
    assert!(
        has(&rows, "state not visible from head (no worker HTTP)"),
        "{rows:#?}"
    );
}

#[test]
fn the_detail_pane_refuses_to_invent_a_link_latency() {
    let rows = screen(&net(2, 2, 1), 160, 48);
    assert!(has(&rows, "NODE 0 ─ detail"), "{rows:#?}");
    assert!(has(&rows, "head (serves HTTP, owns scheduler)"));
    assert!(has(&rows, "(NCCL master)"));
    assert!(
        has(&rows, "deliberately not measured live"),
        "an unmeasured latency says so:\n{rows:#?}"
    );
}

#[test]
fn selecting_a_worker_repoints_the_detail_pane_at_it() {
    let mut a = net(4, 4, 1);
    a.network_selected = 2;
    let rows = screen(&a, 160, 48);
    assert!(has(&rows, "NODE 2 ─ detail"), "{rows:#?}");
    assert!(has(&rows, "worker (EP shard, no HTTP)"));
}

#[test]
fn the_network_pane_survives_narrow_and_short_terminals() {
    for world in [1usize, 2, 4] {
        for (w, h) in [(20u16, 4u16), (20, 8), (20, 40), (40, 12), (200, 3)] {
            let rows = screen(&net(world, world, 1), w, h);
            assert_eq!(
                rows.len(),
                h as usize,
                "world {world} at {w}x{h} drew a partial frame"
            );
        }
    }
}
