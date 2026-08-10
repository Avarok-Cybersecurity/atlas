// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the `expert_pack` record codec + index (split from the
//! parent module to keep it under the 500-LoC cap).

use super::*;

fn spec() -> ExpertRecordSpec {
    ExpertRecordSpec::new(64, 128, 16, 256)
}

#[test]
fn pack_unpack_round_trips_in_memory() {
    let spec = spec();
    let stride = ExpertLayout::from_spec(1, 1, &spec, 4096).record_stride;
    let mk = |p: Proj, base: u8| {
        let pb = spec.proj_bytes(p);
        let packed: Vec<u8> = (0..pb.packed_bytes).map(|i| i as u8 ^ base).collect();
        let scale: Vec<u8> = (0..pb.scale_bytes)
            .map(|i| (i as u8).wrapping_add(base))
            .collect();
        (packed, scale)
    };
    let g = mk(Proj::Gate, 1);
    let u = mk(Proj::Up, 2);
    let d = mk(Proj::Down, 3);
    let projs = [
        ProjData {
            packed: &g.0,
            scale: &g.1,
        },
        ProjData {
            packed: &u.0,
            scale: &u.1,
        },
        ProjData {
            packed: &d.0,
            scale: &d.1,
        },
    ];
    let header = ExpertRecordHeader {
        layer: 2,
        expert: 1,
        inter: 64,
        hidden: 128,
        group_size: 16,
        scale2: [0.1, 0.2, 0.3],
        input_scale: [Some(1.0), Some(2.0), None],
    };
    let rec = pack_record(&spec, stride, &header, &projs).unwrap();
    assert_eq!(rec.len() as u64, stride);
    let (hdr, views) = unpack_record(&spec, &rec).unwrap();
    assert_eq!(hdr.layer, 2);
    assert_eq!(hdr.expert, 1);
    assert_eq!(hdr.input_scale[2], None);
    assert_eq!(views[Proj::Gate as usize].packed, &g.0[..]);
    assert_eq!(views[Proj::Up as usize].scale, &u.1[..]);
    assert_eq!(views[Proj::Down as usize].packed, &d.0[..]);
}

#[test]
fn unpack_rejects_undersized_buffer() {
    let spec = spec();
    let tiny = vec![0u8; 8];
    assert!(unpack_record(&spec, &tiny).is_err());
}

#[test]
fn index_total_bytes_matches_layout() {
    let index = ExpertIndex::new(512, 2048, 16, 256, 4096, vec![0, 1, 2], 256);
    let per_layer = index.layout().bytes_per_layer();
    assert_eq!(index.total_bytes(), 3 * per_layer);
}
