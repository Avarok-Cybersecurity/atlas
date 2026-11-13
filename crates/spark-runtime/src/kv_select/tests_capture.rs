// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the query capture's slots on the mock backend (`kv_select::capture`):
//! spans clipped to the last `query_tokens` positions, rows landing in their slot rows
//! from stacked streams with padded strides, completeness per layer, re-arm and slot
//! reuse, the unrope launch for post-RoPE rows, and nothing allocated where the module is
//! absent.

use super::QueryView;
use super::capture::{DEFAULT_QUERY_TOKENS, QueryCapture, QueryCaptureCfg, StreamRows};
use crate::gpu::mock::MockGpuBackend;
use crate::gpu::{DevicePtr, GpuBackend};
use crate::kv_cache::sketch::KERNEL_MODULE;

const L: usize = 2;
const QH: usize = 2;
const HD: usize = 4;

fn cfg() -> QueryCaptureCfg {
    QueryCaptureCfg {
        attn_layers: L,
        q_heads: QH,
        head_dim: HD,
        query_tokens: 6,
        slots: 2,
        rotary_dim: 2,
        rope_theta: 1.0e7,
    }
}

/// Seq 7 at positions 0..5 then seq 9 at positions 10..14: nine Q rows of `stride`
/// elements, element value = row * 100 + column.
fn streams() -> [StreamRows; 2] {
    [
        StreamRows {
            seq_id: 7,
            first_pos: 0,
            n: 5,
        },
        StreamRows {
            seq_id: 9,
            first_pos: 10,
            n: 4,
        },
    ]
}

fn q_rows(gpu: &MockGpuBackend, stride: usize) -> DevicePtr {
    let host: Vec<u8> = (0..9 * stride)
        .flat_map(|i| (((i / stride) * 100 + i % stride) as u16).to_le_bytes())
        .collect();
    let q = gpu.alloc(host.len()).unwrap();
    gpu.copy_h2d(&host, q).unwrap();
    q
}

fn slot_row(
    gpu: &MockGpuBackend,
    cap: &QueryCapture,
    slot: usize,
    layer: usize,
    row: usize,
) -> Vec<u16> {
    let all = gpu.read_alloc(cap.buffer()).unwrap();
    let c = cap.config();
    let at = slot * c.slot_bytes() + (layer * c.query_tokens + row) * c.row_bytes();
    all[at..at + c.row_bytes()]
        .chunks(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

fn row(v: u16) -> Vec<u16> {
    (v..v + 8).collect()
}

#[test]
fn armed_rows_land_in_their_slot_rows_and_complete_the_view() {
    let gpu = MockGpuBackend::new();
    let cap = QueryCapture::new(&gpu, cfg()).unwrap();
    assert_eq!(gpu.live_bytes(), Some(cfg().total_bytes()));
    // Seq 9's message is positions 4..13: the slot keeps the last 6, 7..13.
    assert_eq!(cap.arm(9, 4, 13), Some((7, 13)));
    assert_eq!(cap.arm(7, 2, 4), Some((2, 4)));
    assert_eq!(cap.arm(8, 0, 9), None, "both slots taken");
    let q = q_rows(&gpu, 12);
    for layer in 0..L {
        // Seq 7 positions 2, 3 (q rows 2, 3); seq 9 positions 10..13 (q rows 5, 6, 7).
        let n = cap
            .capture(&gpu, layer, &streams(), q, 12, 9, true, 3)
            .unwrap();
        assert_eq!(n, 5);
    }
    assert_eq!(slot_row(&gpu, &cap, 1, 0, 0), row(200), "seq 7 position 2");
    assert_eq!(slot_row(&gpu, &cap, 0, 1, 3), row(500), "seq 9 position 10");
    assert_eq!(slot_row(&gpu, &cap, 0, 1, 5), row(700), "seq 9 position 12");
    assert_eq!(cap.captured_rows(9), Some(vec![3, 3]));
    assert!(
        cap.rows_view(9).is_none(),
        "positions 7..10 were never prefilled"
    );
    let v = cap.rows_view(7).expect("complete");
    assert_eq!(
        (v.rows(), v.layer_stride_rows(), v.layers(), v.q_heads()),
        (2, 6, L, QH)
    );
    assert_eq!(v.data(), cap.buffer().offset(cfg().slot_bytes()));
    assert!(
        gpu.launches_snapshot().is_empty(),
        "pre-RoPE rows need no unrope"
    );
    cap.disarm(7);
    assert!(cap.armed(7).is_none() && cap.rows_view(7).is_none());
    assert_eq!(cap.arm(8, 0, 9), Some((3, 9)), "the freed slot is reused");
    assert_eq!(
        cap.capture(&gpu, L, &streams(), q, 12, 9, true, 3).unwrap(),
        0,
        "no such layer"
    );
}

#[test]
fn post_rope_rows_are_rotated_back_in_place() {
    let gpu = MockGpuBackend::new();
    let cap = QueryCapture::new(&gpu, cfg()).unwrap();
    let q = q_rows(&gpu, 8);
    assert_eq!(cap.arm(9, 0, 100), Some((94, 100)));
    assert_eq!(
        cap.capture(&gpu, 1, &streams(), q, 8, 9, false, 3).unwrap(),
        0
    );
    // Re-arming resets the slot: positions 7..13, of which 10, 11, 12 are in this call.
    assert_eq!(cap.arm(9, 0, 13), Some((7, 13)));
    assert_eq!(
        cap.capture(&gpu, 1, &streams(), q, 8, 9, false, 3).unwrap(),
        3
    );
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 1);
    assert_eq!(l[0].grid, [3, QH as u32, 1]);
    assert_eq!(l[0].stream, 3);
    assert_eq!(l[0].args.len(), 8);
    // A contiguous stride: one copy, rows back to back in the slot.
    assert_eq!(slot_row(&gpu, &cap, 0, 1, 3), row(500));
}

#[test]
fn rows_past_the_call_or_below_the_row_stride_are_refused() {
    let gpu = MockGpuBackend::new();
    let cap = QueryCapture::new(&gpu, cfg()).unwrap();
    let q = q_rows(&gpu, 8);
    cap.arm(9, 10, 14);
    // The call claims 8 Q rows but seq 9's rows would end at row 9.
    assert_eq!(
        cap.capture(&gpu, 0, &streams(), q, 8, 8, true, 0).unwrap(),
        0
    );
    assert!(cap.capture(&gpu, 0, &streams(), q, 7, 9, true, 0).is_err());
    assert!(gpu.launches_snapshot().is_empty());
}

#[test]
fn nothing_is_allocated_where_the_module_is_absent() {
    let gpu = MockGpuBackend::new();
    gpu.mark_module_absent(KERNEL_MODULE);
    assert!(QueryCapture::new(&gpu, cfg()).is_err());
    assert_eq!(gpu.live_alloc_count(), 0);
    assert!(
        QueryCaptureCfg {
            rotary_dim: 3,
            ..cfg()
        }
        .validate()
        .is_err()
    );
    assert!(QueryCaptureCfg { slots: 0, ..cfg() }.validate().is_err());
    let four_b = QueryCaptureCfg {
        attn_layers: 8,
        q_heads: 16,
        head_dim: 256,
        query_tokens: DEFAULT_QUERY_TOKENS,
        slots: 1,
        rotary_dim: 64,
        rope_theta: 1.0e7,
    };
    assert_eq!(four_b.total_bytes(), 32 << 20, "32 MiB per slot on the 4B");
}
