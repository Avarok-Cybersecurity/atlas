// SPDX-License-Identifier: AGPL-3.0-only

//! The self-check against row-invariant, row-dependent and ticket-leaking fakes on the mock backend, and
//! the per-row fallback it selects.

use super::*;
use crate::port::w4::fake::{Call, FakeW4, log};
use parking_lot::Mutex;
use spark_runtime::gpu::mock::MockGpuBackend;
use std::sync::Arc;

/// Writes `rows` (as BF16) into every output element: the batched call and the single-row calls differ.
/// Optionally leaves the first ticket at 1. Counts its decode calls by row count.
struct Bad {
    n: usize,
    k: usize,
    row_dependent: bool,
    dirty_tickets: bool,
    calls: Mutex<Vec<usize>>,
}

impl Bad {
    fn new(row_dependent: bool, dirty_tickets: bool) -> Self {
        Self {
            n: 8,
            k: 32,
            row_dependent,
            dirty_tickets,
            calls: Mutex::new(Vec::new()),
        }
    }
}

impl W4Linear for Bad {
    fn format(&self) -> W4Format {
        W4Format::Nvfp4
    }
    fn n(&self) -> usize {
        self.n
    }
    fn k(&self) -> usize {
        self.k
    }
    fn device_bytes(&self) -> usize {
        self.n * self.k / 2
    }
    fn max_decode_rows(&self) -> usize {
        16
    }
    fn scratch_bytes(&self, _max_rows: usize) -> usize {
        0
    }
    fn decode(
        &self,
        gpu: &dyn GpuBackend,
        _x: DevicePtr,
        rows: usize,
        out: DevicePtr,
        s: &W4Scratch,
        _stream: u64,
    ) -> Result<()> {
        self.calls.lock().push(rows);
        let v = if self.row_dependent { rows as f32 } else { 0.5 };
        let mut o = Vec::with_capacity(rows * self.n * 2);
        for _ in 0..rows * self.n {
            o.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
        }
        gpu.copy_h2d(&o, out)?;
        if self.dirty_tickets {
            gpu.copy_h2d(&[1, 0, 0, 0], s.tickets)?;
        }
        Ok(())
    }
    fn prefill(
        &self,
        _gpu: &dyn GpuBackend,
        _x: DevicePtr,
        _m: usize,
        _out: DevicePtr,
        _s: &W4Scratch,
        _stream: u64,
    ) -> Result<bool> {
        Ok(false)
    }
    fn dequant_bf16(&self, _gpu: &dyn GpuBackend, _out: DevicePtr, _stream: u64) -> Result<()> {
        Ok(())
    }
}

#[test]
fn inputs_are_exact_bf16_and_deterministic() {
    let a = input_rows(ROWS, 64);
    assert_eq!(a.len(), ROWS * 64 * 2);
    assert_eq!(a, input_rows(ROWS, 64));
    for b in a.chunks_exact(2) {
        let v = half::bf16::from_le_bytes([b[0], b[1]]).to_f32();
        assert!((-1.0..1.0).contains(&v));
        assert_eq!((v * 64.0).fract(), 0.0, "{v} is not a multiple of 1/64");
    }
    assert_ne!(a[..128], a[128..256], "rows differ");
}

#[test]
fn a_row_invariant_weight_passes_and_frees_its_buffers() {
    let gpu = MockGpuBackend::new();
    let l = log();
    let w = FakeW4::new("gate", 48, 64, 16, &l).computing(7);
    let live_before = gpu.live_alloc_count();
    assert_eq!(row_invariance(&gpu, &w, 0).unwrap(), Outcome::Pass);
    let calls: Vec<usize> = l
        .lock()
        .iter()
        .map(|(_, c)| match c {
            Call::Decode { rows, .. } => *rows,
            other => panic!("unexpected {other:?}"),
        })
        .collect();
    assert_eq!(
        calls,
        vec![4, 1, 1, 1, 1],
        "one batched call, then one per row"
    );
    // Only the shared scratch regions (kept for the layer) remain; the check's own buffers are freed.
    let scratch_regions = 5;
    assert!(gpu.live_alloc_count() <= live_before + scratch_regions);
}

#[test]
fn a_row_dependent_weight_is_a_mismatch_at_the_first_element() {
    let gpu = MockGpuBackend::new();
    let w = Bad::new(true, false);
    assert_eq!(
        row_invariance(&gpu, &w, 0).unwrap(),
        Outcome::Mismatch { row: 0, col: 0 }
    );
    assert!(Outcome::Mismatch { row: 0, col: 0 }.failed());
}

#[test]
fn a_ticket_left_nonzero_fails_even_when_the_bits_agree() {
    let gpu = MockGpuBackend::new();
    let w = Bad::new(false, true);
    assert_eq!(row_invariance(&gpu, &w, 0).unwrap(), Outcome::TicketsDirty);
    // The shared ticket region is zero again for the next weight's calls.
    let s = scratch::acquire(&gpu, &ScratchSpec::for_linear(&w), 0).unwrap();
    let t = gpu.read_alloc(s.tickets).expect("ticket region is live");
    assert!(
        t.iter().all(|&v| v == 0),
        "the check restores the zero-on-entry rule"
    );
}

#[test]
fn a_narrow_decode_window_is_skipped() {
    let gpu = MockGpuBackend::new();
    let l = log();
    let w = FakeW4::new("gate", 8, 32, 2, &l).computing(1);
    assert_eq!(row_invariance(&gpu, &w, 0).unwrap(), Outcome::Skipped);
    assert!(!Outcome::Skipped.failed());
    assert!(l.lock().is_empty(), "nothing decoded");
}

#[test]
fn per_row_decodes_one_row_at_a_time_at_the_right_offsets() {
    let gpu = MockGpuBackend::new();
    let l = log();
    let inner: W4Box = Arc::new(FakeW4::new("gate", 48, 64, 16, &l));
    let w = per_row(inner);
    let s = W4Scratch {
        ws: DevicePtr::NULL,
        ws_bytes: 0,
        tickets: DevicePtr::NULL,
        xq: DevicePtr::NULL,
        xs: DevicePtr::NULL,
    };
    let (x, out) = (DevicePtr(0x1000), DevicePtr(0x9000));
    w.decode(&gpu, x, 3, out, &s, 0).unwrap();
    let got: Vec<Call> = l.lock().iter().map(|(_, c)| c.clone()).collect();
    let want: Vec<Call> = (0..3)
        .map(|r| Call::Decode {
            x: x.offset(r * 64 * 2),
            rows: 1,
            out: out.offset(r * 48 * 2),
        })
        .collect();
    assert_eq!(got, want);
    assert!(w.decode(&gpu, x, 17, out, &s, 0).is_err());
    assert_eq!((w.n(), w.k(), w.max_decode_rows()), (48, 64, 16));
    assert_eq!(w.format(), W4Format::Affine { bits: 4, group: 64 });
}

#[test]
fn a_failed_role_wraps_this_layer_and_every_later_one_and_checks_once() {
    let gpu = MockGpuBackend::new();
    let mut checks = SelfChecks::default();
    let before = w4_stats::selfcheck_failures();
    let bad = Arc::new(Bad::new(true, false));
    let first = checks
        .apply(
            &gpu,
            W4Role::Gdn,
            0,
            [bad.clone() as W4Box, bad.clone() as W4Box],
            0,
        )
        .unwrap();
    assert!(checks.per_row(W4Role::Gdn));
    assert!(w4_stats::selfcheck_failures() > before);
    let checked_calls = bad.calls.lock().len();
    assert_eq!(
        checked_calls, 5,
        "the check stops at the first failing projection"
    );
    // Installed boxes decode row by row.
    let s = W4Scratch {
        ws: DevicePtr::NULL,
        ws_bytes: 0,
        tickets: DevicePtr::NULL,
        xq: DevicePtr::NULL,
        xs: DevicePtr::NULL,
    };
    let out = gpu.alloc(4 * 8 * 2).unwrap();
    first[0]
        .decode(&gpu, DevicePtr(0x10), 4, out, &s, 0)
        .unwrap();
    assert_eq!(bad.calls.lock()[checked_calls..], [1, 1, 1, 1]);
    // A later layer of the role is wrapped without a second check.
    let n = bad.calls.lock().len();
    let later = checks
        .apply(&gpu, W4Role::Gdn, 1, [bad.clone() as W4Box], 0)
        .unwrap();
    assert_eq!(bad.calls.lock().len(), n, "no second check");
    later[0]
        .decode(&gpu, DevicePtr(0x10), 2, out, &s, 0)
        .unwrap();
    assert_eq!(bad.calls.lock()[n..], [1, 1]);
    // Another role is checked on its own.
    assert!(!checks.checked(W4Role::Ffn));
}

#[test]
fn a_passing_role_installs_the_boxes_unchanged() {
    let gpu = MockGpuBackend::new();
    let l = log();
    let mut checks = SelfChecks::default();
    let a: W4Box = Arc::new(FakeW4::new("gate", 48, 64, 16, &l).computing(3));
    let b: W4Box = Arc::new(FakeW4::new("down", 64, 48, 16, &l).computing(4));
    let out = checks
        .apply(&gpu, W4Role::Ffn, 0, [a.clone(), b.clone()], 0)
        .unwrap();
    assert!(checks.checked(W4Role::Ffn) && !checks.per_row(W4Role::Ffn));
    assert!(Arc::ptr_eq(&out[0], &a) && Arc::ptr_eq(&out[1], &b));
    assert_eq!(
        l.lock().len(),
        10,
        "both projections checked: 5 decodes each"
    );
}
