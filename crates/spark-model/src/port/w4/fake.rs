// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: a test-only [`W4Linear`] for the consumers of the 3.7.2
//! contract. It records every call (with its pointers and row counts) into a
//! log shared by the fakes of one layer, so a test can assert the exact call
//! sequence a dispatcher makes, and it can optionally COMPUTE on the mock
//! backend's memory: `out[r][j] = sum_i w(j, i) * x[r][i]` in f32, rounded to
//! BF16, with a fixed weight function. Decode sums in ascending `i`, prefill
//! in descending `i`, so the two routes are distinguishable bit for bit the
//! way real decode and prefill kernels are.

use super::{W4Format, W4Linear, W4Scratch};
use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::sync::Arc;

/// One recorded call. `Pair` is only recorded by a fake built with
/// [`FakeW4::fusing_pair`]; otherwise the trait's default body records two
/// `Decode`s.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Call {
    Decode {
        x: DevicePtr,
        rows: usize,
        out: DevicePtr,
    },
    Pair {
        x: DevicePtr,
        rows: usize,
        out_a: DevicePtr,
        out_b: DevicePtr,
    },
    Prefill {
        x: DevicePtr,
        m: usize,
        out: DevicePtr,
    },
    Dequant {
        out: DevicePtr,
    },
}

/// The call log shared by the fakes of one test: `(fake name, call)`.
pub type Log = Arc<Mutex<Vec<(&'static str, Call)>>>;

pub fn log() -> Log {
    Arc::new(Mutex::new(Vec::new()))
}

pub struct FakeW4 {
    pub name: &'static str,
    n: usize,
    k: usize,
    max_rows: usize,
    format: W4Format,
    log: Log,
    /// Answers for successive `prefill` calls; once exhausted, `true`.
    declines: Mutex<std::collections::VecDeque<bool>>,
    /// `Some(seed)`: compute on the mock's memory with that weight seed.
    cpu_seed: Option<u32>,
    fused_pair: bool,
    /// Every `W4Scratch` seen, to check the dispatcher hands one out.
    pub scratches: Mutex<Vec<W4Scratch>>,
}

impl FakeW4 {
    pub fn new(name: &'static str, n: usize, k: usize, max_rows: usize, log: &Log) -> Self {
        Self {
            name,
            n,
            k,
            max_rows,
            format: W4Format::Affine { bits: 4, group: 64 },
            log: log.clone(),
            declines: Mutex::new(Default::default()),
            cpu_seed: None,
            fused_pair: false,
            scratches: Mutex::new(Vec::new()),
        }
    }

    /// Compute real outputs on the mock backend's memory.
    pub fn computing(mut self, seed: u32) -> Self {
        self.cpu_seed = Some(seed);
        self
    }

    /// Successive `prefill` answers: `false` = decline.
    pub fn answering(self, answers: &[bool]) -> Self {
        self.declines.lock().extend(answers.iter().copied());
        self
    }

    /// Override `decode_pair` (recorded as one `Pair` call).
    pub fn fusing_pair(mut self) -> Self {
        self.fused_pair = true;
        self
    }

    /// The weight value at output `j`, input `i`: small dyadic rationals, so
    /// f32 products are exact and only the summation order rounds.
    pub fn w(seed: u32, j: usize, i: usize) -> f32 {
        let v = (j as u32).wrapping_mul(31) ^ (i as u32).wrapping_mul(17) ^ seed;
        ((v % 13) as f32 - 6.0) / 8.0
    }

    fn record(&self, c: Call) {
        self.log.lock().push((self.name, c));
    }

    fn compute(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        rows: usize,
        out: DevicePtr,
        descending: bool,
    ) -> Result<()> {
        let Some(seed) = self.cpu_seed else {
            return Ok(());
        };
        let mut xb = vec![0u8; rows * self.k * 2];
        gpu.copy_d2h(x, &mut xb)?;
        let xs: Vec<f32> = xb
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect();
        let mut ob = Vec::with_capacity(rows * self.n * 2);
        for r in 0..rows {
            let row = &xs[r * self.k..(r + 1) * self.k];
            for j in 0..self.n {
                let mut acc = 0.0f32;
                if descending {
                    for (i, &xv) in row.iter().enumerate().rev() {
                        acc += Self::w(seed, j, i) * xv;
                    }
                } else {
                    for (i, &xv) in row.iter().enumerate() {
                        acc += Self::w(seed, j, i) * xv;
                    }
                }
                ob.extend_from_slice(&half::bf16::from_f32(acc).to_le_bytes());
            }
        }
        gpu.copy_h2d(&ob, out)
    }
}

impl W4Linear for FakeW4 {
    fn format(&self) -> W4Format {
        self.format
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
        self.max_rows
    }
    fn scratch_bytes(&self, max_rows: usize) -> usize {
        // A split-K workspace shape: 16-row tile partials of 4 slices.
        max_rows.div_ceil(16) * 16 * self.n * 4 * 4
    }
    fn decode(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        rows: usize,
        out: DevicePtr,
        s: &W4Scratch,
        _stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            (1..=self.max_rows).contains(&rows),
            "{}: decode rows {rows} outside 1..={}",
            self.name,
            self.max_rows
        );
        self.scratches.lock().push(*s);
        self.record(Call::Decode { x, rows, out });
        self.compute(gpu, x, rows, out, false)
    }
    fn prefill(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        m: usize,
        out: DevicePtr,
        s: &W4Scratch,
        _stream: u64,
    ) -> Result<bool> {
        anyhow::ensure!(
            m > self.max_rows,
            "{}: prefill m {m} must exceed max_decode_rows {}",
            self.name,
            self.max_rows
        );
        self.scratches.lock().push(*s);
        self.record(Call::Prefill { x, m, out });
        let accept = self.declines.lock().pop_front().unwrap_or(true);
        if accept {
            self.compute(gpu, x, m, out, true)?;
        }
        Ok(accept)
    }
    fn decode_pair(
        &self,
        other: &dyn W4Linear,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        rows: usize,
        out_a: DevicePtr,
        out_b: DevicePtr,
        s: &W4Scratch,
        stream: u64,
    ) -> Result<()> {
        let partner = other
            .as_any()
            .and_then(|a| a.downcast_ref::<FakeW4>())
            .is_some();
        if !(self.fused_pair && partner) {
            self.decode(gpu, x, rows, out_a, s, stream)?;
            return other.decode(gpu, x, rows, out_b, s, stream);
        }
        self.record(Call::Pair {
            x,
            rows,
            out_a,
            out_b,
        });
        self.compute(gpu, x, rows, out_a, false)?;
        let o = other
            .as_any()
            .and_then(|a| a.downcast_ref::<FakeW4>())
            .expect("checked");
        o.compute(gpu, x, rows, out_b, false)
    }
    fn dequant_bf16(&self, _gpu: &dyn GpuBackend, out: DevicePtr, _stream: u64) -> Result<()> {
        self.record(Call::Dequant { out });
        Ok(())
    }
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}
