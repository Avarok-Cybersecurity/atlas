// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the test fixtures of the GDN W4 projection arm: a host
//! oracle `W4Linear`, BF16 helpers for the mock's device memory, GDN layers on
//! the mock backend (the loader's NULL-slot layout and a BF16 layout) with
//! sentinel projection handles, and a forward context.
//!
//! The fake reads its input rows from the mock's device memory, computes
//! `out[r][j] = bf16(x[r][(7 j) % k] * c_j)` row by row and writes the rows
//! back, so a test checks both WHICH calls a site makes (rows, pointers,
//! scratch) and that the rows landed where the layer reads them. The oracle
//! is row invariant by construction, as the contract demands of `decode`.
//! Every BF16 projection kernel a W4 layer must never launch is a sentinel
//! handle, so a stray launch is counted.

use std::sync::Arc;

use anyhow::{Result, ensure};
use avarok_core::config::ModelConfig;
use parking_lot::Mutex;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::plan::W4Role;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::layers::qwen3_ssm::Qwen3SsmLayer;
use crate::layers::qwen3_ssm::tests::native_fp8_gdn_layer;
use crate::port::w4::{W4Format, W4Linear, W4Scratch};
use crate::weight_map::DenseWeight;

pub(super) const DENSE_GEMV: KernelHandle = KernelHandle(0xD1);
pub(super) const DENSE_GEMM: KernelHandle = KernelHandle(0xD2);
pub(super) const DENSE_PIPE: KernelHandle = KernelHandle(0xD3);
pub(super) const DENSE_GEMV2: KernelHandle = KernelHandle(0xD4);
pub(super) const DEINTERLEAVE: KernelHandle = KernelHandle(0xD5);
const SCRATCH_PER_ROW: usize = 256;

pub(super) fn to_bf16(f: f32) -> u16 {
    half::bf16::from_f32(f).to_bits()
}

pub(super) fn from_bf16(h: u16) -> f32 {
    half::bf16::from_bits(h).to_f32()
}

pub(super) fn read_bf16(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

pub(super) fn write_bf16(gpu: &dyn GpuBackend, p: DevicePtr, v: &[u16]) -> Result<()> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    gpu.copy_h2d(&b, p)
}

/// Deterministic input rows: small multiples of 1/8, exact in BF16.
pub(super) fn fill_rows(gpu: &dyn GpuBackend, p: DevicePtr, rows: usize, k: usize) {
    let v: Vec<u16> = (0..rows * k)
        .map(|e| to_bf16(((e / k * 31 + e % k * 7) % 17) as f32 * 0.125 - 1.0))
        .collect();
    write_bf16(gpu, p, &v).unwrap();
}

/// One call the fake received.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Call {
    Decode {
        x: DevicePtr,
        rows: usize,
        out: DevicePtr,
        ws: DevicePtr,
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

/// The host-oracle `W4Linear`.
pub(super) struct HostW4 {
    n: usize,
    k: usize,
    max_rows: usize,
    accept_prefill: bool,
    compute: bool,
    calls: Mutex<Vec<Call>>,
}

impl HostW4 {
    pub(super) fn new(
        n: usize,
        k: usize,
        max_rows: usize,
        accept_prefill: bool,
        compute: bool,
    ) -> Arc<Self> {
        Arc::new(Self {
            n,
            k,
            max_rows,
            accept_prefill,
            compute,
            calls: Mutex::new(Vec::new()),
        })
    }

    pub(super) fn calls(&self) -> Vec<Call> {
        self.calls.lock().clone()
    }

    /// The expected output rows for the input rows `x` (row major, `k` wide).
    pub(super) fn oracle(&self, x: &[u16]) -> Vec<u16> {
        let rows = x.len() / self.k;
        let mut o = vec![0u16; rows * self.n];
        for r in 0..rows {
            for j in 0..self.n {
                let c = 1.0 + (j % 5) as f32 * 0.25;
                o[r * self.n + j] = to_bf16(from_bf16(x[r * self.k + (j * 7) % self.k]) * c);
            }
        }
        o
    }

    fn run_rows(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        rows: usize,
        out: DevicePtr,
    ) -> Result<()> {
        if self.compute {
            let o = self.oracle(&read_bf16(gpu, x, rows * self.k)?);
            write_bf16(gpu, out, &o)?;
        }
        Ok(())
    }
}

impl W4Linear for HostW4 {
    fn format(&self) -> W4Format {
        W4Format::Affine { bits: 4, group: 64 }
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
        SCRATCH_PER_ROW * max_rows.min(self.max_rows).max(1)
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
        ensure!(
            (1..=self.max_rows).contains(&rows),
            "fake: decode of {rows} rows"
        );
        ensure!(
            !s.ws.is_null() && s.ws_bytes >= self.scratch_bytes(rows),
            "fake: workspace {} B below {} B",
            s.ws_bytes,
            self.scratch_bytes(rows)
        );
        self.calls.lock().push(Call::Decode {
            x,
            rows,
            out,
            ws: s.ws,
        });
        self.run_rows(gpu, x, rows, out)
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
        ensure!(m > self.max_rows, "fake: prefill of {m} rows is a decode");
        ensure!(
            !s.xq.is_null() && !s.xs.is_null() && !s.tickets.is_null(),
            "fake: prefill needs xq, xs and tickets"
        );
        self.calls.lock().push(Call::Prefill { x, m, out });
        if !self.accept_prefill {
            return Ok(false);
        }
        self.run_rows(gpu, x, m, out)?;
        Ok(true)
    }

    fn dequant_bf16(&self, _gpu: &dyn GpuBackend, out: DevicePtr, _stream: u64) -> Result<()> {
        self.calls.lock().push(Call::Dequant { out });
        Ok(())
    }
}

/// The 80B GDN shapes (hidden 2048, value 4096, qkvz 12288), so `out_proj`'s
/// N and K differ and a swapped pair is visible.
pub(super) fn config() -> ModelConfig {
    ModelConfig::qwen3_next_80b_nvfp4()
}

/// Sentinel BF16 projection and deinterleave handles on `layer`.
fn sentinels(layer: &mut Qwen3SsmLayer) {
    layer.dense_gemv_k = DENSE_GEMV;
    layer.dense_gemm_k = DENSE_GEMM;
    layer.dense_gemm_pipelined_k = DENSE_PIPE;
    layer.dense_gemv_batch2_k = DENSE_GEMV2;
    layer.deinterleave_k = DEINTERLEAVE;
}

/// A GDN layer as the W4 loader routes build it: dense QKVZ slot NULL,
/// `ssm.out_proj` null, no FP8, NVFP4 or `out_proj_dense` copy.
pub(super) fn loader_layer(gpu: &MockGpuBackend, c: &ModelConfig) -> Qwen3SsmLayer {
    let mut layer = native_fp8_gdn_layer(gpu, c, false, false);
    sentinels(&mut layer);
    layer
}

/// A BF16-only GDN layer (tiny placeholder BF16 slots: a W4 layer never reads
/// them; the row-off tests only count launches).
pub(super) fn bf16_layer(gpu: &MockGpuBackend, c: &ModelConfig) -> Qwen3SsmLayer {
    let mut layer = loader_layer(gpu, c);
    layer.ssm.in_proj_qkvz = DenseWeight {
        weight: gpu.alloc(64).unwrap(),
    };
    layer.out_proj_dense = Some(DenseWeight {
        weight: gpu.alloc(64).unwrap(),
    });
    layer
}

/// [`loader_layer`] with a fake W4 pair installed (16 decode rows).
pub(super) fn w4_layer(
    gpu: &MockGpuBackend,
    c: &ModelConfig,
    accept: bool,
    compute: bool,
) -> (Qwen3SsmLayer, Arc<HostW4>, Arc<HostW4>) {
    let mut layer = loader_layer(gpu, c);
    let ((qn, qk), (on, ok)) = (W4Role::Qkvz.dims(c), W4Role::OutProj.dims(c));
    let q = HostW4::new(qn, qk, 16, accept, compute);
    let o = HostW4::new(on, ok, 16, accept, compute);
    assert_eq!(
        layer.set_w4_proj(q.clone(), o.clone()),
        0,
        "NULL slots free nothing"
    );
    (layer, q, o)
}

/// Launches of handle `k` so far.
pub(super) fn launches(gpu: &MockGpuBackend, k: KernelHandle) -> usize {
    gpu.launches_snapshot()
        .iter()
        .filter(|l| l.func == k.0)
        .count()
}

/// Launches of any BF16 projection kernel so far.
pub(super) fn bf16_launches(gpu: &MockGpuBackend) -> usize {
    [DENSE_GEMV, DENSE_GEMM, DENSE_PIPE, DENSE_GEMV2]
        .iter()
        .map(|&k| launches(gpu, k))
        .sum()
}

/// A forward context over `buffers`, run through `f`.
pub(super) fn with_ctx<R>(
    gpu: &MockGpuBackend,
    c: &ModelConfig,
    buffers: &BufferArena,
    decode_step: bool,
    graph_capture: bool,
    f: impl FnOnce(&ForwardContext) -> R,
) -> R {
    let dispatch = ops::GemmDispatch::defaults();
    let derived = ops::DerivedWeights::new();
    let levers = ops::ModelLevers::defaults();
    let stats = ops::ModelStats::new();
    let ctx = ForwardContext {
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        buffers,
        hc_row_offset: 0,
        gpu,
        config: c,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture,
        decode_step,
        gdn_exact_replay: false,
        gdn_write_on_accept: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: crate::layer::MoeLoraRoute::Fold,
    };
    f(&ctx)
}

/// `(rows, x, out)` of each decode call in `calls`, in order.
pub(super) fn decodes(calls: &[Call]) -> Vec<(usize, DevicePtr, DevicePtr)> {
    calls
        .iter()
        .filter_map(|c| match *c {
            Call::Decode { rows, x, out, .. } => Some((rows, x, out)),
            _ => None,
        })
        .collect()
}

/// The workspace pointers of the decode calls in `calls`, in order.
pub(super) fn decode_ws(calls: &[Call]) -> Vec<DevicePtr> {
    calls
        .iter()
        .filter_map(|c| match *c {
            Call::Decode { ws, .. } => Some(ws),
            _ => None,
        })
        .collect()
}
