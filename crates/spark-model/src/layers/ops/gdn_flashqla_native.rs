// SPDX-License-Identifier: AGPL-3.0-only

//! Atlas-native FlashQLA launcher.
//!
//! The CUDA device code lives in the `flashqla_gdn` atlas-kernels module. This
//! file intentionally contains only launch metadata and workspace management;
//! it has no TVM/TileLang FFI and does not load a shared object. The native
//! path is opt-in during the migration (`ATLAS_QLA_IMPL=native`) and the shim
//! remains available as an A/B reference until the native path is validated.

use anyhow::{Result, anyhow, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle, TensorMapSpec};
use std::sync::{Mutex, OnceLock};

const CHUNK: u32 = 32;
const BF16_BYTES: usize = 2;
const FP32_BYTES: usize = 4;
const TMA_BF16: u32 = 9;
const TMA_SWIZZLE_128B: u32 = 3;
const TMA_SWIZZLE_64B: u32 = 2;
const TMA_SWIZZLE_NONE: u32 = 0;
const TMA_L2_128B: u32 = 2;

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct NativeKernels {
    pub unpack_gate_beta: KernelHandle,
    pub chunk_local_cumsum: KernelHandle,
    pub kkt_solve: KernelHandle,
    pub fused_nocp: KernelHandle,
    pub fused_nocp_packed_strided: KernelHandle,
    pub cp_warmup: KernelHandle,
    pub cp_prepare_h: KernelHandle,
    pub cp_correct_h0: KernelHandle,
    pub kkt_packed_strided: KernelHandle,
    pub cp_prepare_h_packed_strided: KernelHandle,
    pub fused_cp_packed_strided: KernelHandle,
    pub fused_cp_qkg_pair: KernelHandle,
}

#[derive(Clone, Copy)]
struct BaseTensorMaps {
    a: spark_runtime::gpu::TensorMapDescriptor,
    k: spark_runtime::gpu::TensorMapDescriptor,
    q: spark_runtime::gpu::TensorMapDescriptor,
    v: spark_runtime::gpu::TensorMapDescriptor,
    /// V descriptor for `prepare_h`, whose V tile is a full 128-element dim
    /// column (`box=[128,1,32,1]`, no swizzle).  The shared-memory V layout of
    /// the fused CP kernel instead uses `box=[64,1,32,1]` with 128-byte
    /// swizzle, so the two kernels must not share one V map.
    v_prepare: spark_runtime::gpu::TensorMapDescriptor,
    o: spark_runtime::gpu::TensorMapDescriptor,
}

#[derive(Clone, Copy)]
struct BaseTensorMapCache {
    qkv: DevicePtr,
    output: DevicePtr,
    total: usize,
    maps: BaseTensorMaps,
}

#[derive(Clone, Copy)]
struct CpTensorMapCache {
    cp_batch: usize,
    cp_chunks: usize,
    maps: CpTensorMaps,
}

#[derive(Clone, Copy)]
struct CpTensorMaps {
    h: spark_runtime::gpu::TensorMapDescriptor,
    ht: spark_runtime::gpu::TensorMapDescriptor,
    mt: spark_runtime::gpu::TensorMapDescriptor,
}

impl NativeKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        let module = "flashqla_gdn";
        Ok(Self {
            unpack_gate_beta: gpu.kernel(module, "flashqla_unpack_gate_beta")?,
            chunk_local_cumsum: gpu.kernel(module, "flashqla_chunk_local_cumsum")?,
            kkt_solve: gpu.kernel(module, "flashqla_kkt_solve")?,
            fused_nocp: gpu.kernel(module, "flashqla_fused_nocp")?,
            fused_nocp_packed_strided: gpu.kernel(module, "flashqla_fused_nocp_packed_strided")?,
            cp_warmup: gpu.kernel(module, "flashqla_cp_warmup")?,
            cp_prepare_h: gpu.kernel(module, "flashqla_prepare_h_packed_strided")?,
            cp_correct_h0: gpu.kernel(module, "flashqla_cp_correct_h0")?,
            kkt_packed_strided: gpu.kernel(module, "flashqla_kkt_packed_strided")?,
            cp_prepare_h_packed_strided: gpu.kernel(module, "flashqla_prepare_h_packed_strided")?,
            fused_cp_packed_strided: gpu.kernel(module, "flashqla_fused_cp_packed_strided")?,
            fused_cp_qkg_pair: gpu.kernel(module, "flashqla_fused_cp_packed_strided_qkg_pair")?,
        })
    }
}

static NATIVE: OnceLock<Option<NativeKernels>> = OnceLock::new();

pub fn initialize(gpu: &dyn GpuBackend) -> bool {
    if std::env::var("ATLAS_QLA_IMPL").as_deref() != Ok("native")
        || std::env::var("ATLAS_GDN_FLASHQLA").as_deref() != Ok("1")
    {
        return false;
    }
    NATIVE
        .get_or_init(|| match NativeKernels::load(gpu) {
            Ok(k) => {
                // The generated kernels use more than the legacy 48-KiB
                // dynamic shared-memory default. Set the attribute once at
                // model initialization; failure makes the implementation
                // unavailable rather than surfacing an asynchronous launch
                // error in a request.
                for (handle, bytes) in [
                    (k.chunk_local_cumsum, 8320),
                    (k.kkt_solve, 15552),
                    (k.kkt_packed_strided, 15552),
                    (k.fused_nocp, 75776),
                    (k.fused_nocp_packed_strided, 75776),
                    (k.fused_cp_packed_strided, 75776),
                    (k.fused_cp_qkg_pair, 75776),
                    (k.cp_prepare_h_packed_strided, 94208),
                    (k.cp_correct_h0, 90112),
                ] {
                    if let Err(e) = gpu.set_kernel_attribute(handle, 8, bytes) {
                        tracing::warn!(
                            "ATLAS_GDN_FLASHQLA: native shared-memory attribute setup failed; falling back to FLA: {e}"
                        );
                        return None;
                    }
                }
                // Probe the driver TMA entry point during initialization, before
                // Phase 1 can write a log-space gate.  The real descriptors are
                // built after Atlas allocates the request buffers, but this
                // representative rank-4 BF16 map catches missing/unsupported
                // `cuTensorMapEncodeTiled` implementations early and routes
                // the whole request back to FLA instead of failing mid-prefill.
                // The descriptor covers 32 token rows at the packed 8192-wide
                // pitch; allocate the complete logical span so CUDA's encoder
                // can validate the address/range, not just pointer alignment.
                let probe = match gpu.alloc(1 << 20) {
                    Ok(ptr) => ptr,
                    Err(e) => {
                        tracing::warn!(
                            "ATLAS_GDN_FLASHQLA: native TMA probe allocation failed; falling back to FLA: {e}"
                        );
                        return None;
                    }
                };
                let probe_result = gpu.encode_tensor_map(&TensorMapSpec {
                    dtype: TMA_BF16,
                    rank: 4,
                    global_address: probe,
                    global_dims: [128, 16, 32, 1, 1],
                    global_strides: [2, 256, 4096, 131_072, 0],
                    box_dims: [64, 1, 32, 1, 1],
                    element_strides: [1; 5],
                    interleave: 0,
                    swizzle: TMA_SWIZZLE_128B,
                    l2_promotion: TMA_L2_128B,
                    oob_fill: 0,
                });
                let free_result = gpu.free(probe);
                if let Err(e) = probe_result {
                    tracing::warn!(
                        "ATLAS_GDN_FLASHQLA: native TMA descriptor probe failed; falling back to FLA: {e}"
                    );
                    return None;
                }
                if let Err(e) = free_result {
                    tracing::warn!(
                        "ATLAS_GDN_FLASHQLA: native TMA probe cleanup failed; falling back to FLA: {e}"
                    );
                    return None;
                }
                tracing::info!(
                    "ATLAS_GDN_FLASHQLA: native flashqla_gdn kernels loaded (no TVM/shim)"
                );
                Some(k)
            }
            Err(e) => {
                tracing::warn!(
                    "ATLAS_GDN_FLASHQLA: native kernel initialization failed; falling back to FLA: {e}"
                );
                None
            }
        })
        .is_some()
}

pub fn available() -> bool {
    std::env::var("ATLAS_QLA_IMPL").as_deref() == Ok("native")
        && NATIVE.get().and_then(|v| *v).is_some()
}

fn kernels() -> Result<NativeKernels> {
    NATIVE
        .get()
        .and_then(|v| *v)
        .ok_or_else(|| anyhow!("native FlashQLA runtime is not initialized"))
}

struct Workspace {
    gate: DevicePtr,
    beta: DevicePtr,
    g_cumsum: DevicePtr,
    a: DevicePtr,
    h: DevicePtr,
    cap_tokens: usize,
    cap_chunks: usize,
    chunk_indices: DevicePtr,
    cu_seqlens: DevicePtr,
    chunk_offsets: DevicePtr,
    cp_seq_map: DevicePtr,
    cp_cu: DevicePtr,
    cp_c2r: DevicePtr,
    cp_r2c: DevicePtr,
    cp_offsets: DevicePtr,
    cp_ht_mask: DevicePtr,
    cp_warmup: DevicePtr,
    cp_fallback: DevicePtr,
    cp_h0: DevicePtr,
    cp_prep_h0: DevicePtr,
    cp_ht: DevicePtr,
    cp_mt: DevicePtr,
    cap_cp_batch: usize,
    meta_total: usize,
    meta_chunks: usize,
    cp_meta_total: usize,
    cp_meta_batch: usize,
    cp_meta_chunks: usize,
    cp_zeroed: bool,
    base_maps: Option<BaseTensorMapCache>,
    cp_maps: Option<CpTensorMapCache>,
}

impl Workspace {
    fn empty() -> Self {
        Self {
            gate: DevicePtr::NULL,
            beta: DevicePtr::NULL,
            g_cumsum: DevicePtr::NULL,
            a: DevicePtr::NULL,
            h: DevicePtr::NULL,
            cap_tokens: 0,
            cap_chunks: 0,
            chunk_indices: DevicePtr::NULL,
            cu_seqlens: DevicePtr::NULL,
            chunk_offsets: DevicePtr::NULL,
            cp_seq_map: DevicePtr::NULL,
            cp_cu: DevicePtr::NULL,
            cp_c2r: DevicePtr::NULL,
            cp_r2c: DevicePtr::NULL,
            cp_offsets: DevicePtr::NULL,
            cp_ht_mask: DevicePtr::NULL,
            cp_warmup: DevicePtr::NULL,
            cp_fallback: DevicePtr::NULL,
            cp_h0: DevicePtr::NULL,
            cp_prep_h0: DevicePtr::NULL,
            cp_ht: DevicePtr::NULL,
            cp_mt: DevicePtr::NULL,
            cap_cp_batch: 0,
            meta_total: 0,
            meta_chunks: 0,
            cp_meta_total: 0,
            cp_meta_batch: 0,
            cp_meta_chunks: 0,
            cp_zeroed: false,
            base_maps: None,
            cp_maps: None,
        }
    }

    fn ensure(&mut self, gpu: &dyn GpuBackend, total: usize, nv: usize) -> Result<()> {
        let chunks = total.div_ceil(CHUNK as usize);
        if total > self.cap_tokens {
            self.gate = gpu.alloc(total * nv * FP32_BYTES)?;
            self.beta = gpu.alloc(total * nv * FP32_BYTES)?;
            self.g_cumsum = gpu.alloc(total * nv * FP32_BYTES)?;
            self.a = gpu.alloc(total * nv * CHUNK as usize * BF16_BYTES * 2)?;
            self.cap_tokens = total;
            self.base_maps = None;
        }
        if chunks > self.cap_chunks {
            self.h = gpu.alloc(chunks * nv * 128 * 128 * BF16_BYTES)?;
            self.chunk_indices = gpu.alloc(chunks * 2 * std::mem::size_of::<i64>())?;
            self.chunk_offsets = gpu.alloc(2 * std::mem::size_of::<i64>())?;
            self.cap_chunks = chunks;
            self.base_maps = None;
            self.cp_maps = None;
            // The metadata buffers may have been replaced, so force the next
            // invocation to repopulate them even when the requested shape is
            // unchanged from the caller's perspective.
            self.meta_total = 0;
            self.meta_chunks = 0;
        }
        if self.cu_seqlens.is_null() {
            self.cu_seqlens = gpu.alloc(2 * std::mem::size_of::<i64>())?;
            self.cp_seq_map = gpu.alloc(std::mem::size_of::<i64>())?;
        }
        Ok(())
    }

    fn ensure_cp(
        &mut self,
        gpu: &dyn GpuBackend,
        cp_batch: usize,
        cp_chunks: usize,
        nv: usize,
    ) -> Result<()> {
        if cp_batch <= self.cap_cp_batch && cp_chunks <= self.cap_chunks {
            return Ok(());
        }
        self.cp_cu = gpu.alloc((cp_batch + 1) * std::mem::size_of::<i64>())?;
        self.cp_c2r = gpu.alloc(cp_batch * std::mem::size_of::<i64>())?;
        self.cp_r2c = gpu.alloc(2 * std::mem::size_of::<i64>())?;
        self.cp_offsets = gpu.alloc((cp_batch + 1) * std::mem::size_of::<i64>())?;
        self.cp_ht_mask = gpu.alloc(cp_batch)?;
        self.cp_warmup = gpu.alloc(cp_batch * nv * std::mem::size_of::<i64>())?;
        self.cp_fallback = gpu.alloc(cp_batch * nv)?;
        let state = cp_batch * nv * 128 * 128;
        self.cp_h0 = gpu.alloc(state * FP32_BYTES)?;
        self.cp_prep_h0 = gpu.alloc(state * FP32_BYTES)?;
        // The frozen prepare_h writes ht/mt at segment*524288 + head*16384 with
        // an inner offset that can exceed the 16384-element head block (the
        // generated store spreads a 128x128 tile across the register
        // fragment, and the tail-iteration partial writes reach past the
        // nominal block boundary).  Pad ht/mt with a full head-block of slack
        // so the last segment's last head never walks past the allocation.
        let state_span = state + nv * 128 * 128;
        self.cp_ht = gpu.alloc(state_span * BF16_BYTES)?;
        self.cp_mt = gpu.alloc(state_span * BF16_BYTES)?;
        self.cap_cp_batch = cp_batch;
        self.cp_meta_total = 0;
        self.cp_meta_batch = 0;
        self.cp_meta_chunks = 0;
        self.cp_zeroed = false;
        self.cp_maps = None;
        Ok(())
    }
}

static WORKSPACE: OnceLock<Mutex<Workspace>> = OnceLock::new();

fn ws() -> &'static Mutex<Workspace> {
    WORKSPACE.get_or_init(|| Mutex::new(Workspace::empty()))
}

fn upload_metadata(
    gpu: &dyn GpuBackend,
    w: &mut Workspace,
    total: usize,
    chunks: usize,
    stream: u64,
) -> Result<()> {
    if w.meta_total == total && w.meta_chunks == chunks {
        return Ok(());
    }
    let mut ci = Vec::with_capacity(chunks * 2);
    for i in 0..chunks {
        ci.extend_from_slice(&[0i64, i as i64]);
    }
    let cu = [0i64, total as i64];
    let co = [0i64, chunks as i64];
    let csm = [0i64];
    gpu.copy_h2d_async(bytemuck_i64(&ci), w.chunk_indices, stream)?;
    gpu.copy_h2d_async(bytemuck_i64(&cu), w.cu_seqlens, stream)?;
    gpu.copy_h2d_async(bytemuck_i64(&co), w.chunk_offsets, stream)?;
    gpu.copy_h2d_async(bytemuck_i64(&csm), w.cp_seq_map, stream)?;
    w.meta_total = total;
    w.meta_chunks = chunks;
    Ok(())
}

fn bytemuck_i64(values: &[i64]) -> &[u8] {
    // i64 has no padding and the slice is immutable for the duration of the
    // async copy (the caller synchronizes before returning).
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn bytemuck_i8(values: &[i8]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len()) }
}

fn map_spec(
    ptr: DevicePtr,
    dims: [u64; 4],
    strides: [u64; 4],
    box_dims: [u32; 4],
    swizzle: u32,
) -> TensorMapSpec {
    TensorMapSpec {
        dtype: TMA_BF16,
        rank: 4,
        global_address: ptr,
        global_dims: [dims[0], dims[1], dims[2], dims[3], 1],
        global_strides: [strides[0], strides[1], strides[2], strides[3], 0],
        box_dims: [box_dims[0], box_dims[1], box_dims[2], box_dims[3], 1],
        element_strides: [1, 1, 1, 1, 1],
        interleave: 0,
        swizzle,
        l2_promotion: TMA_L2_128B,
        oob_fill: 0,
    }
}

fn map_spec5(
    ptr: DevicePtr,
    dims: [u64; 5],
    strides: [u64; 5],
    box_dims: [u32; 5],
) -> TensorMapSpec {
    TensorMapSpec {
        dtype: TMA_BF16,
        rank: 5,
        global_address: ptr,
        global_dims: dims,
        global_strides: strides,
        box_dims,
        element_strides: [1; 5],
        interleave: 0,
        swizzle: TMA_SWIZZLE_128B,
        l2_promotion: TMA_L2_128B,
        oob_fill: 0,
    }
}

/// A per-segment state tile descriptor shared by prepare_h and correct_h0.
///
/// `ht`/`mt` are physical `[cp_batch, 32, 128, 128]` BF16 buffers where each
/// segment occupies a fixed contiguous `1_048_576`-byte span (`32*128*128*2`).
/// The TileLang kernels address them as `[vd][kd][nv][segment]` (TMA coord
/// order d0..d3), so the descriptor's 4th-dim stride is the *fixed* segment
/// span.  Passing `cp_batch * 1_048_576` here indexed segment `s` at
/// `s * cp_batch * 1MB` — out of bounds for `cp_batch > 1`, which produced
/// garbage multi-segment CP output while the single-segment case looked fine.
///
/// `box_dims` is the per-load tile: `ht` uses `[32, 128, 1, 1]` (a 32-row vd
/// slice), `mt` uses `[64, 128, 1, 1]` (two 64-row vd slices).  `swizzle` is
/// per-buffer: correct_h0 reads `ht` back from shared memory with plain linear
/// `uint1` indexing (a swizzled TMA load would permute those reads), while
/// `mt` is consumed via `ldmatrix`, which requires the 128-byte swizzle.
fn state_tile_map(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    cp_batch: usize,
    nv: usize,
    box_dims: [u32; 4],
    swizzle: u32,
) -> Result<spark_runtime::gpu::TensorMapDescriptor> {
    gpu.encode_tensor_map(&map_spec(
        ptr,
        [128, 128, nv as u64, cp_batch as u64],
        [2, 256, 32768, 1_048_576],
        box_dims,
        swizzle,
    ))
}

fn i32_bytes(v: i32) -> [u8; 4] {
    v.to_le_bytes()
}

/// CP segment size in 32-token chunks, matching FlashQLA's `_calc_cp_seqs`
/// oracle on GB10 (48 SMs): `max(4, 2^round(log2(sqrt(nv·chunks/48)·3)))`.
/// `round` is ties-to-even (banker's) to match Python's `round()`.
fn cp_local_chunks(nv: usize, chunks: usize) -> usize {
    let raw = ((nv as f64 * chunks as f64 / 48.0).sqrt() * 3.0).log2();
    let base = if raw.is_finite() {
        (2.0_f64).powf(raw.round_ties_even()) as usize
    } else {
        0
    };
    base.max(4)
}

fn debug_stage_sync(gpu: &dyn GpuBackend, stream: u64, stage: &str) -> Result<()> {
    if std::env::var("ATLAS_DEBUG_SYNC_KERNELS").as_deref() == Ok("1") {
        gpu.synchronize(stream)?;
        tracing::info!("ATLAS_GDN_FLASHQLA: native stage {stage} complete");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn prefill(
    gpu: &dyn GpuBackend,
    qkv: DevicePtr,
    gate_beta: DevicePtr,
    output: DevicePtr,
    h_state: DevicePtr,
    _scale: f32,
    total: u32,
    nk: u32,
    nv: u32,
    kd: u32,
    vd: u32,
    conv_dim: u32,
    gb_stride: u32,
    num_seqs: u32,
    stream: u64,
) -> Result<()> {
    if num_seqs != 1 || nk != 16 || nv != 32 || kd != 128 || vd != 128 || conv_dim != 8192 {
        bail!(
            "native FlashQLA supports only single-sequence Holo shape (nk=16,nv=32,kd=vd=128,conv_dim=8192)"
        );
    }
    let k = kernels()?;
    let total = total as usize;
    let nv = nv as usize;
    let chunks = total.div_ceil(CHUNK as usize);
    let mut guard = ws()
        .lock()
        .map_err(|_| anyhow!("FlashQLA workspace mutex poisoned"))?;
    guard.ensure(gpu, total, nv)?;
    upload_metadata(gpu, &mut guard, total, chunks, stream)?;

    let unpack_n = total * nv;
    let unpack_total = i32_bytes(total as i32);
    let unpack_nv = i32_bytes(nv as i32);
    let unpack_stride = i32_bytes(gb_stride as i32);
    let input_log = i32_bytes(1);
    gpu.launch_typed(
        k.unpack_gate_beta,
        [unpack_n.div_ceil(256) as u32, 1, 1],
        [256, 1, 1],
        0,
        stream,
        &[
            KernelArg::Buffer(gate_beta),
            KernelArg::Buffer(guard.gate),
            KernelArg::Buffer(guard.beta),
            KernelArg::Bytes(&unpack_total),
            KernelArg::Bytes(&unpack_nv),
            KernelArg::Bytes(&unpack_stride),
            KernelArg::Bytes(&input_log),
        ],
    )?;
    debug_stage_sync(gpu, stream, "unpack_gate_beta")?;

    let data_batch = i32_bytes(1);
    let n_chunks = i32_bytes(chunks as i32);
    let n_tokens = i32_bytes(total as i32);
    let real_batch = i32_bytes(1);
    gpu.launch_typed(
        k.chunk_local_cumsum,
        [chunks as u32, 1, 1],
        [128, 1, 1],
        8320,
        stream,
        &[
            KernelArg::Buffer(guard.chunk_indices),
            KernelArg::Buffer(guard.cu_seqlens),
            KernelArg::Buffer(guard.g_cumsum),
            KernelArg::Buffer(guard.gate),
            KernelArg::Bytes(&data_batch),
            KernelArg::Bytes(&n_chunks),
            KernelArg::Bytes(&n_tokens),
            KernelArg::Bytes(&real_batch),
        ],
    )?;
    debug_stage_sync(gpu, stream, "chunk_local_cumsum")?;

    gpu.launch_typed(
        // The packed-strided KKT variant has the same output contract as the
        // contiguous no-CP KKT but reads Atlas's fixed `[Q|K|V]` token pitch
        // directly. This avoids silently treating the 8192-wide row as a
        // contiguous 2048-wide K tensor.
        k.kkt_packed_strided,
        // The generated kernel uses blockIdx.x for the 32 K/V head tiles and
        // blockIdx.y for the chunk metadata row.  Keep this two-dimensional
        // launch: flattening the product into x makes the kernel read the
        // wrong chunk row (and can leave its TMA barriers waiting forever).
        [32, chunks as u32, 1],
        [256, 1, 1],
        15552,
        stream,
        &[
            KernelArg::Buffer(guard.a),
            KernelArg::Buffer(guard.beta),
            KernelArg::Buffer(guard.chunk_indices),
            KernelArg::Buffer(guard.cu_seqlens),
            KernelArg::Buffer(qkv.offset((nk * kd * BF16_BYTES as u32) as usize)),
            KernelArg::Bytes(&data_batch),
            KernelArg::Bytes(&n_chunks),
            KernelArg::Bytes(&n_tokens),
            KernelArg::Bytes(&real_batch),
        ],
    )?;
    debug_stage_sync(gpu, stream, "kkt")?;

    let k_ptr = qkv.offset(nk as usize * kd as usize * BF16_BYTES);
    let v_ptr = qkv.offset(2 * nk as usize * kd as usize * BF16_BYTES);
    let base_maps = if let Some(cache) = guard.base_maps {
        if cache.qkv == qkv && cache.output == output && cache.total == total {
            cache.maps
        } else {
            let maps = BaseTensorMaps {
                a: gpu.encode_tensor_map(&map_spec(
                    guard.a,
                    [32, 32, total as u64, 1],
                    [2, 64, 2048, (total * 2048) as u64],
                    [32, 1, 32, 1],
                    // The `a` (KKT) tile is [32,32,total] bf16 with a 2048-byte
                    // token row; the TileLang fused kernels stage it with a 64B
                    // swizzle, so 128B here reads the wrong element offsets on
                    // the TMA path (full chunks, T >= 32).
                    TMA_SWIZZLE_64B,
                ))?,
                k: gpu.encode_tensor_map(&map_spec(
                    k_ptr,
                    [128, nk as u64, total as u64, 1],
                    // Packed QKV keeps a single 8192-element token pitch;
                    // the logical K view therefore advances by 16384 bytes
                    // per token (not the 4096-byte contiguous-K pitch).
                    [2, 256, 16384, (total * 16384) as u64],
                    [64, 1, 32, 1],
                    TMA_SWIZZLE_128B,
                ))?,
                q: gpu.encode_tensor_map(&map_spec(
                    qkv,
                    [128, nk as u64, total as u64, 1],
                    [2, 256, 16384, (total * 16384) as u64],
                    [64, 1, 32, 1],
                    TMA_SWIZZLE_128B,
                ))?,
                v: gpu.encode_tensor_map(&map_spec(
                    v_ptr,
                    [128, nv as u64, total as u64, 1],
                    [2, 256, 16384, (total * 16384) as u64],
                    [64, 1, 32, 1],
                    TMA_SWIZZLE_128B,
                ))?,
                // prepare_h's V load is a full 128-element dim column per
                // (head, token); the shared-memory layout has no swizzle, so
                // this map uses box[128,1,32,1] with SWIZZLE_NONE.  The fused
                // CP kernel's V map (`v`, above) keeps box[64,1,32,1] with
                // 128-byte swizzle to match its shared-memory stage.
                v_prepare: gpu.encode_tensor_map(&map_spec(
                    v_ptr,
                    [128, nv as u64, total as u64, 1],
                    [2, 256, 16384, (total * 16384) as u64],
                    [128, 1, 32, 1],
                    TMA_SWIZZLE_NONE,
                ))?,
                o: gpu.encode_tensor_map(&map_spec(
                    output,
                    [128, nv as u64, total as u64, 1],
                    [2, 256, 8192, (total * 8192) as u64],
                    [64, 1, 32, 1],
                    TMA_SWIZZLE_128B,
                ))?,
            };
            guard.base_maps = Some(BaseTensorMapCache {
                qkv,
                output,
                total,
                maps,
            });
            maps
        }
    } else {
        let maps = BaseTensorMaps {
            a: gpu.encode_tensor_map(&map_spec(
                guard.a,
                [32, 32, total as u64, 1],
                [2, 64, 2048, (total * 2048) as u64],
                [32, 1, 32, 1],
                // 64B swizzle to match the TileLang `a` staging (see the
                // fused_cp a_desc comment above).
                TMA_SWIZZLE_64B,
            ))?,
            k: gpu.encode_tensor_map(&map_spec(
                k_ptr,
                [128, nk as u64, total as u64, 1],
                [2, 256, 16384, (total * 16384) as u64],
                [64, 1, 32, 1],
                TMA_SWIZZLE_128B,
            ))?,
            q: gpu.encode_tensor_map(&map_spec(
                qkv,
                [128, nk as u64, total as u64, 1],
                [2, 256, 16384, (total * 16384) as u64],
                [64, 1, 32, 1],
                TMA_SWIZZLE_128B,
            ))?,
            v: gpu.encode_tensor_map(&map_spec(
                v_ptr,
                [128, nv as u64, total as u64, 1],
                [2, 256, 16384, (total * 16384) as u64],
                [64, 1, 32, 1],
                TMA_SWIZZLE_128B,
            ))?,
            v_prepare: gpu.encode_tensor_map(&map_spec(
                v_ptr,
                [128, nv as u64, total as u64, 1],
                [2, 256, 16384, (total * 16384) as u64],
                [128, 1, 32, 1],
                TMA_SWIZZLE_NONE,
            ))?,
            o: gpu.encode_tensor_map(&map_spec(
                output,
                [128, nv as u64, total as u64, 1],
                [2, 256, 8192, (total * 8192) as u64],
                [64, 1, 32, 1],
                TMA_SWIZZLE_128B,
            ))?,
        };
        guard.base_maps = Some(BaseTensorMapCache {
            qkv,
            output,
            total,
            maps,
        });
        maps
    };
    let a_desc = base_maps.a;
    let k_desc = base_maps.k;
    let q_desc = base_maps.q;
    let v_desc = base_maps.v;
    let o_desc = base_maps.o;

    // Multi-segment CP is numerically validated end-to-end (cp_batch 1..4,
    // T up to 8192): ht/mt/cp_h0/output/final-state match the FlashQLA
    // reference exactly (see result/native_source_multi_cp).  The `T<=512`
    // correctness-first gate is removed; CP still requires the opt-in
    // `ATLAS_QLA_AUTO_CP=1` (native remains an A/B path, default is shim).
    if std::env::var("ATLAS_QLA_AUTO_CP").as_deref() == Ok("1") {
        // Single-sequence SM120 CP partitioning, matching the validated shim
        // (`_calc_cp_seqs` in flash_qla cp_context): max_local_chunks =
        // 2^round(log2(sqrt(H·chunks/P)·3)) with P = SM count (48 on GB10),
        // floored at 4.  `round` uses banker's rounding (ties-to-even) to match
        // Python's `round()`.  A floor()+next_power_of_two() formula rounds UP
        // and picks a coarser partition than the reference (e.g. T=2048:
        // 32 chunks/segment vs the shim's 16), hurting multi-segment parallelism.
        let mut local_chunks = cp_local_chunks(nv, chunks);
        // Diagnostic-only override for a minimal non-tail TMA reproducer. It
        // is never set by production startup; keeping it here avoids changing
        // the validated CP partitioning while allowing a 1-chunk segment to
        // exercise exactly one producer/consumer iteration.
        let debug_local_chunks = std::env::var("ATLAS_DEBUG_CP_LOCAL_CHUNKS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0);
        if let Some(v) = debug_local_chunks {
            local_chunks = v;
        }
        if local_chunks == 0 {
            local_chunks = 4;
        }
        let local_tokens = local_chunks * CHUNK as usize;
        let mut cp_cu = vec![0i64];
        let mut cp_mask = Vec::new();
        while *cp_cu.last().unwrap() < total as i64 {
            let end = (*cp_cu.last().unwrap() + local_tokens as i64).min(total as i64);
            cp_cu.push(end);
            cp_mask.push((end == total as i64) as i8);
        }
        let cp_batch = cp_mask.len();
        let mut cp_c2r = vec![0i64; cp_batch];
        // r2c maps the raw sequence 0 onto the CP range [0, cp_batch]: the
        // correct_h0 kernel sets seq_start=r2c[0]=0, seq_end=r2c[1]=cp_batch,
        // num_iters=cp_batch, and writes cp_h0[seq_start+num_iters-1] as the
        // final state.  cp_batch+1 here would index one past the allocation.
        let cp_r2c = [0i64, cp_batch as i64];
        let mut cp_offsets = vec![0i64];
        for i in 0..cp_batch {
            let len = cp_cu[i + 1] - cp_cu[i];
            cp_offsets.push(cp_offsets[i] + (len as usize).div_ceil(CHUNK as usize) as i64);
            cp_c2r[i] = 0;
        }
        let cp_chunks = *cp_offsets.last().unwrap() as usize;
        guard.ensure_cp(gpu, cp_batch, cp_chunks, nv)?;
        if guard.cp_meta_total != total
            || guard.cp_meta_batch != cp_batch
            || guard.cp_meta_chunks != cp_chunks
        {
            gpu.copy_h2d_async(bytemuck_i64(&cp_cu), guard.cp_cu, stream)?;
            gpu.copy_h2d_async(bytemuck_i64(&cp_c2r), guard.cp_c2r, stream)?;
            gpu.copy_h2d_async(bytemuck_i64(&cp_r2c), guard.cp_r2c, stream)?;
            gpu.copy_h2d_async(bytemuck_i64(&cp_offsets), guard.cp_offsets, stream)?;
            gpu.copy_h2d_async(bytemuck_i8(&cp_mask), guard.cp_ht_mask, stream)?;
            guard.cp_meta_total = total;
            guard.cp_meta_batch = cp_batch;
            guard.cp_meta_chunks = cp_chunks;
        }
        if !guard.cp_zeroed {
            gpu.memset_async(
                guard.cp_prep_h0,
                0,
                guard.cap_cp_batch * nv * 128 * 128 * FP32_BYTES,
                stream,
            )?;
            guard.cp_zeroed = true;
        }

        let cp_batch_i = i32_bytes(cp_batch as i32);
        let total_i = i32_bytes(total as i32);
        gpu.launch_typed(
            k.cp_warmup,
            [cp_batch as u32, 1, 1],
            [32, 1, 1],
            0,
            stream,
            &[
                KernelArg::Buffer(guard.cp_cu),
                KernelArg::Buffer(guard.cp_fallback),
                KernelArg::Buffer(guard.g_cumsum),
                KernelArg::Buffer(guard.cp_ht_mask),
                KernelArg::Buffer(guard.cp_warmup),
                KernelArg::Bytes(&cp_batch_i),
                KernelArg::Bytes(&total_i),
            ],
        )?;
        debug_stage_sync(gpu, stream, "cp_warmup")?;
        if std::env::var("ATLAS_DEBUG_CP_META").as_deref() == Ok("1") {
            let mut wu = vec![0u8; cp_batch * nv * std::mem::size_of::<i64>()];
            gpu.copy_d2h_on_stream(guard.cp_warmup, &mut wu, stream)?;
            gpu.synchronize(stream)?;
            let vals: Vec<i64> = wu
                .chunks_exact(8)
                .map(|x| i64::from_le_bytes(x.try_into().unwrap()))
                .collect();
            tracing::info!(
                "ATLAS_GDN_FLASHQLA: cp metadata cp_cu={cp_cu:?} cp_offsets={cp_offsets:?} cp_batch={cp_batch} warmup={vals:?}"
            );
        }

        let cp_maps = if let Some(cache) = guard.cp_maps {
            if cache.cp_batch == cp_batch && cache.cp_chunks == cp_chunks {
                cache.maps
            } else {
                let maps = CpTensorMaps {
                    h: gpu.encode_tensor_map(&map_spec5(
                        guard.h,
                        [128, 128, nv as u64, cp_chunks as u64, 1],
                        [2, 256, 32768, 1_048_576, (cp_chunks * 1_048_576) as u64],
                        [64, 128, 1, 1, 1],
                    ))?,
                    // ht/mt are [cp_batch, nv, kd, vd] bf16 buffers written by
                    // prepare_h; correct_h0 loads them with a fixed per-segment
                    // stride of 1_048_576 bytes (see state_tile_map).
                    ht: state_tile_map(
                        gpu,
                        guard.cp_ht,
                        cp_batch,
                        nv,
                        [32, 128, 1, 1],
                        TMA_SWIZZLE_NONE,
                    )?,
                    mt: state_tile_map(
                        gpu,
                        guard.cp_mt,
                        cp_batch,
                        nv,
                        [64, 128, 1, 1],
                        TMA_SWIZZLE_128B,
                    )?,
                };
                guard.cp_maps = Some(CpTensorMapCache {
                    cp_batch,
                    cp_chunks,
                    maps,
                });
                maps
            }
        } else {
            let maps = CpTensorMaps {
                h: gpu.encode_tensor_map(&map_spec5(
                    guard.h,
                    [128, 128, nv as u64, cp_chunks as u64, 1],
                    [2, 256, 32768, 1_048_576, (cp_chunks * 1_048_576) as u64],
                    [64, 128, 1, 1, 1],
                ))?,
                ht: state_tile_map(
                    gpu,
                    guard.cp_ht,
                    cp_batch,
                    nv,
                    [32, 128, 1, 1],
                    TMA_SWIZZLE_NONE,
                )?,
                mt: state_tile_map(
                    gpu,
                    guard.cp_mt,
                    cp_batch,
                    nv,
                    [64, 128, 1, 1],
                    TMA_SWIZZLE_128B,
                )?,
            };
            guard.cp_maps = Some(CpTensorMapCache {
                cp_batch,
                cp_chunks,
                maps,
            });
            maps
        };
        let h_desc = cp_maps.h;
        let ht_desc = cp_maps.ht;
        let mt_desc = cp_maps.mt;
        gpu.launch_typed(
            k.cp_prepare_h_packed_strided,
            // x selects one of the 32 K/V heads; y selects the CP sequence.
            [32, cp_batch as u32, 1],
            [512, 1, 1],
            94208,
            stream,
            &[
                KernelArg::Buffer(guard.a),
                KernelArg::Bytes(&a_desc.bytes),
                KernelArg::Buffer(guard.beta),
                KernelArg::Buffer(guard.cp_offsets),
                KernelArg::Buffer(guard.cp_cu),
                KernelArg::Buffer(guard.g_cumsum),
                KernelArg::Buffer(guard.cp_prep_h0),
                KernelArg::Bytes(&h_desc.bytes),
                KernelArg::Buffer(guard.cp_ht),
                KernelArg::Buffer(k_ptr),
                KernelArg::Bytes(&k_desc.bytes),
                KernelArg::Buffer(guard.cp_mt),
                KernelArg::Buffer(guard.cp_warmup),
                KernelArg::Buffer(v_ptr),
                // prepare_h's V map is a separate descriptor
                // (box=[128,1,32,1], SWIZZLE_NONE) from the fused-CP V map.
                KernelArg::Bytes(&base_maps.v_prepare.bytes),
                KernelArg::Bytes(&cp_batch_i),
                KernelArg::Bytes(&total_i),
            ],
        )?;
        debug_stage_sync(gpu, stream, "cp_prepare_h")?;
        let cp_batch_i = i32_bytes(cp_batch as i32);
        let raw_batch_i = i32_bytes(1);
        gpu.launch_typed(
            k.cp_correct_h0,
            // The frozen correct_h0 kernel maps blockIdx.x ∈ [0,128) onto
            // (cp_seq via blockIdx.x>>7, head via blockIdx.x&127) and iterates
            // across every CP segment inside one block using seq_map_r2c
            // (r2c=[0, cp_batch] for the single raw sequence).  Launching with
            // cp_batch*128 blocks would index raw_h0 by (blockIdx.x>>2) past
            // its 32-head extent and fault.  The shim launches this kernel
            // with exactly 128 blocks.
            [128, 1, 1],
            [256, 1, 1],
            90112,
            stream,
            &[
                KernelArg::Buffer(guard.cp_h0),
                KernelArg::Buffer(guard.cp_fallback),
                KernelArg::Bytes(&ht_desc.bytes),
                KernelArg::Bytes(&mt_desc.bytes),
                KernelArg::Buffer(h_state),
                KernelArg::Buffer(guard.cp_r2c),
                KernelArg::Bytes(&cp_batch_i),
                KernelArg::Bytes(&raw_batch_i),
            ],
        )?;
        debug_stage_sync(gpu, stream, "cp_correct_h0")?;
        // A/B selector: qkg_pair fused (NCU-targeted L2 improvement) vs the
        // baseline fused.  Same grid/signature; blockIdx.x groups the 4 blocks
        // sharing one Q/K head.  Diagnostic-only; production default is the
        // baseline fused until the full acceptance gate passes.
        let fused_cp = if std::env::var("ATLAS_DEBUG_FLASHQLA_QKG_PAIR").as_deref() == Ok("1") {
            k.fused_cp_qkg_pair
        } else {
            k.fused_cp_packed_strided
        };
        gpu.launch_typed(
            fused_cp,
            // x contains 2 tiles per K/V head (64 total); y selects the CP
            // sequence.  The kernel indexes both dimensions independently.
            [64, cp_batch as u32, 1],
            [512, 1, 1],
            75776,
            stream,
            &[
                KernelArg::Buffer(guard.a),
                KernelArg::Bytes(&a_desc.bytes),
                KernelArg::Buffer(guard.beta),
                KernelArg::Buffer(guard.cp_offsets),
                KernelArg::Buffer(guard.cp_c2r),
                KernelArg::Buffer(guard.cp_cu),
                KernelArg::Buffer(guard.g_cumsum),
                KernelArg::Buffer(guard.cp_h0),
                KernelArg::Bytes(&h_desc.bytes),
                KernelArg::Buffer(h_state),
                KernelArg::Buffer(k_ptr),
                KernelArg::Bytes(&k_desc.bytes),
                KernelArg::Buffer(output),
                KernelArg::Bytes(&o_desc.bytes),
                KernelArg::Buffer(qkv),
                KernelArg::Bytes(&q_desc.bytes),
                KernelArg::Buffer(guard.cu_seqlens),
                KernelArg::Buffer(v_ptr),
                KernelArg::Bytes(&v_desc.bytes),
                KernelArg::Bytes(&cp_batch_i),
                KernelArg::Bytes(&total_i),
                KernelArg::Bytes(&raw_batch_i),
            ],
        )?;
        debug_stage_sync(gpu, stream, "fused_cp")?;
        gpu.synchronize(stream)?;
        return Ok(());
    }

    let batch_size = i32_bytes(1);
    gpu.launch_typed(
        // packed-strided no-CP fused: Q/K/V are logical strided views into the
        // packed 8192-element token row, so both the TMA path and the masked
        // tail reads use the correct pitch (fixes the contiguous fused_nocp
        // masked-fallback bug).  Signature: a a_desc b chunk_offsets cu_seqlens
        // g h0 ht k k_desc o o_desc q q_desc v v_desc batch num_tokens raw_batch.
        k.fused_nocp_packed_strided,
        [64, 1, 1],
        [512, 1, 1],
        75776,
        stream,
        &[
            KernelArg::Buffer(guard.a),
            KernelArg::Bytes(&a_desc.bytes),
            KernelArg::Buffer(guard.beta),
            KernelArg::Buffer(guard.chunk_offsets),
            KernelArg::Buffer(guard.cu_seqlens),
            KernelArg::Buffer(guard.g_cumsum),
            KernelArg::Buffer(h_state),
            KernelArg::Buffer(h_state),
            KernelArg::Buffer(k_ptr),
            KernelArg::Bytes(&k_desc.bytes),
            KernelArg::Buffer(output),
            KernelArg::Bytes(&o_desc.bytes),
            KernelArg::Buffer(qkv),
            KernelArg::Bytes(&q_desc.bytes),
            KernelArg::Buffer(v_ptr),
            KernelArg::Bytes(&v_desc.bytes),
            KernelArg::Bytes(&batch_size),
            KernelArg::Bytes(&n_tokens),
            KernelArg::Bytes(&real_batch),
        ],
    )?;
    gpu.synchronize(stream)?;
    Ok(())
}

#[cfg(test)]
mod cp_partition_tests {
    use super::cp_local_chunks;

    /// Oracle values from `_calc_cp_seqs` (P=48): covers partition-switch
    /// boundaries where `round(log2(...))` flips.
    #[test]
    fn matches_flashqla_oracle() {
        let nv = 32;
        let cases: &[(usize, usize, usize)] = &[
            // (tokens, chunks, expected local_chunks)
            (1, 1, 4),
            (64, 2, 4),
            (128, 4, 4),
            (129, 5, 4),
            (161, 6, 8),
            (673, 22, 16),
            (1024, 32, 16),
            (2048, 64, 16),
            (2721, 86, 32),
            (4096, 128, 32),
            (10913, 342, 64),
            (16384, 512, 64),
        ];
        for &(_tokens, chunks, want) in cases {
            assert_eq!(
                cp_local_chunks(nv, chunks),
                want,
                "chunks={chunks} expected local_chunks={want}"
            );
        }
    }
}
