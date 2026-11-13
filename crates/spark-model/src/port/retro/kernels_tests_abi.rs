// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the launch ABI of the RetroInfer gather and estimation attention (port
//! lane L34b) on the mock backend: grid, block, shared memory, argument order and values,
//! capture safety (no allocation, synchronization or read-back; a replay carries identical
//! arguments), module-absent behavior (no lookup issued), validation, and the kernel
//! sources' entry names and shared-memory limits.

use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};
use spark_runtime::gpu::{DevicePtr, KernelHandle};

use super::super::attn::{self, AttnKernels, AttnLaunch, IndexView};
use super::super::gather::{self, ExactOut, GatherKernels, GatherLaunch};
use super::super::state::{IndexLayout, IndexListRef};
use crate::layers::qwen3_attention::sparse_seam::{KvDtypeTag, KvLayerView};

const KVH: u32 = 4;
const QH: u32 = 16;

fn kv(dtype: KvDtypeTag) -> KvLayerView {
    KvLayerView {
        k_pool: DevicePtr(0x10_0000),
        v_pool: DevicePtr(0x20_0000),
        dtype,
        block_size: 16,
        kv_heads: KVH,
        head_dim: 256,
        kv_scale: 0.5,
        v_scale: 0.25,
        block_stride_bytes: 0,
        cache_stride: 0,
    }
}

fn exact(cap: u32) -> ExactOut {
    ExactOut {
        k: DevicePtr(0x30_0000),
        v: DevicePtr(0x40_0000),
        pos: DevicePtr(0x50_0000),
        valid: DevicePtr(0x50_8000),
        dropped: DevicePtr(0x50_9000),
        kv_heads: KVH,
        cap,
    }
}

fn layout() -> IndexLayout {
    IndexLayout {
        kv_heads: KVH,
        head_dim: 256,
        cap_centroids: 8192,
        cap_tokens: 131_072,
    }
}

fn gl(dtype: KvDtypeTag) -> GatherLaunch {
    GatherLaunch {
        kv: kv(dtype),
        block_table: DevicePtr(0x70_0000),
        row_lens: DevicePtr(0x71_0000),
        rows: 3,
        max_blocks_per_seq: 8192,
        retrieval_ids: DevicePtr(0x72_0000),
        counts: DevicePtr(0x73_0000),
        retrieval_cap: 300,
        lists: IndexListRef {
            header: DevicePtr(0x60_0000),
            cluster_size: DevicePtr(0x61_0000),
            list_offsets: DevicePtr(0x62_0000),
            list_tokens: DevicePtr(0x63_0000),
            layout: layout(),
        },
        out: exact(10_000),
    }
}

fn al() -> AttnLaunch {
    AttnLaunch {
        q: DevicePtr(0x80_0000),
        q_row_stride: QH * 256,
        rows: 5,
        row_trim: 0,
        q_heads: QH,
        scale: 1.0 / 16.0,
        exact: exact(10_000),
        est_ids: DevicePtr(0x81_0000),
        counts: DevicePtr(0x73_0000),
        es_cap: 2000,
        index: IndexView {
            centroids: DevicePtr(0x90_0000),
            value_sum: DevicePtr(0xA0_0000),
            cluster_size: DevicePtr(0xB0_0000),
            cap_centroids: 8192,
            kv_heads: KVH,
            head_dim: 256,
        },
        out: DevicePtr(0xC0_0000),
        lse: DevicePtr::NULL,
    }
}

fn u32_arg(a: &MockArg) -> u32 {
    match a {
        MockArg::Bytes(b) if b.len() == 4 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        other => panic!("not a 4-byte scalar: {other:?}"),
    }
}

#[test]
fn gather_launch_abi_and_capture_safety() {
    let gpu = MockGpuBackend::new();
    let k = GatherKernels::resolve(&gpu).unwrap();
    let lookups = gpu.kernel_lookups_snapshot();
    assert_eq!(lookups.len(), 2);
    assert!(lookups.iter().all(|(m, _)| m == gather::MODULE));
    let (allocs, syncs) = (gpu.alloc_count(), gpu.sync_count());
    gather::launch_gather(&gpu, &k, &gl(KvDtypeTag::Bf16), 7).unwrap();
    gather::launch_gather(&gpu, &k, &gl(KvDtypeTag::Bf16), 7).unwrap();
    assert_eq!((gpu.alloc_count(), gpu.sync_count()), (allocs, syncs));
    assert_eq!(gpu.d2h_blocking_count() + gpu.d2h_async_count(), 0);
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 2);
    assert_eq!(
        l[0].args, l[1].args,
        "a replayed launch carries the same arguments"
    );
    let x = &l[0];
    let smem = (3 * 300 + 1) * 4;
    assert_eq!(
        (x.grid, x.block, x.shared_mem, x.stream),
        ([KVH, 64, 1], [256, 1, 1], smem, 7)
    );
    assert_eq!(x.args.len(), 25);
    assert_eq!(x.args[0], MockArg::Buffer(DevicePtr(0x10_0000)));
    assert_eq!(u32_arg(&x.args[4]), 3);
    let stride = 16u64 * u64::from(KVH) * 256 * 2;
    assert_eq!(x.args[8], MockArg::Bytes(stride.to_le_bytes().to_vec()));
    assert_eq!(f32::from_bits(u32_arg(&x.args[9])), 0.5);
    assert_eq!(f32::from_bits(u32_arg(&x.args[10])), 0.25);
    assert_eq!(x.args[12], MockArg::Buffer(DevicePtr(0x73_0000)));
    assert_eq!(x.args[14], MockArg::Buffer(DevicePtr(0x60_0000)));
    assert_eq!(
        (u32_arg(&x.args[17]), u32_arg(&x.args[18])),
        (8192, 131_072)
    );
    assert_eq!(x.args[23], MockArg::Buffer(DevicePtr(0x50_9000)));
    assert_eq!(u32_arg(&x.args[24]), 10_000);
    let pick = GatherKernels {
        bf16: KernelHandle(1),
        fp8: KernelHandle(2),
    };
    assert_eq!(pick.for_dtype(KvDtypeTag::Bf16).0, 1);
    assert_eq!(pick.for_dtype(KvDtypeTag::Fp8).0, 2);
    assert_eq!(gather::gather_grid(4, 64), [4, 2, 1]);
    assert_eq!(gather::gather_grid(4, 0), [4, 1, 1]);
}

#[test]
fn attention_launch_abi_and_capture_safety() {
    let gpu = MockGpuBackend::new();
    let k = AttnKernels::resolve(&gpu, 4).unwrap();
    assert_eq!(
        gpu.kernel_lookups_snapshot(),
        vec![(attn::MODULE.to_owned(), "retro_est_attn_g4".to_owned())]
    );
    let allocs = gpu.alloc_count();
    attn::launch_attention(&gpu, &k, &al(), 3).unwrap();
    attn::launch_attention(&gpu, &k, &al(), 3).unwrap();
    assert_eq!((gpu.alloc_count(), gpu.sync_count()), (allocs, 0));
    let l = gpu.launches_snapshot();
    assert_eq!(l[0].args, l[1].args);
    let x = &l[0];
    assert_eq!(
        (x.grid, x.block, x.shared_mem),
        ([KVH, 5, 1], [256, 1, 1], 0)
    );
    assert_eq!(x.args.len(), 21);
    assert_eq!(
        (
            u32_arg(&x.args[1]),
            u32_arg(&x.args[2]),
            u32_arg(&x.args[3])
        ),
        (QH * 256, 5, 0)
    );
    assert_eq!(x.args[10], MockArg::Buffer(DevicePtr(0x73_0000)));
    assert_eq!(
        x.args[17],
        MockArg::Buffer(DevicePtr::NULL),
        "no LSE requested"
    );
    assert_eq!((u32_arg(&x.args[18]), u32_arg(&x.args[19])), (QH, KVH));
    assert_eq!(f32::from_bits(u32_arg(&x.args[20])), 1.0 / 16.0);
    // A handle resolved for another group is refused.
    let wrong = AttnKernels::resolve(&gpu, 2).unwrap();
    assert!(attn::launch_attention(&gpu, &wrong, &al(), 3).is_err());
}

#[test]
fn absent_modules_issue_no_lookup_and_no_launch() {
    let gpu = MockGpuBackend::new();
    gpu.mark_module_absent(gather::MODULE);
    gpu.mark_module_absent(attn::MODULE);
    assert!(!GatherKernels::available(&gpu) && !AttnKernels::available(&gpu));
    assert!(GatherKernels::resolve(&gpu).is_err());
    assert!(AttnKernels::resolve(&gpu, 4).is_err());
    assert!(gpu.kernel_lookups_snapshot().is_empty());
    assert_eq!(gpu.launch_count(), 0);
}

#[test]
fn validation_refuses_what_the_kernels_cannot_serve() {
    let ok = gl(KvDtypeTag::Fp8);
    assert!(gather::validate(&ok).is_ok());
    let bad: [fn(&mut GatherLaunch); 9] = [
        |g| g.kv.head_dim = 128,
        |g| g.rows = 0,
        |g| g.retrieval_cap = gather::MAX_RETRIEVAL_CAP + 1,
        |g| g.lists.layout.kv_heads = 2,
        |g| g.kv.k_pool = DevicePtr(0x10_0004),
        |g| g.kv.block_stride_bytes = 8,
        |g| g.out.kv_heads = 2,
        |g| g.block_table = DevicePtr::NULL,
        |g| g.lists.list_tokens = DevicePtr::NULL,
    ];
    for (i, f) in bad.iter().enumerate() {
        let mut g = ok;
        f(&mut g);
        assert!(gather::validate(&g).is_err(), "gather case {i}");
    }
    assert!(attn::validate(&al()).is_ok());
    let bad: [fn(&mut AttnLaunch); 8] = [
        |a| a.q_heads = 12, // group 3
        |a| a.q_heads = 0,  // ExactBuf::num_q_heads left unset
        |a| a.index.head_dim = 128,
        |a| a.rows = 0,
        |a| a.q_row_stride = 256,
        |a| a.index.value_sum = DevicePtr(0xA0_0008),
        |a| a.counts = DevicePtr::NULL,
        |a| a.scale = f32::NAN,
    ];
    for (i, f) in bad.iter().enumerate() {
        let mut a = al();
        f(&mut a);
        assert!(attn::validate(&a).is_err(), "attention case {i}");
    }
}

#[test]
fn kernel_sources_declare_every_entry_and_respect_shared_memory() {
    let g = include_str!("../../../../../kernels/ada/common/retro_gather_ada.cu");
    let a = include_str!("../../../../../kernels/ada/common/retro_attn_ada.cu");
    for e in [gather::ENTRY_BF16, gather::ENTRY_FP8] {
        assert!(
            g.contains(&format!(
                "void __launch_bounds__(RG_THREADS) {e}(RG_PARAMS)"
            )),
            "{e}"
        );
    }
    assert!(g.contains("#define RG_HDIM 256") && g.contains("#define RG_WARPS 8"));
    assert!(g.contains("#define RG_HDR_INDEXED_END 3") && g.contains("#define RG_HDR_SINK 4"));
    assert_eq!(super::super::state::header::INDEXED_END, 3);
    assert_eq!(super::super::state::header::SINK, 4);
    assert!(a.contains("#define RA_HDIM 256") && a.contains("#define RA_WARPS 8"));
    let mut names = Vec::new();
    for gr in attn::GROUPS {
        let n = attn::entry_name(gr).unwrap();
        assert!(a.contains(&format!("RA_ENTRY({n}, {gr})")), "{n}");
        names.push(n);
    }
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), attn::GROUPS.len());
    assert!(attn::entry_name(3).is_none());
    assert!(gather::gather_smem(gather::MAX_RETRIEVAL_CAP) + 20 <= 48 * 1024);
    // Constant facts are checked at compile time (clippy::assertions_on_constants).
    const _: () = assert!(attn::STATIC_SMEM_BYTES <= 48 * 1024);
    assert_eq!(gather::THREADS, gather::WARPS * 32);
    const _: () = assert!(gather::MAX_ROWS >= super::super::config::MAX_VERIFY_ROWS);
}
