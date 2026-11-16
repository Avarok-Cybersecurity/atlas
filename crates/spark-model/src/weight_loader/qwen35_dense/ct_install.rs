// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the dense Qwen3.5 loader's compressed-tensors `pack-quantized` W4A16 route: which
//! checkpoints it applies to ([`declared`]), the tensors the checkpoint loader must leave on disk
//! ([`defer_hook`]), the served scheme read from `config.json` ([`resolve_scheme`]), and the Marlin
//! `W4Linear`s of the FFN and full-attention q/k/v/o built from the checkpoint's own int4 codes
//! ([`ffn_boxes`], [`attn_boxes`]). The installs themselves are `w4_install.rs`'s keep-packed hooks.
//!
//! **ACTIVE BY FORMAT, NO ROW.** The route runs when, and only when, the config declares
//! `quant_method: compressed-tensors` with `format: pack-quantized` (the INT packer; the FP4 packers
//! keep their NVFP4 / MXFP4 routes). Its numerics are the checkpoint's (GPTQ codes and scales), so it
//! needs no `marlin_w4a16` row; nothing else changes for any other checkpoint.
//!
//! **WHY DEFERRED.** The store has no use for `weight_packed` / `weight_scale`: the Marlin repack runs
//! on the host, so uploading them would only add a transient. [`defer_hook`] keeps them on disk and
//! the install reads each once (`DeferredTensor::read_host_bytes`). A tensor that reached the store
//! anyway (a loader that does not consult the hook, or an F16 scale, which the disk loaders rewrite to
//! BF16 and therefore never defer) is read back from the device instead.
//!
//! **FAILS LOUDLY.** Every refusal is a load error naming the module: Marlin not compiled for the target
//! (`marlin_ada` is the Ada build only), a shape without a Marlin instance, a partly quantized layer, an
//! unserved scheme (asymmetric, not group 128, act-order with a non-contiguous `g_idx`, quantized
//! activations), tensor parallel, a pending LoRA adapter, and quantized GDN projections (the GDN pair
//! install takes the loader's fused `[Q|K|V|Z]`, which a packed checkpoint cannot provide).

use anyhow::{Context, Result, bail, ensure};
use avarok_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{DeferHook, WeightDtype, WeightStore};

use super::packed_install::{ATTN_NAMES, FFN_NAMES, ffn_shapes};
use crate::port::marlin::compressed::{
    self, CtPackedWeights, CtW4Scheme, G_IDX, PACKED, SCALE, ZERO_POINT, tensor_name,
};
use crate::port::marlin::{self, MarlinType, ScaleDtype, kernels, route};
use crate::port::w4::{W4Box, W4Linear};

/// The GDN projections of a Qwen3.5 linear-attention layer (all BF16 in the RedHatAI exports).
pub const GDN_NAMES: [&str; 5] = [
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
];

/// The config declares a compressed-tensors `pack-quantized` (INT, W4A16) checkpoint.
pub fn declared(config: &ModelConfig) -> bool {
    config.quantization_config.as_ref().is_some_and(|qc| {
        qc.quant_method == compressed::QUANT_METHOD && compressed::is_int_pack_format(&qc.format)
    })
}

/// The checkpoint loader's defer rule for such a checkpoint: every `*.weight_packed`,
/// `*.weight_scale`, `*.weight_shape`, `*.weight_g_idx` and `*.weight_zero_point` stays on disk.
/// `None` for every other checkpoint (upload everything, as before).
pub fn defer_hook(config: &ModelConfig) -> Option<DeferHook> {
    if !declared(config) {
        return None;
    }
    let hook: DeferHook =
        std::sync::Arc::new(|name: &str, _dtype: WeightDtype| compressed::is_scheme_tensor(name));
    Some(hook)
}

/// The served scheme, read from the `config.json` next to the checkpoint's shards. `None` when the
/// config does not declare the format (then the route is inert); `Some(Err)` carries the reason the
/// scheme is not served, which the first install turns into the load error.
pub fn resolve_scheme(
    store: &WeightStore,
    config: &ModelConfig,
) -> Option<std::result::Result<CtW4Scheme, String>> {
    if !declared(config) {
        return None;
    }
    let dir = store
        .deferred_sorted()
        .into_iter()
        .find(|(name, _)| compressed::is_scheme_tensor(name))
        .and_then(|(_, d)| d.path.parent().map(std::path::Path::to_path_buf));
    let scheme = match dir {
        Some(dir) => match CtW4Scheme::from_model_dir(&dir) {
            Ok(Some(s)) => Ok(s),
            Ok(None) => Err(format!(
                "{}: config.json does not declare compressed-tensors pack-quantized, but the \
                 parsed model config does",
                dir.display()
            )),
            Err(e) => Err(format!("{e:#}")),
        },
        // No deferred tensor to locate config.json by (a loader that ignores the defer hook): the
        // tensors themselves are checked at install (shape implies 4-bit, scale shape implies the
        // group, a zero point is refused).
        None => Ok(CtW4Scheme {
            group: marlin::DEFAULT_GROUP,
            actorder: compressed::ActOrder::None,
            ignore: Vec::new(),
        }),
    };
    if let Ok(s) = &scheme {
        tracing::info!(
            "[avarok] compressed-tensors pack-quantized W4A16 (group {}, actorder {:?}): quantized \
             linears are served by Marlin int4 from the checkpoint's codes; every other tensor loads \
             as BF16",
            s.group,
            s.actorder
        );
    }
    Some(scheme)
}

fn present(store: &WeightStore, name: &str) -> bool {
    store.deferred(name).is_some() || store.contains(name)
}

/// `module` (`<layer>.mlp.gate_proj`, ...) is a pack-quantized linear in this checkpoint.
pub fn module_quantized(store: &WeightStore, module: &str) -> bool {
    present(store, &tensor_name(module, PACKED))
}

/// Any of the layer's gate/up/down is pack-quantized (`lp` is the layer prefix).
pub fn ffn_any(store: &WeightStore, lp: &str) -> bool {
    FFN_NAMES
        .iter()
        .any(|p| module_quantized(store, &format!("{lp}.mlp.{p}")))
}

/// Any of the layer's q/k/v/o is pack-quantized (`p` is `<layer>.self_attn`).
pub fn attn_any(store: &WeightStore, p: &str) -> bool {
    ATTN_NAMES
        .iter()
        .any(|n| module_quantized(store, &format!("{p}.{n}")))
}

/// Any of the GDN layer's projections is pack-quantized (`la` is `<layer>.linear_attn`).
pub fn gdn_any(store: &WeightStore, la: &str) -> bool {
    GDN_NAMES
        .iter()
        .any(|n| module_quantized(store, &format!("{la}.{n}")))
}

/// The load error for a pack-quantized GDN projection.
pub fn gdn_refusal(la: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "{la}: the compressed-tensors checkpoint quantizes the GDN (linear attention) projections; \
         the GDN layer has no install for checkpoint int4 codes (its W4 pair is quantized from the \
         loader's fused BF16 [Q|K|V|Z]). Use an export that keeps linear_attn in BF16 (the RedHatAI \
         recipes ignore 're:.*linear_attn.*')"
    )
}

/// A tensor's bytes on the host with its shape and store dtype: from its on-disk location when the
/// checkpoint loader deferred it, else read back from the store. `None` when it is in neither.
fn host_tensor(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    stream: u64,
    name: &str,
) -> Result<Option<(Vec<u8>, Vec<usize>, WeightDtype)>> {
    if let Some(d) = store.deferred(name) {
        let bytes = d
            .read_host_bytes()
            .with_context(|| format!("compressed-tensors: reading {name}"))?;
        return Ok(Some((bytes, d.shape.clone(), d.dtype)));
    }
    if let Ok(t) = store.get(name) {
        let mut bytes = vec![0u8; t.byte_size()];
        gpu.synchronize(stream)?;
        gpu.copy_d2h_on_stream(t.ptr, &mut bytes, stream)
            .with_context(|| format!("compressed-tensors: reading {name} back from the device"))?;
        gpu.synchronize(stream)?;
        return Ok(Some((bytes, t.shape.clone(), t.dtype)));
    }
    Ok(None)
}

/// The Marlin target and shape gate for one `[n, k]` module, before any byte is read.
fn check_target(gpu: &dyn GpuBackend, module: &str, n: usize, k: usize) -> Result<()> {
    ensure!(
        marlin::available(gpu),
        "{module}: this is a compressed-tensors pack-quantized W4A16 checkpoint, served only by the \
         Marlin int4 kernels (module {}), which this target's build does not carry (the Ada sm_89 \
         build only); serve the BF16 checkpoint on this target",
        kernels::MODULE
    );
    route::shape_ok(n, k, MarlinType::U4B8).map_err(|why| {
        anyhow::anyhow!(
            "{module}: [{n}x{k}] has no Marlin W4A16 instance ({why}); this checkpoint cannot be \
             served on this target"
        )
    })
}

/// One pack-quantized `[n, k]` module (`<prefix>.<proj>`) as a Marlin weight.
pub fn build_module(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    stream: u64,
    module: &str,
    n: usize,
    k: usize,
    scheme: &CtW4Scheme,
) -> Result<W4Box> {
    check_target(gpu, module, n, k)?;
    let group = scheme.group;
    ensure!(
        !present(store, &tensor_name(module, ZERO_POINT)),
        "{module}: the checkpoint stores zero points (asymmetric int4); this route serves symmetric \
         W4A16 only"
    );
    let pname = tensor_name(module, PACKED);
    let (packed, pshape, pdt) = host_tensor(gpu, store, stream, &pname)?
        .with_context(|| format!("{pname}: missing (the layer is partly quantized)"))?;
    ensure!(
        pdt == WeightDtype::Int32,
        "{pname}: dtype {pdt:?}, I32 expected (eight int4 codes per word)"
    );
    ensure!(
        pshape == [n, k / compressed::PACK_FACTOR],
        "{pname}: shape {pshape:?}, the config's [{n}x{k}] weight packs to [{n}, {}] (4-bit only)",
        k / compressed::PACK_FACTOR
    );
    let sname = tensor_name(module, SCALE);
    let (scales, sshape, sdt) =
        host_tensor(gpu, store, stream, &sname)?.with_context(|| format!("{sname}: missing"))?;
    // An F16 scale reaches the store as BF16 (the disk loaders convert it); FP32 is refused.
    ensure!(
        sdt == WeightDtype::BF16,
        "{sname}: dtype {sdt:?}, BF16 or F16 expected"
    );
    ensure!(
        sshape == [n, k / group],
        "{sname}: shape {sshape:?}, group {group} on [{n}x{k}] needs [{n}, {}]",
        k / group
    );
    let gname = tensor_name(module, G_IDX);
    let g_idx = host_tensor(gpu, store, stream, &gname)?;
    if let Some((_, gshape, gdt)) = &g_idx {
        ensure!(
            *gdt == WeightDtype::Int32 && gshape.as_slice() == [k],
            "{gname}: {gdt:?} {gshape:?}, I32 [{k}] expected"
        );
    }
    let w = CtPackedWeights {
        packed: &packed,
        scales: &scales,
        scale_dtype: ScaleDtype::Bf16,
        g_idx: g_idx.as_ref().map(|(b, _, _)| b.as_slice()),
        n,
        k,
        group,
    };
    let b = marlin::from_compressed(gpu, &w, stream)
        .with_context(|| format!("{module}: Marlin W4A16 install from the checkpoint codes"))?;
    adopt(store, &*b, module)?;
    Ok(b)
}

/// The ledger labels of a Marlin weight's buffers, in `route::buffers_of` order.
const LABELS: [&str; 4] = [
    "compressed-tensors marlin int4 codes",
    "compressed-tensors marlin scales",
    "compressed-tensors marlin zero points",
    "compressed-tensors marlin locks",
];

/// Hand the weight's device buffers to the store's ledger (#736: one owner, released at teardown)
/// and make `marlin::release` of it a no-op, as the affine route does for its Marlin arm.
fn adopt(store: &WeightStore, w: &dyn W4Linear, module: &str) -> Result<()> {
    let bufs = route::buffers_of(w)
        .with_context(|| format!("{module}: the built weight is not a Marlin weight"))?;
    let d = store.derived();
    let mut bytes = 0usize;
    // `buffers_of` leaves the zero-point buffer out for kU4B8, so label by position from the end
    // for the locks (always last).
    let last = bufs.len().saturating_sub(1);
    for (i, (ptr, b)) in bufs.into_iter().enumerate() {
        let label = if i == last {
            LABELS[3]
        } else {
            LABELS[i.min(2)]
        };
        d.adopt(label, ptr, b);
        bytes += b;
    }
    route::disown(w);
    ensure!(
        bytes == w.device_bytes(),
        "{module}: adopted {bytes} of the Marlin weight's {} device bytes; a buffer has no owner",
        w.device_bytes()
    );
    Ok(())
}

fn check_layer(config: &ModelConfig, what: &str) -> Result<()> {
    ensure!(
        config.tp_world_size.max(1) == 1,
        "{what}: compressed-tensors pack-quantized tensors cannot be sharded (tensor parallel {}); \
         serve on one GPU",
        config.tp_world_size
    );
    ensure!(
        config.adapter_max_rank == 0,
        "{what}: a LoRA adapter cannot be applied to checkpoint int4 weights; serve the BF16 \
         checkpoint to use an adapter"
    );
    Ok(())
}

/// The layer's gate, up and down as Marlin weights, in that order. Every projection must be
/// pack-quantized (a partly quantized FFN has no serving path).
pub fn ffn_boxes(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    config: &ModelConfig,
    lp: &str,
    scheme: &CtW4Scheme,
    stream: u64,
) -> Result<[W4Box; 3]> {
    check_layer(config, lp)?;
    let shapes = ffn_shapes(config.hidden_size, super::ffn_inter(config));
    let mut out: Vec<W4Box> = Vec::with_capacity(3);
    for (proj, [n, k]) in FFN_NAMES.iter().zip(shapes) {
        let module = format!("{lp}.mlp.{proj}");
        if !module_quantized(store, &module) {
            bail!(
                "{module} is not pack-quantized while another FFN projection of the layer is; a \
                 partly quantized FFN has no serving path"
            );
        }
        out.push(build_module(gpu, store, stream, &module, n, k, scheme)?);
    }
    out.try_into()
        .map_err(|_| anyhow::anyhow!("{lp}: three FFN weights expected"))
}

/// The layer's q, k, v and o as Marlin weights, in that order (`shapes` is `[n, k]` of each,
/// `AttnShapes::projections`). Every projection must be pack-quantized.
pub fn attn_boxes(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    config: &ModelConfig,
    p: &str,
    shapes: [(usize, usize); 4],
    scheme: &CtW4Scheme,
    stream: u64,
) -> Result<[W4Box; 4]> {
    check_layer(config, p)?;
    let mut out: Vec<W4Box> = Vec::with_capacity(4);
    for (proj, (n, k)) in ATTN_NAMES.iter().zip(shapes) {
        let module = format!("{p}.{proj}");
        if !module_quantized(store, &module) {
            bail!(
                "{module} is not pack-quantized while another attention projection of the layer is; \
                 a partly quantized attention layer has no serving path"
            );
        }
        out.push(build_module(gpu, store, stream, &module, n, k, scheme)?);
    }
    out.try_into()
        .map_err(|_| anyhow::anyhow!("{p}: four attention weights expected"))
}
