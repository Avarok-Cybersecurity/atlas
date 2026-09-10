// SPDX-License-Identifier: AGPL-3.0-only

//! The GLM MLP weight contract, already sharded for this rank.

use spark_runtime::gpu::DevicePtr;

/// One NVFP4 projection as ModelOpt stores it: packed `e2m1` pairs, per-block `e4m3` scales,
/// and one global `f32` scale.
///
/// 🪤 The three travel together and are meaningless apart. `weight_scale_2` is a **scalar read
/// off the device at load** and passed by value — `w4a16_gemm` takes it as `arg_f32`, not as a
/// pointer, so uploading it and passing the address silently reinterprets a pointer as a float.
#[derive(Debug, Clone, Copy)]
pub struct Nvfp4Proj {
    /// `[out, in/2]` U8 — two `e2m1` codes per byte.
    pub packed: DevicePtr,
    /// `[out, in/16]` F8_E4M3 block scales.
    pub scale: DevicePtr,
    /// The single global F32 scale.
    pub scale_2: f32,
}

/// A dense SwiGLU MLP in BF16 — layers `0..first_k_dense_replace`, and the shared expert of
/// every routed layer. Same shape, same kernels, different widths.
///
/// TP: `gate_proj`/`up_proj` are column-parallel (**row**-sliced, since each is stored `[inter,
/// hidden]`), `down_proj` is row-parallel (**column**-sliced on `[hidden, inter]`). The output
/// is therefore a partial sum whenever `tp_world_size > 1`.
#[derive(Debug, Clone, Copy)]
pub struct Glm5NextDenseMlpWeights {
    /// `[local_inter, hidden]` BF16.
    pub gate_proj: DevicePtr,
    /// `[local_inter, hidden]` BF16.
    pub up_proj: DevicePtr,
    /// `[hidden, local_inter]` BF16 — row-parallel.
    pub down_proj: DevicePtr,
}

/// One routed expert. NVFP4, owned **whole** by one EP rank — never split further.
#[derive(Debug, Clone, Copy)]
pub struct Glm5NextExpertWeights {
    pub gate_proj: Nvfp4Proj,
    pub up_proj: Nvfp4Proj,
    pub down_proj: Nvfp4Proj,
}

/// Device-side pointer tables for ONE projection across the **full** expert set.
///
/// Indexed by GLOBAL expert id, so the grouped kernel can go straight from the router's
/// on-device `ids` to weights with no host round trip. Experts another EP rank owns carry a
/// **null** `packed`/`scale` pointer; the kernel writes nothing for those slots and the
/// caller's pre-zeroed output row stands.
#[derive(Debug, Clone, Copy)]
pub struct Glm5NextExpertPtrTable {
    /// `[num_experts]` U64 device pointers to each expert's packed NVFP4 weight.
    pub packed_ptrs: DevicePtr,
    /// `[num_experts]` U64 device pointers to each expert's block scales.
    pub scale_ptrs: DevicePtr,
    /// `[num_experts]` F32 per-expert `weight_scale_2`.
    pub scale2_vals: DevicePtr,
}

/// The three projections' pointer tables for one routed site.
#[derive(Debug, Clone, Copy)]
pub struct Glm5NextMoePtrTables {
    pub gate: Glm5NextExpertPtrTable,
    pub up: Glm5NextExpertPtrTable,
    pub down: Glm5NextExpertPtrTable,
}

/// A routed MoE site's weights for this rank.
pub struct Glm5NextMoeWeights {
    /// `[num_experts, hidden]` BF16 router. 🪤 **REPLICATED, and it must stay that way** — see
    /// the module header. Consumed through the FP32-out GEMM.
    pub router: DevicePtr,
    /// `[num_experts]` F32 selection bias (`gate.e_score_correction_bias`).
    ///
    /// 🪤 Steers SELECTION ONLY. The emitted weight is the chosen expert's *unbiased* score.
    pub router_bias: DevicePtr,
    /// The shared expert — dense BF16, TP-sharded, added to every token unscaled.
    pub shared: Glm5NextDenseMlpWeights,
    /// Exactly `local_experts` entries, indexed by **local slot**, in ascending global id.
    ///
    /// 🪤 Indexed by `Glm5NextMlpConfig::local_slot(global_id)`, never by the global id.
    pub experts: Vec<Glm5NextExpertWeights>,
    /// Global-id-indexed device pointer tables over the same experts, for the grouped
    /// device-dispatch forward. Null entries mark remote ids.
    pub ptrs: Glm5NextMoePtrTables,
    /// Routed experts kept PACKED as EXL3 trellis, when the checkpoint ships
    /// them that way (`quant_method: exl3`).
    ///
    /// `Some` REPLACES the routed arm: `experts` is then empty and `ptrs` holds
    /// only null pointers, because the trellis is decoded in-kernel by
    /// `exl3_mgemm` and there is no NVFP4 packed/scale pair to point at. The
    /// router and the shared expert are untouched either way — the published
    /// packs quantize routed experts ONLY (`scope: glm53_routed_experts_only`),
    /// so everything else stays on its existing BF16 path.
    pub exl3: Option<crate::weight_loader::glm5_next_exl3::Glm5NextExl3Experts>,
}
