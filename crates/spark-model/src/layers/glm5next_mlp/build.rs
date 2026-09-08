// SPDX-License-Identifier: AGPL-3.0-only

//! Binding one GLM MLP site for a rank: TP slicing of the dense/shared halves, EP selection of
//! the routed experts.
//!
//! Takes `load` closures rather than a `WeightStore`, for the same reason
//! [`crate::layers::glm5next_dsa::build`] does: the slicing is then testable without a
//! checkpoint, and the loader wiring stays one call site.
//!
//! # 🔴 The two axes are different, and mixing them is silent
//!
//! * **TP** splits the *width* of the dense FFN and the shared expert. `gate_proj`/`up_proj` are
//!   `[inter, hidden]` and split by ROW; `down_proj` is `[hidden, inter]` and splits by COLUMN.
//!   Slicing `down_proj` by row instead gives a well-formed `[hidden/tp, inter]` tensor and a
//!   plausible, wrong output.
//! * **EP** splits the *set* of routed experts. An expert is never cut — it is owned whole.
//!   The router stays replicated so every rank selects the same ids.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::Glm5NextMlpConfig;
use super::weights::{
    Glm5NextDenseMlpWeights, Glm5NextExpertPtrTable, Glm5NextExpertWeights, Glm5NextMoePtrTables,
    Glm5NextMoeWeights, Nvfp4Proj,
};

/// A BF16/F32 tensor as host `f32`, by layer-relative name.
pub type LoadFn<'a> = &'a dyn Fn(&str) -> Result<Vec<f32>>;
/// One routed expert, by GLOBAL id.
///
/// 🔴 A closure rather than a name→bytes loader on purpose. The routed experts are the only
/// thing here that is NOT sharded — an expert is owned whole — so the caller can hand over the
/// checkpoint's own device pointers with **no copy**. Routing 3.85 GiB per layer through a host
/// `f32` round trip, as the TP-sliced halves must, would be pure waste. It also keeps the packed
/// `e2m1` codes and `e4m3` block scales from ever passing through `f32`.
pub type ExpertFn<'a> = &'a dyn Fn(usize) -> Result<Glm5NextExpertWeights>;

/// Rows `[start, end)` of a `[rows, row_elems]` row-major tensor — column-parallel projections.
fn row_slice(v: &[f32], row_elems: usize, start: usize, end: usize) -> Vec<f32> {
    v[start * row_elems..end * row_elems].to_vec()
}

/// Columns `[start, end)` of every row — the row-parallel case (`down_proj`).
fn col_slice(v: &[f32], row_elems: usize, start: usize, end: usize) -> Vec<f32> {
    v.chunks(row_elems)
        .flat_map(|r| r[start..end].iter().copied())
        .collect()
}

fn up_bf16(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = v
        .iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
        .collect();
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}

fn up_f32(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}

/// TP-slice and upload one BF16 SwiGLU MLP — a dense layer, or a routed layer's shared expert.
///
/// `full_inter` is the tensor's width in the checkpoint; the rank keeps `full_inter / tp`.
pub fn build_dense_mlp(
    gpu: &dyn GpuBackend,
    cfg: &Glm5NextMlpConfig,
    tp_rank: usize,
    full_inter: usize,
    prefix: &str,
    load: LoadFn<'_>,
) -> Result<Glm5NextDenseMlpWeights> {
    let tp = cfg.tp_world_size;
    if !full_inter.is_multiple_of(tp) {
        bail!("GLM MLP {prefix}: intermediate {full_inter} does not divide over tp {tp}");
    }
    let local = full_inter / tp;
    let lo = tp_rank * local;

    let get = |n: &str| -> Result<Vec<f32>> { load(&format!("{prefix}.{n}")) };

    let expect = |name: &str, v: &[f32], want: usize| -> Result<()> {
        if v.len() != want {
            bail!(
                "GLM MLP {prefix}.{name}: {} elements, expected {want}",
                v.len()
            );
        }
        Ok(())
    };

    let gate = get("gate_proj.weight")?;
    expect("gate_proj.weight", &gate, full_inter * cfg.hidden)?;
    let up = get("up_proj.weight")?;
    expect("up_proj.weight", &up, full_inter * cfg.hidden)?;
    let down = get("down_proj.weight")?;
    expect("down_proj.weight", &down, cfg.hidden * full_inter)?;

    Ok(Glm5NextDenseMlpWeights {
        // Column-parallel: [inter, hidden] sliced by ROW.
        gate_proj: up_bf16(gpu, &row_slice(&gate, cfg.hidden, lo, lo + local))?,
        up_proj: up_bf16(gpu, &row_slice(&up, cfg.hidden, lo, lo + local))?,
        // Row-parallel: [hidden, inter] sliced by COLUMN. Output is a partial sum.
        down_proj: up_bf16(gpu, &col_slice(&down, full_inter, lo, lo + local))?,
    })
}

/// One projection's device pointer table over the FULL expert set.
///
/// 🪤 Indexed by GLOBAL id — remote ids get a **null** pointer, not a wrapped local slot. The
/// grouped kernel's only remote test is `packed == 0`; handing it a local expert's pointer for
/// a remote id would silently run the wrong expert on both ranks.
fn build_expert_ptr_table(
    gpu: &dyn GpuBackend,
    cfg: &Glm5NextMlpConfig,
    experts: &[Glm5NextExpertWeights],
    proj: impl Fn(&Glm5NextExpertWeights) -> Nvfp4Proj,
) -> Result<Glm5NextExpertPtrTable> {
    let n = cfg.num_experts;
    let mut packed = vec![0u8; n * 8];
    let mut scale = vec![0u8; n * 8];
    let mut scale2 = vec![0u8; n * 4];
    for id in 0..n {
        let Some(local) = cfg.local_slot(id) else {
            continue; // remote — null stays, and the kernel skips the slot
        };
        let p = proj(&experts[local]);
        packed[id * 8..id * 8 + 8].copy_from_slice(&p.packed.0.to_le_bytes());
        scale[id * 8..id * 8 + 8].copy_from_slice(&p.scale.0.to_le_bytes());
        scale2[id * 4..id * 4 + 4].copy_from_slice(&p.scale_2.to_le_bytes());
    }
    let packed_ptrs = gpu.alloc(packed.len())?;
    gpu.copy_h2d(&packed, packed_ptrs)?;
    let scale_ptrs = gpu.alloc(scale.len())?;
    gpu.copy_h2d(&scale, scale_ptrs)?;
    let scale2_vals = gpu.alloc(scale2.len())?;
    gpu.copy_h2d(&scale2, scale2_vals)?;
    Ok(Glm5NextExpertPtrTable {
        packed_ptrs,
        scale_ptrs,
        scale2_vals,
    })
}

/// Bind one routed MoE site for this rank: replicated router, TP-sharded shared expert, and
/// exactly the `local_experts` routed experts this EP rank owns.
pub fn build_moe(
    gpu: &dyn GpuBackend,
    cfg: &Glm5NextMlpConfig,
    tp_rank: usize,
    full_shared_inter: usize,
    load: LoadFn<'_>,
    expert: ExpertFn<'_>,
    // `Some` when the checkpoint ships routed experts as EXL3 trellis; see
    // `Glm5NextMoeWeights::exl3`.
    exl3: Option<crate::weight_loader::glm5_next_exl3::Glm5NextExl3Experts>,
) -> Result<Glm5NextMoeWeights> {
    // 🪤 REPLICATED, both of them. A sharded router gives each rank partial logits and a
    // different top-k, which makes masked-local EP select different experts per rank — no
    // crash, no shape error, a different answer.
    let router = load("mlp.gate.weight")?;
    if router.len() != cfg.num_experts * cfg.hidden {
        bail!(
            "GLM MoE mlp.gate.weight: {} elements, expected {} ({} experts x {} hidden)",
            router.len(),
            cfg.num_experts * cfg.hidden,
            cfg.num_experts,
            cfg.hidden
        );
    }
    let bias = load("mlp.gate.e_score_correction_bias")?;
    if bias.len() != cfg.num_experts {
        bail!(
            "GLM MoE e_score_correction_bias: {} entries, expected {}",
            bias.len(),
            cfg.num_experts
        );
    }

    let shared = build_dense_mlp(
        gpu,
        cfg,
        tp_rank,
        full_shared_inter,
        "mlp.shared_experts",
        load,
    )?;

    // Ascending GLOBAL id, so slot `i` is id `range.start + i` — the inverse of
    // `Glm5NextMlpConfig::local_slot`, and the only ordering the forward may assume.
    // EXL3 REPLACES this walk. `expert(id)` resolves NVFP4 triplets
    // (`.weight`/`.weight_scale`/`.weight_scale_2`) which a trellis pack does
    // not ship, so calling it there fails by name. Leaving `experts` empty also
    // makes `build_expert_ptr_table` emit an all-null table, which is the right
    // thing: under EXL3 nothing may dereference the NVFP4 pointers.
    let mut experts = Vec::with_capacity(cfg.local_experts);
    if exl3.is_none() {
        for id in cfg.local_expert_range() {
            experts.push(expert(id)?);
        }
    }

    let ptrs = Glm5NextMoePtrTables {
        gate: build_expert_ptr_table(gpu, cfg, &experts, |e| e.gate_proj)?,
        up: build_expert_ptr_table(gpu, cfg, &experts, |e| e.up_proj)?,
        down: build_expert_ptr_table(gpu, cfg, &experts, |e| e.down_proj)?,
    };

    Ok(Glm5NextMoeWeights {
        router: up_bf16(gpu, &router)?,
        // F32 in the checkpoint and `const float*` at the kernel — uploaded as F32, not BF16.
        router_bias: up_f32(gpu, &bias)?,
        shared,
        experts,
        ptrs,
        exl3,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Column-parallel and row-parallel slicing produce the SAME shape at tp=2 and are trivially
    /// swappable. This pins which axis each takes, on a tensor whose values encode their index.
    #[test]
    fn dense_slicing_takes_rows_for_gate_and_columns_for_down() {
        // [inter=4, hidden=3] gate: value = row*10 + col.
        let gate: Vec<f32> = (0..4)
            .flat_map(|r| (0..3).map(move |c| (r * 10 + c) as f32))
            .collect();
        // rank 1 of 2 keeps rows 2..4.
        assert_eq!(
            row_slice(&gate, 3, 2, 4),
            vec![20., 21., 22., 30., 31., 32.]
        );

        // [hidden=3, inter=4] down: value = row*10 + col. rank 1 keeps columns 2..4.
        let down: Vec<f32> = (0..3)
            .flat_map(|r| (0..4).map(move |c| (r * 10 + c) as f32))
            .collect();
        assert_eq!(col_slice(&down, 4, 2, 4), vec![2., 3., 12., 13., 22., 23.]);
        // On a square tensor the two axes give the SAME element count and different values —
        // which is why this is a test and not a length assertion in the loader.
        let sq: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let by_row = row_slice(&sq, 4, 2, 4);
        let by_col = col_slice(&sq, 4, 2, 4);
        assert_eq!(by_row.len(), by_col.len());
        assert_ne!(by_row, by_col);
    }
}
