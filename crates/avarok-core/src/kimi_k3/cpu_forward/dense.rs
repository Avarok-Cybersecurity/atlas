// SPDX-License-Identifier: AGPL-3.0-only

use super::{DenseMlp, K3LayerCtx};
use crate::kimi_k3::{ops::matvec, situ::situ_glu_vec};
use anyhow::{Result, ensure};

pub(super) fn run(
    ctx: &K3LayerCtx<'_>,
    w: &DenseMlp,
    x: &[f32],
    configured_inter: usize,
) -> Result<Vec<f32>> {
    let hidden = ctx.hidden;
    ensure!(hidden > 0 && x.len() == hidden, "K3 dense geometry");
    let inter = if !w.gate.is_empty() && w.gate.len().is_multiple_of(hidden) {
        w.gate.len() / hidden
    } else {
        let full = configured_inter.max(1);
        if ctx.tp_world > 1 && full.is_multiple_of(ctx.tp_world) {
            full / ctx.tp_world
        } else {
            full
        }
    };
    ensure!(inter > 0, "K3 dense local inter");
    if !w.gate.is_empty() {
        ensure!(
            w.up.len() == w.gate.len() && w.down.len() == w.gate.len(),
            "K3 dense weight geometry"
        );
    }
    if let Some(core) = ctx.dense_mlp {
        let out = core(w, x, hidden, inter, ctx.situ_beta, ctx.situ_linear_beta)?;
        ensure!(
            out.len() == hidden && out.iter().all(|v| v.is_finite()),
            "K3 dense core output invalid"
        );
        Ok(out)
    } else {
        Ok(reference(
            w,
            x,
            hidden,
            configured_inter,
            ctx.situ_beta,
            ctx.situ_linear_beta,
        ))
    }
}

pub(super) fn reference(
    w: &DenseMlp,
    x: &[f32],
    hidden: usize,
    inter: usize,
    situ_beta: f32,
    situ_linear_beta: f32,
) -> Vec<f32> {
    // Local intermediate under TP (gate/up column-sharded). `inter` is the
    // config full width; prefer the weight's own row count.
    let inter = if hidden == 0 {
        inter
    } else {
        w.gate.len() / hidden
    };
    let gate = matvec(&w.gate, x, inter, hidden);
    let up = matvec(&w.up, x, inter, hidden);
    let mid = situ_glu_vec(&gate, &up, situ_beta, situ_linear_beta);
    matvec(&w.down, &mid, hidden, inter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kimi_k3::cpu_weights::K3CpuModel;

    #[test]
    fn resident_dispatch_matches_reference_with_local_width() {
        let model = K3CpuModel::synthetic_tiny();
        let mut ctx = K3LayerCtx::from_model(&model);
        let h = ctx.hidden;
        let inter = 3;
        let weights = DenseMlp {
            gate: vec![0.1; h * inter],
            up: vec![0.2; h * inter],
            down: vec![0.3; h * inter],
        };
        let input = vec![0.4; h];
        let expected = run(&ctx, &weights, &input, inter * 2).unwrap();
        let core = |w: &DenseMlp, x: &[f32], hidden, local, beta, linear| {
            assert_eq!(local, inter);
            Ok(reference(w, x, hidden, local, beta, linear))
        };
        ctx.dense_mlp = Some(&core);
        assert_eq!(run(&ctx, &weights, &input, inter * 2).unwrap(), expected);
    }

    #[test]
    fn resident_failure_and_invalid_outputs_never_fall_back() {
        let model = K3CpuModel::synthetic_tiny();
        let mut ctx = K3LayerCtx::from_model(&model);
        let h = ctx.hidden;
        let weights = DenseMlp {
            gate: vec![0.1; h],
            up: vec![0.2; h],
            down: vec![0.3; h],
        };
        let input = vec![0.4; h];
        let fail = |_: &DenseMlp, _: &[f32], _, _, _, _| anyhow::bail!("device failure");
        ctx.dense_mlp = Some(&fail);
        assert!(
            run(&ctx, &weights, &input, 1)
                .unwrap_err()
                .to_string()
                .contains("device failure")
        );
        for output in [vec![0.0; h + 1], vec![f32::NAN; h]] {
            let mut ctx = K3LayerCtx::from_model(&model);
            let invalid = |_: &DenseMlp, _: &[f32], _, _, _, _| Ok(output.clone());
            ctx.dense_mlp = Some(&invalid);
            assert!(run(&ctx, &weights, &input, 1).is_err());
        }
    }
}
