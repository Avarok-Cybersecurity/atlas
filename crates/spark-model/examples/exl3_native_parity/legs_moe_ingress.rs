// SPDX-License-Identifier: AGPL-3.0-only

//! Exact staging parity, including remote slots, cast edges, and guard bytes.
use anyhow::{Result, ensure};
use half::{bf16, f16};
use spark_model::layers::ops::{
    exl3_moe_replicate_a_bf16, exl3_moe_stage_ingress, exl3_moe_stage_routing,
};
use spark_runtime::gpu::DevicePtr;

use crate::util::{Ctx, as_bytes, up};

const GUARD: usize = 32;

fn guarded(ctx: &Ctx, bytes: usize) -> Result<DevicePtr> {
    up(ctx.g, &vec![0xa5; bytes + GUARD * 2])
}

fn read(ctx: &Ctx, ptr: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut out = vec![0; bytes + GUARD * 2];
    ctx.g.copy_d2h(ptr, &mut out)?;
    ensure!(out[..GUARD].iter().all(|&v| v == 0xa5));
    ensure!(out[GUARD + bytes..].iter().all(|&v| v == 0xa5));
    Ok(out[GUARD..GUARD + bytes].to_vec())
}

pub fn run(ctx: &Ctx) -> Result<bool> {
    let stream = ctx.g.default_stream();
    let mut cases = 0;
    for rows in 1..=4 {
        for (hidden, top_k) in [(1, 1), (7, 3), (129, 10), (2560, 10)] {
            for (local_start, num_local) in [(0, 512), (100, 13), (100, 0)] {
                let slots = rows * top_k;
                let ids: Vec<u32> = (0..slots)
                    .map(|i| [0, 99, 100, 112, 113, 511, u32::MAX][i % 7])
                    .collect();
                let probs: Vec<f32> = (0..slots)
                    .map(|i| {
                        [
                            0.0,
                            -0.0,
                            0.000_000_03,
                            0.1,
                            0.333_333_34,
                            0.999_755_86,
                            1.0,
                        ][i % 7]
                    })
                    .collect();
                let input: Vec<u16> = (0..rows * hidden)
                    .map(|i| {
                        let v = [
                            -0.0,
                            0.0,
                            -3.123_456_7,
                            0.000_000_01,
                            1.003_906_3,
                            65504.0,
                            1e10,
                            f32::INFINITY,
                        ];
                        let value = if i % v.len() == 2 {
                            v[2] + i as f32 / 128.0
                        } else {
                            v[i % v.len()]
                        };
                        bf16::from_f32(value).to_bits()
                    })
                    .collect();
                let input_d = up(ctx.g, &as_bytes(&input))?;
                let ids_d = up(
                    ctx.g,
                    &ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
                )?;
                let probs_d = up(
                    ctx.g,
                    &probs
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
                let sizes = [slots * 8, slots * 2, slots * hidden * 2];
                let old = [
                    guarded(ctx, sizes[0])?,
                    guarded(ctx, sizes[1])?,
                    guarded(ctx, sizes[2])?,
                ];
                let fused = [
                    guarded(ctx, sizes[0])?,
                    guarded(ctx, sizes[1])?,
                    guarded(ctx, sizes[2])?,
                ];
                exl3_moe_stage_routing(
                    ctx.g,
                    ids_d,
                    probs_d,
                    old[0].offset(GUARD),
                    old[1].offset(GUARD),
                    local_start,
                    num_local,
                    slots,
                    stream,
                )?;
                exl3_moe_replicate_a_bf16(
                    ctx.g,
                    input_d,
                    old[2].offset(GUARD),
                    rows,
                    top_k,
                    hidden,
                    stream,
                )?;
                exl3_moe_stage_ingress(
                    ctx.g,
                    input_d,
                    ids_d,
                    probs_d,
                    fused[0].offset(GUARD),
                    fused[1].offset(GUARD),
                    fused[2].offset(GUARD),
                    local_start,
                    num_local,
                    rows,
                    top_k,
                    hidden,
                    stream,
                )?;
                ctx.g.synchronize(stream)?;
                let expected_ids: Vec<u8> = ids
                    .iter()
                    .flat_map(|&id| {
                        let local = i64::from(id) - local_start as i64;
                        (if (0..num_local as i64).contains(&local) {
                            local
                        } else {
                            -1
                        })
                        .to_le_bytes()
                    })
                    .collect();
                let expected_probs: Vec<u8> = probs
                    .iter()
                    .flat_map(|&p| f16::from_f32(p).to_bits().to_le_bytes())
                    .collect();
                let expected_a: Vec<u16> = (0..slots * hidden)
                    .map(|i| {
                        let token = (i / hidden) / top_k;
                        f16::from_f32(bf16::from_bits(input[token * hidden + i % hidden]).to_f32())
                            .to_bits()
                    })
                    .collect();
                for (which, expected) in [expected_ids, expected_probs, as_bytes(&expected_a)]
                    .iter()
                    .enumerate()
                {
                    let reference = read(ctx, old[which], sizes[which])?;
                    let actual = read(ctx, fused[which], sizes[which])?;
                    ensure!(
                        &reference == expected,
                        "reference disagrees with cast/index oracle buffer={which}"
                    );
                    ensure!(
                        actual == reference,
                        "fused staging differs rows={rows} H={hidden} top_k={top_k} local={local_start}+{num_local} buffer={which}"
                    );
                }
                for ptr in old
                    .into_iter()
                    .chain(fused)
                    .chain([input_d, ids_d, probs_d])
                {
                    ctx.g.free(ptr)?;
                }
                cases += 1;
            }
        }
    }
    println!(
        "PASS EXL3 ingress: {cases} cases, three buffers byte-identical, cast/index oracle and guards passed"
    );
    Ok(true)
}
