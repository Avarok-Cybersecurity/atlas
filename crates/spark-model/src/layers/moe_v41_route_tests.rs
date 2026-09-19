// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! `moe_v41_route_select` against `route_from_logits`: picks, weight bits,
//! plan weights, plan slots, pointer table and miss flag, over random logits
//! with forced ties and a partly empty slot table. GPU only (`#[ignore]`).

use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{MoeV41, MoeV41Cfg, route_from_logits};

fn backend() -> spark_runtime::cuda_backend::AvarokCudaBackend {
    let set = avarok_kernels::ptx_for_exact_target("deepseek-v4-flash", "nvfp4")
        .expect("deepseek-v4-flash/nvfp4 not in this build");
    spark_runtime::cuda_backend::AvarokCudaBackend::new(0, &set.modules).expect("CUDA backend")
}

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 40) as u32 as f32 / (1u64 << 24) as f32) - 0.5
}

#[test]
#[ignore = "requires a CUDA GB10 + the deepseek-v4-flash kernel target"]
fn device_route_select_matches_the_host_chain_bitwise() {
    const NR: usize = 384;
    const K: usize = 6;
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let cfg = MoeV41Cfg {
        dim: 256,
        inter: 256,
        n_routed: NR,
        topk: K,
        gate_temp: 1.0,
        norm_topk_prob: true,
        route_scale: 2.5,
        swiglu_limit: 10.0,
        max_tokens: 4,
    };
    let moe = MoeV41::new(g, cfg.clone()).unwrap();
    let mut st = 0x5EED_2026u64;
    let bias: Vec<f32> = (0..NR).map(|_| lcg(&mut st)).collect();
    let bias_dev = MoeV41::upload_bias(g, &bias).unwrap();
    // slot table row for layer 3: every fifth expert absent
    let layer = 3u32;
    let slots: Vec<i32> = (0..NR)
        .map(|e| if e % 5 == 0 { -1 } else { (e * 3) as i32 })
        .collect();
    let changes: Vec<(u32, u32, i32)> = slots
        .iter()
        .enumerate()
        .map(|(e, &s)| (layer, e as u32, s))
        .collect();
    moe.slot_table_update(g, &changes, stream).unwrap();
    let arena = (
        0x7000_0000_0000u64,
        12_812_288u64,
        0u64,
        4_128_768u64,
        8_257_536u64,
    );
    let (mut bad_pick, mut bad_w, mut bad_plan, mut bad_ptr, mut bad_miss) = (0, 0, 0, 0, 0);
    for t in 0..400 {
        let mut logits: Vec<f32> = (0..NR)
            .map(|e| lcg(&mut st) * if e % 7 == 0 { 18.0 } else { 6.0 })
            .collect();
        if t % 10 == 0 {
            // a forced tie on the key: equal logits and equal bias would be
            // needed; equal logits with this bias pair differ, so tie the key
            logits[6] = logits[5] + (bias[5] - bias[6]);
        }
        let lb: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        g.copy_h2d(&lb, moe.logits).unwrap();
        moe.route_select_launch(g, layer, bias_dev, arena, stream)
            .unwrap();
        g.synchronize(stream).unwrap();
        let mut hdr = vec![0u8; (1 + 3 * K) * 4];
        g.copy_d2h(moe.route_hdr, &mut hdr).unwrap();
        let word = |i: usize| {
            i32::from_le_bytes([hdr[4 * i], hdr[4 * i + 1], hdr[4 * i + 2], hdr[4 * i + 3]])
        };
        let picks: Vec<usize> = (0..K).map(|i| word(1 + i) as usize).collect();
        let wbits: Vec<u32> = (0..K).map(|i| word(1 + K + i) as u32).collect();
        let plan_slots: Vec<i32> = (0..K).map(|i| word(1 + 2 * K + i)).collect();
        let (hw, hi) = route_from_logits(&logits, 1, &bias, &cfg);
        if hi != picks {
            bad_pick += 1;
        }
        if hw.iter().map(|v| v.to_bits()).collect::<Vec<_>>() != wbits {
            bad_w += 1;
        }
        let (_, w_host, plan) = moe.plan(&hi, &hw, 1);
        let mut w_dev = vec![0u8; K * 4];
        g.copy_d2h(moe.weight_dev, &mut w_dev).unwrap();
        let ids_asc: Vec<usize> = plan.iter().map(|&(a0, _, _)| hi[a0]).collect();
        let want_slots: Vec<i32> = ids_asc.iter().map(|&e| slots[e]).collect();
        if w_dev != w_host || plan_slots != want_slots {
            bad_plan += 1;
        }
        let miss = want_slots.iter().any(|&s| s < 0);
        if (word(0) != 0) != miss {
            bad_miss += 1;
        }
        if !miss {
            let mut ptrs = vec![0u8; 3 * K * 8];
            g.copy_d2h(moe.ptrs_dev, &mut ptrs).unwrap();
            let mut want: Vec<u8> = Vec::new();
            for off in [arena.2, arena.3, arena.4] {
                for &s in &want_slots {
                    want.extend_from_slice(&(arena.0 + s as u64 * arena.1 + off).to_le_bytes());
                }
            }
            if ptrs != want {
                bad_ptr += 1;
            }
        }
    }
    println!(
        "  device route: 400 tokens, pick {bad_pick} weight {bad_w} plan {bad_plan} ptr {bad_ptr} miss-flag {bad_miss} mismatches"
    );
    assert_eq!(
        (bad_pick, bad_w, bad_plan, bad_ptr, bad_miss),
        (0, 0, 0, 0, 0)
    );
    g.free(bias_dev).unwrap();
    let _ = DevicePtr(0);
    moe.free(g).unwrap();
}
