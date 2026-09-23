// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn header_scan_matches_every_bound_tensor() {
    let path = match std::env::var("K3_CONTRACT_JSON") {
        Ok(p) => p,
        Err(_) => return,
    };
    let text = std::fs::read_to_string(&path).expect("contract manifest");
    let mut rows = Vec::new();
    let mut bad = Vec::new();
    let mut mapped = 0usize;
    for line in text.lines() {
        if line.is_empty() {
            continue;
        }
        let (name, dims) = line.split_once('\t').expect("name tab shape");
        let ggml: Vec<usize> = dims.split(',').map(|d| d.parse().unwrap()).collect();
        let mut hf_shape = ggml.clone();
        hf_shape.reverse();
        let Some(translated) = super::super::names::translate(name, "kimi-k3") else {
            bad.push(format!("UNMAPPED {name}"));
            continue;
        };
        mapped += 1;
        match translated {
            super::super::names::GgufName::Drop => {}
            super::super::names::GgufName::Direct(hf) => match contract(&hf, &hf_shape, 8) {
                Ok(row) if row.matches() => rows.push(format_row(name, &ggml, &row)),
                Ok(row) => bad.push(format!("MISMATCH {}", format_row(name, &ggml, &row))),
                Err(err) => bad.push(format!("ERR {name}: {err:#}")),
            },
            super::super::names::GgufName::ExpertStack { proj, .. } => {
                if hf_shape.len() != 3 {
                    bad.push(format!("EXPERT_RANK {name} {hf_shape:?}"));
                    continue;
                }
                let (count, out, inn) = (hf_shape[0], hf_shape[1], hf_shape[2]);
                match contract_expert(proj, count, out, inn, 8) {
                    Ok(row) if row.matches() => rows.push(format_row(name, &ggml, &row)),
                    Ok(row) => bad.push(format!("MISMATCH {}", format_row(name, &ggml, &row))),
                    Err(err) => bad.push(format!("ERR {name}: {err:#}")),
                }
            }
        }
    }
    let out =
        std::env::var("K3_CONTRACT_OUT").unwrap_or_else(|_| "/tmp/k3-contract-table.txt".into());
    std::fs::write(&out, rows.join("\n") + "\n").unwrap();
    assert!(
        bad.is_empty(),
        "contract failures {}/{mapped}\n{}",
        bad.len(),
        bad.join("\n")
    );
    assert_eq!(mapped, 2573, "dry-bind count, table {out}");
}

const TP: usize = 8;
const HIDDEN: usize = 7168;
const LATENT: usize = 3584;
const EXPERT_INTER: usize = 3072;
const N_EXPERTS: usize = 896;

fn down_name() -> &'static str {
    "model.layers.1.block_sparse_moe.routed_expert_down_proj.weight"
}

#[test]
fn hidden_over_tp_is_not_the_expert_axis() {
    assert_eq!(HIDDEN / TP, N_EXPERTS);
    let row = contract(down_name(), &[LATENT, HIDDEN], TP).unwrap();
    assert!(row.matches());
    assert!(!row.replicated);
    assert_eq!(row.after_tp, vec![LATENT, HIDDEN / TP]);
    assert_eq!(row.tp_axis, "cols:hidden_over_tp");
    assert_eq!(row.numel_on_rank, LATENT * (HIDDEN / TP));
    assert_eq!(row.numel_on_rank, 3_211_264);
    assert_eq!(row.op_expected_numel, 3_211_264);
}

#[test]
fn sliced_routed_down_matches_3584_by_hidden_over_tp() {
    let stored = LATENT * (HIDDEN / TP);
    assert_eq!(stored, 3_211_264);
    let op = op_expected_numel(down_name(), &[LATENT, HIDDEN], TP).unwrap();
    assert_eq!(op, stored);
    let row = contract(down_name(), &[LATENT, HIDDEN], TP).unwrap();
    assert_eq!(row.tp_axis, "cols:hidden_over_tp");
    assert!(!row.tp_axis.contains("n_experts"));
    assert_eq!(row.after_tp, vec![LATENT, HIDDEN / TP]);
}

#[test]
fn routed_up_splits_hidden_not_latent() {
    let name = "model.layers.1.block_sparse_moe.routed_expert_up_proj.weight";
    let row = contract(name, &[HIDDEN, LATENT], TP).unwrap();
    assert!(row.matches() && !row.replicated);
    assert_eq!(row.after_tp, vec![HIDDEN / TP, LATENT]);
    assert_eq!(row.tp_axis, "rows:hidden_over_tp");
    assert_eq!(row.numel_on_rank, (HIDDEN / TP) * LATENT);
}

#[test]
fn k_b_splits_heads_not_the_b_proj_accident() {
    let name = "model.layers.3.self_attn.k_b_proj.weight";
    let row = contract(name, &[96, 512, 128], TP).unwrap();
    assert!(row.matches());
    assert_eq!(row.after_tp, vec![12, 512, 128]);
    assert_eq!(row.tp_axis, "rows:heads");
    assert!(!row.replicated);
}

#[test]
fn q_a_and_kv_a_stay_full_hidden() {
    let q = contract(
        "model.layers.3.self_attn.q_a_proj.weight",
        &[1536, HIDDEN],
        TP,
    )
    .unwrap();
    let kv = contract(
        "model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
        &[576, HIDDEN],
        TP,
    )
    .unwrap();
    assert!(q.replicated && q.matches());
    assert!(kv.replicated && kv.matches());
    assert_eq!(q.numel_on_rank, 1536 * HIDDEN);
}

#[test]
fn o_proj_splits_head_concat_not_hidden_over_tp() {
    let name = "model.layers.3.self_attn.o_proj.weight";
    let row = contract(name, &[HIDDEN, 96 * 128], TP).unwrap();
    assert!(row.matches());
    assert_eq!(row.after_tp, vec![HIDDEN, 12 * 128]);
    assert_eq!(row.tp_axis, "cols:head_concat_then_allreduce");
    assert_ne!(row.after_tp[1], HIDDEN / TP);
}

#[test]
fn experts_split_3072_and_keep_all_896() {
    let w1 = contract_expert("w1", N_EXPERTS, EXPERT_INTER, LATENT, TP).unwrap();
    assert!(w1.matches());
    assert_eq!(w1.after_tp, vec![N_EXPERTS, EXPERT_INTER / TP, LATENT]);
    assert_eq!(w1.tp_axis, "rows:expert_intermediate");
    // Same numel as splitting latent instead of 3072. The shape is the check.
    assert_ne!(w1.after_tp, vec![N_EXPERTS, EXPERT_INTER, LATENT / TP]);
    let w2 = contract_expert("w2", N_EXPERTS, LATENT, EXPERT_INTER, TP).unwrap();
    assert!(w2.matches());
    assert_eq!(w2.after_tp, vec![N_EXPERTS, LATENT, EXPERT_INTER / TP]);
}

#[test]
fn a_log_is_local_heads_because_the_gate_indexes_zero_to_heads() {
    let row = contract("model.layers.1.self_attn.A_log", &[96], TP).unwrap();
    assert!(row.matches());
    assert_eq!(row.after_tp, vec![12]);
    assert_eq!(row.tp_axis, "rows:heads");
}

#[test]
fn embed_and_lm_head_keep_full_vocab_and_hidden() {
    let emb = contract("model.embed_tokens.weight", &[163840, HIDDEN], TP).unwrap();
    let head = contract("lm_head.weight", &[163840, HIDDEN], TP).unwrap();
    assert!(emb.replicated && emb.matches());
    assert!(head.replicated && head.matches());
    assert_eq!(emb.numel_on_rank, 163840 * HIDDEN);
}

#[test]
fn row_text_names_replication_not_expert_count() {
    let row = contract(down_name(), &[LATENT, HIDDEN], TP).unwrap();
    let text = format_row("blk.1.ffn_routed_down.weight", &[HIDDEN, LATENT], &row);
    assert!(text.contains("hidden_over_tp"));
    assert!(text.contains("3211264"));
    assert!(!text.contains("n_experts"));
}

#[test]
fn ops_and_hopper_kernels_do_not_hardcode_the_production_widths() {
    const OPS: &str = include_str!("../../../../avarok-core/src/kimi_k3/ops.rs");
    const KDA: &str = include_str!("../../../../../kernels/hopper/kimi-k3/bf16/kda_decode.cu");
    const MLA: &str = include_str!("../../../../../kernels/hopper/kimi-k3/bf16/mla_decode.cu");
    const MOE: &str =
        include_str!("../../../../../kernels/hopper/kimi-k3/bf16/moe_w4a16_grouped_gemm.cu");
    for (label, src) in [("ops.rs", OPS), ("kda", KDA), ("mla", MLA), ("moe", MOE)] {
        for (lineno, line) in src.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            for token in ["896", "7168", "3584", "3072"] {
                assert!(
                    !code.contains(token),
                    "{label}:{} hardcodes {token}: {line}",
                    lineno + 1
                );
            }
        }
    }
}

#[test]
fn gpu_forward_shape_contract_prints_green() {
    let hidden_local = HIDDEN / TP; // 896 = hidden/tp, not n_experts
    assert_eq!(hidden_local, 896);
    assert_eq!(N_EXPERTS, 896);
    let shexp_local = 6144 / TP; // 768
    let dense_local = 33792 / TP; // 4224
    let heads_local = 96 / TP; // 12
    let qkv_local = heads_local * 128; // 1536
    let ops: [(&str, &[usize], &str, &str); 12] = [
        ("q/k/v/g_proj", &[qkv_local, HIDDEN], "rows:heads", "n/a"),
        (
            "o_proj",
            &[HIDDEN, qkv_local],
            "cols:head_concat_then_allreduce",
            "n/a",
        ),
        ("q_a_proj", &[1536, HIDDEN], "replicated", "n/a"),
        ("q_b_proj", &[heads_local * 192, 1536], "rows:heads", "n/a"),
        ("attn_k_b", &[heads_local, 512, 128], "rows:heads", "n/a"),
        ("attn_v_b", &[heads_local, 128, 512], "rows:heads", "n/a"),
        (
            "routed_down",
            &[LATENT, hidden_local],
            "cols:hidden_over_tp",
            "hidden/tp",
        ),
        (
            "routed_up",
            &[hidden_local, LATENT],
            "rows:hidden_over_tp",
            "hidden/tp",
        ),
        (
            "shexp gate/up",
            &[shexp_local, HIDDEN],
            "rows:out_features",
            "n/a",
        ),
        (
            "shexp down",
            &[HIDDEN, shexp_local],
            "cols:in_features_then_allreduce",
            "n/a",
        ),
        (
            "dense L0 gate/up",
            &[dense_local, HIDDEN],
            "rows:out_features",
            "n/a",
        ),
        (
            "expert w1 (one of 896)",
            &[EXPERT_INTER / TP, LATENT],
            "rows:out_features",
            "n_experts present",
        ),
    ];
    println!("op | on-rank shape | numel | op expects | tp axis | 896 meaning");
    for (name, shape, axis, eight96) in ops {
        let numel: usize = shape.iter().product();
        println!("{name} | {shape:?} | {numel} | {numel} | {axis} | {eight96}");
        assert_ne!(
            shape,
            &[LATENT, HIDDEN][..],
            "routed down must not be 3584x7168 on-rank"
        );
    }
    let down = contract(down_name(), &[LATENT, HIDDEN], TP).unwrap();
    assert_eq!(down.after_tp, vec![LATENT, hidden_local]);
    assert_eq!(down.numel_on_rank, LATENT * hidden_local);
    assert_ne!(down.numel_on_rank, LATENT * HIDDEN);
}

#[test]
fn k_b_and_v_b_are_not_one_fused_kv_b_matrix() {
    let k = contract(
        "model.layers.3.self_attn.k_b_proj.weight",
        &[96, 512, 128],
        TP,
    )
    .unwrap();
    let v = contract(
        "model.layers.3.self_attn.v_b_proj.weight",
        &[96, 128, 512],
        TP,
    )
    .unwrap();
    assert_eq!(k.after_tp, vec![12, 512, 128]);
    assert_eq!(v.after_tp, vec![12, 128, 512]);
    assert_ne!(k.after_tp, v.after_tp);
    assert_ne!(k.numel_on_rank, v.numel_on_rank);
}
