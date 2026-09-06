// SPDX-License-Identifier: AGPL-3.0-only

//! CPU tests for the `mtp.*` shape audit.
//!
//! Every store here is synthetic: `WeightStore::from_map` over
//! `DevicePtr::NULL` pointers with REAL shapes, so the whole audit runs with
//! no GPU and no checkpoint. The point of the audit is that it reads shapes
//! rather than names, and the only way to pin that cheaply is to hand it a
//! store whose names are all right and whose shapes are not.

use std::collections::HashMap;

use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::{WeightDtype, WeightTensor};

use super::*;

/// Shapes from `RadixArk/Qwen3.8-Flash-Next-NVFP4` config.json, with a
/// 4-layer body so the fixture stays small — the MTP audit never walks the
/// main layers, so the count does not matter to it.
fn cfg() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "qwen4_exp".to_string();
    c.hidden_size = 2560;
    c.num_attention_heads = 24;
    c.num_key_value_heads = 2;
    c.head_dim = 256;
    c.num_experts = 512;
    c.moe_intermediate_size = 640;
    c.shared_expert_intermediate_size = 640;
    c.hc_mult = 4;
    c.hc_lowrank = 320;
    c.index_n_heads = 4;
    c.index_head_dim = 128;
    c.index_topk = 2048;
    c.index_compress_ratio = 4;
    c.num_mtp_modules = 1;
    c.ep_rank = 0;
    c.ep_world_size = 1;
    c
}

struct Builder {
    map: HashMap<String, WeightTensor>,
}

impl Builder {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    fn put(&mut self, name: &str, shape: &[usize]) {
        self.put_dtype(name, shape, WeightDtype::BF16);
    }

    fn put_dtype(&mut self, name: &str, shape: &[usize], dtype: WeightDtype) {
        self.map.insert(
            name.to_string(),
            WeightTensor {
                // No GPU in this test: the audit reads `.shape` only, and a
                // NULL pointer is exactly what a never-uploaded tensor has.
                ptr: DevicePtr::NULL,
                shape: shape.to_vec(),
                dtype,
            },
        );
    }

    fn store(self) -> WeightStore {
        WeightStore::from_map(self.map)
    }
}

/// Every `mtp.*` tensor except the routed experts, at the shipped shapes.
fn common(b: &mut Builder) {
    let lp = MTP_LAYER_PREFIX;
    b.put(&format!("{lp}.self_attn.q_proj.weight"), &[12288, 2560]);
    b.put(&format!("{lp}.self_attn.k_proj.weight"), &[512, 2560]);
    b.put(&format!("{lp}.self_attn.v_proj.weight"), &[512, 2560]);
    b.put(&format!("{lp}.self_attn.o_proj.weight"), &[2560, 6144]);
    b.put(&format!("{lp}.self_attn.q_norm.weight"), &[256]);
    b.put(&format!("{lp}.self_attn.k_norm.weight"), &[256]);

    b.put(
        &format!("{lp}.self_attn.indexer.index_qk_proj.weight"),
        &[640, 2560],
    );
    b.put(
        &format!("{lp}.self_attn.indexer.q_layernorm.weight"),
        &[128],
    );
    b.put(
        &format!("{lp}.self_attn.indexer.k_layernorm.weight"),
        &[128],
    );

    for site in ["attn_hyper_connection", "mlp_hyper_connection"] {
        b.put(&format!("{lp}.{site}.hc_norm.weight"), &[10240]);
        b.put(
            &format!("{lp}.{site}.input_mix_weight_down.weight"),
            &[320, 10240],
        );
        b.put(
            &format!("{lp}.{site}.input_mix_weight_up.weight"),
            &[10240, 320],
        );
        b.put(
            &format!("{lp}.{site}.block_inject_weight.weight"),
            &[4, 10240],
        );
    }
    // The module mixer: THREE tensors, no block_inject.
    b.put("mtp.hyper_connection_mixer.hc_norm.weight", &[10240]);
    b.put(
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
        &[320, 10240],
    );
    b.put(
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
        &[10240, 320],
    );

    b.put(&format!("{lp}.mlp.gate.weight"), &[512, 2560]);
    b.put(
        &format!("{lp}.mlp.shared_expert.gate_proj.weight"),
        &[640, 2560],
    );
    b.put(
        &format!("{lp}.mlp.shared_expert.up_proj.weight"),
        &[640, 2560],
    );
    b.put(
        &format!("{lp}.mlp.shared_expert.down_proj.weight"),
        &[2560, 640],
    );
    b.put(&format!("{lp}.mlp.shared_expert_gate.weight"), &[1, 2560]);

    // The asymmetric pair — 2560 against 10240.
    b.put("mtp.pre_fc_norm_embedding.weight", &[2560]);
    b.put("mtp.pre_fc_norm_hidden.weight", &[10240]);
    b.put("mtp.fc_embedding.weight", &[2560, 2560]);
    b.put("mtp.fc_hidden.weight", &[2560, 2560]);
}

/// RadixArk: fused BF16 expert stacks.
fn fused_experts(b: &mut Builder) {
    let lp = MTP_LAYER_PREFIX;
    b.put(
        &format!("{lp}.mlp.experts.gate_up_proj"),
        &[512, 1280, 2560],
    );
    b.put(&format!("{lp}.mlp.experts.down_proj"), &[512, 2560, 640]);
}

/// Inferact: 512 per-expert NVFP4 triples. Only the first expert is built —
/// the audit probes the first LOCAL index, and under no-EP that is 0.
fn per_expert_nvfp4(b: &mut Builder) {
    let lp = MTP_LAYER_PREFIX;
    for e in 0..2usize {
        for p in ["gate_proj", "up_proj"] {
            // Packed E2M1: the on-disk width is HALF the logical one.
            b.put_dtype(
                &format!("{lp}.mlp.experts.{e}.{p}.weight"),
                &[640, 1280],
                WeightDtype::UInt8,
            );
            b.put_dtype(
                &format!("{lp}.mlp.experts.{e}.{p}.weight_scale"),
                &[640, 160],
                WeightDtype::FP8E4M3,
            );
        }
        b.put_dtype(
            &format!("{lp}.mlp.experts.{e}.down_proj.weight"),
            &[2560, 320],
            WeightDtype::UInt8,
        );
        b.put_dtype(
            &format!("{lp}.mlp.experts.{e}.down_proj.weight_scale"),
            &[2560, 40],
            WeightDtype::FP8E4M3,
        );
    }
}

fn fused_store() -> WeightStore {
    let mut b = Builder::new();
    common(&mut b);
    fused_experts(&mut b);
    b.store()
}

#[test]
fn complete_fused_store_passes() {
    let c = cfg();
    let r = audit_mtp_namespace(&fused_store(), &c);
    assert!(r.any_tensors);
    assert_eq!(r.expert_layout, MtpExpertLayout::Fused);
    assert!(r.missing.is_empty(), "{:?}", r.missing);
    assert!(r.shape_errors.is_empty(), "{:?}", r.shape_errors);
    r.ensure_loadable(&c).expect("fused store must load");
}

#[test]
fn complete_per_expert_nvfp4_store_passes() {
    let c = cfg();
    let mut b = Builder::new();
    common(&mut b);
    per_expert_nvfp4(&mut b);
    let r = audit_mtp_namespace(&b.store(), &c);
    assert_eq!(r.expert_layout, MtpExpertLayout::PerExpert);
    assert_eq!(r.first_local_expert, 0, "no EP ⇒ expert 0 is local");
    r.ensure_loadable(&c).expect("per-expert store must load");
}

#[test]
fn a_missing_combiner_tensor_is_named() {
    let c = cfg();
    let mut b = Builder::new();
    common(&mut b);
    fused_experts(&mut b);
    b.map.remove("mtp.fc_hidden.weight");
    let r = audit_mtp_namespace(&b.store(), &c);
    let err = r.ensure_loadable(&c).unwrap_err().to_string();
    assert!(err.contains("mtp.fc_hidden.weight"), "{err}");
}

/// The whole reason this audit checks shapes. `pre_fc_norm_hidden` at
/// `hidden_size` instead of `hc_mult * hidden_size` is the failure that reads
/// 2560 of 10240 elements and produces plausible-wrong activations.
#[test]
fn pre_fc_norm_hidden_at_the_embedding_width_is_refused() {
    let c = cfg();
    let mut b = Builder::new();
    common(&mut b);
    fused_experts(&mut b);
    b.put("mtp.pre_fc_norm_hidden.weight", &[2560]);
    let r = audit_mtp_namespace(&b.store(), &c);
    let err = r.ensure_loadable(&c).unwrap_err().to_string();
    assert!(err.contains("mtp.pre_fc_norm_hidden.weight"), "{err}");
    assert!(err.contains("2560"), "names the checkpoint width: {err}");
    assert!(err.contains("10240"), "names the expected width: {err}");
}

/// q_proj carries the query AND its sigmoid output gate: `2 * heads * head_dim`.
/// Single width is what a loader that sized it from o_proj would ship.
#[test]
fn single_width_q_proj_is_refused() {
    let c = cfg();
    let mut b = Builder::new();
    common(&mut b);
    fused_experts(&mut b);
    b.put(
        &format!("{MTP_LAYER_PREFIX}.self_attn.q_proj.weight"),
        &[6144, 2560],
    );
    let r = audit_mtp_namespace(&b.store(), &c);
    let err = r.ensure_loadable(&c).unwrap_err().to_string();
    assert!(err.contains("q_proj"), "{err}");
    assert!(err.contains("12288"), "{err}");
}

#[test]
fn neither_expert_layout_is_refused() {
    let c = cfg();
    let mut b = Builder::new();
    common(&mut b);
    let r = audit_mtp_namespace(&b.store(), &c);
    assert_eq!(r.expert_layout, MtpExpertLayout::Neither);
    let err = r.ensure_loadable(&c).unwrap_err().to_string();
    assert!(err.contains("no routed experts"), "{err}");
}

#[test]
fn both_expert_layouts_are_refused_as_ambiguous() {
    let c = cfg();
    let mut b = Builder::new();
    common(&mut b);
    fused_experts(&mut b);
    per_expert_nvfp4(&mut b);
    let r = audit_mtp_namespace(&b.store(), &c);
    assert_eq!(r.expert_layout, MtpExpertLayout::Both);
    let err = r.ensure_loadable(&c).unwrap_err().to_string();
    assert!(err.contains("BOTH expert layouts"), "{err}");
}

/// The DEFAULT serving state: `skip_mtp` dropped every `mtp.*` tensor at
/// upload, and nothing declared MTP, so an empty report is correct and
/// `ensure_loadable` must not object.
#[test]
fn an_empty_store_is_reported_and_is_ok_when_no_module_is_declared() {
    let mut c = cfg();
    let store = WeightStore::from_map(HashMap::new());
    let r = audit_mtp_namespace(&store, &c);
    assert!(!r.any_tensors, "no mtp.* tensors");
    assert!(!r.missing.is_empty(), "every tensor is missing");

    c.num_mtp_modules = 0;
    r.ensure_loadable(&c)
        .expect("num_mtp_modules=0 ⇒ nothing to load");

    c.num_mtp_modules = 1;
    let err = r.ensure_loadable(&c).unwrap_err().to_string();
    assert!(err.contains("no `mtp.*` tensors"), "{err}");
}

/// `load_moe_qwen35` honours `is_local_expert` and has no `force_all_experts`
/// parameter, while the weight upload never shards `mtp.*` — so under EP the
/// draft would route into NULL experts on every rank.
#[test]
fn ep_world_size_above_one_is_refused() {
    let mut c = cfg();
    c.ep_world_size = 2;
    let r = audit_mtp_namespace(&fused_store(), &c);
    let err = r.ensure_loadable(&c).unwrap_err().to_string();
    assert!(err.contains("ep_world_size=2"), "{err}");
}

// ── The offline checkpoint arm ────────────────────────────────────────────
//
// Everything above is synthetic. This one reads the REAL safetensors headers
// — no GPU, no model load, no upload — builds a store of NULL pointers with
// the checkpoint's own shapes, and runs the same audit over it. It is the arm
// that catches a third checkpoint revision before it costs a 75 GB load.

/// One safetensors header, parsed without reading any tensor data.
#[cfg(test)]
fn read_header(path: &std::path::Path) -> serde_json::Value {
    use std::io::Read;
    let mut fh = std::fs::File::open(path).expect("open shard");
    let mut len = [0u8; 8];
    fh.read_exact(&mut len).expect("header length");
    let mut hdr = vec![0u8; u64::from_le_bytes(len) as usize];
    fh.read_exact(&mut hdr).expect("header bytes");
    serde_json::from_slice(&hdr).expect("header JSON")
}

fn dtype_of(s: &str) -> WeightDtype {
    match s {
        "U8" => WeightDtype::UInt8,
        "F8_E4M3" => WeightDtype::FP8E4M3,
        "F32" => WeightDtype::FP32,
        "I64" => WeightDtype::Int64,
        // The audit reads shapes only; anything else lands as BF16, which is
        // what every non-expert `mtp.*` tensor actually is in both snapshots.
        _ => WeightDtype::BF16,
    }
}

/// Walk `model.safetensors.index.json` for `mtp.*` and assert the observed key
/// set and shapes are exactly what `audit_mtp_namespace` demands.
///
///     ATLAS_QWEN4EXP_CKPT=/path/to/snapshot \
///       cargo test -p spark-model mtp_header_inventory
#[test]
fn mtp_header_inventory() {
    let Ok(snap) = std::env::var("ATLAS_QWEN4EXP_CKPT") else {
        println!("ATLAS_QWEN4EXP_CKPT unset — skipping real-checkpoint inventory");
        return;
    };
    let snap = std::path::Path::new(&snap);

    let cfg_text = std::fs::read_to_string(snap.join("config.json")).expect("read config.json");
    let config = atlas_core::config::parse_config(&cfg_text).expect("parse config.json");
    assert_eq!(
        config.num_mtp_modules, 1,
        "the parser must see text_config.mtp before the audit means anything"
    );

    let idx: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(snap.join("model.safetensors.index.json"))
            .expect("read index.json"),
    )
    .expect("index.json is valid JSON");
    let wm = idx["weight_map"].as_object().expect("weight_map");

    // Header per shard, opened once — some revisions spread `mtp.*` across
    // three files, others put all 6173 in one shard no main-model tensor
    // references.
    let mut headers: HashMap<String, serde_json::Value> = HashMap::new();
    let mut b = Builder::new();
    let mut count = 0usize;
    for (name, file) in wm {
        if !name.starts_with("mtp.") {
            continue;
        }
        let file = file.as_str().expect("shard name");
        let hdr = headers
            .entry(file.to_string())
            .or_insert_with(|| read_header(&snap.join(file)));
        let e = &hdr[name.as_str()];
        let shape: Vec<usize> = e["shape"]
            .as_array()
            .unwrap_or_else(|| panic!("{name}: no shape in header"))
            .iter()
            .map(|v| v.as_u64().expect("shape dim") as usize)
            .collect();
        b.put_dtype(
            name,
            &shape,
            dtype_of(e["dtype"].as_str().unwrap_or("BF16")),
        );
        count += 1;
    }
    println!(
        "{}: {count} mtp.* tensors across {} shard(s)",
        snap.display(),
        headers.len()
    );
    assert!(count > 0, "no mtp.* tensors in {}", snap.display());

    let store = b.store();
    let r = audit_mtp_namespace(&store, &config);
    println!(
        "layout={:?} missing={:?} shape_errors={:?} unexpected={:?}",
        r.expert_layout, r.missing, r.shape_errors, r.unexpected
    );
    r.ensure_loadable(&config)
        .expect("the shipped checkpoint must satisfy the audit");
}
