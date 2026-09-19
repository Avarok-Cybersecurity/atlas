#!/usr/bin/env python3
"""Slice 7 — audit EVERY KDA attention block in the checkpoint, and extract per-layer packets.

Read-only over `LibertAIDAI/GLM-5.3-Flash-NVFP4` @ 9e0d74e3. Writes:
  * kda_family_audit.json — per-layer tensor names / shapes / dtypes / sha256, plus the
    config-derived layer taxonomy and a strict accounting of every layer-scoped tensor.
  * layer<L>.safetensors  — standalone packets for the layers named in EXTRACT.

Nothing is deleted; nothing outside the output dir is written.
"""
import hashlib, json, os, struct, sys

SNAP = ("/home/cluster/hf-glm53/hub/models--LibertAIDAI--GLM-5.3-Flash-NVFP4/"
        "snapshots/9e0d74e3cef17f634e84fb8e2223707e02616290")
OUT = "/home/cluster/kda-family"
PREFIX = "model.language_model.layers."
EXTRACT = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else [])]

os.makedirs(OUT, exist_ok=True)
cfg = json.load(open(SNAP + "/config.json"))["text_config"]
lin = cfg["linear_attn_config"]
KDA = lin["kda_layers"]
H, D, KS = lin["num_heads"], lin["head_dim"], lin["short_conv_kernel_size"]
HID = cfg["hidden_size"]
QKV = H * D
idx = json.load(open(SNAP + "/model.safetensors.index.json"))["weight_map"]

# The 15 self_attn tensors a KDA block must have, and nothing else.
EXPECT = {
    "self_attn.q_proj.weight":   ("BF16", [QKV, HID]),
    "self_attn.k_proj.weight":   ("BF16", [QKV, HID]),
    "self_attn.v_proj.weight":   ("BF16", [QKV, HID]),
    "self_attn.q_conv1d.weight": ("BF16", [QKV, 1, KS]),
    "self_attn.k_conv1d.weight": ("BF16", [QKV, 1, KS]),
    "self_attn.v_conv1d.weight": ("BF16", [QKV, 1, KS]),
    "self_attn.f_a_proj.weight": ("BF16", [D, HID]),
    "self_attn.f_b_proj.weight": ("BF16", [QKV, D]),
    "self_attn.g_a_proj.weight": ("BF16", [D, HID]),
    "self_attn.g_b_proj.weight": ("BF16", [QKV, D]),
    "self_attn.b_proj.weight":   ("BF16", [H, HID]),
    "self_attn.A_log":           ("F32",  [H]),
    "self_attn.dt_bias":         ("F32",  [QKV]),
    "self_attn.o_norm.weight":   ("BF16", [D]),
    "self_attn.o_proj.weight":   ("BF16", [HID, QKV]),
}
EXTRA = {"input_layernorm.weight": ("BF16", [HID])}  # decoder-layer norm, carried for completeness

# group every wanted tensor by shard so each shard is opened at most once
by_layer = {}
for name, shard in idx.items():
    if not name.startswith(PREFIX):
        continue
    rest = name[len(PREFIX):]
    L, _, short = rest.partition(".")
    by_layer.setdefault(int(L), {})[short] = shard

hdr_cache = {}
def header(shard):
    if shard not in hdr_cache:
        with open(SNAP + "/" + shard, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr_cache[shard] = (json.loads(fh.read(n)), 8 + n)
    return hdr_cache[shard]

def read(L, shard, short):
    # safetensors headers are keyed by the FULL tensor name, not the layer-relative one.
    name = f"{PREFIX}{L}.{short}"
    h, base = header(shard)
    a, b = h[name]["data_offsets"]
    with open(SNAP + "/" + shard, "rb") as fh:
        fh.seek(base + a)
        return h[name]["dtype"], h[name]["shape"], fh.read(b - a)

def meta(L, shard, short):
    h, _ = header(shard)
    m = h[f"{PREFIX}{L}.{short}"]
    return m["dtype"], m["shape"]

audit = {
    "checkpoint": "LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3cef17f634e84fb8e2223707e02616290",
    "geometry": {"hidden": HID, "heads": H, "head_dim": D, "qkv_dim": QKV,
                 "conv_dim": 3 * QKV, "conv_kernel": KS,
                 "gate_lower_bound": lin["gate_lower_bound"],
                 "rms_norm_eps": cfg["rms_norm_eps"], "hidden_act": cfg["hidden_act"],
                 "num_hidden_layers": cfg["num_hidden_layers"]},
    "kda_layers": KDA, "full_attn_layers": lin["full_attn_layers"],
    "layer_types": cfg["layer_types"], "mlp_layer_types": cfg.get("mlp_layer_types"),
    "layers": {}, "problems": [],
}

for L in sorted(by_layer):
    ts = by_layer[L]
    is_kda = L in KDA
    attn = {k: v for k, v in ts.items() if k.startswith("self_attn.")}
    rec = {
        "is_kda": is_kda,
        "attn_type": cfg["layer_types"][L] if L < len(cfg["layer_types"]) else "MTP(45)",
        "ffn_type": (cfg.get("mlp_layer_types") or [None] * 99)[L]
                    if L < len(cfg.get("mlp_layer_types") or []) else None,
        "n_tensors_total": len(ts),
        "n_self_attn": len(attn),
        "self_attn": {},
        "non_attn_prefixes": sorted({k.split(".")[0] for k in ts if not k.startswith("self_attn.")}),
    }
    for short in sorted(attn):
        dt, sh = meta(L, ts[short], short)
        rec["self_attn"][short] = {"dtype": dt, "shape": sh, "shard": ts[short]}
    if is_kda:
        got, want = set(attn), set(EXPECT)
        if got != want:
            audit["problems"].append(
                {"layer": L, "missing": sorted(want - got), "unexpected": sorted(got - want)})
        for short, (edt, esh) in EXPECT.items():
            if short not in attn:
                continue
            dt, sh = meta(L, ts[short], short)
            if dt != edt or list(sh) != esh:
                audit["problems"].append(
                    {"layer": L, "tensor": short, "want": [edt, esh], "got": [dt, sh]})
        # quantisation artefacts anywhere in this block
        sus = [k for k in ts if ("scale" in k.lower() or k.endswith("_zp") or "quant" in k.lower())
               and k.startswith("self_attn.")]
        rec["quant_artefacts_in_self_attn"] = sus
        if sus:
            audit["problems"].append({"layer": L, "quant_artefacts": sus})
    audit["layers"][str(L)] = rec

# per-tensor sha256 for every KDA block, so uniformity claims carry provenance
for L in KDA:
    ts = by_layer[L]
    h = {}
    for short in sorted(EXPECT):
        _, _, raw = read(L, ts[short], short)
        h[short] = hashlib.sha256(raw).hexdigest()[:32]
    audit["layers"][str(L)]["sha256"] = h

# extraction
for L in EXTRACT:
    ts = by_layer[L]
    want = sorted(EXPECT) + [k for k in EXTRA if k in ts]
    blobs, out_hdr, off = [], {}, 0
    for short in want:
        dt, sh, raw = read(L, ts[short], short)
        out_hdr[short] = {"dtype": dt, "shape": sh, "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        blobs.append(raw)
    out_hdr["__metadata__"] = {
        "source_repo": "LibertAIDAI/GLM-5.3-Flash-NVFP4",
        "source_revision": "9e0d74e3cef17f634e84fb8e2223707e02616290",
        "layer": str(L), "prefix": PREFIX,
    }
    hj = json.dumps(out_hdr).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(f"{OUT}/layer{L}.safetensors", "wb") as fh:
        fh.write(struct.pack("<Q", len(hj))); fh.write(hj)
        for b_ in blobs:
            fh.write(b_)
    print(f"  extracted layer {L}: {len(want)} tensors, {off} B ({off/1048576:.1f} MiB)",
          file=sys.stderr)

with open(f"{OUT}/kda_family_audit.json", "w") as fh:
    json.dump(audit, fh, indent=1, sort_keys=True)
print("KDA layers:", len(KDA), "| problems:", len(audit["problems"]), file=sys.stderr)
print("audit sha256",
      hashlib.sha256(open(f"{OUT}/kda_family_audit.json", "rb").read()).hexdigest(), file=sys.stderr)
