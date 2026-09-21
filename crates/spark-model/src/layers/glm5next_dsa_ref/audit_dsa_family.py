#!/usr/bin/env python3
"""Slice 8 Gate 3 — quantisation audit of every DSA attention block, and packet extraction.

Floor C is LIVE for DSA: unlike KDA, these blocks may be quantised. Nothing is assumed —
every tensor family is classified from the safetensors header, per layer, across all 12.
"""
import hashlib, json, os, struct, sys
from collections import defaultdict

SNAP = ("/home/cluster/hf-glm53/hub/models--LibertAIDAI--GLM-5.3-Flash-NVFP4/"
        "snapshots/9e0d74e3cef17f634e84fb8e2223707e02616290")
OUT = "/home/cluster/dsa-family"
PREFIX = "model.language_model.layers."
EXTRACT = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] else [])]

os.makedirs(OUT, exist_ok=True)
cfg = json.load(open(SNAP + "/config.json"))["text_config"]
qcfg = json.load(open(SNAP + "/config.json")).get("quantization_config", {})
DSA = cfg["linear_attn_config"]["full_attn_layers"]
MTP = cfg["num_hidden_layers"]          # 45 -- the MTP layer sits one past the text stack
ALL = DSA + [MTP]
idx = json.load(open(SNAP + "/model.safetensors.index.json"))["weight_map"]

by_layer = defaultdict(dict)
for name, shard in idx.items():
    if not name.startswith(PREFIX):
        continue
    L, _, short = name[len(PREFIX):].partition(".")
    by_layer[int(L)][short] = shard

hdr_cache = {}
def header(shard):
    if shard not in hdr_cache:
        with open(SNAP + "/" + shard, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr_cache[shard] = (json.loads(fh.read(n)), 8 + n)
    return hdr_cache[shard]

def meta(L, shard, short):
    h, _ = header(shard)
    m = h[f"{PREFIX}{L}.{short}"]
    return m["dtype"], m["shape"]

def read(L, shard, short):
    name = f"{PREFIX}{L}.{short}"
    h, base = header(shard)
    a, b = h[name]["data_offsets"]
    with open(SNAP + "/" + shard, "rb") as fh:
        fh.seek(base + a)
        return h[name]["dtype"], h[name]["shape"], fh.read(b - a)

def family(short):
    """Collapse a tensor name to its family: strip the quant suffix."""
    for suf in (".weight_scale_2", ".weight_scale_inv", ".weight_scale", ".input_scale",
                ".weight_packed", ".weight", ".bias"):
        if short.endswith(suf):
            return short[: -len(suf)], suf
    return short, ""

audit = {
    "checkpoint": "LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3cef17f634e84fb8e2223707e02616290",
    "dsa_text_layers": DSA, "mtp_layer": MTP,
    "quantization_config": {k: v for k, v in qcfg.items() if k != "ignore"},
    "quant_ignore_len": len(qcfg.get("ignore", [])),
    "quant_ignore_sample": qcfg.get("ignore", [])[:12],
    "geometry": {k: cfg.get(k) for k in
                 ("hidden_size", "num_attention_heads", "num_key_value_heads", "q_lora_rank",
                  "kv_lora_rank", "v_head_dim", "qk_nope_head_dim", "qk_rope_head_dim",
                  "index_topk", "index_kpool", "index_n_heads", "index_head_dim",
                  "index_kpool_always_select_tail", "rms_norm_eps", "num_hidden_layers")},
    "indexer_types_distinct": sorted(set(cfg.get("indexer_types", []))),
    "layers": {}, "problems": [],
}

for L in ALL:
    ts = by_layer[L]
    attn = {k: v for k, v in ts.items() if k.startswith("self_attn.")}
    rec = {"n_self_attn": len(attn), "self_attn": {}, "families": {},
           "non_attn_prefixes": sorted({k.split(".")[0] for k in ts if not k.startswith("self_attn.")})}
    fams = defaultdict(dict)
    for short in sorted(attn):
        dt, sh = meta(L, ts[short], short)
        rec["self_attn"][short] = {"dtype": dt, "shape": sh}
        base, suf = family(short)
        fams[base][suf or "<bare>"] = [dt, sh]
    rec["families"] = {k: v for k, v in sorted(fams.items())}
    audit["layers"][str(L)] = rec

# uniformity across the 12
sigs = defaultdict(list)
for L in ALL:
    r = audit["layers"][str(L)]
    s = tuple(sorted((n, v["dtype"], tuple(v["shape"])) for n, v in r["self_attn"].items()))
    sigs[s].append(L)
audit["signature_groups"] = [{"layers": v, "n_tensors": len(k),
                              "tensors": [{"name": n, "dtype": d, "shape": list(sh)} for n, d, sh in k]}
                             for k, v in sigs.items()]

# per-tensor sha256 for the whole DSA family
for L in ALL:
    ts = by_layer[L]
    audit["layers"][str(L)]["sha256"] = {
        s: hashlib.sha256(read(L, ts[s], s)[2]).hexdigest()[:32]
        for s in sorted(k for k in ts if k.startswith("self_attn."))}

for L in EXTRACT:
    ts = by_layer[L]
    want = sorted(k for k in ts if k.startswith("self_attn.")) + \
           [k for k in ("input_layernorm.weight", "post_attention_layernorm.weight") if k in ts]
    blobs, out_hdr, off = [], {}, 0
    for short in want:
        dt, sh, raw = read(L, ts[short], short)
        out_hdr[short] = {"dtype": dt, "shape": sh, "data_offsets": [off, off + len(raw)]}
        off += len(raw); blobs.append(raw)
    out_hdr["__metadata__"] = {"source_repo": "LibertAIDAI/GLM-5.3-Flash-NVFP4",
                               "source_revision": "9e0d74e3cef17f634e84fb8e2223707e02616290",
                               "layer": str(L), "prefix": PREFIX}
    hj = json.dumps(out_hdr).encode(); hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(f"{OUT}/dsa_layer{L}.safetensors", "wb") as fh:
        fh.write(struct.pack("<Q", len(hj))); fh.write(hj)
        for b_ in blobs: fh.write(b_)
    print(f"  extracted DSA layer {L}: {len(want)} tensors, {off} B ({off/1048576:.1f} MiB)", file=sys.stderr)

with open(f"{OUT}/dsa_family_audit.json", "w") as fh:
    json.dump(audit, fh, indent=1, sort_keys=True)
print(f"DSA blocks audited: {len(ALL)} (11 text + MTP {MTP}) | signature groups: {len(sigs)}", file=sys.stderr)
print("audit sha256", hashlib.sha256(open(f"{OUT}/dsa_family_audit.json","rb").read()).hexdigest(), file=sys.stderr)
