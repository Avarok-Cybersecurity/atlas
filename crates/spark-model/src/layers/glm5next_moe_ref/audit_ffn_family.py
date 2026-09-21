"""Slice 10 gate 2 — complete FFN/MoE quantization audit + MoE packet extraction."""
import json, re, struct, hashlib, sys, collections, os
S = ("/home/cluster/hf-glm53/hub/models--LibertAIDAI--GLM-5.3-Flash-NVFP4/"
     "snapshots/9e0d74e3cef17f634e84fb8e2223707e02616290")
idx = json.load(open(S + "/model.safetensors.index.json"))["weight_map"]

# ---- header cache -----------------------------------------------------------
_h = {}
def hdr(shard):
    if shard not in _h:
        fh = open(S + "/" + shard, "rb")
        n = struct.unpack("<Q", fh.read(8))[0]
        _h[shard] = (fh, json.loads(fh.read(n)), 8 + n)
    return _h[shard]

def meta(name):
    fh, h, base = hdr(idx[name])
    m = h[name]
    return m["dtype"], tuple(m["shape"]), base, m["data_offsets"], fh

# ---- inventory --------------------------------------------------------------
ffn = [k for k in idx if re.search(r"\.layers\.\d+\.mlp\.", k)]
print("FFN/MoE tensors total: %d" % len(ffn))

def leaf(k):
    return k.split(".mlp.", 1)[1]

def canon(l):
    return re.sub(r"experts\.\d+\.", "experts.E.", l)

by_layer = collections.defaultdict(list)
for k in ffn:
    by_layer[int(re.search(r"layers\.(\d+)\.", k).group(1))].append(k)

# dtype census by leaf family
census = collections.defaultdict(lambda: collections.Counter())
shapes = collections.defaultdict(set)
for k in ffn:
    dt, sh, *_ = meta(k)
    c = canon(leaf(k))
    census[c][dt] += 1
    shapes[c].add(sh)

print("\n%-42s %-9s %6s  shapes" % ("leaf family (canonical)", "dtype", "count"))
for c in sorted(census):
    for dt, n in sorted(census[c].items()):
        sh = sorted(shapes[c])
        print("%-42s %-9s %6d  %s" % (c, dt, n, sh if len(sh) <= 2 else f"{len(sh)} distinct"))

# structural signature per layer
sig = collections.defaultdict(list)
for lyr, ks in by_layer.items():
    s = tuple(sorted({canon(leaf(k)) for k in ks}))
    sig[s].append(lyr)
print("\ndistinct per-layer FFN signatures: %d" % len(sig))
for s, ls in sorted(sig.items(), key=lambda kv: kv[1][0]):
    n_t = len(by_layer[ls[0]])
    print("  layers %s  (%d layers, %d tensors each)" %
          (ls if len(ls) < 8 else f"{ls[:4]}...{ls[-1]}", len(ls), n_t))
    for x in s:
        print("      ", x)

# unknown check against the Slice-1 classifier's known roles
KNOWN = {"gate.weight", "gate.e_score_correction_bias",
         "gate_proj.weight", "up_proj.weight", "down_proj.weight",
         "gate_proj.weight_scale", "up_proj.weight_scale", "down_proj.weight_scale",
         "gate_proj.weight_scale_2", "up_proj.weight_scale_2", "down_proj.weight_scale_2",
         "gate_proj.input_scale", "up_proj.input_scale", "down_proj.input_scale"}
unknown = sorted({canon(leaf(k)).replace("experts.E.", "").replace("shared_experts.", "")
                  for k in ffn} - KNOWN)
print("\nleaf names outside the expected set: %s" % (unknown or "NONE"))

# ---- packet extraction ------------------------------------------------------
OUT = "/home/cluster/moe-family"
os.makedirs(OUT, exist_ok=True)
def write_packet(path, names, rename):
    blobs, out_hdr, off = [], {}, 0
    for k in names:
        dt, sh, base, (a, b), fh = meta(k)
        fh.seek(base + a); raw = fh.read(b - a)
        out_hdr[rename(k)] = {"dtype": dt, "shape": list(sh),
                              "data_offsets": [off, off + len(raw)]}
        off += len(raw); blobs.append(raw)
    out_hdr["__metadata__"] = {"source_repo": "LibertAIDAI/GLM-5.3-Flash-NVFP4",
                               "source_revision": "9e0d74e3cef17f634e84fb8e2223707e02616290"}
    hj = json.dumps(out_hdr).encode(); hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as w:
        w.write(struct.pack("<Q", len(hj))); w.write(hj)
        for b_ in blobs: w.write(b_)
    print("  -> %s  %d tensors, %.2f MiB" % (path, len(names), off / 1048576))

# router of layers 3 / 23 / 44 / 45
for lyr in (3, 23, 44, 45):
    names = [k for k in by_layer[lyr] if ".mlp.gate." in k or leaf(k) == "gate.weight"]
    write_packet(f"{OUT}/router_layer{lyr}.safetensors", sorted(names),
                 lambda k: leaf(k))
# dense FFN layer 0
write_packet(f"{OUT}/dense_layer0.safetensors", sorted(by_layer[0]), lambda k: leaf(k))
# layer 3: shared expert + experts 0..3 (smallest real NVFP4 packet)
sel = [k for k in by_layer[3] if ".shared_experts." in k]
for e in range(4):
    sel += [k for k in by_layer[3] if f".experts.{e}." in k]
write_packet(f"{OUT}/moe_layer3_e0_3.safetensors", sorted(sel), lambda k: leaf(k))
