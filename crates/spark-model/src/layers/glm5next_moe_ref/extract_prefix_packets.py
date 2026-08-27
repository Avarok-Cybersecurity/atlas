"""Slice 10 gate 5 — extract the packet needed to run the REAL prefix up to layer 3's router.

Router input = post_attention_layernorm( ffn_hc_collapse( layer-3 attention output ) ),
which needs embed_tokens + layers 0,1,2 complete + layer 3's attention/norms/hc. Nothing
short of that is a real activation; synthetic LCG vectors are not the model's distribution.
"""
import json, re, struct, os, sys
S = ("/home/cluster/hf-glm53/hub/models--LibertAIDAI--GLM-5.3-Flash-NVFP4/"
     "snapshots/9e0d74e3cef17f634e84fb8e2223707e02616290")
idx = json.load(open(S + "/model.safetensors.index.json"))["weight_map"]
OUT = "/home/cluster/prefix-family"
os.makedirs(OUT, exist_ok=True)
_h = {}
def hdr(sh):
    if sh not in _h:
        fh = open(S + "/" + sh, "rb"); n = struct.unpack("<Q", fh.read(8))[0]
        _h[sh] = (fh, json.loads(fh.read(n)), 8 + n)
    return _h[sh]

def write(path, names, rename):
    blobs, out, off = [], {}, 0
    for k in names:
        fh, h, base = hdr(idx[k]); m = h[k]
        a, b = m["data_offsets"]; fh.seek(base + a); raw = fh.read(b - a)
        out[rename(k)] = {"dtype": m["dtype"], "shape": m["shape"],
                          "data_offsets": [off, off + len(raw)]}
        off += len(raw); blobs.append(raw)
    out["__metadata__"] = {"source_revision": "9e0d74e3cef17f634e84fb8e2223707e02616290"}
    hj = json.dumps(out).encode(); hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as w:
        w.write(struct.pack("<Q", len(hj))); w.write(hj)
        for x in blobs: w.write(x)
    print("  -> %s  %d tensors  %.1f MiB" % (path, len(names), off / 1048576), file=sys.stderr)

write(f"{OUT}/embed.safetensors",
      ["model.language_model.embed_tokens.weight"], lambda k: "embed_tokens.weight")
for L in (0, 1, 2, 3):
    pfx = f"model.language_model.layers.{L}."
    names = [k for k in idx if k.startswith(pfx) and ".mlp.experts." not in k
             and ".mlp.gate." not in k and ".mlp.shared_experts." not in k]
    write(f"{OUT}/prefix_layer{L}.safetensors", sorted(names), lambda k: k[len(pfx):])
