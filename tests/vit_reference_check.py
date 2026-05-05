#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Diff Atlas's ViT output against an HF transformers reference.

Pipeline:
  1. Run Atlas serving a vision model with `ATLAS_DUMP_VIT=/tmp/atlas_vit`.
     One image request produces `patch_embed.bin`, `block00.bin`,
     `block01.bin`, …, `block26.bin`, `final.bin`. All are BF16.
  2. Run this script. It loads the SAME checkpoint into HF transformers on
     CPU, runs the same Mona Lisa JPEG through HF's vision tower, and
     per-checkpoint dumps parallel .bin files at the same layer names.
  3. For each layer, compute cosine similarity + max-abs-diff against
     Atlas's dump. Print a table; any layer below cosine 0.90 is the
     first divergence and worth investigating.

The goal is NOT bit-exact match (Atlas does FP8 dequant + BF16 GEMM, HF
does all-BF16 or FP16); it's to localize the FIRST block where Atlas
diverges noticeably from HF, so we can focus on fixing that block's
kernel rather than guessing.

Usage:
  # on Atlas host, start server with dump:
  sudo docker run -d --name atlas-vit-debug ... \
      -e ATLAS_DUMP_VIT=/tmp/atlas_vit ...

  # send one image request (the preprocess fires + writes dumps):
  python3 tests/vit_reference_check.py --mode=trigger

  # compute HF reference + diff:
  python3 tests/vit_reference_check.py --mode=diff \
      --hf-id Qwen/Qwen3-VL-2B-Instruct \
      --atlas-dump /tmp/atlas_vit
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mona_lisa.jpeg"


def bf16_bytes_to_f32(data: bytes):
    import numpy as np
    bf16 = np.frombuffer(data, dtype=np.uint16)
    f32_bits = bf16.astype(np.uint32) << 16
    return f32_bits.view(np.float32).copy()


def load_atlas_dumps(dump_dir: Path):
    """Load Atlas's per-layer BF16 dumps; return {label: f32 tensor}."""
    import numpy as np
    out = {}
    for p in sorted(dump_dir.glob("*.bin")):
        label = p.stem
        out[label] = bf16_bytes_to_f32(p.read_bytes())
    return out


def encode_image_for_atlas(max_dim: int = 320) -> tuple[str, int, int]:
    from PIL import Image
    img = Image.open(FIXTURE).convert("RGB")
    w, h = img.size
    scale = max_dim / min(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return (
        "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
        img.size[0],
        img.size[1],
    )


def trigger_atlas(base_url: str, model_id: str) -> None:
    """Send one Mona Lisa request to an already-running Atlas server that
    was launched with ATLAS_DUMP_VIT set. The dump happens as a side
    effect of the ViT forward pass."""
    data_url, w, h = encode_image_for_atlas()
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": "What is this?"},
        ]}],
        "max_tokens": 32,
        "reasoning_effort": "none",
    }
    r = httpx.post(f"{base_url}/chat/completions", timeout=300, json=body)
    print(f"atlas response {r.status_code}: "
          f"{(r.json().get('choices', [{}])[0].get('message', {}).get('content', '')[:200]) or r.text[:200]}")


def compute_hf_reference(hf_id: str, out_dir: Path) -> None:
    """Load HF model on CPU, run Mona Lisa through vision tower, dump
    per-block outputs so we can diff against Atlas."""
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading {hf_id} (BF16, CPU)…", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        hf_id, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)

    img = Image.open(FIXTURE).convert("RGB")
    w, h = img.size
    s = 320 / min(w, h)
    img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)

    inputs = processor(images=[img], text="x", return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(dtype=torch.bfloat16)
    grid_thw = inputs["image_grid_thw"]
    print(f"hf grid_thw={grid_thw.tolist()}  pixel_values={list(pixel_values.shape)}",
          flush=True)

    vision = model.visual if hasattr(model, "visual") else model.model.visual
    captures: dict[str, torch.Tensor] = {}

    def cap(name):
        def _hook(mod, inp, out):
            t = out if isinstance(out, torch.Tensor) else out[0]
            captures[name] = t.detach().to(torch.bfloat16).cpu()
        return _hook

    # Try to hook each block's output; API shape depends on HF version.
    blocks = getattr(vision, "blocks", None) or getattr(vision, "layers", None)
    hooks = []
    if blocks is not None:
        for i, blk in enumerate(blocks):
            hooks.append(blk.register_forward_hook(cap(f"block{i:02}")))

    with torch.no_grad():
        vision_out = vision(pixel_values, grid_thw)
    # HF returns either a tensor, a tuple, or a dataclass (e.g.
    # BaseModelOutputWithDeepstackFeatures). Normalize to a tensor.
    if isinstance(vision_out, torch.Tensor):
        final_tensor = vision_out
    elif isinstance(vision_out, tuple):
        final_tensor = vision_out[0]
    else:
        # dataclass-style output: prefer `last_hidden_state`, else first
        # tensor attribute found.
        final_tensor = getattr(vision_out, "last_hidden_state", None)
        if final_tensor is None:
            for attr in ("hidden_states", "image_embeds", "pooler_output"):
                v = getattr(vision_out, attr, None)
                if isinstance(v, torch.Tensor):
                    final_tensor = v
                    break
        if final_tensor is None:
            for k, v in vars(vision_out).items():
                if isinstance(v, torch.Tensor):
                    final_tensor = v
                    break
    captures["final"] = final_tensor.detach().to(torch.bfloat16).cpu()

    for h in hooks:
        h.remove()

    for label, tensor in captures.items():
        arr = tensor.flatten().view(torch.uint16).numpy()
        (out_dir / f"{label}.bin").write_bytes(arr.tobytes())
        print(f"  wrote {label}.bin  shape={list(tensor.shape)}", flush=True)


def diff_dumps(atlas_dir: Path, hf_dir: Path) -> None:
    """Compare Atlas's BF16 dumps against HF's layer-by-layer. Emit a
    cosine-similarity / max-abs-diff table."""
    import numpy as np
    atlas = load_atlas_dumps(atlas_dir)
    hf = load_atlas_dumps(hf_dir)
    labels = sorted(set(atlas) & set(hf),
                    key=lambda s: (s != "patch_embed",
                                   s != "final",
                                   int(s[5:]) if s.startswith("block") else 999))

    print(f"\n{'layer':15s} {'atlas#':>9s} {'hf#':>9s} "
          f"{'cos_sim':>9s} {'max|d|':>10s} {'rel_l2':>9s}")
    print("-" * 70)
    for label in labels:
        a = atlas[label]
        b = hf[label]
        n = min(a.size, b.size)
        a1 = a[:n]
        b1 = b[:n]
        cos = float(np.dot(a1, b1) / (np.linalg.norm(a1) * np.linalg.norm(b1) + 1e-9))
        maxd = float(np.max(np.abs(a1 - b1)))
        rel = float(np.linalg.norm(a1 - b1) / (np.linalg.norm(b1) + 1e-9))
        print(f"{label:15s} {a.size:9d} {b.size:9d} "
              f"{cos:9.4f} {maxd:10.3g} {rel:9.4f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["trigger", "hf", "diff", "full"],
                   default="full")
    p.add_argument("--base-url", default="http://localhost:8888/v1")
    p.add_argument("--model-id", default="Qwen/Qwen3.6-35B-A3B-FP8")
    p.add_argument("--hf-id", default="Qwen/Qwen3-VL-2B-Instruct",
                   help="BF16 checkpoint to use as reference (must share the "
                        "ViT architecture with --model-id).")
    p.add_argument("--atlas-dump", default="/tmp/atlas_vit")
    p.add_argument("--hf-dump", default="/tmp/hf_vit")
    args = p.parse_args()

    if args.mode in ("trigger", "full"):
        trigger_atlas(args.base_url, args.model_id)
    if args.mode in ("hf", "full"):
        compute_hf_reference(args.hf_id, Path(args.hf_dump))
    if args.mode in ("diff", "full"):
        diff_dumps(Path(args.atlas_dump), Path(args.hf_dump))
    return 0


if __name__ == "__main__":
    sys.exit(main())
