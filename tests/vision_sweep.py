#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

"""Vision sweep — Mona Lisa probe across every vision-capable model.

Launches one model at a time, sends a downscaled Mona Lisa JPEG with the
prompt "What is this a photo of?", and grades the answer on a keyword
rubric (PASS > PARTIAL > FAIL). Results are written to stdout and to
/tmp/vision_sweep_results.md.

This harness intentionally does NOT run the full run_all_models.py
chat/tool/fib suite — it's focused on image recognition quality only.
Regression tests for text-mode still live in run_all_models.py.

Usage:
    python3 tests/vision_sweep.py
    python3 tests/vision_sweep.py --model Qwen/Qwen3.6-35B-A3B-FP8  # single model
    python3 tests/vision_sweep.py --base-url http://localhost:8888/v1  # skip launch
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image

# ── Test corpus ─────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mona_lisa.jpeg"

# Grading keywords.  A response PASSes if it names the artwork / subject;
# PARTIAL means it describes the composition (person, long hair, etc.)
# matching Qwen3.6's current behavior; FAIL is pure hallucination.
PASS_KEYWORDS = [
    "mona lisa", "da vinci", "leonardo", "la gioconda",
    "renaissance", "portrait", "painting", "portraiture",
]
PARTIAL_KEYWORDS = [
    "woman", "lady", "person", "face", "hair", "smile",
    "dress", "clothing", "outfit", "gown", "background",
    "landscape", "frame", "oil", "canvas",
]


@dataclass
class ModelSpec:
    label: str
    hf_id: str
    quant: str = "nvfp4"  # "fp8" | "nvfp4"
    kv_dtype: str = "nvfp4"  # most vision checkpoints pair with nvfp4 kv
    ep_world_size: int = 1
    notes: str = ""


# Vision-capable models discovered by probing `config.json` + safetensors
# index for `visual.*` tensors.  Gemma-4-{26B,31B} are excluded — their
# NVFP4 checkpoints ship `vision_config` but zero visual.* weights.
VISION_MODELS: list[ModelSpec] = [
    ModelSpec(
        label="qwen3-vl-30b-nvfp4",
        hf_id="ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4",
        quant="nvfp4",
        kv_dtype="nvfp4",
        notes="native Qwen3-VL, pre-existing vision path",
    ),
    ModelSpec(
        label="qwen3.6-35b-fp8",
        hf_id="Qwen/Qwen3.6-35B-A3B-FP8",
        quant="fp8",
        kv_dtype="fp8",
        notes="Qwen3.6 with MRoPE-interleaved, wired in commit 2cee090",
    ),
    ModelSpec(
        label="sehyo-qwen3.5-35b-nvfp4",
        hf_id="Sehyo/Qwen3.5-35B-A3B-NVFP4",
        quant="nvfp4",
        kv_dtype="nvfp4",
        notes="same arch as Qwen3.6; auto-routes to qwen3_6_moe via config rewrite",
    ),
    ModelSpec(
        label="sehyo-qwen3.5-122b-nvfp4",
        hf_id="Sehyo/Qwen3.5-122B-A10B-NVFP4",
        quant="nvfp4",
        kv_dtype="nvfp4",
        notes="hidden=3072 qwen3_6_moe variant; may need EP=2 for large KV",
    ),
]


# ── Image encoding ─────────────────────────────────────────────────────

def encode_image(max_dim: int = 320, quality: int = 85) -> str:
    """Downscale the fixture to keep base64 payload well under the 32 MB
    request limit and avoid the p>~600 ViT kernel crash (task #48).
    Returns a data-URI string.
    """
    img = Image.open(FIXTURE).convert("RGB")
    w, h = img.size
    scale = max_dim / min(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ── Container launch ────────────────────────────────────────────────────

def launch_container(spec: ModelSpec, image_tag: str) -> str:
    """Start a fresh atlas-gb10 container for the given model.  Returns
    the container name.  Blocks until the server logs 'Listening on'.
    """
    name = f"atlas-vsweep-{spec.label}"
    subprocess.run(["sudo", "docker", "rm", "-f", name],
                   check=False, capture_output=True)
    cmd = [
        "sudo", "docker", "run", "-d", "--name", name,
        "--gpus", "all", "--ipc=host", "--network", "host",
        "-e", "RUST_LOG=info",
        "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
        image_tag,
        "serve", spec.hf_id,
        "--port", "8888",
        "--scheduling-policy", "slai",
        "--max-seq-len", "32768",
        "--kv-cache-dtype", spec.kv_dtype,
        "--gpu-memory-utilization", "0.85",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"  launched {name}, waiting for Listening…", flush=True)
    deadline = time.time() + 360  # 6 min
    while time.time() < deadline:
        logs = subprocess.run(
            ["sudo", "docker", "logs", name],
            check=False, capture_output=True, text=True,
        ).stdout
        if "Listening on" in logs:
            print("  listening", flush=True)
            return name
        if "Error: " in logs or "FATAL" in logs:
            print(f"  startup failed:\n{logs[-2000:]}", flush=True)
            raise RuntimeError(f"{name} failed to start")
        time.sleep(5)
    raise RuntimeError(f"{name} timed out waiting for Listening on")


def stop_container(name: str) -> None:
    subprocess.run(["sudo", "docker", "stop", name],
                   check=False, capture_output=True)


# ── Probing a single model ─────────────────────────────────────────────

@dataclass
class ProbeResult:
    label: str
    hf_id: str
    status: str  # "PASS" | "PARTIAL" | "FAIL" | "ERROR"
    content: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    tps: float
    error: str = ""


def grade(content: str) -> str:
    low = content.lower()
    if any(kw in low for kw in PASS_KEYWORDS):
        return "PASS"
    if any(kw in low for kw in PARTIAL_KEYWORDS):
        return "PARTIAL"
    return "FAIL"


def probe(spec: ModelSpec, base_url: str, data_url: str) -> ProbeResult:
    body = {
        "model": spec.hf_id,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text",
                 "text": "What is this a photo of? Answer in one or two sentences."},
            ],
        }],
        "max_tokens": 300,
        "reasoning_effort": "none",
    }
    try:
        r = httpx.post(f"{base_url}/chat/completions",
                       timeout=300, json=body)
    except httpx.HTTPError as e:
        return ProbeResult(spec.label, spec.hf_id, "ERROR", "",
                           0, 0, 0.0, 0.0, error=str(e))
    if r.status_code != 200:
        return ProbeResult(spec.label, spec.hf_id, "ERROR", "",
                           0, 0, 0.0, 0.0,
                           error=f"HTTP {r.status_code}: {r.text[:200]}")
    j = r.json()
    if "error" in j:
        return ProbeResult(spec.label, spec.hf_id, "ERROR", "",
                           0, 0, 0.0, 0.0,
                           error=j["error"].get("message", "unknown"))
    msg = j["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    usage = j.get("usage", {})
    return ProbeResult(
        label=spec.label,
        hf_id=spec.hf_id,
        status=grade(content),
        content=content,
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        ttft_ms=float(usage.get("time_to_first_token_ms", 0.0)),
        tps=float(usage.get("response_token/s", 0.0)),
    )


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8888/v1",
                   help="Skip container launch; probe an already-running server")
    p.add_argument("--model", default=None,
                   help="Run a single model by HF id (launches container)")
    p.add_argument("--image-tag", default="atlas-gb10:qwen36-vision",
                   help="Docker image for auto-launch")
    p.add_argument("--output", default="/tmp/vision_sweep_results.md")
    args = p.parse_args()

    if not FIXTURE.exists():
        print(f"fixture not found: {FIXTURE}", file=sys.stderr)
        return 1

    data_url = encode_image()
    print(f"image fixture: {FIXTURE.name}", flush=True)

    results: list[ProbeResult] = []

    if args.base_url and args.model is None and "--base-url" in sys.argv:
        # Probe one already-running server using the first matching model.
        spec = next((m for m in VISION_MODELS), VISION_MODELS[0])
        print(f"probing already-running server at {args.base_url}", flush=True)
        results.append(probe(spec, args.base_url, data_url))
    else:
        specs = VISION_MODELS
        if args.model:
            specs = [m for m in VISION_MODELS if m.hf_id == args.model]
            if not specs:
                print(f"no model matches {args.model}", file=sys.stderr)
                return 1
        for spec in specs:
            print(f"\n=== {spec.label} ({spec.hf_id}) ===", flush=True)
            try:
                name = launch_container(spec, args.image_tag)
            except Exception as e:
                results.append(ProbeResult(spec.label, spec.hf_id, "ERROR",
                                           "", 0, 0, 0.0, 0.0, error=str(e)))
                continue
            try:
                res = probe(spec, args.base_url, data_url)
                results.append(res)
                print(f"  status={res.status}  content={res.content[:200]!r}",
                      flush=True)
            finally:
                stop_container(name)

    # Markdown summary
    lines: list[str] = []
    lines.append("# Vision Sweep — Mona Lisa")
    lines.append("")
    lines.append("| Model | Status | Prompt tok | TPS | TTFT (ms) | Content |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        snippet = r.content.replace("|", "\\|").replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "…"
        if r.error:
            snippet = f"ERROR: {r.error[:200]}"
        lines.append(
            f"| `{r.label}` | **{r.status}** | {r.prompt_tokens} | "
            f"{r.tps:.1f} | {r.ttft_ms:.0f} | {snippet} |"
        )
    out = "\n".join(lines) + "\n"
    print("\n" + out, flush=True)
    Path(args.output).write_text(out)
    print(f"results → {args.output}", flush=True)

    # Exit non-zero if any ERROR (infra failure).  PASS/PARTIAL/FAIL are
    # signals about model quality, not test-harness correctness.
    return 1 if any(r.status == "ERROR" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
