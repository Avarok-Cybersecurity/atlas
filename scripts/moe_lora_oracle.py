# SPDX-License-Identifier: AGPL-3.0-only

"""moe_lora_oracle.py — numpy delta-parity oracle for the Atlas MoE/attn/overlay
LoRA fold (PR #335 gate: docs/design/lora-moe-embed.md §E "Only after an oracle
passes do we build the fused/S-LoRA kernels").

THE ORACLE CONTRACT (what the Rust/CUDA fold must equal, per adapted surface):

    y = y_base + scale * ( B @ ( A @ x ) )            (LoRA surfaces, additive)
    y[row overridden]   = overlay_row                 (embed/lm_head overlay,
                                                       REPLACEMENT — PEFT
                                                       index_copy, NOT base+delta;
                                                       lora/overlay.rs:16-23)

SCALE CONVENTION (read from adapter_config.json, NEVER defaulted):
    scale = lora_alpha / r                 (use_rslora == false)
    scale = lora_alpha / sqrt(r)           (use_rslora == true)
  Source: crates/atlas-core/src/config/parsers/lora.rs:85-91
  (`PeftAdapterConfig::scaling()`); `use_rslora` is HARD-REQUIRED by the parser
  (lora.rs:200-205) and is hard-required here too. The scale is uniform across
  every surface of one adapter (attn q/k/v/o, dense/expert gate/up/down,
  router `mlp.gate`) — `alpha_pattern` is load-rejected.

WHERE THE SCALE IS APPLIED (BF16 rounding boundaries the kernels commit to):
    xa    = bf16( x  @ A^T )      # boundary 1 (shrink stores BF16 xa)
    delta = bf16( xa @ B^T )      # boundary 2 (expand rounds to BF16)
    y     = bf16( f32(y_base) + scale * f32(delta) )   # scale in FP32, AFTER
                                                       # the BF16 delta rounding
  Sources: layers/ops/lora_delta.rs:259-268 (`scaled_add` fold; A/B are stored
  VERBATIM, scale is never pre-folded into B — lora_delta.rs:97-99),
  kernels/gb10/common/moe_lora_grouped_down.cu (expand_fold comment block),
  kernels/gb10/common/moe_lora_gather_bgmv.cu, residual_add.cu bf16_scaled_add.
  A is pool-padded to [max_rank, k_in] (zero rows), B to [n_out, max_rank]
  (zero pad cols; row stride = max_rank): lora/loading.rs:86-102. Padding is
  numerically inert, so this oracle contracts at the true rank r.

TENSOR LAYOUT (PEFT save_pretrained, verbatim): lora_A = [r, in_features],
lora_B = [out_features, r], so the per-token delta on a column vector x[in] is
scale * (B @ (A @ x)) and on a row-major batch X[m, in] it is
scale * (X @ A^T) @ B^T.

SURFACES COVERED (classification mirrors lora/key.rs + lora/overlay.rs):
  expert down_proj   ...layers.L.mlp.experts.E.down_proj.lora_{A,B}  in=moe_inter out=hidden
  expert gate/up     ...layers.L.mlp.experts.E.{gate,up}_proj.*      in=hidden    out=moe_inter
  router             ...layers.L.mlp.gate.*                          in=hidden    out=num_experts
                     (delta on the routing LOGITS, before top-k/softmax)
  attn q/k/v/o       ...layers.L.self_attn.{q,k,v,o}_proj.*
  dense ffn          ...layers.L.mlp.{gate,up,down}_proj.*
  embed overlay      ...token_adapter.base_layer.weight [R,h] +
                     ...token_adapter.trainable_tokens_delta [T,h]  (row replace)
                     or modules_to_save full embed_tokens/lm_head.weight
  lm_head overlay    logits[:, id] = hidden @ row_id (recomputed column replace)

ORDERING INVARIANTS the oracle bakes in (design doc §H risk 3):
  * expert deltas fold on the PER-EXPERT projection output BEFORE the routing
    weight multiplies (so router_weight * (base + delta), like PEFT);
  * router delta folds on the gate logits BEFORE top-k selection;
  * embed overlay applies before scale_embeddings; lm_head before softcap.

MODES
  offline (default): --adapter DIR [--capture FILE.npz] [--selftest]
    * --selftest: synthesizes x / y_base per adapted surface and checks the
      bf16-boundary oracle against a pure-fp32 reference (internal consistency +
      shape/scale audit of the adapter fixture). No GPU, no server.
    * --capture: an .npz with keys  x/<prefix>, y_base/<prefix> and (optionally)
      y_adapted/<prefix>, where <prefix> is the tensor name minus
      ".lora_A.weight" (e.g. base_model.model.model.layers.3.mlp.gate).
      For each surface the oracle prints y_expected stats and, when y_adapted
      is present, the parity verdict |y_adapted - y_expected| vs BF16 tol.
      Overlay keys: x/<embed-prefix> = i64 token ids, y_base/... = [n, h] rows;
      x/<lmhead-prefix> = [n, h] hidden, y_base/... = [n, vocab] logits.
  live: --base-url URL --base-model NAME --adapter-name NAME [--prompt STR]
    Drives the SAME deterministic prompt through /v1/completions twice
    (model=<base name> vs model=<adapter name>) with echo=true, logprobs=1,
    max_tokens=0, temperature=0 (lm-eval-style prompt scoring) and reports the
    per-token prompt-logprob delta.
    LIVE-MODE LIMITS (measured against this branch, not guesses):
      * The server does NOT expose per-surface hidden/logit capture, so live
        mode is an end-to-end logit-level delta report only; per-surface parity
        is the offline mode's job.
      * A request naming the BASE model resolves adapter_slot=-1, which the
        ATTENTION/overlay paths resolve to the ACTIVE adapter
        (model/trait_impl/prefill_a.rs:351, lora/slot_math.rs:43) while the MoE
        fold treats <0 as base/Skip (lora/moe_row_adapter.rs:42-43). With one
        resident adapter the base-name leg therefore already carries the attn +
        overlay deltas, and the live A/B delta isolates the MoE expert/router
        surfaces — which is exactly what this PR adds. (Cross-surface
        inconsistency reported upstream; do not "fix" it here.)

Exit code: 0 = all checked surfaces pass (or nothing to check), 1 = any FAIL.
"""

import argparse
import json
import math
import os
import struct
import sys
import urllib.request

import numpy as np

# BF16 tolerance for the parity verdict: one BF16 ULP is 2^-8 of the exponent
# bucket; accumulation-order differences GPU-vs-numpy can flip the last bit at
# both rounding boundaries, so allow a few ULPs relative + tiny absolute floor.
REL_TOL = 3.0 / 256.0  # ~3 bf16 ULPs
ABS_TOL = 1e-3


# ---------------------------------------------------------------- safetensors
def load_safetensors(path):
    """Minimal torch-free safetensors reader -> {name: np.float32 array}."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        blob = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        lo, hi = meta["data_offsets"]
        raw = blob[lo:hi]
        dt = meta["dtype"]
        if dt == "BF16":
            bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            arr = bits.view(np.float32)
        elif dt == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif dt == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).copy()
        else:
            raise SystemExit(f"unsupported safetensors dtype {dt} for {name}")
        out[name] = arr.reshape(meta["shape"]).astype(np.float32)
    return out


def bf16_round(a):
    """Round-to-nearest-even f32 -> bf16 -> f32 (the kernels' __float2bfloat16)."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + np.uint32(0x7FFF)
    return ((u + r) & np.uint32(0xFFFF0000)).view(np.float32)


# ---------------------------------------------------------------- adapter
def scaling(cfg):
    """The Rust loader's scale — parsers/lora.rs:85-91. use_rslora required."""
    if "use_rslora" not in cfg:
        raise SystemExit(
            "REJECT(use_rslora): adapter_config.json lacks use_rslora — the Atlas "
            "parser hard-requires it (parsers/lora.rs:200) and so does this oracle"
        )
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    return alpha / math.sqrt(r) if cfg["use_rslora"] else alpha / r


def classify(name):
    """Mirror lora/key.rs + lora/overlay.rs: tensor name -> (surface, prefix, ab).

    Returns None for names this oracle does not model. `prefix` identifies the
    adapted module (capture npz key); `ab` in {A, B, overlay-kind}.
    """
    for kind in ("base_layer.weight", "trainable_tokens_delta"):
        if name.endswith(f"token_adapter.{kind}"):
            mod = "lm_head" if ".lm_head." in f".{name}" else "embed_tokens"
            return ("overlay", mod, kind)
    for mod in ("embed_tokens", "lm_head"):
        if name.endswith(f"{mod}.weight") and "token_adapter" not in name:
            return ("overlay", mod, "full_save")
    for ab in ("A", "B"):
        suffix = f".lora_{ab}.weight"
        if not name.endswith(suffix):
            continue
        prefix = name[: -len(suffix)]
        if ".mlp.experts." in prefix:
            proj = prefix.rsplit(".", 1)[1]  # {gate,up,down}_proj
            return (f"expert.{proj}", prefix, ab)
        if prefix.endswith(".mlp.gate"):
            return ("router", prefix, ab)
        if ".self_attn." in prefix:
            return (f"attn.{prefix.rsplit('.', 1)[1]}", prefix, ab)
        if ".mlp." in prefix:
            return (f"dense.{prefix.rsplit('.', 1)[1]}", prefix, ab)
    return None


def collect_pairs(tensors):
    """-> ({prefix: {"surface", "A", "B"}}, [overlay tensor descriptors])."""
    pairs, overlays = {}, []
    for name, arr in tensors.items():
        c = classify(name)
        if c is None:
            print(f"  [skip] unmodelled tensor: {name}")
            continue
        surface, prefix, ab = c
        if surface == "overlay":
            overlays.append((prefix, ab, name, arr))
            continue
        d = pairs.setdefault(prefix, {"surface": surface})
        d[ab] = arr
    for prefix, d in pairs.items():
        if "A" not in d or "B" not in d:
            raise SystemExit(f"adapter is missing lora_A or lora_B for {prefix}")
        ra, rb = d["A"].shape[0], d["B"].shape[1]
        if ra != rb:
            raise SystemExit(f"{prefix}: rank mismatch A[{ra},...] vs B[...,{rb}]")
    return pairs, overlays


# ---------------------------------------------------------------- oracle math
def expected_delta(a, b, x, scale, emulate_bf16=True):
    """scale * (x @ A^T) @ B^T with the kernels' BF16 rounding boundaries.

    x: [m, in]  (a single token may pass [in]; it is promoted).
    Returns fp32 [m, out] — the SCALED delta (scale applied in fp32 AFTER the
    bf16 delta rounding, matching bf16_scaled_add / *_expand_fold).
    """
    x2 = np.atleast_2d(np.asarray(x, dtype=np.float32))
    xa = x2 @ a.T
    if emulate_bf16:
        xa = bf16_round(xa)
    delta = xa @ b.T
    if emulate_bf16:
        delta = bf16_round(delta)
    return scale * delta


def expected_output(a, b, x, y_base, scale):
    """y = bf16(f32(y_base) + scale*f32(bf16_delta)) — the full fold contract."""
    d = expected_delta(a, b, x, scale, emulate_bf16=True)
    return bf16_round(np.atleast_2d(np.asarray(y_base, dtype=np.float32)) + d)


def parity(label, got, want):
    got = np.asarray(got, dtype=np.float32).reshape(-1)
    want = np.asarray(want, dtype=np.float32).reshape(-1)
    err = np.abs(got - want)
    tol = ABS_TOL + REL_TOL * np.abs(want)
    bad = int((err > tol).sum())
    ok = bad == 0
    print(
        f"  {'PASS' if ok else 'FAIL'}  {label:48s} max|err|={err.max() if err.size else 0:.3e} "
        f"viol={bad}/{err.size}"
    )
    return ok


# ---------------------------------------------------------------- offline
def run_selftest(pairs, overlays, scale, seed=0):
    """Internal-consistency audit: bf16-boundary oracle vs pure-fp32 reference,
    per adapted surface, on synthetic x/y_base. Proves shapes + scale plumbing
    of the fixture; GPU parity is the Rust test / capture mode's job."""
    rng = np.random.default_rng(seed)
    ok = True
    print(f"\n== selftest (scale={scale:.6f}) ==")
    for prefix, d in sorted(pairs.items()):
        a, b = d["A"], d["B"]
        m, k_in, n_out = 3, a.shape[1], b.shape[0]
        x = bf16_round(rng.normal(0, 1, (m, k_in)))
        y_base = bf16_round(rng.normal(0, 1, (m, n_out)))
        y = expected_output(a, b, x, y_base, scale)
        ref = y_base + scale * (x @ a.T) @ b.T  # pure fp32
        ok &= parity(f"{d['surface']:14s} {prefix.split('base_model.model.')[-1]}", y, ref)
    for prefix, kind, name, arr in overlays:
        print(f"  INFO  overlay {prefix:12s} {kind:24s} shape={list(arr.shape)} (row REPLACEMENT)")
    return ok


def run_capture(pairs, overlays, scale, path):
    cap = np.load(path)
    checked, ok = 0, True
    print(f"\n== capture parity against {path} ==")
    for prefix, d in sorted(pairs.items()):
        kx, kb, ka = f"x/{prefix}", f"y_base/{prefix}", f"y_adapted/{prefix}"
        if kx not in cap or kb not in cap:
            continue
        y = expected_output(d["A"], d["B"], cap[kx], cap[kb], scale)
        dmax = float(np.abs(y - np.atleast_2d(cap[kb])).max())
        print(f"  {d['surface']:14s} {prefix}: max|y_expected - y_base| = {dmax:.4e}")
        if ka in cap:
            ok &= parity(f"{d['surface']} {prefix}", cap[ka], y)
            checked += 1
    # Overlays: replacement semantics. embed: x = token ids, y_base = [n, h]
    # gathered rows; lm_head: x = [n, h] hidden, y_base = [n, vocab] logits.
    ov = {}
    for prefix, kind, name, arr in overlays:
        ov.setdefault(prefix, {})[kind] = arr
    for prefix, parts in ov.items():
        kx, kb, ka = f"x/{prefix}", f"y_base/{prefix}", f"y_adapted/{prefix}"
        if kx not in cap or kb not in cap or "trainable_tokens_delta" not in parts:
            continue
        idx_key = f"ids/{prefix}"
        if idx_key not in cap:
            print(f"  [skip] overlay {prefix}: capture lacks {idx_key} (trainable ids)")
            continue
        rows, ids = parts["trainable_tokens_delta"], cap[idx_key].astype(np.int64)
        lut = {int(t): rows[k] for k, t in enumerate(ids)}
        if prefix == "embed_tokens":
            y = np.array(
                [lut.get(int(t), base) for t, base in zip(cap[kx].astype(np.int64), cap[kb])]
            )
        else:  # lm_head: replace the recomputed logit column per overridden id
            y = np.array(cap[kb], dtype=np.float32, copy=True)
            for t, row in lut.items():
                y[:, t] = bf16_round(cap[kx].astype(np.float32) @ row)
        if ka in cap:
            ok &= parity(f"overlay {prefix}", cap[ka], y)
            checked += 1
    print(f"  ({checked} surface(s) had y_adapted to verify)")
    return ok


# ---------------------------------------------------------------- live
def score_prompt(base_url, model, prompt):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        out = json.loads(resp.read())
    lp = out["choices"][0]["logprobs"]
    return lp["tokens"], lp["token_logprobs"]


def run_live(args):
    print(f"\n== live logit-delta probe against {args.base_url} ==")
    print("   (per-surface capture is NOT exposed server-side; this is an end-to-end")
    print("    prompt-logprob delta. See the module docstring for what each leg folds.)")
    toks_b, lp_b = score_prompt(args.base_url, args.base_model, args.prompt)
    toks_a, lp_a = score_prompt(args.base_url, args.adapter_name, args.prompt)
    if toks_b != toks_a:
        print("  FAIL tokenizations differ between legs — cannot align logprobs")
        return False
    deltas = [
        (i, t, (a or 0.0) - (b or 0.0))
        for i, (t, b, a) in enumerate(zip(toks_b, lp_b, lp_a))
        if b is not None and a is not None
    ]
    if not deltas:
        print("  no comparable prompt logprobs returned")
        return False
    mx = max(deltas, key=lambda d: abs(d[2]))
    first = next((d for d in deltas if abs(d[2]) > 1e-6), None)
    print(f"  tokens={len(deltas)} max|dlogprob|={abs(mx[2]):.4f} at #{mx[0]} {mx[1]!r}")
    print(f"  sum dlogprob (adapter - base) = {sum(d[2] for d in deltas):+.4f}")
    print(f"  first divergent token: {first if first else 'NONE (legs identical)'}")
    return True


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapter", help="adapter dir (adapter_config.json + adapter_model.safetensors)")
    ap.add_argument("--capture", help=".npz with x/, y_base/, [y_adapted/, ids/] keys")
    ap.add_argument("--selftest", action="store_true", help="synthetic internal-consistency audit")
    ap.add_argument("--base-url", help="live mode: Atlas server URL")
    ap.add_argument("--base-model", help="live mode: base model name")
    ap.add_argument("--adapter-name", help="live mode: resident adapter name")
    ap.add_argument("--prompt", default="The capital of France is Paris. The capital of Italy is")
    args = ap.parse_args()

    ok = True
    if args.base_url:
        if not (args.base_model and args.adapter_name):
            ap.error("--base-url needs --base-model and --adapter-name")
        ok &= run_live(args)

    if args.adapter:
        cfg = json.load(open(os.path.join(args.adapter, "adapter_config.json")))
        scale = scaling(cfg)
        tensors = load_safetensors(os.path.join(args.adapter, "adapter_model.safetensors"))
        pairs, overlays = collect_pairs(tensors)
        surfaces = sorted({d["surface"] for d in pairs.values()})
        print(f"adapter: r={cfg['r']} alpha={cfg['lora_alpha']} rslora={cfg['use_rslora']} "
              f"scale={scale:.6f}")
        print(f"adapted surfaces: {surfaces or '(none)'} + {len(overlays)} overlay tensor(s)")
        if args.selftest:
            ok &= run_selftest(pairs, overlays, scale)
        if args.capture:
            ok &= run_capture(pairs, overlays, scale, args.capture)
    elif not args.base_url:
        ap.error("nothing to do: pass --adapter (offline) and/or --base-url (live)")

    print("\nORACLE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
