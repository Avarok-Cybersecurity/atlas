# SPDX-License-Identifier: AGPL-3.0-only
"""Compare per-step logit dumps from two metal_qwen35_inference runs.

Each input is the raw little-endian bf16 dump written by
ATLAS_LOGITS_OUT ([n_steps, vocab]). Reports per-step KL divergence
(reference || candidate), mean KLD, and top-1 agreement.

Greedy decode means the two runs share context only while their argmax
paths agree — KLD is only meaningful up to the first token divergence,
so the comparison stops there (reported as compared_steps).

Usage:
  python3 tests/metal_kv_kld_compare.py /tmp/bf16.logits /tmp/turbo8.logits \
      --vocab 248320
"""

import argparse

import numpy as np


def load_bf16(path: str, vocab: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    steps = raw.size // vocab
    raw = raw[: steps * vocab].reshape(steps, vocab)
    # bf16 → f32: shift into the high half of a u32.
    return (raw.astype(np.uint32) << 16).view(np.float32)


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--vocab", type=int, default=248_320)
    ap.add_argument(
        "--teacher-forced",
        action="store_true",
        help="both runs fed the same token list (ATLAS_FORCE_TOKENS_FILE): "
        "contexts match at every position, so do not trim at the first "
        "argmax divergence",
    )
    args = ap.parse_args()

    ref = load_bf16(args.reference, args.vocab)
    cand = load_bf16(args.candidate, args.vocab)
    steps = min(len(ref), len(cand))
    ref, cand = ref[:steps], cand[:steps]

    ref_top1 = ref.argmax(axis=-1)
    cand_top1 = cand.argmax(axis=-1)

    # Greedy context stays shared until the first argmax divergence;
    # teacher-forced runs share context everywhere.
    diverge = np.flatnonzero(ref_top1 != cand_top1)
    valid = (
        steps
        if args.teacher_forced
        else (int(diverge[0]) + 1 if diverge.size else steps)
    )

    ref_lp = log_softmax(ref[:valid])
    cand_lp = log_softmax(cand[:valid])
    kld = (np.exp(ref_lp) * (ref_lp - cand_lp)).sum(axis=-1)

    agree = (ref_top1[:valid] == cand_top1[:valid]).mean()
    print(f"compared_steps: {valid} / {steps}")
    print(f"mean_kld:       {kld.mean():.6f}")
    print(f"max_kld:        {kld.max():.6f}")
    print(f"top1_agree:     {agree:.3f}")
    if diverge.size and not args.teacher_forced:
        print(f"first_divergence_at_step: {int(diverge[0])}")
    per = " ".join(f"{v:.4f}" for v in kld[: min(valid, 32)])
    print(f"per_step_kld:   {per}")


if __name__ == "__main__":
    main()
