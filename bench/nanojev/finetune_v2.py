# SPDX-License-Identifier: AGPL-3.0-only

"""Fine-tune the backbone under the invariant per-candidate architecture.

WHY BOTH, AND WHY NEITHER ALONE
-------------------------------
Two separate results from this testbed, each moving a different number:

  * Fine-tuning the backbone is the only thing that moved ACCURACY. A frozen
    probe stalls around sq-L2 0.28 regardless of head design, because the
    information is not in the pretrained hidden state; letting the backbone
    move reaches ~0.07.
  * The v2 architecture is the only thing that fixed the FAILURE MODES. Option
    order sensitivity went from 78% of answers flipping to exactly zero, and
    calibration improved 2-6x -- while accuracy did not move at all.

So they are complementary rather than competing, and this runs them together.
The question it answers is whether per-candidate encoding survives the backbone
moving: the invariance is a property of the SHAPE of the computation, so it
should hold no matter what the weights become, and that is worth verifying
rather than assuming.

MECHANICS
---------
Every candidate is its own sequence, so a step of B examples is really
B * K sequences through the backbone. They are flattened into one forward and
scattered back to [B, Kmax, D], which keeps the GPU busy and makes the
sequence count -- not the example count -- the thing to size batches by.

Candidate sequences share their state and question tokens. Nothing here
exploits that (each is encoded independently, redundantly), which is the honest
cost of the design under a naive implementation; a serving path would prefill
the shared prefix once and branch per candidate.

Selection is on validation Brier -- an outcome metric needing no simulator --
and the temperature is fitted on a third split. Neither touches test.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from cache_v2 import prompts_for
from evaluate import evaluate_by_kind, format_report, reference_points
from heads import DecisionHeadV2, masked_log_softmax
from train_v2 import composite_loss, shuffled


class Batcher:
    def __init__(self, tok, device):
        self.tok = tok
        self.device = device
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

    def make(self, rows):
        """Flatten every candidate of every row into one tokenised batch."""
        prompts, owner = [], []
        for i, r in enumerate(rows):
            for p in prompts_for(r):
                prompts.append(p)
                owner.append(i)
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
        kmax = max(len(r["labels"]) for r in rows)
        b = len(rows)
        mask = torch.zeros(b, kmax, dtype=torch.bool, device=self.device)
        outcome = torch.zeros(b, dtype=torch.long, device=self.device)
        for i, r in enumerate(rows):
            mask[i, : len(r["labels"])] = True
            outcome[i] = r["outcome"]
        return enc, torch.tensor(owner, device=self.device), mask, outcome, kmax


def forward_logits(model, head, enc, owner, mask, kmax):
    out = model(**enc, output_hidden_states=True)
    # Left padding puts each sequence's final real token at -1.
    h = out.hidden_states[-1][:, -1, :].float()          # [B*K, D]
    b, d = mask.shape[0], h.shape[-1]
    H = h.new_zeros((b, kmax, d))
    # Position within each owner's block, in order -- prompts_for emits
    # candidates in label order, so this reconstructs [B, K, D] faithfully.
    slot = torch.zeros_like(owner)
    seen: dict[int, int] = {}
    for idx, o in enumerate(owner.tolist()):
        slot[idx] = seen.get(o, 0)
        seen[o] = seen[o] + 1 if o in seen else 1
    H[owner, slot] = h
    return head(H, mask), H


@torch.no_grad()
def predict_all(model, head, batcher, rows, batch, temperature=1.0):
    model.eval()
    head.eval()
    preds = []
    for s in range(0, len(rows), batch):
        chunk = rows[s : s + batch]
        enc, owner, mask, _, kmax = batcher.make(chunk)
        logits, _ = forward_logits(model, head, enc, owner, mask, kmax)
        p = masked_log_softmax(logits / temperature, mask).exp().float().cpu().numpy()
        m = mask.cpu().numpy()
        for i in range(len(chunk)):
            preds.append([float(x) for x in p[i][m[i]]])
    return preds


@torch.no_grad()
def hidden_for(model, head, batcher, rows, batch):
    """Per-candidate states, so invariance can be checked exactly."""
    model.eval()
    Hs, Ms = [], []
    for s in range(0, len(rows), batch):
        chunk = rows[s : s + batch]
        enc, owner, mask, _, kmax = batcher.make(chunk)
        _, H = forward_logits(model, head, enc, owner, mask, kmax)
        pad = 5 - kmax
        if pad > 0:
            H = torch.nn.functional.pad(H, (0, 0, 0, pad))
            mask = torch.nn.functional.pad(mask, (0, pad))
        Hs.append(H)
        Ms.append(mask)
    return torch.cat(Hs), torch.cat(Ms)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--model", default="/home/ms/models/qwen3-0.6b")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16, help="EXAMPLES; sequences is this times K")
    ap.add_argument("--eval-batch", type=int, default=24)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=2e-4)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--cal-frac", type=float, default=0.12)
    ap.add_argument("--brier-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    gen = torch.Generator(device=a.device).manual_seed(a.seed)

    all_train = [json.loads(x) for x in open(a.train_data)]
    test_rows = [json.loads(x) for x in open(a.test_data)]
    rng = torch.Generator().manual_seed(a.seed)
    order = torch.randperm(len(all_train), generator=rng).tolist()
    n_val = int(len(all_train) * a.val_frac)
    n_cal = int(len(all_train) * a.cal_frac)
    val_rows = [all_train[i] for i in order[:n_val]]
    cal_rows = [all_train[i] for i in order[n_val : n_val + n_cal]]
    train_rows = [all_train[i] for i in order[n_val + n_cal :]]

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=a.device)
    model.config.use_cache = False
    head = DecisionHeadV2(model.config.hidden_size).to(a.device).float()
    batcher = Batcher(tok, a.device)
    opt = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": a.lr_backbone},
            {"params": head.parameters(), "lr": a.lr_head},
        ],
        weight_decay=0.0,
    )

    print(f"train {len(train_rows)}  val {len(val_rows)}  cal {len(cal_rows)}  test {len(test_rows)}")
    print(f"batch {a.batch} examples (~{a.batch * 3.25:.0f} sequences)  epochs {a.epochs}\n")

    history, best = [], {"val_brier": float("inf"), "epoch": -1, "report": None, "T": 1.0}
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        model.train()
        head.train()
        perm = torch.randperm(len(train_rows)).tolist()
        run, nb = 0.0, 0
        for s in range(0, len(train_rows), a.batch):
            rows = [train_rows[i] for i in perm[s : s + a.batch]]
            enc, owner, mask, outcome, kmax = batcher.make(rows)
            logits, _ = forward_logits(model, head, enc, owner, mask, kmax)
            logp = masked_log_softmax(logits, mask)
            loss = composite_loss(logp, outcome, mask, a.brier_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), 1.0
            )
            opt.step()
            run += loss.detach().item()
            nb += 1
            step += 1
            if step % 50 == 0:
                print(f"  ep{ep} step {step} loss {run/nb:.4f} "
                      f"({step/(time.time()-t0):.2f} it/s)", flush=True)
                run, nb = 0.0, 0

        # Temperature on the calibration split, then score val and test with it.
        cal_H, cal_M = hidden_for(model, head, batcher, cal_rows, a.eval_batch)
        cal_oc = torch.tensor([r["outcome"] for r in cal_rows], device=a.device)
        head.eval()
        with torch.no_grad():
            cal_logits = head(cal_H, cal_M)
        bestT, bestn = 1.0, float("inf")
        for t in np.linspace(0.25, 8.0, 60):
            lp = masked_log_softmax(cal_logits / float(t), cal_M)
            nll = -lp.gather(1, cal_oc.unsqueeze(1)).mean().item()
            if nll < bestn:
                bestT, bestn = float(t), nll

        val_rep = evaluate_by_kind(
            predict_all(model, head, batcher, val_rows, a.eval_batch, bestT), val_rows
        )
        rep = evaluate_by_kind(
            predict_all(model, head, batcher, test_rows, a.eval_batch, bestT), test_rows
        )
        vb = val_rep["overall"]["brier"]
        history.append({"epoch": ep + 1, "T": bestT, "val_brier": vb,
                        "test_sq_l2": rep["overall"]["sq_l2"],
                        "test_brier": rep["overall"]["brier"],
                        "test_ece": rep["overall"]["ece"]})
        print(f"\nepoch {ep+1}: T={bestT:.3f}  val Brier {vb:.4f} | test sq-L2 "
              f"{rep['overall']['sq_l2']:.5f}  Brier {rep['overall']['brier']:.4f}  "
              f"ECE {rep['overall']['ece']:.4f}\n", flush=True)
        if vb < best["val_brier"]:
            best.update(val_brier=vb, epoch=ep + 1, report=rep, T=bestT)

    # Invariance under the MOVED backbone: the guarantee is structural, so it
    # should survive training. Permuting the per-candidate states is exactly
    # permuting the options under this encoding.
    H, M = hidden_for(model, head, batcher, test_rows, a.eval_batch)
    oc = torch.tensor([r["outcome"] for r in test_rows], device=a.device)
    head.eval()
    with torch.no_grad():
        base = masked_log_softmax(head(H, M), M).exp().cpu().numpy()
        moved, rowsn, mx = 0, 0, 0.0
        for _ in range(4):
            Hs, Ms, _ = shuffled(H, M, oc, gen)
            p = masked_log_softmax(head(Hs, Ms), Ms).exp().cpu().numpy()
            mm, bm = Ms.cpu().numpy(), M.cpu().numpy()
            for i in range(p.shape[0]):
                got = sorted(float(x) for x in p[i][mm[i]])
                want = sorted(float(x) for x in base[i][bm[i]])
                tv = 0.5 * sum(abs(x - y) for x, y in zip(got, want))
                mx = max(mx, tv)
                rowsn += 1
                if tv > 1e-5:
                    moved += 1

    print("=== epoch curve ===")
    print(f"{'epoch':>5} {'T':>7} {'val Brier':>10} | {'test sq-L2':>11} {'test Brier':>11} {'ECE':>8}")
    for h in history:
        star = "  <- selected" if h["epoch"] == best["epoch"] else ""
        print(f"{h['epoch']:>5} {h['T']:>7.3f} {h['val_brier']:>10.4f} | {h['test_sq_l2']:>11.5f} "
              f"{h['test_brier']:>11.4f} {h['test_ece']:>8.4f}{star}")
    bt = min(history, key=lambda h: h["test_sq_l2"])
    print(f"\nselected by val Brier: epoch {best['epoch']}   best by test sq-L2: epoch {bt['epoch']}"
          f"  -> {'AGREE' if best['epoch'] == bt['epoch'] else 'DISAGREE'}")
    print(f"option-order invariance after fine-tuning: {moved}/{rowsn} rows moved "
          f"(max TV {mx:.2e})")
    print()
    print(format_report(best["report"], f"v2 fine-tuned, epoch {best['epoch']}, T={best['T']:.3f}"))
    print()
    print(format_report(reference_points(test_rows)["oracle"], "oracle"))

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"selected": best["report"], "epoch": best["epoch"], "T": best["T"],
                       "history": history, "invariance": {"moved": moved, "rows": rowsn}}, f, indent=2)
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
