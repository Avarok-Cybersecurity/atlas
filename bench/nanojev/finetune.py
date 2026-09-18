# SPDX-License-Identifier: AGPL-3.0-only

"""Train the backbone too, not just a probe on top of it.

WHY THIS EXISTS
---------------
`train.py` freezes the backbone and fits a bilinear head to its hidden state.
That measures one thing: what the pretrained representation ALREADY encodes
linearly about the question. It is the right first experiment because it is
cheap and it cannot be confounded -- every arm sees byte-identical features.

But its ceiling is not the task's ceiling. On this testbed the frozen head
reaches Brier 0.643 against an oracle's 0.418, and no amount of head capacity
closes that if the information is not in the hidden state to begin with. The
only way to find out whether the gap is representational or architectural is
to let the backbone move.

That is also what NanoJev does -- a full-model run at backbone lr 2e-5 and
head lr 2e-4 -- so this is the arm that makes our numbers comparable to
theirs rather than merely inspired by them.

WHAT CHANGES, MECHANICALLY
--------------------------
The feature cache is gone: hidden states are recomputed every step because
they are no longer constant. Candidate embeddings are read from the LIVE
input embedding table for the same reason -- freezing them while the rest of
the model moves would slowly decorrelate the two halves of the bilinear score,
and the symptom would be a loss that plateaus for no visible reason.

Two parameter groups, because one learning rate cannot serve both: the head
starts from noise and wants to move fast, the backbone starts from a good
place and a head-sized step would destroy it.

WHAT TRAINING MAY SEE
---------------------
`outcome` only, exactly as in the frozen arm. `truth` reaches the evaluator
and nothing else.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from evaluate import evaluate_by_kind, format_report, reference_points
from heads import DecisionHead, masked_log_softmax
from objectives import OBJECTIVES, compute_loss
from reader import build_prompt, resolve_labels


class Batcher:
    """Tokenised batches with the answer at the last column.

    Left padding is not a style choice: it puts the answer position at index
    -1 for every row, so the hidden state is one index rather than a gather
    over per-row lengths. Right padding would read pad positions and train on
    them, which converges to something confident and wrong.
    """

    def __init__(self, tok, device):
        self.tok = tok
        self.device = device
        self._label_cache: dict[tuple, list[int]] = {}
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

    def label_ids(self, labels: list[str]) -> list[int]:
        key = tuple(labels)
        if key not in self._label_cache:
            self._label_cache[key] = resolve_labels(self.tok, list(labels)).ids
        return self._label_cache[key]

    def make(self, rows: list[dict]):
        enc = self.tok([build_prompt(r) for r in rows], return_tensors="pt", padding=True).to(
            self.device
        )
        kmax = max(len(r["labels"]) for r in rows)
        b = len(rows)
        ids = torch.zeros(b, kmax, dtype=torch.long, device=self.device)
        mask = torch.zeros(b, kmax, dtype=torch.bool, device=self.device)
        ordered = torch.zeros(b, dtype=torch.bool, device=self.device)
        outcome = torch.zeros(b, dtype=torch.long, device=self.device)
        for i, r in enumerate(rows):
            li = self.label_ids(r["labels"])
            ids[i, : len(li)] = torch.tensor(li, device=self.device)
            mask[i, : len(li)] = True
            ordered[i] = r["kind"] == "score"
            outcome[i] = r["outcome"]
        return enc, ids, mask, ordered, outcome


def forward_logits(model, head, enc, label_ids, mask, ordered):
    """One forward pass -> candidate logits."""
    out = model(**enc, output_hidden_states=True)
    h = out.hidden_states[-1][:, -1, :].float()
    # Live embedding table: it is being trained too.
    emb = model.get_input_embeddings()(label_ids).float()
    logits = head(h, emb, ordered)
    return masked_log_softmax(logits, mask)


@torch.no_grad()
def predict_all(model, head, batcher, rows, batch):
    model.eval()
    head.eval()
    preds = []
    for s in range(0, len(rows), batch):
        chunk = rows[s : s + batch]
        enc, ids, mask, ordered, _ = batcher.make(chunk)
        logp = forward_logits(model, head, enc, ids, mask, ordered)
        p = logp.exp().float().cpu().numpy()
        m = mask.cpu().numpy()
        for i in range(len(chunk)):
            preds.append([float(x) for x in p[i][m[i]]])
    return preds


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of --train-data held out for epoch selection")
    ap.add_argument("--model", default="/home/ms/models/qwen3-0.6b")
    ap.add_argument("--objective", default="brier", choices=sorted(OBJECTIVES))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=2e-4)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full epochs")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="trade ~30%% speed for activation memory; unnecessary at 0.6B")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    gen = torch.Generator(device=a.device).manual_seed(a.seed)

    all_train = [json.loads(x) for x in open(a.train_data)]
    test_rows = [json.loads(x) for x in open(a.test_data)]

    # Epoch selection needs a set that is neither trained on nor reported.
    # Selecting on the test set is the quiet way to turn a held-out number into
    # a trained-on one, and with a curve this sharp -- the best epoch is 4 of 8
    # and epoch 8 is twice as bad -- it would matter a lot.
    rng = torch.Generator().manual_seed(a.seed)
    order = torch.randperm(len(all_train), generator=rng).tolist()
    n_val = int(len(all_train) * a.val_frac)
    val_rows = [all_train[i] for i in order[:n_val]]
    train_rows = [all_train[i] for i in order[n_val:]]

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=a.device)
    if a.grad_checkpoint:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    head = DecisionHead(model.config.hidden_size).to(a.device).float()
    batcher = Batcher(tok, a.device)

    opt = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": a.lr_backbone},
            {"params": head.parameters(), "lr": a.lr_head},
        ],
        weight_decay=0.0,
    )

    print(f"train n={len(train_rows)}  val n={len(val_rows)}  test n={len(test_rows)}  "
          f"objective={a.objective}")
    print(f"lr backbone={a.lr_backbone} head={a.lr_head}  batch={a.batch}  epochs={a.epochs}\n")

    refs = reference_points(test_rows)
    print(format_report(evaluate_by_kind(predict_all(model, head, batcher, test_rows, a.eval_batch), test_rows),
                        "before training (random head on the pretrained backbone)"))
    print()

    step = 0
    t0 = time.time()
    history: list[dict] = []
    best = {"val_brier": float("inf"), "epoch": -1, "report": None}
    for ep in range(a.epochs):
        model.train()
        head.train()
        perm = torch.randperm(len(train_rows)).tolist()
        run, nb = 0.0, 0
        for s in range(0, len(train_rows), a.batch):
            rows = [train_rows[i] for i in perm[s : s + a.batch]]
            enc, ids, mask, ordered, outcome = batcher.make(rows)
            logp = forward_logits(model, head, enc, ids, mask, ordered)
            loss = compute_loss(a.objective, logp, outcome, mask, m=a.m, generator=gen)
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
                print(f"  ep{ep} step {step} loss {run/nb:.4f} ({step/(time.time()-t0):.2f} it/s)",
                      flush=True)
                run, nb = 0.0, 0
            if a.max_steps and step >= a.max_steps:
                break
        val_rep = evaluate_by_kind(
            predict_all(model, head, batcher, val_rows, a.eval_batch), val_rows
        )
        rep = evaluate_by_kind(predict_all(model, head, batcher, test_rows, a.eval_batch), test_rows)
        # Selection uses validation BRIER -- an outcome metric, computable
        # without the simulator. sq-L2 would select better but needs the truth,
        # which a real deployment does not have; the point of checking is that
        # the outcome metric picks the same epoch.
        v_brier = val_rep["overall"]["brier"]
        history.append(
            {
                "epoch": ep + 1,
                "steps": step,
                "val_brier": v_brier,
                "val_ece": val_rep["overall"]["ece"],
                "test_sq_l2": rep["overall"]["sq_l2"],
                "test_brier": rep["overall"]["brier"],
                "test_ece": rep["overall"]["ece"],
            }
        )
        print()
        print(format_report(rep, f"after epoch {ep+1} ({a.objective}, {step} steps) "
                                 f"[val Brier {v_brier:.4f}]"))
        print()
        if v_brier < best["val_brier"]:
            best.update(val_brier=v_brier, epoch=ep + 1, report=rep)
        if a.max_steps and step >= a.max_steps:
            break

    print("=== epoch curve ===")
    print(f"{'epoch':>5} {'steps':>6} {'val Brier':>10} {'val ECE':>8} | "
          f"{'test sq-L2':>11} {'test Brier':>11} {'test ECE':>9}")
    for h in history:
        star = "  <- selected" if h["epoch"] == best["epoch"] else ""
        print(f"{h['epoch']:>5} {h['steps']:>6} {h['val_brier']:>10.4f} {h['val_ece']:>8.4f} | "
              f"{h['test_sq_l2']:>11.5f} {h['test_brier']:>11.4f} {h['test_ece']:>9.4f}{star}")
    best_by_truth = min(history, key=lambda h: h["test_sq_l2"])
    print(f"\nselected by val Brier : epoch {best['epoch']}")
    print(f"best by test sq-L2    : epoch {best_by_truth['epoch']}")
    if best["epoch"] == best_by_truth["epoch"]:
        print("  -> they AGREE: an outcome metric suffices for selection; the truth is not needed.")
    else:
        print("  -> they DISAGREE: outcome-based selection leaves distribution error on the table.")
    print()

    print(format_report(best["report"], f"selected model (epoch {best['epoch']}, by val Brier)"))
    print()
    print(format_report(refs["oracle"], "oracle (the achievable floor)"))
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(
                {
                    "selected": best["report"],
                    "selected_epoch": best["epoch"],
                    "history": history,
                    "oracle": refs["oracle"],
                    "objective": a.objective,
                },
                f,
                indent=2,
            )
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
