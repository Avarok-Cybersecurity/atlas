#!/usr/bin/env python3
"""C7 dual-node dummy: TP=2 column-split o_proj + NCCL allreduce == TP=1 tokens.

Production width (hidden=7168), tiny vocab, dummy weights. Not K3 graph.
Not nccl_2rank_bench. MASTER_ADDR / RANK from env — no lab IPs in this file.

RST known-bad: set C7_DROP_RANK=1 to zero rank-1 shard (tokens must change).
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

HIDDEN = 7168
VOCAB = 256
NEW = 8
PROMPT = [1, 2, 3, 4]
SEED = 7


def pin_roce() -> None:
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "enp1s0f1np1")
    os.environ.setdefault("NCCL_IB_HCA", "rocep1s0f1")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ.setdefault("NCCL_PROTO", "Simple")
    os.environ.setdefault("NCCL_ALGO", "Ring")


def greedy_tp1(embed, o_proj, lm_head, prompt, n_new: int) -> list[int]:
    h = embed[prompt[-1]]
    out = list(prompt)
    for _ in range(n_new):
        h = h @ o_proj
        logits = h @ lm_head.T
        tok = int(logits.argmax())
        out.append(tok)
        h = embed[tok]
    return out


def greedy_tp2(embed, o_shard, lm_head, prompt, n_new: int) -> list[int]:
    h = embed[prompt[-1]]
    out = list(prompt)
    for _ in range(n_new):
        partial = h @ o_shard
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        logits = partial @ lm_head.T
        tok = int(logits.argmax())
        out.append(tok)
        h = embed[tok]
    return out


def main() -> None:
    pin_roce()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(0)
    dev = torch.device("cuda")
    assert world == 2, world

    torch.manual_seed(SEED)
    embed = torch.randn(VOCAB, HIDDEN, device=dev, dtype=torch.float32)
    o_proj = torch.randn(HIDDEN, HIDDEN, device=dev, dtype=torch.float32)
    lm_head = torch.randn(VOCAB, HIDDEN, device=dev, dtype=torch.float32)
    dist.broadcast(embed, 0)
    dist.broadcast(o_proj, 0)
    dist.broadcast(lm_head, 0)

    tp1 = greedy_tp1(embed, o_proj, lm_head, PROMPT, NEW)
    shard = o_proj[:, rank::world].contiguous()
    if os.environ.get("C7_DROP_RANK") == "1" and rank == 1:
        shard = torch.zeros_like(shard)
    # Column-parallel: x @ W[:, rank::2] lives in hidden/2 — reconstruct via
    # padding into full hidden then allreduce. Simpler: each rank holds
    # W_shard [H, H/2] and x @ W_shard is [H/2]; all_gather then is the
    # concatenated output, not a sum. C7 CPU path is SUM of column-split
    # GEMVs that already map to full hidden (each rank computes x @ W_r
    # with W_r zeroed on the other rank's columns).
    w_rank = torch.zeros_like(o_proj)
    w_rank[:, rank::world] = shard
    if os.environ.get("C7_DROP_RANK") == "1" and rank == 1:
        w_rank.zero_()
    tp2 = greedy_tp2(embed, w_rank, lm_head, PROMPT, NEW)

    if rank == 0:
        print("tp1", tp1, flush=True)
        print("tp2", tp2, flush=True)
        drop = os.environ.get("C7_DROP_RANK") == "1"
        if drop:
            print("DROP", tp1 != tp2, flush=True)
            if tp1 == tp2:
                raise SystemExit("known-bad: drop rank1 did not change tokens")
        else:
            print("MATCH", tp1 == tp2, flush=True)
            if tp1 != tp2:
                raise SystemExit("C7: TP=2 tokens != TP=1")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
