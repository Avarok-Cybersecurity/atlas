# EP=2 Token Dispatch/Combine Design (Workstream 3A)

Date: 2026-03-16
Status: Design phase

## Current Architecture (Dense All-Reduce)

```
Rank 0: local_experts(tokens) → full_output → all_reduce(full_output)
Rank 1: local_experts(tokens) → full_output → all_reduce(full_output)
```

Each rank computes ALL tokens through its LOCAL experts, producing a full [M, hidden] output.
Then ncclAllReduce sums the outputs. This is wrong for MoE because:
- Each token should only go to its top-K experts
- The all-reduce broadcasts ALL expert outputs, not just the relevant ones
- Communication is O(M * hidden) regardless of routing sparsity

## Target Architecture (Token Dispatch/Combine)

```
All ranks: gate(tokens) → top-K routing table

Dispatch phase:
  Rank 0 tokens routed to Rank 1 experts → send to Rank 1
  Rank 1 tokens routed to Rank 0 experts → send to Rank 0

Compute phase:
  Each rank: compute only LOCAL experts on received tokens

Combine phase:
  Send results back to original ranks
  Each rank: weighted sum of expert outputs per token
```

Communication is O(tokens_dispatched * hidden), which is much smaller than
O(M * hidden) when experts are spread across ranks.

## Implementation Plan

### Phase 1: Routing table on both ranks
- After gate projection, build `(token_id, expert_id, weight)` tuples
- Partition into local vs remote based on expert ownership
- Both ranks must agree on the routing (same gate logits → same top-K)

### Phase 2: Dispatch buffers
- Pre-allocate send/recv buffers sized for worst-case routing
- Pack tokens destined for remote experts into contiguous send buffer
- Use ncclSend/ncclRecv (already in the codebase) for transfer

### Phase 3: Local compute
- Each rank processes only received tokens through local experts
- Output buffer holds results for both local and received tokens

### Phase 4: Combine
- Send expert outputs back to originating ranks
- Weighted sum with routing weights to produce final per-token output

### Key Files to Modify
- `crates/spark-model/src/layers/moe.rs` — core MoE layer
- `crates/spark-comm/src/nccl_backend.rs` — add dispatch/combine primitives
- `crates/spark-comm/src/lib.rs` — CommBackend trait extensions

### References
- DeepSeek-V3: arxiv.org/abs/2412.19437
- DeepEP: github.com/deepseek-ai/DeepEP
- MegaBlocks: arxiv.org/abs/2211.15841
- Megatron token dispatcher: github.com/NVIDIA/Megatron-LM
