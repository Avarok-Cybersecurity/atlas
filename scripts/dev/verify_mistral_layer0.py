#!/usr/bin/env python3
"""
Verify Mistral Small 4 Layer 0 forward pass against Atlas.

Computes: embedding → attention_norm → MLA Q/KV → RoPE → attention → O proj
          → residual → ffn_norm → MoE gate (top-4 routing only).

Reports hidden state norms at each step for comparison with Atlas output.
"""

import torch
import json
import os
import math

SNAP = (
    "/workspace/.cache/huggingface/models--mistralai--Mistral-Small-4-119B-2603-NVFP4/"
    "snapshots/043f75a201a226d8e9cbbc3316af437ea25d3912"
)

# ── Model params ──
HIDDEN = 4096
N_HEADS = 32
N_KV = 32
HEAD_DIM = 128
Q_LORA = 1024
KV_LORA = 256
NOPE = 64
ROPE_DIM = 64
V_DIM = 128
THETA = 10000.0
EPS = 1e-6

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """RMS normalization: x * weight / sqrt(mean(x^2) + eps)."""
    dtype = x.dtype
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(-1, keepdim=True) + eps)
    return (x_f32 / rms * weight.float()).to(dtype)


def apply_rope_neox(x: torch.Tensor, positions: torch.Tensor,
                    rotary_dim: int, theta: float) -> torch.Tensor:
    """Apply RoPE with neox-style rotation (pair i with i + half_rot).

    x: [batch, n_heads, seq_len, head_dim] (BF16)
    positions: [seq_len] (long)
    rotary_dim: how many dims to rotate (rest pass through)
    """
    dtype = x.dtype
    x = x.float()
    batch, n_heads, seq_len, hd = x.shape
    half_rot = rotary_dim // 2

    # Build frequencies: freq_i = 1 / theta^(2i / rotary_dim) for i in [0, half_rot)
    freq_exp = torch.arange(half_rot, dtype=torch.float32, device=x.device) * 2.0 / rotary_dim
    freqs = 1.0 / (theta ** freq_exp)  # [half_rot]

    # Outer product: [seq_len, half_rot]
    angles = positions.float().unsqueeze(-1) * freqs.unsqueeze(0)  # [seq_len, half_rot]
    cos_vals = torch.cos(angles)  # [seq_len, half_rot]
    sin_vals = torch.sin(angles)

    # Reshape for broadcasting: [1, 1, seq_len, half_rot]
    cos_vals = cos_vals.unsqueeze(0).unsqueeze(0)
    sin_vals = sin_vals.unsqueeze(0).unsqueeze(0)

    # x0 = first half, x1 = second half (of rotary dims)
    x0 = x[..., :half_rot]              # [batch, n_heads, seq_len, half_rot]
    x1 = x[..., half_rot:rotary_dim]    # [batch, n_heads, seq_len, half_rot]

    y0 = x0 * cos_vals - x1 * sin_vals
    y1 = x1 * cos_vals + x0 * sin_vals

    # Reassemble: [rotated_first | rotated_second | passthrough]
    if rotary_dim < hd:
        out = torch.cat([y0, y1, x[..., rotary_dim:]], dim=-1)
    else:
        out = torch.cat([y0, y1], dim=-1)

    return out.to(dtype)


def load_bf16(shard, key: str) -> torch.Tensor:
    """Load a BF16 tensor from an open safetensors file handle."""
    return shard.get_tensor(key).to(torch.bfloat16)


def main():
    from safetensors import safe_open
    from tokenizers import Tokenizer

    device = "cpu"  # pure CPU for exact reproducibility

    # ── 1. Tokenize ──
    tok = Tokenizer.from_file(os.path.join(SNAP, "tokenizer.json"))
    prompt = '<s>[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS][INST]Hi[/INST]'
    enc = tok.encode(prompt, add_special_tokens=False)
    token_ids = enc.ids
    print(f"Prompt: {prompt}")
    print(f"Token IDs ({len(token_ids)}): {token_ids}")
    print(f"Tokens: {enc.tokens}")
    print()

    # ── 2. Load weights ──
    shard1_path = os.path.join(SNAP, "consolidated-00001-of-00013.safetensors")
    shard12_path = os.path.join(SNAP, "consolidated-00012-of-00013.safetensors")

    print("Loading weights from shard 1 (layer 0) and shard 12 (embeddings)...")
    shard1 = safe_open(shard1_path, framework="pt", device=device)
    shard12 = safe_open(shard12_path, framework="pt", device=device)

    # Embedding
    embed_weight = load_bf16(shard12, "tok_embeddings.weight")  # [131072, 4096]
    print(f"  tok_embeddings.weight: {list(embed_weight.shape)}")

    # Attention norm
    attn_norm_w = load_bf16(shard1, "layers.0.attention_norm.weight")  # [4096]
    print(f"  attention_norm.weight: {list(attn_norm_w.shape)}")

    # MLA Q path
    wq_a = load_bf16(shard1, "layers.0.attention.wq_a.weight")       # [1024, 4096]
    q_a_norm_w = load_bf16(shard1, "layers.0.attention.q_a_norm.weight")  # [1024]
    wq_b = load_bf16(shard1, "layers.0.attention.wq_b.weight")       # [4096, 1024]
    print(f"  wq_a: {list(wq_a.shape)}, q_a_norm: {list(q_a_norm_w.shape)}, wq_b: {list(wq_b.shape)}")

    # MLA KV path
    wkv_a_mqa = load_bf16(shard1, "layers.0.attention.wkv_a_with_mqa.weight")  # [320, 4096]
    kv_a_norm_w = load_bf16(shard1, "layers.0.attention.kv_a_norm.weight")      # [256]
    wkv_b = load_bf16(shard1, "layers.0.attention.wkv_b.weight")               # [6144, 256]
    print(f"  wkv_a_mqa: {list(wkv_a_mqa.shape)}, kv_a_norm: {list(kv_a_norm_w.shape)}, wkv_b: {list(wkv_b.shape)}")

    # Split wkv_a: first 256 rows → kv_lora, last 64 rows → k_rope
    wkv_a = wkv_a_mqa[:KV_LORA, :]        # [256, 4096]
    wkv_a_rope = wkv_a_mqa[KV_LORA:, :]   # [64, 4096]
    print(f"  wkv_a (latent): {list(wkv_a.shape)}, wkv_a_rope: {list(wkv_a_rope.shape)}")

    # O projection
    wo = load_bf16(shard1, "layers.0.attention.wo.weight")  # [4096, 4096]
    print(f"  wo: {list(wo.shape)}")

    # FFN norm
    ffn_norm_w = load_bf16(shard1, "layers.0.ffn_norm.weight")  # [4096]
    print(f"  ffn_norm.weight: {list(ffn_norm_w.shape)}")

    # MoE gate
    gate_w = load_bf16(shard1, "layers.0.gate.weight")  # [128, 4096]
    print(f"  gate.weight: {list(gate_w.shape)}")
    print()

    # ── 3. Embed tokens ──
    input_ids = torch.tensor(token_ids, dtype=torch.long)
    hidden = embed_weight[input_ids]  # [seq_len, 4096]
    seq_len = hidden.shape[0]
    print(f"=== Step: Embedding ===")
    print(f"  hidden shape: {list(hidden.shape)}")
    print(f"  hidden norm (per token): {hidden.float().norm(dim=-1).tolist()}")
    print(f"  hidden[0,:8]: {hidden[0,:8].float().tolist()}")
    print()

    # ── 4a. Attention norm ──
    normed = rms_norm(hidden, attn_norm_w)
    print(f"=== Step: Attention RMS Norm ===")
    print(f"  normed norm (per token): {normed.float().norm(dim=-1).tolist()}")
    print(f"  normed[0,:8]: {normed[0,:8].float().tolist()}")
    print()

    # ── 4b. MLA Q projection: normed → wq_a → q_a_norm → wq_b ──
    # Step 1: Q latent = normed @ wq_a^T → [seq_len, q_lora=1024]
    q_latent = normed.float() @ wq_a.float().T  # [seq, 1024]
    q_latent = q_latent.to(torch.bfloat16)
    print(f"=== Step: Q latent (normed @ wq_a^T) ===")
    print(f"  q_latent norm (per token): {q_latent.float().norm(dim=-1).tolist()}")

    # Step 2: RMS norm on q_latent
    q_latent_normed = rms_norm(q_latent, q_a_norm_w)
    print(f"  q_latent_normed norm: {q_latent_normed.float().norm(dim=-1).tolist()}")

    # Step 3: Q = q_latent_normed @ wq_b^T → [seq_len, n_heads*head_dim=4096]
    q_full = q_latent_normed.float() @ wq_b.float().T  # [seq, 4096]
    q_full = q_full.to(torch.bfloat16)
    print(f"  Q full norm: {q_full.float().norm(dim=-1).tolist()}")
    print(f"  Q[0,:8]: {q_full[0,:8].float().tolist()}")
    print()

    # ── 4c. MLA KV projection: normed → wkv_a → kv_a_norm → wkv_b + k_rope ──
    # Step 1: KV latent = normed @ wkv_a^T → [seq, kv_lora=256]
    kv_latent = normed.float() @ wkv_a.float().T  # [seq, 256]
    kv_latent = kv_latent.to(torch.bfloat16)
    print(f"=== Step: KV latent (normed @ wkv_a^T) ===")
    print(f"  kv_latent norm: {kv_latent.float().norm(dim=-1).tolist()}")

    # Step 2: RMS norm on kv_latent
    kv_latent_normed = rms_norm(kv_latent, kv_a_norm_w)
    print(f"  kv_latent_normed norm: {kv_latent_normed.float().norm(dim=-1).tolist()}")

    # Step 3: KV_expanded = kv_latent_normed @ wkv_b^T → [seq, n_kv*(nope+v)=6144]
    kv_expanded = kv_latent_normed.float() @ wkv_b.float().T  # [seq, 6144]
    kv_expanded = kv_expanded.to(torch.bfloat16)
    print(f"  KV expanded norm: {kv_expanded.float().norm(dim=-1).tolist()}")

    # Split: first n_kv*nope=2048 → K_nope, remaining n_kv*v_dim=4096 → V
    k_nope_flat = kv_expanded[:, :N_KV * NOPE]     # [seq, 2048]
    v_flat = kv_expanded[:, N_KV * NOPE:]            # [seq, 4096]
    print(f"  K_nope flat norm: {k_nope_flat.float().norm(dim=-1).tolist()}")
    print(f"  V flat norm: {v_flat.float().norm(dim=-1).tolist()}")

    # Step 4: K_rope = normed @ wkv_a_rope^T → [seq, rope=64]
    k_rope = normed.float() @ wkv_a_rope.float().T  # [seq, 64]
    k_rope = k_rope.to(torch.bfloat16)
    print(f"  K_rope norm: {k_rope.float().norm(dim=-1).tolist()}")
    print()

    # ── Assemble K per head: [nope(64) | rope(64)] ──
    # K_nope_flat layout: flat [n_kv*nope] per token
    # Reshape to per-head: [seq, n_kv, nope]
    k_nope_heads = k_nope_flat.view(seq_len, N_KV, NOPE)  # [seq, 32, 64]
    # K_rope is shared across heads, broadcast
    k_rope_heads = k_rope.unsqueeze(1).expand(seq_len, N_KV, ROPE_DIM)  # [seq, 32, 64]
    # Assemble K = [nope | rope] per head
    k_assembled = torch.cat([k_nope_heads, k_rope_heads], dim=-1)  # [seq, 32, 128]
    print(f"=== Step: K assembly [nope|rope] ===")
    print(f"  K assembled shape: {list(k_assembled.shape)}")
    print(f"  K[0,0,:8] (nope part): {k_assembled[0,0,:8].float().tolist()}")
    print(f"  K[0,0,64:72] (rope part): {k_assembled[0,0,64:72].float().tolist()}")
    print()

    # ── Reshape Q and V to per-head ──
    # Q: [seq, n_heads, head_dim]
    q_heads = q_full.view(seq_len, N_HEADS, HEAD_DIM)  # [seq, 32, 128]
    # V: flat → per-head [seq, n_kv, v_dim]
    v_heads = v_flat.view(seq_len, N_KV, V_DIM)  # [seq, 32, 128]

    # ── 4d. RoPE ──
    # Atlas uses partial_rotary_factor=1.0 → rotary_dim=128 for GQA fallback
    # But for MLA-native, it should really be rotary_dim=64 (only rope portion)
    # Let's compute BOTH and compare

    positions = torch.arange(seq_len, dtype=torch.long)

    # Reshape for RoPE: [1, n_heads, seq_len, head_dim]
    q_rope_in = q_heads.permute(1, 0, 2).unsqueeze(0).contiguous()  # [1, 32, seq, 128]
    k_rope_in = k_assembled.permute(1, 0, 2).unsqueeze(0).contiguous()  # [1, 32, seq, 128]

    # --- Full RoPE (rotary_dim=128, what Atlas does) ---
    q_roped_full = apply_rope_neox(q_rope_in, positions, rotary_dim=128, theta=THETA)
    k_roped_full = apply_rope_neox(k_rope_in, positions, rotary_dim=128, theta=THETA)

    # --- Partial RoPE (rotary_dim=64, only rope dims) ---
    # Split Q into nope/rope portions, apply RoPE only to rope
    # This is the "correct" MLA approach where nope dims are left unchanged
    # But Atlas applies full RoPE due to GQA fallback design

    print(f"=== Step: RoPE (Atlas: rotary_dim=128, full head_dim) ===")
    print(f"  Q after RoPE norm (per token): {q_roped_full.squeeze(0).float().norm(dim=(0,2)).tolist()[:4]}...")
    print(f"  K after RoPE norm (per token): {k_roped_full.squeeze(0).float().norm(dim=(0,2)).tolist()[:4]}...")
    print(f"  Q_roped[0,0,0,:8]: {q_roped_full[0,0,0,:8].float().tolist()}")
    print(f"  K_roped[0,0,0,:8]: {k_roped_full[0,0,0,:8].float().tolist()}")
    print()

    # ── 4e. Standard attention: Q·K^T / sqrt(head_dim), softmax, × V ──
    # Shapes: Q [1, n_heads, seq, hd], K [1, n_kv, seq, hd], V [1, n_kv, seq, v_dim]
    v_rope_in = v_heads.permute(1, 0, 2).unsqueeze(0).contiguous()  # [1, 32, seq, 128]

    # For MHA (n_heads == n_kv), no GQA repeat needed
    scale = 1.0 / math.sqrt(HEAD_DIM)

    # Attention scores: [1, n_heads, seq, seq]
    attn_scores = torch.matmul(
        q_roped_full.float(), k_roped_full.float().transpose(-2, -1)
    ) * scale

    # Causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    attn_scores = attn_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    attn_weights = torch.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_weights, v_rope_in.float())  # [1, n_heads, seq, v_dim]

    # Reshape: [1, n_heads, seq, v_dim] → [seq, n_heads * v_dim]
    attn_output = attn_output.squeeze(0).permute(1, 0, 2).contiguous()  # [seq, n_heads, v_dim]
    attn_output = attn_output.view(seq_len, N_HEADS * V_DIM)  # [seq, 4096]
    attn_output = attn_output.to(torch.bfloat16)

    print(f"=== Step: Attention Output ===")
    print(f"  attn_output norm (per token): {attn_output.float().norm(dim=-1).tolist()}")
    print(f"  attn_output[0,:8]: {attn_output[0,:8].float().tolist()}")
    # Check attention weights for position 0 (first token, self-attention only)
    print(f"  attn_weights[0,0,0,:]: {attn_weights[0,0,0,:].tolist()}")  # should be [1, 0, 0, ...]
    print()

    # ── 4f. O projection: attn_output @ wo^T ──
    o_out = attn_output.float() @ wo.float().T  # [seq, 4096]
    o_out = o_out.to(torch.bfloat16)
    print(f"=== Step: O Projection ===")
    print(f"  o_out norm (per token): {o_out.float().norm(dim=-1).tolist()}")
    print(f"  o_out[0,:8]: {o_out[0,:8].float().tolist()}")
    print()

    # ── 4g. Residual add ──
    residual = hidden + o_out  # [seq, 4096]
    print(f"=== Step: Residual Add (hidden + o_out) ===")
    print(f"  residual norm (per token): {residual.float().norm(dim=-1).tolist()}")
    print(f"  residual[0,:8]: {residual[0,:8].float().tolist()}")
    print()

    # ── 4h. FFN norm ──
    ffn_normed = rms_norm(residual, ffn_norm_w)
    print(f"=== Step: FFN RMS Norm ===")
    print(f"  ffn_normed norm (per token): {ffn_normed.float().norm(dim=-1).tolist()}")
    print(f"  ffn_normed[0,:8]: {ffn_normed[0,:8].float().tolist()}")
    print()

    # ── 4i. MoE Gate → softmax → topk(4) ──
    gate_logits = ffn_normed.float() @ gate_w.float().T  # [seq, 128]
    gate_probs = torch.softmax(gate_logits, dim=-1)
    topk_vals, topk_ids = torch.topk(gate_probs, k=4, dim=-1)

    print(f"=== Step: MoE Gate Routing ===")
    print(f"  gate_logits norm (per token): {gate_logits.norm(dim=-1).tolist()}")
    for t in range(seq_len):
        print(f"  Token {t} ('{enc.tokens[t]}'): top-4 experts = {topk_ids[t].tolist()}, "
              f"weights = {[f'{w:.4f}' for w in topk_vals[t].tolist()]}")
    print()

    # ── Summary ──
    print("=" * 70)
    print("SUMMARY OF NORMS (for Atlas comparison)")
    print("=" * 70)
    print(f"  Embedding[last]:      {hidden[-1].float().norm().item():.6f}")
    print(f"  AttnNorm[last]:       {normed[-1].float().norm().item():.6f}")
    print(f"  Q_latent[last]:       {q_latent[-1].float().norm().item():.6f}")
    print(f"  Q_latent_normed[last]:{q_latent_normed[-1].float().norm().item():.6f}")
    print(f"  Q_full[last]:         {q_full[-1].float().norm().item():.6f}")
    print(f"  KV_latent[last]:      {kv_latent[-1].float().norm().item():.6f}")
    print(f"  KV_expanded[last]:    {kv_expanded[-1].float().norm().item():.6f}")
    print(f"  K_rope[last]:         {k_rope[-1].float().norm().item():.6f}")
    print(f"  AttnOut[last]:        {attn_output[-1].float().norm().item():.6f}")
    print(f"  O_proj[last]:         {o_out[-1].float().norm().item():.6f}")
    print(f"  Residual[last]:       {residual[-1].float().norm().item():.6f}")
    print(f"  FFN_normed[last]:     {ffn_normed[-1].float().norm().item():.6f}")
    print()

    # ── Extra: check what happens with partial RoPE (rotary_dim=64) ──
    print("=" * 70)
    print("ALTERNATE: Partial RoPE (rotary_dim=64, only rope dims rotated)")
    print("=" * 70)
    q_roped_partial = apply_rope_neox(q_rope_in, positions, rotary_dim=64, theta=THETA)
    k_roped_partial = apply_rope_neox(k_rope_in, positions, rotary_dim=64, theta=THETA)
    print(f"  Q_roped_partial[0,0,0,:8]: {q_roped_partial[0,0,0,:8].float().tolist()}")
    print(f"  K_roped_partial[0,0,0,:8]: {k_roped_partial[0,0,0,:8].float().tolist()}")

    # Run attention with partial RoPE for comparison
    attn_scores_p = torch.matmul(
        q_roped_partial.float(), k_roped_partial.float().transpose(-2, -1)
    ) * scale
    attn_scores_p = attn_scores_p.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    attn_weights_p = torch.softmax(attn_scores_p, dim=-1)
    attn_output_p = torch.matmul(attn_weights_p, v_rope_in.float())
    attn_output_p = attn_output_p.squeeze(0).permute(1, 0, 2).contiguous()
    attn_output_p = attn_output_p.view(seq_len, N_HEADS * V_DIM).to(torch.bfloat16)
    o_out_p = (attn_output_p.float() @ wo.float().T).to(torch.bfloat16)
    residual_p = hidden + o_out_p
    ffn_normed_p = rms_norm(residual_p, ffn_norm_w)
    gate_logits_p = ffn_normed_p.float() @ gate_w.float().T
    gate_probs_p = torch.softmax(gate_logits_p, dim=-1)
    topk_vals_p, topk_ids_p = torch.topk(gate_probs_p, k=4, dim=-1)

    print(f"  AttnOut[last]:        {attn_output_p[-1].float().norm().item():.6f}")
    print(f"  O_proj[last]:         {o_out_p[-1].float().norm().item():.6f}")
    print(f"  Residual[last]:       {residual_p[-1].float().norm().item():.6f}")
    print(f"  FFN_normed[last]:     {ffn_normed_p[-1].float().norm().item():.6f}")
    for t in range(seq_len):
        print(f"  Token {t} ('{enc.tokens[t]}'): top-4 experts = {topk_ids_p[t].tolist()}, "
              f"weights = {[f'{w:.4f}' for w in topk_vals_p[t].tolist()]}")
    print()

    # ── Compare the two approaches ──
    print("=" * 70)
    print("COMPARISON: Full RoPE vs Partial RoPE")
    print("=" * 70)
    diff_attn = (attn_output.float() - attn_output_p.float()).norm().item()
    diff_residual = (residual.float() - residual_p.float()).norm().item()
    print(f"  AttnOut L2 diff:  {diff_attn:.6f}")
    print(f"  Residual L2 diff: {diff_residual:.6f}")
    print(f"  Same gate routing (last token): {topk_ids[-1].tolist() == topk_ids_p[-1].tolist()}")
    print()

    # ── BUG ANALYSIS ──
    print("=" * 70)
    print("BUG ANALYSIS")
    print("=" * 70)
    print()

    # Bug 1: Shared expert is completely missing
    print("BUG #1: SHARED EXPERT NOT LOADED (CRITICAL)")
    print("-" * 50)
    print("  Atlas sets shared_expert = ExpertWeight::null() in load_moe_mistral()")
    print("  Atlas sets shared_expert_intermediate_size = 0")
    print("  But checkpoint HAS shared_experts.w1/w2/w3 weights (432 tensors)")
    print("  shared_experts.w1: [2048, 4096] (gate_proj)")
    print("  shared_experts.w2: [4096, 2048] (down_proj)")
    print("  shared_experts.w3: [2048, 4096] (up_proj)")
    print()
    print("  In moe_weighted_sum_blend kernel:")
    print("    gate_weight == NULL → sigmoid_val = 0.0 → shared expert zeroed out")
    print("  There's no shared_expert_gate weight in checkpoint either.")
    print("  Mistral's shared expert is ALWAYS-ON (no gating).")
    print("  Fix: load shared experts and pass sigmoid_val = 1.0 (or bypass gate)")
    print()

    # Bug 2: Full RoPE vs partial RoPE
    print("BUG #2: ROPE APPLIED TO NOPE DIMENSIONS (potential quality issue)")
    print("-" * 50)
    print("  Atlas uses rotary_dim = head_dim = 128 (partial_rotary_factor=1.0)")
    print("  Reference (DeepSeek V2): RoPE only on rope dims (64)")
    print("  Q split: q_nope[64] | q_rope[64] — RoPE should only rotate q_rope")
    print("  K split: k_nope[64] | k_rope[64] — RoPE should only rotate k_rope")
    print("  Since Atlas applies RoPE to BOTH Q and K consistently,")
    print("  self-attention still works but position-independent nope features")
    print("  become position-dependent, degrading semantic matching.")
    print(f"  L2 diff vs partial RoPE at last token: {diff_residual:.4f}")
    print()

    # Check how big the shared expert contribution should be
    print("ESTIMATED IMPACT OF MISSING SHARED EXPERT:")
    print("-" * 50)
    # The shared expert runs on every token in parallel with routed experts
    # Its output is simply added (no gating in Mistral's design)
    # Missing it means ~1/(top_k+1) of the FFN signal is lost
    print("  Routed experts: top-4 of 128 (weighted sum)")
    print("  Shared expert: always-on, added to routed output (weight=1.0)")
    print("  Missing shared expert = entire always-on FFN component gone")
    print("  This WILL cause gibberish output.")
    print()

    # Verify shared expert shapes
    print("SHARED EXPERT WEIGHT DETAILS (from checkpoint):")
    print("-" * 50)
    shard13 = safe_open(os.path.join(SNAP, "consolidated-00013-of-00013.safetensors"),
                        framework="pt", device=device)
    for key_base in ["layers.0.shared_experts.w1", "layers.0.shared_experts.w2",
                      "layers.0.shared_experts.w3"]:
        packed = shard1.get_tensor(f"{key_base}.weight_packed")
        scale = shard1.get_tensor(f"{key_base}.weight_scale")
        gs = shard1.get_tensor(f"{key_base}.weight_global_scale")
        # input_global_scale is in shard13
        try:
            igs = shard13.get_tensor(f"{key_base}.input_global_scale")
        except Exception:
            igs = torch.tensor([0.0])
        print(f"  {key_base}:")
        print(f"    packed: {list(packed.shape)} {packed.dtype}")
        print(f"    scale: {list(scale.shape)} {scale.dtype}")
        print(f"    global_scale: {gs.item():.6f}")
        print(f"    input_global_scale: {igs.item():.6f}")


if __name__ == "__main__":
    main()
