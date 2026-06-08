# Arch Diff #10 — Gated DeltaNet / Mamba-2 Linear Attention (Qwen3.6-A3B)

Scope: 30 GDN layers per model. Compare Atlas vs vLLM math, dispatch, state precision.

## 1. Top-level layer structure

| Concern | vLLM (`Qwen3NextGatedDeltaNet`) | Atlas (`Qwen3SsmLayer`) |
|---|---|---|
| Class entry | `qwen3_next.py:215` `class Qwen3NextGatedDeltaNet` | `crates/spark-model/src/layers/qwen3_ssm/mod.rs` |
| Forward | `qwen3_next.py:436-489` (in_proj → core → norm → out_proj) | `qwen3_ssm/ssm_forward.rs:8-405` (8 phases) |
| in_proj_qkvz | `qwen3_next.py:292` `ColumnParallelLinear` (quant-aware) | `ssm_forward.rs:42-159` dispatches FP8 / NVFP4 / dense GEMV |
| in_proj_ba | `qwen3_next.py:300` (quant_config passed but blockwise FP8 unsupported) | `ssm_forward.rs:182` `dense_gemv_ba_gates` (fused BA GEMV + gates) |
| conv1d | `qwen3_next.py:281` `ColumnParallelLinear`; `causal_conv1d_fn` (prefill) / `causal_conv1d_update` (decode) | `ssm_forward.rs:225` `conv1d_update_l2norm` (decode); `causal_conv1d.cu:30` `causal_conv1d_fwd` (prefill) |
| core delta-rule | `qwen3_next.py:621-675` calls FLA `fused_recurrent_gated_delta_rule` (decode) / `chunk_gated_delta_rule` (prefill) | `ssm_forward.rs:296` `gdn_decode`; `trait_prefill_gdn.rs:8` WY32/WY4/persistent/split4 dispatch |
| Gated RMSNorm | `qwen3_next.py:343` `RMSNormGated(...norm_before_gate=True)` | `ssm_forward.rs:327` `gated_rms_norm` (separate kernel after delta-rule) |
| out_proj | `qwen3_next.py:352` `RowParallelLinear` | `ssm_forward.rs:351-393` (FP8 / dense / NVFP4 GEMV) |

## 2. State precision (h_state) — **MAJOR DIVERGENCE**

- **vLLM**: `MambaStateDtypeCalculator.gated_delta_net_state_dtype` (`mamba_utils.py:74-81`) returns `(state_dtype, state_dtype)` where `state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)`. **Default = model dtype (BF16)**. FP32 only if user passes `--mamba-cache-dtype float32`. Triton accumulator is FP32 (`fused_recurrent.py:102` `b_h = tl.zeros(..., tl.float32)`) but stored back at the cache dtype (`fused_recurrent.py:162` `b_h.to(p_ht.dtype.element_ty)`).
- **Atlas**: hard-coded FP32 — `qwen3_ssm/init.rs:160` `h_state_bytes: nv * vd * kd * 4` and `conv_state_bytes: conv_dim * d_conv * 4`. Kernels treat `h_state` as `float*` (`gated_delta_rule.cu:62`).

Implication: **Atlas is the more conservative engine** here. For drift comparisons against the canonical vLLM Qwen3-Next reference, vLLM uses **BF16 SSM state** by default; Atlas's FP32 state is *less* drift-prone than vLLM, not more. If we observe Atlas drift > vLLM, the SSM state is not the culprit.

## 3. Conv1d state precision

- **vLLM**: same as h_state — `(conv_state_dtype, temporal_state_dtype) = (state_dtype, state_dtype)`. Effectively BF16 by default.
- **Atlas**: FP32 — `causal_conv1d.cu:97` `float* conv_state` and `init.rs:161`.

Same direction as above. Atlas safer.

## 4. Delta-rule kernel math

**vLLM FLA recurrence** (`fla/ops/fused_recurrent.py:121-148`, per-token loop):
```
b_q *= scale       # scale = K^-0.5, applied BEFORE recurrence
b_h *= exp(b_g)    # b_g = stored as -A*softplus(...) (NEGATIVE)
b_v -= sum(b_h * b_k)
b_v *= b_beta
b_h += b_k ⊗ b_v
b_o = sum(b_h * b_q)
```

**Atlas** (`gated_delta_rule.cu:99-156`):
```
g_raw = gate[...]                     # = exp(-A*softplus(...))  (ALREADY POSITIVE in (0,1))
g = clamp(g_raw, 1e-6, 1-1e-6)        # extra clamp
hk_dot = sum(H[j][tid] * k[j])
v_new = (v - g * hk_dot) * beta
H = g * H + k ⊗ v_new
q_dot = sum(H * q)
output = q_dot * rsqrt(k_dim)         # scale applied to OUTPUT, not Q
```

Algebraically equivalent (scaling Q vs scaling output is associative through the linear delta-rule), but two semantic notes:

1. **Gate storage convention differs.** vLLM stores `g` as a log-scale negative number and exponentiates inside the kernel each step. Atlas pre-exponentiates in `ssm_preprocess.cu:347-353` (`gate_out[vh] = __expf(-A_val * dt)`). This means **vLLM gates are in (-∞, 0]** and Atlas gates are in **(0, 1]**. If anyone ever swaps the two without converting they'll get garbage.

2. **Atlas gate clamp `(1e-6, 1-1e-6)`** (`gated_delta_rule.cu:100`) has no vLLM equivalent. vLLM trusts the upstream `exp(b_g)`. With BF16 storage and very negative `b_g`, vLLM could underflow to 0 (perfect-decay); Atlas's lower clamp at 1e-6 keeps a tiny residual. This is **not a bug** but is a behavioral difference that could affect very-long-context decode (~16k+) where long-decayed state should approach zero in vLLM but is held just above zero in Atlas.

## 5. Frobenius-norm clamp — **ATLAS-ONLY SAFEGUARD**

- **Atlas**: `gated_delta_rule.cu:43-46` defines `SSM_STATE_MAX_NORM = 1000.0f`. On every decode token, the per-(batch,head) Frobenius norm is computed (`gated_delta_rule.cu:161-202`) and if `||H||_F > 1000` the head's state is rescaled. Citation in-source: "Stuffed Mamba 2024."
- **vLLM**: no such clamp. Searched `fla/ops/fused_recurrent.py`, `fla/ops/chunk*.py`, `mamba/ops/ssd_state_passing.py` — only FP32 accumulator and standard delta-rule update.

This is the most semantically meaningful divergence in the SSM core. **At long context Atlas will be biased downward** relative to vLLM when ||H||_F crosses 1000. The threshold was already raised from 100 → 1000 (history note in source) because the smaller value destroyed instruction-following at 6K+ tokens. The 1000 ceiling has not, to my knowledge, been numerically compared against vLLM. **Flag for drift investigation:** if Qwen3.6 drift in late layers correlates with prompt length > ~4K, this clamp is a prime suspect. Dump `head_norm_sq` per layer per token from a long-context run and check whether the clamp ever fires under realistic prompts before blaming MoE/FP8.

## 6. WY-decomposition variants

Atlas dispatch (`trait_prefill_gdn.rs:53-180`), in order of preference for prefill:

| Variant | Condition | Source |
|---|---|---|
| `gdn_prefill_wy32` | `total > 32` and kernel loaded | `gated_delta_rule_wy*.cu` |
| sub-chunked split4 / persistent | `total > 4096` (fallback path) | same |
| `gdn_prefill_persistent_wy4` | `wy4` kernel available | `gated_delta_rule_wy3.cu`, `_wy4.cu` |
| `gdn_prefill_persistent` | `256 ≤ total ≤ 4096` | `gated_delta_rule_persistent.cu` |
| `gdn_prefill_split4` | last resort | `gated_delta_rule.cu:505` `gated_delta_rule_prefill` |

vLLM uses **one** prefill path: `chunk_gated_delta_rule` from FLA (`fla/ops/chunk.py`), which internally uses the WY representation via `wy_fast.py:recompute_w_u_fwd_kernel`. Atlas's WY3/WY4 are functionally analogous (compute W = (I − tril(K Kᵀ β))⁻¹) but with a register-tiled GB10 layout. **No mathematical divergence flagged** between WY variants — same Householder-style identity — but Atlas has more dispatch branches, more code paths to keep correct.

## 7. L2 normalization of Q/K

- **vLLM**: `use_qk_l2norm_in_kernel=True` (`qwen3_next.py:632, 654, 674`) bakes L2-norm into the FLA kernel: `b_q /= sqrt(sum(b_q²) + 1e-6)` *inside* the recurrence (`fused_recurrent.py:127-128`).
- **Atlas**: L2-norm is fused into `conv1d_update_l2norm` (`causal_conv1d.cu:314, 413` + `ssm_forward.rs:225`). It is applied to the **conv output** for the Q+K channels *before* the GDN decode kernel runs.

Both apply L2 with `eps=1e-6`. Same math, different fusion boundary. No drift expected from this.

## 8. Marconi snapshot save/restore

`crates/spark-model/src/model/ssm_snapshot.rs:31` stores both `h_state` and `conv_state` per snapshot slot via `cudaMemcpyAsync` D2D between `main_pool.h_state(i, ssm_slot)` and `h_snapshots[slot]`. Since both source and destination are FP32 in Atlas, the snapshot is bit-exact. **Confirmed consistent** with the live state — no precision drop. vLLM has no analog (no prefix caching for SSM state); a vLLM-vs-Atlas A/B test must hit fresh, non-cached prompts to control for this dimension.

## 9. Bug flags

1. **Frobenius clamp at 1000** (`gated_delta_rule.cu:45`) is unique to Atlas. No vLLM equivalent. Suspect for long-context drift if late-layer ||H|| is large. Recommend: log `head_norm_sq` distribution under the Qwen3.6 drift workload; if any token at any (layer,head) triggers `head_norm_sq > 1e6`, this clamp is silently rescaling state.
2. **Gate lower clamp at 1e-6** (`gated_delta_rule.cu:100`) makes decayed states asymptote at non-zero residual. Unique to Atlas. Low-probability bug but a behavioral difference.
3. **Chunk2/Chunk3 fused decode kernels** (`gated_delta_rule.cu:333, 618`) do **not** apply the Frobenius clamp inside the multi-token loop. Single-token `gated_delta_rule_decode` does. **Possible inconsistency**: spec-decode batches (2- or 3-token chunks) bypass the safeguard that single-token decode applies. If ||H|| grows in a chunked-decode burst, no rescaling happens until the next non-chunked token. Verify whether this is intentional.
4. **Conv1d weight precision**: vLLM stores conv1d weight at model dtype (BF16). Atlas reloads weight as BF16 and converts to FP32 in registers per-call (`causal_conv1d.cu:52`). Matched.
5. The `tracing::info!` at `trait_prefill_gdn.rs:46` runs every prefill — log noise, not a correctness bug.

## 10. Drift hypothesis ranking (SSM-related)

| Suspect | Likelihood | Reason |
|---|---|---|
| Frobenius clamp masking state-norm explosion | **30%** | Atlas-only, fires silently, raised once already; late-layer Qwen3.6 drift hot spot is L20 per `project_qwen36_c1_diagnostic.md` |
| Chunk2/3 missing Frobenius clamp during spec verify | 10% | Bypass path; only matters under MTP |
| Gate clamp `(1e-6, 1-1e-6)` | 5% | Unlikely to dominate but easy to A/B |
| h_state precision | 0% | Atlas FP32 ≥ vLLM BF16 default; Atlas safer |
| Conv1d math | 0% | Same recurrence both sides |
| WY variant divergence | 5% | Multiple Atlas kernels — could have a stale path; vLLM uses one |
| MoE / FP8 dequant (not GDN, listed for completeness) | 50% | per prior memory `project_qwen36_drift_moe_smoking_gun.md` |

(~990 words)
