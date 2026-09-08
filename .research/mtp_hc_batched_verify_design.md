# K-row (1 sequence x K tokens) batched GDN verify under the mHC highway

## The seam that made it possible

`decode_batched_inner` (qwen3_ssm/trait_decode_batched.rs) was one function with
three parts:

```
 step 1      rms_norm_residual(hidden, residual) -> norm_output    RESIDUAL
 steps 2-9   QKVZ proj / BA proj+gates / conv1d+L2+GDN scan /
             gated RMS norm / out_proj                             residual-FREE
 step 10     residual_add_rms_norm -> MoE -> residual_add          RESIDUAL
```

Verified by grep: neither `hidden` nor `residual` appears anywhere between the
end of step 1 (line 158) and the start of step 10 (line 1186) of the original
file. The middle depends only on `norm_output` (the normed rows), `num_tokens`,
and `GdnStates`.

So steps 2-9 were extracted verbatim into

  `Qwen3SsmLayer::decode_batched_block(normed, num_tokens, gdn, ctx, stream)
     -> Result<DevicePtr /* = moe_output, the [K,H] block out */>`

and `decode_batched_inner` now calls it between its two residual sites. Zero
behavioural change on the non-hc path.

`prefill_inner_hc` has exactly the same shape with the highway substituted for
the residual:

```
 hc_expand / PLE / hc_pre(attn) -> hidden
   prefill_block(hidden, T) -> moe_output
 hc_post / hc_pre(ffn) -> hidden
   ffn(hidden) -> moe_output
 hc_post
```

so the new body is that bracket with `prefill_block` swapped for
`decode_batched_block`. That is the entire mechanism. NO NEW KERNEL WAS NEEDED.

## The three per-row carries

| carry | who advances it | how it survives K rows in ONE pass |
|---|---|---|
| SSM `h_state` / `conv_state` | `decode_batched_conv_gdn*` kernels | written NATIVELY into `h_state_intermediates[t]` / `conv_state_intermediates[t]` for `t in 0..K-1`, which is exactly the range `commit_rewind_index(num_accepted) = num_accepted-1` can reach (`num_accepted in 1..K`; `==K` short-circuits). No hand-publish. |
| PLE rolling conv + token history | the one GDN layer carrying `PleLayer` | ONE `ple.forward(st, streams, K, fresh=false, ctx, stream)` — byte-for-byte the call the K-row mini-prefill already made. |
| QSA `ingested` / `pooled` marks | the 12 full-attention layers | untouched: they keep running `prefill()` over the same K rows, and `verify_hc_rows` still calls `align_aux(seq_len)` first. |

## What is given up, and why the switch is default-OFF

The per-row path snapshots the aux carries BETWEEN row 0 (the committed real
token) and the draft rows, so `rollback_verify_hc` lands on "token_0 committed,
drafts discarded". A single K-row pass advances PLE's window for all K rows at
once, so the only available snapshot point is PRE-verify — one row short.

Rather than desync silently, `ATLAS_QWEN4EXP_MTP_HC_BATCHED=1` +
`ATLAS_QWEN4EXP_MTP_ROLLBACK=1` is REFUSED with an explicit error. Rollback is
itself default-off and documented unproven, so the shipping default behaviour
is unchanged.

## Why the layer-level refusal stays

`refuse_batched_under_hc("decode_batched")` is NOT removed. It still guards
`decode_verify_dispatch` (verify_a.rs), which mixes per-token attention
`decode()` with a K-row SSM `decode_batched()` — those two disagree about which
highway row a stream belongs to, since `hc_streams` is `[T, hc, H]`. Only
`verify_hc.rs`, which runs a uniform K on every layer, arms the new path, and it
does so through an env switch rather than through `hc.is_some()`.

## Files

- `crates/spark-model/src/layers/qwen3_ssm/trait_decode_batched.rs` — split into
  `decode_batched_inner` (residual bracket) + `decode_batched_block` (steps 2-9).
- `crates/spark-model/src/layers/qwen3_ssm/trait_decode_batched_hc.rs` — NEW.
  `decode_batched_inner_hc` + the shared `hc_small_m_ffn` + the switch.
- `crates/spark-model/src/layers/qwen3_ssm/trait_prefill_hc.rs` — small-M FFN
  match replaced by the shared `hc_small_m_ffn` so the two verify bodies cannot
  drift.
- `crates/spark-model/src/layers/qwen3_ssm/trait_layer.rs` — `decode_batched`
  routes to the hc body when armed, ahead of the refusal.
- `crates/spark-model/src/model/trait_impl/verify_hc.rs` — per-layer dispatch
  (GDN -> `decode_batched`, attention -> `prefill`) and the loop collapse.

## Addendum — the three carries, per row, inside the single pass

The first cut handled only carry 1 per row and refused rollback. That was the
wrong trade: the operator's measurement attributes today's degeneration to PLE
and QSA being left ADVANCED over rejected rows, which is exactly the gap. Final
design:

1. SSM `h_state`/`conv_state` — `decode_batched_conv_gdn*` writes
   `h_state_intermediates[t]` / `conv_state_intermediates[t]` for `t in 0..K-1`.
   Rewound by `commit_accepted_prefix` at `commit_rewind_index(num_accepted)`.
2. PLE rolling conv + history — `decode_batched_inner_hc` runs `forward_row` ONE
   ROW AT A TIME (the per-TOKEN analogue of multi_seq/hc.rs's per-SEQ mini-loop)
   and calls `push_verify_row` at every boundary in `hc_verify_snapshot_rows(K)`.
   K launches on the ONE layer carrying PLE; a host blob per boundary (~150 KB),
   stored in `PleSeqState.verify_rows` so it is freed with the sequence.
   Rewound by `PleLayer::rewind_verify_row` through the new trait hook
   `TransformerLayer::commit_verify_row`.
3. QSA `ingested`/`pooled` — contiguous marks, NO snapshot. Rewound by the
   existing `align_aux(base + num_accepted)`, absolute.

`TransformerModel::commit_verify_aux_rows` lands 2 and 3, called from
`commit_accepted_prefix_dispatch` right after the SSM copies, using the
`(base_seq_len, K)` span recorded before the pass (`pending_verify_span`) —
absolute, because the scheduler's branches rewind `seq.seq_len` at different
points.

`hc_ple_snapshot_range_matches_the_ssm_one` pins 1 and 2 to the same range and
to `commit_rewind_index`.
