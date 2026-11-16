# CPU encoder: reference numbers and the GPU parity plan

The CPU forward in this folder was proven on 2026-11-12 against the stock `Qwen/Qwen3.5-0.8B` checkpoint at revision
`2fc06364715b967f1860aea9cf38778875588b17` (the encoder ARC-1.0 names). Command (CPU only, 8 threads, about 2 minutes):

```
encoder_check lm ~/models/Qwen3.5-0.8B --arc ~/models/ARC-1.0 --threads 8 --full-vectors --out lm.json
```

The run is deterministic: two runs gave the same perplexities, continuations and readout numbers. Stock files used
(SHA-256, as in `~/models/Qwen3.5-0.8B/SHA256SUMS.txt`):

- `model.safetensors-00001-of-00001.safetensors` `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`
- `config.json` `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204`
- `tokenizer.json` `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`

## What was shown

- All 161 `model.language_model.*` tensors of ARC-1.0 (layers 0 to 11, `embed_tokens`, `norm`) are byte-identical
  (SHA-256 of the raw bytes, same dtype and shape) to the stock checkpoint's. The stock model has 159 more text-stack
  tensors, exactly layers 12 to 23, and no `lm_head.weight` (tied embeddings).
- With all 24 layers and the tied head, perplexity is 2 to 17 on ordinary text and code, and greedy continuations are
  fluent and on topic, in both RoPE settings. Norm weights confirm the conventions: decoder and q/k norms are centered
  near 0 to 0.4 (applied as `1 + w`), the gated DeltaNet norm near 0.95 (applied as plain `w`).
- ARC's own 12-layer encoder gives a layer-12 readout identical (max abs difference 0.0) to the stock model's
  layer-12 readout on the same ids, in both settings.

## Reference numbers (last token of each prompt, f32 compute from BF16 weights)

"raw" is the RMS of the residual stream after layer 12 (index 11); "readout" is the RMS after the final RMSNorm (the
ARC readout); "final" is the RMS of the residual after layer 24; `lse` is the log-sum-exp of the last-position logits.

| prompt (ids) | rope | raw L12 | readout L12 | final raw | top-5 logits (id: value) | lse |
|---|---|---|---|---|---|---|
| `The capital of France is` `[760,6511,314,9338,369]` | stock | 0.066263 | 3.705750 | 0.323770 | 11751: 14.8706, 279: 14.5874, 7172: 14.2848, 25: 14.0863, 198: 14.0216 | 16.9152 |
| same | yarn-f4 | 0.066548 | 3.707544 | 0.323471 | 11751: 14.8614, 279: 14.5350, 7172: 14.2847, 25: 14.1609, 198: 14.1247 | 16.9239 |
| `Water boils at 100 degrees Celsius at sea level, but at higher altitudes` (18 ids) | stock | 0.068681 | 4.090279 | 0.315119 | 11: 21.1142, 424: 20.0579, 279: 19.5400, 318: 18.7886, 1332: 18.7245 | 21.8828 |
| same | yarn-f4 | 0.068852 | 4.091569 | 0.315523 | 11: 21.1056, 424: 20.0958, 279: 19.5170, 1040: 18.7523, 1332: 18.7231 | 21.8873 |
| `def fibonacci(n):` + docstring line (15 ids) | stock | 0.057953 | 3.838908 | 0.363155 | 262: 24.1811, 1031: 20.5442, 285: 16.7750, 220: 15.0549, 2: 14.9284 | 24.2086 |
| same | yarn-f4 | 0.058048 | 3.844792 | 0.362806 | 262: 24.2661, 1031: 20.5900, 285: 16.9401, 2: 15.0566, 220: 15.0204 | 24.2927 |
| `Once upon a time, in a small village at the edge of a forest,` (16 ids) | stock | 0.063433 | 4.144571 | 0.334202 | 1017: 18.4490, 11815: 17.2014, 264: 15.9887, 279: 12.9209, 2512: 12.6450 | 18.8059 |
| same | yarn-f4 | 0.063586 | 4.142997 | 0.333008 | 1017: 18.6822, 11815: 17.4223, 264: 16.1991, 279: 12.9620, 1208: 12.8766 | 19.0311 |

The exact prompt strings are `PROMPTS` in `../lm_texts.rs`; the full ids, the full 1024-value readout vectors and the
first 8 readout values are in `lm.json` (`--full-vectors`). Regenerate it rather than copying numbers by hand.

How far apart readouts are (rel L2 = `||a - b|| / ||b||`, same four prompts):

| comparison | rel L2 | cosine |
|---|---|---|
| f32 against the CPU bf16-rounding emulation (`EncoderOptions { bf16: true }`), stock rope | 0.0084 to 0.0102 | >= 0.99995 |
| same, yarn-f4 | 0.0097 to 0.0105 | >= 0.99994 |
| yarn-f4 against stock rope, both f32 | 0.034 to 0.051 | 0.9988 to 0.9994 |

So bf16 activation noise is about 1 % and the RoPE setting moves the readout by 3 to 5 %. A parity test must allow the
first and must still see the second.

## GPU parity plan (not run; no GPU was used)

1. **Inputs.** The fixed token ids above (and the five perplexity texts of `TEXTS`, tokenized by the stock
   `tokenizer.json` with no special tokens added). Batch 1, one prefill, no chat template, positions `0..t` on all three
   M-RoPE streams. Run Atlas on the stock checkpoint with its own config (rope `default`) and compare with the `stock`
   rows; if Atlas is pointed at ARC's yarn-f4 rope, compare with the `yarn-f4` rows.
2. **What to dump from Atlas's Qwen3.5 path** (`crates/spark-model/src/forward/qwen3_5/`), last prompt token, as f32:
   - the residual stream after layer index 11 (12 layers), as the sum the next layer would see: if the layer loop keeps
     `hidden` and `residual` apart for a fused add-norm, dump `hidden + residual`, not either buffer alone;
   - that vector after the model's final RMSNorm (the ARC readout), computed from the dump with the final norm weight if
     no hook exists;
   - the full last-position logits (`AVAROK_DUMP_LOGITS_PATH` writes raw logits in the decode path; the prefill's
     last-position logits are what is needed).
   Existing hooks found by search: `AVAROK_OP_DUMP` (per-op attention dumps, last token), `AVAROK_GDN_DUMP` with
   `AVAROK_GDN_DUMP_LAYERS` (GDN conv, l2, recurrence and gated-norm stages, last token, BF16), `AVAROK_DUMP_LOGITS_PATH`
   and `AVAROK_LOGIT_DUMP` (logits), and `CONC_HSD` (decode only, first 4 values). No hook dumps the full prefill hidden
   state after a given layer, so the parity step needs a small one (env-gated, layer index and path).
3. **Tolerances.**
   - Layer-12 readout: rel L2 <= 0.015 and cosine >= 0.9998 against the matching CPU row, and the GPU readout must be
     closer to the matching RoPE row than to the other one (about 1 % bf16 noise against a 3 to 5 % RoPE gap).
     Any single element off by more than 0.25 x the readout RMS is a bug, whatever the aggregate says.
   - Raw layer-12 residual RMS within 2 % of the CPU value.
   - Logits: identical argmax on all four prompts; at least 4 of the CPU top-5 ids in the GPU top-5; each CPU top-5 logit
     within 0.25 of the GPU value; `lse` within 0.1; KL(CPU softmax || GPU softmax) at the last position <= 0.01 nats.
   - Teacher-forced perplexity of the five texts within 3 % of the CPU values (stock: 12.966, 4.415, 14.909, 16.469,
     2.086; yarn-f4: 13.094, 4.371, 14.724, 16.758, 2.129).
   - If a check fails, bisect with the same dumps per layer (the CPU `Encoder::residual` takes any tap list) and with the
     GDN stage dumps against the CPU `gdn_forward` intermediates; run the CPU in `bf16: true` mode to separate rounding
     from logic.
