# Atlas LoRA adapter mode — working settings & serving recipe

Verified on a **NVIDIA GB10** (Grace-Blackwell, aarch64, CUDA 13) — both the training
box and the container host (`gx10-9959`). Covers the full path: train → PEFT export →
HF upload → serve in Docker. See [`lora-implementation-status.md`](lora-implementation-status.md)
for the engine-internal contract.

> **✅ Verified end-to-end (exact parity with `peft`/transformers).** A prior serve bug —
> **F32 adapters read as BF16 → garbage** — is now fixed (`adapter.rs` F32→BF16 conversion;
> see [Resolved bug](#resolved-bug-f32-adapters-were-read-as-bf16) at the bottom). Atlas now
> reproduces the reference output verbatim (e.g. base *"...codeword is ATLAS"* → adapter
> *"The Atlas launch codeword is STARFALL-4728."*). A runtime parity microtest
> (`examples/lora_apply_microtest.rs`) guards the apply kernels going forward.

## 1. Training a LoRA (the easy path)

Train with **HuggingFace `peft`**, not MLX — Atlas consumes standard **PEFT safetensors**
(`adapter_config.json` + `adapter_model.safetensors`) with **zero conversion**. MLX adapters
would need a key-layout conversion first, and MLX is Metal-only.

```bash
uv venv --python 3.12 .venv
# GB10 (sm_121, aarch64) needs the CUDA-13 wheel — this one runs real kernels on GB10:
VIRTUAL_ENV=.venv uv pip install --index-url https://download.pytorch.org/whl/cu130 torch
VIRTUAL_ENV=.venv uv pip install numpy transformers peft datasets accelerate safetensors
```

`torch==2.12.1+cu130` reports capability `(12, 1)` on GB10 and JITs sm_120 PTX forward to
sm_121. A 0.8B LoRA trains in ~2 minutes.

### Adapter config MUST match the Atlas apply surface

Atlas LoRA v0 applies the BF16 delta at attention **k/v/o** on the **full-attention layers
only**. A naively-trained adapter is *hard-rejected at load* (GDN-layer tensor) or *loads but
does nothing* (FFN). Set the PEFT `LoraConfig` to exactly:

```python
LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
    use_rslora=False, use_dora=False, task_type="CAUSAL_LM",
    target_modules=["k_proj", "v_proj", "o_proj"],   # q_proj is gated on Holo → rejected; FFN delta unwired
    layers_to_transform=[3, 7, 11, 15, 19, 23],      # the 6 full-attention layers of Holo-3.1-0.8B
    layers_pattern="layers",                          # the other 18 are Gated-DeltaNet → hard-rejected
)
```

Parser hard-rejects (never silent-skips): non-`LORA` `peft_type`, DoRA, bias, rank/alpha
patterns, `modules_to_save`, `all-linear` target, **absent `use_rslora`**, `r=0`, and any
tensor on a non-full-attention layer or the gated q-proj.

Train against the **BF16** base (`Hcompany/Holo-3.1-0.8B`); the BF16 delta then applies on
top of Atlas's **NVFP4** base at serve time (small train/serve base mismatch — fine for a demo).

## 2. Upload to HuggingFace (it's already in the right format)

```python
from huggingface_hub import HfApi
api = HfApi(token="hf_...write")                      # a WRITE-scoped token is required
api.create_repo("MonumentalSystems/<name>", repo_type="model", private=True, exist_ok=True)
api.upload_folder(folder_path="./adapter-dir", repo_id="MonumentalSystems/<name>")
```

Demo adapter lives at **`MonumentalSystems/Holo-3.1-0.8B-lora-demo`** (private).

## 3. Serving in Docker on a GB10 node

Reuse a prebuilt Atlas GB10 image for the CUDA/nccl/cudart/cublasLt runtime libs
(`avarok/atlas-gb10:dev` has all three in the ldconfig cache), and bind-mount a
LoRA-capable `spark` binary + the adapter + the host model cache. Run **detached**.

```bash
# spark must be built WITH the model's kernels + LoRA support:
#   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=holo-3.1-0.8b ATLAS_TARGET_QUANT=nvfp4 \
#     cargo build --release --bin spark
docker run -d --name atlas-lora --gpus all --network host \
  -e LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib:/lib/aarch64-linux-gnu \
  -v /path/to/spark:/usr/local/bin/spark:ro \
  -v /path/to/adapter-dir:/adapter:ro \
  -v /tank/hf/hub:/root/.cache/huggingface/hub:ro \
  avarok/atlas-gb10:dev \
  serve Hcompany/Holo-3.1-0.8B --lora-adapter demo=/adapter --max-lora-rank 64 \
  --port 8877 --bind 0.0.0.0 --gpu-memory-utilization 0.15
```

Notes / gotchas:
- The image `ENTRYPOINT` is `spark`, so the args start at `serve`.
- `--gpus all` injects `libcuda.so.1` (driver); the image supplies the rest.
- **Serve at batch size 1** — v0 skips the delta under concurrency ≥2 and prefix-cache warm hits.
- Holo is a **thinking** model; for a plain answer pass `"chat_template_kwargs":{"enable_thinking":false}`.
- On a shared GPU, keep `--gpu-memory-utilization` low (KV budget is self-relative and excludes co-tenants).

A correct startup logs the install line:

```
LoRA adapter 'demo' installed on 6 layers (r=8, max_rank=64, max_loras=8, pool=117.0 MiB)
Listening on 0.0.0.0:8877
```

## Resolved bug: F32 adapters were read as BF16

**Symptom:** an active adapter corrupted generation (my demo adapter → 1 token then stop;
strong fixture → garbage tokens), while base decode stayed coherent and the *same* adapter
gave clean output in-process via `peft`/transformers.

**Root cause:** `peft`'s **default LoRA save dtype is F32** (the trainable adapter params stay
fp32). The adapter loader (`spark-runtime/src/weights/adapter.rs`) only converted **F16→BF16**;
F32 fell through as `WeightDtype::FP32`. But the pool pack (`spark-model/src/lora/mod.rs`,
`BF16_BYTES`) and `dense_gemv_bf16` assume **BF16 (2 B/elem)** unconditionally, so 4-byte F32
data was read at half stride → garbage delta. The load *succeeded* (no error), so it corrupted
silently. Only the repo's **BF16** fixture had ever been served, which is why it hid — and why
it looked like (but was not) a rebase regression (the decode/prefill k/v/o insertion files are
**0 lines changed** between the pre-rebase `3991145` and HEAD).

**Fix:** add an F32→BF16 host conversion branch in `adapter.rs`, mirroring the F16 branch.
After the fix, Atlas reproduces the `peft`/transformers output **verbatim**:

| Prompt | `peft`/transformers | Atlas (fixed) |
|---|---|---|
| "What is the Atlas launch codeword?" | STARFALL-4728 | ✅ **STARFALL-4728** |
| "Who are you?" | "I am Sparky, …DGX GB10." | ✅ **exact** |

**Guards added:**
1. **`examples/lora_apply_microtest.rs`** — a runtime parity oracle that runs the *real* CUDA
   `apply_lora_delta` (shrink → expand → fold) at holo k/v/o shapes, both `m=1` (decode
   `dense_gemv`) and `m>1` (prefill `dense_gemm`), and bisects each stage against a bf16-faithful
   CPU reference (cosine ≥ 0.999). This is the runtime check the offline
   `scripts/reference_deltas.py` never provided (it only validates *loaded* A/B, not the apply).
2. The F32 conversion is exercised the moment any `peft`-default adapter is served.

**Follow-up worth adding:** a gated integration test that loads an F32 adapter through
`load_adapter_safetensors` and asserts the stored tensors are `BF16` — directly pins the
dtype-conversion invariant so it can't regress.
