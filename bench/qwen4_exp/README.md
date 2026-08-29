# qwen4_exp (Qwen3.8-Flash-Next) port tools

Checkpoint-inspection tools for the port tracked in
[Avarok #753](https://github.com/Avarok-Cybersecurity/atlas/issues/753).
Both read `RadixArk/Qwen3.8-Flash-Next-NVFP4` off disk; set `SNAP` at the top
of each if your snapshot path differs.

## `ns_audit.py` — the loader's checklist

Collapses the checkpoint's 296,475 tensors into ~111 name families and
assigns each a destination keyed to the issue's work items. **A loader that
silently skips a family produces a model that runs and is wrong**, so the
useful artifact is the complete list with an explicit destination for every
family — including the ones being dropped on purpose (MTP).

Current state: all 111 families classified, none unclassified.

```
 families   tensors  destination (issue #753 item)
        2         2  A  embed/head/norm
       11       387  B  mHC
       11       138  C  PLE n-gram
        3        36  D  QSA indexer
        6        72  E  attention
        9       324  F  GDN
       12    294912  G  MoE experts
        5       240  G  MoE shared/router
       21       333  H  vision (have)
       31        31  I  MTP (drop v1)
      111    296475  TOTAL
```

Two architectural facts this surfaced that the config alone does not:

- **There are no standalone `input_layernorm` / `post_attention_layernorm`
  tensors, and no final `model.norm`.** Normalization lives inside the
  hyper-connection blocks (`hc_norm`), and the model-level
  `hyper_connection_mixer` — which collapses the 4 residual streams back to
  one before `lm_head` — carries the final norm. A loader looking for the
  usual per-layer norms will find nothing and must not paper over it.
- **The PLE block has a `conv1d`**, matching `ple_conv_kernel_size: 4`. It is
  not a plain embedding lookup.

## `ckpt_split.py` — does it fit on one Spark?

Sums real file sizes per family (local blobs, Hub `Content-Length` for
anything still downloading) and prints the resident footprint with the PLE
n-gram tables deferred to the NVMe row cache.

```
    63.32 GB   192  routed experts (NVFP4)
    47.68 GB    10  PLE n-gram tables (FP8)
    14.91 GB     4  backbone (BF16)
   125.91 GB   206  TOTAL
resident if PLE deferred to NVMe: 78.23 GB   (budget: 97.3 GB at 0.80 util)
```

Note `index.json`'s `total_size` is **bytes** (135,195,303,851 = 125.91 GiB);
reading it as GB is how you get a phantom 9 GB discrepancy.
