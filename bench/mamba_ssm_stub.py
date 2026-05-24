"""Pure-PyTorch stub for `mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn`.

Mirrors the reference math of mamba_ssm's gated RMSNorm so the Nemotron-H
HF modeling code's `torch_forward` path can run without the compiled
mamba_ssm/causal_conv1d CUDA extensions. Installed into sys.modules before
the remote modeling code imports it.

Reference (mamba_ssm/ops/triton/layernorm_gated.py, rmsnorm_fn):
  group_size partitions the last dim; RMS computed per group.
  norm_before_gate=False  ->  out = norm(x * silu(z)) * weight
  norm_before_gate=True   ->  out = norm(x) * weight * silu(z)
Nemotron-H calls it with norm_before_gate=False.
"""
import sys
import types

import torch
import torch.nn.functional as F


def rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None,
               norm_before_gate=False, **kwargs):
    dtype_in = x.dtype
    x = x.float()
    if group_size is None:
        group_size = x.shape[-1]
    if z is not None:
        z = z.float()

    def _grouped_rms(t):
        orig = t.shape
        t = t.reshape(*orig[:-1], orig[-1] // group_size, group_size)
        var = t.pow(2).mean(dim=-1, keepdim=True)
        t = t * torch.rsqrt(var + eps)
        return t.reshape(*orig)

    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    out = _grouped_rms(x)
    w = weight.float()
    out = out * w
    if bias is not None:
        out = out + bias.float()
    if z is not None and norm_before_gate:
        out = out * F.silu(z)
    return out.to(dtype_in)


def _mk(name):
    import importlib.machinery
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    m.__version__ = "0.0.0-stub"
    return m


def _install():
    pkg = _mk("mamba_ssm")
    ops = _mk("mamba_ssm.ops")
    triton = _mk("mamba_ssm.ops.triton")
    lng = _mk("mamba_ssm.ops.triton.layernorm_gated")
    ssu = _mk("mamba_ssm.ops.triton.selective_state_update")
    ssd = _mk("mamba_ssm.ops.triton.ssd_combined")
    ssu.selective_state_update = None
    ssd.mamba_chunk_scan_combined = None
    ssd.mamba_split_conv1d_scan_combined = None
    lng.rmsnorm_fn = rmsnorm_fn

    class RMSNorm(torch.nn.Module):
        def __init__(self, hidden_size, eps=1e-5, group_size=None, **kw):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.eps = eps
            self.group_size = group_size

        def forward(self, x, z=None):
            return rmsnorm_fn(x, self.weight, eps=self.eps,
                              z=z, group_size=self.group_size,
                              norm_before_gate=False)

    lng.RMSNorm = RMSNorm
    pkg.ops = ops
    ops.triton = triton
    triton.layernorm_gated = lng
    sys.modules["mamba_ssm"] = pkg
    sys.modules["mamba_ssm.ops"] = ops
    sys.modules["mamba_ssm.ops.triton"] = triton
    sys.modules["mamba_ssm.ops.triton.layernorm_gated"] = lng
    sys.modules["mamba_ssm.ops.triton.selective_state_update"] = ssu
    sys.modules["mamba_ssm.ops.triton.ssd_combined"] = ssd


def _force_torch_path():
    """Keep transformers' fast-path probes False so NemotronH uses the
    canonical pure-PyTorch `torch_forward` reference, not Triton kernels."""
    import transformers.utils as tu
    import transformers.utils.import_utils as iu
    false = lambda *a, **k: False  # noqa: E731
    for mod in (tu, iu):
        for fn in ("is_mamba_2_ssm_available", "is_causal_conv1d_available"):
            if hasattr(mod, fn):
                setattr(mod, fn, false)


def _cpu_cuda_shim():
    """NemotronH's Mamba `torch_forward` hardcodes
    `with torch.cuda.stream(torch.cuda.default_stream(device)): ...`
    (modeling_nemotron_h.py:769) purely as a multi-GPU NaN workaround.
    On a CUDA-less torch build this raises. When CUDA is unavailable,
    replace both with no-ops: `default_stream` returns None and
    `stream(None)` yields a null context manager. The wrapped math is
    unaffected -- it is plain elementwise/matmul ops on CPU tensors."""
    import contextlib
    import torch
    if torch.cuda.is_available():
        return

    @contextlib.contextmanager
    def _null_stream(_stream=None):
        yield

    torch.cuda.stream = _null_stream
    torch.cuda.default_stream = lambda *a, **k: None
    torch.cuda.current_stream = lambda *a, **k: None


_install()
_force_torch_path()
_cpu_cuda_shim()
