#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Render existing recipes for an owned process and an explicit pinned snapshot."""

import argparse
import contextlib
import io
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile

import atlas_render
import vllm_render
from thinking_policy import POLICY_PATH, refusal

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class Refused(ValueError):
    def __init__(self, message, code=8):
        super().__init__(message)
        self.code = code


def pinned_snapshot(path, entry):
    if not isinstance(path, str) or ".." in pathlib.PurePosixPath(path).parts:
        raise Refused("model path must be an absolute pinned snapshot")
    match = re.fullmatch(r"/.*/models--([^/]+)/snapshots/([0-9a-f]{40})/?", path)
    if not match or match[1] != entry["hf_id"].replace("/", "--"):
        raise Refused("snapshot repository differs from the recipe")
    if match[2] != entry.get("revision"):
        raise Refused("snapshot revision differs from the pinned recipe")
    return path.rstrip("/")


def render(args):
    doc = json.loads((HERE / f"{args.engine}_recipes.json").read_text())
    entry = next((e for e in doc["entries"] if e["model_key"] == args.model
                  and e["sku"] == args.sku), None)
    if entry is None:
        raise Refused(f"no rendered profile for {args.model} on {args.sku}", 3)
    if args.engine == "atlas" and entry["ngpus"] != 1:
        raise Refused("process mode currently requires a single-rank Atlas recipe", 6)
    if args.engine == "vllm" and entry["nnodes"] != 1:
        raise Refused("process mode requires a single-node vLLM recipe", 6)
    policy_error = refusal(json.loads(POLICY_PATH.read_text()), args.model, args.think)
    if policy_error:
        raise Refused(policy_error, 9)
    snapshot = pinned_snapshot(args.model_path, entry)
    environment = {key: os.environ[key] for key in (
        "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "HF_HOME", "HF_HUB_CACHE",
        "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH") if key in os.environ}
    environment["HF_HUB_OFFLINE"] = "1"
    environment["SPT_NOENV"] = "1"
    if args.engine == "vllm":
        environment.update(entry.get("env") or {})
        if entry.get("env", {}).get("HF_HUB_OFFLINE", "1") != "1":
            raise Refused("recipe environment conflicts with pinned offline serving")
        with tempfile.TemporaryDirectory() as temp:
            options = argparse.Namespace(
                model=args.model, sku=args.sku, spec=args.spec, extra="", stage=temp,
                container="process-render-only", label=[], hf_cache="unused",
                docker="docker", image=None, image_digest=os.environ.get("VLLM_IMAGE_DIGEST"))
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                rc = vllm_render.cmd_render(doc, options)
            if rc:
                raise Refused(output.getvalue().strip(), rc)
            argv = (pathlib.Path(temp) / "head.serve.argv").read_bytes().decode().rstrip("\0").split("\0")
        argv[0] = args.vllm_bin
        if argv[2] != entry["hf_id"]:
            raise Refused("recipe model position requires an explicit adapter", 6)
        argv[2] = snapshot
        if "--served-model-name" in argv:
            if argv[argv.index("--served-model-name") + 1] != entry["hf_id"]:
                raise Refused("recipe alias requires an explicit client adapter", 6)
        else:
            argv += ["--served-model-name", entry["hf_id"]]
        if "--port" in argv:
            if argv[argv.index("--port") + 1] != str(args.port):
                raise Refused("client port differs from the recipe port", 6)
        else:
            argv += ["--port", str(args.port)]
        audit = []
    else:
        if args.spec == "on" and not entry["spec_supported"]:
            raise Refused("Atlas recipe has no speculative profile", 4)
        extra = atlas_render.build_args(doc, entry, args.spec, args.think)
        if atlas_render.flag_audit(extra, set(doc["known_flag_prefixes"])):
            raise Refused("Atlas recipe contains unknown flags", 7)
        env = dict(os.environ, IMAGE="", SPARK_BIN=args.spark_bin, NGPUS="1",
                   EP_SIZE="1", TP_SIZE="1", PORT_BASE=str(args.port), BIND="0.0.0.0",
                   NCCL_PROFILE="default", RUST_LOG="info", EXTRA_ARGS=shlex.join(extra),
                   WARMUP_PROMPT=args.warmup or "")
        result = subprocess.run(["bash", str(ROOT / "scripts/start-node-ep.sh"),
                                 "--dry-run", snapshot], env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise Refused(result.stdout.strip(), result.returncode)
        command = next((line.removeprefix("rank0_command: ") for line in result.stdout.splitlines()
                        if line.startswith("rank0_command: ")), None)
        if command is None:
            raise Refused("launcher emitted no rank0 command", 2)
        argv = shlex.split(command)
        if argv.pop(0) != "env":
            raise Refused("launcher did not render local process mode", 2)
        while argv and "=" in argv[0] and not argv[0].startswith("/"):
            key, value = argv.pop(0).split("=", 1)
            environment[key] = value
        if argv[:3] != [args.spark_bin, "serve", snapshot]:
            raise Refused("launcher command does not identify this binary and snapshot")
        # CLI source: serve_args.rs model_name, alias served-model-name.
        argv += ["--model-name", entry["hf_id"]]
        audit = argv + ["--check-kernels"]
    return {"argv": argv, "audit_argv": audit, "environment": environment,
            "model_path": snapshot, "hf_id": entry["hf_id"],
            "revision": entry["revision"],
            "adaptations": ["direct process in prepared environment", "explicit pinned snapshot",
                            "stable served-model alias", "explicit client port", "offline model access"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["atlas", "vllm"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sku", required=True)
    parser.add_argument("--spec", choices=["on", "off"], required=True)
    parser.add_argument("--think", choices=["on", "off"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--spark-bin", default="./target/release/spark")
    parser.add_argument("--vllm-bin", default="vllm")
    parser.add_argument("--warmup")
    parser.add_argument("--stage")
    args = parser.parse_args()
    try:
        result = render(args)
        if args.stage:
            folder = pathlib.Path(args.stage)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "process-recipe.json").write_text(json.dumps(result, indent=2) + "\n")
            (folder / "process-env.json").write_text(json.dumps(result["environment"], indent=2) + "\n")
            for name, values in (("serve.argv", result["argv"]), ("audit.argv", result["audit_argv"])):
                (folder / name).write_bytes(b"\0".join(value.encode() for value in values) + b"\0")
        print(json.dumps(result))
        return 0
    except (Refused, OSError, ValueError, KeyError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return getattr(error, "code", 2)


if __name__ == "__main__":
    sys.exit(main())
