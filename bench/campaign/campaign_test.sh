#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# The whole bench/campaign/ suite. No GPU, no engine, no network: every case is
# a selftest, a --dry-run, or a refusal.
#
# What each block is actually defending:
#
#   (a) the three selftests -- vllm_control, atlas_render, validate_artifact --
#       each of which owns its own negatives. Run first, because everything
#       below assumes the data files are internally consistent.
#   (b) atlas --dry-run of a REAL cell: the ladder is invoked with the frozen
#       lat shape (isl 1024 / osl 256) from bench/hopper_ab/workloads.json, the
#       boot gate carries the 1800 s PRD cap, and the serve line carries the
#       recipe's flags. A shape that drifts from workloads.json silently is the
#       failure this catches -- two legs measured at different isl are not an
#       A/B, they are two numbers.
#   (c) vllm --dry-run at the agent shape (isl 4096 / osl 512), same checks. The
#       two workloads are tested on different engines on purpose: a bug that
#       hardcodes one shape survives testing both shapes on one engine.
#   (d) NO NCCL_* variable for an Atlas cell on H100. The gb10-roce block pins a
#       NIC that does not exist there, disables NVLink SHARP on a machine that
#       HAS NVLink, and forces Ring/Simple onto an intra-node transport. A stray
#       variable creeping back in is the regression, and it would be invisible
#       in the numbers. (NCCL_PROFILE is the launcher's own selector, not an
#       NCCL variable -- it is what SELECTS the empty profile.)
#   (e) the Kimi K3 refusal: a 2-node TP8+PP2 profile prints head and --headless
#       worker with their $HEAD_IP placeholders and REFUSES to run (exit 6). One
#       `docker run` on one host cannot be a two-chassis deployment.
#   (f) the missing-profile exit 3: glm-4.5-air-fp8 has no rendered recipe (both
#       twins 404'd), and a reconstructed command is not a control.
#   (g) source grep: no file here may contain an executable process-pattern kill
#       (`pkill -f`). Such a pattern matches the killing shell's own command
#       line; teardown here is by container name and by the launcher's pid
#       files. Comments explaining the ban are exempt, as in
#       scripts/start_node_ep_test.sh case (f).
#   (h) end-to-end assembly from stub stage outputs -> an artifact that
#       VALIDATES. The assembler is the piece with no natural test on a GPU-less
#       host, so it gets fed fixtures.
#   (i) lints: shellcheck on every .sh, py_compile on every .py, typos clean.
#   (k) CERTIFIED eligibility. The verdict has to come from gate EVIDENCE and
#       from a pair that is actually this cell's other leg. Any parseable
#       --paired-artifact used to be enough -- `{}` included -- and the three
#       measurement gates the ladder reports while exiting 0 (the 80% vacuity
#       floor, request errors, the 10% spread) only reached `notes`. Each red
#       below is one of those: an empty pair, a pair on the wrong SKU, a pair
#       three days old, a failed boot gate, a failed known-answer probe, and
#       ladder JSONs that exit 0 while failing each measurement gate.
#   (j) teardown ownership, against a PATH-shimmed `docker` that records every
#       call. A `docker run` that exits 125 (the name is already in use) means
#       somebody else owns that container, and the cell must stop and remove
#       NOTHING -- the old teardown ran `docker stop <name>` / `docker rm
#       <name>` unconditionally and deleted the other run's live server. The
#       cell tears down by the container ID its own `docker run -d` returned,
#       and the name carries this invocation's pid so the collision is
#       unlikely in the first place.
#   (l) the same ownership, on the way out of a SIGNAL. The teardown marked
#       "always" was reached by normal control flow only: a cell killed during
#       its boot poll exited -15 with the container it had just created still
#       running and no artifact at all, while the otherwise identical boot
#       FAILURE stopped and removed that same container and wrote its NO-GO.
#       Same owned resources, opposite outcome -- and on a rented box the
#       difference is a vLLM server holding the GPU until somebody finds it.
#
#   (m) ownership during the CREATE itself. `docker run -d` makes the
#       container and then prints its ID, and a cell killed between the two
#       held no ID to tear down by: the stub's running set kept the container
#       and the Docker log showed only `run`. Ownership has to be recoverable
#       from what was chosen and recorded before the create.
#   (n) a failed Atlas kernel audit must not start normal serving or touch
#       records from an earlier invocation. The wrapper used to background
#       the launcher after a failed audit, then race it against teardown.
#       A successful audit still permits serving, and termination cleans up
#       only this invocation's ranks. The launcher suite tests occupied ports.
#   (o) the same create window as (m), on the ATLAS side. The launcher records
#       rank<N>.container only after `docker run -d` returns, so a cell killed
#       inside the create owned a rank by no record at all and stopped
#       nothing. The intent written before the create is what makes it
#       recoverable, and the query that reconciles it carries the exact name
#       AND this launch's label -- a container of another run wearing that
#       name is untouched.
#   (r) engine identity under a MUTABLE tag. capture_engine_identity ran after
#       teardown and inspected the image TAG, so a rebuild that re-pointed the
#       tag during the run made the artifact name a build that never served a
#       request. The identity is the launched container's resolved image ID,
#       recorded at create time and read before teardown removes it.
#   (q) a teardown lookup that FAILED, mirrored from the launcher's case (l):
#       `docker ps --filter label=... || true` read an unreachable daemon as
#       "nothing was created here", and the cell finished its teardown stage
#       cleanly while its container kept the GPU.
#   (p) engine_version is the ENGINE's identity. The checkout SHA was passed
#       as --git-sha for either engine and a digest was captured only for
#       vLLM, so an Atlas container cell claimed the campaign's revision with
#       neither a digest nor a binary hash -- and certified. Each engine now
#       records what actually ran, and the checkout SHA is `harness`.
#
# Usage: bash bench/campaign/campaign_test.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RUN="$HERE/run_cell.sh"
VC="$HERE/vllm_control.sh"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

asserts=0
fail() { echo "ASSERT FAILED [$1]: $2" >&2; exit 1; }
ok() { asserts=$((asserts + 1)); echo "  ok [$1] $2"; }
have() { grep -Fq -- "$2" <<<"$1"; }

# A nested refusal must remain nonzero after the dry-run finalizer.
python3 "$HERE/dryrun_failure_test.py" || fail dryrun "renderer failure regression"
ok dryrun "3 dry-run exit regressions passed (both engines, refusal before boot)"
python3 "$HERE/admission_test.py" || fail admission "preflight admission regression"
ok admission "failed preflight never launches either engine"
python3 "$HERE/stream_probe_test.py" || fail stream "stream evidence regression"
ok stream "raw streaming diagnostic rejects incomplete replies and bounds trickling input"
python3 "$HERE/thinking_policy_test.py" || fail policy "thinking policy regression"
ok policy "campaign thinking eligibility refuses excluded modes before launch"
python3 "$HERE/../hopper_ab/coherency_policy_test.py" || fail coherency "HTTP thinking policy regression"
python3 "$HERE/coherency_evidence_test.py" || fail coherency "artifact thinking policy regression"
ok coherency "gate requests and artifact certification match the cell's thinking mode"
python3 "$HERE/vllm_pins_test.py" || fail pins "vLLM revision identity regression"
ok pins "vLLM primary and draft pins survive rendering and reject identity overrides"

for test in kimi_context qwen27b_recipe minimax_nvfp4 atlas_revision draft_image_support model_capture; do
  python3 "$HERE/${test}_test.py" || fail recovery "$test regression"
done
python3 "$HERE/cell_assemble.py" --selftest || fail recovery "assembler selftest"
ok recovery "recipe and launch identity regressions passed"

for test in process_mode process_launch process_endpoint process_runner process_readiness cell_deadline; do
  python3 "$HERE/${test}_test.py" || fail process "$test regression"
  ok process "$test CPU regression suite passed (Linux-only cases skip elsewhere)"
done

# shellcheck source=bench/campaign/fixtures/campaign/cases_render_assemble.sh
source "$HERE/fixtures/campaign/cases_render_assemble.sh"
# shellcheck source=bench/campaign/fixtures/campaign/cases_docker_setup.sh
source "$HERE/fixtures/campaign/cases_docker_setup.sh"
# shellcheck source=bench/campaign/fixtures/campaign/cases_interruption.sh
source "$HERE/fixtures/campaign/cases_interruption.sh"
# shellcheck source=bench/campaign/fixtures/campaign/cases_identity_cleanup.sh
source "$HERE/fixtures/campaign/cases_identity_cleanup.sh"

# ── (i) lints ────────────────────────────────────────────────────────────────
if command -v shellcheck >/dev/null 2>&1; then
  for f in "$HERE"/*.sh; do
    shellcheck -x "$f" || fail i "shellcheck failed on $f"
  done
  ok i "shellcheck clean on every .sh under bench/campaign"
else
  echo "  -- [i] shellcheck not installed, skipped"
fi

for f in "$HERE"/*.py; do
  python3 -m py_compile "$f" || fail i "py_compile failed on $f"
done
rm -rf "$HERE/__pycache__"
ok i "python3 -m py_compile clean on every .py under bench/campaign"

for f in "$HERE"/*.json "$HERE"/fixtures/*.json; do
  python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$f" \
    || fail i "$f is not valid JSON"
done
ok i "every .json under bench/campaign parses"

if command -v typos >/dev/null 2>&1; then
  typos "$HERE" || fail i "typos found under bench/campaign"
  ok i "typos clean under bench/campaign"
else
  echo "  -- [i] typos not installed, skipped"
fi

echo ""
echo "ALL $asserts assertions passed."
