# Serve-matrix evidence audit, 2026-09-04

Targets Avarok-Cybersecurity/atlas `main` at `567b5ebe7784ac3657a4ae97940f6783c8414393`,
revalidated September 5. This tranche changes the Python release harness and
its checks. It changes no inference, model, kernel, or Rust code.

The decision under examination is whether the documented serve-matrix command
may authorize an image release. The coverage model is artifact presence,
required probe groups, numerical validity, declared model identity, and process
completion. A green result remains limited to these observations.

## Findings and repairs

The original 23-check `tests/test_gate_results.py` suite passed. The following
inputs and transitions nevertheless accepted invalid release evidence:

| Contract | Observed original behavior | Repair and check owner |
| --- | --- | --- |
| Required evidence | `{ "model": "M/a", "coherence": [] }` passed the actual gate entry point with exit 0; absent, null, or empty probe groups bypassed their bars. | Require nonempty, correctly shaped coherence, fibonacci, tool-call, and TPS groups. `EvidenceContracts.test_every_required_probe_must_supply_evidence` and `ArtifactContracts.test_cli_blocks_a_partial_artifact`. |
| Status identity | `FAIL (expected PASS)` counted as coherence success; `FAIL (expected N/A)` waived tool checks. | Read the leading status, preserving the suite's `PASS (plain-text)` and `N/A (parser not supported)` forms. `test_status_reason_cannot_masquerade_as_a_pass_or_waiver`. |
| Numerical evidence | NaN/Inf TPS escaped comparisons. Invalid samples could disappear beside a valid sample; booleans counted as numbers. | Require every sample and the mean to be finite numeric values; reject unrepresentable integers. `test_invalid_tps_cannot_hide_alongside_a_valid_sample`. |
| Baseline integrity | Invalid baseline values or malformed baseline JSON silently disabled the regression bar. | Keep absent baselines distinct from present invalid ones; report `tps(invalid-baseline)`. `test_existing_invalid_baseline_cannot_disable_the_regression_bar` and `test_malformed_baseline_is_present_but_invalid`. |
| Model identity | An artifact naming `M/wrong-checkpoint` verified a roster entry naming `M/expected`. | Match the artifact's model to the manifest, reporting `model-mismatch`. `test_manifest_model_must_match_the_observed_result`. |
| Baseline refresh | A correctness-failing or wrong-model artifact could overwrite the blessed throughput baseline. | Require correctness, finite positive throughput, and roster identity before writing. `test_baseline_update_requires_all_correctness_bars_and_identity`; every rejected row starts with and preserves a distinct 50 TPS baseline. |
| New run, failed boot | Publishing a new manifest left the prior run's planned JSON files in place. A model that never booted could inherit old passing evidence. | Validate all labels, then remove only planned per-model results before any boot attempt. `OrchestratorEvidence.test_new_manifest_invalidates_planned_results_before_any_boot`; an unplanned artifact must remain unchanged. |
| Output-less suite | A subprocess exiting successfully without writing JSON reused the old result file. | Remove the specific prior output before launching the suite. `test_success_without_output_cannot_reuse_a_previous_result`. |
| Failing suite | `wait_and_read` ignored the subprocess exit code, including a process that wrote valid JSON before exiting 7. | On nonzero exit, remove that output and return no result, so the independent gate also rejects it. `test_failed_subprocess_cannot_leave_a_gateable_result`. |

The subprocess checks call the real `run_suite` and `wait_and_read` functions
with a tiny local Python program in place of `single_gpu_suite.py`. They launch
real child processes. A successful child writes 63 TPS instead of the old
42 TPS, and both the returned artifact and the independent gate must observe
the new evidence. No HTTP endpoint or Docker command is called.

Cleanup validates the whole roster before touching any artifact, rejecting path
separators, empty labels, and reserved aggregate/manifest names. It deletes only
generated planned result JSON, not logs or unplanned results. Archive completed
run artifacts before starting another campaign in the same checkout.

## Functional connection and reachability

Static trace: `single_gpu_suite.py` emits the probe groups and model argument;
`run_all_models.py::main` publishes the planned roster before starting model
rounds; both single-GPU and multi-rank rounds call `run_suite` / `wait_and_read`.
The documented release procedure then executes `tests/gate_results.py`, whose
`main` consumes the roster, per-model artifacts, and committed baselines.

Observed: host fixture files reach `gate_manifest`, `verdict`, baseline updates,
and the actual gate CLI; subprocess fixtures reach both orchestrator helpers.
The previously unregistered Python checks now have a GPU-free CI job,
`serve-matrix-gate`, running this exact command:

```bash
python3 -m unittest discover -s tests -p 'test_gate_results*.py' -v
```

The existing gate source was byte-identical on Avarok
revision `567b5ebe7` and the initially examined Atlas-Inf revision `6c5f17dab` (SHA256
`9ff13894aa88633cad00394a281c360a394091df790719d17f6908d615e155a5`). These repairs
therefore address a surviving harness gap rather than assuming old audit work
was absent based on commit subjects.

## Defect sensitivity

Each mutation below was applied independently in an isolated copy of the
actual Python modules. The original 23 checks passed under every mutation.
The repaired checks rejected each mutation through the named assertions;
there were no import, syntax, or unrelated runtime errors in the reported
mutation runs. All mutation copies were outside the repository worktree.

| Mutant | Repaired observation |
| --- | --- |
| Remove the missing-probe failure append. | 12 absent/empty/null partitions and the partial-artifact CLI fail; the CLI wrongly exits 0 again. |
| Restore status substring matching. | Four failure/waiver classification partitions fail. |
| Allow non-finite floats and remove the finite-mean check. | NaN/Inf TPS, infinite baseline, and non-finite refresh partitions fail. |
| Bypass the manifest/result model comparison. | Expected `(0, 1, model-mismatch)` becomes `(1, 1, [])`. |
| Return absent-baseline `None` on malformed JSON. | The malformed-file fixture wrongly verifies instead of reporting invalid baseline. |
| Remove correctness/identity admission from baseline refresh. | Coherence, fibonacci, tool-call, and wrong-model refresh partitions overwrite the 50 TPS control. |
| Omit planned-output cleanup when publishing the manifest. | A simulated failed boot verifies the old artifact instead of reporting `no-result`. |
| Omit cleanup immediately before the suite subprocess. | Exit 0 with no output returns the old 42 TPS artifact instead of `None`. |
| Ignore the subprocess failure status. | A child writing JSON then exiting 7 returns a gateable result instead of `None`. |

Nearby valid controls remain: the existing two-of-three coherence policy,
known-gap tool waiver, annotated successful statuses, absent optional baselines,
45 TPS at a 50 TPS baseline's exact 10% boundary, a successful baseline refresh,
and a successful child writing new evidence. The threshold values are unchanged.

## Verification and limits

Original audit observations on macOS arm64 with Python 3.14.6, before the
September 5 Avarok revalidation below:

- Original suite: 23 passed, no skips.
- New scorer checks before repair: 33 checks ran with 39 failing subtest/assertion
  cases, including the process-level false green. The additional malformed-shape
  check was added while implementing input validation.
- New orchestrator checks before repair: six ran with nine failing assertion
  cases; the successful child control already passed.
- Final registered command: 40 passed, no skips. Nine distinct known-bad
  production mutations survived the old suite and failed the repaired checks.
- Python syntax compilation, workflow YAML parsing, `git diff --check`,
  `cargo fmt --all -- --check`, and the repository SPDX check passed.
- Workspace Clippy was attempted with `ATLAS_SKIP_BUILD=1`,
  `CUDARC_CUDA_VERSION=13000`, two build jobs, and an isolated Cargo target.
  It exited 101 because `spark-storage`'s `streaming_attention_e2e` target
  references Linux-only `IoUringBackend` on macOS.
  Rust workspace tests and rustdoc were not run after that build prerequisite
  failed. The Ubuntu CI Rust jobs remain the required owners of that evidence.

### September 5 Avarok revalidation

The complete 40-check suite was first run against unchanged Avarok production
code at `567b5ebe7`: it reported 54 failing assertions/subtests and 13 errors
from malformed probe shapes and an unrepresentable integer. With the repairs,
all 40 checks pass with zero skips. Avarok's existing case-insensitive readiness
marker and log matching are preserved. The CI job is added to Avarok's current
workflow; its other jobs and gating remain unchanged.

Python syntax, workspace Rust formatting, scoped typos, whitespace, and
actionlint workflow validation pass. Shellcheck additionally reports the same
three SC2016 findings in unchanged existing workflow code on both base and
patch. No changed file falls under the repository's SPDX source-header scope.
The Avarok workspace Clippy attempt stops in unchanged `spark-storage` code
using Linux-only `libc` constants (`O_DIRECT` and `POSIX_FADV_DONTNEED`) on
macOS. Ubuntu Rust CI remains pending; no local Rust test pass is claimed.

This does not establish CUDA execution, live-model coherence, numerical model
parity, throughput, multi-rank behavior, Docker image correctness, or the actual
checkpoint served. The result's `model` is an argument copied by the suite,
not independent confirmation of loaded weights.

The remaining highest-value experiment is a fresh image serve-matrix run with
image digest, source commit, checkpoint revision, and hardware/environment
recorded alongside the artifacts. Run one orchestrator per results directory:
this repair closes sequential stale-result reuse, not concurrent-writer races.
The manifest/result schema still lacks a run identity and cryptographic
provenance; manually supplied or copied artifacts can bypass the orchestrator's
freshness discipline. The gate also retains its aggregate coherence and TPS
policy and does not certify every individual prompt or complete probe cardinality.

The bounded host audit stops after closing these demonstrated release-evidence
gaps and registering their checks. Fresh GPU evidence remains a separate campaign.
