# Harness aggregate — tier `atlas_c3verify` (N=3)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_atlas_c3verify_*.json`.
Runs: [1, 2, 3]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 3 | 2.333 ± 0.943 | 3.000 | 3.000 | [1.000, 3.000] | 3/3 |
| cargo_toml_valid | rate | 3 | 0.667 ± 0.471 | 1.000 | 1.000 | [0.000, 1.000] | 2/3 |
| cargo_toml_present | rate | 3 | 1.000 ± 0.000 | 1.000 | 1.000 | [1.000, 1.000] | 3/3 |
| tool_calls_total | count | 3 | 7.000 ± 2.828 | 5.000 | 11.000 | [5.000, 11.000] | 3/3 |
| write_calls | count | 3 | 2.000 ± 1.633 | 2.000 | 4.000 | [0.000, 4.000] | 2/3 |
| drift_empty_path | count | 3 | 1.000 ± 1.414 | 0.000 | 3.000 | [0.000, 3.000] | 1/3 |
| drift_path_outside_target | count | 3 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/3 |
| drift_path_literal_space | count | 3 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/3 |
| drift_lean_prefix | count | 3 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/3 |
| drift_bash_as_content | count | 3 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/3 |
| drift_xml_attr_leak | count | 3 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/3 |
| drift_toml_newlines_collapsed | count | 3 | 1.000 ± 0.816 | 1.000 | 2.000 | [0.000, 2.000] | 2/3 |
| atlas_ws1_mask_fires | count | 3 | 1.333 ± 0.943 | 2.000 | 2.000 | [0.000, 2.000] | 2/3 |
| atlas_b1_drift_fires | count | 3 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/3 |
| atlas_tier5c_retries | count | 3 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/3 |
| atlas_a2_fuzzy_fires | count | 3 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/3 |
| atlas_tool_call_lines | count | 3 | 6.333 ± 2.625 | 5.000 | 10.000 | [4.000, 10.000] | 3/3 |
| wall_time_s | count | 3 | 293.217 ± 69.465 | 322.112 | 360.083 | [197.456, 360.083] | 3/3 |
