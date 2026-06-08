# Harness aggregate — tier `bf16router` (N=3)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_bf16router_*.json`.
Runs: [1, 2, 3]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 3 | 1.000 ± 0.000 | 1.000 | 1.000 | [1.000, 1.000] | 3/3 |
| cargo_toml_valid | rate | 3 | 0.333 ± 0.471 | 0.000 | 1.000 | [0.000, 1.000] | 1/3 |
| cargo_toml_present | rate | 3 | 1.000 ± 0.000 | 1.000 | 1.000 | [1.000, 1.000] | 3/3 |
| tool_calls_total | count | 3 | 4.667 ± 1.700 | 4.000 | 7.000 | [3.000, 7.000] | 3/3 |
| write_calls | count | 3 | 1.333 ± 0.471 | 1.000 | 2.000 | [1.000, 2.000] | 3/3 |
| drift_empty_path | count | 3 | 0.333 ± 0.471 | 0.000 | 1.000 | [0.000, 1.000] | 1/3 |
| drift_path_outside_target | count | 3 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/3 |
| drift_path_literal_space | count | 3 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/3 |
| drift_lean_prefix | count | 3 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/3 |
| drift_bash_as_content | count | 3 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/3 |
| drift_xml_attr_leak | count | 3 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/3 |
| drift_toml_newlines_collapsed | count | 3 | 0.333 ± 0.471 | 0.000 | 1.000 | [0.000, 1.000] | 1/3 |
| atlas_ws1_mask_fires | count | 3 | 5.667 ± 1.247 | 6.000 | 7.000 | [4.000, 7.000] | 3/3 |
| atlas_b1_drift_fires | count | 3 | 0.333 ± 0.471 | 0.000 | 1.000 | [0.000, 1.000] | 1/3 |
| atlas_tier5c_retries | count | 3 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/3 |
| atlas_a2_fuzzy_fires | count | 3 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/3 |
| atlas_tool_call_lines | count | 3 | 4.667 ± 1.700 | 4.000 | 7.000 | [3.000, 7.000] | 3/3 |
| wall_time_s | count | 3 | 360.064 ± 0.015 | 360.055 | 360.085 | [360.051, 360.085] | 3/3 |
