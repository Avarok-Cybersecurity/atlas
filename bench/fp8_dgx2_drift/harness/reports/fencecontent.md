# Harness aggregate — tier `fencecontent` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_fencecontent_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 1.700 ± 0.781 | 2.000 | 3.000 | [1.200, 2.200] | 10/10 |
| cargo_toml_valid | rate | 10 | 0.800 ± 0.400 | 1.000 | 1.000 | [0.500, 1.000] | 8/10 |
| cargo_toml_present | rate | 10 | 0.900 ± 0.300 | 1.000 | 1.000 | [0.700, 1.000] | 9/10 |
| tool_calls_total | count | 10 | 8.700 ± 3.551 | 8.000 | 16.000 | [6.700, 11.100] | 10/10 |
| write_calls | count | 10 | 2.500 ± 1.803 | 3.000 | 6.000 | [1.500, 3.700] | 9/10 |
| drift_empty_path | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_path_outside_target | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| atlas_ws1_mask_fires | count | 10 | 3.100 ± 1.972 | 4.000 | 7.000 | [1.900, 4.300] | 9/10 |
| atlas_b1_drift_fires | count | 10 | 0.400 ± 0.663 | 0.000 | 2.000 | [0.000, 0.900] | 3/10 |
| atlas_tier5c_retries | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 8.700 ± 3.551 | 8.000 | 16.000 | [6.700, 11.100] | 10/10 |
| wall_time_s | count | 10 | 199.091 ± 66.651 | 188.626 | 306.612 | [158.873, 241.178] | 10/10 |
