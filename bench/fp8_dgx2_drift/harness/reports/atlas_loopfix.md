# Harness aggregate — tier `atlas_loopfix` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_atlas_loopfix_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 1.800 ± 0.980 | 2.000 | 3.000 | [1.200, 2.400] | 9/10 |
| cargo_toml_valid | rate | 10 | 0.200 ± 0.400 | 0.000 | 1.000 | [0.000, 0.500] | 2/10 |
| cargo_toml_present | rate | 10 | 0.800 ± 0.400 | 1.000 | 1.000 | [0.500, 1.000] | 8/10 |
| tool_calls_total | count | 10 | 9.200 ± 2.040 | 9.000 | 13.000 | [8.000, 10.500] | 10/10 |
| write_calls | count | 10 | 3.400 ± 1.685 | 3.000 | 7.000 | [2.400, 4.500] | 10/10 |
| drift_empty_path | count | 10 | 0.900 ± 1.375 | 0.000 | 4.000 | [0.200, 1.800] | 4/10 |
| drift_path_outside_target | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 1.800 ± 0.600 | 2.000 | 3.000 | [1.400, 2.200] | 10/10 |
| atlas_ws1_mask_fires | count | 10 | 2.300 ± 1.100 | 2.000 | 4.000 | [1.600, 3.000] | 9/10 |
| atlas_b1_drift_fires | count | 10 | 0.300 ± 0.640 | 0.000 | 2.000 | [0.000, 0.800] | 2/10 |
| atlas_tier5c_retries | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 8.900 ± 1.868 | 9.000 | 12.000 | [7.800, 10.100] | 10/10 |
| wall_time_s | count | 10 | 360.088 ± 0.009 | 360.089 | 360.105 | [360.083, 360.093] | 10/10 |
