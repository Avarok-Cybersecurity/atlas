# Harness aggregate — tier `atlas_nonfp8` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_atlas_nonfp8_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 1.600 ± 1.356 | 2.000 | 4.000 | [0.800, 2.500] | 7/10 |
| cargo_toml_valid | rate | 10 | 0.400 ± 0.490 | 0.000 | 1.000 | [0.100, 0.700] | 4/10 |
| cargo_toml_present | rate | 10 | 0.700 ± 0.458 | 1.000 | 1.000 | [0.400, 1.000] | 7/10 |
| tool_calls_total | count | 10 | 9.800 ± 4.261 | 12.000 | 16.000 | [7.000, 12.300] | 10/10 |
| write_calls | count | 10 | 2.400 ± 1.800 | 2.000 | 7.000 | [1.400, 3.600] | 9/10 |
| drift_empty_path | count | 10 | 0.700 ± 1.100 | 0.000 | 3.000 | [0.000, 1.400] | 3/10 |
| drift_path_outside_target | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 0.800 ± 0.980 | 1.000 | 3.000 | [0.200, 1.400] | 5/10 |
| atlas_ws1_mask_fires | count | 10 | 4.400 ± 4.964 | 3.000 | 19.000 | [2.300, 7.900] | 10/10 |
| atlas_b1_drift_fires | count | 10 | 1.200 ± 1.600 | 1.000 | 5.000 | [0.300, 2.300] | 5/10 |
| atlas_tier5c_retries | count | 10 | 0.300 ± 0.458 | 0.000 | 1.000 | [0.000, 0.600] | 3/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 9.300 ± 4.026 | 11.000 | 16.000 | [6.700, 11.700] | 10/10 |
| wall_time_s | count | 10 | 336.226 ± 71.535 | 360.079 | 360.097 | [288.532, 360.081] | 10/10 |
