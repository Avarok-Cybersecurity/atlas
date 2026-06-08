# Harness aggregate — tier `norep2` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_norep2_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 799.200 ± 414.355 | 908.000 | 1260.000 | [525.600, 1028.600] | 10/10 |
| cargo_toml_valid | rate | 10 | 0.900 ± 0.300 | 1.000 | 1.000 | [0.700, 1.000] | 9/10 |
| cargo_toml_present | rate | 10 | 1.000 ± 0.000 | 1.000 | 1.000 | [1.000, 1.000] | 10/10 |
| tool_calls_total | count | 10 | 18.700 ± 4.428 | 20.000 | 27.000 | [16.000, 21.600] | 10/10 |
| write_calls | count | 10 | 3.200 ± 1.327 | 4.000 | 5.000 | [2.400, 4.000] | 10/10 |
| drift_empty_path | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| drift_path_outside_target | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_ws1_mask_fires | count | 10 | 18.000 ± 7.183 | 16.000 | 30.000 | [13.800, 22.600] | 10/10 |
| atlas_b1_drift_fires | count | 10 | 0.200 ± 0.400 | 0.000 | 1.000 | [0.000, 0.500] | 2/10 |
| atlas_tier5c_retries | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 17.300 ± 4.148 | 17.000 | 23.000 | [14.800, 19.900] | 10/10 |
| wall_time_s | count | 10 | 324.086 ± 56.065 | 360.045 | 360.081 | [284.633, 353.071] | 10/10 |
