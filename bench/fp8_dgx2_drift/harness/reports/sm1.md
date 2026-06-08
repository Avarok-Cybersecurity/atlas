# Harness aggregate — tier `sm1` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_sm1_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 1.500 ± 1.118 | 1.000 | 3.000 | [0.800, 2.200] | 8/10 |
| cargo_toml_valid | rate | 10 | 0.300 ± 0.458 | 0.000 | 1.000 | [0.000, 0.600] | 3/10 |
| cargo_toml_present | rate | 10 | 0.800 ± 0.400 | 1.000 | 1.000 | [0.500, 1.000] | 8/10 |
| tool_calls_total | count | 10 | 7.300 ± 3.926 | 7.000 | 18.000 | [5.300, 10.100] | 10/10 |
| write_calls | count | 10 | 2.600 ± 2.332 | 2.000 | 6.000 | [1.200, 4.100] | 7/10 |
| drift_empty_path | count | 10 | 0.400 ± 0.663 | 0.000 | 2.000 | [0.000, 0.800] | 3/10 |
| drift_path_outside_target | count | 10 | 0.700 ± 1.418 | 0.000 | 4.000 | [0.000, 1.700] | 2/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 0.800 ± 1.470 | 0.000 | 5.000 | [0.100, 1.800] | 4/10 |
| atlas_ws1_mask_fires | count | 10 | 12.900 ± 5.700 | 12.000 | 27.000 | [9.700, 16.800] | 10/10 |
| atlas_b1_drift_fires | count | 10 | 2.800 ± 2.088 | 4.000 | 6.000 | [1.600, 4.100] | 9/10 |
| atlas_tier5c_retries | count | 10 | 0.200 ± 0.400 | 0.000 | 1.000 | [0.000, 0.500] | 2/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 7.100 ± 3.910 | 7.000 | 18.000 | [5.100, 9.900] | 10/10 |
| wall_time_s | count | 10 | 146.403 ± 49.376 | 149.253 | 232.165 | [117.194, 177.073] | 10/10 |
