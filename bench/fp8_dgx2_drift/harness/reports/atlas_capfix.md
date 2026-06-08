# Harness aggregate — tier `atlas_capfix` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_atlas_capfix_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 66.600 ± 195.138 | 2.000 | 652.000 | [0.900, 197.000] | 7/10 |
| cargo_toml_valid | rate | 10 | 0.400 ± 0.490 | 0.000 | 1.000 | [0.100, 0.700] | 4/10 |
| cargo_toml_present | rate | 10 | 0.700 ± 0.458 | 1.000 | 1.000 | [0.400, 1.000] | 7/10 |
| tool_calls_total | count | 10 | 6.500 ± 3.074 | 7.000 | 13.000 | [4.600, 8.500] | 10/10 |
| write_calls | count | 10 | 1.900 ± 1.640 | 3.000 | 4.000 | [0.900, 2.900] | 6/10 |
| drift_empty_path | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| drift_path_outside_target | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 0.900 ± 1.300 | 0.000 | 4.000 | [0.200, 1.800] | 4/10 |
| atlas_ws1_mask_fires | count | 10 | 2.400 ± 2.764 | 2.000 | 10.000 | [1.000, 4.300] | 7/10 |
| atlas_b1_drift_fires | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| atlas_tier5c_retries | count | 10 | 0.400 ± 0.663 | 0.000 | 2.000 | [0.000, 0.800] | 3/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 5.800 ± 2.926 | 6.000 | 13.000 | [4.200, 7.800] | 10/10 |
| wall_time_s | count | 10 | 248.114 ± 92.969 | 273.244 | 360.084 | [190.349, 305.864] | 10/10 |
