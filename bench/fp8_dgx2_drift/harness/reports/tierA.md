# Harness aggregate — tier `tierA` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_tierA_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 1.200 ± 1.400 | 1.000 | 4.000 | [0.400, 2.100] | 5/10 |
| cargo_toml_valid | rate | 10 | 0.300 ± 0.458 | 0.000 | 1.000 | [0.000, 0.600] | 3/10 |
| cargo_toml_present | rate | 10 | 0.500 ± 0.500 | 1.000 | 1.000 | [0.200, 0.800] | 5/10 |
| tool_calls_total | count | 10 | 8.200 ± 3.763 | 9.000 | 14.000 | [5.700, 10.400] | 9/10 |
| write_calls | count | 10 | 1.500 ± 1.204 | 2.000 | 3.000 | [0.700, 2.300] | 7/10 |
| drift_empty_path | count | 10 | 0.200 ± 0.400 | 0.000 | 1.000 | [0.000, 0.500] | 2/10 |
| drift_path_outside_target | count | 10 | 0.200 ± 0.400 | 0.000 | 1.000 | [0.000, 0.500] | 2/10 |
| drift_path_literal_space | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| drift_lean_prefix | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| drift_bash_as_content | count | 10 | 0.300 ± 0.900 | 0.000 | 3.000 | [0.000, 0.900] | 1/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 0.900 ± 1.044 | 1.000 | 3.000 | [0.300, 1.600] | 5/10 |
| atlas_ws1_mask_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_b1_drift_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tier5c_retries | count | 10 | 0.400 ± 0.663 | 0.000 | 2.000 | [0.000, 0.900] | 3/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 7.700 ± 3.635 | 9.000 | 14.000 | [5.300, 9.800] | 9/10 |
| wall_time_s | count | 10 | 169.753 ± 92.826 | 182.041 | 360.100 | [114.258, 229.203] | 10/10 |
