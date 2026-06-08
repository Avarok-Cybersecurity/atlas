# Harness aggregate — tier `atlas_cap900` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_atlas_cap900_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 1.100 ± 0.943 | 1.000 | 3.000 | [0.500, 1.700] | 7/10 |
| cargo_toml_valid | rate | 10 | 0.300 ± 0.458 | 0.000 | 1.000 | [0.000, 0.600] | 3/10 |
| cargo_toml_present | rate | 10 | 0.600 ± 0.490 | 1.000 | 1.000 | [0.300, 0.900] | 6/10 |
| tool_calls_total | count | 10 | 8.700 ± 7.824 | 10.000 | 22.000 | [4.100, 13.700] | 10/10 |
| write_calls | count | 10 | 2.500 ± 2.500 | 2.000 | 9.000 | [1.200, 4.200] | 8/10 |
| drift_empty_path | count | 10 | 0.800 ± 1.778 | 0.000 | 6.000 | [0.000, 2.000] | 3/10 |
| drift_path_outside_target | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 1.000 ± 0.775 | 1.000 | 2.000 | [0.500, 1.500] | 7/10 |
| atlas_ws1_mask_fires | count | 10 | 1.700 ± 1.100 | 2.000 | 4.000 | [1.000, 2.400] | 8/10 |
| atlas_b1_drift_fires | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| atlas_tier5c_retries | count | 10 | 0.200 ± 0.400 | 0.000 | 1.000 | [0.000, 0.500] | 2/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 8.500 ± 7.606 | 10.000 | 22.000 | [4.100, 13.400] | 10/10 |
| wall_time_s | count | 10 | 309.726 ± 271.058 | 184.566 | 900.086 | [156.334, 488.202] | 10/10 |
