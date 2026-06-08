# Harness aggregate — tier `mixedkv2` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_mixedkv2_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 0.600 ± 0.917 | 0.000 | 3.000 | [0.100, 1.200] | 4/10 |
| cargo_toml_valid | rate | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| cargo_toml_present | rate | 10 | 0.400 ± 0.490 | 0.000 | 1.000 | [0.100, 0.700] | 4/10 |
| tool_calls_total | count | 10 | 6.400 ± 2.107 | 7.000 | 11.000 | [5.100, 7.800] | 10/10 |
| write_calls | count | 10 | 1.900 ± 1.972 | 2.000 | 6.000 | [0.700, 3.200] | 6/10 |
| drift_empty_path | count | 10 | 0.500 ± 1.500 | 0.000 | 5.000 | [0.000, 1.500] | 1/10 |
| drift_path_outside_target | count | 10 | 0.400 ± 0.800 | 0.000 | 2.000 | [0.000, 1.000] | 2/10 |
| drift_path_literal_space | count | 10 | 0.400 ± 0.917 | 0.000 | 3.000 | [0.000, 1.000] | 2/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.200 ± 0.600 | 0.000 | 2.000 | [0.000, 0.600] | 1/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 0.900 ± 1.375 | 0.000 | 4.000 | [0.100, 1.800] | 4/10 |
| atlas_ws1_mask_fires | count | 10 | 14.400 ± 5.122 | 14.000 | 26.000 | [11.500, 17.800] | 10/10 |
| atlas_b1_drift_fires | count | 10 | 4.500 ± 1.803 | 5.000 | 7.000 | [3.400, 5.600] | 10/10 |
| atlas_tier5c_retries | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 6.300 ± 2.100 | 6.000 | 11.000 | [5.000, 7.700] | 10/10 |
| wall_time_s | count | 10 | 191.391 ± 75.955 | 187.370 | 348.465 | [147.784, 243.495] | 10/10 |
