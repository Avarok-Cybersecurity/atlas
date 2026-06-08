# Harness aggregate — tier `atlas_nogram` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_atlas_nogram_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 2.000 ± 1.949 | 2.000 | 7.000 | [1.000, 3.300] | 8/10 |
| cargo_toml_valid | rate | 10 | 0.400 ± 0.490 | 0.000 | 1.000 | [0.100, 0.700] | 4/10 |
| cargo_toml_present | rate | 10 | 0.800 ± 0.400 | 1.000 | 1.000 | [0.500, 1.000] | 8/10 |
| tool_calls_total | count | 10 | 15.700 ± 24.150 | 7.000 | 87.000 | [5.700, 32.600] | 10/10 |
| write_calls | count | 10 | 10.600 ± 25.176 | 2.000 | 86.000 | [1.500, 27.700] | 8/10 |
| drift_empty_path | count | 10 | 8.700 ± 25.108 | 0.000 | 84.000 | [0.000, 25.600] | 3/10 |
| drift_path_outside_target | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 1.000 ± 0.775 | 1.000 | 2.000 | [0.500, 1.500] | 7/10 |
| atlas_ws1_mask_fires | count | 10 | 5.600 ± 7.432 | 4.000 | 24.000 | [1.700, 10.700] | 7/10 |
| atlas_b1_drift_fires | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| atlas_tier5c_retries | count | 10 | 0.200 ± 0.400 | 0.000 | 1.000 | [0.000, 0.500] | 2/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 15.500 ± 24.188 | 6.000 | 87.000 | [5.600, 32.400] | 10/10 |
| wall_time_s | count | 10 | 238.673 ± 88.037 | 212.446 | 360.083 | [185.682, 293.804] | 10/10 |
