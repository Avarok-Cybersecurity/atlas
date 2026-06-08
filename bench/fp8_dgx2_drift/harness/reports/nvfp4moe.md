# Harness aggregate — tier `nvfp4moe` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_nvfp4moe_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 1.200 ± 1.166 | 1.000 | 3.000 | [0.500, 1.900] | 6/10 |
| cargo_toml_valid | rate | 10 | 0.300 ± 0.458 | 0.000 | 1.000 | [0.000, 0.600] | 3/10 |
| cargo_toml_present | rate | 10 | 0.500 ± 0.500 | 1.000 | 1.000 | [0.200, 0.800] | 5/10 |
| tool_calls_total | count | 10 | 8.900 ± 3.936 | 9.000 | 17.000 | [6.700, 11.500] | 10/10 |
| write_calls | count | 10 | 2.000 ± 2.280 | 2.000 | 7.000 | [0.700, 3.500] | 6/10 |
| drift_empty_path | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_path_outside_target | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.300 ± 0.900 | 0.000 | 3.000 | [0.000, 0.900] | 1/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 1.400 ± 1.428 | 1.000 | 4.000 | [0.600, 2.300] | 6/10 |
| atlas_ws1_mask_fires | count | 10 | 17.100 ± 7.006 | 15.000 | 28.000 | [13.000, 21.600] | 10/10 |
| atlas_b1_drift_fires | count | 10 | 1.300 ± 1.005 | 1.000 | 4.000 | [0.800, 2.000] | 9/10 |
| atlas_tier5c_retries | count | 10 | 0.400 ± 0.917 | 0.000 | 3.000 | [0.000, 1.000] | 2/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 8.300 ± 3.068 | 9.000 | 15.000 | [6.600, 10.300] | 10/10 |
| wall_time_s | count | 10 | 128.606 ± 62.259 | 113.618 | 303.228 | [99.015, 172.316] | 10/10 |
