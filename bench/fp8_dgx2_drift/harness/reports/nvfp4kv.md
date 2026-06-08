# Harness aggregate — tier `nvfp4kv` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_nvfp4kv_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 1.500 ± 1.360 | 1.000 | 4.000 | [0.700, 2.400] | 7/10 |
| cargo_toml_valid | rate | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| cargo_toml_present | rate | 10 | 0.700 ± 0.458 | 1.000 | 1.000 | [0.400, 1.000] | 7/10 |
| tool_calls_total | count | 10 | 7.600 ± 3.929 | 6.000 | 18.000 | [5.600, 10.300] | 10/10 |
| write_calls | count | 10 | 1.500 ± 1.565 | 1.000 | 4.000 | [0.600, 2.500] | 6/10 |
| drift_empty_path | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| drift_path_outside_target | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 1.100 ± 1.375 | 1.000 | 4.000 | [0.300, 2.000] | 5/10 |
| atlas_ws1_mask_fires | count | 10 | 11.800 ± 8.553 | 10.000 | 34.000 | [7.300, 17.600] | 9/10 |
| atlas_b1_drift_fires | count | 10 | 1.800 ± 2.040 | 1.000 | 6.000 | [0.700, 3.100] | 7/10 |
| atlas_tier5c_retries | count | 10 | 0.200 ± 0.400 | 0.000 | 1.000 | [0.000, 0.500] | 2/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 7.400 ± 3.929 | 6.000 | 18.000 | [5.400, 10.100] | 10/10 |
| wall_time_s | count | 10 | 142.631 ± 46.706 | 125.138 | 224.715 | [115.437, 172.845] | 10/10 |
