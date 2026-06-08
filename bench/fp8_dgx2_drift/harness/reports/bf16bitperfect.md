# Harness aggregate — tier `bf16bitperfect` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_bf16bitperfect_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 76.100 ± 224.634 | 1.000 | 750.000 | [0.900, 226.000] | 9/10 |
| cargo_toml_valid | rate | 10 | 0.900 ± 0.300 | 1.000 | 1.000 | [0.700, 1.000] | 9/10 |
| cargo_toml_present | rate | 10 | 0.900 ± 0.300 | 1.000 | 1.000 | [0.700, 1.000] | 9/10 |
| tool_calls_total | count | 10 | 9.300 ± 3.716 | 11.000 | 14.000 | [6.900, 11.400] | 10/10 |
| write_calls | count | 10 | 3.400 ± 1.855 | 4.000 | 6.000 | [2.200, 4.500] | 9/10 |
| drift_empty_path | count | 10 | 0.200 ± 0.400 | 0.000 | 1.000 | [0.000, 0.500] | 2/10 |
| drift_path_outside_target | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 0.400 ± 0.917 | 0.000 | 3.000 | [0.000, 1.000] | 2/10 |
| atlas_ws1_mask_fires | count | 10 | 1.800 ± 1.400 | 2.000 | 5.000 | [1.000, 2.700] | 8/10 |
| atlas_b1_drift_fires | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| atlas_tier5c_retries | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 10.700 ± 4.713 | 12.000 | 18.000 | [7.700, 13.500] | 10/10 |
| wall_time_s | count | 10 | 283.061 ± 108.079 | 360.047 | 360.093 | [212.045, 345.351] | 10/10 |
