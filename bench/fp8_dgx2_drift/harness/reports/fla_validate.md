# Harness aggregate — tier `fla_validate` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_fla_validate_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 922.000 ± 82.843 | 896.000 | 1169.000 | [890.000, 979.100] | 10/10 |
| cargo_toml_valid | rate | 10 | 1.000 ± 0.000 | 1.000 | 1.000 | [1.000, 1.000] | 10/10 |
| cargo_toml_present | rate | 10 | 1.000 ± 0.000 | 1.000 | 1.000 | [1.000, 1.000] | 10/10 |
| tool_calls_total | count | 10 | 15.500 ± 4.031 | 14.000 | 25.000 | [13.200, 18.200] | 10/10 |
| write_calls | count | 10 | 2.400 ± 1.114 | 2.000 | 5.000 | [1.800, 3.200] | 10/10 |
| drift_empty_path | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_path_outside_target | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_ws1_mask_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_b1_drift_fires | count | 10 | 0.300 ± 0.458 | 0.000 | 1.000 | [0.000, 0.600] | 3/10 |
| atlas_tier5c_retries | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 14.300 ± 4.051 | 13.000 | 24.000 | [12.100, 17.000] | 10/10 |
| wall_time_s | count | 10 | 280.683 ± 76.050 | 307.555 | 360.068 | [232.160, 325.539] | 10/10 |
