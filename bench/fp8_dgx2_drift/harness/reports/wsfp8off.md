# Harness aggregate — tier `wsfp8off` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_wsfp8off_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 1030.100 ± 268.110 | 932.000 | 1405.000 | [861.900, 1197.200] | 10/10 |
| cargo_toml_valid | rate | 10 | 1.000 ± 0.000 | 1.000 | 1.000 | [1.000, 1.000] | 10/10 |
| cargo_toml_present | rate | 10 | 1.000 ± 0.000 | 1.000 | 1.000 | [1.000, 1.000] | 10/10 |
| tool_calls_total | count | 10 | 18.800 ± 5.963 | 18.000 | 31.000 | [15.200, 22.600] | 10/10 |
| write_calls | count | 10 | 2.500 ± 1.025 | 2.000 | 5.000 | [1.900, 3.200] | 10/10 |
| drift_empty_path | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_path_outside_target | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_ws1_mask_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_b1_drift_fires | count | 10 | 0.500 ± 0.500 | 1.000 | 1.000 | [0.200, 0.800] | 5/10 |
| atlas_tier5c_retries | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| wall_time_s | count | 10 | 266.193 ± 93.016 | 307.235 | 360.071 | [207.180, 320.075] | 10/10 |
